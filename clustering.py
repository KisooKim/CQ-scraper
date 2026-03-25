"""Clustering module — Embed article titles, reduce with UMAP, cluster with HDBSCAN.

Post-processing pipeline:
  title cleaning → embed → UMAP → HDBSCAN → per-article ejection
  → recursive two-pass split → noise recovery → conditional merge
  → force-split (k-means) → dissolve junk

Parameter history:
  exp_033 (2026-03-16): min_article_sim=0.54, two_pass=12, merge=0.75/0.65, dissolve=0.63
  feedback-01 (2026-03-18): min_article_sim=0.57, two_pass=9, merge=0.69/0.68, dissolve=0.65
    + title cleaning (_clean_title) to strip [] column markers before embedding
    Fixes: bracket-based false clustering, outlier ejection, over-splitting related topics
  feedback-02 (2026-03-18): embed title + body snippet (first 300 chars, from R2)
    Full body caused over-merging (all articles too similar); snippet adds topical context
    without diluting title signal. Fixes: sports→election, opinion→lifestyle mixing.
  feedback-03 (2026-03-18): named entity overlap boost in merge step
    Clusters sharing key named entities (persons, orgs) get boosted merge score (+0.12 max).
    Fixes: related sub-clusters with borderline centroid sim not merging.
"""

import logging
import math
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

log = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model upskyy/bge-m3-korean ...")
        _model = SentenceTransformer("upskyy/bge-m3-korean")
    return _model


def _clean_title(title: str) -> str:
    """Remove newspaper formatting noise before embedding.

    Strips leading column/section markers like [사설], [기획], [김정호의 AI시대 전략],
    [동정], [출마합니다] etc. that cause bracket-based false clustering.
    Also strips trailing (종합), (단독), [단독] suffixes.
    """
    # Strip leading [...] or 【...】 markers (up to 30 chars inside)
    cleaned = re.sub(r'^\s*[\[【][^\]】]{1,30}[\]】]\s*', '', title)
    # Strip trailing (종합), (단독), [단독] etc.
    cleaned = re.sub(r'\s*[\(\[【][^)\]】]{1,10}[\)\]】]\s*$', '', cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned else title  # fallback to original if empty


def _extract_key_tokens(titles: list[str]) -> set[str]:
    """Extract named-entity-like tokens from titles for overlap scoring.

    Keeps Korean words of 2+ chars and uppercase English sequences (e.g. BTS, AI, LLM).
    These are likely proper nouns (persons, orgs, brands) that discriminate topics.
    """
    tokens: set[str] = set()
    for t in titles:
        for tok in re.split(r'[\s·,·\-·\/·\|·\[·\]·【·】·\(·\)·\"·\']+', t):
            tok = re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', tok)
            if not tok:
                continue
            # Korean proper nouns: 2+ chars, all Hangul
            if re.match(r'^[가-힣]{2,}$', tok):
                tokens.add(tok)
            # English proper nouns / abbreviations: 2+ chars, all uppercase OR mixed-case starting cap
            elif re.match(r'^[A-Z][A-Za-z]{1,}$', tok) or re.match(r'^[A-Z]{2,}$', tok):
                tokens.add(tok)
    return tokens


def _entity_overlap(titles_a: list[str], titles_b: list[str]) -> float:
    """Jaccard similarity of key tokens between two title sets. Returns 0.0–1.0."""
    ea = _extract_key_tokens(titles_a)
    eb = _extract_key_tokens(titles_b)
    union = ea | eb
    if not union:
        return 0.0
    return len(ea & eb) / len(union)


def _centroid_sim(embeddings: np.ndarray, indices: list[int]) -> tuple[np.ndarray, float]:
    """Compute centroid and average cosine similarity for a set of indices."""
    centroid = embeddings[indices].mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn == 0:
        return centroid, 0.0
    sims = []
    for i in indices:
        s = float(np.dot(embeddings[i], centroid) / (np.linalg.norm(embeddings[i]) * cn))
        sims.append(s)
    return centroid, float(np.mean(sims))


def cluster_articles(articles: list[dict], *,
                     min_cluster_size: int = 7,
                     min_samples: int = 3,
                     n_components: int = 10,
                     n_neighbors: int = 10,
                     cluster_selection_method: str = "leaf",
                     # Post-processing defaults (tuned 2026-03-18 via user feedback)
                     min_article_sim: float = 0.57,   # was 0.54 — eject more outliers
                     two_pass_size: int = 9,           # was 12 — split mixed clusters earlier
                     noise_recovery_sim: float = 0.60,
                     merge_sim: float = 0.69,          # was 0.75 — merge related sub-clusters
                     merge_min_quality: float = 0.68,  # was 0.65 — guard against bad merges
                     force_split_size: int = 12,
                     force_split_max_sim: float = 0.67,
                     dissolve_threshold: float = 0.65) -> dict:  # was 0.63 — dissolve more junk
    """Cluster articles by title similarity with post-processing refinement.

    Args:
        articles: list of dicts with keys: id, title, press_name
        min_cluster_size: HDBSCAN minimum cluster size
        min_samples: HDBSCAN min_samples
        n_components: UMAP target dimensions
        n_neighbors: UMAP n_neighbors
        cluster_selection_method: HDBSCAN selection method — "leaf" or "eom"
        min_article_sim: eject articles with centroid similarity below this
        two_pass_size: recursively re-cluster clusters larger than this
        noise_recovery_sim: assign noise articles to nearest cluster if sim >= this
        merge_sim: merge cluster pairs with centroid similarity above this
        merge_min_quality: only merge if both clusters have avg_sim >= this
        force_split_size: force-split large clusters with k-means if above this size
        force_split_max_sim: only force-split if cluster avg_sim < this
        dissolve_threshold: dissolve clusters with avg_sim below this (final cleanup)

    Returns:
        dict with:
            clusters: list of {label: int, article_ids: list[int], titles: list[str]}
            noise_ids: list[int]  (articles not assigned to any cluster)
            embeddings: np.ndarray  (full embeddings, for downstream use)
    """
    if not articles:
        return {"clusters": [], "noise_ids": [], "embeddings": np.array([])}

    titles = [a["title"] for a in articles]
    id_list = [a["id"] for a in articles]
    titles_clean = [_clean_title(t) for t in titles]

    # Step 1: Embed title + body snippet (first 300 chars of body).
    # Full body dilutes the title signal; a short snippet adds topical context
    # (e.g. disambiguates sports vs. election, opinion vs. lifestyle) without
    # drowning out the title. Body text fetched from R2 by generate_issues.py.
    BODY_SNIPPET_LEN = 300
    texts_to_embed = []
    for i, a in enumerate(articles):
        body = a.get("body_text", "").strip()
        snippet = body[:BODY_SNIPPET_LEN] if body else ""
        if snippet:
            texts_to_embed.append(titles_clean[i] + "\n" + snippet)
        else:
            texts_to_embed.append(titles_clean[i])
    has_body = sum(1 for t in texts_to_embed if "\n" in t)
    log.info("Embedding %d articles (%d with body text) ...", len(titles), has_body)
    t0 = time.time()
    model = _get_model()
    embeddings = model.encode(texts_to_embed, show_progress_bar=False, batch_size=32)
    log.info("  Embedding done in %.1fs", time.time() - t0)

    # Step 2: UMAP dimensionality reduction
    log.info("UMAP reduction (n_components=%d, n_neighbors=%d) ...", n_components, n_neighbors)
    reduced = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    # Step 3: HDBSCAN clustering
    log.info("HDBSCAN (min_cluster_size=%d, min_samples=%d, method=%s) ...",
             min_cluster_size, min_samples, cluster_selection_method)
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    ).fit_predict(reduced)

    # Build index-based cluster map
    cluster_map: dict[int, list[int]] = {}
    noise_idx: list[int] = []
    for i, lab in enumerate(labels):
        if lab == -1:
            noise_idx.append(i)
        else:
            cluster_map.setdefault(int(lab), []).append(i)

    n_raw = len(cluster_map)
    log.info("  Raw: %d clusters, %d noise", n_raw, len(noise_idx))

    # --- Post-processing pipeline ---

    # 1. Per-article ejection
    if min_article_sim > 0:
        ejected = 0
        for lab in list(cluster_map.keys()):
            indices = cluster_map[lab]
            centroid, _ = _centroid_sim(embeddings, indices)
            cn = np.linalg.norm(centroid)
            if cn == 0:
                continue
            keep, eject = [], []
            for i in indices:
                sim = float(np.dot(embeddings[i], centroid) / (np.linalg.norm(embeddings[i]) * cn))
                if sim < min_article_sim:
                    eject.append(i)
                else:
                    keep.append(i)
            if eject:
                ejected += len(eject)
                noise_idx.extend(eject)
                if len(keep) >= min_cluster_size:
                    cluster_map[lab] = keep
                else:
                    noise_idx.extend(keep)
                    del cluster_map[lab]
        if ejected:
            log.info("  Ejection: removed %d articles (sim<%.2f)", ejected, min_article_sim)

    # 2. Recursive two-pass split
    if two_pass_size > 0:
        split_count = 0
        next_label = max(cluster_map.keys(), default=-1) + 1
        sub_mcs = max(3, min_cluster_size // 2)
        changed = True
        while changed:
            changed = False
            for lab in list(cluster_map.keys()):
                indices = cluster_map[lab]
                if len(indices) <= two_pass_size:
                    continue
                sub_labels = HDBSCAN(
                    min_cluster_size=sub_mcs,
                    min_samples=max(1, min_samples // 2),
                    cluster_selection_method="leaf",
                ).fit_predict(reduced[indices])
                n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
                if n_sub < 2:
                    continue
                del cluster_map[lab]
                split_count += 1
                changed = True
                for si, sl in enumerate(sub_labels):
                    orig_i = indices[si]
                    if sl == -1:
                        noise_idx.append(orig_i)
                    else:
                        new_lab = next_label + int(sl)
                        cluster_map.setdefault(new_lab, []).append(orig_i)
                next_label = max(cluster_map.keys(), default=next_label) + 1
        if split_count:
            log.info("  Two-pass: split %d clusters (threshold=%d)", split_count, two_pass_size)

    # 3. Noise recovery
    if noise_recovery_sim > 0 and cluster_map:
        centroids = {lab: embeddings[idx].mean(axis=0) for lab, idx in cluster_map.items()}
        recovered = 0
        remaining = []
        for i in noise_idx:
            emb = embeddings[i]
            best_sim, best_lab = -1.0, -1
            for lab, cent in centroids.items():
                cn = np.linalg.norm(cent)
                if cn > 0:
                    s = float(np.dot(emb, cent) / (np.linalg.norm(emb) * cn))
                    if s > best_sim:
                        best_sim, best_lab = s, lab
            if best_sim >= noise_recovery_sim:
                cluster_map[best_lab].append(i)
                recovered += 1
            else:
                remaining.append(i)
        noise_idx = remaining
        if recovered:
            log.info("  Recovery: assigned %d noise articles (sim>=%.2f)", recovered, noise_recovery_sim)

    # 4. Conditional merge (with named entity overlap boost)
    # Entity overlap can boost effective similarity by up to ENTITY_BOOST_MAX,
    # allowing clusters that share key named entities (persons, orgs) to merge
    # even when centroid sim is slightly below merge_sim.
    ENTITY_BOOST_MAX = 0.12
    if merge_sim > 0 and len(cluster_map) > 1:
        cluster_centroids = {}
        cluster_quality = {}
        cluster_titles: dict[int, list[str]] = {}
        for lab, indices in cluster_map.items():
            cluster_centroids[lab] = embeddings[indices].mean(axis=0)
            _, avg = _centroid_sim(embeddings, indices)
            cluster_quality[lab] = avg
            cluster_titles[lab] = [titles[i] for i in indices]
        merged_count = 0
        merged = True
        while merged:
            merged = False
            labs = list(cluster_centroids.keys())
            best_pair, best_s = None, -1.0
            for ii in range(len(labs)):
                for jj in range(ii + 1, len(labs)):
                    a, b = labs[ii], labs[jj]
                    if cluster_quality[a] < merge_min_quality or cluster_quality[b] < merge_min_quality:
                        continue
                    ca, cb = cluster_centroids[a], cluster_centroids[b]
                    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
                    if na > 0 and nb > 0:
                        s = float(np.dot(ca, cb) / (na * nb))
                        # Boost by named entity overlap (max +ENTITY_BOOST_MAX)
                        overlap = _entity_overlap(cluster_titles[a], cluster_titles[b])
                        s_eff = s + overlap * ENTITY_BOOST_MAX
                        if s_eff > best_s:
                            best_s, best_pair = s_eff, (a, b)
            if best_pair and best_s >= merge_sim:
                a, b = best_pair
                keep, drop = (a, b) if len(cluster_map[a]) >= len(cluster_map[b]) else (b, a)
                cluster_map[keep].extend(cluster_map.pop(drop))
                cluster_centroids[keep] = embeddings[cluster_map[keep]].mean(axis=0)
                _, new_avg = _centroid_sim(embeddings, cluster_map[keep])
                cluster_quality[keep] = new_avg
                cluster_titles[keep] = cluster_titles[keep] + cluster_titles.pop(drop)
                del cluster_centroids[drop], cluster_quality[drop]
                merged_count += 1
                merged = True
        if merged_count:
            log.info("  Merge: %d pairs merged, %d clusters remain", merged_count, len(cluster_map))

    # 5. Force-split (k-means fallback for dense blobs)
    if force_split_size > 0:
        from sklearn.cluster import KMeans
        next_label = max(cluster_map.keys(), default=-1) + 1
        fs_count = 0
        for lab in list(cluster_map.keys()):
            indices = cluster_map[lab]
            if len(indices) <= force_split_size:
                continue
            _, avg_sim = _centroid_sim(embeddings, indices)
            if avg_sim >= force_split_max_sim:
                continue
            k = math.ceil(len(indices) / 10)
            if k < 2:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings[indices])
            del cluster_map[lab]
            fs_count += 1
            for si, sl in enumerate(km.labels_):
                new_lab = next_label + int(sl)
                cluster_map.setdefault(new_lab, []).append(indices[si])
            next_label = max(cluster_map.keys(), default=next_label) + 1
        if fs_count:
            log.info("  Force-split: %d clusters split by k-means", fs_count)

    # 6. Dissolve junk (final cleanup)
    if dissolve_threshold > 0:
        to_dissolve = []
        cluster_sims = []
        for lab, indices in cluster_map.items():
            _, avg_sim = _centroid_sim(embeddings, indices)
            cluster_sims.append((lab, len(indices), avg_sim,
                                 [titles[i][:40] for i in indices[:3]]))
            if avg_sim < dissolve_threshold:
                to_dissolve.append(lab)
        cluster_sims.sort(key=lambda x: x[2])
        for lab, sz, sim, sample in cluster_sims:
            marker = " *** DISSOLVE" if sim < dissolve_threshold else ""
            log.info("  Cluster %d (%d articles, avg_sim=%.3f)%s: %s",
                     lab, sz, sim, marker, " | ".join(sample))
        dissolved = sum(len(cluster_map[lab]) for lab in to_dissolve)
        for lab in to_dissolve:
            noise_idx.extend(cluster_map.pop(lab))
        if to_dissolve:
            log.info("  Dissolve: removed %d junk clusters (%d articles, avg_sim<%.2f)",
                     len(to_dissolve), dissolved, dissolve_threshold)

    # --- Build output ---
    clusters = []
    for lab in sorted(cluster_map, key=lambda k: len(cluster_map[k]), reverse=True):
        indices = cluster_map[lab]
        clusters.append({
            "label": lab,
            "article_ids": [id_list[i] for i in indices],
            "titles": [titles[i] for i in indices],
        })

    noise_ids = [id_list[i] for i in noise_idx]

    log.info("Clustering complete: %d clusters (%d raw → %d refined), %d noise articles (%.0f%%)",
             len(clusters), n_raw, len(clusters), len(noise_ids),
             len(noise_ids) / len(articles) * 100)
    return {
        "clusters": clusters,
        "noise_ids": noise_ids,
        "embeddings": embeddings,
    }
