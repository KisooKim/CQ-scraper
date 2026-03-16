"""Clustering module — Embed article titles, reduce with UMAP, cluster with HDBSCAN."""

import logging
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


def cluster_articles(articles: list[dict]) -> dict:
    """Cluster articles by title similarity.

    Args:
        articles: list of dicts with keys: id, title, press_name

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

    # Step 1: Embed
    log.info("Embedding %d titles ...", len(titles))
    t0 = time.time()
    model = _get_model()
    embeddings = model.encode(titles, show_progress_bar=False, batch_size=32)
    log.info("  Embedding done in %.1fs", time.time() - t0)

    # Step 2: UMAP dimensionality reduction
    log.info("UMAP reduction ...")
    reduced = UMAP(
        n_components=10,
        n_neighbors=20,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    # Step 3: HDBSCAN clustering
    log.info("HDBSCAN clustering ...")
    clusterer = HDBSCAN(
        min_cluster_size=8,
        min_samples=3,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    log.info("  Found %d clusters, %d noise points", n_clusters, noise_count)

    # Build cluster groups
    cluster_map: dict[int, dict] = {}
    noise_ids: list[int] = []

    for i, label in enumerate(labels):
        if label == -1:
            noise_ids.append(id_list[i])
        else:
            label_int = int(label)
            if label_int not in cluster_map:
                cluster_map[label_int] = {"label": label_int, "article_ids": [], "titles": []}
            cluster_map[label_int]["article_ids"].append(id_list[i])
            cluster_map[label_int]["titles"].append(titles[i])

    clusters = sorted(cluster_map.values(), key=lambda c: len(c["article_ids"]), reverse=True)

    log.info("Clustering complete: %d clusters, %d noise articles", len(clusters), len(noise_ids))
    return {
        "clusters": clusters,
        "noise_ids": noise_ids,
        "embeddings": embeddings,
    }
