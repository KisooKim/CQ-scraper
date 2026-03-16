"""Issue naming module — LLM-based cluster naming, merging, and noise reassignment.

Falls back to TF-IDF keyword naming if Claude API is unavailable.
"""

import json
import logging
import os
import re

log = logging.getLogger(__name__)


def _build_llm_prompt(clusters: list[dict], noise_titles: list[dict]) -> str:
    """Build the prompt for Claude to name/merge clusters and reassign noise."""
    lines = ["다음은 한국 뉴스 기사를 임베딩 기반으로 클러스터링한 결과입니다."]
    lines.append("각 클러스터의 기사 제목을 보고, 아래 작업을 수행하세요:\n")
    lines.append("1. 각 클러스터에 한국어로 간결한 이슈 이름을 붙이세요 (예: '미-이란 전쟁·중동 위기')")
    lines.append("2. 내용이 비슷한 클러스터는 하나로 합치세요")
    lines.append("3. 노이즈 기사들을 적절한 이슈에 배정하세요. 어디에도 맞지 않는 기사는 '기타'로 분류하세요\n")

    lines.append("=== 클러스터 목록 ===")
    for i, cluster in enumerate(clusters):
        titles_sample = cluster["titles"][:10]
        lines.append(f"\n클러스터 {i} ({len(cluster['article_ids'])}건):")
        for t in titles_sample:
            lines.append(f"  - {t}")
        if len(cluster["titles"]) > 10:
            lines.append(f"  ... 외 {len(cluster['titles']) - 10}건")

    if noise_titles:
        lines.append(f"\n=== 노이즈 기사 ({len(noise_titles)}건) ===")
        for item in noise_titles[:50]:
            lines.append(f"  - [id:{item['id']}] {item['title']}")
        if len(noise_titles) > 50:
            lines.append(f"  ... 외 {len(noise_titles) - 50}건")

    lines.append("\n=== 응답 형식 (JSON) ===")
    lines.append("다음 JSON 형식으로 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.")
    lines.append("""```json
{
  "issues": [
    {
      "name": "이슈 이름",
      "source_clusters": [0, 3],
      "noise_article_ids": [123, 456]
    }
  ]
}
```""")
    lines.append("\n- source_clusters: 이 이슈에 합칠 클러스터 번호 리스트")
    lines.append("- noise_article_ids: 이 이슈에 배정할 노이즈 기사 ID 리스트")
    lines.append("- 모든 클러스터가 정확히 하나의 이슈에 포함되어야 합니다")
    lines.append("- 어디에도 맞지 않는 노이즈 기사는 '기타' 이슈에 넣으세요")

    return "\n".join(lines)


def _call_claude(prompt: str) -> dict | None:
    """Call Claude API and parse JSON response. Returns None on failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set, skipping LLM naming")
        return None

    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed, skipping LLM naming")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Extract JSON from response (may be wrapped in ```json ... ```)
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        return json.loads(text)

    except Exception as e:
        log.error("Claude API call failed: %s", e)
        return None


def _fallback_keyword_naming(clusters: list[dict]) -> list[str]:
    """Name clusters using TF-IDF top keywords when LLM is unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Collect all titles for fitting the vectorizer
    all_titles = []
    for cluster in clusters:
        all_titles.extend(cluster["titles"])

    if not all_titles:
        return ["기타"] * len(clusters)

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_titles)
    feature_names = vectorizer.get_feature_names_out()

    names = []
    offset = 0
    for cluster in clusters:
        n = len(cluster["titles"])
        if n == 0:
            names.append("기타")
            continue
        cluster_slice = tfidf_matrix[offset:offset + n]
        mean_tfidf = cluster_slice.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-3:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        names.append(" / ".join(keywords))
        offset += n

    return names


def name_and_merge_issues(clusters: list[dict], articles: list[dict]) -> list[dict]:
    """Name clusters, merge similar ones, and reassign noise articles.

    Args:
        clusters: dict from cluster_articles() — has 'clusters', 'noise_ids', 'embeddings'
        articles: original article list (id, title, press_name)

    Returns:
        list of issues: [{name: str, article_ids: list[int], article_count: int}]
    """
    raw_clusters = clusters["clusters"]
    noise_ids = clusters["noise_ids"]

    title_map = {a["id"]: a["title"] for a in articles}
    noise_articles = [{"id": nid, "title": title_map.get(nid, "")} for nid in noise_ids]

    if not raw_clusters:
        # No clusters found — everything is one "기타" group
        all_ids = [a["id"] for a in articles]
        return [{"name": "기타", "article_ids": all_ids, "article_count": len(all_ids)}]

    # Try LLM naming
    prompt = _build_llm_prompt(raw_clusters, noise_articles)
    llm_result = _call_claude(prompt)

    if llm_result and "issues" in llm_result:
        return _apply_llm_result(llm_result, raw_clusters, noise_ids, title_map)
    else:
        log.info("Using fallback keyword naming")
        return _apply_fallback(raw_clusters, noise_ids, title_map)


def _apply_llm_result(llm_result: dict, raw_clusters: list[dict],
                      noise_ids: list[int], title_map: dict) -> list[dict]:
    """Build issue list from LLM merge/naming decisions."""
    issues = []
    assigned_noise = set()

    for item in llm_result["issues"]:
        name = item["name"]
        article_ids = []

        # Collect articles from source clusters
        for idx in item.get("source_clusters", []):
            if 0 <= idx < len(raw_clusters):
                article_ids.extend(raw_clusters[idx]["article_ids"])

        # Add reassigned noise articles
        for nid in item.get("noise_article_ids", []):
            if nid in title_map and nid not in assigned_noise:
                article_ids.append(nid)
                assigned_noise.add(nid)

        if article_ids:
            issues.append({
                "name": name,
                "article_ids": article_ids,
                "article_count": len(article_ids),
            })

    # Remaining noise goes to "기타"
    remaining_noise = [nid for nid in noise_ids if nid not in assigned_noise]
    if remaining_noise:
        # Check if "기타" already exists
        gita_issue = next((iss for iss in issues if iss["name"] == "기타"), None)
        if gita_issue:
            gita_issue["article_ids"].extend(remaining_noise)
            gita_issue["article_count"] = len(gita_issue["article_ids"])
        else:
            issues.append({
                "name": "기타",
                "article_ids": remaining_noise,
                "article_count": len(remaining_noise),
            })

    # Verify all clusters were assigned — catch any missed by the LLM
    assigned_cluster_indices = set()
    for item in llm_result["issues"]:
        assigned_cluster_indices.update(item.get("source_clusters", []))

    missed_ids = []
    for idx, cluster in enumerate(raw_clusters):
        if idx not in assigned_cluster_indices:
            log.warning("Cluster %d not assigned by LLM, adding to 기타", idx)
            missed_ids.extend(cluster["article_ids"])

    if missed_ids:
        gita_issue = next((iss for iss in issues if iss["name"] == "기타"), None)
        if gita_issue:
            gita_issue["article_ids"].extend(missed_ids)
            gita_issue["article_count"] = len(gita_issue["article_ids"])
        else:
            issues.append({
                "name": "기타",
                "article_ids": missed_ids,
                "article_count": len(missed_ids),
            })

    # Sort by article count descending
    issues.sort(key=lambda x: x["article_count"], reverse=True)
    return issues


def _apply_fallback(raw_clusters: list[dict], noise_ids: list[int],
                    title_map: dict) -> list[dict]:
    """Build issue list using TF-IDF keyword naming (no merging)."""
    names = _fallback_keyword_naming(raw_clusters)
    issues = []

    for name, cluster in zip(names, raw_clusters):
        issues.append({
            "name": name,
            "article_ids": cluster["article_ids"],
            "article_count": len(cluster["article_ids"]),
        })

    # All noise goes to "기타"
    if noise_ids:
        issues.append({
            "name": "기타",
            "article_ids": noise_ids,
            "article_count": len(noise_ids),
        })

    issues.sort(key=lambda x: x["article_count"], reverse=True)
    return issues
