"""Generate daily issues — Entry point for the clustering pipeline.

Usage:
    python -m generate_issues                   # process today
    python -m generate_issues --date 20260313   # process specific date
"""

import argparse
import concurrent.futures
import logging
import time
from datetime import datetime, timezone, timedelta

import numpy as np

from db import get_client
from clustering import cluster_articles
from llm_cluster import llm_cluster_articles
from llm_refine import llm_refine_pass
from issue_namer import name_and_merge_issues
from storage import download_article_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

MIN_ARTICLES = 20  # Skip clustering if fewer articles than this

# Daily boundary: 06:45 KST — articles scraped before this time belong to
# the current date's cluster; articles after belong to the next day's cluster.
BOUNDARY_HOUR = 6
BOUNDARY_MINUTE = 45


def load_articles(publish_date: str) -> list[dict]:
    """Load articles for a clustering window from Supabase, then fetch body text from R2.

    The window is defined as [previous day 06:45 KST, current day 06:45 KST).
    For publish_date "2026-03-21", this loads articles scraped between
    2026-03-20 06:45 KST and 2026-03-21 06:45 KST.

    Args:
        publish_date: date string in YYYY-MM-DD format (the cluster date)

    Returns:
        list of dicts with id, title, press_name, body_text
    """
    from datetime import date as date_type

    d = date_type.fromisoformat(publish_date)
    prev = d - timedelta(days=1)

    # Window: prev day 06:45 KST ~ current day 06:45 KST (in UTC)
    window_start = datetime(prev.year, prev.month, prev.day,
                            BOUNDARY_HOUR, BOUNDARY_MINUTE, tzinfo=KST)
    window_end = datetime(d.year, d.month, d.day,
                          BOUNDARY_HOUR, BOUNDARY_MINUTE, tzinfo=KST)

    start_utc = window_start.astimezone(timezone.utc).isoformat()
    end_utc = window_end.astimezone(timezone.utc).isoformat()

    log.info("Loading articles scraped between %s and %s KST",
             window_start.strftime("%Y-%m-%d %H:%M"),
             window_end.strftime("%Y-%m-%d %H:%M"))

    client = get_client()
    resp = (client.table("articles")
            .select("id, title, r2_key, press:press_id(name)")
            .gte("scraped_at", start_utc)
            .lt("scraped_at", end_utc)
            .order("id")
            .execute())

    articles = []
    for row in resp.data:
        articles.append({
            "id": row["id"],
            "title": row["title"],
            "r2_key": row.get("r2_key"),
            "press_name": row["press"]["name"] if row.get("press") else "",
            "body_text": "",
        })

    # Fetch body text from R2 in parallel
    keys_needed = [(i, a["r2_key"]) for i, a in enumerate(articles) if a["r2_key"]]
    if keys_needed:
        log.info("Fetching body text from R2 for %d articles ...", len(keys_needed))
        t0 = time.time()
        def fetch(idx_key):
            idx, key = idx_key
            data = download_article_text(key)
            return idx, (data.get("body_text", "") if data else "")
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as ex:
            for idx, body in ex.map(fetch, keys_needed):
                articles[idx]["body_text"] = body
        log.info("  R2 fetch done in %.1fs", time.time() - t0)

    return articles


def clear_existing_issues(publish_date: str):
    """Delete existing daily issues for a date (idempotent re-run)."""
    client = get_client()

    # Find existing issue IDs for this date
    resp = (client.table("daily_issues")
            .select("id")
            .eq("publish_date", publish_date)
            .execute())

    if not resp.data:
        return

    issue_ids = [row["id"] for row in resp.data]
    log.info("Clearing %d existing issues for %s", len(issue_ids), publish_date)

    # Delete junction rows first (CASCADE should handle this, but be explicit)
    for issue_id in issue_ids:
        (client.table("daily_issue_articles")
         .delete()
         .eq("issue_id", issue_id)
         .execute())

    # Delete issue rows
    (client.table("daily_issues")
     .delete()
     .eq("publish_date", publish_date)
     .execute())


def save_issues(publish_date: str, issues: list[dict],
                issue_centroids: dict[str, list[float]] | None = None):
    """Save issues and their article links to Supabase."""
    client = get_client()
    centroids = issue_centroids or {}

    for issue in issues:
        # Build row data
        row = {
            "publish_date": publish_date,
            "name": issue["name"],
            "article_count": issue["article_count"],
            "is_published": False,
        }
        centroid = centroids.get(issue["name"])
        if centroid:
            # pgvector expects a string like '[0.1, 0.2, ...]'
            row["centroid_embedding"] = str(centroid)

        # Insert issue
        resp = (client.table("daily_issues")
                .insert(row)
                .execute())

        issue_id = resp.data[0]["id"]

        # Insert article links in batches
        if issue["article_ids"]:
            rows = [{"issue_id": issue_id, "article_id": aid}
                    for aid in issue["article_ids"]]
            # Supabase has a row limit per insert; batch in groups of 500
            for i in range(0, len(rows), 500):
                batch = rows[i:i + 500]
                (client.table("daily_issue_articles")
                 .insert(batch)
                 .execute())

        log.info("  Saved issue: [%d articles] %s", issue["article_count"], issue["name"])


def check_existing_issues(publish_date: str) -> str:
    """Check issue state for a date.

    Returns:
        "none"      — no issues exist
        "draft"     — issues exist but not yet published (safe to regenerate)
        "published" — issues are published (protected)
    """
    client = get_client()
    resp = (client.table("daily_issues")
            .select("id, is_published")
            .eq("publish_date", publish_date)
            .limit(1)
            .execute())
    if not resp.data:
        return "none"
    return "published" if resp.data[0]["is_published"] else "draft"


def run(target_date: str, *, force: bool = False, use_llm_cluster: bool = False,
        snapshot_only: bool = False):
    """Run the full clustering pipeline for a date (YYYYMMDD).

    snapshot_only=True: cluster + name, save to snapshot JSON only, skip DB write.
    """
    publish_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"
    log.info("Generating daily issues for %s%s", publish_date,
             " [snapshot-only]" if snapshot_only else "")

    if not snapshot_only:
        # Skip if issues are already published (protect finalized results)
        state = check_existing_issues(publish_date)
        if state == "published" and not force:
            log.info("Published issues already exist for %s. Use --force to regenerate. Skipping.",
                     publish_date)
            return

    start = time.time()

    # Load articles
    articles = load_articles(publish_date)
    log.info("Loaded %d articles", len(articles))

    if len(articles) < MIN_ARTICLES:
        log.warning("Only %d articles found (min %d). Skipping clustering.",
                     len(articles), MIN_ARTICLES)
        return

    if not snapshot_only:
        # Clear previous results (only reached with --force if issues exist)
        clear_existing_issues(publish_date)

    # Step 1: Cluster (no merge, no noise recovery — prevents chaining)
    if use_llm_cluster:
        log.info("Using pure LLM clustering (no embeddings)")
        cluster_result = llm_cluster_articles(articles)
        _snapshot_label = "LLM"
    else:
        cluster_result = cluster_articles(
            articles,
            min_samples=5,
            merge_sim=0,           # disable merge — prevents chaining
            noise_recovery_sim=0,  # disable noise recovery — keep clusters clean
        )
        _snapshot_label = "임베딩"

    # Step 2: Name and merge
    issues = name_and_merge_issues(cluster_result, articles)
    log.info("Generated %d issues", len(issues))

    # Step 2.5: Compute centroid embeddings for each issue
    embeddings = cluster_result.get("embeddings")
    issue_centroids = {}
    if embeddings is not None and len(embeddings) > 0:
        id_to_idx = {a["id"]: i for i, a in enumerate(articles)}
        for issue in issues:
            indices = [id_to_idx[aid] for aid in issue["article_ids"] if aid in id_to_idx]
            if indices:
                centroid = embeddings[indices].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                issue_centroids[issue["name"]] = centroid.tolist()
        log.info("Computed centroid embeddings for %d/%d issues", len(issue_centroids), len(issues))

    duration = time.time() - start
    total_assigned = sum(iss["article_count"] for iss in issues)

    if snapshot_only:
        # Save to snapshot JSON only — do not touch DB
        _save_snapshot_from_issues(publish_date, issues, _snapshot_label, total_assigned, duration, articles)
        return

    # Step 3: Save to DB
    if not force and check_existing_issues(publish_date) == "published":
        log.warning("Issues were published by another run during clustering. Skipping save.")
        return

    save_issues(publish_date, issues, issue_centroids)
    log.info(
        "Done. date=%s | issues=%d | articles_assigned=%d/%d | duration=%.1fs",
        publish_date, len(issues), total_assigned, len(articles), duration,
    )

    # Auto-save snapshot for comparison
    try:
        from save_snapshot import export_snapshot, slugify
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%H%M")
        run_id = f"{slugify(_snapshot_label)}_{ts}"
        export_snapshot(publish_date, f"{_snapshot_label} {ts}", run_id)
    except Exception as e:
        log.warning("Snapshot save failed (non-critical): %s", e)


def _save_snapshot_from_issues(publish_date: str, issues: list[dict],
                                label: str, total: int, duration: float,
                                articles: list[dict] | None = None):
    """Save clustering results directly to snapshot JSON without hitting DB."""
    import datetime as _dt
    from save_snapshot import slugify
    from pathlib import Path as _Path

    ts = _dt.datetime.now().strftime("%H%M")
    run_id = f"{slugify(label)}_{ts}"

    # Build lookup from original articles list
    art_map = {a["id"]: a for a in (articles or [])}

    non_gita = [iss for iss in issues if iss.get("name") != "기타"]
    noise = next((iss["article_count"] for iss in issues if iss.get("name") == "기타"), 0)
    noise_pct = round(noise / total * 100) if total else 0

    clusters = []
    for iss in non_gita:
        arts = []
        for aid in iss.get("article_ids", []):
            a = art_map.get(aid)
            if a:
                arts.append({"id": aid, "title": a["title"],
                             "press": a.get("press_name", "")})
        clusters.append({
            "name": iss.get("name", "?"),
            "size": iss.get("article_count", len(arts)),
            "articles": arts,
        })

    snapshot = {
        "run_id": run_id,
        "label": f"{label} {ts}",
        "date": publish_date,
        "saved_at": _dt.datetime.now().isoformat(),
        "stats": {
            "clusters": len(non_gita),
            "total_articles": total,
            "noise_articles": noise,
            "noise_pct": noise_pct,
            "duration_s": round(duration),
        },
        "clusters": clusters,
    }

    out_dir = _Path(__file__).parent / "feedback" / "snapshots" / publish_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"
    import json as _json
    out_file.write_text(_json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "Snapshot saved → %s  (%d clusters, %d%% noise, %.0fs)",
        out_file, len(non_gita), noise_pct, duration,
    )


def main():
    parser = argparse.ArgumentParser(description="CQ Daily Issue Generator")
    parser.add_argument(
        "--date",
        help="Target date in YYYYMMDD format (default: today KST)",
        default=None,
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        dest="snapshot_only",
        help="Cluster and save to snapshot JSON only — do not write to DB",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if issues already exist",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use pure LLM clustering instead of embeddings",
    )
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(KST).strftime("%Y%m%d")

    run(target_date, force=args.force, use_llm_cluster=args.llm,
        snapshot_only=args.snapshot_only)


if __name__ == "__main__":
    main()
