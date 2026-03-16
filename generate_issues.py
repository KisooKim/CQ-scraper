"""Generate daily issues — Entry point for the clustering pipeline.

Usage:
    python -m generate_issues                   # process today
    python -m generate_issues --date 20260313   # process specific date
"""

import argparse
import logging
import time
from datetime import datetime, timezone, timedelta

from db import get_client
from clustering import cluster_articles
from issue_namer import name_and_merge_issues

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

MIN_ARTICLES = 20  # Skip clustering if fewer articles than this


def load_articles(publish_date: str) -> list[dict]:
    """Load articles for a date from Supabase.

    Args:
        publish_date: date string in YYYY-MM-DD format

    Returns:
        list of dicts with id, title, press_name
    """
    client = get_client()
    resp = (client.table("articles")
            .select("id, title, press:press_id(name)")
            .eq("publish_date", publish_date)
            .order("id")
            .execute())

    articles = []
    for row in resp.data:
        articles.append({
            "id": row["id"],
            "title": row["title"],
            "press_name": row["press"]["name"] if row.get("press") else "",
        })

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


def save_issues(publish_date: str, issues: list[dict]):
    """Save issues and their article links to Supabase."""
    client = get_client()

    for issue in issues:
        # Insert issue
        resp = (client.table("daily_issues")
                .insert({
                    "publish_date": publish_date,
                    "name": issue["name"],
                    "article_count": issue["article_count"],
                })
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


def run(target_date: str):
    """Run the full clustering pipeline for a date (YYYYMMDD)."""
    publish_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"
    log.info("Generating daily issues for %s", publish_date)

    start = time.time()

    # Load articles
    articles = load_articles(publish_date)
    log.info("Loaded %d articles", len(articles))

    if len(articles) < MIN_ARTICLES:
        log.warning("Only %d articles found (min %d). Skipping clustering.",
                     len(articles), MIN_ARTICLES)
        return

    # Clear previous results for idempotent re-runs
    clear_existing_issues(publish_date)

    # Step 1: Cluster
    cluster_result = cluster_articles(articles)

    # Step 2: Name and merge
    issues = name_and_merge_issues(cluster_result, articles)
    log.info("Generated %d issues", len(issues))

    # Step 3: Save
    save_issues(publish_date, issues)

    duration = time.time() - start
    total_assigned = sum(iss["article_count"] for iss in issues)
    log.info(
        "Done. date=%s | issues=%d | articles_assigned=%d/%d | duration=%.1fs",
        publish_date, len(issues), total_assigned, len(articles), duration,
    )


def main():
    parser = argparse.ArgumentParser(description="CQ Daily Issue Generator")
    parser.add_argument(
        "--date",
        help="Target date in YYYYMMDD format (default: today KST)",
        default=None,
    )
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(KST).strftime("%Y%m%d")

    run(target_date)


if __name__ == "__main__":
    main()
