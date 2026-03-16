"""Show daily issues for Claude Code renaming, or apply renames.

Usage:
    python -m rename_issues --date 20260313              # show issues + sample titles
    python -m rename_issues --date 20260313 --apply '{   # apply renames
        "1": "기타",
        "2": "AI·반도체 산업 동향"
    }'
"""

import argparse
import json
from datetime import datetime, timezone, timedelta

from db import get_client

KST = timezone(timedelta(hours=9))


def show_issues(publish_date: str):
    """Print all issues with sample titles for a date."""
    client = get_client()

    resp = (client.table("daily_issues")
            .select("id, name, article_count")
            .eq("publish_date", publish_date)
            .order("article_count", desc=True)
            .execute())

    if not resp.data:
        print(f"No issues found for {publish_date}")
        return

    print(f"\n=== Daily Issues for {publish_date} ({len(resp.data)} issues) ===\n")

    for issue in resp.data:
        issue_id = issue["id"]
        print(f"[{issue_id:3d}] [{issue['article_count']:3d}건] {issue['name']}")

        # Get top 5 articles by response_count
        art_resp = (client.table("daily_issue_articles")
                    .select("article_id")
                    .eq("issue_id", issue_id)
                    .execute())
        article_ids = [r["article_id"] for r in art_resp.data]

        if article_ids:
            titles_resp = (client.table("articles")
                          .select("title, response_count")
                          .in_("id", article_ids[:8])
                          .order("response_count", desc=True)
                          .execute())
            for t in titles_resp.data[:5]:
                print(f"       - [{t['response_count']:3d}] {t['title']}")
        print()


def apply_renames(publish_date: str, renames_json: str):
    """Apply issue renames from a JSON string mapping id -> new_name."""
    renames = json.loads(renames_json)
    client = get_client()

    for issue_id_str, new_name in renames.items():
        issue_id = int(issue_id_str)
        resp = (client.table("daily_issues")
                .update({"name": new_name})
                .eq("id", issue_id)
                .execute())
        if resp.data:
            print(f"  [{issue_id:3d}] -> {new_name}")
        else:
            print(f"  [{issue_id:3d}] FAILED")

    print(f"\nRenamed {len(renames)} issues.")


def main():
    parser = argparse.ArgumentParser(description="CQ Issue Renamer")
    parser.add_argument(
        "--date",
        help="Target date in YYYYMMDD format (default: today KST)",
        default=None,
    )
    parser.add_argument(
        "--apply",
        help="JSON string mapping issue_id -> new_name",
        default=None,
    )
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(KST).strftime("%Y%m%d")

    publish_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"

    if args.apply:
        apply_renames(publish_date, args.apply)
    else:
        show_issues(publish_date)


if __name__ == "__main__":
    main()
