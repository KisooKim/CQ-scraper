"""CQ News Scraper — Entry point.

Usage:
    python -m main                          # scrape today, 5 parallel workers
    python -m main --date 20260314          # scrape specific date
    python -m main --workers 10             # use 10 parallel workers
"""

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

from crawlers.naver import (
    _get_client,
    _delay,
    _extract_ids_from_url,
    scrape_newspaper_page,
    scrape_article_detail,
    fetch_engagement,
)
from db import (
    get_existing_urls,
    get_press_list,
    save_article,
    create_scrape_run,
    complete_scrape_run,
)
from storage import make_r2_key, upload_article_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# Thread-safe set for dedup and counters
_url_lock = threading.Lock()
_counter_lock = threading.Lock()


def _scrape_one_press(press: dict, target_date: str, publish_date: str,
                      existing_urls: set) -> dict:
    """Scrape all articles for one newspaper. Returns stats dict."""
    press_code = press["code"]
    press_name = press["name"]
    press_id = press["id"]
    log.info("Scraping press: %s (%s)", press_name, press_code)

    articles_saved = 0
    errors = 0
    error_messages = []

    # Each thread gets its own httpx client
    client = _get_client()

    try:
        articles = scrape_newspaper_page(client, press_code, target_date)
        log.info("  [%s] Found %d articles", press_name, len(articles))
    except Exception as e:
        log.error("  [%s] Failed newspaper page: %s", press_name, e)
        return {"saved": 0, "errors": 1, "error_messages": [f"{press_name}: newspaper page error: {e}"]}

    for art in articles:
        with _url_lock:
            if art["url"] in existing_urls:
                continue
            existing_urls.add(art["url"])  # Reserve URL early to prevent duplicates across threads

        try:
            _delay()
            detail = scrape_article_detail(client, art["url"])
            response_count, comment_count = fetch_engagement(client, art["url"])

            # Upload body text to R2
            r2_key = None
            body_text = detail.get("body_text")
            if body_text:
                ids = _extract_ids_from_url(art["url"])
                if ids:
                    r2_key = make_r2_key(ids[0], target_date, ids[1])
                    upload_article_text(
                        r2_key, body_text,
                        title=detail.get("title"),
                        journalist_names=detail.get("journalist_names", []),
                    )

            article_id = save_article(
                press_id=press_id,
                title=detail.get("title") or art["title"],
                url=art["url"],
                r2_key=r2_key,
                original_url=detail.get("original_url"),
                thumbnail_url=art.get("thumbnail_url"),
                publish_date=publish_date,
                layout_section=art["section"],
                layout_position=art["position"],
                response_count=response_count,
                comment_count=comment_count,
                journalist_names=detail.get("journalist_names", []),
                image_urls=detail.get("image_urls", []),
            )

            if article_id:
                articles_saved += 1
                log.info("  [%s] Saved: %s", press_name, art["title"][:50])

        except Exception as e:
            log.error("  [%s] Failed article %s: %s", press_name, art["url"], e)
            errors += 1
            error_messages.append(f"{press_name}: {art['title'][:30]}: {e}")

    log.info("  [%s] Done: %d saved, %d errors", press_name, articles_saved, errors)
    return {"saved": articles_saved, "errors": errors, "error_messages": error_messages}


def run(target_date: str, workers: int = 5):
    """Scrape all newspapers for a given date (YYYYMMDD)."""
    log.info("Starting scrape for date=%s with %d workers", target_date, workers)
    start = time.time()

    # Convert YYYYMMDD to YYYY-MM-DD for DB
    publish_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"

    run_id = create_scrape_run(publish_date)
    existing_urls = get_existing_urls(publish_date)
    press_list = get_press_list()

    total_articles = 0
    total_errors = 0
    all_error_messages = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_scrape_one_press, press, target_date, publish_date, existing_urls): press
            for press in press_list
        }

        for future in as_completed(futures):
            result = future.result()
            total_articles += result["saved"]
            total_errors += result["errors"]
            all_error_messages.extend(result.get("error_messages", []))

    duration = time.time() - start
    complete_scrape_run(
        run_id,
        press_count=len(press_list),
        article_count=total_articles,
        error_count=total_errors,
        duration_sec=duration,
        error_log="\n".join(all_error_messages) if all_error_messages else "",
    )

    log.info(
        "Done. date=%s | articles=%d | errors=%d | duration=%.1fs",
        target_date, total_articles, total_errors, duration,
    )


def main():
    parser = argparse.ArgumentParser(description="CQ News Scraper")
    parser.add_argument(
        "--date",
        help="Target date in YYYYMMDD format (default: today KST)",
        default=None,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(KST).strftime("%Y%m%d")

    run(target_date, workers=args.workers)


if __name__ == "__main__":
    main()
