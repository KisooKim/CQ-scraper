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
    scrape_press_page,
    scrape_article_detail,
    fetch_engagement,
)
from db import (
    get_existing_urls,
    get_press_list,
    save_article,
    create_scrape_run,
    complete_scrape_run,
    get_articles_for_engagement_update,
    update_article_engagement,
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
        _delay()  # delay before newspaper page fetch too
        articles = scrape_press_page(client, press_code, target_date)
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

            # Use the article's actual publish date (from data-date-time attr).
            # If parsing fails, skip — do NOT fall back to run date, which
            # historically polluted publish_date on re-scraped old articles.
            parsed_dt = detail.get("publish_datetime")
            if not parsed_dt or len(parsed_dt) < 10:
                log.warning("  [%s] Missing publish_datetime, skipping: %s",
                            press_name, art["url"])
                errors += 1
                error_messages.append(
                    f"{press_name}: {art['title'][:30]}: no publish_datetime")
                continue
            article_publish_date = parsed_dt[:10]  # YYYY-MM-DD

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
                is_portrait_thumb=False,
                publish_date=article_publish_date,
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


def run(target_date: str, workers: int = 5, press_type: str | None = None):
    """Scrape newspapers for a given date (YYYYMMDD).

    Args:
        press_type: 'newspaper' for newspaper outlets only,
                    'other' for non-newspaper (wire/broadcast/online/etc.),
                    None for all.
    """
    label = press_type or "all"
    log.info("Starting scrape for date=%s type=%s with %d workers", target_date, label, workers)
    start = time.time()

    # Convert YYYYMMDD to YYYY-MM-DD for DB
    publish_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"

    run_id = create_scrape_run(publish_date)
    existing_urls = get_existing_urls(publish_date)
    press_list = get_press_list(press_type=press_type)

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

    scrape_duration = time.time() - start

    # --- Engagement update phase ---
    eng_start = time.time()
    eng_articles = get_articles_for_engagement_update(publish_date)
    eng_updated = 0
    eng_errors = 0

    if eng_articles:
        log.info("Engagement update: %d articles to check", len(eng_articles))

        def _update_one(art: dict) -> bool:
            try:
                client = _get_client()
                resp_count, comm_count = fetch_engagement(client, art["url"])
                update_article_engagement(art["id"], resp_count, comm_count)
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_update_one, art): art for art in eng_articles}
            for future in as_completed(futures):
                if future.result():
                    eng_updated += 1
                else:
                    eng_errors += 1

    eng_duration = time.time() - eng_start
    total_duration = time.time() - start

    complete_scrape_run(
        run_id,
        press_count=len(press_list),
        article_count=total_articles,
        error_count=total_errors,
        duration_sec=total_duration,
        error_log="\n".join(all_error_messages) if all_error_messages else "",
    )

    log.info(
        "Done. date=%s | new=%d | errors=%d | scrape=%.0fs | engagement=%d updated (%.0fs) | total=%.0fs",
        target_date, total_articles, total_errors, scrape_duration,
        eng_updated, eng_duration, total_duration,
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
    parser.add_argument(
        "--type",
        choices=["newspaper", "other"],
        default=None,
        help="Press type filter: 'newspaper' (지면), 'other' (통신/방송/온라인), or all if omitted",
    )
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(KST).strftime("%Y%m%d")

    run(target_date, workers=args.workers, press_type=args.type)


if __name__ == "__main__":
    main()
