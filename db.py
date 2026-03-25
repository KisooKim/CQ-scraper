from supabase import create_client, Client

from config import settings

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def get_existing_urls(publish_date: str) -> set[str]:
    """Return set of article URLs already in DB for a given date.

    Paginates to handle >1000 articles (Supabase default limit).
    """
    client = get_client()
    urls: set[str] = set()
    offset = 0
    page_size = 1000
    while True:
        resp = (client.table("articles")
                .select("url")
                .eq("publish_date", publish_date)
                .range(offset, offset + page_size - 1)
                .execute())
        for row in resp.data:
            urls.add(row["url"])
        if len(resp.data) < page_size:
            break
        offset += page_size
    return urls


def get_press_list(press_type: str | None = None) -> list[dict]:
    """Return active press entries, optionally filtered by type.

    Args:
        press_type: 'newspaper' for newspaper outlets only,
                    'other' for non-newspaper (wire/broadcast/online/etc.),
                    None for all active press.
    """
    client = get_client()
    query = (client.table("press")
             .select("id, code, name, press_type")
             .eq("is_active", True)
             .order("id"))

    if press_type == "newspaper":
        query = query.eq("press_type", "newspaper")
    elif press_type == "other":
        query = query.neq("press_type", "newspaper")

    return query.execute().data


def find_or_create_journalist(name: str, press_id: int) -> int:
    """Find existing journalist or create new one. Return journalist id."""
    client = get_client()
    resp = (client.table("journalists")
            .select("id")
            .eq("name", name)
            .eq("press_id", press_id)
            .limit(1)
            .execute())
    if resp.data:
        return resp.data[0]["id"]

    resp = (client.table("journalists")
            .insert({"name": name, "press_id": press_id})
            .execute())
    return resp.data[0]["id"]


def save_article(*, press_id: int, title: str, url: str, r2_key: str | None,
                 original_url: str | None = None,
                 thumbnail_url: str | None = None,
                 is_portrait_thumb: bool = False,
                 publish_date: str, layout_section: str | None, layout_position: int,
                 response_count: int, comment_count: int,
                 journalist_names: list[str], image_urls: list[str]) -> int | None:
    """Insert article and related data. Return article id, or None if duplicate."""
    client = get_client()

    # Upsert article (update on conflict to fill in new fields on re-scrape)
    resp = (client.table("articles")
            .upsert({
                "press_id": press_id,
                "title": title,
                "url": url,
                "r2_key": r2_key,
                "original_url": original_url,
                "thumbnail_url": thumbnail_url,
                "is_portrait_thumb": is_portrait_thumb,
                "publish_date": publish_date,
                "layout_section": layout_section,
                "layout_position": layout_position,
                "response_count": response_count,
                "comment_count": comment_count,
            }, on_conflict="url")
            .execute())

    if not resp.data:
        return None
    article_id = resp.data[0]["id"]

    # Link journalists
    for name in journalist_names:
        journalist_id = find_or_create_journalist(name, press_id)
        (client.table("article_journalists")
         .upsert({"article_id": article_id, "journalist_id": journalist_id},
                 on_conflict="article_id,journalist_id", ignore_duplicates=True)
         .execute())

    # Save images
    if image_urls:
        rows = [{"article_id": article_id, "url": img_url, "display_order": i}
                for i, img_url in enumerate(image_urls)]
        client.table("article_images").insert(rows).execute()

    return article_id


def get_articles_for_engagement_update(publish_date: str, recent_hours: float = 2.0) -> list[dict]:
    """Get articles that need engagement updates.

    Strategy:
      - Articles scraped within recent_hours: ALL (engagement not settled yet)
      - Older articles: only those with response_count + comment_count > 0

    Returns list of dicts with id, url.
    """
    from datetime import datetime, timezone, timedelta

    client = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=recent_hours)).isoformat()
    articles = []
    page_size = 1000

    # 1. Recent articles (all)
    offset = 0
    while True:
        resp = (client.table("articles")
                .select("id, url")
                .eq("publish_date", publish_date)
                .gte("scraped_at", cutoff)
                .range(offset, offset + page_size - 1)
                .execute())
        articles.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size

    recent_ids = {a["id"] for a in articles}

    # 2. Older articles with engagement > 0
    offset = 0
    while True:
        resp = (client.table("articles")
                .select("id, url")
                .eq("publish_date", publish_date)
                .lt("scraped_at", cutoff)
                .gt("response_count", 0)
                .range(offset, offset + page_size - 1)
                .execute())
        for row in resp.data:
            if row["id"] not in recent_ids:
                articles.append(row)
        if len(resp.data) < page_size:
            break
        offset += page_size

    # Also get older articles with comments > 0
    offset = 0
    existing_ids = {a["id"] for a in articles}
    while True:
        resp = (client.table("articles")
                .select("id, url")
                .eq("publish_date", publish_date)
                .lt("scraped_at", cutoff)
                .gt("comment_count", 0)
                .range(offset, offset + page_size - 1)
                .execute())
        for row in resp.data:
            if row["id"] not in existing_ids:
                articles.append(row)
                existing_ids.add(row["id"])
        if len(resp.data) < page_size:
            break
        offset += page_size

    return articles


def update_article_engagement(article_id: int, response_count: int, comment_count: int):
    """Update engagement counts for an article."""
    client = get_client()
    (client.table("articles")
     .update({"response_count": response_count, "comment_count": comment_count})
     .eq("id", article_id)
     .execute())


def create_scrape_run(target_date: str) -> int:
    client = get_client()
    resp = (client.table("scrape_runs")
            .insert({"target_date": target_date})
            .execute())
    return resp.data[0]["id"]


def complete_scrape_run(run_id: int, *, press_count: int, article_count: int,
                        error_count: int, duration_sec: float, error_log: str = ""):
    status = "failed" if error_count > 0 and article_count == 0 else "completed"
    client = get_client()
    (client.table("scrape_runs")
     .update({
         "press_count": press_count,
         "article_count": article_count,
         "error_count": error_count,
         "duration_sec": duration_sec,
         "status": status,
         "error_log": error_log,
         "completed_at": "now()",
     })
     .eq("id", run_id)
     .execute())
