from supabase import create_client, Client

from config import settings

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def get_existing_urls(publish_date: str) -> set[str]:
    """Return set of article URLs already in DB for a given date."""
    client = get_client()
    resp = client.table("articles").select("url").eq("publish_date", publish_date).execute()
    return {row["url"] for row in resp.data}


def get_press_list() -> list[dict]:
    """Return all active press entries."""
    client = get_client()
    resp = (client.table("press")
            .select("id, code, name")
            .eq("is_active", True)
            .order("id")
            .execute())
    return resp.data


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
