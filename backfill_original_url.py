"""Backfill original_url for existing articles by visiting each article page."""
import time
import random
from crawlers.naver import _get_client
from db import get_client
from bs4 import BeautifulSoup

def backfill():
    db = get_client()
    client = _get_client()

    # Get articles missing original_url
    resp = db.table("articles").select("id, url").is_("original_url", "null").execute()
    articles = resp.data
    print(f"Articles missing original_url: {len(articles)}")

    updated = 0
    errors = 0
    for i, art in enumerate(articles):
        try:
            time.sleep(random.uniform(0.3, 0.8))
            r = client.get(art["url"])
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            link = soup.select_one("a.media_end_head_origin_link")
            if link and link.get("href"):
                db.table("articles").update({"original_url": link["href"]}).eq("id", art["id"]).execute()
                updated += 1
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(articles)} (updated: {updated})")
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error article {art['id']}: {e}")

    print(f"Done. Updated: {updated}, Errors: {errors}")

if __name__ == "__main__":
    backfill()
