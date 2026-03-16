"""Naver News newspaper scraper.

Scrapes article lists from media.naver.com/press/{code}/newspaper
and article details from n.news.naver.com/article/newspaper/{code}/{id}.
"""

import json
import random
import re
import time

import httpx
from bs4 import BeautifulSoup

from config import settings

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

NEWSPAPER_URL = "https://media.naver.com/press/{code}/newspaper?date={date}"
COMMENT_API = "https://apis.naver.com/commentBox/cbox5/web_naver_list_jsonp.json"
REACTION_API = "https://news.like.naver.com/v1/search/contents"


def _get_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": random.choice(USER_AGENTS)},
        timeout=settings.request_timeout,
        follow_redirects=True,
    )


def _delay():
    time.sleep(random.uniform(settings.scrape_delay_min, settings.scrape_delay_max))


# ---------------------------------------------------------------------------
# 1. Newspaper page: list of articles per press per date
# ---------------------------------------------------------------------------

def scrape_newspaper_page(client: httpx.Client, press_code: str, date: str) -> list[dict]:
    """Scrape article list from a newspaper page.

    Returns list of dicts with keys:
        title, url, section, position, thumbnail_url
    """
    url = NEWSPAPER_URL.format(code=press_code, date=date)
    resp = client.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    articles = []
    global_position = 0

    for section_div in soup.select("div.newspaper_inner"):
        # Section label: e.g. "A1", "A2", "B1"
        section_em = section_div.select_one("h3 span.page_notation em")
        section = section_em.get_text(strip=True) if section_em else None

        for li in section_div.select("ul.newspaper_article_lst > li"):
            a_tag = li.select_one("a[href]")
            if not a_tag:
                continue

            title_tag = a_tag.select_one("div.newspaper_txt_box strong")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            if not title:
                continue

            href = a_tag.get("href", "")
            global_position += 1

            # Thumbnail
            img_tag = a_tag.select_one("div.newspaper_img_frame img")
            thumbnail = img_tag.get("src") if img_tag else None

            articles.append({
                "title": title,
                "url": href,
                "section": section,
                "position": global_position,
                "thumbnail_url": thumbnail,
            })

    return articles


# ---------------------------------------------------------------------------
# 2. Article detail page
# ---------------------------------------------------------------------------

def scrape_article_detail(client: httpx.Client, article_url: str) -> dict:
    """Scrape article detail page.

    Returns dict with keys: title, body_text, journalist_names, image_urls
    """
    resp = client.get(article_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Title
    title_el = soup.select_one("h2#title_area span")
    title = title_el.get_text(strip=True) if title_el else None

    # Body text
    body_el = soup.select_one("article#dic_area")
    body_text = None
    image_urls = []
    if body_el:
        # Extract images before stripping tags
        for img in body_el.select("img[src]"):
            src = img.get("src", "")
            if src and "pstatic.net" in src:
                image_urls.append(src)

        # Convert <br> to newlines, strip tags
        body_html = str(body_el)
        body_text = re.sub(r"(?i)<br\s*/?>", "\n", body_html)
        body_text = re.sub(r"<[^>]+>", " ", body_text)
        body_text = re.sub(r"\s+", " ", body_text).strip()

    # Journalist names (individual names from the layer)
    journalist_names = []
    for em in soup.select("em.media_end_head_journalist_layer_name"):
        name = em.get_text(strip=True)
        # Strip common suffixes
        name = re.sub(r"\s*(기자|특파원|논설위원|칼럼니스트|객원기자|편집장)\s*$", "", name).strip()
        if name and name not in journalist_names:
            journalist_names.append(name)

    # Original article URL (link to the newspaper's own website)
    origin_link = soup.select_one("a.media_end_head_origin_link")
    original_url = origin_link.get("href") if origin_link else None

    return {
        "title": title,
        "body_text": body_text,
        "journalist_names": journalist_names,
        "image_urls": image_urls,
        "original_url": original_url,
    }


# ---------------------------------------------------------------------------
# 3. Engagement: comment count + reaction count
# ---------------------------------------------------------------------------

def _extract_ids_from_url(article_url: str) -> tuple[str, str] | None:
    """Extract (press_code, article_id) from a Naver News article URL.

    URL patterns:
        https://n.news.naver.com/article/newspaper/025/0003508802?date=...
        https://n.news.naver.com/mnews/article/025/0003508802
        https://news.naver.com/article/025/0003508802
    """
    m = re.search(r"/(?:article|mnews/article|article/newspaper)/(\d+)/(\d+)", article_url)
    if m:
        return m.group(1), m.group(2)
    return None


def fetch_comment_count(client: httpx.Client, press_code: str, article_id: str) -> int:
    """Fetch comment count from Naver comment API."""
    try:
        params = {
            "ticket": "news",
            "templateId": "default_society",
            "pool": "cbox5",
            "lang": "ko",
            "country": "",
            "objectId": f"news{press_code},{article_id}",
            "pageSize": "1",
            "page": "1",
            "sort": "FAVORITE",
            "_callback": "cb",
        }
        resp = client.get(COMMENT_API, params=params)
        # Response is JSONP: cb({...})
        text = resp.text.strip()
        json_str = re.sub(r"^cb\(", "", text).rstrip(");")
        data = json.loads(json_str)
        return data.get("result", {}).get("count", {}).get("comment", 0)
    except Exception:
        return 0


def fetch_reaction_count(client: httpx.Client, press_code: str, article_id: str) -> int:
    """Fetch total reaction count (like, warm, sad, angry, want) from Naver."""
    try:
        params = {
            "q": f"NEWS[ne_{press_code}_{article_id}]",
        }
        resp = client.get(REACTION_API, params=params)
        data = resp.json()
        contents = data.get("contents", [])
        if not contents:
            return 0
        reactions = contents[0].get("reactions", [])
        return sum(r.get("count", 0) for r in reactions)
    except Exception:
        return 0


def fetch_engagement(client: httpx.Client, article_url: str) -> tuple[int, int]:
    """Return (response_count, comment_count) for an article URL."""
    ids = _extract_ids_from_url(article_url)
    if not ids:
        return 0, 0
    press_code, article_id = ids
    comment_count = fetch_comment_count(client, press_code, article_id)
    reaction_count = fetch_reaction_count(client, press_code, article_id)
    return reaction_count, comment_count
