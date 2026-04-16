"""Naver News newspaper scraper.

Scrapes article lists from media.naver.com/press/{code}/newspaper
and article details from n.news.naver.com/article/newspaper/{code}/{id}.
"""

import io
import json
import logging
import random
import re
import time

import httpx
from bs4 import BeautifulSoup
from PIL import Image

from config import settings

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
]

log = logging.getLogger(__name__)

NEWSPAPER_URL = "https://media.naver.com/press/{code}/newspaper?date={date}"
COMMENT_API = "https://apis.naver.com/commentBox/cbox5/web_naver_list_jsonp.json"
REACTION_API = "https://news.like.naver.com/v1/search/contents"


def _get_client() -> httpx.Client:
    ua = random.choice(USER_AGENTS)
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    return httpx.Client(
        headers=headers,
        timeout=settings.request_timeout,
        follow_redirects=True,
        transport=httpx.HTTPTransport(retries=3),
    )


def _delay():
    time.sleep(random.uniform(settings.scrape_delay_min, settings.scrape_delay_max))


def _url_base(url: str) -> str:
    """Strip query string from URL for comparison."""
    return url.split("?")[0]


def is_portrait_thumbnail(
    client: httpx.Client,
    thumbnail_url: str,
    article_image_urls: list[str] | None = None,
) -> bool:
    """Check if a thumbnail is a portrait/headshot.

    Uses two signals:
    1. Whether the thumbnail appears among the article body images.
       If NOT (and the article has body images), it's almost certainly a
       standalone journalist headshot or editorial graphic.
    2. Original image dimensions — portrait orientation or very small.
    """
    # Signal 1: thumbnail not in article body images → journalist headshot
    if article_image_urls is not None and len(article_image_urls) > 0:
        thumb_base = _url_base(thumbnail_url)
        body_bases = {_url_base(u) for u in article_image_urls}
        if thumb_base not in body_bases:
            log.debug("Thumbnail not in body images — flagged: %s", thumbnail_url)
            return True

    # Signal 2: geometric check on original image
    original_url = re.sub(r"\?type=nf\d+_\d+", "", thumbnail_url)
    try:
        resp = client.get(original_url, timeout=5)
        if resp.status_code != 200:
            return False
        img = Image.open(io.BytesIO(resp.content))
        w, h = img.size
        # Portrait orientation, tiny image, or small square (likely headshot)
        if h > w * 1.2 or max(w, h) < 200:
            log.debug("Portrait thumbnail detected: %dx%d — %s", w, h, thumbnail_url)
            return True
        # Small square-ish images not in body → likely headshot (for articles with no body images)
        if article_image_urls is not None and len(article_image_urls) == 0:
            if max(w, h) < 500 and 0.7 < h / w < 1.4:
                log.debug("Small standalone thumbnail: %dx%d — %s", w, h, thumbnail_url)
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# 1. Newspaper page: list of articles per press per date
# ---------------------------------------------------------------------------

def scrape_newspaper_page(client: httpx.Client, press_code: str, date: str,
                          soup: BeautifulSoup | None = None) -> list[dict]:
    """Scrape article list from a newspaper page.

    Returns list of dicts with keys:
        title, url, section, position, thumbnail_url
    """
    if soup is None:
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
            # Strip Naver tracking params
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            _p = urlparse(href)
            _q = {k: v for k, v in parse_qs(_p.query).items() if k != "ref"}
            href = urlunparse(_p._replace(query=urlencode(_q, doseq=True)))
            global_position += 1

            # Thumbnail — request higher-res version
            img_tag = a_tag.select_one("div.newspaper_img_frame img")
            thumbnail = img_tag.get("src") if img_tag else None
            if thumbnail:
                import re as _re
                thumbnail = _re.sub(r"\?type=nf\d+_\d+", "?type=nf528_352", thumbnail)

            articles.append({
                "title": title,
                "url": href,
                "section": section,
                "position": global_position,
                "thumbnail_url": thumbnail,
            })

    return articles


def _scrape_press_home(client: httpx.Client, press_code: str, date: str,
                       soup: BeautifulSoup | None = None) -> list[dict]:
    """Scrape article list from press home page (non-newspaper outlets).

    Used for broadcast/online outlets that don't have a newspaper-format page.
    Returns list of dicts with keys:
        title, url, section, position, thumbnail_url
    """
    if soup is None:
        url = NEWSPAPER_URL.format(code=press_code, date=date)
        resp = client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    seen_urls = set()
    articles = []
    position = 0

    # 1. Main curated articles (press_main_news)
    for li in soup.select("div.press_main_news li.press_news_item"):
        a_tag = li.select_one("a[href]")
        if not a_tag:
            continue
        title_tag = a_tag.select_one("span.press_news_text > strong")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        if not title:
            continue

        href = a_tag.get("href", "")
        _p = urlparse(href)
        _q = {k: v for k, v in parse_qs(_p.query).items() if k not in ("ref", "type")}
        href = urlunparse(_p._replace(query=urlencode(_q, doseq=True)))

        base = href.split("?")[0]
        if base in seen_urls:
            continue
        seen_urls.add(base)
        position += 1

        # Thumbnail from main curated articles
        img_tag = li.select_one("img")
        thumbnail = None
        if img_tag:
            thumbnail = img_tag.get("data-src") or img_tag.get("src")
            if thumbnail and "blank.gif" in thumbnail:
                thumbnail = None
            if thumbnail:
                thumbnail = re.sub(r"\?type=nf\d+_\d+", "?type=nf528_352", thumbnail)

        articles.append({
            "title": title,
            "url": href,
            "section": None,
            "position": position,
            "thumbnail_url": thumbnail,
        })

    # 2. Editorial news list (press_edit_news — bulk of articles)
    for li in soup.select("li.press_edit_news_item"):
        a_tag = li.select_one("a.press_edit_news_link")
        if not a_tag:
            continue
        title_tag = a_tag.select_one("span.press_edit_news_title")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        if not title:
            continue

        href = a_tag.get("href", "")
        _p = urlparse(href)
        _q = {k: v for k, v in parse_qs(_p.query).items() if k not in ("ref", "type")}
        href = urlunparse(_p._replace(query=urlencode(_q, doseq=True)))

        base = href.split("?")[0]
        if base in seen_urls:
            continue
        seen_urls.add(base)
        position += 1

        # Thumbnail — use data-src (src is placeholder gif)
        img_tag = a_tag.select_one("span.press_edit_news_thumb img")
        thumbnail = None
        if img_tag:
            thumbnail = img_tag.get("data-src") or img_tag.get("src")
            if thumbnail and "blank.gif" in thumbnail:
                thumbnail = None
            if thumbnail:
                thumbnail = re.sub(r"\?type=nf\d+_\d+", "?type=nf528_352", thumbnail)

        articles.append({
            "title": title,
            "url": href,
            "section": None,
            "position": position,
            "thumbnail_url": thumbnail,
        })

    return articles


def scrape_press_page(client: httpx.Client, press_code: str, date: str) -> list[dict]:
    """Scrape articles from a press page, auto-detecting format.

    Tries newspaper format first; falls back to press home format.
    Returns list of dicts with keys:
        title, url, section, position, thumbnail_url
    """
    url = NEWSPAPER_URL.format(code=press_code, date=date)
    resp = client.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    if soup.select("div.newspaper_inner"):
        return scrape_newspaper_page(client, press_code, date, soup=soup)
    else:
        return _scrape_press_home(client, press_code, date, soup=soup)


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

    # Journalist names
    journalist_names = []
    for em in soup.select("em.media_end_head_journalist_name"):
        name = em.get_text(strip=True)
        # Strip common suffixes
        name = re.sub(r"\s*(기자|특파원|논설위원|칼럼니스트|객원기자|편집장)\s*$", "", name).strip()
        if name and name not in journalist_names:
            journalist_names.append(name)

    # Original article URL (link to the newspaper's own website)
    origin_link = soup.select_one("a.media_end_head_origin_link")
    original_url = origin_link.get("href") if origin_link else None

    # Publish datetime (KST) — parse from data-date-time attr
    # Format: "YYYY-MM-DD HH:MM:SS". Fall back to modify time if missing.
    publish_datetime = None
    ts_el = soup.select_one(".media_end_head_info_datestamp_time._ARTICLE_DATE_TIME")
    if ts_el and ts_el.get("data-date-time"):
        publish_datetime = ts_el["data-date-time"].strip()
    else:
        mod_el = soup.select_one(".media_end_head_info_datestamp_time._ARTICLE_MODIFY_DATE_TIME")
        if mod_el and mod_el.get("data-modify-date-time"):
            publish_datetime = mod_el["data-modify-date-time"].strip()

    return {
        "title": title,
        "body_text": body_text,
        "journalist_names": journalist_names,
        "image_urls": image_urls,
        "original_url": original_url,
        "publish_datetime": publish_datetime,
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
        headers = {"Referer": f"https://n.news.naver.com/article/{press_code}/{article_id}"}
        resp = client.get(COMMENT_API, params=params, headers=headers)
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
