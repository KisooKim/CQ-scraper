"""Name unpublished daily issues and generate briefings using Claude Code CLI.

Usage:
    python name_issues.py              # name all unpublished dates
    python name_issues.py --date 20260316  # name specific date

Designed to run on a schedule (Windows Task Scheduler) on the home computer.
Requires `claude` CLI to be installed and authenticated.
"""

import argparse
import json
import logging
import os
import subprocess
import sys

from db import get_client
from storage import download_article_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_unpublished_dates() -> list[str]:
    """Find dates that have unpublished issues."""
    client = get_client()
    resp = (client.table("daily_issues")
            .select("publish_date")
            .eq("is_published", False)
            .execute())

    dates = sorted(set(row["publish_date"] for row in resp.data))
    return dates


def get_issues_with_titles(publish_date: str) -> list[dict]:
    """Get unpublished issues with sample article titles for a date."""
    client = get_client()

    resp = (client.table("daily_issues")
            .select("id, name, article_count")
            .eq("publish_date", publish_date)
            .eq("is_published", False)
            .order("article_count", desc=True)
            .execute())

    issues = []
    for iss in resp.data:
        if iss["name"] == "기타":
            issues.append({**iss, "titles": []})
            continue

        links = (client.table("daily_issue_articles")
                 .select("article_id")
                 .eq("issue_id", iss["id"])
                 .limit(10)
                 .execute())
        aids = [r["article_id"] for r in links.data]

        titles = []
        if aids:
            arts = (client.table("articles")
                    .select("title")
                    .in_("id", aids)
                    .execute())
            titles = [a["title"] for a in arts.data]

        issues.append({**iss, "titles": titles})

    return issues


def build_prompt(publish_date: str, issues: list[dict]) -> str:
    """Build a prompt for Claude Code to name the issues."""
    lines = [
        f"{publish_date} 한국 뉴스 클러스터에 이슈 이름을 붙여주세요.",
        "",
        "규칙:",
        "- 5~15자 내외, 신문 1면 헤드라인 스타일",
        "- '인물+사건' 또는 '주제+핵심 키워드' 조합",
        "- 좋은 예: '트럼프 호르무즈 파병 요구', '서울 아파트 공급난', 'BTS 광화문 콘서트'",
        "- 나쁜 예: '트럼프 / 호르무즈 / 파병' (키워드 나열 금지)",
        "",
    ]

    non_gita = [iss for iss in issues if iss.get("name") != "기타"]
    for iss in non_gita:
        lines.append(f"[ID:{iss['id']}] ({iss['article_count']}건):")
        for t in iss["titles"]:
            lines.append(f"  - {t}")
        lines.append("")

    lines.append("JSON으로만 응답하세요. 다른 텍스트 없이:")
    lines.append('{"<issue_id>": "<이슈 이름>", ...}')

    return "\n".join(lines)


def call_claude(prompt: str) -> dict | None:
    """Call Claude Code CLI with a prompt and parse JSON response."""
    try:
        env = {**subprocess.os.environ}
        env.pop("CLAUDECODE", None)  # allow running inside Claude Code session

        claude_cmd = os.environ.get("CLAUDE_CMD", "claude")
        # On Windows, npm installs claude as .cmd — need shell=True
        use_shell = sys.platform == "win32"
        result = subprocess.run(
            [claude_cmd, "-p", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            encoding="utf-8",
            shell=use_shell,
        )

        if result.returncode != 0:
            log.error("Claude CLI failed (code %d): %s", result.returncode, result.stderr)
            return None

        text = result.stdout.strip()

        # Try to extract JSON from response
        # Handle ```json ... ``` wrapping
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)

        return json.loads(text)

    except subprocess.TimeoutExpired:
        log.error("Claude CLI timed out")
        return None
    except json.JSONDecodeError:
        # Fallback: extract summary/briefing via regex for single-issue responses
        import re
        m_key = re.search(r'"(\d+)"\s*:', text)
        m_sum = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        m_brief = re.search(r'"briefing"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if m_key and m_sum and m_brief:
            key = m_key.group(1)
            summary = m_sum.group(1).replace('\\"', '"').replace('\\n', '\n')
            briefing = m_brief.group(1).replace('\\"', '"').replace('\\n', '\n')
            log.warning("  JSON parse failed, recovered via regex for key %s", key)
            return {key: {"summary": summary, "briefing": briefing}}
        log.error("Failed to parse Claude response as JSON and regex fallback failed\nResponse: %s", text[:500])
        return None
    except FileNotFoundError:
        log.error("claude CLI not found. Install Claude Code: https://claude.ai/code")
        return None


def get_issue_articles_with_body(issue_id: int, limit: int = 5) -> list[dict]:
    """Get top articles for an issue with body text from R2."""
    client = get_client()

    # Get article IDs sorted by engagement
    links = (client.table("daily_issue_articles")
             .select("article:article_id(id, title, r2_key, response_count, comment_count)")
             .eq("issue_id", issue_id)
             .execute())

    articles = []
    for row in links.data:
        art = row["article"]
        articles.append(art)

    # Sort by engagement (responses + comments)
    articles.sort(key=lambda a: (a["response_count"] or 0) + (a["comment_count"] or 0), reverse=True)

    # Fetch body text from R2 for top articles
    result = []
    for art in articles[:limit]:
        body = ""
        if art.get("r2_key"):
            r2_data = download_article_text(art["r2_key"])
            if r2_data:
                body = r2_data.get("body_text", "")
        result.append({
            "title": art["title"],
            "body": body[:3000] if body else "",  # truncate long articles
        })

    return result


def build_briefing_prompt(publish_date: str, issues: list[dict]) -> str:
    """Build a prompt for generating summaries and briefings.

    Each issue in the list should have: id, name, articles (from get_issue_articles_with_body).
    """
    lines = [
        f"{publish_date} 한국 뉴스 이슈별 브리핑을 작성해주세요.",
        "",
        "## 작성 규칙",
        "",
        "각 이슈에 대해 summary와 briefing 두 가지를 작성합니다.",
        "",
        "### summary (짧은 요약)",
        "- 2~3문장으로 이 이슈의 핵심을 자연어로 설명",
        "- 무슨 일이 왜 일어났고, 왜 중요한지",
        "- 독자가 한눈에 상황을 파악할 수 있게",
        "- **순수 텍스트만 사용** — 마크다운 기호(**, ##, - 등) 절대 사용 금지",
        "",
        "### briefing (상세 브리핑)",
        "- 나무위키/위키백과 스타일로 A4 1장 분량 (800~1200자)",
        "- 마크다운 형식 (## 소제목, 볼드, 리스트 등 활용)",
        "- 볼드 안에 따옴표 금지: **\"텍스트\"** (X) → \"**텍스트**\" (O)",
        "- 구성: 사건 개요 → 배경/맥락 → 쟁점/논란 → 전망/의미",
        "- 어려운 용어는 쉽게 풀어서 설명",
        "- 중립적 톤, 팩트 중심",
        "",
        "## 이슈별 기사 내용",
        "",
    ]

    for iss in issues:
        lines.append(f"### [ID:{iss['id']}] {iss['name']}")
        for i, art in enumerate(iss["articles"], 1):
            lines.append(f"\n**기사 {i}: {art['title']}**")
            if art["body"]:
                lines.append(art["body"])
        lines.append("")

    lines.append("## 응답 형식")
    lines.append("JSON으로만 응답하세요. 다른 텍스트 없이:")
    lines.append("""{
  "<issue_id>": {
    "summary": "짧은 요약 2~3문장",
    "briefing": "마크다운 상세 브리핑"
  }
}""")

    return "\n".join(lines)


def _generate_batch_briefings(publish_date: str, batch: list[dict]) -> dict[int, dict]:
    """Generate briefings for a single batch of issues."""
    prompt = build_briefing_prompt(publish_date, batch)
    result = call_claude(prompt)
    if not result:
        return {}

    briefing_map = {}
    for key, val in result.items():
        try:
            issue_id = int(key)
            if isinstance(val, dict):
                briefing_map[issue_id] = {
                    "summary": val.get("summary", ""),
                    "briefing": val.get("briefing", ""),
                }
        except (ValueError, AttributeError):
            log.warning("  Skipping invalid briefing key: %s", key)
    return briefing_map


BRIEFING_BATCH_SIZE = 1
BRIEFING_MAX_WORKERS = 8


def generate_briefings(publish_date: str, name_map: dict[int, str],
                       issues: list[dict]) -> dict[int, dict] | None:
    """Generate summary + briefing for each named issue (batched + parallel)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    non_gita = [iss for iss in issues if iss["name"] != "기타"]
    if not non_gita:
        return {}

    log.info("  Fetching article bodies for briefing generation...")
    briefing_issues = []
    for iss in non_gita:
        issue_id = iss["id"]
        display_name = name_map.get(issue_id, iss["name"])
        articles = get_issue_articles_with_body(issue_id, limit=5)
        briefing_issues.append({
            "id": issue_id,
            "name": display_name,
            "articles": articles,
        })

    # Split into batches
    batches = [briefing_issues[i:i + BRIEFING_BATCH_SIZE]
               for i in range(0, len(briefing_issues), BRIEFING_BATCH_SIZE)]
    log.info("  Generating briefings: %d issues in %d batches (max %d parallel)...",
             len(briefing_issues), len(batches), BRIEFING_MAX_WORKERS)

    # Run batches in parallel
    briefing_map = {}
    with ThreadPoolExecutor(max_workers=BRIEFING_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_generate_batch_briefings, publish_date, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_result = future.result()
                briefing_map.update(batch_result)
                log.info("    Batch %d/%d done (%d briefings)",
                         batch_idx + 1, len(batches), len(batch_result))
            except Exception as e:
                log.error("    Batch %d/%d failed: %s", batch_idx + 1, len(batches), e)

    if not briefing_map:
        log.error("  All briefing batches failed")
        return None

    log.info("  Briefings complete: %d/%d issues", len(briefing_map), len(briefing_issues))
    return briefing_map


def _sanitize_briefing(text: str) -> str:
    """Post-process briefing text for clean rendering.

    - Escape lone ~ to prevent GFM strikethrough
    (Bold ** handling is done in the frontend via regex → <strong> + rehype-raw)
    """
    import re
    text = re.sub(r"(?<!\\)(?<!~)~(?!~)", r"\\~", text)
    return text


def _sanitize_summary(text: str) -> str:
    """Strip markdown formatting from summary (should be plain text)."""
    import re
    # Remove bold/italic markers: **, *, __, _
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bullet markers at start of lines
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def save_briefings(briefing_map: dict[int, dict]):
    """Save summaries and briefings to daily_issues."""
    client = get_client()
    for issue_id, data in briefing_map.items():
        summary = _sanitize_summary(data.get("summary", ""))
        briefing = _sanitize_briefing(data.get("briefing", ""))
        (client.table("daily_issues")
         .update({"summary": summary, "briefing": briefing})
         .eq("id", issue_id)
         .execute())
        log.info("  [%d] briefing saved (%d chars)", issue_id, len(briefing))


def publish_date_issues(publish_date: str, name_map: dict[int, str]):
    """Update issue names and set is_published = true."""
    client = get_client()

    for issue_id, name in name_map.items():
        (client.table("daily_issues")
         .update({"name": name, "is_published": True})
         .eq("id", issue_id)
         .execute())
        log.info("  [%d] → %s", issue_id, name)

    # Also publish 기타 issues for this date
    (client.table("daily_issues")
     .update({"is_published": True})
     .eq("publish_date", publish_date)
     .eq("name", "기타")
     .execute())


def process_date(publish_date: str) -> bool:
    """Name and publish issues for a single date. Returns True on success."""
    log.info("Processing %s", publish_date)

    issues = get_issues_with_titles(publish_date)
    if not issues:
        log.info("  No unpublished issues found")
        return True

    non_gita = [iss for iss in issues if iss["name"] != "기타"]
    if not non_gita:
        # Only 기타 — just publish it
        (get_client().table("daily_issues")
         .update({"is_published": True})
         .eq("publish_date", publish_date)
         .execute())
        log.info("  Only 기타 issues — published as-is")
        return True

    # Batch naming: max 5 issues per call to avoid long-prompt failures
    NAMING_BATCH = 5
    name_map = {}
    for i in range(0, len(non_gita), NAMING_BATCH):
        batch_issues = non_gita[i:i + NAMING_BATCH]
        # Build prompt with only this batch (pass as issues list including 기타 check)
        prompt = build_prompt(publish_date, batch_issues)
        result = call_claude(prompt)
        if not result:
            log.error("  Failed to get names from Claude (batch %d-%d)", i, i + len(batch_issues))
            continue
        for key, name in result.items():
            try:
                name_map[int(key)] = name
            except ValueError:
                log.warning("  Skipping invalid key: %s", key)

    if not name_map:
        log.error("  Failed to get names from Claude")
        return False

    # Verify all non-기타 issues got names
    expected_ids = {iss["id"] for iss in non_gita}
    named_ids = set(name_map.keys())
    missing = expected_ids - named_ids
    if missing:
        log.warning("  Missing names for issue IDs: %s", missing)

    # Generate briefings using article body text
    briefing_map = generate_briefings(publish_date, name_map, issues)
    if briefing_map:
        save_briefings(briefing_map)
    else:
        log.warning("  Briefing generation failed or empty — publishing without briefings")

    publish_date_issues(publish_date, name_map)
    log.info("  Published %d issues for %s", len(name_map) + 1, publish_date)

    # Backfill thumbnails for visible articles (non-critical)
    try:
        backfill_thumbnails(publish_date)
    except Exception as e:
        log.warning("  Thumbnail backfill failed (non-critical): %s", e)

    # Match daily issues to persistent issues (non-critical)
    try:
        from persistent_issues import match_daily_to_persistent
        match_daily_to_persistent(publish_date)
    except Exception as e:
        log.warning("  Persistent issue matching failed (non-critical): %s", e)

    # Update persistent issue content (non-critical)
    try:
        from persistent_content import update_todays_persistent_issues
        update_todays_persistent_issues(publish_date)
    except Exception as e:
        log.warning("  Persistent content update failed (non-critical): %s", e)

    return True


def backfill_thumbnails(publish_date: str):
    """Fetch thumbnails for published issue articles that are missing them.

    Only runs for articles in non-기타 published issues (i.e., visible on the site).
    Uses og:image meta tag from article detail page. Saves Naver CDN URL directly.
    TODO: When R2 public access is configured, upload images to R2 instead.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import httpx
    from bs4 import BeautifulSoup

    client_db = get_client()

    # Get all non-기타 published issue IDs
    issues = (client_db.table("daily_issues")
              .select("id")
              .eq("publish_date", publish_date)
              .eq("is_published", True)
              .neq("name", "기타")
              .execute())

    if not issues.data:
        return

    # Get articles without thumbnails (deduplicate across issues)
    seen_ids = set()
    articles_to_fix = []
    for iss in issues.data:
        arts = (client_db.table("daily_issue_articles")
                .select("article:article_id(id, url, thumbnail_url)")
                .eq("issue_id", iss["id"])
                .execute())
        for a in arts.data:
            art = a["article"]
            if art["id"] not in seen_ids and not art["thumbnail_url"] and art["url"]:
                seen_ids.add(art["id"])
                articles_to_fix.append(art)

    if not articles_to_fix:
        log.info("  Thumbnails: all articles already have thumbnails")
        return

    log.info("  Thumbnails: fetching for %d articles...", len(articles_to_fix))

    def _fetch_og_image(article: dict) -> tuple[int, str | None]:
        """Fetch og:image from article page."""
        try:
            http = httpx.Client(timeout=10, follow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp = http.get(article["url"])
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            og = soup.select_one('meta[property="og:image"]')
            if og and og.get("content"):
                return article["id"], og["content"]
        except Exception:
            pass
        return article["id"], None

    updated = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_og_image, art): art for art in articles_to_fix}
        for future in as_completed(futures):
            art_id, thumb_url = future.result()
            if thumb_url:
                (client_db.table("articles")
                 .update({"thumbnail_url": thumb_url})
                 .eq("id", art_id)
                 .execute())
                updated += 1

    log.info("  Thumbnails: updated %d/%d articles", updated, len(articles_to_fix))


def backfill_briefings(publish_date: str) -> bool:
    """Generate briefings for published issues that are missing them."""
    client = get_client()

    resp = (client.table("daily_issues")
            .select("id, name, article_count")
            .eq("publish_date", publish_date)
            .eq("is_published", True)
            .is_("summary", "null")
            .neq("name", "기타")
            .order("article_count", desc=True)
            .execute())

    if not resp.data:
        log.info("  No issues missing briefings for %s", publish_date)
        return True

    log.info("  %d issues missing briefings for %s", len(resp.data), publish_date)

    name_map = {iss["id"]: iss["name"] for iss in resp.data}
    briefing_map = generate_briefings(publish_date, name_map, resp.data)
    if briefing_map:
        save_briefings(briefing_map)
        log.info("  Saved %d briefings for %s", len(briefing_map), publish_date)
        return True

    log.error("  Briefing generation failed for %s", publish_date)
    return False


def get_published_dates() -> list[str]:
    """Get all dates that have published issues."""
    client = get_client()
    resp = (client.table("daily_issues")
            .select("publish_date")
            .eq("is_published", True)
            .execute())
    return sorted(set(row["publish_date"] for row in resp.data))


def main():
    parser = argparse.ArgumentParser(description="CQ Issue Namer (Claude Code)")
    parser.add_argument("--date", help="Target date YYYYMMDD (default: all unpublished)")
    parser.add_argument("--briefings-only", action="store_true",
                        help="Only generate briefings for published issues missing them")
    args = parser.parse_args()

    if args.briefings_only:
        if args.date:
            d = args.date
            publish_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            backfill_briefings(publish_date)
        else:
            dates = get_published_dates()
            log.info("Backfilling briefings for %d date(s): %s", len(dates), dates)
            for date in dates:
                backfill_briefings(date)
        return

    if args.date:
        d = args.date
        publish_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        process_date(publish_date)
    else:
        dates = get_unpublished_dates()
        if not dates:
            log.info("No unpublished dates found")
            return

        log.info("Found %d unpublished date(s): %s", len(dates), dates)
        for date in dates:
            success = process_date(date)
            if not success:
                log.error("Stopping due to failure on %s", date)
                sys.exit(1)


if __name__ == "__main__":
    main()
