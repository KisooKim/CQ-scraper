"""Cloudflare R2 storage for article body text."""

import json

import boto3

from config import settings

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
        )
    return _s3_client


def make_r2_key(press_code: str, date: str, article_id: str) -> str:
    """Generate R2 object key: {press_code}/{date}/{article_id}.json"""
    return f"{press_code}/{date}/{article_id}.json"


def upload_article_text(r2_key: str, body_text: str, title: str | None = None,
                        journalist_names: list[str] | None = None) -> None:
    """Upload article body text to R2 as JSON."""
    s3 = get_s3_client()
    payload = {
        "title": title,
        "body_text": body_text,
        "journalist_names": journalist_names or [],
    }
    s3.put_object(
        Bucket=settings.r2_bucket,
        Key=r2_key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def download_article_text(r2_key: str) -> dict | None:
    """Download article body text from R2. Returns dict or None."""
    s3 = get_s3_client()
    try:
        resp = s3.get_object(Bucket=settings.r2_bucket, Key=r2_key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return None


def upload_thumbnail(article_id: int, image_bytes: bytes,
                     content_type: str = "image/jpeg") -> str:
    """Upload thumbnail image to R2. Returns R2 key."""
    s3 = get_s3_client()
    r2_key = f"thumbs/{article_id}.jpg"
    s3.put_object(
        Bucket=settings.r2_bucket,
        Key=r2_key,
        Body=image_bytes,
        ContentType=content_type,
    )
    return r2_key


def delete_thumbnail(article_id: int) -> None:
    """Delete thumbnail from R2."""
    s3 = get_s3_client()
    s3.delete_object(Bucket=settings.r2_bucket, Key=f"thumbs/{article_id}.jpg")


def get_r2_public_url(r2_key: str) -> str:
    """Get public URL for an R2 object.

    Requires R2 public access (r2.dev or custom domain) to be enabled.
    Configure R2_PUBLIC_URL in .env (e.g., https://pub-xxx.r2.dev).
    """
    base = settings.r2_public_url.rstrip("/") if hasattr(settings, "r2_public_url") and settings.r2_public_url else ""
    if not base:
        # Fallback: construct from account ID (won't work without public access)
        base = f"https://{settings.r2_account_id}.r2.dev"
    return f"{base}/{r2_key}"
