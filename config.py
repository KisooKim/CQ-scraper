from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = "cq-articles"
    r2_public_url: str = ""  # e.g., https://pub-xxx.r2.dev — set after enabling R2 public access
    scrape_delay_min: float = 1.0
    scrape_delay_max: float = 2.5
    request_timeout: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
