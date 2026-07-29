from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AgentOS Local"
    database_path: str = "data/agentos.db"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    request_timeout_seconds: float = 3600.0
    max_evidence_items: int = 12
    max_output_tokens: int = 350
    context_window: int = 4096

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTOS_")


settings = Settings()
