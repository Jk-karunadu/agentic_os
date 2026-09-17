from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AgentOS Local"
    database_path: str = "data/agentos.db"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    request_timeout_seconds: float = 3600.0
    max_evidence_items: int = 12
    max_output_tokens: int = 1500
    context_window: int = 8192
    soft_timeout_ratio: float = 0.80
    max_tool_generation_attempts: int = 2
    max_agent_iterations: int = 12
    chat_history_limit: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTOS_")


settings = Settings()
