from pydantic_settings import BaseSettings


class Setting(BaseSettings):
    DATABASE_URL: str
    GROQ_PROVIDER_API_KEY: str
    GROQ_PROVIDER_BASE_URL: str = "https://api.groq.com/openai/v1"
    CHAT_MODEL_NAME: str
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    NVIDIA_PROVIDER_API_KEY: str
    NVIDIA_PROVIDER_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    class Config:
        env_file_encoding = "utf-8"
        env_file = ".env"


settings = Setting()
