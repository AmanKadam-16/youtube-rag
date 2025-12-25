from openai import OpenAI
from src.yt_rag.core.config import settings


client = OpenAI(
    base_url=settings.GROQ_PROVIDER_BASE_URL, api_key=settings.GROQ_PROVIDER_API_KEY
)

embedding_client = OpenAI(
    base_url=settings.NVIDIA_PROVIDER_BASE_URL, api_key=settings.NVIDIA_PROVIDER_API_KEY
)


def call_llm(system_prompt: str, user_msg: str) -> str:
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def call_agentic_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL_NAME, messages=messages, temperature=0
    )
    return response.choices[0].message.content
