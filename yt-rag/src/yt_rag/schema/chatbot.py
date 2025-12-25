from pydantic import BaseModel, Field


class ChatInput(BaseModel):
    user_prompt: str = Field(..., min_length=1)
