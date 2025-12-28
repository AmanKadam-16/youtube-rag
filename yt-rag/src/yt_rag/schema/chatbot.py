from pydantic import BaseModel, Field


class ChatInput(BaseModel):
    user_prompt: str = Field(..., min_length=1)
    collection_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
