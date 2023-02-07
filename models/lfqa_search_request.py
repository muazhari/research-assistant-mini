from typing import Optional

from pydantic import BaseModel


class LFQARequest(BaseModel):
    model_format: str
    generator_model: str
    answer_min_length: Optional[int]
    answer_max_length: Optional[int]
    answer_max_tokens: Optional[int]
    openai_api_key: Optional[str]
