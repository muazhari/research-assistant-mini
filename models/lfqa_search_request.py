from typing import Optional

from pydantic import BaseModel


class LFQARequest(BaseModel):
    generator_model_format: Optional[str]
    generator_model: Optional[str]
    generator_model_source_type: Optional[str]
    prompt: Optional[str]
    answer_min_length: Optional[int]
    answer_max_length: Optional[int]
    answer_max_tokens: Optional[int]
    api_key: Optional[str]
