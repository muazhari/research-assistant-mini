from typing import Optional

from pydantic import BaseModel

from models.embedding_model import EmbeddingModel


class PassageSearchRequest(BaseModel):
    source_type: str
    corpus: str
    query: str
    granularity: str
    window_sizes: str
    percentage: float
    model_format: str
    embedding_model: EmbeddingModel
    embedding_dimension: int
    similarity_function: str
    openai_api_key: Optional[str]
