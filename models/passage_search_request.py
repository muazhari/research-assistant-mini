from typing import Optional

from pydantic import BaseModel

from models.embedding_model import EmbeddingModel


class PassageSearchRequest(BaseModel):
    corpus_source_type: str
    corpus: str
    query: str
    granularity: str
    window_sizes: str
    percentage: float
    retriever_source_type: str
    retriever: str
    embedding_model: EmbeddingModel
    embedding_dimension: int
    num_iterations: int
    similarity_function: str
    api_key: Optional[str]
