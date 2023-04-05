from typing import Optional

from pydantic import BaseModel

from models.embedding_model import EmbeddingModel


class PassageSearchRequest(BaseModel):
    corpus_source_type: Optional[str]
    corpus: Optional[str]
    query: Optional[str]
    granularity: Optional[str]
    window_sizes: Optional[str]
    retriever_top_k: Optional[float]
    retriever_source_type: Optional[str]
    dense_retriever: Optional[str]
    sparse_retriever: Optional[str]
    embedding_model: Optional[EmbeddingModel]
    embedding_dimension: Optional[int]
    num_iterations: Optional[int]
    similarity_function: Optional[str]
    api_key: Optional[str]
