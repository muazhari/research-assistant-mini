from typing import Optional

from pydantic import BaseModel


class EmbeddingModel(BaseModel):
    query_embedding_model: Optional[str]
    passage_embedding_model: Optional[str]
