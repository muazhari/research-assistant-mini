from typing import Optional

from pydantic import BaseModel


class EmbeddingModel(BaseModel):
    query_model: Optional[str]
    passage_model: Optional[str]
    ranker_model: Optional[str]
