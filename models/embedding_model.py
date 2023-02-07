from pydantic import BaseModel


class EmbeddingModel(BaseModel):
    query_embedding_model: str
    passage_embedding_model: str
