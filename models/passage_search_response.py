from pydantic import BaseModel


class PassageSearchResponse(BaseModel):
    retrieval_result: dict
    process_duration: float
