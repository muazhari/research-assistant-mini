from pydantic import BaseModel


class LFQAResponse(BaseModel):
    generative_qa_result: dict
    process_duration: float
