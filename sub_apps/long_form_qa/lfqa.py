from datetime import datetime, timedelta
from typing import List

from haystack import Pipeline
from haystack.nodes import BaseGenerator
from haystack.schema import Document

from models.lfqa_response import LFQAResponse
from models.lfqa_search_request import LFQARequest
from models.passage_search_request import PassageSearchRequest
from sub_apps.long_form_qa.generator_model import generator_model
from sub_apps.passage_search.passage_search import passage_search


class LFQA:

    def get_pipeline(self, passage_search_request: PassageSearchRequest, lfqa_request: LFQARequest,
                     documents: List[Document]) -> Pipeline:
        generator: BaseGenerator = generator_model.get_generator(
            lfqa_request=lfqa_request
        )

        pipeline: Pipeline = passage_search.get_pipeline(
            passage_search_request=passage_search_request,
            documents=documents
        )

        pipeline.add_node(
            component=generator,
            name="Generator",
            inputs=["Ranker"]
        )

        return pipeline

    def qa(self, passage_search_request: PassageSearchRequest, lfqa_request: LFQARequest):
        time_start: datetime = datetime.now()

        window_sized_documents: List[Document] = passage_search.get_window_sized_documents(
            passage_search_request=passage_search_request
        )

        pipeline: Pipeline = self.get_pipeline(
            passage_search_request=passage_search_request,
            lfqa_request=lfqa_request,
            documents=window_sized_documents
        )

        generative_qa_result: dict = pipeline.run(
            query=passage_search_request.query,
            params={
                "DenseRetriever": {"top_k": passage_search_request.retriever_top_k},
                "SparseRetriever": {"top_k": passage_search_request.retriever_top_k},
                "Ranker": {"top_k": passage_search_request.ranker_top_k},
            },
            debug=True
        )

        time_finish: datetime = datetime.now()
        time_delta: timedelta = time_finish - time_start

        response: LFQAResponse = LFQAResponse(
            generative_qa_result=generative_qa_result,
            process_duration=time_delta.total_seconds()
        )

        return response


long_form_qa = LFQA()
