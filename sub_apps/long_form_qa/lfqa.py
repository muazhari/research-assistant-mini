from datetime import datetime, timedelta
from typing import List

from haystack import Pipeline
from haystack.nodes import BaseGenerator, JoinDocuments, BaseRetriever
from haystack.schema import Document

from models.lfqa_response import LFQAResponse
from models.lfqa_search_request import LFQARequest
from models.passage_search_request import PassageSearchRequest
from sub_apps.long_form_qa.generator_model import generator_model
from sub_apps.passage_search.passage_search import passage_search


class LFQA:

    def get_pipeline(self, passage_search_request: PassageSearchRequest, lfqa_request: LFQARequest,
                     documents: List[Document]) -> Pipeline:
        dense_retriever: BaseRetriever = passage_search.get_dense_retriever(
            passage_search_request=passage_search_request,
            documents=documents
        )

        sparse_retriever: BaseRetriever = passage_search.get_sparse_retriever(
            passage_search_request=passage_search_request,
            documents=documents
        )

        document_joiner: JoinDocuments = JoinDocuments(
            join_mode="reciprocal_rank_fusion"
        )

        generator: BaseGenerator = generator_model.get_generator(
            lfqa_request=lfqa_request
        )

        pipeline = Pipeline()
        pipeline.add_node(
            component=dense_retriever,
            name="DenseRetriever",
            inputs=["Query"]
        )
        pipeline.add_node(
            component=sparse_retriever,
            name="SparseRetriever",
            inputs=["Query"]
        )
        pipeline.add_node(
            component=document_joiner,
            name="DocumentJoiner",
            inputs=["DenseRetriever", "SparseRetriever"]
        )
        pipeline.add_node(
            component=generator,
            name="Generator",
            inputs=["DocumentJoiner"]
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
