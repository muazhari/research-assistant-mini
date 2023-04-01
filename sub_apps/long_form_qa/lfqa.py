from datetime import datetime, timedelta
from typing import List

from haystack.nodes import EmbeddingRetriever, BaseGenerator, Shaper
from haystack.pipelines import GenerativeQAPipeline
from haystack.schema import Document

from models.lfqa_response import LFQAResponse
from models.lfqa_search_request import LFQARequest
from models.passage_search_request import PassageSearchRequest
from sub_apps.long_form_qa.generator_model import generator_model
from sub_apps.passage_search.passage_search import passage_search


class LFQA:

    def qa(self, passage_search_request: PassageSearchRequest, lfqa_request: LFQARequest):
        time_start: datetime = datetime.now()

        window_sized_documents: List[Document] = passage_search.get_window_sized_documents(
            passage_search_request=passage_search_request
        )

        retriever: EmbeddingRetriever = passage_search.get_retriever(
            passage_search_request=passage_search_request,
            documents=window_sized_documents
        )

        shaper = Shaper(
            func="join_documents",
            inputs={"documents": "documents"},
            outputs=["documents"]
        )

        generator: BaseGenerator = generator_model.get_generator(
            lfqa_request=lfqa_request
        )

        pipeline = GenerativeQAPipeline(
            retriever=retriever,
            generator=generator,
        )

        generative_qa_result: dict = pipeline.run(
            query=passage_search_request.query,
            params={
                "Retriever": {"top_k": int(passage_search_request.percentage * len(window_sized_documents))},
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
