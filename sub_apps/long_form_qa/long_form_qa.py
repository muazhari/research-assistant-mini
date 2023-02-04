from datetime import datetime, timedelta
import os

from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline, GenerativeQAPipeline
from haystack.nodes import EmbeddingRetriever, Seq2SeqGenerator, BaseGenerator
from haystack.document_stores import FAISSDocumentStore, PineconeDocumentStore, BaseDocumentStore
from haystack.schema import Document
from utilities.pre_processor import pre_processor
from sub_apps.passage_search.retriever_model import retriever_model
import hashlib
from typing import List, Tuple, Optional, Any
from sub_apps.passage_search.passage_search import passage_search
from sub_apps.long_form_qa.generator_model import generator_model


class LongFormQA:

    def qa(self, passage_search_request, lfqa_request: dict):
        time_start: datetime = datetime.now()

        window_sized_documents: List[Document] = passage_search.get_window_sized_documents(
            passage_search_request=passage_search_request
        )

        retriever: EmbeddingRetriever = passage_search.get_retriever(
            passage_search_request=passage_search_request,
            documents=window_sized_documents
        )

        generator: BaseGenerator = generator_model.get_generator(
            lfqa_request=lfqa_request
        )

        generative_qa_pipeline: GenerativeQAPipeline = GenerativeQAPipeline(
            retriever=retriever,
            generator=generator,
        )

        generative_qa_result: dict = generative_qa_pipeline.run(
            query=passage_search_request["query"],
            params={"Retriever": {"top_k": int(passage_search_request["percentage"] * len(window_sized_documents))}},
            debug=True
        )

        time_finish: datetime = datetime.now()
        time_delta: timedelta = time_finish - time_start

        response: dict = {
            "generative_qa_result": generative_qa_result,
            "process_duration": time_delta.total_seconds()
        }

        return response


long_form_qa = LongFormQA()
