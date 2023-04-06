import hashlib
import os
from datetime import datetime, timedelta
from typing import List

from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.nodes import BaseRetriever, JoinDocuments, BaseRanker
from haystack.schema import Document

from models.passage_search_request import PassageSearchRequest
from models.passage_search_response import PassageSearchResponse
from sub_apps.passage_search.ranker_model import ranker_model
from sub_apps.passage_search.retriever_model import retriever_model
from utilities.pre_processor import pre_processor


class PassageSearch:

    def get_window_sized_documents(self, passage_search_request: PassageSearchRequest) -> List[Document]:
        window_sized_processed_corpuses: List[dict] = pre_processor.get_window_sized_processed_corpuses(
            corpus=passage_search_request.corpus,
            corpus_source_type=passage_search_request.corpus_source_type,
            granularity=passage_search_request.granularity,
            window_sizes=list(map(int, passage_search_request.window_sizes.split(" ")))
        )

        window_sized_documents: List[Document] = []

        for window_sized_processed_corpus in window_sized_processed_corpuses:
            for index_window, window in enumerate(window_sized_processed_corpus["processed_corpus"]):
                window_sized_document = Document(
                    content=pre_processor.deprocess_windowed_corpus(
                        windowed_corpus=window,
                        granularity_source=passage_search_request.granularity
                    ),
                    meta={"index_window": index_window,
                          "window_size": window_sized_processed_corpus["window_size"]}
                )

                window_sized_documents.append(window_sized_document)

        return window_sized_documents

    def get_document_store_index_hash(self, passage_search_request: PassageSearchRequest) -> str:
        corpus_hash: str = hashlib.md5(passage_search_request.corpus.encode("utf-8")).hexdigest()
        window_sizes_hash: str = hashlib.md5(passage_search_request.window_sizes.encode("utf-8")).hexdigest()
        embedding_model_hash = hashlib.md5(str(passage_search_request.embedding_model).encode("utf-8")).hexdigest()

        document_store_index_hash: str = f"{embedding_model_hash}_{corpus_hash}_{window_sizes_hash}"

        return document_store_index_hash

    def get_dense_retriever(self, passage_search_request: PassageSearchRequest,
                            documents: List[Document]) -> BaseRetriever:
        document_store_index_hash: str = self.get_document_store_index_hash(
            passage_search_request=passage_search_request
        )
        faiss_index_path: str = f"document_store/faiss_index_{document_store_index_hash}"
        faiss_config_path: str = f"document_store/faiss_config_{document_store_index_hash}"

        if all(map(os.path.exists, [faiss_index_path, faiss_config_path])):
            document_store: FAISSDocumentStore = FAISSDocumentStore.load(
                index_path=faiss_index_path,
                config_path=faiss_config_path,
            )
            retriever: BaseRetriever = retriever_model.get_dense_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request,
            )
        else:
            document_store: FAISSDocumentStore = FAISSDocumentStore(
                sql_url="sqlite:///document_store/document_store.db",
                index=document_store_index_hash,
                embedding_dim=passage_search_request.embedding_dimension,
                return_embedding=True,
                similarity=passage_search_request.similarity_function,
                duplicate_documents="skip",
            )
            retriever: BaseRetriever = retriever_model.get_dense_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request,
            )
            document_store.write_documents(documents)
            document_store.update_embeddings(retriever)
            document_store.save(faiss_index_path, faiss_config_path)

        return retriever

    def get_sparse_retriever(self, passage_search_request: PassageSearchRequest,
                             documents: List[Document]) -> BaseRetriever:
        document_store_index_hash: str = self.get_document_store_index_hash(
            passage_search_request=passage_search_request
        )

        document_store: InMemoryDocumentStore = InMemoryDocumentStore(
            index=document_store_index_hash,
            embedding_dim=passage_search_request.embedding_dimension,
            return_embedding=True,
            similarity=passage_search_request.similarity_function,
            duplicate_documents="skip",
            use_bm25=True
        )
        retriever: BaseRetriever = retriever_model.get_sparse_retriever(
            document_store=document_store,
            passage_search_request=passage_search_request,
        )
        document_store.write_documents(documents)

        return retriever

    def get_ranker(self, passage_search_request: PassageSearchRequest) -> BaseRanker:
        return ranker_model.get_ranker(
            passage_search_request=passage_search_request
        )

    def get_pipeline(self, passage_search_request: PassageSearchRequest, documents: List[Document]) -> Pipeline:
        dense_retriever: BaseRetriever = self.get_dense_retriever(
            passage_search_request=passage_search_request,
            documents=documents
        )
        sparse_retriever: BaseRetriever = self.get_sparse_retriever(
            passage_search_request=passage_search_request,
            documents=documents
        )
        document_joiner: JoinDocuments = JoinDocuments(
            join_mode="reciprocal_rank_fusion"
        )
        ranker: BaseRanker = self.get_ranker(
            passage_search_request=passage_search_request
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
            component=ranker,
            name="Ranker",
            inputs=["DocumentJoiner"]
        )

        return pipeline

    def search(self, passage_search_request: PassageSearchRequest) -> PassageSearchResponse:
        time_start: datetime = datetime.now()

        window_sized_documents: List[Document] = self.get_window_sized_documents(
            passage_search_request=passage_search_request
        )

        pipeline: Pipeline = self.get_pipeline(
            passage_search_request=passage_search_request,
            documents=window_sized_documents
        )

        retrieval_result: dict = pipeline.run(
            query=passage_search_request.query,
            params={
                "DenseRetriever": {"top_k": passage_search_request.retriever_top_k},
                "SparseRetriever": {"top_k": passage_search_request.retriever_top_k},
                "Ranker": {"top_k": passage_search_request.ranker_top_k}
            },
            debug=True
        )

        time_finish: datetime = datetime.now()
        time_delta: timedelta = time_finish - time_start

        response: PassageSearchResponse = PassageSearchResponse(
            retrieval_result=retrieval_result,
            process_duration=time_delta.total_seconds()
        )

        return response


passage_search = PassageSearch()
