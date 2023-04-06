from haystack.document_stores import BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, BaseRetriever, DensePassageRetriever, MultihopEmbeddingRetriever, \
    BM25Retriever, TfidfRetriever

from models.passage_search_request import PassageSearchRequest


class RetrieverModel:
    def get_multihop_retriever(self, document_store: BaseDocumentStore,
                               passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: MultihopEmbeddingRetriever = MultihopEmbeddingRetriever(
            document_store=document_store,
            embedding_model=passage_search_request.embedding_model.query_model,
            num_iterations=passage_search_request.num_iterations,
        )
        return retriever

    def get_basic_retriever(self, document_store: BaseDocumentStore,
                            passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: EmbeddingRetriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=passage_search_request.embedding_model.query_model,
            api_key=passage_search_request.api_key,
        )
        return retriever

    def get_dense_passage_retriever(self, document_store: BaseDocumentStore,
                                    passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: DensePassageRetriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=passage_search_request.embedding_model.query_model,
            passage_embedding_model=passage_search_request.embedding_model.passage_model,
        )
        return retriever

    def get_bm25_retriever(self, document_store: BaseDocumentStore) -> BaseRetriever:
        retriever: BM25Retriever = BM25Retriever(
            document_store=document_store
        )
        return retriever

    def get_tfidf_retriever(self, document_store: BaseDocumentStore) -> BaseRetriever:
        retriever: TfidfRetriever = TfidfRetriever(
            document_store=document_store
        )
        return retriever

    def get_dense_retriever(self, document_store: BaseDocumentStore,
                            passage_search_request: PassageSearchRequest) -> BaseRetriever:
        if passage_search_request.dense_retriever == "basic":
            retriever = self.get_basic_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        elif passage_search_request.dense_retriever == "multihop":
            retriever = self.get_multihop_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        elif passage_search_request.dense_retriever == "dense_passage":
            retriever = self.get_dense_passage_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        else:
            raise ValueError(f"Dense retriever {passage_search_request.dense_retriever} is not supported.")
        return retriever

    def get_sparse_retriever(self, document_store: BaseDocumentStore,
                             passage_search_request: PassageSearchRequest) -> BaseRetriever:
        if passage_search_request.sparse_retriever == "tfidf":
            retriever = self.get_tfidf_retriever(
                document_store=document_store,
            )
        elif passage_search_request.sparse_retriever == "bm25":
            retriever = self.get_bm25_retriever(
                document_store=document_store,
            )
        else:
            raise ValueError(f"Sparse retriever {passage_search_request.sparse_retriever} is not supported.")
        return retriever


retriever_model = RetrieverModel()
