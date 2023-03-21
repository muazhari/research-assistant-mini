from haystack.document_stores import BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, BaseRetriever, DensePassageRetriever, MultihopEmbeddingRetriever

from models.passage_search_request import PassageSearchRequest


class RetrieverModel:
    def get_multihop_retriever(self, document_store: BaseDocumentStore,
                               passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: EmbeddingRetriever = MultihopEmbeddingRetriever(
            document_store=document_store,
            embedding_model=passage_search_request.embedding_model.query_embedding_model,
            num_iterations=passage_search_request.num_iterations,
            use_gpu=True
        )
        return retriever

    def get_basic_retriever(self, document_store: BaseDocumentStore,
                            passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: EmbeddingRetriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=passage_search_request.embedding_model.query_embedding_model,
            api_key=passage_search_request.api_key,
            use_gpu=True
        )
        return retriever

    def get_dense_passage_retriever(self, document_store: BaseDocumentStore,
                                    passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: DensePassageRetriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=passage_search_request.embedding_model.query_embedding_model,
            passage_embedding_model=passage_search_request.embedding_model.passage_embedding_model,
            use_gpu=True
        )
        return retriever

    def get_retriever(self, document_store: BaseDocumentStore,
                      passage_search_request: PassageSearchRequest) -> BaseRetriever:
        if passage_search_request.retriever == "basic":
            retriever = self.get_basic_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        elif passage_search_request.retriever == "multihop":
            retriever = self.get_multihop_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        elif passage_search_request.retriever == "dense_passage":
            retriever = self.get_dense_passage_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        else:
            raise ValueError(f"Retriever {passage_search_request.retriever} is not supported.")
        return retriever


retriever_model = RetrieverModel()
