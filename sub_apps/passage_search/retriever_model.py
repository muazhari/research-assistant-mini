from haystack.document_stores import BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, BaseRetriever, DensePassageRetriever

from models.passage_search_request import PassageSearchRequest


class RetrieverModel:
    def get_sentence_transformer_retriever(self, document_store: BaseDocumentStore,
                                           passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: EmbeddingRetriever = EmbeddingRetriever(
            document_store=document_store,
            model_format=passage_search_request.model_format,
            embedding_model=passage_search_request.embedding_model.query_embedding_model,
            use_gpu=True
        )
        return retriever

    def get_openai_retriever(self, document_store: BaseDocumentStore,
                             passage_search_request: PassageSearchRequest) -> BaseRetriever:
        retriever: EmbeddingRetriever = EmbeddingRetriever(
            document_store=document_store,
            model_format=passage_search_request.model_format,
            embedding_model=passage_search_request.embedding_model.query_embedding_model,
            api_key=passage_search_request.openai_api_key,
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
        if passage_search_request.model_format == "sentence_transformers":
            retriever = self.get_sentence_transformer_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        elif passage_search_request.model_format == "openai":
            retriever = self.get_openai_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        elif passage_search_request.model_format == "dense_passage":
            retriever = self.get_dense_passage_retriever(
                document_store=document_store,
                passage_search_request=passage_search_request
            )
        else:
            raise ValueError(f"Model format {passage_search_request.model_format} is not supported.")
        return retriever


retriever_model = RetrieverModel()
