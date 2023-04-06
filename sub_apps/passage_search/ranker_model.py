from haystack.nodes import SentenceTransformersRanker, BaseRanker

from models.passage_search_request import PassageSearchRequest


class RankerModel:
    def get_sentence_transformers_ranker(self, passage_search_request: PassageSearchRequest) -> BaseRanker:
        ranker: SentenceTransformersRanker = SentenceTransformersRanker(
            model_name_or_path=passage_search_request.embedding_model.ranker_model,
        )
        return ranker

    def get_ranker(self, passage_search_request: PassageSearchRequest) -> BaseRanker:
        if passage_search_request.ranker == "sentence_transformers":
            ranker = self.get_sentence_transformers_ranker(
                passage_search_request=passage_search_request
            )
        else:
            raise ValueError(f"Ranker {passage_search_request.ranker} is not supported.")
        return ranker


ranker_model = RankerModel()
