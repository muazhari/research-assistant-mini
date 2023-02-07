from typing import List, Tuple

import more_itertools
from txtai.pipeline import Textractor, Segmentation


class PreProcessor:

    def segment(self, corpus: str, granularity: str) -> List[str]:
        granularized_corpus = None
        if (granularity == "sentence"):
            segmentator = Segmentation(sentences=True)
            granularized_corpus = segmentator(text=corpus)
        elif granularity == "paragraph":
            segmentator = Segmentation(paragraphs=True)
            granularized_corpus = segmentator(text=corpus)
        elif granularity == "word":
            granularized_corpus = corpus.split(" ")
        else:
            ValueError(f"Granularity {granularity} is not supported.")
        return granularized_corpus

    def textract(self, corpus: str, granularity: str) -> List[str]:
        granularized_corpus = None
        if granularity == "word":
            granularized_corpus = corpus.split(" ")
        elif granularity == "sentence":
            textractor = Textractor(sentences=True)
            granularized_corpus = textractor(text=corpus)
        elif granularity == "paragraph":
            textractor = Textractor(paragraphs=True)
            granularized_corpus = textractor(text=corpus)
        else:
            ValueError(f"Granularity {granularity} is not supported.")
        return granularized_corpus

    def granularize(self, corpus: str, source_type: str, granularity: str) -> List[str]:
        granularized_corpus = None
        if source_type in ["text"]:
            granularized_corpus = self.segment(corpus, granularity)
        elif source_type in ["file", "web"]:
            granularized_corpus = self.textract(corpus, granularity)
        else:
            raise ValueError(f"Source type {source_type} is not supported.")
        return granularized_corpus

    def dewindowize(self, windowed_corpus: List[Tuple[str]]) -> List[str]:
        result = {}
        for window in windowed_corpus:
            for document in window:
                result[document] = None

        return result.keys()

    def degranularize(self, granularized_corpus: List[str], granularity_source: str) -> str:
        degranularized_corpus = None
        if granularity_source in ["word", "sentence"]:
            degranularized_corpus = " ".join(granularized_corpus)
        elif granularity_source in ["paragraph"]:
            degranularized_corpus = "\n".join(granularized_corpus)
        else:
            ValueError(f"Granularity {granularity_source} is not supported.")
        return degranularized_corpus

    def deprocess(self, processed_corpus: List[Tuple[str]], granularity_source: str) -> str:
        dewindowized_corpus = self.dewindowize(processed_corpus)
        degranularized_corpus = self.degranularize(dewindowized_corpus, granularity_source)
        deprocessed_corpus = degranularized_corpus
        return deprocessed_corpus

    def windowize(self, corpus: str, window_size: int) -> List[Tuple[str, ...]]:
        return list(more_itertools.windowed(corpus, window_size))

    def process(self, corpus: str, source_type: str, granularity: str, window_size: int):
        granularized_corpus = self.granularize(corpus, source_type, granularity)
        windowed_granularized_corpus = self.windowize(granularized_corpus, window_size)
        return windowed_granularized_corpus

    def get_window_sized_processed_corpuses(self, corpus: str, source_type: str, granularity: str,
                                            window_sizes: List[int]) -> List[dict]:
        window_sized_processed_corpus = []
        for window_size in window_sizes:
            processed_corpus = pre_processor.process(corpus, source_type, granularity, window_size)

            window_sized_pre_processed_corpus = {
                "window_size": window_size,
                "processed_corpus": processed_corpus
            }

            window_sized_processed_corpus.append(window_sized_pre_processed_corpus)

        return window_sized_processed_corpus


pre_processor = PreProcessor()
