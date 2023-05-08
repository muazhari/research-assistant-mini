from typing import List, Tuple

import more_itertools
from haystack import Document
from txtai.pipeline import Segmentation, Textractor


class DocumentProcessor:

    def segment(self, corpus: str, granularity: str) -> List[str]:
        granularized_corpus: List[str] = []
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
        granularized_corpus: List[str] = []
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

    def extract_corpus(self, corpus: str, corpus_source_type: str, granularity: str) -> List[str]:
        if corpus_source_type in ["text"]:
            extracted_corpus = self.segment(corpus, granularity)
        elif corpus_source_type in ["file", "web"]:
            extracted_corpus = self.textract(corpus, granularity)
        else:
            raise ValueError(f"Source type {corpus_source_type} is not supported.")
        return extracted_corpus

    def windowize(self, corpus: List[str], window_size: int) -> List[Tuple[str, ...]]:
        return list(more_itertools.windowed(corpus, window_size))

    def degranularize(self, windowed_corpus: Tuple[str], granularity_source: str) -> str:
        degranularized_corpus = None
        if granularity_source in ["word", "sentence"]:
            degranularized_corpus = " ".join(windowed_corpus)
        elif granularity_source in ["paragraph"]:
            degranularized_corpus = "\n".join(windowed_corpus)
        else:
            ValueError(f"Granularity {granularity_source} is not supported.")
        return degranularized_corpus

    def process(self, corpus: str, corpus_source_type: str, granularity: str,
                window_sizes: List[int]) -> List[Document]:
        extracted_corpus: List[str] = self.extract_corpus(corpus, corpus_source_type, granularity)

        processed_documents_with_many_window = []
        for window_size in window_sizes:
            windowed_corpus: List[Tuple[str, ...]] = self.windowize(extracted_corpus, window_size)
            for index_window, content_window in enumerate(windowed_corpus):
                document: Document = Document(
                    content=self.degranularize(content_window, granularity),
                    meta={"index_window": index_window, "window_size": window_size}
                )
                processed_documents_with_many_window.append(document)

        return processed_documents_with_many_window


document_processor = DocumentProcessor()
