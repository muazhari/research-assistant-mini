from typing import List, Optional

from haystack import Document
from transformers import PreTrainedTokenizer, BatchEncoding

from sub_apps.long_form_qa.generator_model_converter.base_generator_model_converter import BaseGeneratorModelConverter


class T5LFQAGeneratorModelConverter(BaseGeneratorModelConverter):
    def __call__(
            self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        conditioned_doc = "<P> " + " <P> ".join([d.content for d in documents])

        # concatenate question and support document into BART input
        query_and_docs = "question: {} context: {}".format(query, conditioned_doc)

        return tokenizer([(query_and_docs, "A")], truncation=True, padding=True, return_tensors="pt")
