from typing import List, Optional

from haystack import Document
from transformers import PreTrainedTokenizer, BatchEncoding

from sub_apps.long_form_qa.generator_model_converter.base_generator_model_converter import BaseGeneratorModelConverter


class   FlanT5SummarizerGeneratorModelConverter(BaseGeneratorModelConverter):
    def __call__(
            self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        context = " ".join([d.content for d in documents])
        prompt = f"question: {query} context: {context} rephrase context as answer with logical transitions: "
        return tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
