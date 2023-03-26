from typing import List, Optional

from haystack import Document
from transformers import PreTrainedTokenizer, BatchEncoding

from sub_apps.long_form_qa.generator_model_converter.base_generator_model_converter import BaseGeneratorModelConverter


class FlanT5GeneratorModelConverter(BaseGeneratorModelConverter):
    def __call__(
            self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        paragraphs = " ".join([d.content for d in documents])
        prompt = f"Synthesize a comprehensive answer from the following topk most relevant paragraphs and the given question. Provide an elaborated long answer from the key points and information in the paragraphs. Say irrelevant if the paragraphs are irrelevant to the question, then explain why it is irrelevant. \n\n Paragraphs: {paragraphs} \n\n Question: {query} \n\n Answer:"
        return tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
