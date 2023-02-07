import re
from typing import List, Optional

from haystack import Document
from transformers import PreTrainedTokenizer, BatchEncoding

from sub_apps.long_form_qa.generator_model_converter.base_generator_model_converter import BaseGeneratorModelConverter


class ParrotParaphraserGeneratorModelConverter(BaseGeneratorModelConverter):
    def __call__(
            self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        conditioned_doc = " ".join([d.content for d in documents])

        input_phrase = re.sub("[^a-zA-Z0-9 \?\'\-\/\:\.]", "", conditioned_doc)
        input_phrase = f"paraphrase: {input_phrase}"

        return tokenizer(input_phrase, truncation=True, padding=True, return_tensors="pt")
