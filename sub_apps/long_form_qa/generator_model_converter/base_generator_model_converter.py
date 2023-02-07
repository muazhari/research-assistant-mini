from typing import List, Optional

from haystack import Document
from haystack.nodes.answer_generator.transformers import _BartEli5Converter
from transformers import PreTrainedTokenizer, BatchEncoding


class BaseGeneratorModelConverter(_BartEli5Converter):
    def __call__(
            self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        pass
