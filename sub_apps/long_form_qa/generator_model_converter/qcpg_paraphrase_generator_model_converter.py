from typing import List, Optional

from haystack import Document
from transformers import PreTrainedTokenizer, BatchEncoding

from sub_apps.long_form_qa.generator_model_converter.base_generator_model_converter import BaseGeneratorModelConverter


class QCPGParaphraseGeneratorModelConverter(BaseGeneratorModelConverter):

    def __init__(self, model_type: str, lexical: float, syntactic: float, semantic: float, **kwargs: any):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.lexical = lexical
        self.syntactic = syntactic
        self.semantic = semantic
        assert self.model_type in ["captions", "questions", "sentences"]
        self.ranges = {
            "captions": {"lex": [0, 90], "syn": [0, 80], "sem": [0, 95]},
            "sentences": {"lex": [0, 100], "syn": [0, 80], "sem": [0, 95]},
            "questions": {"lex": [0, 90], "syn": [0, 75], "sem": [0, 95]}
        }[self.model_type]

    def __call__(
            self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        text = " ".join([d.content for d in documents])

        assert all([0 <= val <= 1 for val in [self.lexical, self.syntactic, self.semantic]]), \
            f" control values must be between 0 and 1, got {self.lexical}, {self.syntactic}, {self.semantic}"
        names = ["semantic_sim", "lexical_div", "syntactic_div"]
        control = [int(5 * round(val * 100 / 5)) for val in [self.semantic, self.lexical, self.syntactic]]
        control = {name: max(min(val, self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in
                   zip(names, control)}
        control = [f"COND_{name.upper()}_{control[name]}" for name in names]
        assert all(cond in tokenizer.additional_special_tokens for cond in control)
        text = " ".join(control) + text if isinstance(text, str) else [" ".join(control) for t in text]

        return tokenizer(text, truncation=True, padding=True, return_tensors="pt")
