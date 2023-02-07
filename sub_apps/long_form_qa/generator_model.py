from haystack.nodes import OpenAIAnswerGenerator, Seq2SeqGenerator, BaseGenerator
from haystack.nodes.answer_generator.transformers import _BartEli5Converter

from models.lfqa_search_request import LFQARequest
from sub_apps.long_form_qa.generator_model_converter.base_generator_model_converter import BaseGeneratorModelConverter
from sub_apps.long_form_qa.generator_model_converter.parrot_paraphraser_generator_model_converter import \
    ParrotParaphraserGeneratorModelConverter
from sub_apps.long_form_qa.generator_model_converter.qcpg_paraphrase_generator_model_converter import \
    QCPGParaphraseGeneratorModelConverter
from sub_apps.long_form_qa.generator_model_converter.t5_lfqa_generator_model_converter import \
    T5LFQAGeneratorModelConverter


class GeneratorModel:

    def get_seq2seq_generator_input_converter(self, lfqa_request: LFQARequest) -> BaseGeneratorModelConverter:
        if lfqa_request.generator_model in ["yjernite/bart_eli5", "vblagoje/bart_lfqa"]:
            generator_model_input_converter: _BartEli5Converter = _BartEli5Converter()
        elif lfqa_request.generator_model == "prithivida/parrot_paraphraser_on_T5":
            generator_model_input_converter: ParrotParaphraserGeneratorModelConverter = ParrotParaphraserGeneratorModelConverter()
        elif lfqa_request.generator_model == "ibm/qcpg-sentences":
            generator_model_input_converter: QCPGParaphraseGeneratorModelConverter = QCPGParaphraseGeneratorModelConverter(
                model_type="sentences",
                lexical=0.3,
                syntactic=0.5,
                semantic=0.8,
            )
        elif lfqa_request.generator_model == "pszemraj/t5-base-askscience-lfqa":
            generator_model_input_converter: T5LFQAGeneratorModelConverter = T5LFQAGeneratorModelConverter()
        else:
            raise ValueError(f"Generator model {lfqa_request.generator_model} is not supported.")

        return generator_model_input_converter

    def get_seq2seq_generator(self, lfqa_request: LFQARequest) -> Seq2SeqGenerator:
        generator: Seq2SeqGenerator = Seq2SeqGenerator(
            model_name_or_path=lfqa_request.generator_model,
            min_length=lfqa_request.answer_min_length,
            max_length=lfqa_request.answer_max_length,
            use_gpu=True,
            input_converter=self.get_seq2seq_generator_input_converter(lfqa_request=lfqa_request),
        )
        return generator

    def get_openai_answer_generator(self, lfqa_request: LFQARequest) -> OpenAIAnswerGenerator:
        generator: OpenAIAnswerGenerator = OpenAIAnswerGenerator(
            api_key=lfqa_request.openai_api_key,
            model=lfqa_request.generator_model,
            max_tokens=lfqa_request.answer_max_tokens,
            top_k=1,
        )
        return generator

    def get_generator(self, lfqa_request: LFQARequest) -> BaseGenerator:
        if lfqa_request.model_format == "seq2seq":
            generator: Seq2SeqGenerator = self.get_seq2seq_generator(
                lfqa_request=lfqa_request,
            )
        elif lfqa_request.model_format == "openai_answer":
            generator: OpenAIAnswerGenerator = self.get_openai_answer_generator(
                lfqa_request=lfqa_request,
            )
        else:
            raise ValueError(f"Model format {lfqa_request.model_format} is not supported.")
        return generator


generator_model = GeneratorModel()
