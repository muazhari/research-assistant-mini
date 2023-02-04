from haystack.nodes import OpenAIAnswerGenerator, Seq2SeqGenerator, BaseGenerator

from haystack.document_stores import FAISSDocumentStore, PineconeDocumentStore, BaseDocumentStore

class GeneratorModel:
    
    def get_seq2seq_generator(self, lfqa_request: dict) -> Seq2SeqGenerator:
        generator: Seq2SeqGenerator = Seq2SeqGenerator(
            model_name_or_path=lfqa_request["generator_model"],
            min_length=lfqa_request["answer_min_length"],
            max_length=lfqa_request["answer_max_length"],
            use_gpu=True,
        )
        return generator
    
    def get_openai_answer_generator(self, lfqa_request: dict) -> OpenAIAnswerGenerator:
        generator: OpenAIAnswerGenerator = OpenAIAnswerGenerator(
            api_key=lfqa_request["api_key"],
            model=lfqa_request["generator_model"],
            max_tokens=lfqa_request["answer_max_tokens"],
            top_k=1,
        )
        return generator
    
    def get_generator(self, lfqa_request: dict) -> BaseGenerator:
        generator: BaseGenerator = None
        if lfqa_request["model_format"] == "seq2seq":
            generator = self.get_seq2seq_generator(
                lfqa_request=lfqa_request,
            )
        elif lfqa_request["model_format"] == "openai_answer":
            generator = self.get_openai_answer_generator(
                lfqa_request=lfqa_request,
            )
        else:
            raise ValueError(f"Model format {lfqa_request['model_format']} is not supported.")
        return generator
            

generator_model = GeneratorModel()
            