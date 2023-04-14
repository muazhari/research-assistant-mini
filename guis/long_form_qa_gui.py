import hashlib
import os
import pathlib
from pathlib import Path
from typing import Optional

import nltk
import pandas as pd
import streamlit as st
from pandas import DataFrame

from models.embedding_model import EmbeddingModel
from models.lfqa_response import LFQAResponse
from models.lfqa_search_request import LFQARequest
from models.passage_search_request import PassageSearchRequest
from sub_apps.long_form_qa.lfqa import long_form_qa
from sub_apps.passage_search.document_conversion import document_conversion


class LongFormQAGUI:
    def __init__(self) -> None:
        nltk.download('punkt')
        self.STREAMLIT_STATIC_PATH: Path = pathlib.Path(st.__path__[0]) / 'static' / 'static'

        self.passage_search_request: PassageSearchRequest = PassageSearchRequest()
        self.passage_search_request.embedding_model = EmbeddingModel()
        self.lfqa_request: LFQARequest = LFQARequest()

    def display(self) -> None:
        st.title("Long Form QA")

        st.subheader("Configurations")

        self.passage_search_request.retriever_source_type = st.radio(
            label="Pick a retriever source type.",
            options=['local', 'openai'],
            index=0
        )

        if self.passage_search_request.retriever_source_type == 'local':
            self.passage_search_request.dense_retriever = st.radio(
                label="Pick a dense retriever.",
                options=['multihop', 'dense_passage'],
                index=1
            )

            if self.passage_search_request.dense_retriever == 'multihop':
                self.passage_search_request.embedding_model.query_model = self.passage_search_request.embedding_model.passage_model = st.text_input(
                    label="Enter an embedding model.",
                    value="sentence-transformers/all-mpnet-base-v2"
                )
                self.passage_search_request.embedding_dimension = st.number_input(
                    label="Enter an embedding dimension.",
                    value=768
                )
                self.passage_search_request.num_iterations = st.number_input(
                    label="Enter a number of iterations/hops.",
                    value=2
                )

            elif self.passage_search_request.dense_retriever == 'dense_passage':
                self.passage_search_request.embedding_model.query_model = st.text_input(
                    label="Enter a query embedding model.",
                    value="vblagoje/dpr-question_encoder-single-lfqa-wiki"
                )
                self.passage_search_request.embedding_model.passage_model = st.text_input(
                    label="Enter a passage embedding model.",
                    value="vblagoje/dpr-ctx_encoder-single-lfqa-wiki"
                )
                self.passage_search_request.embedding_dimension = st.number_input(
                    label="Enter an embedding dimension.",
                    value=128
                )
            else:
                st.error("Please select a right dense retriever.")
            self.passage_search_request.api_key = None

        elif self.passage_search_request.retriever_source_type == 'openai':
            self.passage_search_request.dense_retriever = "basic"
            open_ai_model = {
                "ada": 1024,
                "babbage": 2048,
                "curie": 4096,
                "davinci": 12288
            }
            self.passage_search_request.embedding_model.query_model = self.passage_search_request.embedding_model.passage_model = st.radio(
                label="Enter an embedding model.",
                options=open_ai_model.keys(),
                index=3
            )
            self.passage_search_request.embedding_dimension = open_ai_model[
                self.passage_search_request.embedding_model.query_model]
            self.passage_search_request.api_key = st.text_input(
                label="Enter an OpenAI API key.",
                value="",
                type="password"
            )
        else:
            st.error("Please select a right retriever source.")

        self.passage_search_request.similarity_function = st.radio(
            label="Pick an embedding similarity function.",
            options=['cosine', 'dot_product'],
            index=1
        )

        self.passage_search_request.sparse_retriever = st.radio(
            label="Pick a sparse retriever.",
            options=['tfidf', 'bm25'],
            index=1
        )

        self.passage_search_request.ranker = st.radio(
            label="Pick a ranker.",
            options=['sentence_transformers'],
            index=0
        )

        if self.passage_search_request.ranker == 'sentence_transformers':
            self.passage_search_request.embedding_model.ranker_model = st.text_input(
                label="Enter a ranker model.",
                value="naver/trecdl22-crossencoder-electra"
            )
        else:
            st.error("Please select a right ranker.")

        self.lfqa_request.generator_model_format = st.radio(
            label="Pick a generator model format.",
            options=['seq2seq', 'llm_prompt'],
            index=1
        )

        if self.lfqa_request.generator_model_format == 'seq2seq':
            self.lfqa_request.generator_model = st.text_input(
                label="Enter a generator model.",
                value="google/flan-t5-large"
            )
            self.lfqa_request.prompt = None
            self.lfqa_request.answer_min_length = st.number_input(
                label="Enter a minimum length of the answer.",
                value=300
            )
            self.lfqa_request.answer_max_length = st.number_input(
                label="Enter a maximum length of the answer.",
                value=500
            )
            self.lfqa_request.answer_max_tokens = None
            self.lfqa_request.api_key = None
        elif self.lfqa_request.generator_model_format == 'llm_prompt':
            self.lfqa_request.generator_model_source_type = st.radio(
                label="Pick a generator model source type.",
                options=['local', 'online'],
                index=1
            )

            if self.lfqa_request.generator_model_source_type == 'online':
                self.lfqa_request.api_key = st.text_input(
                    label="Enter an API key for generator model.",
                    value="",
                    type="password"
                )
                self.lfqa_request.generator_model = st.text_input(
                    label="Enter a generator model.",
                    value="gpt-3.5-turbo"
                )
            elif self.lfqa_request.generator_model_source_type == 'local':
                self.lfqa_request.api_key = None
                self.lfqa_request.generator_model = st.text_input(
                    label="Enter a generator model.",
                    value="google/flan-t5-large"
                )
            else:
                st.error("Please select a right generator source type.")

            self.lfqa_request.prompt = st.text_area(
                label="Enter a prompt.",
                value="Synthesize a comprehensive answer from the following topk most relevant paragraphs and the given question. Provide an elaborated long answer from the key points and information in the paragraphs. Say irrelevant if the paragraphs are irrelevant to the question, then explain why it is irrelevant. \n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:"
            )
            self.lfqa_request.answer_min_length = None
            self.lfqa_request.answer_max_length = st.number_input(
                label="Enter a maximum length of the answer.",
                value=500
            )
            self.lfqa_request.answer_max_tokens = None
        else:
            st.error("Please select a right model format.")

        self.passage_search_request.corpus_source_type = st.radio(
            label="Pick a source type.",
            options=['file', 'text', 'web'],
            index=0
        )

        uploaded_file_path: Optional[Path] = None
        if self.passage_search_request.corpus_source_type in ['file']:
            uploaded_file = st.file_uploader(
                label="Upload a file.",
                type=['pdf'],
                accept_multiple_files=False
            )

            if None not in [uploaded_file]:
                uploaded_file_hash = hashlib.md5(
                    uploaded_file.getbuffer()).hexdigest()
                uploaded_file_name = "{}.pdf".format(uploaded_file_hash)
                uploaded_file_path = self.STREAMLIT_STATIC_PATH / uploaded_file_name
                self.passage_search_request.corpus = str(document_conversion.file_bytes_to_pdf(
                    uploaded_file.getbuffer(), uploaded_file_path))
                st.success("File uploaded!")
        elif self.passage_search_request.corpus_source_type in ['text', 'web']:
            self.passage_search_request.corpus = st.text_area(
                label="Enter a corpus.",
                value=""
            )
        else:
            st.error("Please select a right source type.")

        if self.passage_search_request.corpus not in ["", None] and self.passage_search_request.corpus_source_type in [
            'file']:
            uploaded_file_name = os.path.splitext(os.path.basename(self.passage_search_request.corpus))[0]
            uploaded_file_page_length = document_conversion.get_pdf_page_length(self.passage_search_request.corpus)

            start_page = st.number_input(
                label=f"Enter the start page of the pdf you want to be retrieved (1-{uploaded_file_page_length}).",
                min_value=1,
                max_value=uploaded_file_page_length,
                value=1
            )
            end_page = st.number_input(
                f"Enter the end page of the pdf you want to be retrieved (1-{uploaded_file_page_length}).",
                min_value=1,
                max_value=uploaded_file_page_length,
                value=uploaded_file_page_length
            )

            split_uploaded_file_name = f'{uploaded_file_name}_split_{start_page}_to_{end_page}.pdf'
            split_uploaded_file_path = self.STREAMLIT_STATIC_PATH / f"{split_uploaded_file_name}"
            self.passage_search_request.corpus = str(document_conversion.split_pdf_page(
                start_page, end_page, uploaded_file_path, split_uploaded_file_path))

        self.passage_search_request.query = st.text_area(
            label="Enter a query.",
            value=""
        )

        self.passage_search_request.granularity = st.radio(
            label="Pick a granularity.",
            options=['word', 'sentence', 'paragraph'],
            index=1
        )

        self.passage_search_request.window_sizes = st.text_input(
            label='Enter a list of window sizes that seperated by a space.',
            value='1 3 5'
        )

        self.passage_search_request.retriever_top_k = st.number_input(
            label="Enter a top-k for each retriever.",
            value=100
        )

        self.passage_search_request.ranker_top_k = st.number_input(
            label="Enter a top-k for each rerank.",
            value=15
        )

        passage_search_request_dict: dict = self.passage_search_request.dict(
            exclude={"api_key", "num_iterations"}
        )
        lfqa_request_dict: dict = self.lfqa_request.dict(
            exclude={"api_key", "answer_min_length", "answer_max_length", "answer_max_tokens",
                     "generator_model_source_type", "prompt"}
        )

        if all(value not in [None, ""] for value in
               list(passage_search_request_dict.values())
               + list(lfqa_request_dict.values())
               ):
            lfqa_search_response: LFQAResponse = long_form_qa.qa(
                passage_search_request=self.passage_search_request,
                lfqa_request=self.lfqa_request
            )

            documents_response: list = lfqa_search_response.generative_qa_result["_debug"]["Ranker"]["output"][
                "documents"]
            answers_response: list = lfqa_search_response.generative_qa_result["answers"]

            st.subheader("Output Score Overview")
            st.caption(
                "Metric to determine how sure the meaning of the query is in the retrieved documents (score to document in descending order).")

            chart_df: DataFrame = pd.DataFrame(
                data=[(doc.score) for doc in documents_response],
                columns=['score']
            )
            st.line_chart(chart_df)

            st.subheader("Output Process Duration")
            st.write(f"{lfqa_search_response.process_duration} seconds")

            st.subheader("Output Content")
            st.write(f"Answer:")
            st.write(f"{answers_response[0].answer}")
            st.write(f"Retrieved documents:")
            retrieved_documents_df: DataFrame = pd.DataFrame(
                columns=["Content", "Score"],
                data=[(doc.content, doc.score) for doc in documents_response]
            )
            st.table(retrieved_documents_df)


long_form_qa_gui = LongFormQAGUI()
