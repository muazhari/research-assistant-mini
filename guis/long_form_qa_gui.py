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

    def display(self) -> None:
        st.title("Long Form QA")

        st.subheader("Configurations")

        retriever_model_format: Optional[str] = st.radio(
            label="Pick a retriever model format.",
            options=['dense_passage', 'openai'],
            index=0
        )

        query_embedding_model: Optional[str] = None
        passage_embedding_model: Optional[str] = None
        embedding_dimension: Optional[str] = None
        retriever_openai_api_key: Optional[str] = None
        if retriever_model_format == 'dense_passage':
            query_embedding_model = st.text_input(
                label="Enter a query embedding model.",
                value="vblagoje/dpr-question_encoder-single-lfqa-wiki"
            )
            passage_embedding_model = st.text_input(
                label="Enter a passage embedding model.",
                value="vblagoje/dpr-ctx_encoder-single-lfqa-wiki"
            )
            embedding_dimension = st.number_input(
                label="Enter a embedding dimension.",
                value=128,
            )
            retriever_openai_api_key = None
        elif retriever_model_format == 'openai':
            open_ai_embedding_model = {
                "ada": 1024,
                "babage": 2048,
                "curie": 4096,
                "davinci": 12288
            }
            query_embedding_model = passage_embedding_model = st.radio(
                label="Enter an openai embedding model.",
                options=open_ai_embedding_model.keys(),
                index=0
            )
            embedding_dimension = open_ai_embedding_model[query_embedding_model]
            retriever_openai_api_key = st.text_input(
                label="Enter an OpenAI API key for retriever.",
                value=""
            )
        else:
            st.error("Please select a right model format.")

        embedding_model: EmbeddingModel = EmbeddingModel(
            query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
        )

        similarity_function: Optional[str] = st.radio(
            label="Pick an embedding similarity function.",
            options=['cosine', 'dot_product'],
            index=1
        )

        generator_model_format: Optional[str] = st.radio(
            label="Pick a generator model format.",
            options=['seq2seq', 'openai_answer'],
            index=0
        )

        generator_model: Optional[str] = None
        answer_min_length: Optional[int] = None
        answer_max_length: Optional[int] = None
        answer_max_tokens: Optional[int] = None
        generator_openai_api_key: Optional[str] = None
        if generator_model_format == 'seq2seq':
            generator_model = st.text_input(
                label="Enter a generator model.",
                value="google/flan-t5-large"
            )
            answer_min_length = st.number_input(
                label="Enter a minimum length of the answer.",
                value=50
            )
            answer_max_length = st.number_input(
                label="Enter a maximum length of the answer.",
                value=300
            )
            answer_max_tokens = None
            generator_openai_api_key = None
        elif generator_model_format == 'openai_answer':
            open_ai_generator_model = [
                "text-ada-001",
                "text-babbage-001",
                "text-curie-001",
                "text-davinci-003"
            ]
            generator_model = st.radio(
                label="Enter an openai embedding model.",
                options=open_ai_generator_model,
                index=0
            )
            answer_min_length = None
            answer_max_length = None
            answer_max_tokens = st.number_input(
                label="Enter a maximum tokens in the answer.",
                value=13
            )
            generator_openai_api_key = st.text_input(
                label="Enter an OpenAI API key for generator.",
                value=""
            )
        else:
            st.error("Please select a right model format.")

        source_type: Optional[str] = st.radio(
            label="Pick a source type.",
            options=['file', 'text', 'web'],
            index=1
        )

        corpus: Optional[str] = ""
        if source_type in ['file']:
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
                corpus = str(document_conversion.file_bytes_to_pdf(
                    uploaded_file.getbuffer(), uploaded_file_path))
                st.success("File uploaded!")
        elif source_type in ['text', 'web']:
            corpus = st.text_area(
                label="Enter a corpus.",
                value=""
            )
        else:
            st.error("Please select a right source type.")

        if corpus != "" and source_type in ['file']:
            uploaded_file_name = os.path.splitext(os.path.basename(corpus))[0]
            uploaded_file_page_length = document_conversion.get_pdf_page_length(
                corpus)

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
                value=1
            )

            split_uploaded_file_name = f'{uploaded_file_name}_split_{start_page}_to_{end_page}.pdf'
            split_uploaded_file_path = self.STREAMLIT_STATIC_PATH / f"{split_uploaded_file_name}"
            corpus = str(document_conversion.split_pdf_page(
                start_page, end_page, uploaded_file_path, split_uploaded_file_path))

        query: Optional[str] = st.text_area(
            label="Enter a query.",
            value=""
        )

        granularity: Optional[str] = st.radio(
            label="Pick a granularity.",
            options=['word', 'sentence', 'paragraph'],
            index=1
        )

        window_sizes: Optional[str] = st.text_input(
            label='Enter a list of window sizes that seperated by a space.',
            value='1'
        )

        percentage: Optional[float] = st.slider(
            label='Pick a retriever percentage.',
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            value=10.0
        )
        percentage = percentage / 100

        passage_search_request: PassageSearchRequest = PassageSearchRequest(
            source_type=source_type,
            corpus=corpus,
            query=query,
            granularity=granularity,
            window_sizes=window_sizes,
            percentage=percentage,
            model_format=retriever_model_format,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            similarity_function=similarity_function,
            openai_api_key=retriever_openai_api_key
        )

        lfqa_request: LFQARequest = LFQARequest(
            model_format=generator_model_format,
            generator_model=generator_model,
            answer_min_length=answer_min_length,
            answer_max_length=answer_max_length,
            answer_max_tokens=answer_max_tokens,
            openai_api_key=generator_openai_api_key
        )

        passage_search_request_dict: dict = passage_search_request.dict(exclude={"openai_api_key"})
        lfqa_request_dict: dict = lfqa_request.dict(
            exclude={"openai_api_key", "answer_min_length", "answer_max_length", "answer_max_tokens"})
        if all(value not in [None, ""] for value in
               list(passage_search_request_dict.values())
               + list(lfqa_request_dict.values())
               ):
            lfqa_search_response: LFQAResponse = long_form_qa.qa(
                passage_search_request=passage_search_request,
                lfqa_request=lfqa_request
            )

            answers_response: list = lfqa_search_response.generative_qa_result["answers"]
            metadata_response: dict = answers_response[0].meta
            st.subheader("Output Process Duration")
            st.write(f"{lfqa_search_response.process_duration} seconds")

            st.subheader("Output Content")
            st.write(f"Answer:")
            st.write(f"{answers_response[0].answer}")
            st.write(f"Retrieved documents:")
            retrieved_documents_df: DataFrame = pd.DataFrame(
                columns=["Content", "Score"],
                data=zip(metadata_response["content"],
                         metadata_response["doc_scores"])
            )
            st.table(retrieved_documents_df)


long_form_qa_gui = LongFormQAGUI()
