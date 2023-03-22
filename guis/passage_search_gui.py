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
from models.passage_search_request import PassageSearchRequest
from models.passage_search_response import PassageSearchResponse
from sub_apps.passage_search.annotater import annotater
from sub_apps.passage_search.document_conversion import document_conversion
from sub_apps.passage_search.passage_search import passage_search
from sub_apps.passage_search.search_statistics import search_statistics
from utilities.pre_processor import pre_processor


class PassageSearchGUI:

    def __init__(self) -> None:
        nltk.download('punkt')
        self.STREAMLIT_STATIC_PATH: Path = pathlib.Path(st.__path__[0]) / 'static' / 'static'

    def display(self) -> None:
        st.title("Passage Search")

        st.subheader("Configurations")

        retriever_source_type: Optional[str] = st.radio(
            label="Pick a retriever source.",
            options=['local', 'openai'],
            index=0
        )

        retriever: Optional[str] = None
        query_embedding_model: Optional[str] = None
        passage_embedding_model: Optional[str] = None
        embedding_dimension: Optional[int] = None
        num_iterations: Optional[int] = None
        api_key: Optional[str] = None
        if retriever_source_type == 'local':
            retriever = st.radio(
                label="Pick a retriever.",
                options=['multihop', 'dense_passage'],
                index=0
            )

            if retriever == 'multihop':
                query_embedding_model = passage_embedding_model = st.text_input(
                    label="Enter an embedding model.",
                    value="sentence-transformers/all-mpnet-base-v2"
                )
            elif retriever == 'dense_passage':
                query_embedding_model = st.text_input(
                    label="Enter a query embedding model.",
                    value="facebook/dpr-question_encoder-multiset-base"
                )
                passage_embedding_model = st.text_input(
                    label="Enter a passage embedding model.",
                    value="facebook/dpr-ctx_encoder-multiset-base"
                )
            else:
                st.error("Please select a right retriever.")

            embedding_dimension = st.number_input(
                label="Enter an embedding dimension.",
                value=768
            )

            num_iterations = st.number_input(
                label="Enter a number of iterations/hops.",
                value=2
            )

            api_key = None
        elif retriever_source_type == 'openai':
            retriever = "basic"
            open_ai_model = {
                "ada": 1024,
                "babbage": 2048,
                "curie": 4096,
                "davinci": 12288
            }
            query_embedding_model = passage_embedding_model = st.radio(
                label="Enter an embedding model.",
                options=open_ai_model.keys(),
                index=3
            )
            embedding_dimension = open_ai_model[query_embedding_model]
            api_key = st.text_input(
                label="Enter an OpenAI API key.",
                value=""
            )
        else:
            st.error("Please select a right retriever source.")

        embedding_model = EmbeddingModel(
            query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
        )

        similarity_function = st.radio(
            label="Pick an embedding similarity function.",
            options=['cosine', 'dot_product'],
            index=1
        )

        corpus_source_type = st.radio(
            label="Pick a corpus source type.",
            options=['file', 'text', 'web'],
            index=0
        )

        corpus: Optional[str] = None
        uploaded_file_path: Optional[Path] = None
        if corpus_source_type in ['file']:
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
                corpus = str(document_conversion.file_bytes_to_pdf(uploaded_file.getbuffer(), uploaded_file_path))
                st.success("File uploaded!")
        elif corpus_source_type in ['text', 'web']:
            corpus = st.text_area(
                label="Enter a corpus.",
                value=""
            )
        else:
            st.error("Please select a right source type.")

        if corpus is not None and corpus_source_type in ['file']:
            uploaded_file_name = os.path.splitext(os.path.basename(corpus))[0]
            uploaded_file_page_length = document_conversion.get_pdf_page_length(
                corpus)

            start_page: int = st.number_input(
                label=f"Enter the start page of the pdf you want to be retrieved (1-{uploaded_file_page_length}).",
                min_value=1,
                max_value=uploaded_file_page_length,
                value=1
            )
            end_page: int = st.number_input(
                f"Enter the end page of the pdf you want to be retrieved (1-{uploaded_file_page_length}).",
                min_value=1,
                max_value=uploaded_file_page_length,
                value=uploaded_file_page_length
            )

            split_uploaded_file_name: str = f'{uploaded_file_name}_split_{start_page}_to_{end_page}.pdf'
            split_uploaded_file_path: Path = self.STREAMLIT_STATIC_PATH / f"{split_uploaded_file_name}"
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
            value='1 3 5'
        )

        percentage: Optional[float] = st.slider(
            label='Pick a percentage.',
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            value=10.0
        )
        percentage = percentage / 100

        passage_search_request: PassageSearchRequest = PassageSearchRequest(
            corpus_source_type=corpus_source_type,
            corpus=corpus,
            query=query,
            granularity=granularity,
            window_sizes=window_sizes,
            percentage=percentage,
            retriever_source_type=retriever_source_type,
            retriever=retriever,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            num_iterations=num_iterations,
            similarity_function=similarity_function,
            api_key=api_key
        )

        passage_search_request_dict: dict = passage_search_request.dict(exclude={"api_key"})
        if all(value not in [None, ""] for value in passage_search_request_dict.values()):
            passage_search_response: PassageSearchResponse = passage_search.search(
                passage_search_request=passage_search_request
            )

            result_windowed_documents: list = passage_search_response.retrieval_result["documents"]
            result_documents = pre_processor.granularize(
                corpus=passage_search_request.corpus,
                corpus_source_type=passage_search_request.corpus_source_type,
                granularity=passage_search_request.granularity
            )

            result_document_indexes_with_overlapped_scores: dict[int, dict] = \
                search_statistics.get_document_indexes_with_overlapped_scores(result_windowed_documents)

            selected_result_labels: list = search_statistics.get_selected_labels(
                document_indexes_with_overlapped_scores=result_document_indexes_with_overlapped_scores,
                percentage=passage_search_request.percentage
            )
            selected_result_documents: list[str] = search_statistics.get_selected_documents(
                document_indexes_with_overlapped_scores=result_document_indexes_with_overlapped_scores,
                percentage=passage_search_request.percentage,
                source_documents=result_documents
            )

            passage_search_request_hash: str = hashlib.md5(str(passage_search_request).encode("utf-8")).hexdigest()
            pdf_output_file_path: Path = document_conversion.corpus_to_pdf(
                passage_search_request,
                output_file_path=self.STREAMLIT_STATIC_PATH / f"output_{passage_search_request_hash}.pdf"
            )
            highlighted_pdf_output_file_path: Path = annotater.annotate(
                labels=selected_result_labels,
                documents=selected_result_documents,
                input_file_path=pdf_output_file_path,
                output_file_path=self.STREAMLIT_STATIC_PATH / f"highlighted_output_{passage_search_request_hash}.pdf",
            )
            highlighted_pdf_output_file_name: str = os.path.basename(highlighted_pdf_output_file_path)

            st.subheader("Output Score Overview")
            st.caption(
                "Metric to determine how sure the meaning of the query is in the corpus (score_mean to document in descending order).")
            sorted_result_document_indexes_with_overlapped_scores_items: list = sorted(
                result_document_indexes_with_overlapped_scores.items(),
                key=lambda item: item[1]["score_mean"],
                reverse=True
            )
            chart_df: DataFrame = pd.DataFrame(
                data=[value['score_mean']
                      for key, value in sorted_result_document_indexes_with_overlapped_scores_items],
                columns=['score']
            )
            st.line_chart(chart_df)

            st.subheader("Output Process Duration")
            st.write("{} seconds".format(
                passage_search_response.process_duration))

            st.subheader("Output Content")

            st.write(f"Highlighted documents:")
            pdf_display = f'<iframe src="static/{highlighted_pdf_output_file_name}" width="700" height="1000"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.write(f"Retrieved documents:")

            retrieved_documents_df: DataFrame = pd.DataFrame(
                columns=["Content", "Score Mean", "Score Count"],
                data=zip(
                    [result_documents[key] for key, value in
                     sorted_result_document_indexes_with_overlapped_scores_items],
                    [value["score_mean"] for key, value in
                     sorted_result_document_indexes_with_overlapped_scores_items],
                    [value["count"] for key, value in
                     sorted_result_document_indexes_with_overlapped_scores_items]
                )
            )
            st.table(retrieved_documents_df)


passage_search_gui = PassageSearchGUI()
