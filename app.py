import pandas as pd
from passage_search import passage_search
from document_conversion import document_conversion
from annotater import annotater
from search_statistics import search_statistics

from pre_processor import pre_processor
import hashlib
import streamlit as st
import os
import pathlib

from datetime import datetime


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static' / 'static'


st.title("Passage Search")

st.subheader("Configurations")

open_ai_api_key = st.text_input(
    label="Enter an OpenAI API key.",
    value=""
)

source_type = st.radio(
    label="Pick a source type.", 
    options=['file', 'text', 'web'], 
    index=1
)

corpus = ""
if (source_type in ['file']):
    uploaded_file = st.file_uploader(
        label="Upload a file.", 
        type=['pdf'], 
        accept_multiple_files=False
    )

    if None not in [uploaded_file]:
        uploaded_file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
        uploaded_file_name = "{}.pdf".format(uploaded_file_hash)
        uploaded_file_path = STREAMLIT_STATIC_PATH / uploaded_file_name
        corpus = str(document_conversion.file_bytes_to_pdf(uploaded_file.getbuffer(), uploaded_file_path))
        st.success("File uploaded!")
elif (source_type in ['text', 'web']):
    corpus = st.text_area(
        label="Enter a corpus.",
        value=""
    )
else:
    st.error("Please select a right source type.")


if (corpus != "" and source_type in ['file']):
    uploaded_file_name = os.path.splitext(os.path.basename(corpus))[0]
    uploaded_file_page_length = document_conversion.get_pdf_page_length(corpus)

    start_page = st.number_input(
        label=f"Enter the start page of the pdf you want to be highlighted (1-{uploaded_file_page_length}).", 
        min_value=1, 
        max_value=uploaded_file_page_length, 
        value=1
    )
    end_page = st.number_input(
        f"Enter the end page of the pdf you want to be highlighted (1-{uploaded_file_page_length}).", 
        min_value=1,
        max_value=uploaded_file_page_length, 
        value=1
    )

    splitted_uploaded_file_name = f'{uploaded_file_name}_split_{start_page}_to_{end_page}.pdf'
    splitted_uploaded_file_path = STREAMLIT_STATIC_PATH/f"{splitted_uploaded_file_name}"
    corpus = str(document_conversion.split_pdf_page(start_page, end_page, uploaded_file_path, splitted_uploaded_file_path))

query = st.text_area(
    label="Enter a query.",
    value=""
)

granularity = st.radio(
    label="Pick a granularity.", 
    options=['word', 'sentence', 'paragraph'], 
    index=1
)

window_sizes = st.text_input(
    label='Enter a list of window sizes that seperated by a space.', 
    value='1'
)

percentage = st.slider(
    label='Pick a percentage.', 
    min_value=0.0, 
    max_value=1.0,
    step=0.01,
    value=0.1
)

passage_search_request = {
    "source_type": source_type,
    "corpus": corpus,
    "query": query,
    "granularity": granularity,
    "window_sizes": window_sizes,
    "percentage": percentage,
    "open_ai_api_key": open_ai_api_key
}

if (None not in passage_search_request.values() and all(value != "" for value in passage_search_request.values())):
    passage_search_response = passage_search.search(passage_search_request)
                    
    result_windowed_documents = passage_search_response["retrieval_result"]["documents"]
    result_documents = pre_processor.granularize(
        corpus=passage_search_request["corpus"],
        source_type=passage_search_request["source_type"],
        granularity=passage_search_request["granularity"]
    )

    result_document_indexes_with_overlapped_scores = search_statistics.get_document_indexes_with_overlapped_scores(result_windowed_documents)

    result_labels = search_statistics.get_selected_labels(
        document_indexes_with_overlapped_scores=result_document_indexes_with_overlapped_scores,
        percentage=passage_search_request["percentage"]
    )
    
    
    passage_search_request_hash = hashlib.md5(str(passage_search_request).encode("utf-8")).hexdigest()
    pdf_output_file_path = document_conversion.corpus_to_pdf(
        passage_search_request,
        output_file_path = STREAMLIT_STATIC_PATH / f"output_{passage_search_request_hash}.pdf"
    )
    passage_search_request_hash = hashlib.md5(str(passage_search_request).encode("utf-8")).hexdigest()
    highlighted_pdf_output_file_name = f"highlighted_output_{passage_search_request_hash}.pdf"
    highlighted_pdf_output_file_path = STREAMLIT_STATIC_PATH / highlighted_pdf_output_file_name
    highlights = annotater.annotate(
        labels=result_labels, 
        documents=result_documents, 
        input_file_path=pdf_output_file_path, 
        output_file_path=highlighted_pdf_output_file_path
    )
    
    st.subheader("Output Score Overview")
    st.caption(
        "Metric to determine how sure the meaning of the query is in the corpus (score to document in descending order).")
    chart_df = pd.DataFrame(
        data=[value['score_mean'] for value in sorted(
                    result_document_indexes_with_overlapped_scores.values(), 
                    key=lambda value: value["score_mean"], 
                    reverse=True
                )
            ],
        columns=['score']
    )
    st.line_chart(chart_df)
        
    st.subheader("Output Process Duration")
    st.write("{} seconds".format(passage_search_response["process_duration"]))
    
    st.subheader("Output Content")
    pdf_display = f'<iframe src="static/{highlighted_pdf_output_file_name}" width="700" height="1000"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)