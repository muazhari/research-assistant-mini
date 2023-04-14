
# Research Assistant Mini

Research Assistant Mini App using Streamlit user interface.

## Features

- Passage Search (Done)
- Long Form QA (Done)
- Document Network (Not yet implemented)
- Document Search (Not yet implemented)

## Authors

- [@muazhari](https://github.com/muazhari) 

## Demo

- Passage Search

[![passage_search_demo](https://img.youtube.com/vi/3CR1Vnyx8ik/0.jpg)](https://youtu.be/3CR1Vnyx8ik)

- Long Form QA

[![long-form_qa_demo](https://img.youtube.com/vi/Ih-qgRqUpzc/0.jpg)](https://youtu.be/Ih-qgRqUpzc)



## Walkthrough 

### Local
1. Install the Nvidia GPU driver and CUDA package, then ensure the system can use GPU CUDA cores (via anaconda).
2. Install Python 3.9.x, [JDK 8](https://www.oracle.com/id/java/technologies/javase/javase8-archive-downloads.html), wkhtmltopdf, & Apache Tika (via chocolatey or apt).
3. Run `git clone https://github.com/muazhari/research-assistant-mini.git`.
4. Go to `research-assistant-mini` directory.
5. Run `pip install -r requirements.txt && pip install farm-haystack[only-faiss,only-faiss-gpu,crawler,preprocessing,ocr] txtai[pipeline]`. 
6. Get your Open AI API key.
7. Run `python run.py --server.address 0.0.0.0 --server.port 8501` (Run as Administrator if in Windows).
8. Open URL `http://localhost:8501` in a browser.
9. Use the app.

### Jupyter Notebook (Recommended)
1. Get your Open AI API key.
2. Get your Ngrok Authentication Token.
3. Create cell based on below Jupyter Notebook script in Kaggle, or other alternatives.

```python
#@title Research Assistant Mini App
NGROK_TOKEN = "" #@param {type:"string"} 


# Python version upgrade script. Use this if the python version is not equal to 3.9.
!conda create -n newCondaEnvironment -c cctbx202208 -y
!source /opt/conda/bin/activate newCondaEnvironment && conda install -c cctbx202208 python=3.9 -y
!/opt/conda/envs/newCondaEnvironment/bin/python3 --version
!echo 'print("Hello, World!")' > test.py
!/opt/conda/envs/newCondaEnvironment/bin/python3 test.py
!sudo rm /opt/conda/bin/python3
!sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3
!sudo rm /opt/conda/bin/python3.7
!sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3.7
!sudo rm /opt/conda/bin/python
!sudo ln -s /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python
!sudo ln -s /opt/conda/envs/newCondaEnvironment/bin/ngrok /opt/conda/bin/ngrok
!sudo ln -s /opt/conda/envs/newCondaEnvironment/bin/streamlit /opt/conda/bin/streamlit
!python --version

# Installation script.
%cd ~
!git clone https://github.com/muazhari/research-assistant-mini.git
%cd ~/research-assistant-mini/
!git fetch --all
!git reset --hard origin

!apt-get update -y
!yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq wkhtmltopdf xvfb libopenblas-dev libomp-dev poppler-utils openjdk-8-jdk jq

!pip install -r requirements.txt
!pip install pyngrok farm-haystack[only-faiss,only-faiss-gpu,crawler,preprocessing,ocr] txtai[pipeline]

!nvidia-smi

get_ipython().system_raw(f'ngrok authtoken {NGROK_TOKEN}')
get_ipython().system_raw('ngrok http 8501 &')
print("Open public URL:")
!curl -s http://localhost:4040/api/tunnels | jq ".tunnels[0].public_url"
!streamlit run ~/research-assistant-mini/app.py

!sleep 10000000
```

4. Submit your ngrok Authentication Token to `NGROK_TOKEN` column in the cell form.
5. Enable GPU in the Notebook.
6. Run the cell.
7. Wait until the setups are done.
8. Open Ngrok public URL.
9. Use the app.

## Notes

1. If you want to process text in another language, you can use the other language model or the multilingual model. For example, use the settings below if you want to process Indonesian text:
   - Retriever DPR Query: `voidful/dpr-question_encoder-bert-base-multilingual`
    - Retriever DPR Passage:  `voidful/dpr-ctx_encoder-bert-base-multilingual`
    - Retriever Embedding Dimension: `768`
    - Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
    - Prompt: 
```
Sintesiskan jawaban komprehensif dari paragraf-paragraf berikut yang paling relevan dan pertanyaan yang diberikan. Berikan jawaban panjang yang diuraikan dari poin-poin utama dan informasi dalam paragraf-paragraf. Katakan tidak relevan jika paragraf-paragraf tidak relevan dengan pertanyaan, lalu jelaskan mengapa itu tidak relevan.
Paragraf-paragraf: {join(documents)}
Pertanyaan: {query}
Jawaban:
```
2. Delete the entire contents of the `document_store` folder if there is an error. Just delete the contents, don't delete the folder.

## Warning
- This repository not yet peer reviewed, so be careful when using it.
