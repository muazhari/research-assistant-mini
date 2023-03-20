
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

- Passage Search (Old Version)
[![demo](http://img.youtube.com/vi/bu93G6YesaQ/0.jpg)](http://www.youtube.com/watch?v=bu93G6YesaQ)

## Walkthrough 

### Local
1. Run `python run.py --server.address 0.0.0.0 --server.port 8501` (Run as Administrator if in Windows).
2. Open URL `http://localhost:8501` in a browser.

### Jupyter Notebook
1. Get your Open AI API key.
2. Get your Ngrok Authentication Token.
3. Create cell based on below Jupyter Notebook script in Google Colab, Kaggle, or other alternatives.

```python
#@title Research Assistant Mini App
NGROK_TOKEN = "" #@param {type:"string"} 

%cd ~
!git clone https://github.com/muazhari/research-assistant-mini.git
%cd ~/research-assistant-mini/
!git fetch --all
!git reset --hard origin

!apt-get update -y
!yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq wkhtmltopdf xvfb libopenblas-dev libomp-dev poppler-utils openjdk-8-jdk jq

!pip install -r requirements.txt
!pip install pyngrok farm-haystack[all] txtai[all]

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

## Warning
- This repository not yet peer reviewed, so be careful when using it.
