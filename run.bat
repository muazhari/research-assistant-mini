@echo off
cmd /k "conda activate research-assistant && cd C:\Data\Apps\research-assistant-mini && python -m streamlit.web.cli run app.py --server.address 0.0.0.0 --server.port 8501"
pause