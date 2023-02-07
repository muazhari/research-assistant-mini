import os
import sys

run_command: str = f"streamlit run app.py {' '.join(sys.argv[1:])}"
os.system(run_command)
