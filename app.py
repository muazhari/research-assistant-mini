import streamlit as st

from guis.long_form_qa_gui import long_form_qa_gui
from guis.passage_search_gui import passage_search_gui


class App:
    def __init__(self) -> None:
        pass

    def display(self) -> None:

        sub_app = st.sidebar.radio(
            label="Select a sub app.",
            options=["document_network", "document_search",
                     "passage_search", "long_form_qa"],
            index=2
        )

        if sub_app == "document_network":
            pass
        elif sub_app == "document_search":
            pass
        elif sub_app == "passage_search":
            passage_search_gui.display()
        elif sub_app == "long_form_qa":
            long_form_qa_gui.display()
        else:
            raise ValueError(f"Sub app {sub_app} is not supported.")


app = App()
app.display()
