from pathlib import Path
import uuid
import pdfkit

import pdfkit
import pdfrw
from pdfrw import PdfReader, PdfWriter
from pyvirtualdisplay import Display
import hashlib

from pre_processor import pre_processor

class DocumentConversion:
    def __init__(self):
        self.options = {
            'page-size': 'Letter',
            'margin-top': '0.25in',
            'margin-right': '1.00in',
            'margin-bottom': '0.25in',
            'margin-left': '1.00in',
        }
    
    def text_to_pdf(self, text: str, output_file_path: Path) -> bytes:
        return pdfkit.from_string(text, output_file_path, options=self.options)
    
    def web_to_pdf(self, url: str, output_file_path: Path) -> bytes: 
        return pdfkit.from_url(url, output_file_path, options=self.options)
        
    def file_to_pdf(self, input_file_path: Path, output_file_path: Path) -> bytes:
        return pdfkit.from_file(input_file_path, output_file_path, options=self.options)
    
    def corpus_to_pdf(self, passage_search_request: dict, output_file_path: Path) -> str:
        if passage_search_request["source_type"] == "text":
            self.text_to_pdf(
                text=passage_search_request["corpus"],
                output_file_path=output_file_path
            )
        elif passage_search_request["source_type"] == "file":
            self.file_to_pdf(
                input_file_path=passage_search_request["corpus"],
                output_file_path=output_file_path
            )
        elif passage_search_request["source_type"] == "web":
            self.web_to_pdf(
                url=passage_search_request["corpus"],
                output_file_path=output_file_path
            )
        else:
            raise ValueError(f"Source type {passage_search_request['source_type']} is not supported.")
        
        return output_file_path
    
    def split_pdf_page(self, start_page: int, end_page: int, input_file_path: Path, output_file_path: Path):
        pdf_reader = PdfReader(input_file_path)
        pdf_writer = PdfWriter(output_file_path)

        for page_num in range(start_page - 1, end_page):
            pdf_writer.addpage(pdf_reader.pages[page_num])

        pdf_writer.write()

        return output_file_path
    
    def file_bytes_to_pdf(self, file_bytes: bytes, output_file_path: Path):
        file_hash = hashlib.md5(file_bytes).hexdigest()
        file_name = "{}.pdf".format(str(file_hash))
        file_path = output_file_path / file_name
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        return output_file_path
    
    def get_pdf_page_length(self, input_file_path: str) -> int:
        pdf_reader = PdfReader(input_file_path)
        pdf_page_length = len(pdf_reader.pages)
        return pdf_page_length

        

document_conversion = DocumentConversion()