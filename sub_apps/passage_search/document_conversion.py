from pathlib import Path

import pdfkit
from pdfrw import PdfReader, PdfWriter

from models.passage_search_request import PassageSearchRequest


class DocumentConversion:
    def __init__(self):
        self.options = {
            'page-size': 'Letter',
            'margin-top': '0.25in',
            'margin-right': '1.00in',
            'margin-bottom': '0.25in',
            'margin-left': '1.00in',
        }

    def text_to_pdf(self, text: str, output_file_path: Path) -> Path:
        pdfkit.from_string(text, output_file_path, options=self.options)
        return output_file_path

    def web_to_pdf(self, url: str, output_file_path: Path) -> Path:
        pdfkit.from_url(url, output_file_path, options=self.options)
        return output_file_path

    def file_to_pdf(self, input_file_path: Path, output_file_path: Path) -> Path:
        with open(input_file_path, "rb") as i_f:
            input_file_bytes = i_f.read()
            if input_file_path.suffix == ".pdf":
                with open(output_file_path, "wb") as o_f:
                    o_f.write(input_file_bytes)
            else:
                raise ValueError(f"File type {input_file_path.suffix} is not supported.")
        return output_file_path

    def corpus_to_pdf(self, passage_search_request: PassageSearchRequest, output_file_path: Path) -> Path:
        if passage_search_request.source_type == "text":
            result_output_file_path = self.text_to_pdf(
                text=passage_search_request.corpus,
                output_file_path=output_file_path
            )
        elif passage_search_request.source_type == "file":
            result_output_file_path = self.file_to_pdf(
                input_file_path=Path(passage_search_request.corpus),
                output_file_path=output_file_path
            )
        elif passage_search_request.source_type == "web":
            result_output_file_path = self.web_to_pdf(
                url=passage_search_request.corpus,
                output_file_path=output_file_path
            )
        else:
            raise ValueError(f"Source type {passage_search_request.source_type} is not supported.")

        return result_output_file_path

    def split_pdf_page(self, start_page: int, end_page: int, input_file_path: Path, output_file_path: Path) -> Path:
        pdf_reader = PdfReader(input_file_path)
        pdf_writer = PdfWriter(output_file_path)

        for page_num in range(start_page - 1, end_page):
            pdf_writer.addpage(pdf_reader.pages[page_num])

        pdf_writer.write()

        return output_file_path

    def file_bytes_to_pdf(self, file_bytes: bytes, output_file_path: Path) -> Path:
        with open(output_file_path, "wb") as f:
            f.write(file_bytes)
        return output_file_path

    def get_pdf_page_length(self, input_file_path: Path) -> int:
        pdf_reader = PdfReader(input_file_path)
        pdf_page_length = len(pdf_reader.pages)
        return pdf_page_length


document_conversion = DocumentConversion()
