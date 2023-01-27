import re
from typing import List, Tuple, Optional, Any
from txtmarker.factory import Factory
from pathlib import Path


class Annotater():

    def annotate(self, labels: List[str], documents: List[str], input_file_path: Path, output_file_path: Path) -> List[Tuple[str, str]]:
        highlights = []
        for label, document in zip(labels, documents):
            highlight = (label, document)
            highlights.append(highlight)
            
        highlighter = Factory.create("pdf", self.formatter, 4)
        highlighter.highlight(str(input_file_path), str(output_file_path), highlights)
        
        return highlights

    def formatter(self, text):
        """
        Custom formatter that is passed to PDF Annotation method. This logic maps data cleansing logic in paperetl.

        Reference: https://github.com/neuml/paperetl/blob/master/src/python/paperetl/text.py

        Args:
            text: input text

        Returns:
            clean text
        """

        # List of patterns
        patterns = []

        # Remove emails
        patterns.append(r"\w+@\w+(\.[a-z]{2,})+")

        # Remove urls
        patterns.append(r"http(s)?\:\/\/\S+")

        # Remove single characters repeated at least 3 times (ex. j o u r n a l)
        patterns.append(r"(^|\s)(\w\s+){3,}")

        # Remove citations references (ex. [3] [4] [5])
        patterns.append(r"(\[\d+\]\,?\s?){3,}(\.|\,)?")

        # Remove citations references (ex. [3, 4, 5])
        patterns.append(r"\[[\d\,\s]+\]")

        # Remove citations references (ex. (NUM1) repeated at least 3 times with whitespace
        patterns.append(r"(\(\d+\)\s){3,}")

        # Build regex pattern
        pattern = re.compile("|".join([f"({p})" for p in patterns]))

        # Clean/transform text
        text = pattern.sub(" ", text)

        # Remove extra spacing either caused by replacements or already in text
        text = re.sub(r" {2,}|\.{2,}", " ", text)

        # Limit to alphanumeric characters
        text = re.sub(r"[^A-Za-z0-9]", "", text)

        return text


annotater = Annotater()