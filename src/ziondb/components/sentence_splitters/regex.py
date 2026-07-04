import re
from typing import List
from ziondb.core.interfaces import SentenceSplitter
from ziondb.core.models import TextDocument, Sentence

class RegexSentenceSplitter(SentenceSplitter):
    """A rules/regex-based sentence splitter that tracks precise character offsets."""

    def __init__(self, pattern: str = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+") -> None:
        """
        Initialize the RegexSentenceSplitter.

        Args:
            pattern: Regular expression pattern representing the delimiter (whitespace following sentence-ending punctuation).
                     Includes negative lookbehinds for abbreviations like "U.S.A." or "Mr.".
        """
        self.pattern = re.compile(pattern)

    def split(self, document: TextDocument) -> List[Sentence]:
        """
        Splits a text document into a list of Sentence objects.

        Args:
            document: The TextDocument to split.

        Returns:
            List[Sentence]: A list of Sentence objects with text, index, and original character offsets.
        """
        text = document.text
        if not text.strip():
            return []

        sentences: List[Sentence] = []
        doc_id = document.id or "doc"
        
        last_idx = 0
        sent_index = 0

        # Iterate over all punctuation-delimiter matches
        for match in self.pattern.finditer(text):
            start, end = match.span()
            sent_slice = text[last_idx:start]
            
            # Clean up leading/trailing whitespace offsets
            stripped = sent_slice.strip()
            if stripped:
                lstrip_len = len(sent_slice) - len(sent_slice.lstrip())
                rstrip_len = len(sent_slice) - len(sent_slice.rstrip())
                
                sentences.append(
                    Sentence(
                        text=stripped,
                        index=sent_index,
                        start_char=last_idx + lstrip_len,
                        end_char=start - rstrip_len,
                        document_id=doc_id
                    )
                )
                sent_index += 1
            last_idx = end

        # Handle the remaining trailing portion of the text
        remaining = text[last_idx:]
        stripped_remaining = remaining.strip()
        if stripped_remaining:
            lstrip_len = len(remaining) - len(remaining.lstrip())
            rstrip_len = len(remaining) - len(remaining.rstrip())
            
            sentences.append(
                Sentence(
                    text=stripped_remaining,
                    index=sent_index,
                    start_char=last_idx + lstrip_len,
                    end_char=len(text) - rstrip_len,
                    document_id=doc_id
                )
            )

        return sentences
