import logging
from typing import List
import spacy
import spacy.cli
from ziondb.core.interfaces import SentenceSplitter
from ziondb.core.models import TextDocument, Sentence

logger = logging.getLogger(__name__)

class SpacySentenceSplitter(SentenceSplitter):
    """A sentence splitter leveraging spaCy's linguistic parsing or rule-based sentencizer."""

    def __init__(self, model_name: str = "en_sentencizer") -> None:
        """
        Initialize the SpacySentenceSplitter.

        Args:
            model_name: Either "en_sentencizer" to use a blank English pipeline with only the sentencizer
                        component, or a full spaCy pipeline name (e.g. "en_core_web_sm").
        """
        self.model_name = model_name

        if model_name == "en_sentencizer":
            logger.info("Initializing blank English spaCy pipeline with sentencizer...")
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        else:
            logger.info(f"Loading spaCy model '{model_name}'...")
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not found locally. Attempting to download...")
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)


    def split(self, document: TextDocument) -> List[Sentence]:
        """
        Splits a text document into a list of Sentence objects.

        Args:
            document: The TextDocument to split.

        Returns:
            List[Sentence]: A list of Sentence objects with text, index, and offsets.
        """
        text = document.text
        if not text.strip():
            return []

        doc = self.nlp(text)
        sentences: List[Sentence] = []
        doc_id = document.id or "doc"

        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text
            stripped = sent_text.strip()
            
            if not stripped:
                continue

            # Adjust character offsets to exclude leading/trailing whitespace
            lstrip_len = len(sent_text) - len(sent_text.lstrip())
            rstrip_len = len(sent_text) - len(sent_text.rstrip())

            sentences.append(
                Sentence(
                    text=stripped,
                    index=idx,
                    start_char=sent.start_char + lstrip_len,
                    end_char=sent.end_char - rstrip_len,
                    document_id=doc_id
                )
            )

        return sentences
