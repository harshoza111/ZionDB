import logging
from typing import List, Tuple
from ziondb.core.interfaces import (
    SentenceSplitter, EmbeddingProvider, BoundaryDetector,
    ChunkBuilder, ChunkEmbedder
)
from ziondb.core.models import (
    TextDocument, Sentence, SentenceEmbedding, 
    BoundaryDecision, Chunk, ChunkEmbedding
)

logger = logging.getLogger(__name__)

class DocumentIndexingPipeline:
    """Orchestrates the entire document indexing pipeline from raw document to embedded chunks."""

    def __init__(
        self,
        splitter: SentenceSplitter,
        embedder: EmbeddingProvider,
        detector: BoundaryDetector,
        builder: ChunkBuilder,
        chunk_embedder: ChunkEmbedder
    ) -> None:
        """
        Initialize the indexing pipeline with stage implementations.

        Args:
            splitter: Splits the input document into sentences.
            embedder: Generates embeddings for text segments (sentences).
            detector: Detects semantic boundaries between sentences.
            builder: Assembles sentences into final Chunk objects.
            chunk_embedder: Generates embeddings for the final Chunk objects.
        """
        self.splitter = splitter
        self.embedder = embedder
        self.detector = detector
        self.builder = builder
        self.chunk_embedder = chunk_embedder

    def run(self, document: TextDocument) -> Tuple[List[Chunk], List[ChunkEmbedding]]:
        """
        Runs the indexing pipeline on a TextDocument.

        Args:
            document: The input TextDocument.

        Returns:
            Tuple[List[Chunk], List[ChunkEmbedding]]: A tuple of assembled Chunk objects and their embeddings.
        """
        logger.info(f"Starting indexing pipeline for document ID: '{document.id or 'unknown'}'")
        
        # 1. Split document into sentences
        sentences = self.splitter.split(document)
        n_sentences = len(sentences)
        logger.info(f"Pipeline: split document into {n_sentences} sentences.")
        
        if not sentences:
            logger.info("Pipeline: document text is empty. Returning empty chunks and embeddings.")
            return [], []

        # 2. Embed each individual sentence
        logger.info("Pipeline: generating embeddings for sentences...")
        sentence_texts = [s.text for s in sentences]
        embeddings_raw = self.embedder.embed(sentence_texts)
        
        sentence_embeddings = [
            SentenceEmbedding(sentence_index=s.index, embedding=emb)
            for s, emb in zip(sentences, embeddings_raw)
        ]

        # 3. Detect boundaries
        logger.info("Pipeline: running semantic boundary detector...")
        decisions = self.detector.detect_boundaries(sentences, sentence_embeddings)

        # 4. Group sentences into chunks based on boundary decisions
        logger.info("Pipeline: building chunks...")
        chunks = self.builder.build_chunks(sentences, decisions)

        # 5. Embed the final chunk objects
        logger.info("Pipeline: generating embeddings for final chunks...")
        chunk_embeddings = self.chunk_embedder.embed_chunks(chunks)

        logger.info(f"Pipeline: finished. Generated {len(chunks)} chunks and embeddings.")
        return chunks, chunk_embeddings
