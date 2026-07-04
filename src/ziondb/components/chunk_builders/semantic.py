import logging
from typing import List
from ziondb.core.interfaces import ChunkBuilder
from ziondb.core.models import Sentence, BoundaryDecision, Chunk

logger = logging.getLogger(__name__)

class SemanticChunkBuilder(ChunkBuilder):
    """Assembles individual Sentence objects into final Chunk objects using boundary decisions."""

    def build_chunks(
        self, 
        sentences: List[Sentence], 
        decisions: List[BoundaryDecision]
    ) -> List[Chunk]:
        """
        Assembles individual sentences into Chunk objects based on boundary decisions.

        Args:
            sentences: The original list of Sentence objects.
            decisions: The boundary decisions representing where splits should occur.

        Returns:
            List[Chunk]: The assembled Chunk objects.
        """
        if not sentences:
            logger.debug("No sentences provided to ChunkBuilder. Returning empty list.")
            return []

        doc_id = sentences[0].document_id
        
        # Collect indices of sentences that form a boundary after them
        boundary_indices = {d.index for d in decisions if d.is_boundary}
        
        chunks: List[Chunk] = []
        current_sentences: List[Sentence] = []
        chunk_idx = 0

        for idx, sentence in enumerate(sentences):
            current_sentences.append(sentence)

            # Trigger a split if this sentence is marked as a boundary, or if it is the last sentence
            if idx in boundary_indices or idx == len(sentences) - 1:
                chunk_text = " ".join(s.text for s in current_sentences)
                sentence_indices = [s.index for s in current_sentences]

                # Extract bounding offsets from sentences
                start_char = current_sentences[0].start_char
                end_char = current_sentences[-1].end_char

                metadata = {
                    "start_char": start_char,
                    "end_char": end_char,
                    "sentence_count": len(current_sentences)
                }

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=chunk_idx,
                        sentence_indices=sentence_indices,
                        document_id=doc_id,
                        metadata=metadata
                    )
                )
                chunk_idx += 1
                current_sentences = []

        logger.info(f"Assembled {len(chunks)} chunks from {len(sentences)} sentences.")
        return chunks
