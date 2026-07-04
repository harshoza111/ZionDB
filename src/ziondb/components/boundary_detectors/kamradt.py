import logging
from typing import List, Optional
import numpy as np

from ziondb.core.interfaces import BoundaryDetector, EmbeddingProvider
from ziondb.core.models import Sentence, SentenceEmbedding, BoundaryDecision

logger = logging.getLogger(__name__)

class KamradtBoundaryDetector(BoundaryDetector):
    """
    Implements the Kamradt semantic boundary detection algorithm.
    Computes cosine distances of adjacent sentence group representations and applies a threshold.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingProvider] = None,
        buffer_size: int = 1,
        threshold_type: str = "percentile",
        threshold_value: float = 95.0
    ) -> None:
        """
        Initialize the KamradtBoundaryDetector.

        Args:
            embedder: Optional EmbeddingProvider. If provided, sentence groups will be concatenated as text
                      and embedded. If None, individual sentence embeddings will be mathematically averaged.
            buffer_size: The window size (k) of neighboring sentences to group on each side.
            threshold_type: Either "percentile" or "standard_deviation".
            threshold_value: The value for the threshold (e.g. 95.0 for percentile, or 1.2 for standard_deviation).
        """
        self.embedder = embedder
        self.buffer_size = buffer_size
        self.threshold_type = threshold_type.lower()
        self.threshold_value = threshold_value

        if self.threshold_type not in ("percentile", "standard_deviation"):
            raise ValueError(
                f"Invalid threshold_type: '{threshold_type}'. Must be 'percentile' or 'standard_deviation'."
            )

    def detect_boundaries(
        self, 
        sentences: List[Sentence], 
        embeddings: List[SentenceEmbedding]
    ) -> List[BoundaryDecision]:
        """
        Analyzes sentences and their embeddings to identify semantic boundary locations.

        Args:
            sentences: List of Sentence objects.
            embeddings: List of SentenceEmbedding objects matching the sentences.

        Returns:
            List[BoundaryDecision]: Decisions for each sentence split point.
        """
        n_sentences = len(sentences)
        if n_sentences < 2:
            logger.info("Fewer than 2 sentences provided. No boundary decisions can be made.")
            return []

        if len(embeddings) != n_sentences:
            raise ValueError(
                f"Mismatch in count of sentences ({n_sentences}) and embeddings ({len(embeddings)})"
            )

        logger.info(
            f"Detecting boundaries for {n_sentences} sentences using "
            f"buffer_size={self.buffer_size}, threshold_type={self.threshold_type}, "
            f"threshold_value={self.threshold_value}"
        )

        group_vectors: List[np.ndarray] = []

        if self.embedder is not None:
            # 1. Text-based grouping: concatenate sentences in the buffer window and re-embed
            logger.debug("Generating group representations using text-concatenation embedding...")
            grouped_texts: List[str] = []
            for i in range(n_sentences):
                start = max(0, i - self.buffer_size)
                end = min(n_sentences - 1, i + self.buffer_size)
                group_text = " ".join(sentences[j].text for j in range(start, end + 1))
                grouped_texts.append(group_text)
                
            group_vectors = self.embedder.embed(grouped_texts)
        else:
            # 2. Vector-based grouping: average the individual sentence embedding vectors in the buffer window
            logger.debug("Generating group representations using sliding-window vector averaging...")
            emb_dict = {se.sentence_index: se.embedding for se in embeddings}
            
            for i in range(n_sentences):
                start = max(0, i - self.buffer_size)
                end = min(n_sentences - 1, i + self.buffer_size)
                
                # Fetch vectors in window
                vectors = [emb_dict[j] for j in range(start, end + 1)]
                mean_vec = np.mean(vectors, axis=0)
                
                # Normalize the averaged vector
                norm = np.linalg.norm(mean_vec)
                normalized_vec = mean_vec / max(norm, 1e-9)
                group_vectors.append(normalized_vec)

        # 3. Calculate cosine distances between consecutive group embeddings
        distances: List[float] = []
        for i in range(n_sentences - 1):
            vec_a = group_vectors[i]
            vec_b = group_vectors[i + 1]
            similarity = float(np.dot(vec_a, vec_b))
            distance = 1.0 - similarity
            distances.append(distance)

        # 4. Compute boundary threshold
        if self.threshold_type == "percentile":
            threshold = float(np.percentile(distances, self.threshold_value))
        else:  # standard_deviation
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            threshold = mean_dist + self.threshold_value * std_dist

        logger.info(f"Calculated boundary distance threshold: {threshold:.4f}")

        # 5. Build boundary decisions (decision i corresponds to the gap after sentence i)
        decisions: List[BoundaryDecision] = []
        for i, dist in enumerate(distances):
            is_boundary = dist >= threshold
            distance_to_threshold = dist - threshold
            decisions.append(
                BoundaryDecision(
                    index=i,
                    distance=dist,
                    is_boundary=is_boundary,
                    distance_to_threshold=distance_to_threshold
                )
            )

        return decisions
