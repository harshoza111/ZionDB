from abc import ABC, abstractmethod
import numpy as np

class SimilarityMetric(ABC):
    """Abstract interface for vector similarity calculation."""

    @property
    @abstractmethod
    def greater_is_better(self) -> bool:
        """
        Indicates if higher scores represent better matches.

        Returns:
            bool: True if higher scores are better (e.g. Cosine Similarity), 
                  False if lower scores are better (e.g. Euclidean Distance).
        """
        pass

    @abstractmethod
    def calculate(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates similarity score between two 1D vectors.

        Args:
            v1: First 1D numpy array.
            v2: Second 1D numpy array.

        Returns:
            float: The calculated similarity score.
        """
        pass


class CosineSimilarity(SimilarityMetric):
    """Cosine similarity metric calculator."""

    @property
    def greater_is_better(self) -> bool:
        return True

    def calculate(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Computes the cosine similarity: (v1 . v2) / (||v1|| ||v2||).

        Args:
            v1: First 1D numpy array.
            v2: Second 1D numpy array.

        Returns:
            float: Cosine similarity score (between -1.0 and 1.0).
        """
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
            
        similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return float(similarity)
