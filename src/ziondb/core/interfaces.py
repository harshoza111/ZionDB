from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ziondb.core.models import (
    TextDocument, Sentence, SentenceEmbedding, 
    BoundaryDecision, Chunk, ChunkEmbedding
)

class SentenceSplitter(ABC):
    """Abstract interface for splitting documents into sentences."""
    
    @abstractmethod
    def split(self, document: TextDocument) -> List[Sentence]:
        """
        Splits a text document into a list of Sentence objects.
        
        Args:
            document: The TextDocument to be split.
            
        Returns:
            List[Sentence]: A list of Sentence objects with text, index, and character bounds.
        """
        pass

class EmbeddingProvider(ABC):
    """Abstract interface for generating embeddings for text lists."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates embedding vectors for a list of input strings.
        
        Args:
            texts: A list of text segments to embed.
            
        Returns:
            List[np.ndarray]: A list of 1D numpy float arrays representing embeddings.
        """
        pass

class BoundaryDetector(ABC):
    """Abstract interface for detecting semantic boundaries between sentences."""
    
    @abstractmethod
    def detect_boundaries(
        self, 
        sentences: List[Sentence], 
        embeddings: List[SentenceEmbedding]
    ) -> List[BoundaryDecision]:
        """
        Analyzes sentences and their embeddings to identify semantic boundary locations.
        
        Args:
            sentences: The list of original Sentence objects.
            embeddings: The list of SentenceEmbedding objects matching the sentences.
            
        Returns:
            List[BoundaryDecision]: Decisions for each boundary checkpoint.
        """
        pass

class ChunkBuilder(ABC):
    """Abstract interface for assembling Sentence objects into final Chunk objects."""
    
    @abstractmethod
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
        pass

class ChunkEmbedder(ABC):
    """Abstract interface for generating embeddings for constructed Chunk objects."""
    
    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[ChunkEmbedding]:
        """
        Computes embeddings for a list of constructed Chunk objects.
        
        Args:
            chunks: The list of Chunk objects to embed.
            
        Returns:
            List[ChunkEmbedding]: The embeddings mapping to the chunks.
        """
        pass
