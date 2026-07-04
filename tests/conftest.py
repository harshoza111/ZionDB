import pytest
import numpy as np
from typing import List
from ziondb.core.models import TextDocument
from ziondb.core.interfaces import EmbeddingProvider

@pytest.fixture
def sample_document() -> TextDocument:
    return TextDocument(
        text="This is sentence one. This is sentence two! Sentence three is here? Yes, sentence four.",
        metadata={"source": "test_suite"},
        id="doc_1"
    )

class MockEmbeddingProvider(EmbeddingProvider):
    """Generates dummy 384-dimensional normalized vectors for testing."""
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        for i, text in enumerate(texts):
            # Create a 384-dimensional vector with distinct values based on text length or index
            vec = np.zeros(384)
            # Add some seed differences
            val = float(len(text) % 10) + float(i)
            vec[0] = val
            vec[1] = 10.0 - val
            # Normalize to unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return embeddings

@pytest.fixture
def mock_embedder() -> EmbeddingProvider:
    return MockEmbeddingProvider()
