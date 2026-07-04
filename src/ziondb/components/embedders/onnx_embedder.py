import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

from ziondb.core.interfaces import EmbeddingProvider, ChunkEmbedder
from ziondb.core.models import Chunk, ChunkEmbedding

logger = logging.getLogger(__name__)

class ONNXEmbeddingProvider(EmbeddingProvider):
    """Concrete implementation of EmbeddingProvider using ONNX Runtime and Transformers Tokenizer."""

    def __init__(
        self, 
        model_dir: Union[str, Path], 
        session: Optional[ort.InferenceSession] = None,
        max_length: int = 256
    ) -> None:
        """
        Initialize the ONNX Embedding Provider.

        Args:
            model_dir: Path to directory containing local model files and tokenizer.
            session: Optional pre-initialized ONNX session. If None, it will be loaded from model_dir/onnx/model.onnx.
            max_length: Maximum sequence length for the tokenizer.
        """
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        logger.info(f"Initializing tokenizer from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        if session is not None:
            self.session = session
        else:
            onnx_path = self.model_dir / "onnx" / "model.onnx"
            logger.info(f"Initializing ONNX session from {onnx_path}...")
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        self.session_inputs = {node.name for node in self.session.get_inputs()}

    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Perform mean pooling on token embeddings, taking the attention mask into account.

        Args:
            token_embeddings: Output from ONNX model of shape [batch_size, seq_len, hidden_size].
            attention_mask: Mask of shape [batch_size, seq_len] with 1s for real tokens and 0s for padding.

        Returns:
            np.ndarray: Mean-pooled embeddings of shape [batch_size, hidden_size].
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        
        # Sum token embeddings weighted by attention mask
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        
        # Sum attention mask weight along the sequence length
        sum_mask = np.sum(input_mask_expanded, axis=1)
        
        # Clamp sum_mask to avoid division by zero
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to project embeddings onto the unit hypersphere.

        Args:
            embeddings: Embeddings of shape [batch_size, hidden_size].

        Returns:
            np.ndarray: L2 normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates embedding vectors for a list of input strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List[np.ndarray]: List of 1D numpy float arrays representing embeddings.
        """
        if not texts:
            return []

        # Tokenize all text strings
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )

        # Build dynamic inputs dict matching what the ONNX model expects
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded and "token_type_ids" in self.session_inputs:
            inputs["token_type_ids"] = encoded["token_type_ids"]

        # Run model inference
        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0]  # Shape: [batch_size, seq_len, hidden_size]

        # Apply mean pooling followed by L2 normalization
        pooled = self._mean_pooling(last_hidden_state, encoded["attention_mask"])
        normalized = self._l2_normalize(pooled)

        # Unpack batch into a list of 1D arrays
        return [normalized[i] for i in range(normalized.shape[0])]


class ONNXChunkEmbedder(ChunkEmbedder):
    """Concrete implementation of ChunkEmbedder leveraging an underlying EmbeddingProvider."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        """
        Initialize the ONNX Chunk Embedder.

        Args:
            embedder: Any EmbeddingProvider implementation to run the embedding math.
        """
        self.embedder = embedder

    def embed_chunks(self, chunks: List[Chunk]) -> List[ChunkEmbedding]:
        """
        Computes embeddings for a list of constructed Chunk objects.

        Args:
            chunks: The list of Chunk objects to embed.

        Returns:
            List[ChunkEmbedding]: The generated chunk embeddings.
        """
        if not chunks:
            return []

        # Extract text from chunks and run the embedder
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        return [
            ChunkEmbedding(chunk_index=chunk.index, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]
