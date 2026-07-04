from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from ziondb.collection.collection_config import CollectionConfig
from ziondb.core.interfaces import EmbeddingProvider
from ziondb.core.models import TextDocument
from ziondb.storage.record import ChunkRecord, SystemMetadata
from ziondb.storage.in_memory_storage import InMemoryStorage
from ziondb.index.brute_force_index import BruteForceIndex
from ziondb.index.similarity import CosineSimilarity
from ziondb.retrieval.retriever import Retriever
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_result import SearchResult


class Collection:
    """Represents an independent collection of documents and vectors in ZionDB."""

    def __init__(
        self,
        name: str,
        config: CollectionConfig,
        embedding_provider: Optional[EmbeddingProvider] = None
    ) -> None:
        """
        Initialize a Collection.

        Args:
            name: The unique name of the collection.
            config: The CollectionConfig specifying models and parameters.
            embedding_provider: Optional pre-configured embedding provider (e.g. for testing).
        """
        self.name = name
        self.config = config

        # 1. Initialize the embedding provider (lazy import to optimize load times)
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            self._init_embedding_provider()

        # 2. Setup storage, index, retriever, and pipeline based on self.config
        self._setup_components()

    def _init_embedding_provider(self) -> None:
        """Helper to initialize the default ONNX embedding provider."""
        from ziondb.model_manager.manager import ModelManager
        from ziondb.components.embedders.onnx_embedder import ONNXEmbeddingProvider

        manager = ModelManager(
            model_name=self.config.embedding_model,
            cache_dir=self.config.cache_dir
        )
        model_dir = manager.download_model()
        session = manager.get_onnx_session()
        self.embedding_provider = ONNXEmbeddingProvider(
            model_dir=model_dir,
            session=session,
            max_length=self.config.max_length
        )

    def _setup_components(self) -> None:
        """Helper to initialize storage, vector index, retriever, and chunking pipeline."""
        # 1. Initialize storage
        if self.config.storage_type == "in_memory":
            self.storage = InMemoryStorage()
        else:
            raise ValueError(f"Unsupported storage type: '{self.config.storage_type}'")

        # 2. Resolve similarity metric
        if self.config.similarity_metric.lower() == "cosine":
            metric = CosineSimilarity()
        else:
            raise ValueError(f"Unsupported similarity metric: '{self.config.similarity_metric}'")

        # 3. Initialize vector index
        if self.config.index_type == "brute_force":
            self.index = BruteForceIndex(
                record_provider=self.storage,
                metric=metric
            )
        else:
            raise ValueError(f"Unsupported index type: '{self.config.index_type}'")

        # 4. Initialize retriever
        self.retriever = Retriever(
            vector_index=self.index,
            record_provider=self.storage,
            embedding_provider=self.embedding_provider
        )

        # 5. Initialize chunking pipeline
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        """Helper to initialize the document processing and chunking pipeline."""
        from ziondb.core.pipeline import DocumentIndexingPipeline
        from ziondb.components.chunk_builders import SemanticChunkBuilder
        from ziondb.components.embedders.onnx_embedder import ONNXChunkEmbedder
        from ziondb.components.boundary_detectors import KamradtBoundaryDetector

        # Sentence Splitter
        if self.config.splitter_type == "regex":
            from ziondb.components.sentence_splitters import RegexSentenceSplitter
            if self.config.regex_pattern is not None:
                splitter = RegexSentenceSplitter(pattern=self.config.regex_pattern)
            else:
                splitter = RegexSentenceSplitter()
        elif self.config.splitter_type == "spacy":
            from ziondb.components.sentence_splitters import SpacySentenceSplitter
            splitter = SpacySentenceSplitter(model_name=self.config.spacy_model)
        else:
            raise ValueError(f"Unsupported splitter type: '{self.config.splitter_type}'")

        # Boundary Detector
        detector = KamradtBoundaryDetector(
            embedder=self.embedding_provider if self.config.use_text_embedding else None,
            buffer_size=self.config.buffer_size,
            threshold_type=self.config.threshold_type,
            threshold_value=self.config.threshold_value
        )

        # Semantic Chunk Builder
        builder = SemanticChunkBuilder()

        # Chunk Embedder
        chunk_embedder = ONNXChunkEmbedder(self.embedding_provider)

        # Main Pipeline Orchestrator
        self.pipeline = DocumentIndexingPipeline(
            splitter=splitter,
            embedder=self.embedding_provider,
            detector=detector,
            builder=builder,
            chunk_embedder=chunk_embedder
        )

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Adds a single text document to the collection.
        Slices the document into chunks, embeds them, stores them, and updates the vector index.

        Args:
            text: The text content of the document.
            metadata: Optional dictionary of metadata.
            id: Optional unique identifier for the document. Auto-generated if not provided.

        Returns:
            str: The document ID.
        """
        doc_id = id or str(uuid.uuid4())
        doc_metadata = metadata or {}
        doc = TextDocument(text=text, metadata=doc_metadata, id=doc_id)
        self.add_documents([doc])
        return doc_id

    def add_documents(self, documents: List[TextDocument]) -> None:
        """
        Adds multiple TextDocument objects to the collection.
        Slices, embeds, stores, and indexes the chunks.

        Args:
            documents: List of TextDocument objects.
        """
        for doc in documents:
            if not doc.id:
                doc.id = str(uuid.uuid4())

            # Overwrite/Upsert logic: remove any existing records with the same document ID
            existing_ids = []
            for record in self.storage.iterate():
                if record.system_metadata.document_id == doc.id:
                    existing_ids.append(record.id)

            for record_id in existing_ids:
                try:
                    self.storage.delete(record_id)
                    self.index.remove(record_id)
                except Exception:
                    pass

            # Run the chunking and embedding pipeline
            chunks, chunk_embeddings = self.pipeline.run(doc)

            if not chunks:
                continue

            for chunk, chunk_emb in zip(chunks, chunk_embeddings):
                # Format a unique ID for the chunk record
                record_id = f"{doc.id}#chunk_{chunk.index}"

                # Approximate token count (simple word count split)
                token_count = len(chunk.text.split())

                sys_meta = SystemMetadata(
                    document_id=doc.id,
                    embedding_model=self.config.embedding_model,
                    chunking_method=self.config.chunking_method,
                    token_count=token_count,
                    created_at=datetime.now(timezone.utc)
                )

                record = ChunkRecord(
                    id=record_id,
                    text=chunk.text,
                    embedding=chunk_emb.embedding,
                    metadata=chunk.metadata,
                    system_metadata=sys_meta
                )

                # Store chunk and vector embedding representation
                self.storage.insert(record)
                self.index.insert(record)

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_metric: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Performs similarity search over the collection for the query.

        Args:
            query: The text query.
            top_k: The number of results to return.
            metadata_filter: Optional metadata filtering criteria (not implemented).
            similarity_metric: Optional similarity metric override (not implemented).

        Returns:
            List[SearchResult]: The search results.
        """
        request = SearchRequest(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            similarity_metric=similarity_metric
        )
        return self.retriever.retrieve(request)

    def count(self) -> int:
        """Returns the total number of chunks/records stored in the collection."""
        return self.storage.size()

    def clear(self) -> None:
        """Clears all documents/chunks and index representations in the collection."""
        self.storage.clear()
        self.index.rebuild()

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the collection state to the specified directory.

        Args:
            path: Directory path where the collection files should be written.
        """
        from ziondb.persistence.collection_persistence import CollectionPersistence
        persistence = CollectionPersistence()
        persistence.save(Path(path), self)

    def load(self, path: Union[str, Path]) -> None:
        """
        Loads and restores the collection state from the specified directory.

        Args:
            path: Directory path from which collection files should be loaded.
        """
        from ziondb.persistence.collection_persistence import CollectionPersistence
        persistence = CollectionPersistence()

        dir_path = Path(path)
        name, config, records = persistence.load(dir_path)

        # Re-initialize embedding provider if model/cache config changed
        model_changed = (
            self.config.embedding_model != config.embedding_model or
            self.config.cache_dir != config.cache_dir or
            self.config.max_length != config.max_length
        )

        self.name = name
        self.config = config

        if model_changed:
            from ziondb.components.embedders.onnx_embedder import ONNXEmbeddingProvider
            # Only trigger reload if currently using ONNX (allows retaining mock providers in tests)
            if isinstance(self.embedding_provider, ONNXEmbeddingProvider):
                self._init_embedding_provider()

        # Setup all sub-components (storage, metrics, index, pipeline) with new configuration
        self._setup_components()

        # Re-populate storage
        self.storage.clear()
        for record in records:
            self.storage.insert(record)

        # Rebuild/load index dynamically
        persistence.load_index(dir_path, self.index, config.index_type)
