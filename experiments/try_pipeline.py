import logging
from pathlib import Path
import numpy as np

from ziondb.core.models import TextDocument
from ziondb.core.pipeline import DocumentIndexingPipeline
from ziondb.model_manager.manager import ModelManager
from ziondb.components.sentence_splitters import SpacySentenceSplitter, RegexSentenceSplitter
from ziondb.components.embedders.onnx_embedder import ONNXEmbeddingProvider, ONNXChunkEmbedder
from ziondb.components.boundary_detectors import KamradtBoundaryDetector
from ziondb.components.chunk_builders import SemanticChunkBuilder
from ziondb.utils import load_config, setup_logging

def main() -> None:
    # 1. Setup logging to console
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("ziondb.experiments.try_pipeline")
    logger.info("Initializing ZionDB End-to-End Verification...")

    # 2. Find and load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "pipeline_config.yaml"
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # 3. Model Downloader and Manager
    logger.info("Initializing Model Manager...")
    model_manager = ModelManager(
        model_name=config.model.name,
        cache_dir=project_root / config.model.cache_dir
    )
    # Download assets from HF Hub
    logger.info("Checking/Downloading embedding model (all-MiniLM-L6-v2)...")
    model_dir = model_manager.download_model()
    onnx_session = model_manager.get_onnx_session()

    # 4. Instantiate Pipeline Components
    logger.info("Instantiating pipeline components...")
    
    # Splitter
    if config.splitter.type == "spacy":
        splitter = SpacySentenceSplitter(model_name=config.splitter.model_name)
    else:
        splitter = RegexSentenceSplitter(pattern=config.splitter.regex_pattern)

    # Embedders
    embedder = ONNXEmbeddingProvider(
        model_dir=model_dir,
        session=onnx_session,
        max_length=config.model.max_length
    )
    chunk_embedder = ONNXChunkEmbedder(embedder=embedder)

    # Boundary Detector (injecting embedder for text re-embedding)
    detector = KamradtBoundaryDetector(
        embedder=embedder if config.boundary_detector.use_text_embedding else None,
        buffer_size=config.boundary_detector.buffer_size,
        threshold_type=config.boundary_detector.threshold_type,
        threshold_value=config.boundary_detector.threshold_value
    )

    # Chunk Builder
    builder = SemanticChunkBuilder()

    # 5. Assemble and Run the Pipeline
    pipeline = DocumentIndexingPipeline(
        splitter=splitter,
        embedder=embedder,
        detector=detector,
        builder=builder,
        chunk_embedder=chunk_embedder
    )

    # 6. Sample text document with a distinct transition in topic
    # Topic 1: Vector Databases -> Topic 2: Space Colonization / Mars
    document_text = (
        "Vector databases are specifically designed to store and query high-dimensional vector embeddings. "
        "Unlike relational databases that search based on exact rows or keyword indexes, vector databases perform semantic searches. "
        "They utilize distance metrics like cosine similarity or Euclidean distance to retrieve the most relevant documents. "
        "These systems are key components in modern Retrieval-Augmented Generation (RAG) architectures. "
        "On the other hand, humanity's future might lie among the stars. "
        "Mars is currently the primary target for human colonization and space exploration. "
        "Sending astronauts to the Red Planet requires solving complex radiation protection and atmospheric logistics. "
        "NASA and SpaceX are actively building heavy rocket infrastructure like Starship to support these deep space missions. "
        "We must become a multi-planetary species to ensure the survival of human consciousness."
    )

    document = TextDocument(
        text=document_text,
        metadata={"category": "technology_science_transition"},
        id="demo_e2e_001"
    )

    logger.info("Executing indexing pipeline on sample document...")
    chunks, chunk_embeddings = pipeline.run(document)

    # 7. Print and Validate the output
    print("\n" + "="*80)
    print("ZIONDB INDEXING PIPELINE E2E RESULTS")
    print("="*80)
    print(f"Original Text Length: {len(document_text)} characters")
    print(f"Generated Chunks count: {len(chunks)}")
    print(f"Generated Embeddings count: {len(chunk_embeddings)}")
    print("-"*80)

    for chunk, emb in zip(chunks, chunk_embeddings):
        print(f"CHUNK {chunk.index} (Character bounds: [{chunk.metadata['start_char']}:{chunk.metadata['end_char']}], Sentences: {chunk.metadata['sentence_count']}):")
        print(f"  Content: \"{chunk.text}\"")
        print(f"  Embedding shape: {emb.embedding.shape}, Norm: {np.linalg.norm(emb.embedding):.4f}")
        print("-"*80)

if __name__ == "__main__":
    main()
