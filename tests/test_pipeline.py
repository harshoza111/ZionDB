import pytest
import numpy as np
from ziondb.core.models import (
    TextDocument, Sentence, SentenceEmbedding, BoundaryDecision, Chunk
)
from ziondb.core.pipeline import DocumentIndexingPipeline
from ziondb.components.sentence_splitters import RegexSentenceSplitter, SpacySentenceSplitter
from ziondb.components.boundary_detectors import KamradtBoundaryDetector
from ziondb.components.chunk_builders import SemanticChunkBuilder
from ziondb.components.embedders.onnx_embedder import ONNXChunkEmbedder

# ==========================================
# 1. Test Sentence Splitters
# ==========================================

def test_regex_sentence_splitter(sample_document):
    splitter = RegexSentenceSplitter()
    sentences = splitter.split(sample_document)
    
    assert len(sentences) == 4
    assert sentences[0].text == "This is sentence one."
    assert sentences[1].text == "This is sentence two!"
    assert sentences[2].text == "Sentence three is here?"
    assert sentences[3].text == "Yes, sentence four."
    
    # Verify indexes and character offsets
    for idx, s in enumerate(sentences):
        assert s.index == idx
        assert s.document_id == sample_document.id
        # Sliced text from document should match exactly
        assert sample_document.text[s.start_char:s.end_char] == s.text

def test_spacy_sentence_splitter(sample_document):
    splitter = SpacySentenceSplitter(model_name="en_sentencizer")
    sentences = splitter.split(sample_document)
    
    assert len(sentences) == 4
    assert sentences[0].text == "This is sentence one."
    assert sentences[1].text == "This is sentence two!"
    assert sentences[2].text == "Sentence three is here?"
    assert sentences[3].text == "Yes, sentence four."
    
    # Verify character offsets
    for idx, s in enumerate(sentences):
        assert s.index == idx
        assert sample_document.text[s.start_char:s.end_char] == s.text

def test_splitters_empty_document():
    doc = TextDocument(text="   \n   ", metadata={}, id="empty")
    
    reg_splitter = RegexSentenceSplitter()
    sp_splitter = SpacySentenceSplitter(model_name="en_sentencizer")
    
    assert len(reg_splitter.split(doc)) == 0
    assert len(sp_splitter.split(doc)) == 0


# ==========================================
# 2. Test Boundary Detector
# ==========================================

def test_kamradt_boundary_detector_percentile(mock_embedder):
    sentences = [
        Sentence(text="A", index=0, start_char=0, end_char=1, document_id="doc"),
        Sentence(text="B", index=1, start_char=2, end_char=3, document_id="doc"),
        Sentence(text="C", index=2, start_char=4, end_char=5, document_id="doc"),
        Sentence(text="D", index=3, start_char=6, end_char=7, document_id="doc"),
    ]
    
    # Generate mock embeddings (all distinct vectors)
    raw_embs = mock_embedder.embed([s.text for s in sentences])
    embeddings = [
        SentenceEmbedding(sentence_index=i, embedding=emb) 
        for i, emb in enumerate(raw_embs)
    ]
    
    # Detector with percentile threshold
    detector = KamradtBoundaryDetector(
        embedder=mock_embedder, 
        buffer_size=1, 
        threshold_type="percentile", 
        threshold_value=50.0
    )
    
    decisions = detector.detect_boundaries(sentences, embeddings)
    # 4 sentences -> 3 boundary check gaps
    assert len(decisions) == 3
    for d in decisions:
        assert isinstance(d, BoundaryDecision)
        assert d.distance >= 0.0
        
    # Since threshold is 50th percentile, at least one boundary decision should be True
    # (depending on distances relative to percentile)
    has_boundary = any(d.is_boundary for d in decisions)
    assert has_boundary is True

def test_kamradt_boundary_detector_std(mock_embedder):
    sentences = [
        Sentence(text="Hello world", index=0, start_char=0, end_char=11, document_id="doc"),
        Sentence(text="Hello world indeed", index=1, start_char=12, end_char=30, document_id="doc"),
        Sentence(text="Completely different topic here", index=2, start_char=31, end_char=62, document_id="doc"),
    ]
    raw_embs = mock_embedder.embed([s.text for s in sentences])
    embeddings = [
        SentenceEmbedding(sentence_index=i, embedding=emb) 
        for i, emb in enumerate(raw_embs)
    ]
    
    # Detector with standard deviation (0.0 means threshold = mean)
    detector = KamradtBoundaryDetector(
        embedder=None,  # Tests vector averaging instead of text re-embedding
        buffer_size=1, 
        threshold_type="standard_deviation", 
        threshold_value=0.0
    )
    
    decisions = detector.detect_boundaries(sentences, embeddings)
    assert len(decisions) == 2

def test_kamradt_boundary_detector_edge_cases():
    detector = KamradtBoundaryDetector()
    # Less than 2 sentences
    decisions = detector.detect_boundaries(
        [Sentence(text="One", index=0, start_char=0, end_char=3, document_id="doc")],
        [SentenceEmbedding(sentence_index=0, embedding=np.zeros(384))]
    )
    assert len(decisions) == 0


# ==========================================
# 3. Test Semantic Chunk Builder
# ==========================================

def test_semantic_chunk_builder():
    sentences = [
        Sentence(text="S1", index=0, start_char=0, end_char=2, document_id="doc"),
        Sentence(text="S2", index=1, start_char=3, end_char=5, document_id="doc"),
        Sentence(text="S3", index=2, start_char=6, end_char=8, document_id="doc"),
        Sentence(text="S4", index=3, start_char=9, end_char=11, document_id="doc"),
    ]
    
    # Decisions marking a boundary after S2 (index 1)
    decisions = [
        BoundaryDecision(index=0, distance=0.1, is_boundary=False, distance_to_threshold=-0.1),
        BoundaryDecision(index=1, distance=0.5, is_boundary=True, distance_to_threshold=0.3),
        BoundaryDecision(index=2, distance=0.2, is_boundary=False, distance_to_threshold=-0.0),
    ]
    
    builder = SemanticChunkBuilder()
    chunks = builder.build_chunks(sentences, decisions)
    
    # Should result in 2 chunks: [S1, S2] and [S3, S4]
    assert len(chunks) == 2
    
    assert chunks[0].index == 0
    assert chunks[0].text == "S1 S2"
    assert chunks[0].sentence_indices == [0, 1]
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == 5
    assert chunks[0].metadata["sentence_count"] == 2
    
    assert chunks[1].index == 1
    assert chunks[1].text == "S3 S4"
    assert chunks[1].sentence_indices == [2, 3]
    assert chunks[1].metadata["start_char"] == 6
    assert chunks[1].metadata["end_char"] == 11
    assert chunks[1].metadata["sentence_count"] == 2


# ==========================================
# 4. Test Orchestrator Pipeline
# ==========================================

def test_indexing_pipeline_e2e(sample_document, mock_embedder):
    splitter = SpacySentenceSplitter(model_name="en_sentencizer")
    detector = KamradtBoundaryDetector(embedder=mock_embedder, buffer_size=1, threshold_value=50.0)
    builder = SemanticChunkBuilder()
    chunk_embedder = ONNXChunkEmbedder(mock_embedder)
    
    pipeline = DocumentIndexingPipeline(
        splitter=splitter,
        embedder=mock_embedder,
        detector=detector,
        builder=builder,
        chunk_embedder=chunk_embedder
    )
    
    chunks, chunk_embeddings = pipeline.run(sample_document)
    
    assert len(chunks) > 0
    assert len(chunk_embeddings) == len(chunks)
    
    # Assert data formats
    for i, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
        assert isinstance(chunk, Chunk)
        assert chunk.index == i
        assert chunk.document_id == sample_document.id
        
        assert chunk_emb.chunk_index == chunk.index
        assert isinstance(chunk_emb.embedding, np.ndarray)
        assert chunk_emb.embedding.shape == (384,)
