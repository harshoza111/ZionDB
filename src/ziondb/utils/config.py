import yaml
from pathlib import Path
from typing import Any, Dict, Union
from ziondb.core.models import (
    PipelineConfig, ModelConfig, SplitterConfig, BoundaryDetectorConfig
)

def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    Loads and parses the YAML configuration file into a PipelineConfig object.

    Args:
        config_path: Path to the pipeline config YAML file.

    Returns:
        PipelineConfig: The strongly-typed configuration object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pipeline_data = data.get("pipeline", {})
    
    # 1. Parse ModelConfig
    model_data = pipeline_data.get("model", {})
    model_config = ModelConfig(
        name=model_data.get("name", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_dir=model_data.get("cache_dir", "models"),
        max_length=model_data.get("max_length", 256)
    )

    # 2. Parse SplitterConfig
    splitter_data = pipeline_data.get("splitter", {})
    splitter_config = SplitterConfig(
        type=splitter_data.get("type", "spacy"),
        model_name=splitter_data.get("model_name", "en_sentencizer"),
        regex_pattern=splitter_data.get("regex_pattern", r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+")
    )

    # 3. Parse BoundaryDetectorConfig
    bd_data = pipeline_data.get("boundary_detector", {})
    bd_config = BoundaryDetectorConfig(
        buffer_size=bd_data.get("buffer_size", 1),
        threshold_type=bd_data.get("threshold_type", "percentile"),
        threshold_value=float(bd_data.get("threshold_value", 95.0)),
        use_text_embedding=bool(bd_data.get("use_text_embedding", True))
    )

    return PipelineConfig(
        model=model_config,
        splitter=splitter_config,
        boundary_detector=bd_config
    )
