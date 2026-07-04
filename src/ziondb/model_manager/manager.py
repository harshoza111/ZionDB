import logging
from pathlib import Path
from typing import List, Optional, Union
from huggingface_hub import snapshot_download
import onnxruntime as ort

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages downloading, local caching, and loading of ONNX embedding models."""

    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
        cache_dir: Union[str, Path] = "models"
    ) -> None:
        """
        Initialize the ModelManager.
        
        Args:
            model_name: Hugging Face model repository name.
            cache_dir: Directory where models are downloaded and cached.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir).resolve()
        self.model_dir: Optional[Path] = None

    def download_model(self) -> Path:
        """
        Downloads required model files from Hugging Face Hub if they do not exist locally.
        
        Returns:
            Path: The local directory containing the model assets.
        """
        # Replace / with -- for clean folder naming
        safe_name = self.model_name.replace("/", "--")
        target_dir = self.cache_dir / safe_name

        logger.info(f"Checking model files for '{self.model_name}' in {target_dir}")
        
        # Verify if crucial files are already present
        onnx_path = target_dir / "onnx" / "model.onnx"
        config_path = target_dir / "config.json"
        
        if onnx_path.exists() and config_path.exists():
            logger.info(f"Model '{self.model_name}' found locally at {target_dir}")
            self.model_dir = target_dir
            return target_dir

        logger.info(f"Downloading model '{self.model_name}' files from Hugging Face Hub...")
        
        # Ensure target cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download only the necessary files for tokenization and ONNX inference
        downloaded_dir = snapshot_download(
            repo_id=self.model_name,
            local_dir=target_dir,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "special_tokens_map.json",
                "onnx/model.onnx",
                "1_Pooling/config.json",
            ]
        )
        
        self.model_dir = Path(downloaded_dir)
        logger.info(f"Model downloaded and cached successfully at {self.model_dir}")
        return self.model_dir

    def get_onnx_session(self, providers: Optional[List[str]] = None) -> ort.InferenceSession:
        """
        Loads and returns the ONNX inference session.
        
        Args:
            providers: Execution providers for ONNX Runtime (default: ["CPUExecutionProvider"]).
            
        Returns:
            ort.InferenceSession: Loaded ONNX runtime session.
        """
        if not self.model_dir:
            self.download_model()

        assert self.model_dir is not None
        onnx_path = self.model_dir / "onnx" / "model.onnx"
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")

        logger.info(f"Loading ONNX session from {onnx_path}...")
        if providers is None:
            providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(str(onnx_path), providers=providers)
