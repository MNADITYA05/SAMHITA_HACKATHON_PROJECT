from __future__ import annotations
from dataclasses import dataclass, field
import os


@dataclass
class Settings:
    # Paths
    temp_dir: str = "temp"
    models_dir: str = "models"

    # Detection
    detection_confidence: float = 0.5
    min_face_size: int = 30

    # Display
    max_image_size: tuple[int, int] = (800, 600)
    supported_formats: list[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp", "tiff"])

    # Performance
    resize_factor: float = 1.0
    enable_gpu: bool = True
    enable_performance_monitoring: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"

    # GPU
    insightface_ctx_id: int = 0  # -1 for CPU
    insightface_det_size: tuple[int, int] = (640, 640)

    # DeepFace
    deepface_actions: list[str] = field(default_factory=lambda: ["age", "gender", "emotion"])
    deepface_enforce_detection: bool = False
    deepface_detector_backend: str = "opencv"
    deepface_align: bool = True

    # Colors in BGR
    colors: dict = field(default_factory=lambda: {
        "face_box": (0, 255, 0),
        "text_bg": (0, 0, 0),
        "text": (0, 255, 0),
        "male": (255, 0, 0),
        "female": (255, 0, 255),
    })

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            enable_gpu=os.environ.get("ENABLE_GPU", "true").lower() == "true",
            detection_confidence=float(os.environ.get("DETECTION_CONFIDENCE", "0.5")),
            min_face_size=int(os.environ.get("MIN_FACE_SIZE", "30")),
            debug_mode=os.environ.get("DEBUG_MODE", "false").lower() == "true",
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


settings = Settings.from_env()
