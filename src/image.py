from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.config import settings
from src.models import AnalysisResult, Gender


def validate_image(image: Image.Image) -> tuple[bool, str]:
    w, h = image.size
    if w < 50 or h < 50:
        return False, "Image too small. Minimum size: 50x50 pixels"
    if w > 5000 or h > 5000:
        return False, "Image too large. Maximum size: 5000x5000 pixels"
    if image.mode not in ("RGB", "RGBA", "L"):
        return False, "Invalid image format"
    return True, "Valid image"


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    if len(arr.shape) == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return arr
    if len(arr.shape) == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr


def resize_for_display(image: Image.Image | np.ndarray, max_size: tuple[int, int] | None = None) -> Image.Image | np.ndarray:
    size = max_size or settings.max_image_size
    max_w, max_h = size

    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return image

    new_w, new_h = int(w * scale), int(h * scale)
    if isinstance(image, np.ndarray):
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def draw_analysis_results(
    image: np.ndarray,
    results: list[AnalysisResult],
    show_dual: bool = True,
    show_emotion: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    annotated = image.copy()

    for i, result in enumerate(results):
        bbox = result.bbox
        color = settings.colors["male"] if result.gender_final == Gender.MALE else settings.colors["female"] if result.gender_final == Gender.FEMALE else settings.colors["face_box"]

        cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
        cv2.circle(annotated, (bbox.x1 + 15, bbox.y1 + 15), 12, color, -1)
        cv2.putText(annotated, str(i + 1), (bbox.x1 + 10, bbox.y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        labels = []
        if show_dual and result.age_deepface is not None:
            labels.append(f"Age: {result.age_insightface} | {result.age_deepface}")
            labels.append(f"Gender: {result.gender_insightface} | {result.gender_deepface}")
        else:
            labels.append(f"Age: {result.age_final} ({result.age_group})")
            labels.append(f"Gender: {result.gender_final}")

        if show_emotion and result.emotion.name != "UNKNOWN":
            labels.append(f"Emotion: {result.emotion}")

        if show_confidence:
            labels.append(f"Detection: {result.detection_score:.2%}")
            if result.gender_confidence > 0:
                labels.append(f"Gender Conf: {result.gender_confidence:.1%}")

        y_offset = 0
        for label in labels:
            text_y = bbox.y2 + 20 + y_offset
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated,
                          (bbox.x1, text_y - th - 5),
                          (bbox.x1 + tw + 10, text_y + baseline + 5),
                          settings.colors["text_bg"], -1)
            cv2.putText(annotated, label, (bbox.x1 + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += th + 10

    return annotated
