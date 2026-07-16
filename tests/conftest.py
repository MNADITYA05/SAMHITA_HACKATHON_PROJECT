import numpy as np
import pytest
from PIL import Image

from src.models import AnalysisResult, BBox, Gender, AgeGroup, Emotion


def _make_result(age: int = 30, gender: Gender = Gender.MALE) -> AnalysisResult:
    return AnalysisResult(
        bbox=BBox(10, 10, 100, 120),
        detection_score=0.95,
        age_insightface=age,
        age_deepface=age,
        age_final=age,
        age_group=AgeGroup.from_age(age),
        gender_insightface=gender,
        gender_deepface=gender,
        gender_confidence=0.98,
        gender_final=gender,
        emotion=Emotion.HAPPY,
        emotion_scores={"happy": 0.9, "neutral": 0.1},
        insightface_time=0.05,
        deepface_time=0.3,
    )


@pytest.fixture
def sample_results() -> list[AnalysisResult]:
    return [
        _make_result(age=25, gender=Gender.FEMALE),
        _make_result(age=40, gender=Gender.MALE),
        _make_result(age=70, gender=Gender.MALE),
    ]


@pytest.fixture
def rgb_image() -> np.ndarray:
    return np.ones((200, 200, 3), dtype=np.uint8) * 128


@pytest.fixture
def pil_image() -> Image.Image:
    return Image.new("RGB", (200, 200), color="gray")
