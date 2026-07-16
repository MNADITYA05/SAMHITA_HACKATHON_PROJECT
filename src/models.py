from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class Gender(Enum):
    MALE = auto()
    FEMALE = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name.title()


class AgeGroup(Enum):
    CHILD = (0, 12)
    TEENAGER = (13, 19)
    YOUNG_ADULT = (20, 34)
    ADULT = (35, 54)
    SENIOR = (55, 100)
    UNKNOWN = (0, 0)

    @classmethod
    def from_age(cls, age: int) -> AgeGroup:
        for group in cls:
            if group == cls.UNKNOWN:
                continue
            lo, hi = group.value
            if lo <= age <= hi:
                return group
        return cls.UNKNOWN

    def __str__(self) -> str:
        return self.name.replace('_', ' ').title()


class Emotion(Enum):
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    FEAR = auto()
    SURPRISE = auto()
    DISGUST = auto()
    NEUTRAL = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name.title()


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class FaceDetection:
    bbox: BBox
    age: int
    gender: Gender
    detection_score: float
    processing_time: float


@dataclass
class FaceAttributes:
    age: int | None
    gender: Gender
    gender_confidence: float
    emotion: Emotion
    emotion_scores: dict[str, float]
    processing_time: float


@dataclass
class AnalysisResult:
    bbox: BBox
    detection_score: float
    age_insightface: int
    age_deepface: int | None
    age_final: int
    age_group: AgeGroup
    gender_insightface: Gender
    gender_deepface: Gender | None
    gender_confidence: float
    gender_final: Gender
    emotion: Emotion
    emotion_scores: dict[str, float]
    insightface_time: float
    deepface_time: float


@dataclass
class AnalysisStats:
    total_faces: int
    avg_age: float
    min_age: int
    max_age: int
    male_count: int
    female_count: int
    age_groups: dict[str, int] = field(default_factory=dict)
