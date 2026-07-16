from __future__ import annotations
import time
import logging

import cv2
import numpy as np
import insightface
from deepface import DeepFace

from src.config import settings
from src.models import (
    AgeGroup, BBox, Emotion, Gender,
    FaceDetection, FaceAttributes, AnalysisResult, AnalysisStats,
)

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


class AnalysisError(Exception):
    pass


def create_face_app(providers: list[str] | None = None) -> insightface.app.FaceAnalysis:
    app = insightface.app.FaceAnalysis(providers=providers)
    app.prepare(ctx_id=settings.insightface_ctx_id, det_size=settings.insightface_det_size)
    return app


class AgeGenderAnalyzer:
    def __init__(self, face_app: insightface.app.FaceAnalysis | None = None):
        self.face_app = face_app
        self.model_loaded = face_app is not None

    @classmethod
    def from_defaults(cls, enable_gpu: bool | None = None) -> AgeGenderAnalyzer:
        gpu = settings.enable_gpu if enable_gpu is None else enable_gpu
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        face_app = create_face_app(providers)
        return cls(face_app)

    def detect_faces(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
        min_face_size: int | None = None,
    ) -> list[FaceDetection]:
        if not self.model_loaded or self.face_app is None:
            raise ModelLoadError("Face analysis model not loaded")

        threshold = confidence_threshold if confidence_threshold is not None else settings.detection_confidence
        min_size = min_face_size if min_face_size is not None else settings.min_face_size

        try:
            t0 = time.time()
            faces = self.face_app.get(image)
            elapsed = time.time() - t0
        except Exception as e:
            logger.error(f"InsightFace detection error: {e}")
            raise AnalysisError(f"Face detection failed: {e}") from e

        results = []
        for face in faces:
            if face.det_score < threshold:
                continue
            bbox_arr = face.bbox.astype(int)
            bbox = BBox(x1=int(bbox_arr[0]), y1=int(bbox_arr[1]), x2=int(bbox_arr[2]), y2=int(bbox_arr[3]))
            if bbox.width < min_size or bbox.height < min_size:
                continue
            results.append(FaceDetection(
                bbox=bbox,
                age=int(face.age),
                gender=Gender.MALE if face.gender == 1 else Gender.FEMALE,
                detection_score=float(face.det_score),
                processing_time=elapsed,
            ))

        return results

    def analyze_face(self, image: np.ndarray, bbox: BBox) -> FaceAttributes:
        x1, y1, x2, y2 = bbox.to_tuple()
        pad = 20
        h, w = image.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.shape[0] < 48 or face_crop.shape[1] < 48:
            face_crop = cv2.resize(face_crop, (48, 48))

        try:
            t0 = time.time()
            analysis = DeepFace.analyze(
                face_crop,
                actions=settings.deepface_actions,
                enforce_detection=settings.deepface_enforce_detection,
                detector_backend=settings.deepface_detector_backend,
                align=settings.deepface_align,
                silent=True,
            )
            elapsed = time.time() - t0

            if isinstance(analysis, list):
                analysis = analysis[0]

            raw_gender = analysis.get("dominant_gender", "Unknown")
            gender = Gender.MALE if raw_gender.lower() == "male" else Gender.FEMALE if raw_gender.lower() == "female" else Gender.UNKNOWN
            gender_conf = analysis.get("gender", {}).get(analysis.get("dominant_gender", ""), 50) / 100.0
            raw_emotion = analysis.get("dominant_emotion", "unknown")
            emotion = Emotion.UNKNOWN
            for e in Emotion:
                if e.name.lower() == str(raw_emotion).lower():
                    emotion = e
                    break
            emotion_scores = analysis.get("emotion", {})
            age = analysis.get("age")

            return FaceAttributes(
                age=int(age) if age is not None else None,
                gender=gender,
                gender_confidence=gender_conf,
                emotion=emotion,
                emotion_scores=emotion_scores,
                processing_time=elapsed,
            )

        except Exception as e:
            logger.warning(f"DeepFace analysis error: {e}")
            return FaceAttributes(
                age=None,
                gender=Gender.UNKNOWN,
                gender_confidence=0.0,
                emotion=Emotion.UNKNOWN,
                emotion_scores={},
                processing_time=0.0,
            )

    @staticmethod
    def classify_age_group(age: int) -> AgeGroup:
        return AgeGroup.from_age(age)

    def analyze_image(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
        min_face_size: int | None = None,
    ) -> list[AnalysisResult]:
        if not self.model_loaded:
            raise ModelLoadError("Models not loaded")

        t_start = time.time()

        detections = self.detect_faces(image, confidence_threshold, min_face_size)
        if not detections:
            return []

        results = []
        for det in detections:
            attrs = self.analyze_face(image, det.bbox)

            final_age = attrs.age if attrs.age is not None else det.age
            final_gender = attrs.gender if attrs.gender != Gender.UNKNOWN else det.gender

            result = AnalysisResult(
                bbox=det.bbox,
                detection_score=det.detection_score,
                age_insightface=det.age,
                age_deepface=attrs.age,
                age_final=final_age,
                age_group=self.classify_age_group(final_age),
                gender_insightface=det.gender,
                gender_deepface=attrs.gender,
                gender_confidence=attrs.gender_confidence,
                gender_final=final_gender,
                emotion=attrs.emotion,
                emotion_scores=attrs.emotion_scores,
                insightface_time=det.processing_time,
                deepface_time=attrs.processing_time,
            )
            results.append(result)

        if settings.enable_performance_monitoring:
            logger.info(f"Analysis: {time.time() - t_start:.2f}s for {len(results)} faces")

        return results


def calculate_statistics(results: list[AnalysisResult]) -> AnalysisStats:
    if not results:
        return AnalysisStats(total_faces=0, avg_age=0, min_age=0, max_age=0, male_count=0, female_count=0)

    ages = [r.age_final for r in results]
    return AnalysisStats(
        total_faces=len(results),
        avg_age=float(np.mean(ages)),
        min_age=min(ages),
        max_age=max(ages),
        male_count=sum(1 for r in results if r.gender_final == Gender.MALE),
        female_count=sum(1 for r in results if r.gender_final == Gender.FEMALE),
        age_groups={str(g): sum(1 for r in results if r.age_group == g) for g in AgeGroup if g != AgeGroup.UNKNOWN},
    )


def format_processing_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"
