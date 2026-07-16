from unittest.mock import patch, MagicMock

import numpy as np

from src.analyzer import AgeGenderAnalyzer, calculate_statistics, format_processing_time
from src.models import Gender, AgeGroup, AnalysisResult, BBox, Emotion
from tests.conftest import _make_result


class TestAgeGenderAnalyzer:
    def test_init_no_face_app(self) -> None:
        analyzer = AgeGenderAnalyzer(face_app=None)
        assert not analyzer.model_loaded

    def test_from_defaults_failure(self) -> None:
        with patch("src.analyzer.insightface.app.FaceAnalysis", side_effect=ImportError("no insightface")):
            analyzer = AgeGenderAnalyzer(face_app=None)
            assert not analyzer.model_loaded

    def test_detect_faces_raises_if_not_loaded(self) -> None:
        analyzer = AgeGenderAnalyzer(face_app=None)
        try:
            analyzer.detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))
            assert False, "Should have raised"
        except Exception:
            pass

    def test_analyze_image_raises_if_not_loaded(self) -> None:
        analyzer = AgeGenderAnalyzer(face_app=None)
        try:
            analyzer.analyze_image(np.zeros((100, 100, 3), dtype=np.uint8))
            assert False, "Should have raised"
        except Exception:
            pass

    def test_classify_age_group(self) -> None:
        analyzer = AgeGenderAnalyzer(face_app=None)
        assert analyzer.classify_age_group(10) == AgeGroup.CHILD
        assert analyzer.classify_age_group(16) == AgeGroup.TEENAGER
        assert analyzer.classify_age_group(25) == AgeGroup.YOUNG_ADULT
        assert analyzer.classify_age_group(40) == AgeGroup.ADULT
        assert analyzer.classify_age_group(70) == AgeGroup.SENIOR


class TestCalculateStatistics:
    def test_empty(self) -> None:
        stats = calculate_statistics([])
        assert stats.total_faces == 0
        assert stats.avg_age == 0

    def test_with_results(self, sample_results: list[AnalysisResult]) -> None:
        stats = calculate_statistics(sample_results)
        assert stats.total_faces == 3
        assert stats.avg_age == 45.0
        assert stats.min_age == 25
        assert stats.max_age == 70
        assert stats.male_count == 2
        assert stats.female_count == 1


class TestFormatProcessingTime:
    def test_milliseconds(self) -> None:
        assert format_processing_time(0.5) == "500ms"

    def test_seconds(self) -> None:
        assert format_processing_time(1.5) == "1.50s"

    def test_edge_case(self) -> None:
        assert format_processing_time(1.0) == "1.00s"
