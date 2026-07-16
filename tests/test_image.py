import numpy as np
from PIL import Image

from src.image import validate_image, image_to_rgb_array, resize_for_display, draw_analysis_results
from tests.conftest import _make_result


class TestValidateImage:
    def test_valid_rgb(self, pil_image: Image.Image) -> None:
        valid, msg = validate_image(pil_image)
        assert valid

    def test_too_small(self) -> None:
        img = Image.new("RGB", (10, 10))
        valid, msg = validate_image(img)
        assert not valid
        assert "too small" in msg

    def test_too_large(self) -> None:
        img = Image.new("RGB", (6000, 6000))
        valid, msg = validate_image(img)
        assert not valid
        assert "too large" in msg

    def test_invalid_mode(self) -> None:
        img = Image.new("CMYK", (100, 100))
        valid, msg = validate_image(img)
        assert not valid
        assert "Invalid" in msg


class TestImageToRgbArray:
    def test_rgb_passthrough(self) -> None:
        img = Image.new("RGB", (10, 10))
        arr = image_to_rgb_array(img)
        assert arr.shape == (10, 10, 3)

    def test_rgba_conversion(self) -> None:
        img = Image.new("RGBA", (10, 10), (128, 128, 128, 255))
        arr = image_to_rgb_array(img)
        assert arr.shape == (10, 10, 3)

    def test_grayscale_conversion(self) -> None:
        img = Image.new("L", (10, 10), 128)
        arr = image_to_rgb_array(img)
        assert arr.shape == (10, 10, 3)


class TestResizeForDisplay:
    def test_smaller_than_max(self, pil_image: Image.Image) -> None:
        result = resize_for_display(pil_image, max_size=(800, 600))
        assert result.size == pil_image.size

    def test_resize_down(self) -> None:
        img = Image.new("RGB", (1600, 1200))
        result = resize_for_display(img, max_size=(800, 600))
        w, h = result.size
        assert w <= 800
        assert h <= 600


class TestDrawAnalysisResults:
    def test_draw_returns_same_shape(self, rgb_image: np.ndarray) -> None:
        results = [_make_result()]
        annotated = draw_analysis_results(rgb_image, results)
        assert annotated.shape == rgb_image.shape

    def test_draw_empty_results(self, rgb_image: np.ndarray) -> None:
        annotated = draw_analysis_results(rgb_image, [])
        assert annotated.shape == rgb_image.shape
        assert np.array_equal(annotated, rgb_image)
