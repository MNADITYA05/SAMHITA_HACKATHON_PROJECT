from src.models import AgeGroup, BBox, Gender, Emotion


class TestAgeGroup:
    def test_from_age_child(self) -> None:
        assert AgeGroup.from_age(5) == AgeGroup.CHILD

    def test_from_age_teen(self) -> None:
        assert AgeGroup.from_age(15) == AgeGroup.TEENAGER

    def test_from_age_young_adult(self) -> None:
        assert AgeGroup.from_age(25) == AgeGroup.YOUNG_ADULT

    def test_from_age_adult(self) -> None:
        assert AgeGroup.from_age(40) == AgeGroup.ADULT

    def test_from_age_senior(self) -> None:
        assert AgeGroup.from_age(65) == AgeGroup.SENIOR

    def test_from_age_out_of_range(self) -> None:
        assert AgeGroup.from_age(999) == AgeGroup.UNKNOWN

    def test_string_representation(self) -> None:
        assert "Young Adult" in str(AgeGroup.from_age(25))
        assert "Child" in str(AgeGroup.CHILD)


class TestBBox:
    def test_properties(self) -> None:
        b = BBox(10, 20, 110, 70)
        assert b.width == 100
        assert b.height == 50
        assert b.area == 5000

    def test_to_tuple(self) -> None:
        b = BBox(1, 2, 3, 4)
        assert b.to_tuple() == (1, 2, 3, 4)


class TestEnums:
    def test_gender_str(self) -> None:
        assert str(Gender.MALE) == "Male"
        assert str(Gender.FEMALE) == "Female"

    def test_emotion_str(self) -> None:
        assert str(Emotion.HAPPY) == "Happy"
