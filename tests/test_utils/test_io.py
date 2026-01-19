"""
IO 유틸리티 함수들 단위 테스트
"""
import pandas as pd
import pytest

from src.utils.io import artifact_exists, load_artifact, save_artifact


class TestArtifactExists:
    """artifact_exists 함수 테스트"""

    def test_artifact_exists_with_parquet(self, tmp_path):
        """parquet 파일이 존재하는 경우 True 반환"""
        file_path = tmp_path / "test"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(file_path.with_suffix(".parquet"))

        assert artifact_exists(file_path) is True

    def test_artifact_exists_with_csv(self, tmp_path):
        """csv 파일이 존재하는 경우 True 반환"""
        file_path = tmp_path / "test"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_csv(file_path.with_suffix(".csv"), index=False)

        assert artifact_exists(file_path) is True

    def test_artifact_exists_none_exist(self, tmp_path):
        """파일이 존재하지 않는 경우 False 반환"""
        file_path = tmp_path / "test"
        assert artifact_exists(file_path) is False

    def test_artifact_exists_custom_formats(self, tmp_path):
        """사용자 지정 포맷으로만 존재하는 경우"""
        file_path = tmp_path / "test"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_json(file_path.with_suffix(".json"), orient="records")

        # json 포맷만 지정하면 존재함
        assert artifact_exists(file_path, formats=["json"]) is True
        # parquet/csv만 지정하면 존재하지 않음
        assert artifact_exists(file_path, formats=["parquet", "csv"]) is False


class TestLoadArtifact:
    """load_artifact 함수 테스트"""

    def test_load_parquet_file(self, tmp_path):
        """parquet 파일 로드 테스트"""
        file_path = tmp_path / "test"
        original_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "value": [1.0, 2.0, 3.0]
        })
        original_df.to_parquet(file_path.with_suffix(".parquet"))

        loaded_df = load_artifact(file_path)

        pd.testing.assert_frame_equal(original_df, loaded_df)

    def test_load_csv_file(self, tmp_path):
        """csv 파일 로드 테스트"""
        file_path = tmp_path / "test"
        original_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "value": [1.0, 2.0, 3.0]
        })
        original_df.to_csv(file_path.with_suffix(".csv"), index=False)

        loaded_df = load_artifact(file_path)

        # 날짜 컬럼이 datetime으로 변환되는지 확인
        assert pd.api.types.is_datetime64_any_dtype(loaded_df["date"])
        assert loaded_df["value"].tolist() == [1.0, 2.0, 3.0]

    def test_load_csv_without_date_column(self, tmp_path):
        """date 컬럼이 없는 csv 파일 로드 테스트"""
        file_path = tmp_path / "test"
        original_df = pd.DataFrame({
            "ticker": ["AAPL", "GOOGL"],
            "value": [1.0, 2.0]
        })
        original_df.to_csv(file_path.with_suffix(".csv"), index=False)

        loaded_df = load_artifact(file_path)

        pd.testing.assert_frame_equal(original_df, loaded_df)

    def test_load_artifact_not_found(self, tmp_path):
        """파일이 존재하지 않는 경우 FileNotFoundError 발생"""
        file_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Artifact not found"):
            load_artifact(file_path)

    def test_load_parquet_takes_precedence(self, tmp_path):
        """parquet와 csv가 모두 존재하는 경우 parquet 우선 로드"""
        file_path = tmp_path / "test"

        # parquet 파일 생성
        parquet_df = pd.DataFrame({"source": ["parquet"], "value": [1.0]})
        parquet_df.to_parquet(file_path.with_suffix(".parquet"))

        # csv 파일 생성 (다른 내용)
        csv_df = pd.DataFrame({"source": ["csv"], "value": [2.0]})
        csv_df.to_csv(file_path.with_suffix(".csv"), index=False)

        loaded_df = load_artifact(file_path)

        assert loaded_df["source"].iloc[0] == "parquet"
        assert loaded_df["value"].iloc[0] == 1.0


class TestSaveArtifact:
    """save_artifact 함수 테스트"""

    def test_save_as_parquet(self, tmp_path, sample_data_basic):
        """parquet 형식으로 저장 테스트"""
        file_path = tmp_path / "test"

        save_artifact(sample_data_basic, file_path, formats=["parquet"])

        assert (file_path.with_suffix(".parquet")).exists()
        loaded_df = pd.read_parquet(file_path.with_suffix(".parquet"))
        pd.testing.assert_frame_equal(sample_data_basic, loaded_df)

    def test_save_as_csv(self, tmp_path, sample_data_basic):
        """csv 형식으로 저장 테스트"""
        file_path = tmp_path / "test"

        save_artifact(sample_data_basic, file_path, formats=["csv"])

        assert (file_path.with_suffix(".csv")).exists()
        loaded_df = pd.read_csv(file_path.with_suffix(".csv"))
        # 날짜 타입은 문자열로 저장되므로 비교 시 주의
        assert len(loaded_df) == len(sample_data_basic)
        assert list(loaded_df.columns) == list(sample_data_basic.columns)

    def test_save_multiple_formats(self, tmp_path, sample_data_basic):
        """여러 형식으로 동시에 저장 테스트"""
        file_path = tmp_path / "test"

        save_artifact(sample_data_basic, file_path, formats=["parquet", "csv"])

        assert (file_path.with_suffix(".parquet")).exists()
        assert (file_path.with_suffix(".csv")).exists()

    def test_save_force_overwrite(self, tmp_path, sample_data_basic):
        """force=True로 기존 파일 덮어쓰기 테스트"""
        file_path = tmp_path / "test"

        # 첫 번째 저장
        save_artifact(sample_data_basic, file_path, formats=["parquet"])

        # 두 번째 저장 (force=True)
        modified_df = sample_data_basic.copy()
        modified_df["new_col"] = [1] * len(modified_df)
        save_artifact(modified_df, file_path, formats=["parquet"], force=True)

        loaded_df = pd.read_parquet(file_path.with_suffix(".parquet"))
        assert "new_col" in loaded_df.columns

    def test_save_creates_parent_directories(self, tmp_path, sample_data_basic):
        """부모 디렉토리가 없는 경우 자동 생성 테스트"""
        file_path = tmp_path / "nested" / "deep" / "test"

        save_artifact(sample_data_basic, file_path, formats=["parquet"])

        assert (file_path.with_suffix(".parquet")).exists()