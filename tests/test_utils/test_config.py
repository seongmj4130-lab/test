"""
설정 로딩 유틸리티 함수들 단위 테스트
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils.config import _replace_base_dir, _to_posix, find_repo_root, load_config


class TestFindRepoRoot:
    """find_repo_root 함수 테스트"""

    def test_find_repo_root_from_file(self, tmp_path):
        """파일 경로에서 리포지토리 루트 찾기 테스트"""
        # .git 디렉토리 생성
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # 테스트 파일 생성
        test_file = tmp_path / "subdir" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("print('test')")

        # test_file에서 루트 찾기
        root = find_repo_root(test_file)

        assert root == tmp_path

    def test_find_repo_root_from_pyproject(self, tmp_path):
        """pyproject.toml 파일로 리포지토리 루트 찾기 테스트"""
        # pyproject.toml 파일 생성
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text('[tool.poetry]\nname = "test"')

        # 하위 디렉토리에서 루트 찾기
        sub_dir = tmp_path / "src" / "utils"
        sub_dir.mkdir(parents=True)

        root = find_repo_root(sub_dir)

        assert root == tmp_path

    def test_find_repo_root_not_found(self, tmp_path):
        """리포지토리 루트가 없는 경우 현재 작업 디렉토리 반환"""
        # .git이나 pyproject.toml이 없는 디렉토리
        isolated_dir = tmp_path / "isolated"
        isolated_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=isolated_dir):
            root = find_repo_root()

            assert root == isolated_dir

    def test_find_repo_root_none_start_path(self):
        """start_path가 None인 경우 현재 파일 위치에서 찾기"""
        # 이 테스트는 실제 파일 시스템에 의존하므로 mock 사용
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/fake/path")

            # 실제로는 파일 시스템에 따라 다르지만 호출 자체는 성공해야 함
            root = find_repo_root(None)

            # mock_cwd가 호출되었는지 확인
            mock_cwd.assert_called_once()


class TestToPosix:
    """_to_posix 함수 테스트"""

    def test_to_posix_windows_path(self):
        """Windows 경로를 POSIX 형식으로 변환"""
        windows_path = r"C:\Users\test\file.txt"
        posix_path = _to_posix(windows_path)

        assert posix_path == "C:/Users/test/file.txt"

    def test_to_posix_posix_path(self):
        """POSIX 경로 그대로 반환"""
        posix_path = "/home/user/file.txt"
        result = _to_posix(posix_path)

        assert result == posix_path

    def test_to_posix_pathlib_path(self):
        """Path 객체를 POSIX 형식으로 변환"""
        path_obj = Path(r"C:\Users\test\file.txt")
        result = _to_posix(path_obj)

        assert result == "C:/Users/test/file.txt"


class TestReplaceBaseDir:
    """_replace_base_dir 함수 테스트"""

    def test_replace_base_dir_in_string(self):
        """문자열에서 {base_dir} 플레이스홀더 교체"""
        base_dir = "/project/root"
        input_str = "{base_dir}/data/file.csv"

        result = _replace_base_dir(input_str, base_dir)

        assert result == "/project/root/data/file.csv"

    def test_replace_base_dir_in_dict(self):
        """딕셔너리에서 {base_dir} 플레이스홀더 교체"""
        base_dir = "/project/root"
        input_dict = {
            "data_path": "{base_dir}/data",
            "config_path": "{base_dir}/config",
            "nested": {"artifact_path": "{base_dir}/artifacts"},
        }

        result = _replace_base_dir(input_dict, base_dir)

        assert result["data_path"] == "/project/root/data"
        assert result["config_path"] == "/project/root/config"
        assert result["nested"]["artifact_path"] == "/project/root/artifacts"

    def test_replace_base_dir_in_list(self):
        """리스트에서 {base_dir} 플레이스홀더 교체"""
        base_dir = "/project/root"
        input_list = ["{base_dir}/file1.txt", "{base_dir}/file2.txt"]

        result = _replace_base_dir(input_list, base_dir)

        assert result == ["/project/root/file1.txt", "/project/root/file2.txt"]

    def test_replace_base_dir_mixed_types(self):
        """혼합 타입에서 {base_dir} 플레이스홀더 교체"""
        base_dir = "/project/root"
        input_data = {
            "paths": ["{base_dir}/data", "{base_dir}/config"],
            "settings": {
                "log_path": "{base_dir}/logs",
                "count": 42,  # 숫자는 변경되지 않아야 함
            },
        }

        result = _replace_base_dir(input_data, base_dir)

        assert result["paths"] == ["/project/root/data", "/project/root/config"]
        assert result["settings"]["log_path"] == "/project/root/logs"
        assert result["settings"]["count"] == 42

    def test_replace_base_dir_no_placeholder(self):
        """플레이스홀더가 없는 경우 원본 반환"""
        base_dir = "/project/root"
        input_data = {"normal_path": "/absolute/path", "relative_path": "relative/path"}

        result = _replace_base_dir(input_data, base_dir)

        assert result == input_data


class TestLoadConfig:
    """load_config 함수 테스트"""

    def test_load_config_file_not_found(self):
        """설정 파일이 존재하지 않는 경우 FileNotFoundError 발생"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_config_basic(self, tmp_path, sample_config_minimal):
        """기본 설정 파일 로드 테스트"""
        config_file = tmp_path / "config.yaml"

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(sample_config_minimal, f)

        loaded_config = load_config(config_file)

        assert "paths" in loaded_config
        assert "base_dir" in loaded_config["paths"]

    def test_load_config_with_base_dir_replacement(self, tmp_path):
        """base_dir 플레이스홀더 교체 테스트"""
        config_data = {
            "paths": {
                "base_dir": "/custom/base",
                "data_dir": "{base_dir}/data",
                "artifacts_dir": "{base_dir}/artifacts",
            },
            "other": {"log_path": "{base_dir}/logs"},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)

        loaded_config = load_config(config_file)

        assert loaded_config["paths"]["data_dir"] == "/custom/base/data"
        assert loaded_config["paths"]["artifacts_dir"] == "/custom/base/artifacts"
        assert loaded_config["other"]["log_path"] == "/custom/base/logs"

    def test_load_config_missing_base_dir(self, tmp_path):
        """base_dir이 없는 경우 RuntimeError 발생"""
        config_data = {
            "paths": {
                # base_dir이 없음
                "data_dir": "{base_dir}/data"
            }
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)

        with pytest.raises(RuntimeError, match="base_dir.*not found"):
            load_config(config_file)

    @patch.dict(os.environ, {"BASE_DIR": "/env/base"})
    def test_load_config_env_fallback(self, tmp_path):
        """환경 변수에서 base_dir 폴백 테스트"""
        config_data = {
            "paths": {
                # base_dir이 설정에 없음
                "data_dir": "{base_dir}/data"
            }
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)

        loaded_config = load_config(config_file)

        assert loaded_config["paths"]["data_dir"] == "/env/base/data"
