from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def find_repo_root(start_path: Path = None) -> Path:
    """Find repository root by looking for .git directory or pyproject.toml."""
    if start_path is None:
        start_path = Path(__file__).resolve()

    current = start_path
    while current.parent != current:  # Stop at filesystem root
        if (current / ".git").is_dir() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback to current working directory if repo root not found
    return Path.cwd()


def _to_posix(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def _replace_base_dir(value: Any, base_dir_posix: str) -> Any:
    """Replace {base_dir} placeholder recursively."""
    if isinstance(value, str):
        return value.replace("{base_dir}", base_dir_posix)
    if isinstance(value, dict):
        return {k: _replace_base_dir(v, base_dir_posix) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_base_dir(v, base_dir_posix) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.get("paths", {})
    base_dir = paths.get("base_dir")
    if not base_dir:
        raise KeyError("configs/config.yaml must define: paths.base_dir")

    # [경로 고정] base_dir 깨진 문자열 검증 및 제거
    base_dir_str = str(base_dir).strip()
    if "???" in base_dir_str or "??" in base_dir_str or not base_dir_str:
        expected_base_dir = os.getenv(
            "BASE_DIR", "C:/Users/seong/OneDrive/Desktop/bootcamp/000_code"
        )
        raise ValueError(
            f"[FATAL] base_dir contains corrupted characters: {base_dir_str}\n"
            f"Fix configs/config.yaml paths.base_dir to: {expected_base_dir}"
        )

    base_dir_posix = _to_posix(base_dir)
    cfg = _replace_base_dir(cfg, base_dir_posix)

    # Convert common path fields to Path objects (optional but useful)
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        if isinstance(v, str) and ("/" in v or "\\" in v):
            paths[k] = Path(v)
    cfg["paths"] = paths

    # [경로 고정] 런타임 강제 검증: base_dir이 정확한 경로인지 확인
    expected_base_dir = os.getenv("BASE_DIR", str(find_repo_root()))
    EXPECTED = Path(expected_base_dir).resolve()
    ACTUAL = Path(paths["base_dir"]).resolve()

    if ACTUAL != EXPECTED:
        raise RuntimeError(
            f"[FATAL] base_dir mismatch.\n"
            f"  expected: {EXPECTED}\n"
            f"  actual  : {ACTUAL}\n"
            f"Fix configs/config.yaml paths.base_dir to: {expected_base_dir}"
        )

    # [경로 고정] 런타임 로그 출력
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[RUNTIME] cwd={Path.cwd().resolve()}")
    logger.info(f"[RUNTIME] base_dir={ACTUAL}")

    return cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"Missing paths.{key} in config")
    v = paths[key]
    return v if isinstance(v, Path) else Path(str(v))


# 기본값 정의
DEFAULT_CONFIG = {
    "params": {
        "start_date": "2016-01-01",
        "end_date": "2024-12-31"
    },
    "run": {
        "fail_on_validation_error": True,
        "save_formats": ["parquet", "csv"],
        "skip_if_exists": False,
        "timezone": "Asia/Seoul",
        "write_meta": True
    },
    "l7": {
        "cost_bps": 10.0,
        "holding_days": 20,
        "top_k": 12,
        "rebalance_interval": 1,
        "target_volatility": 0.15
    }
}


def load_yaml(path: Union[str, Path], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    YAML 파일을 로드하고 기본값을 병합합니다.

    Args:
        path: YAML 파일 경로
        defaults: 기본값 딕셔너리 (옵션)

    Returns:
        병합된 설정 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        yaml.YAMLError: YAML 파싱 오류
        ValueError: 유효하지 않은 기본값 키
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 파싱 오류 ({path}): {e}")

    # 기본값 병합
    if defaults:
        config = merge_defaults(config, defaults)

    # 환경변수 치환
    config = substitute_env_vars(config)

    return config


def merge_defaults(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정에 기본값을 재귀적으로 병합합니다.

    Args:
        config: 사용자 설정
        defaults: 기본값

    Returns:
        병합된 설정
    """
    merged = defaults.copy()

    def _merge_dict(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    target[key] = _merge_dict(target[key], value)
                else:
                    target[key] = value
            else:
                target[key] = value
        return target

    return _merge_dict(merged, config)


def substitute_env_vars(config: Any) -> Any:
    """
    설정에서 환경변수 플레이스홀더를 치환합니다.
    ${VAR_NAME} 또는 $VAR_NAME 형식을 지원합니다.

    Args:
        config: 설정 딕셔너리 (재귀적 처리)

    Returns:
        환경변수가 치환된 설정
    """
    if isinstance(config, str):
        import re
        # ${VAR} 또는 $VAR 패턴 찾기
        pattern = re.compile(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)')

        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"환경변수가 설정되지 않음: {var_name}")
            return value

        return pattern.sub(replace_var, config)

    elif isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    else:
        return config


def load_yaml_with_defaults(path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAML 파일을 로드하고 글로벌 기본값을 병합합니다.

    Args:
        path: YAML 파일 경로

    Returns:
        기본값이 병합된 설정
    """
    return load_yaml(path, DEFAULT_CONFIG)
