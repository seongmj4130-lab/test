from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
