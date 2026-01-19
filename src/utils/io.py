# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/utils/io.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def artifact_exists(out_base: Path, formats: Optional[List[str]] = None) -> bool:
    formats = formats or ["parquet", "csv"]
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        if p.exists():
            return True
    return False

def load_artifact(out_base: Path) -> pd.DataFrame:
    """
    out_base = .../data/interim/<name>
    우선순위: parquet -> csv
    """
    p_parq = out_base.with_suffix(".parquet")
    p_csv = out_base.with_suffix(".csv")

    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    raise FileNotFoundError(f"Artifact not found: {p_parq} or {p_csv}")

def save_artifact(
    df: pd.DataFrame,
    out_base: Path,
    *,
    formats: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    out_base = .../data/interim/<name> or .../data/interim/{run_tag}/<name>
    formats: ["parquet","csv"]

    [Stage0] 태그 폴더 구조 강제: out_base의 부모 디렉토리가 반드시 생성됨

    [공통 프롬프트 v2] force-rebuild 모드:
    - force=True면 기존 파일이 있어도 무조건 덮어쓰기 (skip_if_exists 무시)
    - L2 산출물(fundamentals_annual.parquet)은 이 함수를 통해 저장되지 않음 (재사용 고정)
    """
    formats = formats or ["parquet", "csv"]
    # [Stage0] 부모 디렉토리 강제 생성 (태그 폴더 구조 보장)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # [Stage0] 디버그: 실제 저장 경로 확인
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"[save_artifact] Saving to: {out_base}")

    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        # [공통 프롬프트 v2] force=True면 기존 파일이 있어도 무조건 덮어쓰기
        if out_path.exists() and not force:
            continue

        if fmt == "parquet":
            df.to_parquet(out_path, index=False)
        elif fmt == "csv":
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
