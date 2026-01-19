# -*- coding: utf-8 -*-
"""
UI 인터페이스 모듈

[리팩토링 2단계] 함수/모듈화
- UI에서 import 가능한 형태로 제공
- 간단한 함수 인터페이스 제공
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data_collection.pipeline import DataCollectionPipeline
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact

logger = logging.getLogger(__name__)


def get_universe(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    유니버스 데이터 조회 (UI용)

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame: date, ym, ticker 컬럼 포함
    """
    pipeline = DataCollectionPipeline(config_path=config_path, force_rebuild=force_rebuild)
    return pipeline.run_l0()


def get_ohlcv(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    OHLCV 데이터 조회 (UI용)

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame: date, ticker, open, high, low, close, volume, value 및 기술적 지표 컬럼 포함
    """
    pipeline = DataCollectionPipeline(config_path=config_path, force_rebuild=force_rebuild)
    return pipeline.run_l1()


def get_panel(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    패널 데이터 조회 (UI용)

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame: 병합된 패널 데이터
    """
    pipeline = DataCollectionPipeline(config_path=config_path, force_rebuild=force_rebuild)
    return pipeline.run_l3()


def get_dataset(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> Dict[str, any]:
    """
    데이터셋 조회 (UI용)

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        dict: {
            "dataset_daily": DataFrame,
            "cv_folds_short": list,
            "cv_folds_long": list,
        }
    """
    pipeline = DataCollectionPipeline(config_path=config_path, force_rebuild=force_rebuild)
    return pipeline.run_l4()


def check_data_availability(
    config_path: str = "configs/config.yaml",
) -> Dict[str, bool]:
    """
    데이터 가용성 확인 (UI용)

    Args:
        config_path: 설정 파일 경로

    Returns:
        dict: {
            "universe": bool,
            "ohlcv": bool,
            "fundamentals": bool,
            "panel": bool,
            "dataset": bool,
            "cv_short": bool,
            "cv_long": bool,
        }
    """
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))

    return {
        "universe": artifact_exists(interim_dir / "universe_k200_membership_monthly"),
        "ohlcv": artifact_exists(interim_dir / "ohlcv_daily"),
        "fundamentals": artifact_exists(interim_dir / "fundamentals_annual"),
        "panel": artifact_exists(interim_dir / "panel_merged_daily"),
        "dataset": artifact_exists(interim_dir / "dataset_daily"),
        "cv_short": artifact_exists(interim_dir / "cv_folds_short"),
        "cv_long": artifact_exists(interim_dir / "cv_folds_long"),
    }


def collect_data_for_ui(
    config_path: str = "configs/config.yaml",
    stages: Optional[List[str]] = None,
    force_rebuild: bool = False,
) -> Dict[str, any]:
    """
    UI에서 사용할 데이터 수집 (편의 함수)

    Args:
        config_path: 설정 파일 경로
        stages: 실행할 단계 리스트 (None이면 전체 실행)
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        dict: {
            "universe": DataFrame,
            "ohlcv": DataFrame,
            "panel": DataFrame,
            "dataset": DataFrame,
            "cv_folds_short": list,
            "cv_folds_long": list,
            "available": Dict[str, bool],
        }
    """
    from src.data_collection.pipeline import run_data_collection_pipeline

    # 데이터 수집
    result = run_data_collection_pipeline(
        config_path=config_path,
        stages=stages,
        force_rebuild=force_rebuild,
    )

    artifacts = result["artifacts"]

    # 가용성 확인
    available = check_data_availability(config_path=config_path)

    return {
        "universe": artifacts.get("universe_k200_membership_monthly"),
        "ohlcv": artifacts.get("ohlcv_daily"),
        "fundamentals": artifacts.get("fundamentals_annual"),
        "panel": artifacts.get("panel_merged_daily"),
        "dataset": artifacts.get("dataset_daily"),
        "cv_folds_short": artifacts.get("cv_folds_short"),
        "cv_folds_long": artifacts.get("cv_folds_long"),
        "available": available,
        "artifacts_path": result["artifacts_path"],
    }
