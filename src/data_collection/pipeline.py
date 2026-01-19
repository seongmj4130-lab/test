# -*- coding: utf-8 -*-
"""
데이터 수집 파이프라인 모듈

[리팩토링 3단계] 파이프라인 재조립
- 재현성 보장
- 실행 간편화
- 단계별 실행 가능
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data_collection.collectors import (
    collect_dataset,
    collect_fundamentals,
    collect_ohlcv,
    collect_panel,
    collect_universe,
)
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact

logger = logging.getLogger(__name__)


class DataCollectionPipeline:
    """
    데이터 수집 파이프라인 클래스

    재현성과 실행 간편화를 위한 파이프라인 래퍼
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        force_rebuild: bool = False,
    ):
        """
        Args:
            config_path: 설정 파일 경로
            force_rebuild: True면 캐시 무시하고 재생성
        """
        self.config_path = config_path
        self.force_rebuild = force_rebuild
        self.cfg = load_config(config_path)
        self.params = self.cfg.get("params", {})
        self.interim_dir = Path(get_path(self.cfg, "data_interim"))
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        # 아티팩트 저장소
        self.artifacts: Dict[str, any] = {}
        self.artifacts_path: Dict[str, str] = {}

    def run_l0(self) -> pd.DataFrame:
        """L0: 유니버스 구성"""
        logger.info("[Pipeline L0] 유니버스 구성")

        cache_path = self.interim_dir / "universe_k200_membership_monthly"

        if artifact_exists(cache_path) and not self.force_rebuild:
            self.artifacts["universe_k200_membership_monthly"] = load_artifact(cache_path)
            self.artifacts_path["universe"] = str(cache_path)
            logger.info(f"  ✓ 캐시에서 로드: {len(self.artifacts['universe_k200_membership_monthly']):,}행")
            return self.artifacts["universe_k200_membership_monthly"]

        df = collect_universe(
            start_date=self.params.get("start_date", "2016-01-01"),
            end_date=self.params.get("end_date", "2024-12-31"),
            index_code=self.params.get("index_code", "1028"),
            anchor_ticker=self.params.get("anchor_ticker", "005930"),
            config_path=self.config_path,
            save_to_cache=True,
            force_rebuild=self.force_rebuild,
        )

        self.artifacts["universe_k200_membership_monthly"] = df
        self.artifacts_path["universe"] = str(cache_path)
        return df

    def run_l1(self) -> pd.DataFrame:
        """L1: OHLCV 다운로드"""
        logger.info("[Pipeline L1] OHLCV 다운로드")

        # L0 의존성 확인
        if "universe_k200_membership_monthly" not in self.artifacts:
            logger.info("  → L0 먼저 실행")
            self.run_l0()

        cache_path = self.interim_dir / "ohlcv_daily"

        if artifact_exists(cache_path) and not self.force_rebuild:
            self.artifacts["ohlcv_daily"] = load_artifact(cache_path)
            self.artifacts_path["ohlcv"] = str(cache_path)
            logger.info(f"  ✓ 캐시에서 로드: {len(self.artifacts['ohlcv_daily']):,}행")
            return self.artifacts["ohlcv_daily"]

        df = collect_ohlcv(
            universe=self.artifacts["universe_k200_membership_monthly"],
            start_date=self.params.get("start_date", "2016-01-01"),
            end_date=self.params.get("end_date", "2024-12-31"),
            calculate_technical_features=True,
            config_path=self.config_path,
            save_to_cache=True,
            force_rebuild=self.force_rebuild,
        )

        self.artifacts["ohlcv_daily"] = df
        self.artifacts_path["ohlcv"] = str(cache_path)
        return df

    def run_l2(self) -> Optional[pd.DataFrame]:
        """L2: 재무 데이터 로드"""
        logger.info("[Pipeline L2] 재무 데이터 로드")

        cache_path = self.interim_dir / "fundamentals_annual"

        if artifact_exists(cache_path) and not self.force_rebuild:
            self.artifacts["fundamentals_annual"] = load_artifact(cache_path)
            self.artifacts_path["fundamentals"] = str(cache_path)
            logger.info(f"  ✓ 캐시에서 로드: {len(self.artifacts['fundamentals_annual']):,}행")
            return self.artifacts["fundamentals_annual"]

        df = collect_fundamentals(
            config_path=self.config_path,
            save_to_cache=True,
            force_rebuild=self.force_rebuild,
        )

        if df is not None:
            self.artifacts["fundamentals_annual"] = df
            self.artifacts_path["fundamentals"] = str(cache_path)
        else:
            logger.warning("  ⚠ 재무 데이터가 없습니다. L3에서 재무 데이터 없이 진행합니다.")

        return df

    def run_l3(self) -> pd.DataFrame:
        """L3: 패널 병합"""
        logger.info("[Pipeline L3] 패널 병합")

        # L1 의존성 확인
        if "ohlcv_daily" not in self.artifacts:
            logger.info("  → L1 먼저 실행")
            self.run_l1()

        # L2는 선택적
        if "fundamentals_annual" not in self.artifacts:
            self.run_l2()

        cache_path = self.interim_dir / "panel_merged_daily"

        if artifact_exists(cache_path) and not self.force_rebuild:
            self.artifacts["panel_merged_daily"] = load_artifact(cache_path)
            self.artifacts_path["panel"] = str(cache_path)
            logger.info(f"  ✓ 캐시에서 로드: {len(self.artifacts['panel_merged_daily']):,}행")
            return self.artifacts["panel_merged_daily"]

        df = collect_panel(
            ohlcv_daily=self.artifacts["ohlcv_daily"],
            fundamentals_annual=self.artifacts.get("fundamentals_annual"),
            universe_membership_monthly=self.artifacts.get("universe_k200_membership_monthly"),
            fundamental_lag_days=self.params.get("fundamental_lag_days", 90),
            filter_k200_members_only=self.params.get("filter_k200_members_only", False),
            config_path=self.config_path,
            save_to_cache=True,
            force_rebuild=self.force_rebuild,
        )

        self.artifacts["panel_merged_daily"] = df
        self.artifacts_path["panel"] = str(cache_path)
        return df

    def run_l4(self) -> Dict[str, any]:
        """L4: CV 분할 및 타겟 생성"""
        logger.info("[Pipeline L4] CV 분할 및 타겟 생성")

        # L3 의존성 확인
        if "panel_merged_daily" not in self.artifacts:
            logger.info("  → L3 먼저 실행")
            self.run_l3()

        dataset_path = self.interim_dir / "dataset_daily"
        cv_short_path = self.interim_dir / "cv_folds_short"
        cv_long_path = self.interim_dir / "cv_folds_long"

        if (artifact_exists(dataset_path) and
            artifact_exists(cv_short_path) and
            artifact_exists(cv_long_path) and
            not self.force_rebuild):
            self.artifacts["dataset_daily"] = load_artifact(dataset_path)
            self.artifacts["cv_folds_short"] = load_artifact(cv_short_path)
            self.artifacts["cv_folds_long"] = load_artifact(cv_long_path)
            self.artifacts_path["dataset"] = str(dataset_path)
            self.artifacts_path["cv_short"] = str(cv_short_path)
            self.artifacts_path["cv_long"] = str(cv_long_path)
            logger.info(f"  ✓ 캐시에서 로드: dataset {len(self.artifacts['dataset_daily']):,}행")
            return {
                "dataset_daily": self.artifacts["dataset_daily"],
                "cv_folds_short": self.artifacts["cv_folds_short"],
                "cv_folds_long": self.artifacts["cv_folds_long"],
            }

        result = collect_dataset(
            panel_merged_daily=self.artifacts["panel_merged_daily"],
            config_path=self.config_path,
            save_to_cache=True,
            force_rebuild=self.force_rebuild,
        )

        self.artifacts["dataset_daily"] = result["dataset_daily"]
        self.artifacts["cv_folds_short"] = result["cv_folds_short"]
        self.artifacts["cv_folds_long"] = result["cv_folds_long"]
        self.artifacts_path["dataset"] = str(dataset_path)
        self.artifacts_path["cv_short"] = str(cv_short_path)
        self.artifacts_path["cv_long"] = str(cv_long_path)

        return result

    def run_all(self) -> Dict[str, any]:
        """L0~L4 전체 실행"""
        logger.info("=" * 80)
        logger.info("데이터 수집 파이프라인 전체 실행 (L0~L4)")
        logger.info("=" * 80)

        # L0
        self.run_l0()

        # L1
        self.run_l1()

        # L2 (선택적)
        self.run_l2()

        # L3
        self.run_l3()

        # L4
        self.run_l4()

        logger.info("=" * 80)
        logger.info("✅ 데이터 수집 파이프라인 완료")
        logger.info("=" * 80)

        return {
            "artifacts": self.artifacts,
            "artifacts_path": self.artifacts_path,
        }

    def get_artifacts(self) -> Dict[str, any]:
        """수집된 아티팩트 반환"""
        return self.artifacts

    def get_artifacts_path(self) -> Dict[str, str]:
        """아티팩트 경로 반환"""
        return self.artifacts_path


def run_data_collection_pipeline(
    config_path: str = "configs/config.yaml",
    stages: Optional[List[str]] = None,
    force_rebuild: bool = False,
) -> Dict[str, any]:
    """
    데이터 수집 파이프라인 실행 (편의 함수)

    Args:
        config_path: 설정 파일 경로
        stages: 실행할 단계 리스트 (None이면 전체 실행)
                예: ["L0", "L1", "L3"] 또는 None (전체)
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        dict: {
            "artifacts": Dict[str, DataFrame],
            "artifacts_path": Dict[str, str],
        }

    Example:
        # 전체 실행
        result = run_data_collection_pipeline()

        # 특정 단계만 실행
        result = run_data_collection_pipeline(stages=["L0", "L1"])

        # 캐시 무시하고 재생성
        result = run_data_collection_pipeline(force_rebuild=True)
    """
    pipeline = DataCollectionPipeline(
        config_path=config_path,
        force_rebuild=force_rebuild,
    )

    if stages is None:
        # 전체 실행
        return pipeline.run_all()

    # 단계별 실행
    stage_map = {
        "L0": pipeline.run_l0,
        "L1": pipeline.run_l1,
        "L2": pipeline.run_l2,
        "L3": pipeline.run_l3,
        "L4": pipeline.run_l4,
    }

    for stage in stages:
        if stage not in stage_map:
            raise ValueError(f"Unknown stage: {stage}. Use one of {list(stage_map.keys())}")
        stage_map[stage]()

    return {
        "artifacts": pipeline.get_artifacts(),
        "artifacts_path": pipeline.get_artifacts_path(),
    }

