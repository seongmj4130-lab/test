# -*- coding: utf-8 -*-
"""
공통 데이터 준비 파이프라인 (L0~L4)

이 모듈은 Track A와 Track B 모두에서 사용하는 공통 데이터를 준비합니다.
데이터 수집 로직을 완전히 분리하여 "데이터는 기존 그대로" 보장합니다.

사용 방법:
    from src.tracks.shared.data_pipeline import prepare_common_data
    
    artifacts = prepare_common_data(config_path="configs/config.yaml")
"""
from pathlib import Path
from typing import Dict, Optional, List
import logging
import pandas as pd

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists, save_artifact
from src.utils.meta import build_meta, save_meta
from src.utils.validate import validate_df, raise_if_invalid
from src.utils.quality import fundamental_coverage_report, walkforward_quality_report

# 데이터 수집 함수들 (기존 그대로 사용)
from src.tracks.shared.stages.data.l0_universe import build_k200_membership_month_end
from src.tracks.shared.stages.data.l1_ohlcv import download_ohlcv_panel
from src.tracks.shared.stages.data.l2_fundamentals_dart import download_annual_fundamentals
from src.tracks.shared.stages.data.l3_panel_merge import build_panel_merged_daily
from src.tracks.shared.stages.data.l4_walkforward_split import build_targets_and_folds

logger = logging.getLogger(__name__)


def prepare_common_data(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    공통 데이터 준비 파이프라인 실행 (L0~L4)
    
    Track A와 Track B 모두에서 사용하는 공통 데이터를 준비합니다.
    캐시 우선 방식으로 동작하며, 이미 생성된 데이터가 있으면 재사용합니다.
    
    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재계산
    
    Returns:
        dict: 공통 데이터 아티팩트
        {
            "universe_k200_membership_monthly": DataFrame,
            "ohlcv_daily": DataFrame,
            "fundamentals_annual": DataFrame (선택적),
            "panel_merged_daily": DataFrame,
            "dataset_daily": DataFrame,
            "cv_folds_short": DataFrame,
            "cv_folds_long": DataFrame,
        }
    
    Example:
        >>> artifacts = prepare_common_data()
        >>> universe = artifacts["universe_k200_membership_monthly"]
        >>> panel = artifacts["panel_merged_daily"]
    """
    logger.info("=" * 80)
    logger.info("공통 데이터 준비 파이프라인 실행 (L0~L4)")
    logger.info("=" * 80)
    
    # 설정 로드
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {}
    p = cfg.get("params", {})
    
    # L0: 유니버스 구성
    logger.info("[L0] 유니버스 구성")
    uni_path = interim_dir / "universe_k200_membership_monthly"
    if artifact_exists(uni_path) and not force_rebuild:
        artifacts["universe_k200_membership_monthly"] = load_artifact(uni_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['universe_k200_membership_monthly']):,}행")
    else:
        logger.info("  → 유니버스 생성 중...")
        df = build_k200_membership_month_end(
            start_date=p.get("start_date", "2016-01-01"),
            end_date=p.get("end_date", "2024-12-31"),
            index_code=p.get("index_code", "1028"),
            anchor_ticker=p.get("anchor_ticker", "005930"),
        )
        artifacts["universe_k200_membership_monthly"] = df
        save_artifact(df, uni_path, force=True)
        logger.info(f"  ✓ 생성 완료: {len(df):,}행")
    
    # L1: OHLCV 다운로드 + 기술적 지표 계산
    logger.info("[L1] OHLCV 다운로드 + 기술적 지표 계산")
    ohlcv_path = interim_dir / "ohlcv_daily"
    if artifact_exists(ohlcv_path) and not force_rebuild:
        artifacts["ohlcv_daily"] = load_artifact(ohlcv_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['ohlcv_daily']):,}행")
    else:
        logger.info("  → OHLCV 다운로드 중...")
        tickers = sorted(artifacts["universe_k200_membership_monthly"]["ticker"].astype(str).unique().tolist())
        df = download_ohlcv_panel(
            tickers=tickers,
            start_date=p.get("start_date", "2016-01-01"),
            end_date=p.get("end_date", "2024-12-31"),
            calculate_technical_features=True,
        )
        artifacts["ohlcv_daily"] = df
        save_artifact(df, ohlcv_path, force=True)
        logger.info(f"  ✓ 생성 완료: {len(df):,}행, {len(df.columns)}컬럼")
    
    # L2: 재무 데이터 로드
    logger.info("[L2] 재무 데이터 로드")
    fund_path = interim_dir / "fundamentals_annual"
    if artifact_exists(fund_path) and not force_rebuild:
        artifacts["fundamentals_annual"] = load_artifact(fund_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['fundamentals_annual']):,}행")
    else:
        logger.info("  → 재무 데이터 확인 중...")
        # 기존 데이터가 있으면 사용, 없으면 None (L3에서 처리)
        if fund_path.exists():
            artifacts["fundamentals_annual"] = load_artifact(fund_path)
            logger.info(f"  ✓ 기존 데이터 사용: {len(artifacts['fundamentals_annual']):,}행")
        else:
            logger.warning("  ⚠ 재무 데이터가 없습니다. L3에서 재무 데이터 없이 진행합니다.")
            artifacts["fundamentals_annual"] = None
    
    # L3: 패널 병합
    logger.info("[L3] 패널 병합")
    panel_path = interim_dir / "panel_merged_daily"
    if artifact_exists(panel_path) and not force_rebuild:
        artifacts["panel_merged_daily"] = load_artifact(panel_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['panel_merged_daily']):,}행")
    else:
        logger.info("  → 패널 병합 중...")
        lag_days = int(p.get("fundamental_lag_days", 90))
        l3_cfg = cfg.get("l3", {}) or {}
        
        df = build_panel_merged_daily(
            ohlcv_daily=artifacts["ohlcv_daily"],
            fundamentals_annual=artifacts.get("fundamentals_annual"),
            news_sentiment_daily=None,  # 선택적
            esg_sentiment_daily=None,    # 선택적
            universe_k200_membership_monthly=artifacts["universe_k200_membership_monthly"],
            disclosure_lag_days=lag_days,
            sector_map_enabled=l3_cfg.get("sector_map_enabled", True),
        )
        artifacts["panel_merged_daily"] = df
        save_artifact(df, panel_path, force=True)
        logger.info(f"  ✓ 생성 완료: {len(df):,}행")
    
    # L4: Walk-Forward CV 분할 및 타겟 생성
    logger.info("[L4] Walk-Forward CV 분할 및 타겟 생성")
    dataset_path = interim_dir / "dataset_daily"
    cv_short_path = interim_dir / "cv_folds_short"
    cv_long_path = interim_dir / "cv_folds_long"
    
    if (artifact_exists(dataset_path) and 
        artifact_exists(cv_short_path) and 
        artifact_exists(cv_long_path) and 
        not force_rebuild):
        artifacts["dataset_daily"] = load_artifact(dataset_path)
        artifacts["cv_folds_short"] = load_artifact(cv_short_path)
        artifacts["cv_folds_long"] = load_artifact(cv_long_path)
        logger.info(f"  ✓ 캐시에서 로드: dataset {len(artifacts['dataset_daily']):,}행")
    else:
        logger.info("  → CV 분할 및 타겟 생성 중...")
        l4_cfg = cfg.get("l4", {}) or {}
        
        dataset, folds_short, folds_long = build_targets_and_folds(
            panel_merged_daily=artifacts["panel_merged_daily"],
            holdout_years=int(l4_cfg.get("holdout_years", 2)),
            step_days=int(l4_cfg.get("step_days", 20)),
            test_window_days=int(l4_cfg.get("test_window_days", 20)),
            embargo_days=int(l4_cfg.get("embargo_days", 20)),
            horizon_short=int(l4_cfg.get("horizon_short", 20)),
            horizon_long=int(l4_cfg.get("horizon_long", 120)),
            rolling_train_years_short=int(l4_cfg.get("rolling_train_years_short", 3)),
            rolling_train_years_long=int(l4_cfg.get("rolling_train_years_long", 5)),
            market_neutral=bool(l4_cfg.get("market_neutral", False)),
        )
        
        artifacts["dataset_daily"] = dataset
        artifacts["cv_folds_short"] = folds_short
        artifacts["cv_folds_long"] = folds_long
        
        save_artifact(dataset, dataset_path, force=True)
        save_artifact(folds_short, cv_short_path, force=True)
        save_artifact(folds_long, cv_long_path, force=True)
        logger.info(f"  ✓ 생성 완료: dataset {len(dataset):,}행, folds_short {len(folds_short):,}행, folds_long {len(folds_long):,}행")
    
    logger.info("=" * 80)
    logger.info("✅ 공통 데이터 준비 완료")
    logger.info("=" * 80)
    
    return artifacts


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    force = "--force" in sys.argv
    artifacts = prepare_common_data(force_rebuild=force)
    print(f"\n✅ 완료: {len(artifacts)}개 아티팩트 생성")

