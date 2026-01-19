"""
데이터 수집 함수 모듈

[리팩토링 1단계] 데이터 수집 완전 분리
- L0~L4를 독립적인 함수로 분리
- 기존 데이터는 그대로 유지
- UI에서 import 가능한 형태로 제공
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact

# L0~L4 스테이지 함수 import
# [리팩토링 1단계] 기존 스테이지 함수 재사용 (데이터는 그대로 유지)
try:
    from src.tracks.shared.stages.data.l0_universe import (
        build_k200_membership_month_end,
    )
except ImportError:
    from src.stages.data.l0_universe import build_k200_membership_month_end

try:
    from src.tracks.shared.stages.data.l1_ohlcv import download_ohlcv_panel
except ImportError:
    from src.stages.data.l1_ohlcv import download_ohlcv_panel

try:
    from src.tracks.shared.stages.data.l2_fundamentals_dart import (
        download_annual_fundamentals,
    )
except ImportError:
    pass

try:
    from src.tracks.shared.stages.data.l3_panel_merge import build_panel_merged_daily
except ImportError:
    from src.stages.data.l3_panel_merge import build_panel_merged_daily

try:
    from src.tracks.shared.stages.data.l4_walkforward_split import (
        build_targets_and_folds,
    )
except ImportError:
    from src.stages.data.l4_walkforward_split import build_targets_and_folds

logger = logging.getLogger(__name__)


def collect_universe(
    start_date: str = "2016-01-01",
    end_date: str = "2024-12-31",
    index_code: str = "1028",
    anchor_ticker: str = "005930",
    config_path: Optional[str] = None,
    save_to_cache: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    L0: 유니버스 구성 (KOSPI200 멤버십)

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        index_code: 지수 코드 (기본값: 1028 = KOSPI200)
        anchor_ticker: 기준 종목 코드 (기본값: 005930 = 삼성전자)
        config_path: 설정 파일 경로 (캐시 경로 확인용)
        save_to_cache: 캐시에 저장 여부
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame: date, ym, ticker 컬럼 포함
    """
    logger.info(f"[L0] 유니버스 구성 시작: {start_date} ~ {end_date}")

    # 캐시 확인
    if config_path and not force_rebuild:
        cfg = load_config(config_path)
        interim_dir = Path(get_path(cfg, "data_interim"))
        cache_path = interim_dir / "universe_k200_membership_monthly"

        if artifact_exists(cache_path):
            logger.info(f"[L0] 캐시에서 로드: {cache_path}")
            return load_artifact(cache_path)

    # 데이터 수집
    df = build_k200_membership_month_end(
        start_date=start_date,
        end_date=end_date,
        index_code=index_code,
        anchor_ticker=anchor_ticker,
        strict=True,
    )

    # 캐시 저장
    if config_path and save_to_cache:
        cfg = load_config(config_path)
        interim_dir = Path(get_path(cfg, "data_interim"))
        cache_path = interim_dir / "universe_k200_membership_monthly"
        save_artifact(df, cache_path, force=True)
        logger.info(f"[L0] 캐시에 저장: {cache_path}")

    logger.info(f"[L0] 완료: {len(df):,}행")
    return df


def collect_ohlcv(
    universe: pd.DataFrame,
    start_date: str = "2016-01-01",
    end_date: str = "2024-12-31",
    calculate_technical_features: bool = True,
    config_path: Optional[str] = None,
    save_to_cache: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    L1: OHLCV 데이터 다운로드 + 기술적 지표 계산

    Args:
        universe: 유니버스 DataFrame (ticker 컬럼 필요)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        calculate_technical_features: 기술적 지표 계산 여부
        config_path: 설정 파일 경로 (캐시 경로 확인용)
        save_to_cache: 캐시에 저장 여부
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame: date, ticker, open, high, low, close, volume, value 및 기술적 지표 컬럼 포함
    """
    logger.info(f"[L1] OHLCV 다운로드 시작: {start_date} ~ {end_date}")

    # 캐시 확인
    if config_path and not force_rebuild:
        cfg = load_config(config_path)
        interim_dir = Path(get_path(cfg, "data_interim"))
        cache_path = interim_dir / "ohlcv_daily"

        if artifact_exists(cache_path):
            logger.info(f"[L1] 캐시에서 로드: {cache_path}")
            return load_artifact(cache_path)

    # ticker 추출
    tickers = sorted(universe["ticker"].astype(str).unique().tolist())
    logger.info(f"[L1] 종목 수: {len(tickers)}개")

    # 데이터 수집
    df = download_ohlcv_panel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        calculate_technical_features=calculate_technical_features,
    )

    # 캐시 저장
    if config_path and save_to_cache:
        cfg = load_config(config_path)
        interim_dir = Path(get_path(cfg, "data_interim"))
        cache_path = interim_dir / "ohlcv_daily"
        save_artifact(df, cache_path, force=True)
        logger.info(f"[L1] 캐시에 저장: {cache_path}")

    logger.info(f"[L1] 완료: {len(df):,}행, {len(df.columns)}컬럼")
    return df


def collect_fundamentals(
    config_path: Optional[str] = None,
    save_to_cache: bool = True,
    force_rebuild: bool = False,
) -> Optional[pd.DataFrame]:
    """
    L2: 재무 데이터 로드 (기존 데이터 사용, 새로 다운로드 안 함)

    Args:
        config_path: 설정 파일 경로 (캐시 경로 확인용)
        save_to_cache: 캐시에 저장 여부
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame 또는 None: 재무 데이터가 없으면 None 반환
    """
    logger.info("[L2] 재무 데이터 로드 시작")

    if not config_path:
        logger.warning("[L2] config_path가 없어 기존 데이터를 찾을 수 없습니다.")
        return None

    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    cache_path = interim_dir / "fundamentals_annual"

    # 기존 데이터 확인
    if artifact_exists(cache_path) and not force_rebuild:
        logger.info(f"[L2] 기존 데이터 로드: {cache_path}")
        return load_artifact(cache_path)

    # 새로 다운로드 (선택적)
    logger.warning("[L2] 기존 fundamentals_annual 데이터가 없습니다.")
    logger.warning("[L2] L3에서 재무 데이터 없이 진행합니다.")
    return None


def collect_panel(
    ohlcv_daily: pd.DataFrame,
    fundamentals_annual: Optional[pd.DataFrame] = None,
    universe_membership_monthly: Optional[pd.DataFrame] = None,
    fundamental_lag_days: int = 90,
    filter_k200_members_only: bool = False,
    config_path: Optional[str] = None,
    save_to_cache: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    L3: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)

    Args:
        ohlcv_daily: OHLCV 데이터
        fundamentals_annual: 재무 데이터 (선택적)
        universe_membership_monthly: 유니버스 멤버십 (선택적)
        fundamental_lag_days: 재무 데이터 지연 일수
        filter_k200_members_only: K200 멤버만 필터링 여부
        config_path: 설정 파일 경로 (캐시 경로 확인용)
        save_to_cache: 캐시에 저장 여부
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        DataFrame: 병합된 패널 데이터
    """
    logger.info("[L3] 패널 병합 시작")

    # 캐시 확인
    if config_path and not force_rebuild:
        cfg = load_config(config_path)
        interim_dir = Path(get_path(cfg, "data_interim"))
        cache_path = interim_dir / "panel_merged_daily"

        if artifact_exists(cache_path):
            logger.info(f"[L3] 캐시에서 로드: {cache_path}")
            cached_df = load_artifact(cache_path)

            # 기술적 지표가 포함되어 있는지 확인하고 없으면 병합
            technical_cols = [
                c
                for c in cached_df.columns
                if c in ["price_momentum_20d", "volatility_20d", "market_cap"]
            ]
            if not technical_cols and ohlcv_daily is not None:
                technical_cols_ohlcv = [
                    c
                    for c in ohlcv_daily.columns
                    if c
                    in [
                        "price_momentum_20d",
                        "price_momentum_60d",
                        "volatility_20d",
                        "volatility_60d",
                        "max_drawdown_60d",
                        "volume_ratio",
                        "momentum_reversal",
                    ]
                ]
                if technical_cols_ohlcv:
                    cached_df = cached_df.merge(
                        ohlcv_daily[["date", "ticker"] + technical_cols_ohlcv],
                        on=["date", "ticker"],
                        how="left",
                        suffixes=("", "_new"),
                    )
                    for col in technical_cols_ohlcv:
                        if (
                            col not in cached_df.columns
                            and f"{col}_new" in cached_df.columns
                        ):
                            cached_df[col] = cached_df[f"{col}_new"]
                            cached_df = cached_df.drop(columns=[f"{col}_new"])
                    logger.info(
                        f"[L3] 기술적 지표 병합 완료: {len(technical_cols_ohlcv)}개"
                    )

            return cached_df

    # 설정 로드
    cfg = None
    l3_cfg = {}
    if config_path:
        cfg = load_config(config_path)
        l3_cfg = cfg.get("l3", {}) or {}

    # 데이터 수집
    df, warns = build_panel_merged_daily(
        ohlcv_daily=ohlcv_daily,
        fundamentals_annual=fundamentals_annual,
        universe_membership_monthly=universe_membership_monthly,
        fundamental_lag_days=fundamental_lag_days,
        filter_k200_members_only=filter_k200_members_only,
        fundamentals_effective_date_col=l3_cfg.get(
            "fundamentals_effective_date_col", "effective_date"
        ),
    )

    if warns:
        for w in warns:
            logger.warning(f"[L3] {w}")

    # 캐시 저장
    if config_path and save_to_cache:
        cfg = load_config(config_path)
        interim_dir = Path(get_path(cfg, "data_interim"))
        cache_path = interim_dir / "panel_merged_daily"
        save_artifact(df, cache_path, force=True)
        logger.info(f"[L3] 캐시에 저장: {cache_path}")

    logger.info(f"[L3] 완료: {len(df):,}행")
    return df


def collect_dataset(
    panel_merged_daily: pd.DataFrame,
    config_path: Optional[str] = None,
    save_to_cache: bool = True,
    force_rebuild: bool = False,
) -> dict[str, any]:
    """
    L4: Walk-Forward CV 분할 및 타겟 생성

    Args:
        panel_merged_daily: 병합된 패널 데이터
        config_path: 설정 파일 경로
        save_to_cache: 캐시에 저장 여부
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        dict: {
            "dataset_daily": DataFrame,
            "cv_folds_short": list,
            "cv_folds_long": list,
        }
    """
    logger.info("[L4] CV 분할 및 타겟 생성 시작")

    # 설정 로드
    if not config_path:
        raise ValueError("config_path가 필요합니다.")

    cfg = load_config(config_path)
    l4 = cfg.get("l4", {}) or {}

    # 캐시 확인
    if not force_rebuild:
        interim_dir = Path(get_path(cfg, "data_interim"))
        dataset_path = interim_dir / "dataset_daily"
        cv_short_path = interim_dir / "cv_folds_short"
        cv_long_path = interim_dir / "cv_folds_long"

        if (
            artifact_exists(dataset_path)
            and artifact_exists(cv_short_path)
            and artifact_exists(cv_long_path)
        ):
            logger.info("[L4] 캐시에서 로드")
            return {
                "dataset_daily": load_artifact(dataset_path),
                "cv_folds_short": load_artifact(cv_short_path),
                "cv_folds_long": load_artifact(cv_long_path),
            }

    # 데이터 수집
    df, cv_s, cv_l, warns = build_targets_and_folds(
        panel_merged_daily=panel_merged_daily,
        holdout_years=int(l4.get("holdout_years", 2)),
        step_days=int(l4.get("step_days", 20)),
        test_window_days=int(l4.get("test_window_days", 20)),
        embargo_days=int(l4.get("embargo_days", 20)),
        horizon_short=int(l4.get("horizon_short", 20)),
        horizon_long=int(l4.get("horizon_long", 120)),
        rolling_train_years_short=int(l4.get("rolling_train_years_short", 3)),
        rolling_train_years_long=int(l4.get("rolling_train_years_long", 5)),
        price_col=l4.get("price_col", None),
    )

    if warns:
        for w in warns:
            logger.warning(f"[L4] {w}")

    # 캐시 저장
    if save_to_cache:
        interim_dir = Path(get_path(cfg, "data_interim"))
        save_artifact(df, interim_dir / "dataset_daily", force=True)
        save_artifact(cv_s, interim_dir / "cv_folds_short", force=True)
        save_artifact(cv_l, interim_dir / "cv_folds_long", force=True)
        logger.info("[L4] 캐시에 저장")

    logger.info(
        f"[L4] 완료: dataset_daily {len(df):,}행, cv_folds_short {len(cv_s)}개, cv_folds_long {len(cv_l)}개"
    )

    return {
        "dataset_daily": df,
        "cv_folds_short": cv_s,
        "cv_folds_long": cv_l,
    }


def collect_all_data(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> dict[str, any]:
    """
    L0~L4 전체 데이터 수집

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재생성

    Returns:
        dict: {
            "universe_k200_membership_monthly": DataFrame,
            "ohlcv_daily": DataFrame,
            "fundamentals_annual": DataFrame 또는 None,
            "panel_merged_daily": DataFrame,
            "dataset_daily": DataFrame,
            "cv_folds_short": list,
            "cv_folds_long": list,
        }
    """
    logger.info("=" * 80)
    logger.info("전체 데이터 수집 시작 (L0~L4)")
    logger.info("=" * 80)

    cfg = load_config(config_path)
    params = cfg.get("params", {})

    # L0: 유니버스
    universe = collect_universe(
        start_date=params.get("start_date", "2016-01-01"),
        end_date=params.get("end_date", "2024-12-31"),
        index_code=params.get("index_code", "1028"),
        anchor_ticker=params.get("anchor_ticker", "005930"),
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=force_rebuild,
    )

    # L1: OHLCV
    ohlcv = collect_ohlcv(
        universe=universe,
        start_date=params.get("start_date", "2016-01-01"),
        end_date=params.get("end_date", "2024-12-31"),
        calculate_technical_features=True,
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=force_rebuild,
    )

    # L2: 재무 데이터 (선택적)
    fundamentals = collect_fundamentals(
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=force_rebuild,
    )

    # L3: 패널 병합
    panel = collect_panel(
        ohlcv_daily=ohlcv,
        fundamentals_annual=fundamentals,
        universe_membership_monthly=universe,
        fundamental_lag_days=params.get("fundamental_lag_days", 90),
        filter_k200_members_only=params.get("filter_k200_members_only", False),
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=force_rebuild,
    )

    # L4: CV 분할
    dataset_result = collect_dataset(
        panel_merged_daily=panel,
        config_path=config_path,
        save_to_cache=True,
        force_rebuild=force_rebuild,
    )

    logger.info("=" * 80)
    logger.info("✅ 전체 데이터 수집 완료")
    logger.info("=" * 80)

    return {
        "universe_k200_membership_monthly": universe,
        "ohlcv_daily": ohlcv,
        "fundamentals_annual": fundamentals,
        "panel_merged_daily": panel,
        "dataset_daily": dataset_result["dataset_daily"],
        "cv_folds_short": dataset_result["cv_folds_short"],
        "cv_folds_long": dataset_result["cv_folds_long"],
    }
