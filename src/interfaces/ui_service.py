# -*- coding: utf-8 -*-
"""
UI 서비스 인터페이스 모듈

Flask 등 UI 프레임워크에서 사용할 수 있는 함수들을 제공합니다.
- 랭킹 조회 (Track A)
- 백테스트 성과 조회 (Track B)
- 공통 데이터 준비 상태 확인

[리팩토링 2단계] UI에서 import 가능한 형태로 모듈화
"""
from typing import List, Literal, TypedDict, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import logging

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists

logger = logging.getLogger(__name__)


class RankingItem(TypedDict):
    """랭킹 항목 타입 정의"""
    ticker: str
    score: float
    rank: int
    horizon: Literal["short", "long", "combined"]


def _load_ranking_data(
    horizon: Literal["short", "long"],
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    """
    랭킹 데이터를 로드합니다 (Track A 산출물).
    
    Args:
        horizon: "short" 또는 "long"
        config_path: 설정 파일 경로
    
    Returns:
        DataFrame: 랭킹 데이터
    """
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    
    if horizon == "short":
        ranking_path = interim_dir / "ranking_short_daily"
    else:
        ranking_path = interim_dir / "ranking_long_daily"
    
    if not artifact_exists(ranking_path):
        raise FileNotFoundError(
            f"ranking_{horizon}_daily not found at {ranking_path}. "
            f"Run Track A first: python -m src.pipeline.track_a_pipeline"
        )
    
    return load_artifact(ranking_path)


def _load_rebalance_scores(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    rebalance_scores 데이터를 로드합니다 (Track B 산출물).
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        DataFrame: rebalance_scores 데이터
    """
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    scores_path = interim_dir / "rebalance_scores_from_ranking"
    
    if not artifact_exists(scores_path):
        raise FileNotFoundError(
            f"rebalance_scores_from_ranking not found at {scores_path}. "
            "Run Track B first: python -m src.pipeline.track_b_pipeline bt20_short"
        )
    
    return load_artifact(scores_path)


def get_short_term_ranking(
    as_of: str,
    top_k: int = 20,
    config_path: str = "configs/config.yaml",
) -> List[RankingItem]:
    """
    UI에서 '단기 랭킹' 요청 시 호출할 함수 (Track A 산출물 사용).
    
    [리팩토링 2단계] Track A의 ranking_short_daily를 직접 사용하도록 개선
    
    Args:
        as_of: 기준일 (YYYY-MM-DD 형식)
        top_k: 상위 K개 종목 반환
        config_path: 설정 파일 경로
    
    Returns:
        List[RankingItem]: 단기 랭킹 리스트
    
    Example:
        >>> rankings = get_short_term_ranking("2024-12-31", top_k=20)
        >>> print(rankings[0])
        {'ticker': '005930', 'score': 0.85, 'rank': 1, 'horizon': 'short'}
    """
    # Track A 산출물 직접 사용
    df = _load_ranking_data("short", config_path)
    
    # 날짜 필터링
    df["date"] = pd.to_datetime(df["date"])
    as_of_dt = pd.to_datetime(as_of)
    df_filtered = df[df["date"] == as_of_dt].copy()
    
    if len(df_filtered) == 0:
        logger.warning(f"기준일 {as_of}에 해당하는 데이터가 없습니다.")
        return []
    
    # 유니버스 필터링
    if "in_universe" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["in_universe"] == True].copy()
    
    # score_total과 rank_total 사용
    score_col = "score_total"
    rank_col = "rank_total"
    
    if score_col not in df_filtered.columns:
        raise ValueError(f"랭킹 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df_filtered.columns)}")
    
    # 랭킹 정렬 (높은 점수 순, 낮은 rank_total 순)
    df_sorted = df_filtered.sort_values([score_col, rank_col], ascending=[False, True]).head(top_k)
    
    # 결과 생성
    result = []
    for _, row in df_sorted.iterrows():
        result.append({
            "ticker": str(row["ticker"]).zfill(6),
            "score": float(row[score_col]) if pd.notna(row[score_col]) else 0.0,
            "rank": int(row[rank_col]) if pd.notna(row[rank_col]) else 999,
            "horizon": "short",
        })
    
    return result


def get_long_term_ranking(
    as_of: str,
    top_k: int = 20,
    config_path: str = "configs/config.yaml",
) -> List[RankingItem]:
    """
    UI에서 '장기 랭킹' 요청 시 호출할 함수 (Track A 산출물 사용).
    
    [리팩토링 2단계] Track A의 ranking_long_daily를 직접 사용하도록 개선
    
    Args:
        as_of: 기준일 (YYYY-MM-DD 형식)
        top_k: 상위 K개 종목 반환
        config_path: 설정 파일 경로
    
    Returns:
        List[RankingItem]: 장기 랭킹 리스트
    
    Example:
        >>> rankings = get_long_term_ranking("2024-12-31", top_k=20)
        >>> print(rankings[0])
        {'ticker': '005930', 'score': 0.82, 'rank': 1, 'horizon': 'long'}
    """
    # Track A 산출물 직접 사용
    df = _load_ranking_data("long", config_path)
    
    # 날짜 필터링
    df["date"] = pd.to_datetime(df["date"])
    as_of_dt = pd.to_datetime(as_of)
    df_filtered = df[df["date"] == as_of_dt].copy()
    
    if len(df_filtered) == 0:
        logger.warning(f"기준일 {as_of}에 해당하는 데이터가 없습니다.")
        return []
    
    # 유니버스 필터링
    if "in_universe" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["in_universe"] == True].copy()
    
    # score_total과 rank_total 사용
    score_col = "score_total"
    rank_col = "rank_total"
    
    if score_col not in df_filtered.columns:
        raise ValueError(f"랭킹 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df_filtered.columns)}")
    
    # 랭킹 정렬 (높은 점수 순, 낮은 rank_total 순)
    df_sorted = df_filtered.sort_values([score_col, rank_col], ascending=[False, True]).head(top_k)
    
    # 결과 생성
    result = []
    for _, row in df_sorted.iterrows():
        result.append({
            "ticker": str(row["ticker"]).zfill(6),
            "score": float(row[score_col]) if pd.notna(row[score_col]) else 0.0,
            "rank": int(row[rank_col]) if pd.notna(row[rank_col]) else 999,
            "horizon": "long",
        })
    
    return result


def get_combined_ranking(
    as_of: str,
    top_k: int = 20,
    config_path: str = "configs/config.yaml",
) -> List[RankingItem]:
    """
    UI에서 '통합 랭킹' 요청 시 호출할 함수 (Track B 산출물 사용).
    
    [리팩토링 2단계] Track B의 rebalance_scores_from_ranking에서 score_ens 사용
    
    Args:
        as_of: 기준일 (YYYY-MM-DD 형식)
        top_k: 상위 K개 종목 반환
        config_path: 설정 파일 경로
    
    Returns:
        List[RankingItem]: 통합 랭킹 리스트
    
    Example:
        >>> rankings = get_combined_ranking("2024-12-31", top_k=20)
        >>> print(rankings[0])
        {'ticker': '005930', 'score': 0.84, 'rank': 1, 'horizon': 'combined'}
    """
    # Track B 산출물 사용 (통합 스코어 포함)
    df = _load_rebalance_scores(config_path)
    
    # 날짜 필터링
    df["date"] = pd.to_datetime(df["date"])
    as_of_dt = pd.to_datetime(as_of)
    df_filtered = df[df["date"] == as_of_dt].copy()
    
    if len(df_filtered) == 0:
        logger.warning(f"기준일 {as_of}에 해당하는 데이터가 없습니다.")
        return []
    
    # 유니버스 필터링
    if "in_universe" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["in_universe"] == True].copy()
    
    # score_ens 컬럼 확인
    score_col = "score_ens"
    if score_col not in df_filtered.columns:
        # 대체 컬럼 시도
        for alt_col in ["score_ensemble", "score_total"]:
            if alt_col in df_filtered.columns:
                score_col = alt_col
                break
        else:
            raise ValueError(f"통합 랭킹 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df_filtered.columns)}")
    
    # 랭킹 정렬 (높은 점수 순)
    df_sorted = df_filtered.sort_values(score_col, ascending=False).head(top_k)
    
    # 결과 생성
    result = []
    for i, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        result.append({
            "ticker": str(row["ticker"]).zfill(6),
            "score": float(row[score_col]) if pd.notna(row[score_col]) else 0.0,
            "rank": i,
            "horizon": "combined",
        })
    
    return result


def get_backtest_metrics(
    strategy: str = "bt20_short",
    phase: Optional[Literal["dev", "holdout"]] = None,
    config_path: str = "configs/config.yaml",
) -> Dict[str, Any]:
    """
    UI에서 백테스트 성과 지표 조회 시 호출할 함수 (Track B 산출물 사용).
    
    [리팩토링 2단계] Track B의 bt_metrics를 조회하는 함수 추가
    
    Args:
        strategy: 전략 이름 ("bt20_short", "bt20_ens", "bt120_long", "bt120_ens")
        phase: 구간 필터링 ("dev" 또는 "holdout"), None이면 전체 반환
        config_path: 설정 파일 경로
    
    Returns:
        Dict: 백테스트 성과 지표
    
    Example:
        >>> metrics = get_backtest_metrics("bt20_short", phase="holdout")
        >>> print(metrics["net_sharpe"])
        0.7370
    """
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    metrics_path = interim_dir / f"bt_metrics_{strategy}"
    
    if not artifact_exists(metrics_path):
        raise FileNotFoundError(
            f"bt_metrics_{strategy} not found at {metrics_path}. "
            f"Run Track B first: python -m src.pipeline.track_b_pipeline {strategy}"
        )
    
    df = load_artifact(metrics_path)
    
    # Phase 필터링
    if phase and "phase" in df.columns:
        df = df[df["phase"] == phase].copy()
    
    if len(df) == 0:
        logger.warning(f"전략 {strategy}, 구간 {phase}에 해당하는 데이터가 없습니다.")
        return {}
    
    # 첫 번째 행을 딕셔너리로 변환
    result = df.iloc[0].to_dict()
    
    return result


def check_data_availability(
    config_path: str = "configs/config.yaml",
) -> Dict[str, bool]:
    """
    공통 데이터 준비 상태를 확인하는 함수.
    
    [리팩토링 2단계] 데이터 준비 상태 확인 함수 추가
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        Dict: 각 데이터 파일의 존재 여부
    
    Example:
        >>> status = check_data_availability()
        >>> print(status)
        {'universe': True, 'ohlcv': True, 'panel': True, 'dataset': True, ...}
    """
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    
    status = {
        "universe": artifact_exists(interim_dir / "universe_k200_membership_monthly"),
        "ohlcv": artifact_exists(interim_dir / "ohlcv_daily"),
        "fundamentals": artifact_exists(interim_dir / "fundamentals_annual"),
        "panel": artifact_exists(interim_dir / "panel_merged_daily"),
        "dataset": artifact_exists(interim_dir / "dataset_daily"),
        "cv_folds_short": artifact_exists(interim_dir / "cv_folds_short"),
        "cv_folds_long": artifact_exists(interim_dir / "cv_folds_long"),
        "ranking_short": artifact_exists(interim_dir / "ranking_short_daily"),
        "ranking_long": artifact_exists(interim_dir / "ranking_long_daily"),
        "rebalance_scores": artifact_exists(interim_dir / "rebalance_scores_from_ranking"),
    }
    
    return status


# Flask 사용 예시 (주석)
"""
# Flask app.py에서 사용 예시:
from flask import Flask, jsonify, request
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
    get_backtest_metrics,
    check_data_availability,
)

app = Flask(__name__)

# Track A: 랭킹 API
@app.get("/api/ranking/short")
def short_ranking():
    as_of = request.args.get("as_of", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_short_term_ranking(as_of, top_k))

@app.get("/api/ranking/long")
def long_ranking():
    as_of = request.args.get("as_of", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_long_term_ranking(as_of, top_k))

@app.get("/api/ranking/combined")
def combined_ranking():
    as_of = request.args.get("as_of", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_combined_ranking(as_of, top_k))

# Track B: 백테스트 성과 API
@app.get("/api/backtest/metrics")
def backtest_metrics():
    strategy = request.args.get("strategy", default="bt20_short")
    phase = request.args.get("phase", default=None)
    return jsonify(get_backtest_metrics(strategy, phase))

# 데이터 준비 상태 확인 API
@app.get("/api/data/status")
def data_status():
    return jsonify(check_data_availability())
"""

