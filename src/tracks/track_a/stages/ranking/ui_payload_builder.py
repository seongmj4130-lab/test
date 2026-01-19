# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/ranking/ui_payload_builder.py
"""
[Stage11] UI Payload Builder
Top/Bottom 일별 데이터 생성 및 최신 스냅샷 생성

입력:
- ranking_daily.parquet (date, ticker, rank_total, score_total, ...)
- ohlcv_daily.parquet (수익률 계산용)

출력:
- ui_top_bottom_daily.parquet (date, top_list, bottom_list, ...)
- ui_snapshot.csv (최신일 기준 Top10/Bottom10)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_ui_top_bottom_daily(
    ranking_daily: pd.DataFrame,
    top_k: int = 10,
    bottom_k: int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Top/Bottom 일별 데이터 생성

    Args:
        ranking_daily: ranking_daily DataFrame (date, ticker, rank_total, score_total, ...)
        top_k: 상위 종목 수 (기본 10)
        bottom_k: 하위 종목 수 (기본 10)

    Returns:
        (ui_top_bottom_daily DataFrame, warnings)
    """
    warns: List[str] = []

    # 필수 컬럼 확인
    required_cols = ["date", "ticker", "rank_total"]
    missing_cols = [c for c in required_cols if c not in ranking_daily.columns]
    if missing_cols:
        raise KeyError(f"ranking_daily missing required columns: {missing_cols}")

    # 날짜 정규화
    ranking_daily = ranking_daily.copy()
    ranking_daily["date"] = pd.to_datetime(ranking_daily["date"])

    # in_universe 필터링 (있으면)
    if "in_universe" in ranking_daily.columns:
        ranking_daily = ranking_daily[ranking_daily["in_universe"]].copy()

    # rank_total이 결측인 행 제거
    before = len(ranking_daily)
    ranking_daily = ranking_daily[ranking_daily["rank_total"].notna()].copy()
    if len(ranking_daily) < before:
        warns.append(f"dropped {before - len(ranking_daily)} rows with NA rank_total")

    # 날짜별로 Top/Bottom 추출
    rows = []

    for date, group in ranking_daily.groupby("date"):
        # rank_total 기준 정렬 (낮을수록 좋음)
        group_sorted = group.sort_values("rank_total", ascending=True)

        # Top K
        top_df = group_sorted.head(top_k)
        top_list = top_df["ticker"].tolist()
        top_scores = top_df["score_total"].tolist() if "score_total" in top_df.columns else [None] * len(top_list)

        # Bottom K
        bottom_df = group_sorted.tail(bottom_k)
        bottom_list = bottom_df["ticker"].tolist()
        bottom_scores = bottom_df["score_total"].tolist() if "score_total" in bottom_df.columns else [None] * len(bottom_list)

        # 기여도 요약 (contrib_* 컬럼이 있으면)
        contrib_cols = [c for c in group.columns if c.startswith("contrib_")]
        top_contrib_summary = {}
        bottom_contrib_summary = {}

        if contrib_cols:
            top_contrib_summary = {
                col: float(top_df[col].mean()) if col in top_df.columns else None
                for col in contrib_cols[:5]  # 최대 5개만
            }
            bottom_contrib_summary = {
                col: float(bottom_df[col].mean()) if col in bottom_df.columns else None
                for col in contrib_cols[:5]
            }

        row = {
            "date": date,
            "top_list": ",".join(top_list),
            "bottom_list": ",".join(bottom_list),
            "top_scores": ",".join([str(s) if s is not None else "" for s in top_scores]),
            "bottom_scores": ",".join([str(s) if s is not None else "" for s in bottom_scores]),
            "top_count": len(top_list),
            "bottom_count": len(bottom_list),
        }

        # 기여도 요약 추가
        for col, val in top_contrib_summary.items():
            row[f"top_{col}"] = val
        for col, val in bottom_contrib_summary.items():
            row[f"bottom_{col}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # 날짜 정렬
    df = df.sort_values("date").reset_index(drop=True)

    return df, warns

def build_ui_snapshot(
    ranking_daily: pd.DataFrame,
    top_k: int = 10,
    bottom_k: int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    최신일 기준 Top10/Bottom10 스냅샷 생성

    Args:
        ranking_daily: ranking_daily DataFrame
        top_k: 상위 종목 수 (기본 10)
        bottom_k: 하위 종목 수 (기본 10)

    Returns:
        (ui_snapshot DataFrame, warnings)
    """
    warns: List[str] = []

    # 필수 컬럼 확인
    required_cols = ["date", "ticker", "rank_total"]
    missing_cols = [c for c in required_cols if c not in ranking_daily.columns]
    if missing_cols:
        raise KeyError(f"ranking_daily missing required columns: {missing_cols}")

    # 날짜 정규화
    ranking_daily = ranking_daily.copy()
    ranking_daily["date"] = pd.to_datetime(ranking_daily["date"])

    # in_universe 필터링 (있으면)
    if "in_universe" in ranking_daily.columns:
        ranking_daily = ranking_daily[ranking_daily["in_universe"]].copy()

    # 최신일 추출
    latest_date = ranking_daily["date"].max()
    snapshot_df = ranking_daily[ranking_daily["date"] == latest_date].copy()

    if len(snapshot_df) == 0:
        raise ValueError(f"No data found for latest date: {latest_date}")

    # rank_total 기준 정렬
    snapshot_df = snapshot_df[snapshot_df["rank_total"].notna()].copy()
    snapshot_df = snapshot_df.sort_values("rank_total", ascending=True)

    # Top K
    top_df = snapshot_df.head(top_k).copy()
    top_df["snapshot_type"] = "top"
    top_df["snapshot_rank"] = range(1, len(top_df) + 1)

    # Bottom K
    bottom_df = snapshot_df.tail(bottom_k).copy()
    bottom_df["snapshot_type"] = "bottom"
    bottom_df["snapshot_rank"] = range(len(snapshot_df) - len(bottom_df) + 1, len(snapshot_df) + 1)

    # 병합
    result_df = pd.concat([top_df, bottom_df], ignore_index=True)

    # 컬럼 선택 (필요한 컬럼만)
    output_cols = ["snapshot_date", "snapshot_type", "snapshot_rank", "ticker", "rank_total", "score_total"]

    # 추가 컬럼 (있으면)
    optional_cols = ["regime_label", "regime_score", "top_features"]
    for col in optional_cols:
        if col in result_df.columns:
            output_cols.append(col)

    # contrib_* 컬럼 (최대 3개만)
    contrib_cols = [c for c in result_df.columns if c.startswith("contrib_")][:3]
    output_cols.extend(contrib_cols)

    # 존재하는 컬럼만 선택
    available_cols = [c for c in output_cols if c in result_df.columns]
    result_df = result_df[available_cols].copy()

    # snapshot_date 컬럼 추가
    result_df["snapshot_date"] = latest_date

    # 컬럼 순서 재정렬
    col_order = ["snapshot_date", "snapshot_type", "snapshot_rank"] + [c for c in available_cols if c not in ["snapshot_date", "snapshot_type", "snapshot_rank"]]
    result_df = result_df[col_order]

    return result_df, warns

def run_L11_ui_payload(
    cfg: dict,
    artifacts: dict,
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [Stage11] UI Payload Builder 실행 (통합)

    Args:
        cfg: 설정 딕셔너리
        artifacts: 이전 스테이지 산출물 딕셔너리
        force: 강제 재생성 플래그

    Returns:
        (outputs, warnings) 튜플
        - outputs: {"ui_top_bottom_daily": DataFrame, "ui_snapshot": DataFrame, "ui_equity_curves": DataFrame, "ui_metrics": DataFrame}
        - warnings: 경고 메시지 리스트
    """
    warns: list[str] = []

    # 입력 데이터 확인
    ranking_daily = artifacts.get("ranking_daily")
    ohlcv_daily = artifacts.get("ohlcv_daily")

    if ranking_daily is None:
        raise ValueError("L11 requires 'ranking_daily' in artifacts")

    if ohlcv_daily is None:
        raise ValueError("L11 requires 'ohlcv_daily' in artifacts")

    if len(ranking_daily) == 0:
        raise ValueError("ranking_daily is empty")

    if len(ohlcv_daily) == 0:
        raise ValueError("ohlcv_daily is empty")

    # 설정 읽기
    l11 = cfg.get("l11", {}) or {}
    top_k = int(l11.get("top_k", 10))
    bottom_k = int(l11.get("bottom_k", 10))
    top_k_perf = int(l11.get("top_k_perf", 20))  # 성과 곡선용 top_k
    benchmark_type = l11.get("benchmark_type", "universe_mean")

    # Top/Bottom 일별 데이터 생성
    ui_top_bottom_daily, warns_top_bottom = build_ui_top_bottom_daily(
        ranking_daily,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    warns.extend(warns_top_bottom)

    # 최신 스냅샷 생성
    ui_snapshot, warns_snapshot = build_ui_snapshot(
        ranking_daily,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    warns.extend(warns_snapshot)

    # 성과 곡선 생성 (ranking_demo_performance 모듈 사용)
    from src.stages.ranking.ranking_demo_performance import (  # [개선안 17번] 멀티 벤치마크 지원
        build_benchmark_returns,
        build_equity_curves,
        build_equity_curves_multi,
        build_top20_equal_weight_returns,
        calculate_performance_metrics,
        calculate_performance_metrics_multi,
        calculate_returns,
    )

    # 수익률 계산
    returns = calculate_returns(ohlcv_daily)

    # Top20 equal-weight 수익률 계산
    strategy_returns = build_top20_equal_weight_returns(
        ranking_daily,
        returns,
        top_k=top_k_perf,
    )

    # [개선안 17번] 벤치마크 표준화: 단일/멀티 지원
    benchmark_types = l11.get("benchmark_types", None)
    if isinstance(benchmark_types, list) and len(benchmark_types) > 0:
        bench_map = {}
        for bt in benchmark_types:
            bench_map[str(bt)] = build_benchmark_returns(
                ohlcv_daily,
                benchmark_type=str(bt),
                cfg=cfg,
            )
        ui_equity_curves = build_equity_curves_multi(
            strategy_returns=strategy_returns,
            benchmark_returns_by_type=bench_map,
        )
        ui_metrics = calculate_performance_metrics_multi(ui_equity_curves)
    else:
        # 벤치마크 수익률 계산(단일)
        benchmark_returns = build_benchmark_returns(
            ohlcv_daily,
            benchmark_type=benchmark_type,
            cfg=cfg,
        )
        # 누적 곡선 계산
        ui_equity_curves = build_equity_curves(
            strategy_returns,
            benchmark_returns,
        )
        # 성과 지표 계산
        metrics = calculate_performance_metrics(ui_equity_curves)
        ui_metrics = pd.DataFrame([metrics])

    outputs = {
        "ui_top_bottom_daily": ui_top_bottom_daily,
        "ui_snapshot": ui_snapshot,
        "ui_equity_curves": ui_equity_curves,
        "ui_metrics": ui_metrics,
    }

    return outputs, warns
