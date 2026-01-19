# -*- coding: utf-8 -*-
# [개선안 36번] Track A: Holdout 단일 일자 TopK 랭킹 + 팩터셋(그룹) 기여도 Top3 분석 (함수화)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import pandas as pd
import yaml

from src.components.ranking.contribution_engine import (
    ContributionConfig,
    compute_group_contributions_for_day,
    load_group_map_from_yaml,
    pick_top_groups_per_row,
)


Horizon = Literal["short", "long", "both"]


@dataclass(frozen=True)
class HoldoutDayInspectResult:
    """
    [개선안 36번] Holdout 단일 일자 분석 결과 컨테이너
    """

    date: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    short: Optional[pd.DataFrame]
    long: Optional[pd.DataFrame]


def holdout_range_from_cfg(cfg: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    [개선안 36번] 현재 dataset_daily에 phase/dev/holdout 컬럼이 없어서 config 기반으로 holdout 기간을 산출.
    - end_date 기준 마지막 N년을 holdout으로 가정 (holdout_years)
    """
    params = cfg.get("params", {}) or {}
    l4 = cfg.get("l4", {}) or {}
    end_date = pd.Timestamp(params.get("end_date"))
    holdout_years = int(l4.get("holdout_years", 2))
    start_year = end_date.year - holdout_years + 1
    holdout_start = pd.Timestamp(f"{start_year}-01-01")
    holdout_end = end_date.normalize()
    return holdout_start, holdout_end


def _pick_ranking_file(interim_dir: Path, base_name: str, normalization_method: str) -> Path:
    """
    ranking_short_daily_zscore.parquet 같은 변형 파일이 있으면 우선 사용.
    """
    method = str(normalization_method).lower()
    cand = interim_dir / f"{base_name}_{method}.parquet"
    if cand.exists():
        return cand
    return interim_dir / f"{base_name}.parquet"


def _load_feature_weights_and_group_map(
    cfg: dict,
    *,
    horizon: Literal["short", "long"],
) -> Tuple[Dict[str, float], Optional[Dict[str, list[str]]]]:
    """
    [개선안 36번] l8_short/l8_long의 feature_weights_config를 로드하고,
    feature_groups_config가 존재하면 group_map도 함께 로드.
    """
    base_dir = Path(cfg.get("paths", {}).get("base_dir", "."))
    l8_key = "l8_short" if horizon == "short" else "l8_long"
    l8 = cfg.get(l8_key, {}) or {}

    weights_rel = l8.get("feature_weights_config")
    if not weights_rel:
        raise ValueError(f"{l8_key}.feature_weights_config is missing in config.yaml")
    weights_path = base_dir / str(weights_rel)
    if not weights_path.exists():
        raise FileNotFoundError(f"feature_weights_config not found: {weights_path}")

    with open(weights_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    feature_weights = data.get("feature_weights", {}) or {}
    feature_weights = {str(k): float(v) for k, v in feature_weights.items()}

    groups_rel = l8.get("feature_groups_config")
    groups_path = (base_dir / str(groups_rel)) if groups_rel else None
    if groups_path is not None and not groups_path.exists():
        groups_path = None

    group_map = load_group_map_from_yaml(groups_path) if groups_path else None
    return feature_weights, group_map


def _format_top_groups_row(row: pd.Series) -> str:
    parts = []
    for i in range(1, 4):
        name = row.get(f"group_top{i}_name")
        val = row.get(f"group_top{i}_value")
        if pd.isna(name) or pd.isna(val):
            continue
        parts.append(f"{name}({val:+.4f})")
    return ", ".join(parts)


def inspect_holdout_day(
    *,
    cfg: dict,
    date: str | pd.Timestamp,
    dataset_daily: pd.DataFrame,
    ranking_short_daily: pd.DataFrame,
    ranking_long_daily: pd.DataFrame,
    topk: int = 10,
    horizon: Horizon = "both",
) -> HoldoutDayInspectResult:
    """
    [개선안 36번] Track A Holdout 기간 중 특정 날짜의 TopK 랭킹과 팩터셋(그룹) 기여도 Top3를 계산.

    Args:
        cfg: load_config()로 로드한 설정 딕셔너리
        date: 분석 대상 날짜 (예: "2024-12-30")
        dataset_daily: data/interim/dataset_daily 아티팩트 (그 날짜의 cross-sectional 정규화에 사용)
        ranking_short_daily: Track A 단기 랭킹 daily
        ranking_long_daily: Track A 장기 랭킹 daily
        topk: TopK 종목 수
        horizon: "short"|"long"|"both"

    Returns:
        HoldoutDayInspectResult
    """
    date_ts = pd.Timestamp(date)
    hold_s, hold_e = holdout_range_from_cfg(cfg)
    if not (hold_s <= date_ts <= hold_e):
        raise ValueError(
            f"입력 date={date_ts.date()} 가 holdout 범위 밖입니다. "
            f"holdout={hold_s.date()}~{hold_e.date()} (config 기반 추정)"
        )

    # 공통 전처리
    dd = dataset_daily.copy()
    dd["date"] = pd.to_datetime(dd["date"], errors="raise")
    dd["ticker"] = dd["ticker"].astype(str).str.zfill(6)
    day_df = dd[dd["date"] == date_ts].copy()
    if len(day_df) == 0:
        raise ValueError(f"dataset_daily에 date={date_ts.date()} 행이 없습니다.")

    def _one(h: Literal["short", "long"], ranking_df: pd.DataFrame) -> pd.DataFrame:
        l8_key = "l8_short" if h == "short" else "l8_long"
        l8 = cfg.get(l8_key, {}) or {}
        normalization_method = l8.get("normalization_method", "percentile")

        r = ranking_df.copy()
        r["date"] = pd.to_datetime(r["date"], errors="raise")
        r["ticker"] = r["ticker"].astype(str).str.zfill(6)
        day_rank = r[r["date"] == date_ts].copy()
        if len(day_rank) == 0:
            raise ValueError(f"[{h}] ranking_daily에 date={date_ts.date()} 행이 없습니다.")

        day_rank = day_rank[day_rank["rank_total"].notna()].copy()
        day_top = day_rank.nsmallest(topk, "rank_total").copy()

        feature_weights, group_map = _load_feature_weights_and_group_map(cfg, horizon=h)
        contrib_cfg = ContributionConfig(
            normalization_method=normalization_method,
            date_col="date",
            ticker_col="ticker",
            universe_col="in_universe",
            sector_col=l8.get("sector_col"),
            use_sector_relative=bool(l8.get("use_sector_relative", False)),
        )
        contrib = compute_group_contributions_for_day(
            day_df,
            feature_weights=feature_weights,
            group_map=group_map,
            cfg=contrib_cfg,
        )
        contrib = pick_top_groups_per_row(contrib, top_n=3)

        merged = day_top.merge(contrib, on=["date", "ticker"], how="left")
        merged["top_groups"] = merged.apply(_format_top_groups_row, axis=1)

        if "score_total" in merged.columns:
            merged["score_gap"] = (merged["score_total"] - merged["score_total_calc"]).astype(float)

        cols = ["date", "rank_total", "ticker", "score_total", "score_total_calc", "score_gap", "top_groups"]
        cols = [c for c in cols if c in merged.columns]
        return merged[cols].sort_values("rank_total").reset_index(drop=True)

    out_short = None
    out_long = None
    if horizon in ("short", "both"):
        out_short = _one("short", ranking_short_daily)
    if horizon in ("long", "both"):
        out_long = _one("long", ranking_long_daily)

    return HoldoutDayInspectResult(
        date=date_ts,
        holdout_start=hold_s,
        holdout_end=hold_e,
        short=out_short,
        long=out_long,
    )


