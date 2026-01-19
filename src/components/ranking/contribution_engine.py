# -*- coding: utf-8 -*-
# [개선안 36번] Track A Holdout 단일 일자 랭킹 Top10 + 팩터셋(그룹) 기여도 Top3 분석 유틸
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.components.ranking.score_engine import normalize_feature_cross_sectional
from src.utils.feature_groups import get_feature_groups, load_feature_groups


@dataclass(frozen=True)
class ContributionConfig:
    """
    [개선안 36번] 하루치(단일 date)에서 score_total을 팩터셋(그룹) 기여도로 분해하기 위한 설정
    """

    normalization_method: str  # "percentile" | "zscore" | "robust_zscore"
    date_col: str = "date"
    ticker_col: str = "ticker"
    universe_col: str = "in_universe"
    sector_col: Optional[str] = None
    use_sector_relative: bool = False


def _zfill_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def infer_feature_group(feature: str) -> str:
    """
    [개선안 36번] feature_groups.yaml이 없을 때 fallback용 그룹 추론.
    (현재 프로젝트의 group_weights: value/profitability/technical/news/other 기준)
    """
    f = str(feature).lower()
    if f.startswith("news_"):
        return "news"
    if f in {"roe"}:
        return "profitability"
    if f in {"net_income"}:
        return "profitability"
    if f in {"equity", "total_liabilities", "debt_ratio"}:
        return "value"
    if "esg" in f or f.endswith("_score") or f in {"environmental_score", "social_score", "governance_score"}:
        return "other"
    if (
        "momentum" in f
        or "volatility" in f
        or "drawdown" in f
        or f in {"turnover", "volume_ratio", "ret_daily"}
    ):
        return "technical"
    return "other"


def load_group_map_from_yaml(config_path: Path) -> Dict[str, List[str]]:
    """
    [개선안 36번] feature_groups*.yaml에서 {group -> features} 로드
    """
    cfg = load_feature_groups(config_path)
    return get_feature_groups(cfg)


def compute_group_contributions_for_day(
    day_df: pd.DataFrame,
    *,
    feature_weights: Dict[str, float],
    group_map: Optional[Dict[str, List[str]]] = None,
    cfg: Optional[ContributionConfig] = None,
) -> pd.DataFrame:
    """
    [개선안 36번] 단일 date(day_df)에서 score_total을 그룹별 기여도로 분해.

    Args:
        day_df: 단일 date만 포함한 DataFrame (date, ticker, feature cols 포함)
        feature_weights: {feature -> weight}. 실제 score_total 계산에 사용될 피처/가중치
        group_map: {group -> [features...]}. 없으면 infer_feature_group()로 feature->group 추정
        cfg: ContributionConfig

    Returns:
        DataFrame:
        - date, ticker, in_universe(있으면)
        - score_total_calc
        - group_contrib__<group> 컬럼들
    """
    if cfg is None:
        cfg = ContributionConfig(normalization_method="percentile")

    if cfg.date_col not in day_df.columns:
        raise KeyError(f"Missing date col: {cfg.date_col}")
    if cfg.ticker_col not in day_df.columns:
        raise KeyError(f"Missing ticker col: {cfg.ticker_col}")

    out = day_df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="raise")
    out[cfg.ticker_col] = _zfill_ticker(out[cfg.ticker_col])

    unique_dates = out[cfg.date_col].dropna().unique()
    if len(unique_dates) != 1:
        raise ValueError(f"day_df must contain exactly one date. got={len(unique_dates)} dates")

    # 피처 컬럼은 '가중치 파일에 존재' AND 'day_df에 존재' 조건으로 결정
    feature_cols = [f for f in feature_weights.keys() if f in out.columns]
    if len(feature_cols) == 0:
        raise ValueError("No features from feature_weights exist in day_df columns.")

    # 정규화 (cross-sectional). sector-relative 옵션은 지금 데이터에 sector가 없어서 보통 False.
    actual_sector_col = None
    if cfg.use_sector_relative and cfg.sector_col and cfg.sector_col in out.columns:
        if out[cfg.sector_col].notna().sum() > 0:
            actual_sector_col = cfg.sector_col

    normalized: Dict[str, pd.Series] = {}
    for feat in feature_cols:
        normalized[feat] = normalize_feature_cross_sectional(
            out,
            feature_col=feat,
            date_col=cfg.date_col,
            method=cfg.normalization_method,  # type: ignore[arg-type]
            sector_col=actual_sector_col,
        ).fillna(0.0)

    # 그룹 매핑 준비
    feature_to_group: Dict[str, str] = {}
    if group_map:
        for g, feats in group_map.items():
            for f in feats:
                feature_to_group[str(f)] = str(g)

    groups: List[str] = []
    for feat in feature_cols:
        g = feature_to_group.get(feat) if feature_to_group else None
        if g is None:
            g = infer_feature_group(feat)
        if g not in groups:
            groups.append(g)

    group_contrib = {g: pd.Series(0.0, index=out.index) for g in groups}
    score_total = pd.Series(0.0, index=out.index)

    for feat in feature_cols:
        w = float(feature_weights.get(feat, 0.0))
        contrib_feat = w * normalized[feat]
        score_total += contrib_feat

        g = feature_to_group.get(feat) if feature_to_group else None
        if g is None:
            g = infer_feature_group(feat)
        group_contrib[g] = group_contrib[g] + contrib_feat

    res_cols = [cfg.date_col, cfg.ticker_col]
    if cfg.universe_col in out.columns:
        res_cols.append(cfg.universe_col)

    result = out[res_cols].copy()
    result["score_total_calc"] = score_total.astype(float)

    for g in groups:
        result[f"group_contrib__{g}"] = group_contrib[g].astype(float)

    return result


def pick_top_groups_per_row(
    contrib_df: pd.DataFrame,
    *,
    top_n: int = 3,
    prefix: str = "group_contrib__",
) -> pd.DataFrame:
    """
    [개선안 36번] 각 row(=ticker)별로 그룹 기여도 TopN(절대값 기준)을 뽑아
    group_top1_name/group_top1_value ... 형태로 반환.
    """
    group_cols = [c for c in contrib_df.columns if c.startswith(prefix)]
    if len(group_cols) == 0:
        raise ValueError(f"No group contribution columns found with prefix={prefix}")

    out = contrib_df.copy()

    values = out[group_cols].to_numpy(dtype=float)
    abs_values = np.abs(values)
    # 내림차순 top_n 인덱스
    top_idx = np.argsort(-abs_values, axis=1)[:, :top_n]

    for k in range(top_n):
        col_idx = top_idx[:, k]
        names = [group_cols[i].replace(prefix, "") for i in col_idx]
        vals = values[np.arange(values.shape[0]), col_idx]
        out[f"group_top{k+1}_name"] = names
        out[f"group_top{k+1}_value"] = vals

    return out
