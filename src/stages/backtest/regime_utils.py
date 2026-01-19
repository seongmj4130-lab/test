# -*- coding: utf-8 -*-
"""
[개선안 22번] 시장 국면(regime) 유틸리티

목표:
  - 연구용 5단계(bull_strong/bull_weak/neutral/bear_weak/bear_strong)를
    최종 제출/운영용 3단계(bull/neutral/bear)로 단순화한다.
  - lookback_return_pct 기반 임계값 방식도 지원한다.
"""

from __future__ import annotations

import pandas as pd


def map_regime_to_3level(regime: str) -> str:
    """
    [개선안 22번] 5단계/기타 라벨을 3단계(bull/neutral/bear)로 매핑
    """
    r = (regime or "").strip().lower()
    if r in ("bull", "bull_strong", "bull_weak"):
        return "bull"
    if r in ("bear", "bear_strong", "bear_weak"):
        return "bear"
    return "neutral"


def apply_3level_regime(
    market_regime_df: pd.DataFrame,
    *,
    regime_col: str = "regime",
    out_col: str = "regime_3",
) -> pd.DataFrame:
    """
    [개선안 22번] market_regime(df)에 regime_3 컬럼을 추가
    """
    if market_regime_df is None or market_regime_df.empty:
        return market_regime_df
    if regime_col not in market_regime_df.columns:
        raise KeyError(f"market_regime_df missing column: {regime_col}")
    out = market_regime_df.copy()
    out[out_col] = out[regime_col].astype(str).map(map_regime_to_3level)
    return out


def classify_3level_from_return(
    lookback_return_pct: float,
    *,
    neutral_band_pct: float = 0.0,
) -> str:
    """
    [개선안 22번] lookback_return_pct 하나로 3단계 국면을 분류

    Args:
        lookback_return_pct: 과거 lookback 기간 수익률(%)
        neutral_band_pct: 중립 밴드(%) 예: 1.0이면 [-1%, +1%]는 neutral
    """
    try:
        x = float(lookback_return_pct)
    except Exception:
        return "neutral"
    b = float(neutral_band_pct)
    if x > b:
        return "bull"
    if x < -b:
        return "bear"
    return "neutral"
