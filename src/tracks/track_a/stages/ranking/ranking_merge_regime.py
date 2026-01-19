# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/ranking/ranking_merge_regime.py
"""
[Stage10] ranking_daily에 시장 국면(regime) 조인
"""
from __future__ import annotations

import pandas as pd


def merge_regime_to_ranking(
    ranking_daily: pd.DataFrame,
    market_regime_daily: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    ranking_daily에 market_regime_daily를 date 기준으로 조인

    Args:
        ranking_daily: Stage9의 ranking_daily (date, ticker, score_total, rank_total 등 포함)
        market_regime_daily: market_regime_daily (date, regime_score, regime_label 포함)
        date_col: 날짜 컬럼명

    Returns:
        regime 컬럼이 추가된 ranking_daily
    """
    result = ranking_daily.copy()

    # date 컬럼 정규화
    result[date_col] = pd.to_datetime(result[date_col])
    market_regime_daily[date_col] = pd.to_datetime(market_regime_daily[date_col])

    # date 기준 조인
    regime_cols = ["regime_score", "regime_label"]
    available_regime_cols = [c for c in regime_cols if c in market_regime_daily.columns]

    if len(available_regime_cols) == 0:
        raise ValueError(
            "market_regime_daily에 regime_score 또는 regime_label 컬럼이 없습니다."
        )

    merge_df = market_regime_daily[[date_col] + available_regime_cols].copy()
    result = result.merge(merge_df, on=date_col, how="left")

    # 조인 실패한 날짜 확인
    missing_dates = result[result["regime_score"].isna()][date_col].unique()
    if len(missing_dates) > 0:
        print(f"[WARN] {len(missing_dates)}개 날짜에 regime 데이터가 없습니다.")

    return result
