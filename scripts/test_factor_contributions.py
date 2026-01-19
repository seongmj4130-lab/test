# -*- coding: utf-8 -*-
# [개선안 36번] 팩터셋(그룹) 기여도 계산 최소 테스트 (합=score_total_calc 검증)
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.ranking.contribution_engine import (
    ContributionConfig,
    compute_group_contributions_for_day,
)


def main():
    date = pd.Timestamp("2024-06-03")
    df = pd.DataFrame(
        {
            "date": [date, date, date],
            "ticker": ["000001", "000002", "000003"],
            "in_universe": [True, True, True],
            # 3개 피처: 2개는 technical, 1개는 profitability로 분류되도록 이름을 구성
            "volatility_20d": [0.1, 0.2, 0.3],
            "price_momentum_20d": [1.0, 0.0, -1.0],
            "roe": [0.05, 0.10, 0.20],
        }
    )

    feature_weights = {
        "volatility_20d": 0.2,
        "price_momentum_20d": 0.3,
        "roe": 0.5,
    }

    cfg = ContributionConfig(normalization_method="zscore")
    contrib = compute_group_contributions_for_day(df, feature_weights=feature_weights, group_map=None, cfg=cfg)

    # group_contrib__* 합 == score_total_calc
    group_cols = [c for c in contrib.columns if c.startswith("group_contrib__")]
    recon = contrib[group_cols].sum(axis=1).to_numpy()
    score = contrib["score_total_calc"].to_numpy()

    if not np.allclose(recon, score, atol=1e-10):
        raise AssertionError(f"sum(group_contrib)!=score_total_calc. max_gap={np.max(np.abs(recon-score))}")

    print("[OK] group contribution sum matches score_total_calc")


if __name__ == "__main__":
    main()
