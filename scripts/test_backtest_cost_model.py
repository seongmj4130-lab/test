# -*- coding: utf-8 -*-
"""
[개선안 1번][개선안 3번] L7 거래비용/슬리피지(턴오버 기반) 계산 테스트

목표:
- "리밸런싱 발생 시 고정 10bp 차감" 같은 과다 차감이 다시 들어오지 않도록 방지
- turnover_oneway, exposure, (cost_bps + slippage_bps) 조합이 정확히 total_cost로 반영되는지 확인

실행:
  cd C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code
  python scripts/test_backtest_cost_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 경로에 추가 (scripts/*.py 실행 시 src import 보장)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest


def _approx(a: float, b: float, tol: float = 1e-12) -> None:
    assert abs(float(a) - float(b)) <= tol, f"not close: a={a}, b={b}, tol={tol}"


def test_cost_model_turnover_based() -> None:
    # Given: 2 리밸런싱 날짜, 2~3 종목, score로 top_k=2 선택
    # date1: A,B 선택 (prev empty)
    # date2: A,C 선택 (B -> C 교체)
    rows = [
        {"date": "2024-01-02", "phase": "dev", "ticker": "A", "score_ens": 2.0, "true_short": 0.10},
        {"date": "2024-01-02", "phase": "dev", "ticker": "B", "score_ens": 1.0, "true_short": 0.00},
        {"date": "2024-01-02", "phase": "dev", "ticker": "C", "score_ens": 0.0, "true_short": 0.00},
        {"date": "2024-01-22", "phase": "dev", "ticker": "A", "score_ens": 2.0, "true_short": 0.00},
        {"date": "2024-01-22", "phase": "dev", "ticker": "B", "score_ens": 0.0, "true_short": 0.00},
        {"date": "2024-01-22", "phase": "dev", "ticker": "C", "score_ens": 1.0, "true_short": 0.00},
    ]
    rebalance_scores = pd.DataFrame(rows)

    cfg = BacktestConfig(
        holding_days=20,
        top_k=2,
        cost_bps=10.0,
        slippage_bps=5.0,  # total 15bp
        score_col="score_ens",
        ret_col="true_short",
        weighting="equal",
        buffer_k=0,
        rebalance_interval=1,
        smart_buffer_enabled=False,
        volatility_adjustment_enabled=False,
        risk_scaling_enabled=False,
    )

    bt_pos, bt_ret, bt_eq, bt_met, quality, warns, *_ = run_backtest(
        rebalance_scores=rebalance_scores,
        cfg=cfg,
        config_cost_bps=10.0,
        market_regime=None,
    )

    # When/Then: 비용이 traded_value(=turnover*|exposure|) * (10+5)bp 로 계산되는지 검증
    # date1:
    # - prev empty -> new weights (0.5,0.5) => one-way turnover = 0.5 * (0.5+0.5) = 0.5
    # - traded_value=0.5 (exposure=1)
    # - total_cost=0.5 * 15/10000 = 0.00075
    r1 = bt_ret.sort_values("date").iloc[0]
    _approx(r1["turnover_oneway"], 0.5, 1e-12)
    _approx(r1["traded_value"], 0.5, 1e-12)
    _approx(r1["total_cost"], 0.00075, 1e-12)
    _approx(r1["slippage_cost"], 0.5 * 5.0 / 10000.0, 1e-12)
    _approx(r1["turnover_cost"], 0.5 * 10.0 / 10000.0, 1e-12)

    # date2:
    # - weights change: from (A=0.5,B=0.5) to (A=0.5,C=0.5)
    # - delta: B 0.5->0, C 0->0.5 => sum abs=1.0 => one-way turnover=0.5
    r2 = bt_ret.sort_values("date").iloc[1]
    _approx(r2["turnover_oneway"], 0.5, 1e-12)
    _approx(r2["traded_value"], 0.5, 1e-12)
    _approx(r2["total_cost"], 0.00075, 1e-12)

    assert not warns, f"unexpected warns: {warns}"


if __name__ == "__main__":
    test_cost_model_turnover_based()
    print("✅ PASS: turnover 기반 거래비용/슬리피지 계산이 기대값과 일치합니다.")


