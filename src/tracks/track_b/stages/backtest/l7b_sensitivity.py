# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/backtest/l7b_sensitivity.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

from src.stages.backtest.l7_backtest import BacktestConfig, run_backtest

def _resolve_section(cfg: dict, name: str) -> dict:
    """config.yaml이 params 아래에 섹션을 두는 구조를 지원한다."""
    if not isinstance(cfg, dict):
        return {}
    params = cfg.get("params", {}) or {}
    if isinstance(params, dict):
        sec = params.get(name, None)
        if isinstance(sec, dict):
            return sec
    sec2 = cfg.get(name, None)
    return sec2 if isinstance(sec2, dict) else {}

def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _parse_int_list(x: Any) -> list[int]:
    items = _as_list(x)
    out: list[int] = []
    for v in items:
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            # "0,10,20" 같은 형태 지원
            if "," in s:
                for tok in s.split(","):
                    tok = tok.strip()
                    if tok:
                        out.append(int(float(tok)))
                continue
            out.append(int(float(s)))
        else:
            out.append(int(float(v)))
    # unique + sort
    out = sorted(set(out))
    return out

def _parse_float_list(x: Any) -> list[float]:
    items = _as_list(x)
    out: list[float] = []
    for v in items:
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            if "," in s:
                for tok in s.split(","):
                    tok = tok.strip()
                    if tok:
                        out.append(float(tok))
                continue
            out.append(float(s))
        else:
            out.append(float(v))
    out = sorted(set(out))
    return out

def _parse_str_list(x: Any) -> list[str]:
    items = _as_list(x)
    out: list[str] = []
    for v in items:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    # unique preserve order
    seen = set()
    res = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        res.append(s)
    return res

def run_sensitivity(
    rebalance_scores: pd.DataFrame,
    *,
    holding_days: int,
    top_k_grid: List[int],
    cost_bps_grid: List[float],
    weighting_grid: List[str],
    score_col: str,
    ret_col: str,
    buffer_k_grid: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    warns: List[str] = []
    rows = []

    if buffer_k_grid is None or len(buffer_k_grid) == 0:
        buffer_k_grid = [0]

    for w in weighting_grid:
        for k in top_k_grid:
            for c in cost_bps_grid:
                for bk in buffer_k_grid:
                    cfg_bt = BacktestConfig(
                        holding_days=int(holding_days),
                        top_k=int(k),
                        cost_bps=float(c),
                        score_col=str(score_col),
                        ret_col=str(ret_col),
                        weighting=str(w),
                        buffer_k=int(bk),
                    )
                    _, _, _, met, _, wns, _, _, _ = run_backtest(rebalance_scores, cfg_bt)
                    warns.extend(wns or [])
                    met = met.copy()
                    met["grid_top_k"] = int(k)
                    met["grid_cost_bps"] = float(c)
                    met["grid_weighting"] = str(w)
                    met["grid_buffer_k"] = int(bk)
                    rows.append(met)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    quality = {
        "holding_days": int(holding_days),
        "top_k_grid": list(map(int, top_k_grid)),
        "cost_bps_grid": list(map(float, cost_bps_grid)),
        "weighting_grid": list(map(str, weighting_grid)),
        "buffer_k_grid": list(map(int, buffer_k_grid)),
        "n_runs": int(len(top_k_grid) * len(cost_bps_grid) * len(weighting_grid) * len(buffer_k_grid)),
        "rows_expected": int(2 * len(top_k_grid) * len(cost_bps_grid) * len(weighting_grid) * len(buffer_k_grid)),
    }
    quality_df = pd.DataFrame([quality])

    return out, quality_df, warns

def run_l7b_sensitivity(
    *,
    rebalance_scores: pd.DataFrame,
    cfg: dict,
) -> Tuple[dict, List[str]]:
    """
    run_all.py에서 호출하는 표준 엔트리포인트.
    output artifact: bt_sensitivity
    """
    l7 = _resolve_section(cfg, "l7")
    l7b = _resolve_section(cfg, "l7b")

    holding_days = int(l7.get("holding_days", 20))
    score_col = str(l7.get("score_col", "score_ens"))
    ret_col = str(l7.get("ret_col", l7.get("return_col", "true_short")))

    top_k_grid = _parse_int_list(l7b.get("top_k_grid", [10, 20, 30]))
    cost_bps_grid = _parse_float_list(l7b.get("cost_bps_grid", [0.0, 10.0, 20.0]))
    weighting_grid = _parse_str_list(l7b.get("weighting_grid", ["equal"]))

    # ✅ buffer grid: l7b가 없거나 비면 l7.buffer_k로 보정(너 케이스에서 즉시 효과)
    raw_bk = l7b.get("buffer_k_grid", None)
    buffer_k_grid = _parse_int_list(raw_bk)

    if len(buffer_k_grid) == 0:
        fallback_bk = int(l7.get("buffer_k", 0))
        buffer_k_grid = [fallback_bk] if fallback_bk != 0 else [0]

    # 그래도 혹시 l7.buffer_k(예:20)가 있고 grid에 없으면 0과 함께 넣어줌(실험 비교용)
    l7_bk = int(l7.get("buffer_k", 0))
    if l7_bk != 0 and l7_bk not in buffer_k_grid:
        buffer_k_grid = sorted(set(buffer_k_grid + [0, l7_bk]))

    df, quality_df, warns = run_sensitivity(
        rebalance_scores,
        holding_days=holding_days,
        top_k_grid=top_k_grid,
        cost_bps_grid=cost_bps_grid,
        weighting_grid=weighting_grid,
        score_col=score_col,
        ret_col=ret_col,
        buffer_k_grid=buffer_k_grid,
    )

    # ✅ _l7b_quality를 DataFrame으로 반환(검증/저장 충돌 방지)
    return {"bt_sensitivity": df, "_l7b_quality": quality_df}, (warns or [])
