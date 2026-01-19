# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/backtest/l7d_stability.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


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

@dataclass(frozen=True)
class L7DConfig:
    holding_days: int = 20
    date_col: str = "date"
    phase_col: str = "phase"
    net_return_candidates: Tuple[str, ...] = ("net_return", "net_ret", "net_period_return")

def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _pick_net_return_col(bt_returns: pd.DataFrame, cfg: L7DConfig) -> str:
    for c in cfg.net_return_candidates:
        if c in bt_returns.columns:
            return c
    raise KeyError(
        f"[L7D] bt_returns must contain one of {list(cfg.net_return_candidates)}. "
        f"got={bt_returns.columns.tolist()}"
    )

def _max_drawdown_from_returns(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(np.min(dd)) if dd.size else 0.0

def build_bt_yearly_metrics(
    bt_returns: pd.DataFrame,
    *,
    holding_days: int = 20,
    date_col: str = "date",
    phase_col: str = "phase",
    net_return_col: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(bt_returns, pd.DataFrame) or bt_returns.empty:
        raise ValueError("[L7D] bt_returns must be a non-empty DataFrame.")

    cfg = L7DConfig(holding_days=holding_days, date_col=date_col, phase_col=phase_col)
    df = bt_returns.copy()

    for c in (cfg.date_col, cfg.phase_col):
        if c not in df.columns:
            raise KeyError(f"[L7D] bt_returns missing required column: {c}")

    df = _ensure_datetime(df, cfg.date_col)
    if df[cfg.date_col].isna().any():
        bad = df[df[cfg.date_col].isna()].head(5)
        raise ValueError(f"[L7D] bt_returns has non-parsable dates. sample:\n{bad}")

    used_col = net_return_col if net_return_col else _pick_net_return_col(df, cfg)
    if used_col not in df.columns:
        raise KeyError(f"[L7D] net_return_col='{used_col}' not found in bt_returns columns.")

    df[used_col] = pd.to_numeric(df[used_col], errors="coerce")
    df["year"] = df[cfg.date_col].dt.year.astype(int)

    ann_factor = math.sqrt(252.0 / float(cfg.holding_days))

    rows: List[dict] = []
    g = df.groupby([cfg.phase_col, "year"], sort=True)

    for (phase, year), d in g:
        d = d.sort_values(cfg.date_col)
        r = d[used_col].to_numpy(dtype=float)
        r = r[~np.isnan(r)]
        n = int(r.size)

        if n == 0:
            net_total_return = 0.0
            net_vol_ann = 0.0
            net_sharpe = 0.0
            net_mdd = 0.0
            net_hit_ratio = 0.0
        else:
            net_total_return = float(np.prod(1.0 + r) - 1.0)
            std = float(np.std(r, ddof=1)) if n >= 2 else 0.0
            mean = float(np.mean(r))

            net_vol_ann = std * ann_factor if std > 0 else 0.0
            net_sharpe = (mean / std) * ann_factor if std > 0 else 0.0
            net_mdd = _max_drawdown_from_returns(r)
            net_hit_ratio = float(np.mean(r > 0.0))

        rows.append(
            {
                "phase": phase,
                "year": int(year),
                "n_rebalances": int(d.shape[0]),
                "net_total_return": net_total_return,
                "net_vol_ann": net_vol_ann,
                "net_sharpe": net_sharpe,
                "net_mdd": net_mdd,
                "net_hit_ratio": net_hit_ratio,
                "date_start": pd.Timestamp(d[cfg.date_col].min()),
                "date_end": pd.Timestamp(d[cfg.date_col].max()),
                "net_return_col_used": used_col,
            }
        )

    out = pd.DataFrame(rows)
    out = out[
        [
            "phase",
            "year",
            "n_rebalances",
            "net_total_return",
            "net_vol_ann",
            "net_sharpe",
            "net_mdd",
            "net_hit_ratio",
            "date_start",
            "date_end",
            "net_return_col_used",
        ]
    ].sort_values(["phase", "year"], ignore_index=True)
    return out

def build_bt_rolling_sharpe(
    bt_returns: pd.DataFrame,
    *,
    holding_days: int = 20,
    window_rebalances: int = 12,
    net_return_col: str = "net_return",
) -> pd.DataFrame:
    if bt_returns is None or bt_returns.empty:
        return pd.DataFrame()

    if net_return_col not in bt_returns.columns:
        raise KeyError(f"[L7D] bt_returns missing '{net_return_col}'")

    df = bt_returns[["date", "phase", net_return_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["phase"] = df["phase"].astype(str)
    df[net_return_col] = pd.to_numeric(df[net_return_col], errors="coerce")
    df[net_return_col] = df[net_return_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    periods_per_year = 252.0 / float(holding_days) if holding_days > 0 else 12.6
    ann_factor = np.sqrt(periods_per_year)

    out = []
    for phase, g in df.groupby("phase", sort=False):
        s = g.sort_values("date").reset_index(drop=True)
        r = s[net_return_col].astype(float)

        roll_n = r.rolling(window_rebalances, min_periods=1).count()
        roll_mean = r.rolling(window_rebalances, min_periods=1).mean()
        roll_std = r.rolling(window_rebalances, min_periods=2).std(ddof=1)

        roll_mean = roll_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        roll_std = roll_std.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        roll_vol_ann = roll_std * ann_factor

        mean_np = roll_mean.to_numpy(dtype=float)
        std_np = roll_std.to_numpy(dtype=float)
        ratio = np.zeros_like(mean_np, dtype=float)
        np.divide(mean_np, std_np, out=ratio, where=(std_np > 0.0))
        roll_sharpe = ratio * ann_factor

        out.append(
            pd.DataFrame(
                {
                    "phase": phase,
                    "date": s["date"],
                    "net_rolling_n": roll_n.astype(int),
                    "net_rolling_mean": roll_mean.astype(float),
                    "net_rolling_vol_ann": roll_vol_ann.astype(float),
                    "net_rolling_sharpe": pd.Series(roll_sharpe, index=s.index).astype(float),
                    "net_return_col_used": net_return_col,
                }
            )
        )

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    for c in ["net_rolling_mean", "net_rolling_vol_ann", "net_rolling_sharpe"]:
        if c in res.columns:
            res[c] = pd.to_numeric(res[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keys = ["date", "phase", "net_return_col_used"]
    if not res.empty and res.duplicated(keys).any():
        raise ValueError("[L7D] bt_rolling_sharpe must be unique on (date,phase,net_return_col_used)")

    return res

def build_drawdown_events(
    bt_equity_curve: pd.DataFrame,
    *,
    date_col: str = "date",
    phase_col: str = "phase",
    equity_col: str = "equity",
) -> pd.DataFrame:
    if bt_equity_curve is None or bt_equity_curve.empty:
        return pd.DataFrame(
            columns=["phase", "peak_date", "trough_date", "drawdown", "length_days", "peak_equity", "trough_equity"]
        )

    df = bt_equity_curve.copy()
    for c in [date_col, phase_col, equity_col]:
        if c not in df.columns:
            raise KeyError(f"[L7D] bt_equity_curve missing required column: {c}")

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df[phase_col] = df[phase_col].astype(str)
    df[equity_col] = pd.to_numeric(df[equity_col], errors="coerce")

    rows: List[dict] = []

    for phase, g in df.groupby(phase_col, sort=False):
        s = g.sort_values(date_col).reset_index(drop=True)
        eq = s[equity_col].to_numpy(dtype=float)
        dt = s[date_col].to_numpy()

        if len(eq) == 0:
            continue

        running_max = np.maximum.accumulate(eq)
        is_peak = eq == running_max

        peak_idx = np.where(is_peak)[0]
        if len(peak_idx) == 0:
            continue

        for i, p_idx in enumerate(peak_idx):
            start = p_idx
            end = peak_idx[i + 1] if i + 1 < len(peak_idx) else len(eq)

            seg_eq = eq[start:end]
            seg_dt = dt[start:end]
            if len(seg_eq) == 0:
                continue

            trough_local = int(np.argmin(seg_eq))
            trough_idx = start + trough_local

            peak_equity = float(eq[p_idx])
            trough_equity = float(eq[trough_idx])
            dd = (trough_equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0

            peak_date = pd.Timestamp(dt[p_idx])
            trough_date = pd.Timestamp(dt[trough_idx])
            length_days = int((trough_date - peak_date).days)

            rows.append(
                {
                    "phase": phase,
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "drawdown": float(dd),
                    "length_days": int(length_days),
                    "peak_equity": float(peak_equity),
                    "trough_equity": float(trough_equity),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["phase", "drawdown"]).reset_index(drop=True)
    return out

def run_l7d_stability_from_artifacts(
    *,
    bt_returns: pd.DataFrame,
    bt_equity_curve: Optional[pd.DataFrame] = None,
    holding_days: int = 20,
    cfg: Optional[dict] = None,
) -> Tuple[dict, List[str]]:
    """
    run_all.py에서 호출하는 표준 엔트리포인트.
    output artifacts:
      - bt_yearly_metrics
      - bt_rolling_sharpe
      - bt_drawdown_events

    + alias:
      - l7d_stability (yearly)
      - _l7d_stability (rolling)
    """
    warns: List[str] = []

    l7d = _resolve_section(cfg or {}, "l7d")
    window_rebalances = int(l7d.get("rolling_window_rebalances", 12))
    want_col = str(l7d.get("net_return_col", "net_return"))

    # ✅ 컬럼이 없으면 후보 컬럼으로 자동 폴백
    cfg_local = L7DConfig(holding_days=holding_days)
    net_col_used = want_col
    if want_col not in bt_returns.columns:
        net_col_used = _pick_net_return_col(bt_returns, cfg_local)
        warns.append(f"[L7D] net_return_col '{want_col}' not found. fallback to '{net_col_used}'")

    yearly = build_bt_yearly_metrics(bt_returns, holding_days=holding_days, net_return_col=net_col_used)
    rolling = build_bt_rolling_sharpe(
        bt_returns,
        holding_days=holding_days,
        window_rebalances=window_rebalances,
        net_return_col=net_col_used,
    )

    if bt_equity_curve is not None and not bt_equity_curve.empty and "equity" in bt_equity_curve.columns:
        dd = build_drawdown_events(bt_equity_curve)
    else:
        dd = build_drawdown_events(pd.DataFrame(columns=["date", "phase", "equity"]))

    quality_df = pd.DataFrame([{
        "holding_days": int(holding_days),
        "rolling_window_rebalances": int(window_rebalances),
        "net_return_col_requested": want_col,
        "net_return_col_used": net_col_used,
        "rows_yearly": int(len(yearly)),
        "rows_rolling": int(len(rolling)),
        "rows_drawdown_events": int(len(dd)),
    }])

    return {
        "bt_yearly_metrics": yearly,
        "bt_rolling_sharpe": rolling,
        "bt_drawdown_events": dd,
        "_l7d_quality": quality_df,

        # ✅ 네 확인 스크립트가 찾는 이름으로 alias 제공
        "l7d_stability": yearly,
        "_l7d_stability": rolling,
    }, (warns or [])
