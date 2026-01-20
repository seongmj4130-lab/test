# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/modeling/l6_scoring.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"missing column: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="raise")


def _ensure_ticker(df: pd.DataFrame) -> None:
    if "ticker" not in df.columns:
        raise KeyError("missing column: ticker")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name} missing columns: {miss}. got={list(df.columns)}")


def _add_ym(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """date -> ym('YYYY-MM')"""
    out = df.copy()
    _ensure_datetime(out, date_col)
    out["ym"] = out[date_col].dt.to_period("M").astype(str)
    return out


def _attach_in_universe_monthly(
    *,
    df: pd.DataFrame,
    universe_monthly: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    universe_monthly: 최소 컬럼 (date,ticker) 또는 (ym,ticker)를 가진 월별 멤버십 테이블
    반환:
      - df_out: df에 in_universe(bool), ym 추가
      - univ_counts: ym별 멤버 수 (ym, universe_n_tickers)
    """
    if universe_monthly is None or len(universe_monthly) == 0:
        raise ValueError("universe_k200_membership_monthly is empty")

    un = universe_monthly.copy()

    if "ym" not in un.columns:
        if "date" not in un.columns:
            raise KeyError("universe_monthly must have 'date' or 'ym' column")
        un = _add_ym(un, "date")
    else:
        un["ym"] = un["ym"].astype(str)

    if "ticker" not in un.columns:
        raise KeyError("universe_monthly missing column: ticker")

    _ensure_ticker(un)
    un_key = un[["ym", "ticker"]].drop_duplicates()

    out = _add_ym(df, date_col)
    if ticker_col != "ticker":
        out = out.rename(columns={ticker_col: "ticker"})
    _ensure_ticker(out)

    merged = out.merge(un_key, on=["ym", "ticker"], how="left", indicator=True)
    merged["in_universe"] = merged["_merge"].eq("both")
    merged.drop(columns=["_merge"], inplace=True)

    univ_counts = (
        un_key.groupby("ym", sort=False)["ticker"]
        .nunique()
        .rename("universe_n_tickers")
        .reset_index()
    )
    return merged, univ_counts


def _agg_across_models(df: pd.DataFrame, score_col: str = "y_pred") -> pd.DataFrame:
    """
    (date,ticker,fold_id,phase,horizon) 단위로 모델별 예측을 평균내서 단일 score로 만든다.
    """
    gcols = ["date", "ticker", "fold_id", "phase", "horizon"]
    keep_true = "y_true" in df.columns
    keep_univ = "in_universe" in df.columns

    agg: dict[str, object] = {score_col: "mean"}
    if keep_true:
        agg["y_true"] = "mean"
    if keep_univ:
        agg["in_universe"] = "max"

    out = (
        df.groupby(gcols, sort=False, as_index=False)
        .agg(agg)
        .rename(columns={score_col: "score", "y_true": "true"})
    )
    return out


def _pick_rebalance_rows_by_fold_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    fold별 test window 마지막 날짜(=fold 내 max(date))만 남긴다.
    결과: (fold_id, phase, horizon, ticker)당 1행
    """
    end = (
        df.groupby(["fold_id", "phase", "horizon"], sort=False)["date"]
        .max()
        .rename("rebalance_date")
        .reset_index()
    )
    out = df.merge(
        end, on=["fold_id", "phase", "horizon"], how="inner", validate="many_to_one"
    )
    out = out.loc[out["date"] == out["rebalance_date"]].copy()
    out.drop(columns=["date"], inplace=True)
    out.rename(columns={"rebalance_date": "date"}, inplace=True)
    return out


def _rank_within_date(df: pd.DataFrame, col: str, suffix: str) -> pd.DataFrame:
    df[f"rank_{suffix}"] = df.groupby(["date", "phase"], sort=False)[col].rank(
        ascending=False, method="first"
    )
    df[f"pct_{suffix}"] = df.groupby(["date", "phase"], sort=False)[col].rank(
        pct=True, ascending=False, method="first"
    )
    return df


def build_rebalance_scores(
    *,
    pred_short_oos: pd.DataFrame,
    pred_long_oos: pd.DataFrame,
    universe_k200_membership_monthly: pd.DataFrame | None = None,
    weight_short: float = 0.5,
    weight_long: float = 0.5,
    universe_gate_warn_below: float = 0.95,
    # [Stage 4] sector_name carry를 위한 추가 파라미터
    dataset_daily: pd.DataFrame | None = None,
    # [예측력 개선] Score 부호 반전 옵션
    invert_score_sign: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """
    L6:
    - L5 OOS 예측을 리밸런싱(date) 단위 스코어로 축약
    - fold별 test window 마지막 날짜를 리밸런싱 기준일로 사용
    - 월별 KOSPI200 유니버스 강제 적용
    """
    warns: list[str] = []

    req = ["date", "ticker", "y_pred", "fold_id", "phase", "horizon"]
    _require_cols(pred_short_oos, req, "pred_short_oos")
    _require_cols(pred_long_oos, req, "pred_long_oos")

    ps = pred_short_oos.copy()
    pl = pred_long_oos.copy()

    _ensure_datetime(ps, "date")
    _ensure_datetime(pl, "date")
    _ensure_ticker(ps)
    _ensure_ticker(pl)

    ps1 = _agg_across_models(ps, score_col="y_pred")
    pl1 = _agg_across_models(pl, score_col="y_pred")

    ps2 = _pick_rebalance_rows_by_fold_end(ps1)
    pl2 = _pick_rebalance_rows_by_fold_end(pl1)

    ps2 = ps2.rename(columns={"score": "score_short", "true": "true_short"})
    pl2 = pl2.rename(columns={"score": "score_long", "true": "true_long"})

    key = ["date", "ticker", "phase"]
    out = ps2.merge(pl2, on=key, how="outer", validate="one_to_one")

    # [Stage 4] sector_name carry (dataset_daily에서 가져오기)
    if dataset_daily is not None and "sector_name" in dataset_daily.columns:
        # dataset_daily에서 (date, ticker) 기준으로 sector_name 가져오기
        sector_info = dataset_daily[["date", "ticker", "sector_name"]].drop_duplicates(
            ["date", "ticker"]
        )
        out = out.merge(
            sector_info, on=["date", "ticker"], how="left", validate="many_to_one"
        )
        warns.append("[L6 Stage4] sector_name carried from dataset_daily")
    else:
        warns.append(
            "[L6 Stage4] sector_name not found in dataset_daily -> not carried"
        )

    w_s = float(weight_short)
    w_l = float(weight_long)

    s_s = out["score_short"]
    s_l = out["score_long"]

    mask_s = s_s.notna()
    mask_l = s_l.notna()

    den = (w_s * mask_s.astype(float)) + (w_l * mask_l.astype(float))
    num = (w_s * s_s.fillna(0.0)) + (w_l * s_l.fillna(0.0))

    # 둘 다 없으면 NaN 유지, 하나라도 있으면 있는 쪽으로 스코어 생성
    out["score_ens"] = num / den.replace(0.0, np.nan)

    # ---------- Universe attach + filter ----------
    universe_mode = "none"
    universe_ratio_before = np.nan
    dropped_rows = 0
    univ_counts = None

    if (
        universe_k200_membership_monthly is not None
        and len(universe_k200_membership_monthly) > 0
    ):
        universe_mode = "monthly_table"
        out2, univ_counts = _attach_in_universe_monthly(
            df=out,
            universe_monthly=universe_k200_membership_monthly,
            date_col="date",
            ticker_col="ticker",
        )
        universe_ratio_before = (
            float(out2["in_universe"].mean()) if len(out2) else np.nan
        )
        dropped_rows = int((~out2["in_universe"]).sum())
        out = out2.loc[out2["in_universe"]].copy()
    else:
        if "in_universe" not in out.columns:
            raise ValueError(
                "Universe enforcement failed: provide universe_k200_membership_monthly "
                "or ensure 'in_universe' exists in pred_*_oos and is preserved into L6."
            )
        universe_mode = "from_pred"
        out["in_universe"] = out["in_universe"].astype(bool)
        universe_ratio_before = float(out["in_universe"].mean()) if len(out) else np.nan
        dropped_rows = int((~out["in_universe"]).sum())
        out = out.loc[out["in_universe"]].copy()

    if (
        np.isfinite(universe_ratio_before)
        and universe_ratio_before < universe_gate_warn_below
    ):
        warns.append(
            f"[L6] Universe coverage is low BEFORE filtering: {universe_ratio_before:.1%}. "
            f"(dropped_rows={dropped_rows}) Check universe_k200_membership_monthly completeness/mapping."
        )

    # [예측력 개선] Score 부호 반전 옵션 (음의 상관관계 해결)
    # Score와 실제 수익률이 음의 상관관계를 보일 때 부호 반전
    # 이는 임시 해결책이며, 근본적으로는 모델 예측력 개선이 필요함
    if invert_score_sign:
        out["score_ens"] = -out["score_ens"]
        out["score_short"] = -out["score_short"]
        out["score_long"] = -out["score_long"]
        warns.append("[L6 예측력 개선] Score 부호 반전 적용됨 (음의 상관관계 해결)")

    # [UI 개선] 단기/장기/통합 스코어 각각의 랭킹 계산
    # 단기 스코어 랭킹 (NaN 값은 제외하고 랭킹 계산)
    if "score_short" in out.columns:
        out = _rank_within_date(out, "score_short", "short")
    # 장기 스코어 랭킹 (NaN 값은 제외하고 랭킹 계산)
    if "score_long" in out.columns:
        out = _rank_within_date(out, "score_long", "long")
    # 통합 스코어 랭킹
    out = _rank_within_date(out, "score_ens", "ens")

    dup = int(out.duplicated(subset=key).sum())
    if dup != 0:
        raise ValueError(f"rebalance_scores duplicate keys(date,ticker,phase)={dup}")

    uni_tickers = int(out["ticker"].nunique())
    summary = (
        out.groupby(["date", "phase"], sort=False)
        .agg(
            n_tickers=("ticker", "nunique"),
            score_ens_mean=("score_ens", "mean"),
            score_ens_std=("score_ens", "std"),
            score_short_missing=("score_short", lambda s: float(s.isna().mean())),
            score_long_missing=("score_long", lambda s: float(s.isna().mean())),
            score_ens_missing=("score_ens", lambda s: float(s.isna().mean())),
        )
        .reset_index()
    )
    summary["coverage_ticker_pct"] = summary["n_tickers"] / max(uni_tickers, 1) * 100.0

    if univ_counts is not None and len(univ_counts) > 0:
        summary2 = _add_ym(summary, "date")
        summary2 = summary2.merge(univ_counts, on="ym", how="left")
        summary2["coverage_vs_universe_pct"] = (
            summary2["n_tickers"] / summary2["universe_n_tickers"] * 100.0
        )
        summary = summary2

    quality = {
        "scoring": {
            "rows": int(len(out)),
            "unique_tickers": int(uni_tickers),
            "unique_dates": int(out["date"].nunique()),
            "phases": sorted(out["phase"].dropna().unique().tolist()),
            "avg_coverage_ticker_pct": float(
                round(summary["coverage_ticker_pct"].mean(), 4)
            ),
            "avg_score_short_missing_pct": float(
                round(summary["score_short_missing"].mean() * 100.0, 6)
            ),
            "avg_score_long_missing_pct": float(
                round(summary["score_long_missing"].mean() * 100.0, 6)
            ),
            "avg_score_ens_missing_pct": float(
                round(summary["score_ens_missing"].mean() * 100.0, 6)
            ),
            "weights": {"short": float(weight_short), "long": float(weight_long)},
        },
        "universe": {
            "mode": universe_mode,
            "coverage_before_filter": (
                None
                if not np.isfinite(universe_ratio_before)
                else float(universe_ratio_before)
            ),
            "dropped_rows": int(dropped_rows),
        },
    }

    return out, summary, quality, warns
