# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/validate_l6_outputs.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]

def _fail(msg: str) -> None:
    raise SystemExit(msg)

def _ensure_datetime(s: pd.Series, name: str) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="raise")
    return s

def _read_meta(interim: Path, artifact_name: str) -> Dict[str, Any]:
    meta_path = interim / f"{artifact_name}__meta.json"
    if not meta_path.exists():
        _fail(f"[FAIL] meta file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _get_l6_weights(cfg: dict) -> Tuple[float, float]:
    p = cfg.get("params", {}) or {}
    l6 = p.get("l6", {})
    if not isinstance(l6, dict):
        l6 = {}
    w_s = float(l6.get("weight_short", 0.5))
    w_l = float(l6.get("weight_long", 0.5))
    if w_s < 0 or w_l < 0 or (w_s + w_l) <= 0:
        _fail(f"[FAIL] invalid L6 weights: weight_short={w_s}, weight_long={w_l}")
    s = w_s + w_l
    return (w_s / s, w_l / s)

def _detect_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

@dataclass(frozen=True)
class L6CheckResult:
    ok: bool
    messages: List[str]

def _validate_fold_windows_nonoverlap(cv: pd.DataFrame, *, name: str) -> None:
    # segment(phase)별로 test window가 서로 겹치지 않는지 확인(겹치면 fold_id->date 검증이 애매해짐)
    for seg, g in cv.groupby("segment", sort=False):
        gg = g.sort_values(["test_start", "test_end"]).reset_index(drop=True)
        # 바로 이전 test_end 다음날 이후에 시작해야(동일 구간 중복 방지)
        prev_end = None
        for i, row in gg.iterrows():
            ts = row["test_start"]
            te = row["test_end"]
            if prev_end is not None and ts <= prev_end:
                _fail(
                    f"[FAIL] {name} has overlapping test windows in segment='{seg}'. "
                    f"prev_end={prev_end.date()} cur_start={ts.date()} fold_id={row['fold_id']}"
                )
            prev_end = te

def _fold_membership_check(
    scores: pd.DataFrame,
    cv: pd.DataFrame,
    *,
    fold_col: str,
    score_present_mask: pd.Series,
    name: str,
) -> None:
    """
    scores의 (date, phase, fold_id_*)가 cv_folds의 (segment, fold_id, test_start~test_end)에
    논리적으로 부합하는지 확인한다.
    - score_present_mask=True인 행만 검증 대상으로 삼는다(예: long 없는 구간 제외)
    """
    # 필요한 컬럼만
    s = scores.loc[score_present_mask, ["date", "phase", fold_col]].copy()
    s = s.dropna(subset=[fold_col])
    s[fold_col] = s[fold_col].astype(str)

    cv2 = cv.copy()
    cv2["test_start"] = _ensure_datetime(cv2["test_start"], f"{name}.test_start")
    cv2["test_end"] = _ensure_datetime(cv2["test_end"], f"{name}.test_end")
    cv2["segment"] = cv2["segment"].astype(str)
    cv2["fold_id"] = cv2["fold_id"].astype(str)

    # fold_id + phase(segment)로 붙이고, date가 test window 안인지 확인
    m = s.merge(
        cv2[["fold_id", "segment", "test_start", "test_end"]],
        left_on=[fold_col, "phase"],
        right_on=["fold_id", "segment"],
        how="left",
        validate="many_to_one",
    )

    if m["test_start"].isna().any():
        bad = m.loc[m["test_start"].isna(), ["date", "phase", fold_col]].head(20)
        _fail(f"[FAIL] {name}: fold_id not found in cv_folds for given phase(segment). sample:\n{bad}")

    in_window = (m["date"] >= m["test_start"]) & (m["date"] <= m["test_end"])
    bad_cnt = int((~in_window).sum())
    if bad_cnt > 0:
        bad = m.loc[~in_window, ["date", "phase", fold_col, "test_start", "test_end"]].head(20)
        _fail(f"[FAIL] {name}: date not in the test window of its fold_id. sample:\n{bad}")

def validate_l6_outputs(cfg_path: Path) -> L6CheckResult:
    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")

    msgs: List[str] = []
    msgs.append("=== L6 Validation Runner ===")
    msgs.append(f"ROOT  : {_root_dir()}")
    msgs.append(f"CFG   : {cfg_path}")
    msgs.append(f"INTERIM: {interim}")

    required = ["rebalance_scores", "rebalance_scores_summary", "cv_folds_short", "cv_folds_long"]
    for n in required:
        if not artifact_exists(interim / n):
            _fail(f"[FAIL] missing artifact: {n} at {interim / n}")

    scores = load_artifact(interim / "rebalance_scores")
    summary = load_artifact(interim / "rebalance_scores_summary")
    cv_s = load_artifact(interim / "cv_folds_short")
    cv_l = load_artifact(interim / "cv_folds_long")

    meta_scores = _read_meta(interim, "rebalance_scores")
    meta_summary = _read_meta(interim, "rebalance_scores_summary")

    msgs.append("\n=== Meta check ===")
    msgs.append(f"- rebalance_scores meta.stage: {meta_scores.get('stage')}")
    msgs.append(f"- rebalance_scores meta.quality keys: {list((meta_scores.get('quality') or {}).keys())}")
    msgs.append(f"- rebalance_scores_summary meta.stage: {meta_summary.get('stage')}")
    msgs.append(f"- rebalance_scores_summary meta.quality keys: {list((meta_summary.get('quality') or {}).keys())}")

    if "scoring" not in (meta_scores.get("quality") or {}):
        _fail("[FAIL] rebalance_scores meta.quality missing key 'scoring'")
    if "scoring" not in (meta_summary.get("quality") or {}):
        _fail("[FAIL] rebalance_scores_summary meta.quality missing key 'scoring'")

    msgs.append("\n=== Schema check ===")
    for c in ["date", "ticker", "phase"]:
        if c not in scores.columns:
            _fail(f"[FAIL] rebalance_scores missing required col: {c}")
    for c in ["date", "phase", "n_tickers", "coverage_ticker_pct"]:
        if c not in summary.columns:
            _fail(f"[FAIL] rebalance_scores_summary missing required col: {c}")
    for c in ["fold_id", "segment", "test_start", "test_end"]:
        if c not in cv_s.columns:
            _fail(f"[FAIL] cv_folds_short missing required col: {c}")
        if c not in cv_l.columns:
            _fail(f"[FAIL] cv_folds_long missing required col: {c}")

    scores = scores.copy()
    summary = summary.copy()
    scores["date"] = _ensure_datetime(scores["date"], "scores.date")
    summary["date"] = _ensure_datetime(summary["date"], "summary.date")
    scores["ticker"] = scores["ticker"].astype(str)
    scores["phase"] = scores["phase"].astype(str)
    summary["phase"] = summary["phase"].astype(str)

    msgs.append("\n=== Duplicate check (date,ticker,phase) ===")
    dup = scores.duplicated(["date", "ticker", "phase"], keep=False)
    if int(dup.sum()) > 0:
        sample = scores.loc[dup, ["date", "ticker", "phase"]].head(20)
        _fail(f"[FAIL] duplicates detected in rebalance_scores. sample:\n{sample}")

    score_short_col = _detect_col(scores, ["score_short"])
    score_long_col = _detect_col(scores, ["score_long"])
    score_ens_col = _detect_col(scores, ["score_ens", "score_ensemble", "score"])
    if score_short_col is None:
        _fail("[FAIL] rebalance_scores missing 'score_short'")
    if score_long_col is None:
        _fail("[FAIL] rebalance_scores missing 'score_long'")
    if score_ens_col is None:
        _fail("[FAIL] rebalance_scores missing ensemble score col (score_ens/score_ensemble/score)")

    # fold cols (merge suffix 대응)
    fold_short = _detect_col(scores, ["fold_id_short", "fold_id_x", "fold_id"])
    fold_long = _detect_col(scores, ["fold_id_long", "fold_id_y"])
    if fold_short is None:
        _fail("[FAIL] rebalance_scores missing short fold id col (fold_id_short/fold_id_x/fold_id)")
    # long은 없는 기간이 있으므로 컬럼 자체는 있어야 한다
    if fold_long is None:
        _fail("[FAIL] rebalance_scores missing long fold id col (fold_id_long/fold_id_y)")

    s_short = pd.to_numeric(scores[score_short_col], errors="coerce")
    s_long = pd.to_numeric(scores[score_long_col], errors="coerce")
    s_ens = pd.to_numeric(scores[score_ens_col], errors="coerce")

    long_present = s_long.notna()
    long_missing = ~long_present

    # long이 없는 행에서는 fold_long도 결측이어야(정합성)
    fl = scores[fold_long].astype("string")
    bad = long_missing & fl.notna()
    if int(bad.sum()) > 0:
        sample = scores.loc[bad, ["date", "ticker", "phase", fold_long, score_long_col]].head(20)
        _fail(f"[FAIL] long missing rows have non-null fold_id_long. sample:\n{sample}")

    msgs.append("\n=== CV windows sanity ===")
    cv_s2 = cv_s.copy()
    cv_l2 = cv_l.copy()
    cv_s2["test_start"] = _ensure_datetime(cv_s2["test_start"], "cv_s.test_start")
    cv_s2["test_end"] = _ensure_datetime(cv_s2["test_end"], "cv_s.test_end")
    cv_l2["test_start"] = _ensure_datetime(cv_l2["test_start"], "cv_l.test_start")
    cv_l2["test_end"] = _ensure_datetime(cv_l2["test_end"], "cv_l.test_end")
    _validate_fold_windows_nonoverlap(cv_s2, name="cv_folds_short")
    _validate_fold_windows_nonoverlap(cv_l2, name="cv_folds_long")

    # 핵심: fold_id equality가 아니라, 각 fold_id가 자기 test window에 date를 포함하는지 확인
    msgs.append("\n=== Fold membership check ===")
    _fold_membership_check(
        scores=scores,
        cv=cv_s2,
        fold_col=fold_short,
        score_present_mask=s_short.notna(),
        name="SHORT",
    )
    _fold_membership_check(
        scores=scores,
        cv=cv_l2,
        fold_col=fold_long,
        score_present_mask=long_present,
        name="LONG",
    )

    # Ensemble 검증(가중 평균)
    w_s, w_l = _get_l6_weights(cfg)
    msgs.append("\n=== Ensemble score check ===")
    msgs.append(f"- weights: short={w_s:.6f}, long={w_l:.6f}")

    wsum = (~s_short.isna()).astype(float) * w_s + (~s_long.isna()).astype(float) * w_l
    exp = (s_short.fillna(0) * w_s + s_long.fillna(0) * w_l)
    exp = exp.where(wsum > 0, np.nan) / wsum.where(wsum > 0, np.nan)

    diff = (s_ens - exp).abs()
    med = float(diff.median(skipna=True))
    p99 = float(diff.quantile(0.99))
    msgs.append(f"- |ens - expected| median={med:.10f}, p99={p99:.10f}")
    if not np.isfinite(med) or p99 > 1e-6:
        _fail("[FAIL] ensemble score does not match expected weighted merge (check build_rebalance_scores)")

    # Summary 정합성
    msgs.append("\n=== Coverage vs summary ===")
    calc = (
        scores.groupby(["date", "phase"], as_index=False)
        .agg(
            n_tickers_calc=("ticker", "nunique"),
            ens_missing=(score_ens_col, lambda x: float(pd.to_numeric(x, errors="coerce").isna().mean())),
        )
    )
    merged = summary.merge(calc, on=["date", "phase"], how="left")
    if merged["n_tickers_calc"].isna().any():
        miss = merged.loc[merged["n_tickers_calc"].isna(), ["date", "phase"]].head(20)
        _fail(f"[FAIL] summary has (date,phase) not found in scores. sample:\n{miss}")

    diff_nt = (merged["n_tickers"] - merged["n_tickers_calc"]).abs()
    if int(diff_nt.max()) != 0:
        bad = merged.loc[diff_nt != 0, ["date", "phase", "n_tickers", "n_tickers_calc"]].head(20)
        _fail(f"[FAIL] n_tickers mismatch between summary and scores. sample:\n{bad}")

    if (summary["coverage_ticker_pct"] < 0).any() or (summary["coverage_ticker_pct"] > 100).any():
        bad = summary.loc[(summary["coverage_ticker_pct"] < 0) | (summary["coverage_ticker_pct"] > 100)].head(20)
        _fail(f"[FAIL] coverage_ticker_pct out of range [0,100]. sample:\n{bad}")

    msgs.append("\n✅ L6 VALIDATION COMPLETE: All critical checks passed.")
    msgs.append("➡️ Next: proceed to L7 (backtest / portfolio construction).")
    return L6CheckResult(ok=True, messages=msgs)

def main():
    root = _root_dir()
    cfg_path = root / "configs" / "config.yaml"
    res = validate_l6_outputs(cfg_path)
    for m in res.messages:
        print(m)

if __name__ == "__main__":
    main()
