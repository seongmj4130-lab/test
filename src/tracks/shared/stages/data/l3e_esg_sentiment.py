# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l3e_esg_sentiment.py
"""
[ESG 통합] ESG 감성 피처 통합 모듈

목적:
  - ESG 데이터(date, ticker, pred_label, ESG_Label)를 일별×종목별 ESG 점수 피처로 집계
  - L3 단계에서 panel_merged_daily에 머지하여 이후 L8/L6R 랭킹 엔진이 사용 가능하게 함

핵심 정책:
  - lookahead 방지: 기본 lag_days=1 (t일 매매 신호에 t-1일까지의 ESG만 반영)
  - 입력 파일이 없으면: 절대 실패하지 않고 "스킵" + 경고만 남김
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ESGSentimentConfig:
    """
    [ESG 통합] ESG 감성 피처 로더/집계 설정

    목적:
      - (기사 단위 -1/0/1 라벨) -> (일별×종목별) ESG 점수 피처로 집계
      - L3 단계에서 panel_merged_daily에 머지하여 이후 L8/L6R 랭킹 엔진이 사용 가능하게 함

    핵심 정책:
      - lookahead 방지: 기본 lag_days=1 (t일 매매 신호에 t-1일까지의 ESG만 반영)
      - 입력 파일이 없으면: 절대 실패하지 않고 "스킵" + 경고만 남김 (placeholder 뼈대)
    """
    enabled: bool = False
    source_path: str = "data/external/esg_daily.parquet"
    lag_days: int = 1
    shrink_k: int = 5  # (P-N)/(T+k) 안정화 상수

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")

def _ensure_date_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "ticker" not in out.columns:
        raise KeyError("ESG sentiment input must have columns: date, ticker")
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out

def _aggregate_esg_by_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ESG 통합] ESG 데이터를 일별×종목별로 집계

    입력: date, ticker, pred_label, ESG_Label
    출력: date, ticker, P, N, Z, T (전체), environmental_P, environmental_N, environmental_Z, environmental_T, ...
    """
    x = _ensure_date_ticker(df)

    if "pred_label" not in x.columns or "ESG_Label" not in x.columns:
        raise KeyError("ESG data must have columns: pred_label, ESG_Label")

    # pred_label을 숫자로 변환
    lab = pd.to_numeric(x["pred_label"], errors="coerce")
    x = x.assign(_lab=lab)
    x = x.dropna(subset=["_lab"])
    x["_lab"] = x["_lab"].astype(int)

    # 전체 ESG 카운트 (P, N, Z)
    g_all = x.groupby(["date", "ticker"], sort=False)["_lab"]
    agg_all = g_all.value_counts().unstack(fill_value=0)
    for col, key in [(1, "P"), (-1, "N"), (0, "Z")]:
        if col not in agg_all.columns:
            agg_all[key] = 0
        elif col != key:
            agg_all[key] = agg_all[col]
            agg_all = agg_all.drop(columns=[col], errors="ignore")
    agg_all = agg_all[["P", "N", "Z"]].reset_index()
    agg_all["T"] = agg_all["P"] + agg_all["N"] + agg_all["Z"]

    # ESG_Label별 카운트
    esg_labels = ["Environmental", "Social", "Governance"]
    result = agg_all.copy()

    for label in esg_labels:
        label_data = x[x["ESG_Label"] == label].copy()
        if len(label_data) > 0:
            g_label = label_data.groupby(["date", "ticker"], sort=False)["_lab"]
            agg_label = g_label.value_counts().unstack(fill_value=0)
            for col, key in [(1, "P"), (-1, "N"), (0, "Z")]:
                if col not in agg_label.columns:
                    agg_label[f"{label.lower()}_{key}"] = 0
                else:
                    agg_label[f"{label.lower()}_{key}"] = agg_label[col]
                    agg_label = agg_label.drop(columns=[col], errors="ignore")
            agg_label = agg_label[[f"{label.lower()}_P", f"{label.lower()}_N", f"{label.lower()}_Z"]].reset_index()
            agg_label[f"{label.lower()}_T"] = agg_label[f"{label.lower()}_P"] + agg_label[f"{label.lower()}_N"] + agg_label[f"{label.lower()}_Z"]

            result = result.merge(agg_label, on=["date", "ticker"], how="left", validate="one_to_one")
        else:
            # 해당 라벨이 없으면 0으로 채움
            for key in ["P", "N", "Z"]:
                result[f"{label.lower()}_{key}"] = 0
            result[f"{label.lower()}_T"] = 0

    # 결측값 처리
    for col in result.columns:
        if col not in ["date", "ticker"]:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    return result

def _compute_esg_features_from_counts(counts: pd.DataFrame, *, shrink_k: int) -> pd.DataFrame:
    """
    [ESG 통합] counts(P, N, Z, T, ...) -> ESG 점수 피처 생성

    생성 피처:
      - esg_score: 전체 ESG 점수 ((P - N) / (T + k))
      - environmental_score: Environmental 점수
      - social_score: Social 점수
      - governance_score: Governance 점수
    """
    x = counts.copy()

    k = float(max(int(shrink_k), 0))

    # 전체 ESG 점수 (뉴스 감성과 동일한 공식)
    P = pd.to_numeric(x.get("P", 0), errors="coerce").fillna(0.0)
    N = pd.to_numeric(x.get("N", 0), errors="coerce").fillna(0.0)
    T = pd.to_numeric(x.get("T", 0), errors="coerce").fillna(0.0)
    denom_all = (T + k).replace(0.0, np.nan)
    x["esg_score"] = (P - N) / denom_all

    # ESG_Label별 점수
    esg_labels = ["Environmental", "Social", "Governance"]
    for label in esg_labels:
        label_lower = label.lower()
        label_P = pd.to_numeric(x.get(f"{label_lower}_P", 0), errors="coerce").fillna(0.0)
        label_N = pd.to_numeric(x.get(f"{label_lower}_N", 0), errors="coerce").fillna(0.0)
        label_T = pd.to_numeric(x.get(f"{label_lower}_T", 0), errors="coerce").fillna(0.0)
        denom_label = (label_T + k).replace(0.0, np.nan)
        x[f"{label_lower}_score"] = (label_P - label_N) / denom_label

    return x[["date", "ticker", "esg_score", "environmental_score", "social_score", "governance_score"]].copy()

def _apply_lag_by_ticker(df: pd.DataFrame, *, lag_days: int) -> pd.DataFrame:
    """
    [ESG 통합] lookahead 방지: ticker별로 lag_days만큼 shift
    """
    x = _ensure_date_ticker(df)
    lag = int(lag_days)

    if lag <= 0:
        # no lag
        return x[["date", "ticker", "esg_score", "environmental_score", "social_score", "governance_score"]].copy()

    x = x.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    for col in ["esg_score", "environmental_score", "social_score", "governance_score"]:
        if col not in x.columns:
            raise KeyError(f"missing column: {col}")
        x[col] = x.groupby("ticker", sort=False)[col].shift(lag)

    return x[["date", "ticker", "esg_score", "environmental_score", "social_score", "governance_score"]].copy()

def build_esg_sentiment_daily_features(
    *,
    esg_df: pd.DataFrame,
    shrink_k: int = 5,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    [ESG 통합] ESG 감성 피처 생성

    입력:
      - date, ticker, pred_label, ESG_Label

    출력:
      - date, ticker, esg_score, environmental_score, social_score, governance_score

    Notes:
      - 최종 출력은 lag_days 적용 후 값 (lookahead 방지)
    """
    df = _ensure_date_ticker(esg_df)

    # ESG 데이터 집계
    counts = _aggregate_esg_by_label(df)

    # ESG 점수 피처 생성
    feats_raw = _compute_esg_features_from_counts(counts, shrink_k=shrink_k)

    # lag 적용
    feats = _apply_lag_by_ticker(feats_raw, lag_days=lag_days)

    return feats

def maybe_merge_esg_sentiment(
    *,
    panel_merged_daily: pd.DataFrame,
    cfg: dict,
    project_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    [ESG 통합] ESG 감성 피처를 panel_merged_daily에 머지

    - 설정(esg.enabled)이 False면 그대로 반환
    - enabled=True인데 파일이 없으면 그대로 반환(경고만)
    - 파일이 있으면 피처 4개를 생성/머지:
        esg_score, environmental_score, social_score, governance_score
    """
    warns: List[str] = []
    if panel_merged_daily is None or panel_merged_daily.empty:
        raise ValueError("panel_merged_daily is empty")

    esg_cfg_raw = (cfg.get("esg", {}) if isinstance(cfg, dict) else {}) or (cfg.get("params", {}).get("esg", {}) if isinstance(cfg, dict) else {}) or {}
    esg_cfg = ESGSentimentConfig(
        enabled=bool(esg_cfg_raw.get("enabled", False)),
        source_path=str(esg_cfg_raw.get("source_path", "data/external/esg_daily.parquet")),
        lag_days=int(esg_cfg_raw.get("lag_days", 1)),
        shrink_k=int(esg_cfg_raw.get("shrink_k", 5)),
    )

    if not esg_cfg.enabled:
        return panel_merged_daily, warns

    root = project_root if project_root is not None else Path(cfg.get("paths", {}).get("base_dir", Path.cwd())) if isinstance(cfg, dict) else Path.cwd()
    src = Path(esg_cfg.source_path)
    src_path = src if src.is_absolute() else (root / src)

    if not src_path.exists():
        warns.append(f"[L3E] esg.enabled=True but file not found -> skipped: {src_path}")
        return panel_merged_daily, warns

    try:
        esg_df = _read_table(src_path)
        feats = build_esg_sentiment_daily_features(
            esg_df=esg_df,
            shrink_k=esg_cfg.shrink_k,
            lag_days=esg_cfg.lag_days,
        )
    except Exception as e:
        warns.append(f"[L3E] failed to load/build ESG sentiment features -> skipped: {type(e).__name__}: {e}")
        return panel_merged_daily, warns

    # merge
    out = panel_merged_daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    feats = _ensure_date_ticker(feats)

    before_cols = set(out.columns)
    out = out.merge(feats, on=["date", "ticker"], how="left", validate="many_to_one")
    added = sorted(list(set(out.columns) - before_cols))
    warns.append(f"[L3E] merged ESG sentiment features: {added} (rows={len(feats):,}, file={src_path})")

    return out, warns
