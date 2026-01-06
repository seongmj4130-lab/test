# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l3n_news_sentiment.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class NewsSentimentConfig:
    """
    [개선안 15번] 뉴스 감성 피처(뼈대) 로더/집계 설정

    목적:
      - (기사 단위 -1/0/1 라벨) -> (일별×종목별) 피처로 집계
      - L3 단계에서 panel_merged_daily에 머지하여 이후 L8/L6R 랭킹 엔진이 사용 가능하게 함

    핵심 정책:
      - lookahead 방지: 기본 lag_days=1 (t일 매매 신호에 t-1일까지의 뉴스만 반영)
      - 입력 파일이 없으면: 절대 실패하지 않고 "스킵" + 경고만 남김 (placeholder 뼈대)
    """
    enabled: bool = False
    source_path: str = "data/external/news_sentiment_daily.parquet"
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
        raise KeyError("news sentiment input must have columns: date, ticker")
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out

def _aggregate_from_article_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    [개선안 15번] 기사 단위 라벨(-1/0/1)에서 일별×종목별 카운트 집계
    필요 컬럼: date, ticker, label
    """
    x = _ensure_date_ticker(df)
    if "label" not in x.columns:
        raise KeyError("article-level news sentiment must have 'label' column (-1/0/1)")

    lab = pd.to_numeric(x["label"], errors="coerce")
    x = x.assign(_lab=lab)
    x = x.dropna(subset=["_lab"])
    x["_lab"] = x["_lab"].astype(int)

    # count P/N/Z
    g = x.groupby(["date", "ticker"], sort=False)["_lab"]
    agg = g.value_counts().unstack(fill_value=0)
    for col, key in [(1, "P"), (-1, "N"), (0, "Z")]:
        if col not in agg.columns:
            agg[col] = 0
        agg = agg.rename(columns={col: key})
    agg = agg[["P", "N", "Z"]].reset_index()
    agg["T"] = agg["P"] + agg["N"] + agg["Z"]
    return agg

def _compute_features_from_counts(counts: pd.DataFrame, *, shrink_k: int) -> pd.DataFrame:
    """
    [개선안 15번] counts(P,N,Z,T) -> (news_sentiment, news_conviction, news_volume)
    """
    x = counts.copy()
    for c in ["P", "N", "Z", "T"]:
        if c not in x.columns:
            raise KeyError(f"counts missing required column: {c}")

    P = pd.to_numeric(x["P"], errors="coerce").fillna(0.0)
    N = pd.to_numeric(x["N"], errors="coerce").fillna(0.0)
    T = pd.to_numeric(x["T"], errors="coerce").fillna(0.0)

    k = float(max(int(shrink_k), 0))
    denom = (T + k).replace(0.0, np.nan)

    x["news_sentiment_raw"] = (P - N) / denom
    x["news_conviction_raw"] = (P - N).abs() / denom
    x["news_volume_raw"] = np.log1p(T)

    return x[["date", "ticker", "news_sentiment_raw", "news_conviction_raw", "news_volume_raw"]].copy()

def _apply_lag_by_ticker(df: pd.DataFrame, *, lag_days: int) -> pd.DataFrame:
    """
    [개선안 15번] lookahead 방지: ticker별로 lag_days만큼 shift
    """
    x = _ensure_date_ticker(df)
    lag = int(lag_days)
    if lag <= 0:
        # no lag
        out = x.rename(
            columns={
                "news_sentiment_raw": "news_sentiment",
                "news_conviction_raw": "news_conviction",
                "news_volume_raw": "news_volume",
            }
        )
        return out[["date", "ticker", "news_sentiment", "news_conviction", "news_volume"]].copy()

    x = x.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    for raw, col in [
        ("news_sentiment_raw", "news_sentiment"),
        ("news_conviction_raw", "news_conviction"),
        ("news_volume_raw", "news_volume"),
    ]:
        if raw not in x.columns:
            raise KeyError(f"missing column: {raw}")
        x[col] = x.groupby("ticker", sort=False)[raw].shift(lag)
    return x[["date", "ticker", "news_sentiment", "news_conviction", "news_volume"]].copy()

def _add_ewm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [개선안 18번] 뉴스 감성 파생 피처 추가

    생성:
      - news_sentiment_ewm5
      - news_sentiment_ewm20
      - news_sentiment_surprise (= news_sentiment - news_sentiment_ewm20)

    Notes:
      - df는 이미 lag 적용된 news_sentiment를 포함해야 함(lookahead 방지)
    """
    x = _ensure_date_ticker(df)
    for c in ["news_sentiment", "news_conviction", "news_volume"]:
        if c not in x.columns:
            raise KeyError(f"missing required column: {c}")

    x = x.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    s = pd.to_numeric(x["news_sentiment"], errors="coerce")
    x["news_sentiment_ewm5"] = x.groupby("ticker", sort=False)["news_sentiment"].transform(
        lambda v: pd.to_numeric(v, errors="coerce").ewm(span=5, adjust=False, min_periods=1).mean()
    )
    x["news_sentiment_ewm20"] = x.groupby("ticker", sort=False)["news_sentiment"].transform(
        lambda v: pd.to_numeric(v, errors="coerce").ewm(span=20, adjust=False, min_periods=1).mean()
    )
    x["news_sentiment_surprise"] = s - pd.to_numeric(x["news_sentiment_ewm20"], errors="coerce")
    return x

def build_news_sentiment_daily_features(
    *,
    news_df: pd.DataFrame,
    shrink_k: int = 5,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    [개선안 15번] 뉴스 감성 피처 생성(기사 단위 or 일별 단위 입력 모두 지원)

    입력 지원:
      - (A) 기사 단위: date,ticker,label(-1/0/1)
      - (B) 일별 단위(이미 집계됨): date,ticker,P,N,Z,T
      - (C) 일별 단위(이미 피처 있음): date,ticker,news_sentiment/news_conviction/news_volume

    출력:
      - date, ticker, news_sentiment, news_conviction, news_volume

    Notes:
      - 최종 출력은 lag_days 적용 후 값 (lookahead 방지)
    """
    df = _ensure_date_ticker(news_df)

    # Case C: already feature columns exist -> just lag and return
    if {"news_sentiment", "news_conviction", "news_volume"}.issubset(set(df.columns)):
        x = df[["date", "ticker", "news_sentiment", "news_conviction", "news_volume"]].copy()
        # treat them as raw and lag
        x = x.rename(
            columns={
                "news_sentiment": "news_sentiment_raw",
                "news_conviction": "news_conviction_raw",
                "news_volume": "news_volume_raw",
            }
        )
        feats = _apply_lag_by_ticker(x, lag_days=lag_days)
        return _add_ewm_features(feats)

    # Case B: counts exist
    if {"P", "N", "Z"}.issubset(set(df.columns)):
        x = df.copy()
        if "T" not in x.columns:
            x["T"] = pd.to_numeric(x["P"], errors="coerce").fillna(0.0) + pd.to_numeric(x["N"], errors="coerce").fillna(0.0) + pd.to_numeric(x["Z"], errors="coerce").fillna(0.0)
        feats_raw = _compute_features_from_counts(x[["date", "ticker", "P", "N", "Z", "T"]], shrink_k=shrink_k)
        feats = _apply_lag_by_ticker(feats_raw, lag_days=lag_days)
        return _add_ewm_features(feats)

    # Case A: article labels
    if "label" in df.columns:
        counts = _aggregate_from_article_labels(df)
        feats_raw = _compute_features_from_counts(counts, shrink_k=shrink_k)
        feats = _apply_lag_by_ticker(feats_raw, lag_days=lag_days)
        return _add_ewm_features(feats)

    raise KeyError(
        "Unsupported news sentiment schema. Need one of: "
        "(date,ticker,label) or (date,ticker,P,N,Z[,T]) or (date,ticker,news_sentiment,news_conviction,news_volume)"
    )

def maybe_merge_news_sentiment(
    *,
    panel_merged_daily: pd.DataFrame,
    cfg: dict,
    project_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    [개선안 15번] (뼈대) 뉴스 감성 피처를 panel_merged_daily에 머지

    - 설정(news.enabled)이 False면 그대로 반환
    - enabled=True인데 파일이 없으면 그대로 반환(경고만)
    - 파일이 있으면 피처 3개를 생성/머지:
        news_sentiment, news_conviction, news_volume
    """
    warns: List[str] = []
    if panel_merged_daily is None or panel_merged_daily.empty:
        raise ValueError("panel_merged_daily is empty")

    news_cfg_raw = (cfg.get("news", {}) if isinstance(cfg, dict) else {}) or (cfg.get("params", {}).get("news", {}) if isinstance(cfg, dict) else {}) or {}
    news_cfg = NewsSentimentConfig(
        enabled=bool(news_cfg_raw.get("enabled", False)),
        source_path=str(news_cfg_raw.get("source_path", "data/external/news_sentiment_daily.parquet")),
        lag_days=int(news_cfg_raw.get("lag_days", 1)),
        shrink_k=int(news_cfg_raw.get("shrink_k", 5)),
    )

    if not news_cfg.enabled:
        return panel_merged_daily, warns

    root = project_root if project_root is not None else Path(cfg.get("paths", {}).get("base_dir", Path.cwd())) if isinstance(cfg, dict) else Path.cwd()
    src = Path(news_cfg.source_path)
    src_path = src if src.is_absolute() else (root / src)

    if not src_path.exists():
        warns.append(f"[L3N] news.enabled=True but file not found -> skipped: {src_path}")
        return panel_merged_daily, warns

    try:
        news_df = _read_table(src_path)
        feats = build_news_sentiment_daily_features(
            news_df=news_df,
            shrink_k=news_cfg.shrink_k,
            lag_days=news_cfg.lag_days,
        )
    except Exception as e:
        warns.append(f"[L3N] failed to load/build news sentiment features -> skipped: {type(e).__name__}: {e}")
        return panel_merged_daily, warns

    # merge
    out = panel_merged_daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    feats = _ensure_date_ticker(feats)

    before_cols = set(out.columns)
    out = out.merge(feats, on=["date", "ticker"], how="left", validate="many_to_one")
    added = sorted(list(set(out.columns) - before_cols))
    warns.append(f"[L3N] merged news sentiment features: {added} (rows={len(feats):,}, file={src_path})")

    return out, warns
