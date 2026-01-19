# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/news_sentiment_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class NewsSentimentConfig:
    """
    [개선안 15번] 뉴스 감성 피처(뼈대) attach 설정

    목적:
      - 랭킹 기반 전략에서 dataset_daily에 뉴스 감성 feature를 추가할 수 있도록,
        "파일이 있으면 merge, 없으면 no-op" 구조를 제공한다.

    누수 방지:
      - 기본적으로 lag_days=1을 권장 (t일 매매 신호에 t-1일 뉴스만 사용)
    """
    # [개선안 18번] 기본값은 False로 두고, config.yaml의 news.enabled로만 켠다.
    enabled: bool = False
    lag_days: int = 1
    # [개선안 18번] config.yaml의 news.source_path를 우선 사용한다.
    # 기대 파일: date,ticker + feature columns
    rel_path: str = "news_sentiment_daily.parquet"  # data/external/news_sentiment_daily.parquet

    # 기대 컬럼(없어도 됨: 존재하는 것만 붙임)
    feature_cols: Tuple[str, ...] = (
        "news_sentiment",        # 방향성: (P-N)/(T+k) 등 [-1,1] 근처 추천
        "news_conviction",       # 확신도: |P-N|/(T+k)
        "news_volume",           # log(1+T)
        "news_sentiment_ewm5",
        "news_sentiment_ewm20",
        "news_sentiment_surprise",
    )

def _ensure_date_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "ticker" not in out.columns:
        raise KeyError("required columns missing: date, ticker")
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out

def _resolve_external_path(cfg: dict) -> Path:
    paths = (cfg.get("paths", {}) if isinstance(cfg, dict) else {}) or {}
    base_dir = Path(paths.get("base_dir", Path.cwd()))
    data_ext = Path(paths.get("data_ext", base_dir / "data" / "external"))
    # config.yaml에 "{base_dir}/data/external" 템플릿이 있을 수 있으므로 단순히 base_dir replace는 하지 않고,
    # Path로 그대로 취급한다(Windows 절대경로면 그대로).
    return data_ext

def attach_news_sentiment_features(
    dataset_daily: pd.DataFrame,
    *,
    cfg: dict,
    config: Optional[NewsSentimentConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    [개선안 15번] dataset_daily에 뉴스 감성 피처를 (옵션) merge 한다.

    - 파일이 없으면: dataset_daily를 그대로 반환 + 경고 1줄
    - 파일이 있으면:
      - (date,ticker) 기준 left merge
      - lag_days>0이면 news_df.date를 +lag_days 이동하여 "t일 row에 t-lag 뉴스"가 붙도록 맞춤

    Returns:
        (dataset_daily_out, warnings)
    """
    warns: List[str] = []
    # [개선안 18번] config.yaml(news 섹션)을 단일 진실 소스로 사용
    news_cfg_raw = (cfg.get("news", {}) if isinstance(cfg, dict) else {}) or (cfg.get("params", {}).get("news", {}) if isinstance(cfg, dict) else {}) or {}
    if config is None:
        config = NewsSentimentConfig(
            enabled=bool(news_cfg_raw.get("enabled", False)),
            lag_days=int(news_cfg_raw.get("lag_days", 1)),
            rel_path=str(news_cfg_raw.get("source_path", "data/external/news_sentiment_daily.parquet")),
        )

    if (not config.enabled) or dataset_daily is None or dataset_daily.empty:
        return dataset_daily, warns

    ds = _ensure_date_ticker(dataset_daily)

    # [뉴스 피처 병합 문제 해결] 이미 뉴스 피처가 있으면 스킵 (L3에서 이미 병합됨)
    existing_news_cols = [c for c in ds.columns if c in config.feature_cols]
    if existing_news_cols:
        warns.append(f"[L4 News] news features already exist -> skipped (from L3): {existing_news_cols}")
        return ds, warns

    # source_path는 절대/상대 둘 다 지원
    root = Path((cfg.get("paths", {}) or {}).get("base_dir", Path.cwd())) if isinstance(cfg, dict) else Path.cwd()
    src = Path(config.rel_path)
    src_path = src if src.is_absolute() else (root / src)
    if not src_path.exists():
        warns.append(f"[L4 News] news sentiment file not found -> skipped: {src_path}")
        return ds, warns

    # 파일 로드
    if src_path.suffix.lower() == ".csv":
        news = pd.read_csv(src_path)
    else:
        news = pd.read_parquet(src_path)

    news = _ensure_date_ticker(news)

    # 사용할 컬럼만 선택 (존재하는 것만)
    keep = ["date", "ticker"] + [c for c in config.feature_cols if c in news.columns]
    news = news[keep].copy()

    if len(keep) <= 2:
        warns.append(f"[L4 News] news file loaded but no known feature columns -> skipped. path={src_path}")
        return ds, warns

    # 누수 방지 lag: news의 date를 +lag_days 이동시키면,
    # dataset의 date=t에 news의 원래 date=t-lag 가 merge됨.
    lag_days = int(config.lag_days)
    if lag_days != 0:
        news["date"] = news["date"] + pd.Timedelta(days=lag_days)
        warns.append(f"[L4 News] applied lag_days={lag_days} (news.date shifted +{lag_days}d)")

    # 중복 제거
    if news.duplicated(["date", "ticker"]).any():
        before = len(news)
        news = news.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
        warns.append(f"[L4 News] dropped duplicate (date,ticker) rows: {before - len(news)}")

    out = ds.merge(news, on=["date", "ticker"], how="left", validate="one_to_one")
    warns.append(f"[L4 News] attached cols={keep[2:]} from {src_path} (rows={len(out):,})")

    return out, warns
