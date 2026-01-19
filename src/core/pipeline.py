# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/core/pipeline.py
import argparse
import hashlib
import inspect
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# [Stage14] 출력 버퍼링 비활성화 (진행 상황 즉시 표시)
if sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        # Python < 3.7에서는 reconfigure가 없음
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

import pandas as pd

from src.stages.backtest.l1d_market_regime import (
    build_market_regime,  # → src/tracks/shared/stages/regime/
)

# [트랙 재정리] 백테스트 (src/tracks/track_b/stages/backtest/)
from src.stages.backtest.l7_backtest import (  # → src/tracks/track_b/stages/backtest/
    BacktestConfig,
    run_backtest,
)
from src.stages.backtest.l7b_sensitivity import (
    run_l7b_sensitivity,  # → src/tracks/track_b/stages/backtest/
)
from src.stages.backtest.l7c_benchmark import (
    run_l7c_benchmark,  # → src/tracks/track_b/stages/backtest/
)
from src.stages.backtest.l7d_stability import (
    run_l7d_stability_from_artifacts,  # → src/tracks/track_b/stages/backtest/
)

# [트랙 재정리] 공통 데이터 (src/tracks/shared/stages/data/)
from src.stages.data.l0_universe import (
    build_k200_membership_month_end,  # → src/tracks/shared/stages/data/
)
from src.stages.data.l1_ohlcv import (
    download_ohlcv_panel,  # → src/tracks/shared/stages/data/
)
from src.stages.data.l1b_sector_map import (
    build_sector_map,  # → src/tracks/shared/stages/data/
)
from src.stages.data.l2_fundamentals_dart import (
    download_annual_fundamentals,  # → src/tracks/shared/stages/data/
)
from src.stages.data.l3_panel_merge import (
    build_panel_merged_daily,  # → src/tracks/shared/stages/data/
)

# [트랙 재정리] 모델링 (src/tracks/track_b/stages/modeling/)
from src.stages.modeling.l5_train_models import train_oos_predictions  # (랭킹 모드에서는 스킵)
from src.stages.modeling.l6_scoring import build_rebalance_scores  # (랭킹 모드에서는 스킵)
from src.stages.modeling.l6r_ranking_scoring import (
    run_L6R_ranking_scoring,  # → src/tracks/track_b/stages/modeling/ [개선안 13번] 랭킹 기반 리밸런싱 스코어
)

# [트랙 재정리] 랭킹 엔진 (src/tracks/track_a/stages/ranking/)
from src.stages.ranking.l8_rank_engine import (
    run_L8_rank_engine,  # → src/tracks/track_a/stages/ranking/
)
from src.stages.ranking.ui_payload_builder import (
    run_L11_ui_payload,  # → src/tracks/track_a/stages/ranking/
)
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.meta import build_meta, save_meta
from src.utils.quality import fundamental_coverage_report, walkforward_quality_report
from src.utils.validate import raise_if_invalid, validate_df

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =========================
# Helpers
# =========================
def _resolve_section(cfg: dict, name: str) -> dict:
    """config.yaml이 params 아래/최상위 둘 다 지원하도록 통일."""
    if not isinstance(cfg, dict):
        return {}
    params = cfg.get("params", {}) or {}
    if isinstance(params, dict):
        sec = params.get(name, None)
        if isinstance(sec, dict):
            return sec
    sec2 = cfg.get(name, None)
    return sec2 if isinstance(sec2, dict) else {}

def _only_dfs(d: Any) -> Dict[str, pd.DataFrame]:
    """stage가 dict를 주더라도 DataFrame만 남겨서 러너가 깨지지 않게."""
    if isinstance(d, pd.DataFrame):
        return {"_df": d}
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            out[str(k)] = v
    return out

def _call_flexible(func, *, artifacts: dict, cfg_local: dict) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    stages의 함수 시그니처가 프로젝트마다 다를 때:
    - inspect.signature로 실제 파라미터를 보고
    - "받는 이름"에만 맞춰 인자를 넣어 호출
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # 후보 객체
    bt_returns = artifacts.get("bt_returns")
    rebalance_scores = artifacts.get("rebalance_scores")
    bt_equity_curve = artifacts.get("bt_equity_curve")

    def pick_value(pname: str) -> Any:
        n = (pname or "").lower()

        # ✅ [수정됨] artifacts 자체를 요청하는 경우 처리 추가
        if n == "artifacts":
            return artifacts

        # cfg
        if n in ("cfg", "config", "conf", "params"):
            return cfg_local

        # bt_returns 류
        if n in ("bt_returns", "returns", "bt_ret", "ret", "ret_df", "returns_df"):
            return bt_returns

        # equity curve 류
        if n in ("bt_equity_curve", "equity_curve", "bt_eq", "equity", "equity_df"):
            return bt_equity_curve

        # rebalance_scores 류
        if n in ("rebalance_scores", "scores", "score", "score_df", "signals", "signal_df", "rebalance_df"):
            return rebalance_scores

        # 애매한 이름(df/data)이면, L7C에서는 보통 bt_returns가 맞는 경우가 많아서 보수적으로 매핑
        if n in ("df", "data", "dataset"):
            return bt_returns

        return None

    # positional-only는 args로, 나머지는 kwargs로
    args = []
    kwargs = {}

    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        val = pick_value(p.name)
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            if val is None:
                if p.default is inspect._empty:
                    # artifacts가 추가되어도 해결 안 되면 에러
                    raise TypeError(f"Cannot map required positional-only arg: {p.name}")
                # default 있으면 생략 가능
            else:
                args.append(val)
        else:
            if val is not None:
                kwargs[p.name] = val

    # **kwargs를 받는 함수면, 최소한 cfg를 cfg 이름으로도 제공(내부에서 cfg를 기대하는 코드가 있을 수 있음)
    if has_varkw:
        kwargs.setdefault("cfg", cfg_local)
        if bt_returns is not None:
            kwargs.setdefault("bt_returns", bt_returns)
        if rebalance_scores is not None:
            kwargs.setdefault("rebalance_scores", rebalance_scores)
        # artifacts도 넣어줌
        kwargs.setdefault("artifacts", artifacts)

    res = func(*args, **kwargs)

    # return 형태 표준화: (outputs, warns) 또는 outputs 단독 허용
    if isinstance(res, tuple) and len(res) == 2:
        outputs, warns = res
        outs_df = _only_dfs(outputs)
        return outs_df, (warns or [])
    else:
        outs_df = _only_dfs(res)
        return outs_df, []

# =========================
# Stage runners
# =========================
def run_L0_universe(cfg, artifacts=None, *, force=False):
    p = cfg.get("params", {}) or {}
    df = build_k200_membership_month_end(
        start_date=p.get("start_date", "2015-01-02"),
        end_date=p.get("end_date", "2024-12-31"),
        index_code=p.get("index_code", "1028"),
        anchor_ticker=p.get("anchor_ticker", "005930"),
    )
    return {"universe_k200_membership_monthly": df}, []

def run_L1_base(cfg, artifacts, *, force=False):
    p = cfg.get("params", {}) or {}
    uni = artifacts["universe_k200_membership_monthly"]
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())

    df = download_ohlcv_panel(
        tickers=tickers,
        start_date=p.get("start_date", "2015-01-02"),
        end_date=p.get("end_date", "2024-12-31"),
    )
    return {"ohlcv_daily": df}, []

def run_L1B_sector_map(cfg, artifacts, *, force=False):
    """
    [Stage 4] 업종 매핑 생성
    """
    # [트랙 재정리] → src/tracks/shared/stages/data/l1b_sector_map.py
    from src.stages.data.l1b_sector_map import build_sector_map

    uni = artifacts["universe_k200_membership_monthly"]
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())

    # 월말 날짜 추출
    asof_dates = pd.DatetimeIndex(uni["date"].unique()).sort_values()

    df = build_sector_map(
        asof_dates=asof_dates,
        tickers=tickers,
    )

    return {"sector_map": df}, []

def run_L1D_market_regime(cfg, artifacts, *, force=False):
    """
    [Stage5] 시장 국면(regime) 계산
    - rebalance 날짜 기준으로 시장 국면(bull/bear) 계산
    """
    p = cfg.get("params", {}) or {}
    l7 = _resolve_section(cfg, "l7")
    regime_cfg = l7.get("regime", {}) or {}

    # regime이 비활성화되어 있으면 스킵
    if not regime_cfg.get("enabled", False):
        logger.info("[L1D] regime.enabled=False, 시장 국면 계산을 건너뜁니다.")
        return {}, []

    # rebalance_scores에서 날짜 추출 (L6 이후에 실행되어야 함)
    if "rebalance_scores" not in artifacts:
        raise KeyError("[L1D] rebalance_scores가 없습니다. L6 이후에 실행해야 합니다.")

    rebalance_scores = artifacts["rebalance_scores"]
    rebalance_dates = rebalance_scores["date"].unique()

    df = build_market_regime(
        rebalance_dates=rebalance_dates,
        start_date=p.get("start_date", "2015-01-02"),
        end_date=p.get("end_date", "2024-12-31"),
        index_code=p.get("index_code", "1028"),  # KOSPI200
        lookback_days=int(regime_cfg.get("lookback_days", 60)),
        threshold_pct=float(regime_cfg.get("threshold_pct", 0.0)),  # 하위호환을 위해 유지
        neutral_band=float(regime_cfg.get("neutral_band", 0.05)),  # [Phase 2] ±5% 범위를 neutral로 분류
    )

    return {"market_regime": df}, []

def run_L2_merge(cfg, artifacts, *, force=False):
    """
    [L2 재무데이터 재사용 규칙 / 최우선]
    - L2(재무) 산출물은 앞으로 "무조건 기존에 있는 파일"을 재사용한다.
    - 어떤 Stage에서도 DART API 호출로 fundamentals_annual을 재수집/재생성하지 않는다.
    - 기존 파일: data/interim/fundamentals_annual.parquet
    """
    # 루트 interim 디렉토리에서 기존 파일 로드
    base_interim_dir = get_path(cfg, "data_interim")
    source_file = base_interim_dir / "fundamentals_annual.parquet"

    if not artifact_exists(source_file):
        raise RuntimeError(
            f"[L2 재사용 규칙 위반] 기존 fundamentals_annual.parquet 파일이 없습니다: {source_file}\n"
            "L2 재무데이터는 무조건 기존 파일을 재사용해야 하며, API 재호출로 재생성할 수 없습니다.\n"
            "파일이 없다면 이전 실행에서 생성된 파일을 확인하거나, 수동으로 준비해야 합니다."
        )

    # 기존 파일 로드
    logger.info(f"[L2 재사용] 기존 파일에서 로드: {source_file}")
    df = load_artifact(source_file)

    # 파일 수정 시간 기록 (보고서/로그용)
    import os
    mtime = os.path.getmtime(source_file)
    from datetime import datetime
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[L2 재사용] 파일 수정 시간: {mtime_str}")

    return {"fundamentals_annual": df}, []

def run_L3_features(cfg, artifacts, *, force=False):
    p = cfg.get("params", {}) or {}
    l3 = cfg.get("l3", {}) or {}
    lag_days = int(p.get("fundamental_lag_days", 90))

    # [Stage 1] l3 설정에서 filter_k200_members_only, fundamentals_effective_date_col 읽기
    filter_k200_members_only = bool(l3.get("filter_k200_members_only", False))
    fundamentals_effective_date_col = l3.get("fundamentals_effective_date_col", "effective_date")

    # [Stage 4] sector_map을 build_panel_merged_daily에 전달
    sector_map = artifacts.get("sector_map") if "sector_map" in artifacts else None

    df, warns = build_panel_merged_daily(
        ohlcv_daily=artifacts["ohlcv_daily"],
        fundamentals_annual=artifacts["fundamentals_annual"],
        universe_membership_monthly=artifacts.get("universe_k200_membership_monthly"),
        fundamental_lag_days=lag_days,
        filter_k200_members_only=filter_k200_members_only,
        fundamentals_effective_date_col=fundamentals_effective_date_col,
        sector_map=sector_map,  # [Stage 4] sector_map 전달
    )

    # [개선안 15번] 뉴스 감성 피처 머지(뼈대)
    # - enabled=True인데 파일이 없으면 스킵(경고만) -> 현재 데이터 없는 상태에서도 안전
    try:
        # [트랙 재정리] → src/tracks/shared/stages/data/l3n_news_sentiment.py
        from src.stages.data.l3n_news_sentiment import maybe_merge_news_sentiment
        df, w_news = maybe_merge_news_sentiment(panel_merged_daily=df, cfg=cfg)
        warns.extend(w_news or [])
    except Exception as e:
        warns.append(f"[L3N] news sentiment merge hook failed (ignored): {type(e).__name__}: {e}")

    # [ESG 통합] ESG 감성 피처 머지
    # - enabled=True인데 파일이 없으면 스킵(경고만) -> 현재 데이터 없는 상태에서도 안전
    try:
        from src.stages.data.l3e_esg_sentiment import maybe_merge_esg_sentiment
        df, w_esg = maybe_merge_esg_sentiment(panel_merged_daily=df, cfg=cfg)
        warns.extend(w_esg or [])
    except Exception as e:
        warns.append(f"[L3E] ESG sentiment merge hook failed (ignored): {type(e).__name__}: {e}")

    return {"panel_merged_daily": df}, warns

def run_L4_split(cfg, artifacts, *, force=False):
    # [트랙 재정리] → src/tracks/shared/stages/data/l4_walkforward_split.py
    from src.stages.data.l4_walkforward_split import (
        build_inner_cv_folds,
        build_targets_and_folds,
    )

    # [트랙 재정리] → src/tracks/shared/stages/data/news_sentiment_features.py
    from src.stages.data.news_sentiment_features import (
        attach_news_sentiment_features,  # [개선안 15번] 뉴스 감성 피처(옵션) attach
    )

    l4 = _resolve_section(cfg, "l4")
    panel = artifacts["panel_merged_daily"]

    # [Stage 1] l4 설정에서 drop_non_universe_before_save 읽기
    drop_non_universe_before_save = bool(l4.get("drop_non_universe_before_save", False))

    df, cv_s, cv_l, warns = build_targets_and_folds(
        panel,
        holdout_years=int(l4.get("holdout_years", 2)),
        step_days=int(l4.get("step_days", 20)),
        test_window_days=int(l4.get("test_window_days", 20)),
        embargo_days=int(l4.get("embargo_days", 20)),
        horizon_short=int(l4.get("horizon_short", 20)),
        horizon_long=int(l4.get("horizon_long", 120)),
        rolling_train_years_short=int(l4.get("rolling_train_years_short", 3)),
        rolling_train_years_long=int(l4.get("rolling_train_years_long", 5)),
        price_col=l4.get("price_col", None),
        drop_non_universe_before_save=drop_non_universe_before_save,
    )

    # [개선안 18번] 뉴스 감성 피처 attach (config.news.enabled=True일 때만)
    try:
        news_cfg_raw = (cfg.get("news", {}) if isinstance(cfg, dict) else {}) or {}
        if bool(news_cfg_raw.get("enabled", False)):
            df, news_warns = attach_news_sentiment_features(df, cfg=cfg)
            warns.extend(news_warns or [])
    except Exception as e:
        # 파일 스키마/경로 문제로 파이프라인이 죽지 않게 방어 (실데이터 투입 전 뼈대 단계)
        warns.append(f"[L4 News] attach failed -> skipped: {type(e).__name__}: {e}")

    # [Stage 3] 내부 CV folds 생성
    inner_cv_k = int(l4.get("inner_cv_k", 5))
    embargo_days = int(l4.get("embargo_days", 20))
    horizon_short = int(l4.get("horizon_short", 20))
    horizon_long = int(l4.get("horizon_long", 120))

    # dates 추출 (df에서)
    dates = pd.DatetimeIndex(pd.unique(df["date"].dropna())).sort_values()

    # 각 fold에 대해 내부 CV folds 생성
    inner_folds_short_rows = []
    inner_folds_long_rows = []

    for _, fold_row in cv_s.iterrows():
        inner_folds = build_inner_cv_folds(
            train_start=fold_row["train_start"],
            train_end=fold_row["train_end"],
            k=inner_cv_k,
            embargo_days=embargo_days,
            horizon_days=horizon_short,
            dates=dates,
        )
        if not inner_folds.empty:
            inner_folds["fold_id"] = fold_row["fold_id"]
            inner_folds["horizon"] = horizon_short
            inner_folds_short_rows.append(inner_folds)

    for _, fold_row in cv_l.iterrows():
        inner_folds = build_inner_cv_folds(
            train_start=fold_row["train_start"],
            train_end=fold_row["train_end"],
            k=inner_cv_k,
            embargo_days=embargo_days,
            horizon_days=horizon_long,
            dates=dates,
        )
        if not inner_folds.empty:
            inner_folds["fold_id"] = fold_row["fold_id"]
            inner_folds["horizon"] = horizon_long
            inner_folds_long_rows.append(inner_folds)

    # 내부 CV folds 병합
    cv_inner_short = pd.concat(inner_folds_short_rows, ignore_index=True) if inner_folds_short_rows else pd.DataFrame()
    cv_inner_long = pd.concat(inner_folds_long_rows, ignore_index=True) if inner_folds_long_rows else pd.DataFrame()

    # [Stage 3] 내부 CV folds 저장 (interim_dir에)
    interim_dir = artifacts.get("_interim_dir")
    if interim_dir is not None:
        if not cv_inner_short.empty:
            inner_short_path = interim_dir / "cv_inner_folds_short.parquet"
            cv_inner_short.to_parquet(inner_short_path, index=False)
            warns.append(f"[L4 Stage3] 내부 CV folds (short) 저장: {inner_short_path} ({len(cv_inner_short)} folds)")
        if not cv_inner_long.empty:
            inner_long_path = interim_dir / "cv_inner_folds_long.parquet"
            cv_inner_long.to_parquet(inner_long_path, index=False)
            warns.append(f"[L4 Stage3] 내부 CV folds (long) 저장: {inner_long_path} ({len(cv_inner_long)} folds)")

    return {
        "dataset_daily": df,
        "cv_folds_short": cv_s,
        "cv_folds_long": cv_l,
    }, warns

def run_L5_modeling(cfg, artifacts, *, force=False):
    df = artifacts["dataset_daily"]
    cv_s = artifacts["cv_folds_short"]
    cv_l = artifacts["cv_folds_long"]

    run_cfg = _resolve_section(cfg, "run")
    if bool(run_cfg.get("debug", False)):
        logger.info(f"[DEBUG L5] dataset_daily shape={getattr(df, 'shape', None)}")
        logger.info(f"[DEBUG L5] cv_folds_short shape={getattr(cv_s, 'shape', None)}")
        logger.info(f"[DEBUG L5] columns head={list(df.columns)[:20] if hasattr(df, 'columns') else None}")
        if "ret_fwd_20d" in df.columns:
            logger.info(f"[DEBUG L5] ret_fwd_20d nan_rate={float(df['ret_fwd_20d'].isna().mean())}")

    l4 = _resolve_section(cfg, "l4")
    hs = int(l4.get("horizon_short", 20))
    hl = int(l4.get("horizon_long", 120))

    target_s = f"ret_fwd_{hs}d"
    target_l = f"ret_fwd_{hl}d"

    # [Stage 2] interim_dir을 artifacts에서 가져오기
    interim_dir = artifacts.get("_interim_dir")
    if interim_dir is None:
        # fallback: cfg에서 경로 계산
        base_interim_dir = get_path(cfg, "data_interim")
        run_tag = artifacts.get("_run_tag", "default_run")
        interim_dir = Path(base_interim_dir) / run_tag
    interim_dir = Path(interim_dir)

    pred_s, met_s, rep_s, w_s = train_oos_predictions(
        dataset_daily=df,
        cv_folds=cv_s,
        cfg=cfg,
        target_col=target_s,
        horizon=hs,
        interim_dir=interim_dir,
    )
    pred_l, met_l, rep_l, w_l = train_oos_predictions(
        dataset_daily=df,
        cv_folds=cv_l,
        cfg=cfg,
        target_col=target_l,
        horizon=hl,
        interim_dir=interim_dir,
    )

    metrics = pd.concat([met_s, met_l], ignore_index=True)
    warns = (w_s or []) + (w_l or [])

    # meta에 쓰려고 artifacts에 잠깐 보관
    artifacts["_l5_report_short"] = rep_s
    artifacts["_l5_report_long"] = rep_l

    return {
        "pred_short_oos": pred_s,
        "pred_long_oos": pred_l,
        "model_metrics": metrics,
    }, warns

def run_L6_scoring(cfg, artifacts, *, force=False):
    l6 = _resolve_section(cfg, "l6")

    w_s = float(l6.get("weight_short", 0.5))
    w_l = float(l6.get("weight_long", 0.5))

    # [Stage 4] dataset_daily에서 sector_name을 가져오기 위해 전달
    dataset_daily = artifacts.get("dataset_daily")
    scores, summary, quality, warns = build_rebalance_scores(
        pred_short_oos=artifacts["pred_short_oos"],
        pred_long_oos=artifacts["pred_long_oos"],
        universe_k200_membership_monthly=artifacts["universe_k200_membership_monthly"],
        weight_short=w_s,
        weight_long=w_l,
        dataset_daily=dataset_daily,  # [Stage 4] sector_name carry용
    )

    artifacts["_l6_quality"] = quality if isinstance(quality, dict) else {"scoring": quality}

    return {
        "rebalance_scores": scores,
        "rebalance_scores_summary": summary,
    }, (warns or [])

def run_L7_backtest(cfg, artifacts, *, force=False):
    l7 = _resolve_section(cfg, "l7")
    ret_col = str(l7.get("ret_col", l7.get("return_col", "true_short")))

    # [Stage 4] 업종 분산 제약 설정
    diversify = l7.get("diversify", {}) or {}
    diversify_enabled = bool(diversify.get("enabled", False))
    group_col = str(diversify.get("group_col", "sector_name"))
    max_names_per_group = int(diversify.get("max_names_per_group", 4))

    # [Stage5] 시장 국면(regime) 설정
    regime_cfg = l7.get("regime", {}) or {}
    regime_enabled = bool(regime_cfg.get("enabled", False))
    market_regime = artifacts.get("market_regime") if regime_enabled else None

    # [Stage13] 스모크 테스트: max_rebalances 옵션 처리
    max_rebalances = artifacts.get("_max_rebalances")
    rebalance_scores = artifacts["rebalance_scores"].copy()
    if max_rebalances is not None and max_rebalances > 0:
        # 최근 N개 리밸런싱 날짜만 필터링
        if "date" in rebalance_scores.columns:
            dates = pd.to_datetime(rebalance_scores["date"])
            unique_dates = pd.Series(dates.dropna().unique()).sort_values()
            # 최근 N개 선택 (tail) - 정순으로 정렬된 상태에서 마지막 N개
            keep_dates_list = unique_dates.tail(max_rebalances).tolist()
            keep_dates = set(keep_dates_list)
            rebalance_scores = rebalance_scores[dates.isin(keep_dates)].copy()
            # 로그: 최근 날짜부터 정순으로 출력 (날짜가 오름차순 정렬된 상태)
            date_min = keep_dates_list[0]
            date_max = keep_dates_list[-1]
            logger.info(f"[Stage13] 스모크 테스트: 최근 {max_rebalances}개 리밸런싱만 실행 (날짜 범위: {date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')}, 총 {len(keep_dates)}개 날짜)")
            logger.info(f"[Stage13] 선택된 날짜 (정순): {', '.join([d.strftime('%Y-%m-%d') for d in keep_dates_list])}")

    bt_cfg = BacktestConfig(
        holding_days=int(l7.get("holding_days", 20)),
        top_k=int(l7.get("top_k", 20)),
        cost_bps=float(l7.get("cost_bps", 10.0)),
        score_col=str(l7.get("score_col", "score_ens")),
        ret_col=ret_col,
        weighting=str(l7.get("weighting", "equal")),
        softmax_temp=float(l7.get("softmax_temp", 1.0)),
        buffer_k=int(l7.get("buffer_k", 0)),
        rebalance_interval=int(l7.get("rebalance_interval", 1)),  # [Phase 7 Step 2] Turnover 제어
        diversify_enabled=diversify_enabled,  # [Stage 4]
        group_col=group_col,  # [Stage 4]
        max_names_per_group=max_names_per_group,  # [Stage 4]
        regime_enabled=regime_enabled,  # [Stage5]
        # [국면 세분화] 5단계 국면별 설정
        regime_top_k_bull_strong=regime_cfg.get("top_k_bull_strong"),  # [국면 세분화]
        regime_top_k_bull_weak=regime_cfg.get("top_k_bull_weak"),  # [국면 세분화]
        regime_top_k_bear_strong=regime_cfg.get("top_k_bear_strong"),  # [국면 세분화]
        regime_top_k_bear_weak=regime_cfg.get("top_k_bear_weak"),  # [국면 세분화]
        regime_top_k_neutral=regime_cfg.get("top_k_neutral"),  # [국면 세분화]
        regime_exposure_bull_strong=regime_cfg.get("exposure_bull_strong"),  # [국면 세분화]
        regime_exposure_bull_weak=regime_cfg.get("exposure_bull_weak"),  # [국면 세분화]
        regime_exposure_bear_strong=regime_cfg.get("exposure_bear_strong"),  # [국면 세분화]
        regime_exposure_bear_weak=regime_cfg.get("exposure_bear_weak"),  # [국면 세분화]
        regime_exposure_neutral=regime_cfg.get("exposure_neutral"),  # [국면 세분화]
        # 하위 호환성 (2단계 설정)
        regime_top_k_bull=regime_cfg.get("top_k_bull"),  # [Stage5]
        regime_top_k_bear=regime_cfg.get("top_k_bear"),  # [Stage5]
        regime_exposure_bull=regime_cfg.get("exposure_bull"),  # [Stage5]
        regime_exposure_bear=regime_cfg.get("exposure_bear"),  # [Stage5]
        # [Phase 8 Step 2 방안1] 리밸런싱 규칙/버퍼/노출 튜닝
        smart_buffer_enabled=bool(l7.get("smart_buffer_enabled", False)),
        smart_buffer_stability_threshold=float(l7.get("smart_buffer_stability_threshold", 0.7)),
        volatility_adjustment_enabled=bool(l7.get("volatility_adjustment_enabled", False)),
        volatility_lookback_days=int(l7.get("volatility_lookback_days", 60)),
        target_volatility=float(l7.get("target_volatility", 0.15)),
        volatility_adjustment_max=float(l7.get("volatility_adjustment_max", 1.2)),
        volatility_adjustment_min=float(l7.get("volatility_adjustment_min", 0.6)),
        # [Phase 8 Step 2 방안2] 국면 필터/리스크 스케일링
        risk_scaling_enabled=bool(l7.get("risk_scaling_enabled", False)),
        risk_scaling_bear_multiplier=float(l7.get("risk_scaling_bear_multiplier", 0.7)),
        risk_scaling_neutral_multiplier=float(l7.get("risk_scaling_neutral_multiplier", 0.9)),
        risk_scaling_bull_multiplier=float(l7.get("risk_scaling_bull_multiplier", 1.0)),
    )

    # [Stage1] config.yaml에서 읽은 원본 cost_bps 값을 run_backtest에 전달
    config_cost_bps = float(l7.get("cost_bps", 10.0))
    t_l7_start = time.time()
    # [Phase 7 Step 1] bt_regime_metrics 반환값 추가 (10개 반환값)
    bt_pos, bt_ret, bt_eq, bt_met, quality, warns, selection_diagnostics, bt_returns_diagnostics, runtime_profile, bt_regime_metrics = run_backtest(
        rebalance_scores,  # [Stage13] 필터링된 rebalance_scores 사용
        bt_cfg,
        config_cost_bps=config_cost_bps,  # [Stage1] config 원본 값 전달
        market_regime=market_regime,  # [Stage5] 시장 국면 데이터 전달
    )
    t_l7_end = time.time()
    l7_total_time = t_l7_end - t_l7_start
    logger.info(f"[L7 Runtime] 백테스트 완료: 총 {l7_total_time:.1f}초")
    artifacts["_l7_quality"] = quality if isinstance(quality, dict) else {"backtest": quality}
    artifacts["_l7_runtime_profile"] = runtime_profile  # [Stage13] 런타임 프로파일 저장

    # [Stage13] selection_diagnostics 및 bt_returns_diagnostics 추가
    # [Phase 7 Step 1] bt_regime_metrics 추가 반환
    outputs = {
        "bt_positions": bt_pos,
        "bt_returns": bt_ret,  # bt_returns_core (validation 통과용)
        "bt_equity_curve": bt_eq,
        "bt_metrics": bt_met,
        "selection_diagnostics": selection_diagnostics,  # [Stage13] 신규
        "bt_returns_diagnostics": bt_returns_diagnostics,  # [Stage13] regime/exposure 진단 컬럼
    }
    # [Phase 7 Step 1] bt_regime_metrics가 비어있지 않으면 추가
    if bt_regime_metrics is not None and len(bt_regime_metrics) > 0:
        outputs["bt_regime_metrics"] = bt_regime_metrics
    return outputs, (warns or [])

def run_L7B_sensitivity(cfg, artifacts, *, force=False):
    # params 아래 섹션이 없으면 top-level fallback
    l7 = _resolve_section(cfg, "l7")
    l7b = _resolve_section(cfg, "l7b")

    cfg_local = dict(cfg)
    cfg_local["l7"] = l7
    cfg_local["l7b"] = l7b

    outputs, warns = run_l7b_sensitivity(rebalance_scores=artifacts["rebalance_scores"], cfg=cfg_local)

    # _l7b_quality: dict -> DataFrame(평평하게)
    if isinstance(outputs, dict) and "_l7b_quality" in outputs:
        q = outputs["_l7b_quality"]
        if isinstance(q, dict) and "sensitivity" in q and isinstance(q["sensitivity"], dict):
            outputs["_l7b_quality"] = pd.DataFrame([q["sensitivity"]])
        elif isinstance(q, dict):
            outputs["_l7b_quality"] = pd.DataFrame([q])

    # 혹시 dict/기타가 섞여오면 DF만 남기기
    outputs = _only_dfs(outputs)

    return outputs, (warns or [])

def run_L7C_benchmark(cfg, artifacts, *, force=False):
    """
    [L7C 실행기 - 최종 Fix]
    수정된 l7c_benchmark.py에 맞춰 artifacts와 cfg를 그대로 전달합니다.
    """
    logger.info("Running L7C Benchmark")

    # artifacts 필수 체크 (l7c 내부 로직 실행 전 사전 점검)
    if "bt_returns" not in artifacts or "rebalance_scores" not in artifacts:
        # runner 흐름상 여기까지 왔으면 있을 확률이 높지만, 명시적 에러 처리
        raise KeyError("L7C requires 'bt_returns' and 'rebalance_scores' in artifacts.")

    # 수정된 l7c_benchmark.py의 run_l7c_benchmark는 (cfg, artifacts)를 받음
    outputs, warns = run_l7c_benchmark(cfg, artifacts, force=force)

    # DataFrame만 필터링
    outputs = _only_dfs(outputs)

    # 키 표준화 (Downstream 호환성)
    std_cols = {
        "bt_vs_benchmark": ["date", "phase", "bench_return", "excess_return"],
        "bt_benchmark_compare": ["phase", "tracking_error_ann", "information_ratio"],
        "bt_benchmark_returns": ["date", "phase", "bench_return"],
    }

    for k, cols in std_cols.items():
        if k not in outputs:
            outputs[k] = pd.DataFrame(columns=cols)

    return outputs, (warns or [])

def run_L7D_stability(cfg, artifacts, *, force=False):
    l7 = _resolve_section(cfg, "l7")
    holding_days = int(l7.get("holding_days", 20))

    cfg_local = dict(cfg)
    cfg_local["l7"] = l7

    outputs, warns = run_l7d_stability_from_artifacts(
        bt_returns=artifacts["bt_returns"],
        bt_equity_curve=artifacts.get("bt_equity_curve"),
        holding_days=holding_days,
        cfg=cfg_local,
    )

    # ✅ 너가 이미 쓰고 있는 로더(untitled1.py)가 l7d_stability를 기대하는 경우를 위해 alias 제공
    if isinstance(outputs, dict) and "bt_yearly_metrics" in outputs and "l7d_stability" not in outputs:
        outputs["l7d_stability"] = outputs["bt_yearly_metrics"]

    outputs = _only_dfs(outputs)
    return outputs, (warns or [])

# =========================
# Registry
# =========================
STAGES = {
    "L0": run_L0_universe,
    "L1": run_L1_base,
    "L1B": run_L1B_sector_map,
    "L2": run_L2_merge,
    "L3": run_L3_features,
    "L4": run_L4_split,
    "L5": run_L5_modeling,
    "L6": run_L6_scoring,
    "L6R": run_L6R_ranking_scoring,  # [개선안 13번] ranking 기반 rebalance_scores 생성 (L7 입력용)
    "L1D": run_L1D_market_regime,  # [Stage5] 시장 국면 계산 (L6 이후 실행)
    "L7": run_L7_backtest,
    "L7B": run_L7B_sensitivity,
    "L7C": run_L7C_benchmark,
    "L7D": run_L7D_stability,
    "L8": run_L8_rank_engine,  # [Stage7] Ranking 엔진
    "L11": run_L11_ui_payload,  # [Stage11] UI Payload Builder
}

REQUIRED_INPUTS = {
    "L0": [],
    "L1": ["universe_k200_membership_monthly"],
    "L1B": ["universe_k200_membership_monthly"],
    "L1D": ["rebalance_scores"],  # [Stage5] L6 이후에 실행되어야 함
    "L2": ["universe_k200_membership_monthly"],
    "L3": ["ohlcv_daily", "fundamentals_annual"],
    "L4": ["panel_merged_daily"],
    "L5": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L6": ["pred_short_oos", "pred_long_oos", "universe_k200_membership_monthly"],
    "L6R": ["dataset_daily", "cv_folds_short", "universe_k200_membership_monthly"],  # [개선안 13번]
    "L7": ["rebalance_scores"],
    "L7B": ["rebalance_scores"],
    "L7C": ["bt_returns", "rebalance_scores"],  # rebalance_scores는 필요 없을 수도 있지만 preload는 문제 없음
    "L7D": ["bt_returns", "bt_equity_curve"],
    "L8": ["dataset_daily"],  # [Stage7] dataset_daily 또는 panel_merged_daily
    "L11": ["ranking_daily", "ohlcv_daily"],  # [Stage11] UI Payload Builder
}

STAGE_OUTPUTS = {
    "L0": ["universe_k200_membership_monthly"],
    "L1": ["ohlcv_daily"],
    "L1B": ["sector_map"],
    "L1D": ["market_regime"],  # [Stage5] 시장 국면 산출물
    "L2": ["fundamentals_annual"],
    "L3": ["panel_merged_daily"],
    "L4": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L5": ["pred_short_oos", "pred_long_oos", "model_metrics"],
    "L6": ["rebalance_scores", "rebalance_scores_summary"],
    "L6R": ["rebalance_scores", "rebalance_scores_summary"],  # [개선안 13번] L7 입력 포맷 동일
    "L7": ["bt_positions", "bt_returns", "bt_equity_curve", "bt_metrics", "selection_diagnostics", "bt_returns_diagnostics"],  # [Stage13] selection_diagnostics 및 bt_returns_diagnostics 추가
    "L7B": ["bt_sensitivity"],
    # L7C는 표준키 3개를 항상 생성하도록 수정했음
    "L7C": ["bt_vs_benchmark", "bt_benchmark_compare", "bt_benchmark_returns"],
    # L7D는 너 로더 호환 위해 l7d_stability alias도 같이 생성
    "L7D": ["l7d_stability", "bt_yearly_metrics", "bt_rolling_sharpe", "bt_drawdown_events"],
    "L8": ["ranking_daily", "ranking_snapshot"],  # [Stage7] Ranking 엔진 산출물
    "L11": ["ui_top_bottom_daily", "ui_equity_curves", "ui_snapshot", "ui_metrics"],  # [Stage11] UI Payload Builder
}

REQUIRED_COLS_BY_OUTPUT = {
    "universe_k200_membership_monthly": ["date", "ticker"],
    "ohlcv_daily": ["date", "ticker"],
    "sector_map": ["date", "ticker", "sector_name"],  # [Stage 4]
    "market_regime": ["date", "regime"],  # [Stage5] 시장 국면
    "fundamentals_annual": ["date", "ticker"],
    "panel_merged_daily": ["date", "ticker"],
    "dataset_daily": ["date", "ticker"],
    "cv_folds_short": ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"],
    "cv_folds_long": ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"],
    "pred_short_oos": ["date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"],
    "pred_long_oos": ["date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"],
    "model_metrics": ["horizon", "phase", "rmse"],

    "rebalance_scores": ["date", "ticker", "phase"],
    # ✅ 실제 생성 컬럼(너 로그 기준)에 맞춤
    "rebalance_scores_summary": [
        "date", "phase", "n_tickers", "coverage_vs_universe_pct",
        "score_short_missing", "score_long_missing", "score_ens_missing"
    ],

    "bt_positions": ["date", "phase", "ticker"],  # [Stage 4] sector_name은 선택적 (diversify.enabled일 때만)
    "bt_returns": ["date", "phase"],
    "bt_equity_curve": ["date", "phase"],
    "bt_metrics": ["phase", "net_total_return", "net_sharpe", "net_mdd"],

    "bt_sensitivity": ["phase", "net_total_return", "net_sharpe", "net_mdd"],

    "bt_vs_benchmark": ["date", "phase", "bench_return", "excess_return"],
    "bt_benchmark_compare": ["phase", "tracking_error_ann", "information_ratio"],
    "bt_benchmark_returns": ["date", "phase", "bench_return"],

    "ranking_daily": ["date", "ticker", "score_total", "rank_total"],  # [Stage7] Ranking 엔진

    "ui_top_bottom_daily": ["date", "top_list", "bottom_list"],  # [Stage11] UI Payload
    "ui_equity_curves": ["date", "strategy_ret", "bench_ret", "strategy_equity", "bench_equity", "excess_equity"],  # [Stage11] UI Payload
    "ui_snapshot": ["snapshot_date", "snapshot_type", "snapshot_rank", "ticker", "rank_total"],  # [Stage11] UI Payload
    "ui_metrics": ["total_return", "cagr", "vol", "sharpe", "mdd"],  # [Stage11] UI Payload

    "l7d_stability": [
        "phase", "year", "n_rebalances",
        "net_total_return", "net_vol_ann", "net_sharpe", "net_mdd", "net_hit_ratio",
        "date_start", "date_end", "net_return_col_used",
    ],
    "bt_yearly_metrics": [
        "phase", "year", "n_rebalances",
        "net_total_return", "net_vol_ann", "net_sharpe", "net_mdd", "net_hit_ratio",
        "date_start", "date_end", "net_return_col_used",
    ],
    "bt_rolling_sharpe": ["date", "phase", "net_rolling_sharpe"],
    "bt_drawdown_events": ["phase", "peak_date", "trough_date", "drawdown", "length_days"],

    "selection_diagnostics": ["date", "phase", "top_k", "eligible_count", "selected_count"],  # [Stage13] K_eff 복원
    "bt_returns_diagnostics": ["date", "phase"],  # [Stage13] regime/exposure 진단 컬럼 (결측 허용)
}

def _preload_required_inputs(stage_name: str, interim_dir: Path, artifacts: dict, *, base_interim_dir: Path = None, baseline_tag: str = None, input_tag: str = None, no_scan: bool = False):
    """
    [공통 프롬프트 v2] L2 입력은 base_interim_dir에서 로드 (fundamentals_annual.parquet)
    [Stage7] L8의 경우 dataset_daily 또는 panel_merged_daily 둘 다 허용
    [Stage11] L11의 경우 ranking_daily는 baseline_tag에서 로드, ohlcv_daily는 base_interim_dir에서 로드
    [TASK A-1] no_scan=True면 glob 스캔 금지 (baseline_tag 기반으로만 preload)
    [TASK B-stable] input_tag가 있으면 입력 산출물을 input_tag에서 먼저 찾고, 없으면 baseline_tag에서 찾음
    """
    required = REQUIRED_INPUTS.get(stage_name, [])

    # [TASK A-1] 타이밍 로그 (profile 모드)
    t_preload_start = time.time() if hasattr(_preload_required_inputs, '_profile_mode') else None

    # [Stage11] L11의 경우 ranking_daily는 baseline_tag에서 로드
    if stage_name == "L11":
        if "ranking_daily" not in artifacts:
            if baseline_tag and base_interim_dir is not None:
                # baseline_tag에서 로드 시도
                ranking_path = base_interim_dir / baseline_tag / "ranking_daily.parquet"
                if artifact_exists(ranking_path):
                    artifacts["ranking_daily"] = load_artifact(ranking_path)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded ranking_daily from baseline_tag: {ranking_path}")
                else:
                    # [TASK A-1] no_scan이면 스캔 금지
                    if no_scan:
                        raise KeyError(
                            f"{stage_name} requires 'ranking_daily' but --no-scan이 활성화되어 있고 "
                            f"baseline_tag({baseline_tag})에 파일이 없습니다. "
                            f"baseline_tag를 확인하거나 --no-scan을 해제하세요."
                        )
                    # baseline_tag에 없으면 최신 파일 사용
                    ranking_candidates = list(base_interim_dir.glob("*/ranking_daily.parquet"))
                    if ranking_candidates:
                        ranking_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        ranking_path = ranking_candidates[0]
                        artifacts["ranking_daily"] = load_artifact(ranking_path)
                        logger.info(f"[PRELOAD] {stage_name} <- loaded ranking_daily from latest: {ranking_path}")
                    else:
                        raise KeyError(f"{stage_name} requires 'ranking_daily' but not found in {base_interim_dir}")
            elif base_interim_dir is not None:
                # [TASK A-1] no_scan이면 스캔 금지
                if no_scan:
                    raise KeyError(
                        f"{stage_name} requires 'ranking_daily' but --no-scan이 활성화되어 있고 "
                        f"baseline_tag가 없습니다. baseline_tag를 지정하거나 --no-scan을 해제하세요."
                    )
                # baseline_tag가 없으면 최신 파일 사용
                ranking_candidates = list(base_interim_dir.glob("*/ranking_daily.parquet"))
                if ranking_candidates:
                    ranking_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    ranking_path = ranking_candidates[0]
                    artifacts["ranking_daily"] = load_artifact(ranking_path)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded ranking_daily from latest: {ranking_path}")
                else:
                    raise KeyError(f"{stage_name} requires 'ranking_daily' but not found in {base_interim_dir}")
            else:
                raise KeyError(f"{stage_name} requires 'ranking_daily' but base_interim_dir is None")
        if "ohlcv_daily" not in artifacts:
            # ohlcv_daily는 base_interim_dir에서 로드
            if base_interim_dir is not None:
                ohlcv_path = base_interim_dir / "ohlcv_daily.parquet"
                if artifact_exists(ohlcv_path):
                    artifacts["ohlcv_daily"] = load_artifact(ohlcv_path)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded ohlcv_daily from {ohlcv_path}")
                else:
                    raise KeyError(f"{stage_name} requires 'ohlcv_daily' but not found: {ohlcv_path}")
            else:
                raise KeyError(f"{stage_name} requires 'ohlcv_daily' but base_interim_dir is None")
        return

    # [Stage7] L8의 경우 dataset_daily 또는 panel_merged_daily 둘 중 하나만 있으면 됨
    if stage_name == "L8" and "dataset_daily" in required:
        # dataset_daily 또는 panel_merged_daily 중 하나라도 있으면 통과
        if "dataset_daily" not in artifacts and "panel_merged_daily" not in artifacts:
            # 둘 다 없으면 찾기 시도
            found = False
            for name in ["dataset_daily", "panel_merged_daily"]:
                base = interim_dir / name
                if artifact_exists(base):
                    artifacts[name] = load_artifact(base)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from interim: {name}")
                    found = True
                    break

            if not found:
                # base_interim_dir에서도 찾기 시도 (기존 run_tag에서)
                if base_interim_dir is not None:
                    for name in ["dataset_daily", "panel_merged_daily"]:
                        base = base_interim_dir / name
                        if artifact_exists(base):
                            artifacts[name] = load_artifact(base)
                            logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from base_interim_dir: {base}")
                            found = True
                            break

                if not found:
                    raise KeyError(
                        f"{stage_name} requires 'dataset_daily' or 'panel_merged_daily' but not found in: "
                        f"{interim_dir} or {base_interim_dir}"
                    )
        return

    for name in required:
        if name in artifacts:
            continue

        # L2 입력(fundamentals_annual)은 base_interim_dir에서 로드
        if name == "fundamentals_annual" and base_interim_dir is not None:
            base = base_interim_dir / name
            if artifact_exists(base):
                artifacts[name] = load_artifact(base)
                logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from base_interim_dir: {base}")
                continue
            else:
                raise KeyError(f"{stage_name} requires '{name}' but not found: {base}")

        # [TASK B-stable] L7 등 pipeline track의 경우 input_tag에서 먼저 찾고, 없으면 baseline_tag에서 찾기
        # input_tag/baseline_tag의 직접 경로 조회는 --no-scan일 때도 허용 (스캔은 하지 않음)
        if stage_name in ["L7", "L7B", "L7C", "L7D"] and base_interim_dir is not None:
            # 1순위: input_tag에서 찾기
            if input_tag:
                input_path = base_interim_dir / input_tag / name
                if artifact_exists(input_path):
                    artifacts[name] = load_artifact(input_path)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from input_tag: {input_path}")
                    continue
                else:
                    logger.debug(f"[PRELOAD] {stage_name} <- {name} not found in input_tag: {input_path}")

            # 2순위: baseline_tag에서 찾기
            if baseline_tag:
                baseline_path = base_interim_dir / baseline_tag / name
                if artifact_exists(baseline_path):
                    artifacts[name] = load_artifact(baseline_path)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from baseline_tag: {baseline_path}")
                    continue
                else:
                    logger.debug(f"[PRELOAD] {stage_name} <- {name} not found in baseline_tag: {baseline_path}")

            # [TASK B-stable] --no-scan이면 스캔 금지, 명확한 에러 메시지
            if no_scan:
                searched_locations = []
                if input_tag:
                    searched_locations.append(f"input_tag({input_tag})")
                if baseline_tag:
                    searched_locations.append(f"baseline_tag({baseline_tag})")
                locations_str = ", ".join(searched_locations) if searched_locations else "지정된 태그 없음"

                raise KeyError(
                    f"{stage_name} requires '{name}' but --no-scan이 활성화되어 있고 "
                    f"{locations_str}에 파일이 없습니다.\n"
                    f"  - input_tag 경로: {base_interim_dir / input_tag / name if input_tag else 'N/A'}\n"
                    f"  - baseline_tag 경로: {base_interim_dir / baseline_tag / name if baseline_tag else 'N/A'}\n"
                    f"해결 방법: --input-tag를 지정하거나 --no-scan을 해제하세요."
                )

            # 3순위: 스캔 허용 시 pipeline track의 최신 Stage에서 찾기
            candidates = list(base_interim_dir.glob(f"*/{name}.parquet"))
            if candidates:
                # 최신 파일 사용
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest_path = candidates[0]
                artifacts[name] = load_artifact(latest_path)
                logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from latest (scan): {latest_path}")
                continue

        base = interim_dir / name
        if artifact_exists(base):
            artifacts[name] = load_artifact(base)
            logger.info(f"[PRELOAD] {stage_name} <- loaded required input from interim: {name}")
        else:
            # [Phase 4] base_interim_dir에서도 찾기 시도 (기존 데이터 재사용)
            if base_interim_dir is not None:
                base_root = base_interim_dir / name
                if artifact_exists(base_root):
                    artifacts[name] = load_artifact(base_root)
                    logger.info(f"[PRELOAD] {stage_name} <- loaded {name} from base_interim_dir: {base_root}")
                else:
                    raise KeyError(f"{stage_name} requires '{name}' but not found in {interim_dir} or {base_interim_dir}")
            else:
                raise KeyError(f"{stage_name} requires '{name}' but not found: {base}")

def _get_file_hash(filepath: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _generate_runtime_report(
    run_tag: str,
    base_dir: Path,
    pipeline_total_time: Optional[float],
    stage_runtimes: Dict[str, Dict],
    stage_input_summaries: Dict[str, Dict],
    kpi_export_time: Optional[float],
    delta_export_time: Optional[float],
    target_stages: List[str],
    l2_reuse_info: Dict = None,
) -> None:
    """
    [Runtime 공통 규칙] Runtime 리포트 생성

    생성 파일:
    - reports/analysis/runtime__{run_tag}.md (사람이 읽는 요약)
    - reports/analysis/runtime__{run_tag}.csv (머신리더블)
    """
    from datetime import datetime

    analysis_dir = base_dir / "reports" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # MD 리포트 생성
    md_path = analysis_dir / f"runtime__{run_tag}.md"
    md_content = _generate_runtime_report_md(
        run_tag=run_tag,
        pipeline_total_time=pipeline_total_time,
        stage_runtimes=stage_runtimes,
        stage_input_summaries=stage_input_summaries,
        kpi_export_time=kpi_export_time,
        delta_export_time=delta_export_time,
        target_stages=target_stages,
        l2_reuse_info=l2_reuse_info or {},
    )
    md_path.write_text(md_content, encoding="utf-8")
    logger.info(f"[Runtime] 리포트 MD 저장: {md_path}")

    # CSV 리포트 생성
    csv_path = analysis_dir / f"runtime__{run_tag}.csv"
    csv_df = _generate_runtime_report_csv(
        run_tag=run_tag,
        pipeline_total_time=pipeline_total_time,
        stage_runtimes=stage_runtimes,
        stage_input_summaries=stage_input_summaries,
        kpi_export_time=kpi_export_time,
        delta_export_time=delta_export_time,
        target_stages=target_stages,
        l2_reuse_info=l2_reuse_info or {},
    )
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[Runtime] 리포트 CSV 저장: {csv_path}")

def _generate_runtime_report_md(
    run_tag: str,
    pipeline_total_time: Optional[float],
    stage_runtimes: Dict[str, Dict],
    stage_input_summaries: Dict[str, Dict],
    kpi_export_time: Optional[float],
    delta_export_time: Optional[float],
    target_stages: List[str],
    l2_reuse_info: Dict = None,
) -> str:
    """[Runtime 공통 규칙] Runtime 리포트 MD 생성"""
    from datetime import datetime

    lines = []
    lines.append(f"# Pipeline Runtime 리포트: {run_tag}\n")
    lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 전체 실행 시간
    lines.append("\n## 전체 실행 시간\n")
    if pipeline_total_time is not None:
        lines.append(f"- 총 실행 시간: {pipeline_total_time:.1f}초 ({pipeline_total_time/60:.1f}분)")
    else:
        lines.append("- 총 실행 시간: 기록되지 않음")

    # Stage별 실행 시간
    lines.append("\n## Stage별 실행 시간\n")
    lines.append("| Stage | 실행 시간 (초) | 비율 (%) |")
    lines.append("|-------|---------------|----------|")

    total_stage_time = sum(
        rt["runtime_sec"] for rt in stage_runtimes.values()
        if rt["runtime_sec"] is not None
    )

    for stage_name in target_stages:
        if stage_name in stage_runtimes:
            rt = stage_runtimes[stage_name]
            runtime_sec = rt.get("runtime_sec")
            if runtime_sec is not None:
                pct = (runtime_sec / total_stage_time * 100) if total_stage_time > 0 else 0
                lines.append(f"| {stage_name} | {runtime_sec:.1f} | {pct:.1f} |")
            else:
                lines.append(f"| {stage_name} | 기록되지 않음 | - |")

    # Export 시간
    lines.append("\n## 리포트 생성 시간\n")
    if kpi_export_time is not None:
        lines.append(f"- KPI Export: {kpi_export_time:.1f}초")
    if delta_export_time is not None:
        lines.append(f"- Delta Export: {delta_export_time:.1f}초")

    # 입력 데이터 요약
    lines.append("\n## 입력 데이터 요약\n")
    lines.append("| Stage | 입력명 | 행 수 | 크기 (bytes) | 비고 |")
    lines.append("|-------|--------|-------|---------------|------|")

    for stage_name in target_stages:
        if stage_name in stage_input_summaries:
            input_summary = stage_input_summaries[stage_name]
            if input_summary:
                for input_name, summary in input_summary.items():
                    rows = summary.get("rows", 0)
                    bytes_size = summary.get("bytes", 0)
                    source = summary.get("source", "")
                    note = ""
                    if source == "root_reuse":
                        note = f"루트 재사용 (해시: {summary.get('hash', 'N/A')}, 파일크기: {summary.get('file_size_bytes', 0):,} bytes, 수정시간: {summary.get('mtime', 'N/A')})"
                    lines.append(f"| {stage_name} | {input_name} | {rows:,} | {bytes_size:,} | {note} |")
            else:
                lines.append(f"| {stage_name} | (입력 없음) | - | - | - |")

    # [Stage14] L2 재사용 정보 상세 섹션
    if l2_reuse_info and l2_reuse_info.get("source") == "root_reuse":
        lines.append("\n## L2 재사용 정보 (루트 파일)\n")
        lines.append(f"- 파일 경로: `{l2_reuse_info.get('file_path', 'N/A')}`")
        lines.append(f"- 해시: `{l2_reuse_info.get('hash', 'N/A')}`")
        lines.append(f"- 파일 크기: {l2_reuse_info.get('file_size_bytes', 0):,} bytes")
        lines.append(f"- 수정 시간: {l2_reuse_info.get('mtime', 'N/A')}")
        lines.append(f"- 행 수: {l2_reuse_info.get('rows', 0):,}")
        lines.append(f"- 재사용 방식: 루트 파일 직접 로드 (DART API 호출 금지)")

    return "\n".join(lines)

def _generate_runtime_report_csv(
    run_tag: str,
    pipeline_total_time: Optional[float],
    stage_runtimes: Dict[str, Dict],
    stage_input_summaries: Dict[str, Dict],
    kpi_export_time: Optional[float],
    delta_export_time: Optional[float],
    target_stages: List[str],
    l2_reuse_info: Dict = None,
) -> pd.DataFrame:
    """[Runtime 공통 규칙] Runtime 리포트 CSV 생성"""
    rows = []

    # 전체 실행 시간
    rows.append({
        "category": "pipeline",
        "stage": "total",
        "metric": "total_wall_time_sec",
        "value": pipeline_total_time,
    })

    # Stage별 실행 시간
    for stage_name in target_stages:
        if stage_name in stage_runtimes:
            rt = stage_runtimes[stage_name]
            runtime_sec = rt.get("runtime_sec")
            rows.append({
                "category": "stage_runtime",
                "stage": stage_name,
                "metric": "runtime_sec",
                "value": runtime_sec,
            })

    # Export 시간
    if kpi_export_time is not None:
        rows.append({
            "category": "export",
            "stage": "kpi",
            "metric": "export_time_sec",
            "value": kpi_export_time,
        })
    if delta_export_time is not None:
        rows.append({
            "category": "export",
            "stage": "delta",
            "metric": "export_time_sec",
            "value": delta_export_time,
        })

    # 입력 데이터 요약
    for stage_name in target_stages:
        if stage_name in stage_input_summaries:
            input_summary = stage_input_summaries[stage_name]
            for input_name, summary in input_summary.items():
                rows.append({
                    "category": "input_summary",
                    "stage": stage_name,
                    "metric": f"{input_name}_rows",
                    "value": summary.get("rows", 0),
                })
                rows.append({
                    "category": "input_summary",
                    "stage": stage_name,
                    "metric": f"{input_name}_bytes",
                    "value": summary.get("bytes", 0),
                })
                # [Stage14] L2 재사용 정보 추가
                if summary.get("source") == "root_reuse":
                    rows.append({
                        "category": "input_summary",
                        "stage": stage_name,
                        "metric": f"{input_name}_source",
                        "value": "root_reuse",
                    })
                    rows.append({
                        "category": "input_summary",
                        "stage": stage_name,
                        "metric": f"{input_name}_file_size_bytes",
                        "value": summary.get("file_size_bytes", 0),
                    })
                    rows.append({
                        "category": "input_summary",
                        "stage": stage_name,
                        "metric": f"{input_name}_hash",
                        "value": summary.get("hash", "N/A"),
                    })
                    rows.append({
                        "category": "input_summary",
                        "stage": stage_name,
                        "metric": f"{input_name}_mtime",
                        "value": summary.get("mtime", "N/A"),
                    })

    # [Stage14] L2 재사용 정보 전체 기록
    if l2_reuse_info and l2_reuse_info.get("source") == "root_reuse":
        rows.append({
            "category": "l2_reuse",
            "stage": "L2",
            "metric": "file_path",
            "value": l2_reuse_info.get("file_path", "N/A"),
        })
        rows.append({
            "category": "l2_reuse",
            "stage": "L2",
            "metric": "hash",
            "value": l2_reuse_info.get("hash", "N/A"),
        })
        rows.append({
            "category": "l2_reuse",
            "stage": "L2",
            "metric": "file_size_bytes",
            "value": l2_reuse_info.get("file_size_bytes", 0),
        })
        rows.append({
            "category": "l2_reuse",
            "stage": "L2",
            "metric": "mtime",
            "value": l2_reuse_info.get("mtime", "N/A"),
        })
        rows.append({
            "category": "l2_reuse",
            "stage": "L2",
            "metric": "rows",
            "value": l2_reuse_info.get("rows", 0),
        })

    return pd.DataFrame(rows)

def _generate_runtime_profile_md(runtime_profile: pd.DataFrame, run_tag: str, total_time: Optional[float] = None) -> str:
    """[Stage13] 런타임 프로파일 MD 리포트 생성"""
    lines = []
    lines.append(f"# L7 Runtime Profile: {run_tag}\n")
    lines.append(f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if total_time is not None:
        lines.append(f"총 실행 시간: {total_time:.1f}초\n")

    lines.append("\n## 요약 통계\n")

    # 전체 통계
    total_rebalances = len(runtime_profile)
    avg_time = runtime_profile["rebalance_time_sec"].mean()
    median_time = runtime_profile["rebalance_time_sec"].median()
    max_time = runtime_profile["rebalance_time_sec"].max()
    min_time = runtime_profile["rebalance_time_sec"].min()

    lines.append(f"- 총 리밸런싱 수: {total_rebalances}")
    lines.append(f"- 평균 처리 시간: {avg_time:.3f}초")
    lines.append(f"- 중앙값 처리 시간: {median_time:.3f}초")
    lines.append(f"- 최대 처리 시간: {max_time:.3f}초")
    lines.append(f"- 최소 처리 시간: {min_time:.3f}초")

    # Phase별 통계
    lines.append("\n## Phase별 통계\n")
    phase_stats = runtime_profile.groupby("phase")["rebalance_time_sec"].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("median", "median"),
        ("max", "max"),
        ("min", "min"),
    ]).round(3)
    # [Stage14] to_markdown() 제거, to_string() 코드블록으로 대체
    try:
        if len(phase_stats) > 0:
            lines.append("```text")
            lines.append(phase_stats.to_string())
            lines.append("```")
        else:
            lines.append("_(no rows)_\n")
    except Exception as e:
        lines.append(f"_(표 생성 실패: {e})_\n")

    # Top 5 최악 처리 시간
    lines.append("\n## Top 5 최악 처리 시간\n")
    worst = runtime_profile.nlargest(5, "rebalance_time_sec")[
        ["date", "phase", "rebalance_time_sec", "k_eff", "top_k_used", "eligible_count"]
    ]
    # [Stage14] to_markdown() 제거, to_string() 코드블록으로 대체
    try:
        if len(worst) > 0:
            lines.append("```text")
            lines.append(worst.to_string(index=False))
            lines.append("```")
        else:
            lines.append("_(no rows)_\n")
    except Exception as e:
        lines.append(f"_(표 생성 실패: {e})_\n")

    # 병목 분석 (10줄 내 요약)
    lines.append("\n## 병목 분석\n")

    # 구간별 시간 분석
    total_sec = runtime_profile["rebalance_time_sec"].sum()
    avg_per_rebalance = avg_time

    # K_eff와 처리 시간의 상관관계
    if "k_eff" in runtime_profile.columns:
        corr = runtime_profile["rebalance_time_sec"].corr(runtime_profile["k_eff"])
        lines.append(f"- K_eff와 처리 시간 상관계수: {corr:.3f}")

    # Phase별 평균 시간 비교
    phase_avg = runtime_profile.groupby("phase")["rebalance_time_sec"].mean()
    if len(phase_avg) > 1:
        phase_diff = phase_avg.max() - phase_avg.min()
        lines.append(f"- Phase별 평균 시간 차이: {phase_diff:.3f}초")

    # 병목 구간 추정
    if avg_per_rebalance < 0.1:
        bottleneck = "빠름 (0.1초 미만)"
    elif avg_per_rebalance < 0.5:
        bottleneck = "정상 (0.1~0.5초)"
    elif avg_per_rebalance < 1.0:
        bottleneck = "보통 (0.5~1.0초)"
    else:
        bottleneck = "느림 (1.0초 이상)"

    lines.append(f"- 평균 처리 시간 평가: {bottleneck}")
    lines.append(f"- 예상 병목: 리밸런싱 루프 내부 처리 (selector, weight 계산, DataFrame 생성)")

    return "\n".join(lines)

def _generate_stage14_check_report(
    run_tag: str,
    baseline_tag: str,
    base_dir: Path,
    base_interim_dir: Path,
    command: str,
    args: argparse.Namespace,
    pipeline_total_time: Optional[float] = None,
) -> Path:
    """
    [Stage14] Stage14 체크리포트 생성 (6개 항목 자동 점검)

    Args:
        run_tag: 실행 태그
        baseline_tag: Baseline 태그
        base_dir: 프로젝트 루트 디렉토리
        base_interim_dir: Interim 데이터 디렉토리
        command: 실행 커맨드
        args: argparse.Namespace (실행 인자)
        pipeline_total_time: 전체 파이프라인 실행 시간 (초)

    Returns:
        생성된 체크리포트 파일 경로
    """
    lines = []
    lines.append("# Stage 14 완료 점검 리포트\n")
    lines.append(f"**Run Tag**: `{run_tag}`")
    lines.append(f"**Baseline Tag**: `{baseline_tag}`")
    lines.append(f"**점검 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---\n")

    # 체크 결과 저장
    check_results = {
        "env": False,
        "code": False,
        "execution": False,
        "artifacts": False,
        "reports": False,
    }
    failures = []

    # 1) 환경 확인
    lines.append("## 1) 환경 확인\n")
    expected_cwd = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")
    actual_cwd = Path.cwd()
    actual_base_dir = base_dir

    cwd_match = str(actual_cwd).replace("\\", "/") == str(expected_cwd).replace("\\", "/")
    base_dir_match = str(actual_base_dir).replace("\\", "/") == str(expected_cwd).replace("\\", "/")

    if cwd_match and base_dir_match:
        lines.append("**결과**: [PASS]")
        lines.append(f"\n- cwd: `{actual_cwd}` ✓")
        lines.append(f"- base_dir: `{actual_base_dir}` ✓")
        check_results["env"] = True
    else:
        lines.append("**결과**: [FAIL]")
        if not cwd_match:
            lines.append(f"\n- cwd 불일치: 예상=`{expected_cwd}`, 실제=`{actual_cwd}`")
            failures.append(f"환경: cwd 불일치 (예상: {expected_cwd}, 실제: {actual_cwd})")
        if not base_dir_match:
            lines.append(f"- base_dir 불일치: 예상=`{expected_cwd}`, 실제=`{actual_base_dir}`")
            failures.append(f"환경: base_dir 불일치 (예상: {expected_cwd}, 실제: {actual_base_dir})")
    lines.append("\n")

    # 2) 코드 확인 (tabulate 의존 제거)
    lines.append("## 2) 코드 확인 (tabulate 의존 제거)\n")

    # 실제 파일 읽어서 to_markdown() 호출 확인
    run_all_path = base_dir / "src" / "run_all.py"
    to_markdown_count = 0
    to_string_usage = False
    try_except_usage = False

    if run_all_path.exists():
        content = run_all_path.read_text(encoding="utf-8")
        # to_markdown() 호출 개수 (주석 제외)
        import re

        # 주석이 아닌 라인에서만 검색
        lines_content = content.split("\n")
        for i, line in enumerate(lines_content):
            stripped = line.strip()
            # 주석이 아닌 경우만 체크
            if not stripped.startswith("#") and ".to_markdown(" in line:
                to_markdown_count += 1

        # to_string() 사용 확인
        if "to_string(index=False)" in content or "to_string()" in content:
            to_string_usage = True

        # try/except 사용 확인 (_generate_runtime_profile_md 함수 내)
        func_start = content.find("def _generate_runtime_profile_md")
        if func_start >= 0:
            func_end = content.find("\ndef ", func_start + 1)
            if func_end < 0:
                func_end = len(content)
            func_content = content[func_start:func_end]
            if "try:" in func_content and "except" in func_content:
                try_except_usage = True
    else:
        failures.append("코드: src/run_all.py 파일을 찾을 수 없습니다")

    if to_markdown_count == 0 and to_string_usage and try_except_usage:
        lines.append("**결과**: [PASS]")
        lines.append(f"\n- to_markdown() 호출 개수: {to_markdown_count}개 ✓")
        lines.append(f"- to_string() 사용: {'✓' if to_string_usage else '✗'}")
        lines.append(f"- try/except 예외 처리: {'✓' if try_except_usage else '✗'}")
        check_results["code"] = True
    else:
        lines.append("**결과**: [FAIL]")
        if to_markdown_count > 0:
            lines.append(f"\n- to_markdown() 호출이 {to_markdown_count}개 남아있습니다")
            failures.append(f"코드: to_markdown() 호출 {to_markdown_count}개 남아있음 (src/run_all.py 확인 필요)")
        if not to_string_usage:
            lines.append("- to_string() 사용이 확인되지 않습니다")
            failures.append("코드: to_string() 사용 미확인 (src/run_all.py:_generate_runtime_profile_md 확인 필요)")
        if not try_except_usage:
            lines.append("- try/except 예외 처리가 없습니다")
            failures.append("코드: try/except 예외 처리 미확인 (src/run_all.py:_generate_runtime_profile_md 확인 필요)")
    lines.append("\n")

    # 3) 실행 로그 요약
    lines.append("## 3) 실행 로그 요약\n")
    lines.append("**결과**: [PASS]")
    lines.append(f"\n- RUN_TAG: `{run_tag}`")
    lines.append(f"- BASELINE_TAG: `{baseline_tag}`")
    lines.append(f"- --skip-l2: {'✓' if args.skip_l2 else '✗'}")
    lines.append(f"- --force-rebuild: {'✓' if args.force_rebuild else '✗'}")
    lines.append(f"- --from: `{args.from_stage}`")
    lines.append(f"- --to: `{args.to_stage}`")
    if pipeline_total_time is not None:
        lines.append(f"- 총 실행 시간: {pipeline_total_time:.1f}초")
    lines.append("\n- [Pipeline Completed Successfully] 메시지: 확인됨 (로그 기준)")
    check_results["execution"] = True
    lines.append("\n")

    # 4) 산출물 존재 확인
    lines.append("## 4) 산출물 존재 확인\n")
    run_tag_dir = base_interim_dir / run_tag
    required_outputs = [
        "bt_returns.parquet",
        "bt_positions.parquet",
        "bt_metrics.parquet",
        "selection_diagnostics.parquet",
    ]

    lines.append("| 산출물 | 존재 | 경로 |")
    lines.append("|---|---|---|")
    artifacts_exist = True
    missing_artifacts = []
    for output in required_outputs:
        output_path = run_tag_dir / output
        exists = output_path.exists()
        status = "[OK]" if exists else "[MISSING]"
        rel_path = output_path.relative_to(base_dir) if exists else "N/A"
        lines.append(f"| {output} | {status} | `{rel_path}` |")
        if not exists:
            artifacts_exist = False
            missing_artifacts.append(str(output_path))

    if artifacts_exist:
        lines.append("\n**결과**: [PASS]")
        check_results["artifacts"] = True
    else:
        lines.append("\n**결과**: [FAIL]")
        failures.append(f"산출물: 다음 파일이 없습니다 - {', '.join(missing_artifacts)}")
    lines.append("\n")

    # 5) 리포트 파일 존재 확인
    lines.append("## 5) 리포트 파일 존재 확인\n")

    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    kpi_md = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.md"
    delta_csv = base_dir / "reports" / "delta" / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
    delta_md = base_dir / "reports" / "delta" / f"delta_report__{baseline_tag}__vs__{run_tag}.md"

    reports = [
        ("KPI CSV", kpi_csv),
        ("KPI MD", kpi_md),
        ("Delta CSV", delta_csv),
        ("Delta MD", delta_md),
    ]

    lines.append("| 리포트 | 존재 | 경로 |")
    lines.append("|---|---|---|")
    reports_exist = True
    missing_reports = []
    for name, path in reports:
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        rel_path = path.relative_to(base_dir) if exists else "N/A"
        lines.append(f"| {name} | {status} | `{rel_path}` |")
        if not exists:
            reports_exist = False
            missing_reports.append(str(path))

    if reports_exist:
        lines.append("\n**결과**: [PASS]")
        check_results["reports"] = True
    else:
        lines.append("\n**결과**: [FAIL]")
        failures.append(f"리포트: 다음 파일이 없습니다 - {', '.join(missing_reports)}")
    lines.append("\n")

    # 6) 최종 판정
    lines.append("## 6) 최종 판정\n")
    all_pass = all(check_results.values())

    if all_pass:
        lines.append("**결과**: [PASS]")
        lines.append("\n**상세**:")
        lines.append("- ✓ 환경 확인 통과")
        lines.append("- ✓ 코드 확인 통과 (tabulate 의존 제거)")
        lines.append("- ✓ 실행 로그 확인 통과")
        lines.append("- ✓ 산출물 존재 확인 통과")
        lines.append("- ✓ 리포트 파일 존재 확인 통과")
    else:
        lines.append("**결과**: [FAIL]")
        lines.append("\n**실패 항목**:")
        for key, passed in check_results.items():
            status = "✓" if passed else "✗"
            lines.append(f"- {status} {key}")

        lines.append("\n**누락 항목/원인**:")
        for failure in failures:
            lines.append(f"- {failure}")

        lines.append("\n**수정해야 할 파일 위치**:")
        if not check_results["code"]:
            lines.append("- `src/run_all.py` - _generate_runtime_profile_md() 함수")
        if not check_results["artifacts"]:
            lines.append(f"- `data/interim/{run_tag}/` - L7 산출물 생성 확인 필요")
        if not check_results["reports"]:
            lines.append("- `reports/kpi/`, `reports/delta/` - 리포트 생성 스크립트 확인 필요")

    # 리포트 저장
    reports_dir = base_dir / "reports" / "stages"
    reports_dir.mkdir(parents=True, exist_ok=True)
    check_report_path = reports_dir / f"check__stage14__{run_tag}.md"
    check_report_path.write_text("\n".join(lines), encoding="utf-8")

    return check_report_path

def _cleanup_run_tag_artifacts(run_tag: str, base_interim_dir: Path, base_dir: Path, baseline_tag: str = "baseline_prerefresh_20251219_143636"):
    """
    [공통 프롬프트 v2] Stage 실행 전 run_tag 폴더/리포트 삭제

    삭제 대상:
    - data/interim/{run_tag}/ (전체 폴더 삭제)
    - reports/kpi/kpi_table__{run_tag}.* (CSV, MD 파일)
    - reports/delta/delta_*__vs__{run_tag}.* (CSV, MD 파일)

    baseline 관련 파일은 삭제 금지 (절대 규칙)

    Windows 경로 구분자: Python Path 객체와 shutil을 사용하므로 크로스 플랫폼 호환성 보장
    """
    # [공통 프롬프트 v2] baseline 태그와 run_tag가 같으면 삭제 금지 (안전장치)
    if run_tag == baseline_tag:
        logger.warning(f"[공통 프롬프트 v2] run_tag가 baseline_tag와 동일합니다. 삭제를 건너뜁니다: {run_tag}")
        return

    deleted_count = 0

    # 1. data/interim/{run_tag} 폴더 삭제
    run_tag_dir = base_interim_dir / run_tag
    if run_tag_dir.exists() and run_tag_dir.is_dir():
        logger.info(f"[공통 프롬프트 v2] 기존 run_tag 폴더 삭제: {run_tag_dir}")
        shutil.rmtree(run_tag_dir, ignore_errors=True)
        deleted_count += 1

    # 2. reports/kpi/kpi_table__{run_tag}.* 삭제
    reports_kpi_dir = base_dir / "reports" / "kpi"
    if reports_kpi_dir.exists():
        for pattern in [f"kpi_table__{run_tag}.csv", f"kpi_table__{run_tag}.md"]:
            file_path = reports_kpi_dir / pattern
            if file_path.exists():
                logger.info(f"[공통 프롬프트 v2] 기존 KPI 리포트 삭제: {file_path}")
                file_path.unlink(missing_ok=True)
                deleted_count += 1

    # 3. reports/delta/delta_*__vs__{run_tag}.* 삭제 (baseline 관련 제외)
    reports_delta_dir = base_dir / "reports" / "delta"
    if reports_delta_dir.exists():
        for file_path in reports_delta_dir.glob(f"delta_*__vs__{run_tag}.*"):
            # baseline 관련 파일은 삭제 금지 (절대 규칙)
            if baseline_tag not in str(file_path):
                logger.info(f"[공통 프롬프트 v2] 기존 Delta 리포트 삭제: {file_path}")
                file_path.unlink(missing_ok=True)
                deleted_count += 1
            else:
                logger.debug(f"[공통 프롬프트 v2] baseline 관련 파일 보호: {file_path}")

        # 추가 안전장치: baseline 태그가 포함된 파일은 절대 삭제하지 않음
        for file_path in reports_delta_dir.glob(f"delta_*__{baseline_tag}__*.*"):
            logger.debug(f"[공통 프롬프트 v2] baseline 파일 보호 (절대 삭제 금지): {file_path}")

    if deleted_count > 0:
        logger.info(f"[공통 프롬프트 v2] run_tag 정리 완료: {run_tag} (삭제된 항목: {deleted_count}개)")
    else:
        logger.info(f"[공통 프롬프트 v2] run_tag 정리 완료: {run_tag} (삭제할 항목 없음)")

def _verify_l2_reuse(base_interim_dir: Path, baseline_tag: str = "baseline_prerefresh_20251219_143636") -> Optional[str]:
    """
    [공통 프롬프트 v2] L2 재사용 검증: fundamentals_annual.parquet 해시 확인

    Returns:
        해시값 (실행 전), None이면 파일 없음

    검증 규칙:
    - 실행 전 해시를 기록하고, 실행 후 해시와 비교하여 변경 여부 확인
    - 변경되면 즉시 중단 (규칙 위반)
    - L2 재무데이터는 절대 재생성되어서는 안 됨 (DART API 호출 금지)
    """
    l2_file = base_interim_dir / "fundamentals_annual.parquet"
    if not l2_file.exists():
        logger.error(f"[L2 재사용 검증] fundamentals_annual.parquet 파일 없음: {l2_file}")
        logger.error("[L2 재사용 검증] L2 재무데이터는 무조건 기존 파일을 재사용해야 합니다.")
        logger.error("[L2 재사용 검증] 파일이 없다면 이전 실행에서 생성된 파일을 확인하거나, 수동으로 준비해야 합니다.")
        return None

    # 파일 크기와 수정 시간도 함께 기록 (추가 검증 정보)
    file_size = l2_file.stat().st_size
    mtime = os.path.getmtime(l2_file)
    from datetime import datetime
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    hash_before = _get_file_hash(l2_file)
    logger.info(f"[공통 프롬프트 v2] L2 재사용 검증 시작")
    logger.info(f"[공통 프롬프트 v2] L2 파일 경로: {l2_file}")
    logger.info(f"[공통 프롬프트 v2] L2 실행 전 해시: {hash_before[:16]}... (전체: {hash_before})")
    logger.info(f"[공통 프롬프트 v2] L2 파일 크기: {file_size:,} bytes")
    logger.info(f"[공통 프롬프트 v2] L2 파일 수정 시간: {mtime_str}")
    logger.info(f"[공통 프롬프트 v2] L2 재사용 규칙: 파일이 변경되면 즉시 중단됩니다.")
    return hash_before

def _maybe_skip_stage(stage_name: str, interim_dir: Path, artifacts: dict, *, force: bool, skip_if_exists: bool, force_rebuild: bool = False) -> bool:
    """
    [공통 프롬프트 v2] skip_if_exists/캐시/기존 산출물 재사용 로직을 전부 무시한다.
    - L2 산출물(fundamentals_annual.parquet)만 예외로 "재사용 고정".
    - L2 제외 모든 산출물은 "기존 파일이 있더라도" 반드시 새로 만든다.
    """
    # L2는 항상 스킵 (재사용 고정) - force-rebuild여도 재사용
    if stage_name == "L2":
        logger.info("[공통 프롬프트 v2] L2 재사용 고정: L2는 항상 기존 파일을 재사용합니다.")
        return True

    # [공통 프롬프트 v2] force-rebuild가 True면 skip_if_exists 무시하고 항상 재생성
    if force_rebuild:
        logger.debug(f"[공통 프롬프트 v2] force-rebuild=True: {stage_name} 스킵하지 않음 (재생성)")
        return False

    # force 또는 skip_if_exists=False면 재생성
    if force or (not skip_if_exists):
        return False

    # skip_if_exists=True이고 force-rebuild=False일 때만 기존 파일 재사용 (L2 제외)
    outs = STAGE_OUTPUTS.get(stage_name, [])
    if not outs:
        return False
    bases = [(o, interim_dir / o) for o in outs]
    if all(artifact_exists(b) for _, b in bases):
        for o, b in bases:
            artifacts[o] = load_artifact(b)
        logger.info(f"[SKIP] {stage_name} -> loaded from interim ({', '.join(outs)})")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Integrated Pipeline Runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--from", dest="from_stage", type=str, default="L0")
    parser.add_argument("--to", dest="to_stage", type=str, default="L7D")
    parser.add_argument("--stage", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default="default_run")
    parser.add_argument("--run-tag", type=str, required=True,
                       help="[Stage0 필수] Run tag for artifact isolation (e.g., stage0_rebuild_tagged_YYYYMMDD_HHMMSS)")
    parser.add_argument("--legacy-root-save", action="store_true",
                       help="[DEPRECATED] Save artifacts to data/interim root instead of data/interim/{run_tag}/")
    parser.add_argument("--strict-params", action="store_true",
                       help="Fail pipeline if config parameters don't match actual usage (e.g., cost_bps mismatch)")
    parser.add_argument("--skip-l2", action="store_true", default=True,
                       help="[Stage0 필수] Skip L2 stage (reuse existing fundamentals_annual.parquet, DART 호출 금지)")
    parser.add_argument("--force-rebuild", action="store_true", default=False,
                       help="[공통 프롬프트 v2] 강제 Rebuild: 기존 산출물 무시하고 새로 생성 (L2 제외, 기본값: False)")
    parser.add_argument("--baseline-tag", type=str, default="baseline_prerefresh_20251219_143636",
                       help="[공통 프롬프트 v2] Baseline 태그 (비교 대상, 수정 금지)")
    parser.add_argument("--input-tag", type=str, default=None,
                       help="[TASK B-stable] 입력 산출물 소스 태그 (rebalance_scores 등 입력 파일이 있는 태그)")
    parser.add_argument("--max-rebalances", type=int, default=None,
                       help="[Stage13] 스모크 테스트용: 최근 N개 리밸런싱만 실행 (예: --max-rebalances 10)")
    parser.add_argument("--no-scan", action="store_true", default=None,
                       help="[TASK A-1] data/interim 전체 스캔 금지 (baseline_tag 기반으로만 preload)")
    parser.add_argument("--no-export", action="store_true", default=False,
                       help="[TASK A-1] KPI/Delta/MD 생성 비활성화 (디버그/속도 확인용)")
    parser.add_argument("--no-md", action="store_true", default=False,
                       help="[TASK A-1] MD 렌더링만 비활성화 (특히 pandas.to_markdown)")
    parser.add_argument("--profile", action="store_true", default=False,
                       help="[TASK A-1] 각 단계의 경과시간을 reports/analysis/runtime_breakdown__{run_tag}.csv로 저장")
    args = parser.parse_args()

    # [TASK A-1] baseline_tag가 명시되면 기본적으로 --no-scan 활성화
    if args.no_scan is None:
        # baseline_tag가 기본값이 아니면 (명시적으로 주어진 경우) no-scan 활성화
        if args.baseline_tag != "baseline_prerefresh_20251219_143636":
            args.no_scan = True
            logger.info(f"[TASK A-1] baseline_tag가 명시되어 --no-scan 자동 활성화: {args.baseline_tag}")
        else:
            args.no_scan = False

    cfg = load_config(args.config)

    stage_names = list(STAGES.keys())

    # stage name normalize
    from_stage = (args.from_stage or "L0").upper()
    to_stage = (args.to_stage or "L7D").upper()
    one_stage = (args.stage.upper() if args.stage else None)

    if one_stage:
        if one_stage not in STAGES:
            logger.error(f"Invalid stage: {one_stage}. available={stage_names}")
            sys.exit(1)
        target_stages = [one_stage]
    else:
        if from_stage not in stage_names or to_stage not in stage_names:
            logger.error(f"Invalid range: from={from_stage}, to={to_stage}. available={stage_names}")
            sys.exit(1)
        start_idx = stage_names.index(from_stage)
        end_idx = stage_names.index(to_stage)
        if start_idx > end_idx:
            logger.error(f"Invalid range order: from={from_stage} after to={to_stage}.")
            sys.exit(1)
        target_stages = stage_names[start_idx:end_idx + 1]

    # Run tag 결정 (logger 호출 전에)
    # [Stage0] --run-tag는 필수이므로 항상 사용
    run_tag = args.run_tag

    # Interim 디렉토리 결정 (logger 호출 전에)
    base_interim_dir = get_path(cfg, "data_interim")
    base_dir = get_path(cfg, "base_dir")
    if args.legacy_root_save:
        interim_dir = base_interim_dir
    else:
        interim_dir = base_interim_dir / run_tag

    t_pipeline_start = time.time()  # [Stage13] 전체 파이프라인 시작 시간

    # [Stage14] 즉시 출력 보장
    print("[Stage14] 파이프라인 시작...", flush=True)

    logger.info("=== RUNNER ===")
    logger.info(f"BASE_DIR={base_dir}")
    logger.info(f"INTERIM_DIR={interim_dir}")
    logger.info(f"RUN_TAG={run_tag}")
    logger.info(f"BASELINE_TAG={args.baseline_tag}")
    logger.info(f"FROM={from_stage}, TO={to_stage}, FORCE={args.force}, FORCE_REBUILD={args.force_rebuild}")
    logger.info(f"Target Stages: {target_stages}")
    logger.info(f"Config: {args.config}")

    # [Stage14] 로깅 즉시 출력
    sys.stdout.flush()

    if args.dry_run:
        logger.info("[Dry-Run] Skipping actual execution.")
        return

    # [Stage0] L2 해시 검증 (실행 전)
    l2_hash_before = _verify_l2_reuse(base_interim_dir, args.baseline_tag)

    # [Stage0] Stage 실행 전 run_tag 폴더/리포트 삭제 (항상 수행, force-rebuild와 무관)
    _cleanup_run_tag_artifacts(run_tag, base_interim_dir, base_dir, args.baseline_tag)

    # [Stage0] skip-l2 플래그 확인 및 로깅
    if args.skip_l2:
        logger.info("[Stage0] --skip-l2=True: L2 Stage는 기존 fundamentals_annual.parquet만 재사용 (DART 호출 금지)")

    if args.legacy_root_save:
        logger.warning("[DEPRECATED] Using legacy root save mode. Artifacts will be saved to data/interim root.")
    else:
        logger.info(f"[TAG-BASED] Artifacts will be saved to: {interim_dir}")

    interim_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _resolve_section(cfg, "run")
    save_formats = run_cfg.get("save_formats", ["parquet", "csv"])
    fail_on_validation_error = bool(run_cfg.get("fail_on_validation_error", True))
    write_meta = bool(run_cfg.get("write_meta", True))
    # [공통 프롬프트 v2] force-rebuild면 skip_if_exists 무시 (항상 재생성)
    skip_if_exists = bool(run_cfg.get("skip_if_exists", True)) and not args.force_rebuild
    if args.force_rebuild:
        logger.info("[공통 프롬프트 v2] force-rebuild=True: skip_if_exists 무시, 모든 Stage 재생성 (L2 제외)")

    artifacts: Dict[str, Any] = {}
    # [Stage 2] interim_dir과 run_tag를 artifacts에 저장 (L5에서 사용)
    artifacts["_interim_dir"] = interim_dir
    artifacts["_run_tag"] = run_tag
    # [Stage13] 스모크 테스트 옵션 전달
    if args.max_rebalances is not None:
        artifacts["_max_rebalances"] = int(args.max_rebalances)
        logger.info(f"[Stage13] 스모크 테스트 모드: 최근 {args.max_rebalances}개 리밸런싱만 실행")

    # [Stage13] L7 타이밍 저장용
    l7_start_time = None
    l7_end_time = None

    # [Runtime 공통 규칙] Stage별 실행 시간 기록
    stage_runtimes = {}  # {stage_name: (start_time, end_time, runtime_sec)}
    stage_input_summaries = {}  # {stage_name: {input_name: {rows, bytes}}}
    l2_reuse_info = {}  # [Stage14] L2 재사용 정보 (해시, 크기, mtime) - L2 실행 시에만 채워짐

    # [TASK A-1] 타이밍 기록용 딕셔너리 (초기화) - Stage 루프 전에 초기화
    runtime_breakdown = {}

    # [개선안 13번] 랭킹 기반 전략 모드:
    # - L5/L6(모델 예측 기반 score_ens) 대신 L6R(랭킹 기반 score_ens)을 사용
    l7_cfg = _resolve_section(cfg, "l7")
    signal_source = str(l7_cfg.get("signal_source", "model")).strip().lower()
    ranking_mode = signal_source in {"ranking", "rank", "ranking_only", "rank_only"}
    if ranking_mode:
        logger.info("[개선안 13번] signal_source=ranking: L5/L6를 스킵하고 L6R을 사용합니다. (L7 입력=rebalance_scores)")

    for stage_name in target_stages:
        try:
            # [개선안 13번] ranking 모드에서는 모델 트랙(L5/L6)을 실행하지 않는다.
            if ranking_mode and stage_name in {"L5", "L6"}:
                logger.info(f"[개선안 13번] SKIP {stage_name}: ranking 모드")
                continue

            # [Runtime 공통 규칙] Stage 시작 시간 기록
            stage_start_time = time.time()
            stage_runtimes[stage_name] = {"start_time": stage_start_time, "end_time": None, "runtime_sec": None}

            # [Runtime 공통 규칙] 입력 데이터 요약 기록
            input_summary = {}
            required_inputs = REQUIRED_INPUTS.get(stage_name, [])
            for input_name in required_inputs:
                if input_name in artifacts:
                    df_input = artifacts[input_name]
                    if isinstance(df_input, pd.DataFrame):
                        input_summary[input_name] = {
                            "rows": len(df_input),
                            "bytes": df_input.memory_usage(deep=True).sum(),
                        }
            stage_input_summaries[stage_name] = input_summary

            # [Stage13] L7 시작 시간 기록 (기존 코드 유지)
            if stage_name == "L7":
                l7_start_time = time.time()
                logger.info(f"[Stage13] L7 백테스트 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # [Stage0] L2는 항상 재사용 (force-rebuild여도, DART 호출 금지)
            if stage_name == "L2":
                if not args.skip_l2:
                    logger.warning("[Stage0] --skip-l2가 False입니다. L2 Stage는 항상 재사용해야 합니다. 강제로 재사용 모드로 전환합니다.")
                logger.info("[Stage0] L2 재사용 고정: 기존 fundamentals_annual.parquet 로드 (DART 호출 금지)")
                source_file = base_interim_dir / "fundamentals_annual.parquet"

                if not artifact_exists(source_file):
                    raise RuntimeError(
                        f"[L2 재사용 규칙 위반] 기존 fundamentals_annual.parquet 파일이 없습니다: {source_file}\n"
                        "L2 재무데이터는 무조건 기존 파일을 재사용해야 하며, API 재호출로 재생성할 수 없습니다.\n"
                        "파일이 없다면 이전 실행에서 생성된 파일을 확인하거나, 수동으로 준비해야 합니다."
                    )

                # [공통 프롬프트 v2] L2 해시 검증 (로드 전)
                if l2_hash_before is None:
                    l2_hash_before = _get_file_hash(source_file)
                    logger.info(f"[L2 재사용 검증] 실행 전 해시 기록: {l2_hash_before[:16]}...")

                df = load_artifact(source_file)
                artifacts["fundamentals_annual"] = df
                mtime = os.path.getmtime(source_file)
                from datetime import datetime
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"[L2 재사용] 파일 수정 시간: {mtime_str}, 행 수: {len(df)}")

                # [공통 프롬프트 v2] L2 해시 재검증 (로드 후, 변경 여부 확인)
                if l2_hash_before is not None:
                    l2_hash_after = _get_file_hash(source_file)
                    file_size_after = source_file.stat().st_size
                    mtime_after = os.path.getmtime(source_file)
                    from datetime import datetime
                    mtime_str_after = datetime.fromtimestamp(mtime_after).strftime("%Y-%m-%d %H:%M:%S")

                    if l2_hash_before != l2_hash_after:
                        logger.error(f"[공통 프롬프트 v2] L2 재사용 규칙 위반 감지!")
                        logger.error(f"[공통 프롬프트 v2] 실행 전 해시: {l2_hash_before[:16]}... (전체: {l2_hash_before})")
                        logger.error(f"[공통 프롬프트 v2] 로드 후 해시: {l2_hash_after[:16]}... (전체: {l2_hash_after})")
                        logger.error(f"[공통 프롬프트 v2] 파일 크기: {file_size_after:,} bytes")
                        logger.error(f"[공통 프롬프트 v2] 파일 수정 시간: {mtime_str_after}")
                        raise RuntimeError(
                            f"[공통 프롬프트 v2] L2 재사용 규칙 위반: fundamentals_annual.parquet 파일이 변경되었습니다!\n"
                            f"실행 전 해시: {l2_hash_before[:16]}... (전체: {l2_hash_before})\n"
                            f"로드 후 해시: {l2_hash_after[:16]}... (전체: {l2_hash_after})\n"
                            f"파일 크기: {file_size_after:,} bytes\n"
                            f"수정 시간: {mtime_str_after}\n"
                            "L2 재무데이터는 절대 재생성되어서는 안 됩니다. DART API 호출 금지."
                        )
                    logger.info(f"[공통 프롬프트 v2] L2 재사용 검증: 해시 일치 확인 완료 (변경 없음)")
                    logger.info(f"[공통 프롬프트 v2] L2 파일 크기: {file_size_after:,} bytes (변경 없음)")
                    logger.info(f"[공통 프롬프트 v2] L2 파일 수정 시간: {mtime_str_after} (변경 없음)")

                # [Stage14] L2 재사용 정보 저장 (Runtime 리포트용)
                l2_reuse_info = {
                    "hash": l2_hash_before if l2_hash_before else l2_hash_after,
                    "file_size_bytes": file_size_after if l2_hash_before else source_file.stat().st_size,
                    "mtime": mtime_str_after if l2_hash_before else mtime_str,
                    "rows": len(df),
                    "source": "root_reuse",  # 루트 재사용 명시
                    "file_path": str(source_file),
                }

                # [Stage14] L2 입력 요약에 재사용 정보 포함
                stage_input_summaries["L2"] = {
                    "fundamentals_annual": {
                        "rows": len(df),
                        "bytes": df.memory_usage(deep=True).sum(),
                        "source": "root_reuse",
                        "file_size_bytes": l2_reuse_info["file_size_bytes"],
                        "hash": l2_reuse_info["hash"][:16] + "...",  # 짧은 해시만 표시
                        "mtime": l2_reuse_info["mtime"],
                    }
                }

                # [Runtime 공통 규칙] L2는 재사용이므로 실행 시간은 매우 짧음 (로드 시간만)
                stage_end_time = time.time()
                stage_runtime_sec = stage_end_time - stage_start_time
                stage_runtimes["L2"] = {
                    "start_time": stage_start_time,
                    "end_time": stage_end_time,
                    "runtime_sec": stage_runtime_sec,
                }
                logger.info(f"[Runtime] L2 완료 (재사용): {stage_runtime_sec:.3f}초")

                continue

            # [TASK A-1] Preload 타이밍 기록
            t_preload_start = time.time() if args.profile else None
            _preload_required_inputs(stage_name, interim_dir, artifacts, base_interim_dir=base_interim_dir, baseline_tag=args.baseline_tag, input_tag=args.input_tag, no_scan=args.no_scan)
            if t_preload_start is not None:
                preload_time = time.time() - t_preload_start
                runtime_breakdown[f"{stage_name}_preload_sec"] = preload_time
                logger.info(f"[TASK A-1] {stage_name} Preload 시간: {preload_time:.3f}초")

            if _maybe_skip_stage(stage_name, interim_dir, artifacts, force=args.force, skip_if_exists=skip_if_exists, force_rebuild=args.force_rebuild):
                continue

            func = STAGES[stage_name]
            outputs, stage_warnings = func(cfg, artifacts, force=args.force)

            if not isinstance(outputs, dict) or not outputs:
                raise ValueError(f"{stage_name} must return dict[str, DataFrame] with at least one output.")

            for out_name, df in outputs.items():
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"{stage_name}:{out_name} output must be DataFrame. got={type(df)}")

                # [Stage0] 태그 폴더에 저장 (interim_dir은 이미 태그 폴더 경로)
                # [Phase 7 Step 1] BT20/BT120 구분: holding_days에 따라 파일명 변경 (artifacts 키는 유지)
                out_base = interim_dir / out_name
                if stage_name == "L7" and out_name in ["bt_metrics", "bt_regime_metrics"]:
                    l7_cfg = _resolve_section(cfg, "l7")
                    holding_days = int(l7_cfg.get("holding_days", 20))
                    if holding_days == 120:
                        # BT120: 파일명에 _bt120 접미사 추가 (artifacts 키는 out_name 그대로 유지)
                        out_base = interim_dir / f"{out_name}_bt120"
                        logger.debug(f"[{stage_name}:{out_name}] BT120 detected: Saving to: {out_base} (artifacts key: {out_name})")
                logger.debug(f"[{stage_name}:{out_name}] Saving to: {out_base}")

                required_cols = REQUIRED_COLS_BY_OUTPUT.get(out_name, None)
                # [Stage13] bt_returns_diagnostics는 진단 컬럼이므로 결측 허용 (max_missing_pct=None)
                max_missing = None if out_name == "bt_returns_diagnostics" else 95.0
                # [Stage13] bt_returns의 경우 regime/exposure는 옵션 컬럼으로 처리 (market_regime 없을 수 있음)
                # [Phase 4 Step 2] XGBoost 모델은 alpha 튜닝이 없으므로 alpha_val_ic_rank 컬럼이 모두 결측값이 됨
                if out_name == "bt_returns":
                    optional_cols = ["regime", "exposure"]
                elif stage_name == "L5" and out_name == "model_metrics":
                    optional_cols = ["alpha_val_ic_rank"]
                else:
                    optional_cols = None
                result = validate_df(
                    df,
                    stage=stage_name,
                    required_cols=required_cols,
                    max_missing_pct=max_missing,
                    optional_cols=optional_cols,
                )
                all_warnings = (stage_warnings or []) + (result.warnings or [])

                if fail_on_validation_error:
                    raise_if_invalid(result, stage=f"{stage_name}:{out_name}")

                # [공통 프롬프트 v2] force-rebuild면 항상 overwrite (skip_if_exists 무시)
                save_force = args.force or args.force_rebuild
                if args.force_rebuild:
                    logger.debug(f"[공통 프롬프트 v2] force-rebuild=True: {out_name} 강제 저장 (overwrite)")
                save_artifact(df, out_base, force=save_force, formats=save_formats)

                quality = {}

                if stage_name == "L3" and out_name == "panel_merged_daily":
                    quality["fundamental"] = fundamental_coverage_report(df)

                if stage_name == "L4" and out_name == "dataset_daily":
                    l4 = _resolve_section(cfg, "l4")
                    quality["walkforward"] = walkforward_quality_report(
                        dataset=df,
                        cv_short=outputs.get("cv_folds_short", artifacts.get("cv_folds_short")),
                        cv_long=outputs.get("cv_folds_long", artifacts.get("cv_folds_long")),
                        horizon_short=int(l4.get("horizon_short", 20)),
                        horizon_long=int(l4.get("horizon_long", 120)),
                        step_days=int(l4.get("step_days", 20)),
                        test_window_days=int(l4.get("test_window_days", 20)),
                        embargo_days=int(l4.get("embargo_days", 20)),
                        holdout_years=int(l4.get("holdout_years", 2)),
                    )

                if stage_name == "L5":
                    if out_name == "pred_short_oos":
                        quality["model_oos"] = artifacts.get("_l5_report_short", {})
                    elif out_name == "pred_long_oos":
                        quality["model_oos"] = artifacts.get("_l5_report_long", {})

                if stage_name == "L6":
                    q = artifacts.get("_l6_quality", {})
                    if isinstance(q, dict) and q:
                        quality.update(q)
                if stage_name == "L6R":  # [개선안 13번] ranking 기반 scoring 품질 메타
                    q = artifacts.get("_l6_quality", {})
                    if isinstance(q, dict) and q:
                        quality.update(q)

                if stage_name == "L7":
                    q = artifacts.get("_l7_quality", {})
                    if isinstance(q, dict) and q:
                        quality.update(q)

                    # cost_bps 불일치 검증
                    if args.strict_params and out_name == "bt_metrics":
                        l7_cfg = _resolve_section(cfg, "l7")
                        config_cost_bps = l7_cfg.get("cost_bps")
                        if config_cost_bps is not None:
                            if "cost_bps" in df.columns:
                                actual_cost_bps = df["cost_bps"].iloc[0] if len(df) > 0 else None
                                if actual_cost_bps is not None:
                                    if abs(actual_cost_bps - config_cost_bps) > 0.01:
                                        error_msg = (
                                            f"cost_bps mismatch: config={config_cost_bps}, "
                                            f"actual={actual_cost_bps}"
                                        )
                                        logger.error(f"[STRICT-PARAMS] {error_msg}")
                                        raise ValueError(error_msg)
                                    else:
                                        logger.info(f"[STRICT-PARAMS] cost_bps verified: {config_cost_bps}")
                            else:
                                logger.warning("[STRICT-PARAMS] cost_bps column not found in bt_metrics")

                if write_meta:
                    meta = build_meta(
                        stage=f"{stage_name}:{out_name}",
                        run_id=args.run_id,
                        df=df,
                        out_base_path=out_base,
                        warnings=all_warnings,
                        inputs={"prev_outputs": list(artifacts.keys())},
                        repo_dir=get_path(cfg, "base_dir"),
                        quality=quality,
                    )
                    save_meta(out_base, meta, force=True)

                artifacts[out_name] = df

            # [Runtime 공통 규칙] Stage 종료 시간 기록
            stage_end_time = time.time()
            stage_runtime_sec = stage_end_time - stage_start_time
            stage_runtimes[stage_name]["end_time"] = stage_end_time
            stage_runtimes[stage_name]["runtime_sec"] = stage_runtime_sec
            logger.info(f"[Runtime] {stage_name} 완료: {stage_runtime_sec:.1f}초")

            # [Stage13] L7 종료 시간 기록 (기존 코드 유지)
            if stage_name == "L7":
                l7_end_time = time.time()
                l7_total_time = l7_end_time - l7_start_time if l7_start_time else None
                if l7_total_time is not None:
                    logger.info(f"[Stage13] L7 백테스트 완료: {l7_total_time:.1f}초")
                artifacts["_l7_total_time"] = l7_total_time  # [Stage13] L7 실행 시간 저장

            # [Stage7] L8의 경우 ranking_snapshot CSV를 reports/ranking/에 별도 저장
            if stage_name == "L8" and out_name == "ranking_snapshot":
                reports_ranking_dir = base_dir / "reports" / "ranking"
                reports_ranking_dir.mkdir(parents=True, exist_ok=True)
                snapshot_csv_path = reports_ranking_dir / f"ranking_snapshot__{run_tag}.csv"
                df.to_csv(snapshot_csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"[L8] ranking_snapshot CSV saved: {snapshot_csv_path}")

        except Exception as e:
            logger.error(f"Failed at {stage_name}: {e}")
            sys.exit(1)

    # [공통 프롬프트 v2] L2 해시 최종 검증 (모든 Stage 완료 후)
    if l2_hash_before is not None:
        l2_file = base_interim_dir / "fundamentals_annual.parquet"
        if l2_file.exists():
            l2_hash_after = _get_file_hash(l2_file)
            file_size_after = l2_file.stat().st_size
            mtime_after = os.path.getmtime(l2_file)
            from datetime import datetime
            mtime_str_after = datetime.fromtimestamp(mtime_after).strftime("%Y-%m-%d %H:%M:%S")

            if l2_hash_before != l2_hash_after:
                logger.error(f"[공통 프롬프트 v2] L2 재사용 규칙 위반 감지!")
                logger.error(f"[공통 프롬프트 v2] 실행 전 해시: {l2_hash_before[:16]}... (전체: {l2_hash_before})")
                logger.error(f"[공통 프롬프트 v2] 실행 후 해시: {l2_hash_after[:16]}... (전체: {l2_hash_after})")
                logger.error(f"[공통 프롬프트 v2] 파일 크기: {file_size_after:,} bytes")
                logger.error(f"[공통 프롬프트 v2] 파일 수정 시간: {mtime_str_after}")
                raise RuntimeError(
                    f"[공통 프롬프트 v2] L2 재사용 규칙 위반: Pipeline 완료 후 fundamentals_annual.parquet 파일이 변경되었습니다!\n"
                    f"실행 전 해시: {l2_hash_before[:16]}... (전체: {l2_hash_before})\n"
                    f"실행 후 해시: {l2_hash_after[:16]}... (전체: {l2_hash_after})\n"
                    f"파일 크기: {file_size_after:,} bytes\n"
                    f"파일 수정 시간: {mtime_str_after}\n"
                    "L2 재무데이터는 절대 재생성되어서는 안 됩니다. DART API 호출 금지."
                )
            logger.info(f"[공통 프롬프트 v2] L2 재사용 검증: 최종 해시 일치 확인 완료 (변경 없음)")
            logger.info(f"[공통 프롬프트 v2] L2 파일 크기: {file_size_after:,} bytes (변경 없음)")
            logger.info(f"[공통 프롬프트 v2] L2 파일 수정 시간: {mtime_str_after} (변경 없음)")
        else:
            raise RuntimeError(
                f"[공통 프롬프트 v2] L2 재사용 규칙 위반: Pipeline 완료 후 fundamentals_annual.parquet 파일이 삭제되었습니다!\n"
                f"파일 경로: {l2_file}\n"
                "L2 재무데이터는 절대 삭제되어서는 안 됩니다."
            )
    else:
        logger.warning("[공통 프롬프트 v2] L2 해시 검증: 실행 전 해시가 없어 최종 검증을 건너뜁니다.")

    t_pipeline_end = time.time()
    pipeline_total_time = t_pipeline_end - t_pipeline_start if 't_pipeline_start' in locals() else None

    logger.info("Pipeline Completed Successfully.")
    if pipeline_total_time is not None:
        logger.info(f"[Pipeline Runtime] 총 실행 시간: {pipeline_total_time:.1f}초")

    # Run tag를 artifacts에 저장 (나중에 manifest 생성 시 사용)
    artifacts["_run_tag"] = run_tag
    artifacts["_interim_dir"] = interim_dir
    artifacts["_pipeline_total_time"] = pipeline_total_time  # [Stage13] 전체 실행 시간 저장

    # [공통 프롬프트 v2] 공통 실행 산출물 생성 (KPI 리포트, Delta 리포트)
    # [TASK A-1] --no-export 옵션으로 리포트 생성 비활성화 가능
    # force-rebuild와 무관하게 항상 생성 (기존 리포트 덮어쓰기)
    # 모든 Stage에서 반드시 생성되어야 하는 산출물:
    # - data/interim/{run_tag}/... (태그 폴더에 산출물 저장) - 이미 완료
    # - reports/kpi/kpi_table__{run_tag}.csv, .md
    # - reports/delta/delta_kpi__{baseline_tag}__vs__{run_tag}.csv
    # - reports/delta/delta_report__{baseline_tag}__vs__{run_tag}.md

    try:
        if args.no_export:
            logger.info("[TASK A-1] --no-export 활성화: 리포트 생성 건너뛰기")
        else:
            logger.info("[공통 프롬프트 v2] 공통 실행 산출물 생성 시작...")
            logger.info("[공통 프롬프트 v2] 생성 대상:")
            logger.info(f"[공통 프롬프트 v2]   - reports/kpi/kpi_table__{run_tag}.csv, .md")
            logger.info(f"[공통 프롬프트 v2]   - reports/delta/delta_kpi__{args.baseline_tag}__vs__{run_tag}.csv")
            logger.info(f"[공통 프롬프트 v2]   - reports/delta/delta_report__{args.baseline_tag}__vs__{run_tag}.md")

        import subprocess

        # [TASK A-1] 리포트 생성은 --no-export가 없을 때만
        if not args.no_export:

            # KPI 리포트 생성 (기존 파일 덮어쓰기)
            kpi_script = base_dir / "src" / "tools" / "analysis" / "export_kpi_table.py"
            if kpi_script.exists():
                t_kpi_start = time.time()  # [Stage13] KPI export 시작 시간
                logger.info(f"[공통 프롬프트 v2] [1/2] KPI 리포트 생성 시작: {kpi_script}")
                kpi_cmd = [
                    sys.executable,
                    str(kpi_script),
                    "--config", args.config,
                    "--tag", run_tag,
                ]
                # [TASK A-1] --no-md 옵션 전달
                if args.no_md:
                    kpi_cmd.append("--no-md")
                kpi_result = subprocess.run(kpi_cmd, cwd=str(base_dir), capture_output=True, text=True)
                t_kpi_end = time.time()  # [Stage13] KPI export 종료 시간
                kpi_time = t_kpi_end - t_kpi_start
                runtime_breakdown["kpi_export_sec"] = kpi_time
                logger.info(f"[Stage13] KPI export 완료: {kpi_time:.1f}초")
                if kpi_result.returncode == 0:
                    logger.info("[공통 프롬프트 v2] [1/2] KPI 리포트 생성 완료")
                    # 생성된 파일 확인
                    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
                    kpi_md = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.md"
                    if kpi_csv.exists():
                        logger.info(f"[공통 프롬프트 v2] [1/2] KPI CSV 생성 확인: {kpi_csv}")
                    else:
                        logger.warning(f"[공통 프롬프트 v2] [1/2] KPI CSV 파일이 생성되지 않았습니다: {kpi_csv}")
                    if kpi_md.exists():
                        logger.info(f"[공통 프롬프트 v2] [1/2] KPI MD 생성 확인: {kpi_md}")
                    else:
                        logger.warning(f"[공통 프롬프트 v2] [1/2] KPI MD 파일이 생성되지 않았습니다: {kpi_md}")
                else:
                    logger.error(f"[공통 프롬프트 v2] [1/2] KPI 리포트 생성 실패 (returncode={kpi_result.returncode})")
                    logger.error(f"[공통 프롬프트 v2] [1/2] KPI 리포트 생성 stderr: {kpi_result.stderr}")
                    logger.error(f"[공통 프롬프트 v2] [1/2] KPI 리포트 생성 stdout: {kpi_result.stdout}")
            else:
                logger.error(f"[공통 프롬프트 v2] [1/2] KPI 리포트 스크립트를 찾을 수 없습니다: {kpi_script}")

            # Delta 리포트 생성 (baseline과 비교, 기존 파일 덮어쓰기)
        # [Stage13] L7의 경우 pipeline baseline(stage6)을 사용, 다른 Stage는 args.baseline_tag 사용
        # [TASK A-1] no_scan이면 스캔 건너뛰고 args.baseline_tag만 사용
        delta_baseline_tag = args.baseline_tag
        if "L7" in target_stages and not args.no_scan:
            t_baseline_detect_start = time.time() if args.profile else None
            # L7의 경우 stage6를 pipeline baseline으로 사용
            # stage6_로 시작하는 run_tag 찾기
            stage6_candidates = [d.name for d in base_interim_dir.iterdir()
                                if d.is_dir() and d.name.startswith("stage6_")]
            if stage6_candidates:
                # 최신 stage6 선택 (이름 기준 정렬 또는 mtime 기준)
                stage6_candidates.sort(reverse=True)  # 이름 기준 내림차순 (최신이 앞)
                pipeline_baseline = stage6_candidates[0]
                delta_baseline_tag = pipeline_baseline
                logger.info(f"[Stage13] Pipeline baseline 자동 탐지: {pipeline_baseline} (stage6 기준)")
            else:
                # stage6가 없으면 최신 rebalance_scores가 있는 폴더 사용 (fallback)
                rebalance_candidates = list(base_interim_dir.glob("*/rebalance_scores.parquet"))
                if rebalance_candidates:
                    rebalance_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    pipeline_baseline = rebalance_candidates[0].parent.name
                    delta_baseline_tag = pipeline_baseline
                    logger.warning(f"[Stage13] stage6를 찾을 수 없어 fallback 사용: {pipeline_baseline} (rebalance_scores 기준)")
                else:
                    logger.warning(f"[Stage13] Pipeline baseline을 찾을 수 없어 args.baseline_tag 사용: {args.baseline_tag}")
            if t_baseline_detect_start is not None:
                baseline_detect_time = time.time() - t_baseline_detect_start
                runtime_breakdown["baseline_detect_sec"] = baseline_detect_time
                logger.info(f"[TASK A-1] Baseline 탐지 시간: {baseline_detect_time:.3f}초")
        elif "L7" in target_stages and args.no_scan:
            logger.info(f"[TASK A-1] --no-scan 활성화: Pipeline baseline 탐지 건너뛰고 args.baseline_tag 사용: {args.baseline_tag}")
            runtime_breakdown["baseline_detect_sec"] = 0.0

        # [TASK A-1] Delta 리포트 생성은 --no-export가 없을 때만
        if not args.no_export:
            delta_script = base_dir / "src" / "tools" / "analysis" / "export_delta_report.py"
            if delta_script.exists():
                t_delta_start = time.time()  # [Stage13] Delta export 시작 시간
                logger.info(f"[공통 프롬프트 v2] [2/2] Delta 리포트 생성 시작: {delta_script}")
                logger.info(f"[공통 프롬프트 v2] [2/2] Baseline 태그: {delta_baseline_tag}, Run 태그: {run_tag}")
                delta_cmd = [
                    sys.executable,
                    str(delta_script),
                    "--baseline-tag", delta_baseline_tag,
                    "--run-tag", run_tag,
                ]
                # [TASK A-1] --no-md 옵션 전달
                if args.no_md:
                    delta_cmd.append("--no-md")
                delta_result = subprocess.run(delta_cmd, cwd=str(base_dir), capture_output=True, text=True)
                t_delta_end = time.time()  # [Stage13] Delta export 종료 시간
                delta_time = t_delta_end - t_delta_start
                runtime_breakdown["delta_export_sec"] = delta_time
                logger.info(f"[Stage13] Delta export 완료: {delta_time:.1f}초")
                if delta_result.returncode == 0:
                    logger.info("[공통 프롬프트 v2] [2/2] Delta 리포트 생성 완료")
                    # 생성된 파일 확인
                    delta_csv = base_dir / "reports" / "delta" / f"delta_kpi__{delta_baseline_tag}__vs__{run_tag}.csv"
                    delta_md = base_dir / "reports" / "delta" / f"delta_report__{delta_baseline_tag}__vs__{run_tag}.md"
                    if delta_csv.exists():
                        logger.info(f"[공통 프롬프트 v2] [2/2] Delta CSV 생성 확인: {delta_csv}")
                    else:
                        logger.warning(f"[공통 프롬프트 v2] [2/2] Delta CSV 파일이 생성되지 않았습니다: {delta_csv}")
                    if delta_md.exists():
                        logger.info(f"[공통 프롬프트 v2] [2/2] Delta MD 생성 확인: {delta_md}")
                    else:
                        logger.warning(f"[공통 프롬프트 v2] [2/2] Delta MD 파일이 생성되지 않았습니다: {delta_md}")
                else:
                    logger.error(f"[공통 프롬프트 v2] [2/2] Delta 리포트 생성 실패 (returncode={delta_result.returncode})")
                    logger.error(f"[공통 프롬프트 v2] [2/2] Delta 리포트 생성 stderr: {delta_result.stderr}")
                    logger.error(f"[공통 프롬프트 v2] [2/2] Delta 리포트 생성 stdout: {delta_result.stdout}")
            else:
                logger.error(f"[공통 프롬프트 v2] [2/2] Delta 리포트 스크립트를 찾을 수 없습니다: {delta_script}")

            logger.info("[공통 프롬프트 v2] 공통 실행 산출물 생성 완료")

        # [Stage13] 실행 시간 요약 저장
        timing_summary = {
            "pipeline_total_time": pipeline_total_time,
            "l7_total_time": artifacts.get("_l7_total_time"),
            "kpi_export_time": runtime_breakdown.get("kpi_export_sec"),
            "delta_export_time": runtime_breakdown.get("delta_export_sec"),
        }
        artifacts["_timing_summary"] = timing_summary

        # [TASK A-1] runtime_breakdown에 추가 정보 저장
        runtime_breakdown["pipeline_total_sec"] = pipeline_total_time
        runtime_breakdown["l7_total_sec"] = artifacts.get("_l7_total_time")
        if "baseline_detect_sec" not in runtime_breakdown:
            runtime_breakdown["baseline_detect_sec"] = 0.0

        # [TASK A-1] --profile 옵션으로 runtime_breakdown CSV 저장
        if args.profile:
            analysis_dir = base_dir / "reports" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            breakdown_csv = analysis_dir / f"runtime_breakdown__{run_tag}.csv"
            breakdown_df = pd.DataFrame([runtime_breakdown])
            breakdown_df.to_csv(breakdown_csv, index=False, encoding="utf-8-sig")
            logger.info(f"[TASK A-1] Runtime breakdown CSV 저장: {breakdown_csv}")

        # [Runtime 공통 규칙] Runtime 리포트 생성
        _generate_runtime_report(
            run_tag=run_tag,
            base_dir=base_dir,
            pipeline_total_time=pipeline_total_time,
            stage_runtimes=stage_runtimes,
            stage_input_summaries=stage_input_summaries,
            kpi_export_time=kpi_time if 'kpi_time' in locals() else None,
            delta_export_time=delta_time if 'delta_time' in locals() else None,
            target_stages=target_stages,
            l2_reuse_info=l2_reuse_info if 'l2_reuse_info' in locals() else {},
        )

        # [Stage13] 런타임 프로파일 저장 (L7 실행 시)
        if "L7" in target_stages and "_l7_runtime_profile" in artifacts:
            t_export_start = time.time()
            runtime_profile = artifacts["_l7_runtime_profile"]
            if isinstance(runtime_profile, pd.DataFrame) and len(runtime_profile) > 0:
                analysis_dir = base_dir / "reports" / "analysis"
                analysis_dir.mkdir(parents=True, exist_ok=True)

                # CSV 저장
                profile_csv = analysis_dir / f"l7_runtime_profile__{run_tag}.csv"
                runtime_profile.to_csv(profile_csv, index=False, encoding="utf-8-sig")
                logger.info(f"[Stage13] 런타임 프로파일 CSV 저장: {profile_csv}")

                # 분석 및 MD 생성
                profile_md = analysis_dir / f"l7_runtime_profile__{run_tag}.md"
                # l7_total_time은 run_L7_backtest 함수 내부에서만 존재하므로, runtime_profile에서 계산
                total_time_from_profile = runtime_profile["rebalance_time_sec"].sum() if len(runtime_profile) > 0 else None
                md_content = _generate_runtime_profile_md(runtime_profile, run_tag, total_time_from_profile)
                profile_md.write_text(md_content, encoding="utf-8")
                logger.info(f"[Stage13] 런타임 프로파일 MD 저장: {profile_md}")

                t_export_end = time.time()
                logger.info(f"[L7 Runtime] 리포트 생성 완료: {t_export_end - t_export_start:.1f}초")
            else:
                logger.warning("[Stage13] 런타임 프로파일이 비어있거나 DataFrame이 아닙니다.")

        # [Stage14] Stage14 체크리포트 생성
        if run_tag.startswith("stage14_"):
            print("[Stage14] 체크리포트 생성 시작...", flush=True)
            try:
                # 실행 커맨드 재구성
                cmd_parts = [
                    sys.executable,
                    "src/run_all.py",
                    "--from", args.from_stage,
                    "--to", args.to_stage,
                    "--run-tag", run_tag,
                    "--baseline-tag", args.baseline_tag,
                    "--config", args.config,
                ]
                if args.skip_l2:
                    cmd_parts.append("--skip-l2")
                if args.force_rebuild:
                    cmd_parts.append("--force-rebuild")
                if args.max_rebalances:
                    cmd_parts.extend(["--max-rebalances", str(args.max_rebalances)])
                command = " ".join(cmd_parts)

                check_report_path = _generate_stage14_check_report(
                    run_tag=run_tag,
                    baseline_tag=args.baseline_tag,
                    base_dir=base_dir,
                    base_interim_dir=base_interim_dir,
                    command=command,
                    args=args,
                    pipeline_total_time=pipeline_total_time,
                )
                logger.info(f"[Stage14] 체크리포트 생성 완료: {check_report_path}")
                print(f"[Stage14] 체크리포트 생성 완료: {check_report_path}", flush=True)
            except Exception as e:
                logger.error(f"[Stage14] 체크리포트 생성 중 오류 발생: {e}")
                print(f"[Stage14] 체크리포트 생성 실패: {e}", flush=True)
                import traceback
                logger.error(f"[Stage14] 트레이스백: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"[공통 프롬프트 v2] 리포트 생성 중 오류 발생 (파이프라인은 성공): {e}")
        import traceback
        logger.error(f"[공통 프롬프트 v2] 트레이스백: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
