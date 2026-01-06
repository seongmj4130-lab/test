# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/backtest/l7_backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import logging
import time
import numpy as np
import pandas as pd

# [Stage13] selector import
from src.components.portfolio.selector import select_topk_with_fallback

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 20
    top_k: int = 20
    cost_bps: float = 10.0
    score_col: str = "score_ens"
    ret_col: str = "true_short"
    weighting: str = "equal"  # equal | softmax
    softmax_temp: float = 1.0

    # Turnover 완화 옵션(기본 OFF)
    # 이전 보유 종목이 상위(top_k + buffer_k) 안에 있으면 유지 후보로 인정
    buffer_k: int = 0
    
    # [Phase 7 Step 2] Turnover 제어: 리밸런싱 주기 완화
    # rebalance_interval=1이면 모든 리밸런싱 실행, 2면 매 2번째만 실행 (빈도 50% 감소)
    # 목표: Avg Turnover ≤ 500% 달성
    rebalance_interval: int = 1  # 기본값 1 (모든 리밸런싱 실행)
    
    # [Stage 4] 업종 분산 제약 옵션
    diversify_enabled: bool = False
    group_col: str = "sector_name"
    max_names_per_group: int = 4
    
    # [Stage5] 시장 국면(regime) 기반 전략 옵션
    regime_enabled: bool = False
    # [국면 세분화] 5단계 국면별 설정
    regime_top_k_bull_strong: Optional[int] = None
    regime_top_k_bull_weak: Optional[int] = None
    regime_top_k_bear_strong: Optional[int] = None
    regime_top_k_bear_weak: Optional[int] = None
    regime_top_k_neutral: Optional[int] = None
    regime_exposure_bull_strong: Optional[float] = None
    regime_exposure_bull_weak: Optional[float] = None
    regime_exposure_bear_strong: Optional[float] = None
    regime_exposure_bear_weak: Optional[float] = None
    regime_exposure_neutral: Optional[float] = None
    # 하위 호환성 (2단계 설정)
    regime_top_k_bull: Optional[int] = None  # bull 시장에서 사용할 top_k (5단계 미지정 시 사용)
    regime_top_k_bear: Optional[int] = None  # bear 시장에서 사용할 top_k (5단계 미지정 시 사용)
    regime_exposure_bull: Optional[float] = None  # bull 시장에서 사용할 exposure (5단계 미지정 시 사용)
    regime_exposure_bear: Optional[float] = None  # bear 시장에서 사용할 exposure (5단계 미지정 시 사용)
    
    # [Phase 8 Step 2 방안1] 리밸런싱 규칙/버퍼/노출 튜닝: Dev 붕괴 완화
    smart_buffer_enabled: bool = False  # 스마트 버퍼링 활성화
    smart_buffer_stability_threshold: float = 0.7  # 보유 종목 안정성 임계값
    volatility_adjustment_enabled: bool = False  # 변동성 기반 exposure 조정 활성화
    volatility_lookback_days: int = 60  # 변동성 계산 기간
    target_volatility: float = 0.15  # 목표 변동성 (15%)
    volatility_adjustment_max: float = 1.2  # 변동성 조정 최대 배수
    volatility_adjustment_min: float = 0.6  # 변동성 조정 최소 배수
    
    # [Phase 8 Step 2 방안2] 국면 필터/리스크 스케일링: Bear 구간 방어 강화
    risk_scaling_enabled: bool = False  # 국면별 리스크 스케일링 활성화
    risk_scaling_bear_multiplier: float = 0.7  # Bear 구간 추가 포지션 축소 배수
    risk_scaling_neutral_multiplier: float = 0.9  # Neutral 구간 포지션 축소 배수
    risk_scaling_bull_multiplier: float = 1.0  # Bull 구간 포지션 축소 배수

def _ensure_datetime(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s)
    return s

def _pick_score_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in ["score_ens", "score", "score_total", "score_ensemble"]:
        if c in df.columns:
            return c
    raise KeyError(
        f"score column not found. tried: {preferred}, score_ens, score, score_total, score_ensemble"
    )

def _pick_ret_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in ["true_short", "y_true", "ret_fwd_20d", "ret"]:
        if c in df.columns:
            return c
    raise KeyError(
        f"return/true column not found. tried: {preferred}, true_short, y_true, ret_fwd_20d, ret"
    )

def _compute_turnover_oneway(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    keys = set(prev_w) | set(new_w)
    s = 0.0
    for k in keys:
        s += abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0))
    return 0.5 * s

def _mdd(rr: np.ndarray) -> float:
    """[최종 수치셋] MDD 계산 함수"""
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rr:
        eq *= (1.0 + float(r))
        peak = max(peak, eq)
        mdd = min(mdd, (eq / peak) - 1.0)
    return float(mdd)

def _weights_from_scores(scores: pd.Series, method: str, temp: float) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.array([], dtype=float)

    if method == "equal":
        return np.full(n, 1.0 / n, dtype=float)

    if method == "softmax":
        x = scores.astype(float).to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        t = float(temp) if float(temp) > 0 else 1.0
        x = x / t
        x = x - np.max(x)  # 안정화
        w = np.exp(x)
        sw = w.sum()
        if sw <= 0:
            return np.full(n, 1.0 / n, dtype=float)
        return w / sw

    raise ValueError(f"unknown weighting: {method}. expected equal|softmax")

def _select_with_diversification(
    g_sorted: pd.DataFrame,
    *,
    ticker_col: str,
    top_k: int,
    buffer_k: int,
    prev_holdings: List[str],
    group_col: str | None = None,
    max_names_per_group: int | None = None,
) -> pd.DataFrame:
    """
    [Stage 4] 업종 분산 제약을 적용한 종목 선택
    
    Args:
        g_sorted: score desc 정렬된 DataFrame
        ticker_col: 티커 컬럼명
        top_k: 선택할 종목 수
        buffer_k: 버퍼 종목 수
        prev_holdings: 이전 보유 종목 리스트
        group_col: 그룹 컬럼명 (예: "sector_name")
        max_names_per_group: 그룹당 최대 종목 수
    
    Returns:
        선택된 종목 DataFrame
    """
    top_k = int(top_k)
    buffer_k = int(buffer_k)
    
    # [Stage 4] 업종 분산 제약이 없으면 기존 로직 사용
    if group_col is None or max_names_per_group is None or group_col not in g_sorted.columns:
        return _select_with_buffer(g_sorted, ticker_col=ticker_col, top_k=top_k, buffer_k=buffer_k, prev_holdings=prev_holdings)
    
    # [Stage 4] 업종 분산 제약 적용
    allow_n = top_k + buffer_k if buffer_k > 0 else top_k
    allow = g_sorted.head(allow_n).copy()
    
    # 이전 보유 종목 중 허용 범위에 있는 것들
    allow_set = set(allow[ticker_col].astype(str).tolist())
    keep = [t for t in prev_holdings if t in allow_set]
    
    # cap keep to top_k
    if len(keep) > top_k:
        keep = keep[:top_k]
    
    selected = []
    selected_set = set()
    group_counts: Dict[str, int] = {}
    
    # 1) keep 먼저 (업종 제약 고려)
    for t in keep:
        ticker_row = allow[allow[ticker_col].astype(str) == t]
        if len(ticker_row) > 0:
            sector = str(ticker_row.iloc[0][group_col]) if pd.notna(ticker_row.iloc[0][group_col]) else "기타"
            current_count = group_counts.get(sector, 0)
            if current_count < max_names_per_group:
                selected.append(t)
                selected_set.add(t)
                group_counts[sector] = current_count + 1
    
    # 2) 부족한 만큼 상위에서 채움 (업종 제약 고려)
    for _, row in allow.iterrows():
        if len(selected) >= top_k:
            break
        
        t = str(row[ticker_col])
        if t in selected_set:
            continue
        
        sector = str(row[group_col]) if pd.notna(row[group_col]) else "기타"
        current_count = group_counts.get(sector, 0)
        
        if current_count < max_names_per_group:
            selected.append(t)
            selected_set.add(t)
            group_counts[sector] = current_count + 1
    
    # final safety: never exceed top_k
    if len(selected) > top_k:
        selected = selected[:top_k]
    
    return g_sorted[g_sorted[ticker_col].astype(str).isin(selected)].copy()

def _select_with_smart_buffer(
    g_sorted: pd.DataFrame,
    *,
    ticker_col: str,
    top_k: int,
    buffer_k: int,
    prev_holdings: List[str],
    stability_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    [Phase 8 Step 2 방안1] 스마트 버퍼링: 기존 보유 종목에 더 높은 우선순위 부여
    
    기존 _select_with_buffer와 차이:
    - 보유 종목이 상위 X% 내에 있으면 더 강력하게 유지
    - 안정적인 포지션 유지로 Dev 구간 붕괴 완화
    
    Args:
        g_sorted: score desc 정렬된 DataFrame
        ticker_col: 티커 컬럼명
        top_k: 선택할 종목 수
        buffer_k: 버퍼 종목 수
        prev_holdings: 이전 보유 종목 리스트
        stability_threshold: 보유 종목 안정성 임계값 (0.0~1.0)
    
    Returns:
        선택된 종목 DataFrame
    """
    top_k = int(top_k)
    buffer_k = int(buffer_k)

    if buffer_k <= 0 or len(prev_holdings) == 0:
        return g_sorted.head(top_k).copy()

    allow_n = top_k + buffer_k
    allow = g_sorted.head(allow_n)

    allow_set = set(allow[ticker_col].astype(str).tolist())
    total_count = len(g_sorted)
    
    # [Phase 8 Step 2 방안1] 스마트 버퍼링: 안정성 임계값 기반 필터링
    keep = []
    for t in prev_holdings:
        if t in allow_set:
            # 해당 종목의 현재 순위 확인
            ticker_idx = g_sorted[g_sorted[ticker_col].astype(str) == t].index
            if len(ticker_idx) > 0:
                rank = ticker_idx[0]  # 첫 번째 매치
                rank_pct = rank / max(total_count - 1, 1)  # 0~1 정규화
                # 순위가 상위 X% 내에 있으면 유지 (낮을수록 상위)
                if rank_pct <= stability_threshold:
                    keep.append(t)
    
    # cap keep to top_k
    if len(keep) > top_k:
        keep = keep[:top_k]

    selected = []
    selected_set = set()

    # 1) keep 먼저 (스마트 필터링된 종목들)
    for t in keep:
        selected.append(t)
        selected_set.add(t)

    # 2) 부족한 만큼 상위에서 채움
    for t in g_sorted[ticker_col].astype(str).tolist():
        if len(selected) >= top_k:
            break
        if t in selected_set:
            continue
        selected.append(t)
        selected_set.add(t)

    # final safety: never exceed top_k
    if len(selected) > top_k:
        selected = selected[:top_k]
    return g_sorted[g_sorted[ticker_col].astype(str).isin(selected)].copy()

def _select_with_buffer(
    g_sorted: pd.DataFrame,
    *,
    ticker_col: str,
    top_k: int,
    buffer_k: int,
    prev_holdings: List[str],
) -> pd.DataFrame:
    """
    g_sorted: score desc 정렬된 DataFrame
    buffer_k=0이면 그냥 top_k 선택
    buffer_k>0이면:
      - 허용범위: top_k + buffer_k
      - 이전보유 중 허용범위에 있으면 우선 keep
      - 남은 자리는 상위 점수부터 채움
    """
    top_k = int(top_k)
    buffer_k = int(buffer_k)

    if buffer_k <= 0 or len(prev_holdings) == 0:
        return g_sorted.head(top_k).copy()

    allow_n = top_k + buffer_k
    allow = g_sorted.head(allow_n)

    allow_set = set(allow[ticker_col].astype(str).tolist())
    keep = [t for t in prev_holdings if t in allow_set]

    # cap keep to top_k to avoid selecting > top_k when buffer window is wide
    if len(keep) > top_k:
        keep = keep[:top_k]

    selected = []
    selected_set = set()

    # 1) keep 먼저
    for t in keep:
        selected.append(t)
        selected_set.add(t)

    # 2) 부족한 만큼 상위에서 채움
    for t in g_sorted[ticker_col].astype(str).tolist():
        if len(selected) >= top_k:
            break
        if t in selected_set:
            continue
        selected.append(t)
        selected_set.add(t)

    # final safety: never exceed top_k
    if len(selected) > top_k:
        selected = selected[:top_k]
    return g_sorted[g_sorted[ticker_col].astype(str).isin(selected)].copy()

def _calculate_volatility_adjustment(
    recent_returns: np.ndarray,
    target_vol: float,
    lookback_days: int,
    max_mult: float = 1.2,
    min_mult: float = 0.6,
) -> float:
    """
    [Phase 8 Step 2 방안1] 변동성 기반 exposure 조정
    
    Args:
        recent_returns: 최근 수익률 배열
        target_vol: 목표 변동성 (예: 0.15 = 15%)
        lookback_days: 변동성 계산 기간
        max_mult: 최대 조정 배수
        min_mult: 최소 조정 배수
    
    Returns:
        exposure 조정 배수 (0.6 ~ 1.2)
    """
    if len(recent_returns) < 2:
        return 1.0
    
    # 최근 N일 수익률의 표준편차 계산 (연율화)
    recent_window = recent_returns[-lookback_days:] if len(recent_returns) >= lookback_days else recent_returns
    if len(recent_window) < 2:
        return 1.0
    
    current_vol = float(np.std(recent_window)) * np.sqrt(252.0)  # 연율화
    
    if current_vol <= 0 or target_vol <= 0:
        return 1.0
    
    # 목표 변동성 대비 현재 변동성 비율
    vol_ratio = target_vol / current_vol
    
    # 조정 배수 계산 (클리핑)
    adjustment = float(np.clip(vol_ratio, min_mult, max_mult))
    
    return adjustment

def _apply_risk_scaling(
    base_exposure: float,
    regime: Optional[str],
    risk_scaling_enabled: bool,
    bear_multiplier: float = 0.7,
    neutral_multiplier: float = 0.9,
    bull_multiplier: float = 1.0,
) -> float:
    """
    [Phase 8 Step 2 방안2] 국면별 리스크 스케일링: Bear 구간 방어 강화
    
    Args:
        base_exposure: 기본 exposure (국면별 설정값)
        regime: 현재 시장 국면 ("bear_strong", "bear_weak", "neutral", "bull_weak", "bull_strong" 등)
        risk_scaling_enabled: 리스크 스케일링 활성화 여부
        bear_multiplier: Bear 구간 추가 포지션 축소 배수
        neutral_multiplier: Neutral 구간 포지션 축소 배수
        bull_multiplier: Bull 구간 포지션 축소 배수
    
    Returns:
        조정된 exposure
    """
    if not risk_scaling_enabled:
        return base_exposure
    
    if regime is None:
        return base_exposure
    
    regime_lower = str(regime).lower()
    
    # Bear 구간: 추가 포지션 축소
    if "bear" in regime_lower:
        return base_exposure * bear_multiplier
    # Neutral 구간: 약간 축소
    elif "neutral" in regime_lower:
        return base_exposure * neutral_multiplier
    # Bull 구간: 변경 없음
    elif "bull" in regime_lower:
        return base_exposure * bull_multiplier
    else:
        return base_exposure

def run_backtest(
    rebalance_scores: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    phase_col: str = "phase",
    config_cost_bps: Optional[float] = None,  # [Stage1] config.yaml에서 읽은 원본 cost_bps 값
    market_regime: Optional[pd.DataFrame] = None,  # [Stage5] 시장 국면 DataFrame (date, regime 컬럼)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, List[str]]:
    warns: List[str] = []

    # [Stage1] config_cost_bps가 없으면 cfg.cost_bps를 사용 (하위 호환성)
    cost_bps_config = float(config_cost_bps) if config_cost_bps is not None else float(cfg.cost_bps)
    cost_bps_used = float(cfg.cost_bps)  # 실제 백테스트에 사용된 cost_bps
    cost_bps_mismatch_flag = bool(abs(cost_bps_used - cost_bps_config) > 1e-6)  # 부동소수점 오차 고려
    
    if cost_bps_mismatch_flag:
        warns.append(f"[Stage1] cost_bps mismatch: config={cost_bps_config}, used={cost_bps_used}")
    
    # [Stage5] 시장 국면(regime) 처리
    regime_dict: Dict[pd.Timestamp, str] = {}
    if cfg.regime_enabled:
        if market_regime is None or len(market_regime) == 0:
            warns.append("[Stage5] regime_enabled=True이지만 market_regime이 제공되지 않았습니다. regime 기능을 비활성화합니다.")
            regime_enabled_actual = False
        else:
            # market_regime에서 date, regime 컬럼 추출
            if "date" not in market_regime.columns or "regime" not in market_regime.columns:
                warns.append("[Stage5] market_regime에 'date' 또는 'regime' 컬럼이 없습니다. regime 기능을 비활성화합니다.")
                regime_enabled_actual = False
            else:
                regime_df = market_regime[["date", "regime"]].copy()
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                regime_dict = dict(zip(regime_df["date"], regime_df["regime"]))
                regime_enabled_actual = True
                logger.info(f"[Stage5] 시장 국면 데이터 로드 완료: {len(regime_dict)}개 날짜")
    else:
        regime_enabled_actual = False

    for c in [date_col, ticker_col, phase_col]:
        if c not in rebalance_scores.columns:
            raise KeyError(f"rebalance_scores missing required column: {c}")

    score_col = _pick_score_col(rebalance_scores, cfg.score_col)
    ret_col = _pick_ret_col(rebalance_scores, cfg.ret_col)

    # [Stage 4] sector_name이 있으면 포함
    df_cols = [date_col, ticker_col, phase_col, score_col, ret_col]
    if cfg.diversify_enabled and cfg.group_col in rebalance_scores.columns:
        df_cols.append(cfg.group_col)
    
    df = rebalance_scores[df_cols].copy()
    df[date_col] = _ensure_datetime(df[date_col])
    df[ticker_col] = df[ticker_col].astype(str)
    df[phase_col] = df[phase_col].astype(str)

    before = len(df)
    df = df.dropna(subset=[ret_col])
    dropped = before - len(df)
    if dropped > 0:
        warns.append(f"[L7] dropped {dropped} rows with NA {ret_col}")

    df_sorted = df.sort_values(
        [phase_col, date_col, score_col, ticker_col],
        ascending=[True, True, False, True],
    )

    positions_rows: List[dict] = []
    returns_rows: List[dict] = []
    # [Stage13] selection diagnostics 저장용
    selection_diagnostics_rows: List[dict] = []
    
    # [Stage13] 런타임 프로파일 저장용
    runtime_profile_rows: List[dict] = []
    t_load_inputs = time.time()
    logger.info(f"[L7 Runtime] 입력 로딩 완료: {time.time() - t_load_inputs:.3f}초")

    for phase, dphase in df_sorted.groupby(phase_col, sort=False):
        t_phase_start = time.time()
        rebalance_dates_all = sorted(dphase[date_col].unique())
        
        # [Phase 7 Step 2] Turnover 제어: 리밸런싱 주기 완화
        rebalance_interval = int(cfg.rebalance_interval) if hasattr(cfg, 'rebalance_interval') else 1
        if rebalance_interval > 1:
            # 매 N번째 리밸런싱만 선택 (0-indexed이므로 interval-1, interval*2-1, ...)
            rebalance_dates_filtered = [rebalance_dates_all[i] for i in range(0, len(rebalance_dates_all), rebalance_interval)]
            logger.info(f"[Phase 7 Step 2] 리밸런싱 주기 완화: interval={rebalance_interval}, 전체 {len(rebalance_dates_all)}개 → 필터링 {len(rebalance_dates_filtered)}개")
            # 필터링된 날짜만 포함하도록 dphase 필터링
            dphase = dphase[dphase[date_col].isin(rebalance_dates_filtered)].copy()
        else:
            rebalance_dates_filtered = rebalance_dates_all
        
        logger.info(f"[L7 Runtime] Phase '{phase}' 시작 (리밸런싱 {len(rebalance_dates_filtered)}개)")
        prev_w: Dict[str, float] = {}
        prev_holdings: List[str] = []
        
        rebalance_dates = sorted(dphase[date_col].unique())
        rebalance_count = 0
        
        # [Phase 8 Step 2 방안1] 변동성 기반 exposure 조정을 위한 수익률 히스토리 추적
        recent_returns_history: List[float] = []  # 최근 수익률 저장
        
        for dt, g in dphase.groupby(date_col, sort=True):
            t_rebalance_start = time.time()
            rebalance_count += 1
            g = g.sort_values([score_col, ticker_col], ascending=[False, True]).reset_index(drop=True)

            # [Stage5] 시장 국면에 따른 top_k 및 exposure 결정
            dt_ts = pd.to_datetime(dt)
            current_regime = None
            current_top_k = int(cfg.top_k)
            current_exposure = 1.0
            
            if regime_enabled_actual:
                # 가장 가까운 rebalance 날짜의 regime 조회 (forward fill)
                regime_dates = sorted([d for d in regime_dict.keys() if d <= dt_ts])
                if len(regime_dates) > 0:
                    nearest_date = regime_dates[-1]
                    current_regime = regime_dict[nearest_date]
                    
                    # [국면 세분화] 5단계 국면별 top_k/exposure 결정
                    if current_regime == "bull_strong":
                        if cfg.regime_top_k_bull_strong is not None:
                            current_top_k = int(cfg.regime_top_k_bull_strong)
                        elif cfg.regime_top_k_bull is not None:
                            current_top_k = int(cfg.regime_top_k_bull)
                        if cfg.regime_exposure_bull_strong is not None:
                            current_exposure = float(cfg.regime_exposure_bull_strong)
                        elif cfg.regime_exposure_bull is not None:
                            current_exposure = float(cfg.regime_exposure_bull)
                    elif current_regime == "bull_weak":
                        if cfg.regime_top_k_bull_weak is not None:
                            current_top_k = int(cfg.regime_top_k_bull_weak)
                        elif cfg.regime_top_k_bull is not None:
                            current_top_k = int(cfg.regime_top_k_bull)
                        if cfg.regime_exposure_bull_weak is not None:
                            current_exposure = float(cfg.regime_exposure_bull_weak)
                        elif cfg.regime_exposure_bull is not None:
                            current_exposure = float(cfg.regime_exposure_bull)
                    elif current_regime == "bear_strong":
                        if cfg.regime_top_k_bear_strong is not None:
                            current_top_k = int(cfg.regime_top_k_bear_strong)
                        elif cfg.regime_top_k_bear is not None:
                            current_top_k = int(cfg.regime_top_k_bear)
                        if cfg.regime_exposure_bear_strong is not None:
                            current_exposure = float(cfg.regime_exposure_bear_strong)
                        elif cfg.regime_exposure_bear is not None:
                            current_exposure = float(cfg.regime_exposure_bear)
                    elif current_regime == "bear_weak":
                        if cfg.regime_top_k_bear_weak is not None:
                            current_top_k = int(cfg.regime_top_k_bear_weak)
                        elif cfg.regime_top_k_bear is not None:
                            current_top_k = int(cfg.regime_top_k_bear)
                        if cfg.regime_exposure_bear_weak is not None:
                            current_exposure = float(cfg.regime_exposure_bear_weak)
                        elif cfg.regime_exposure_bear is not None:
                            current_exposure = float(cfg.regime_exposure_bear)
                    elif current_regime == "neutral":
                        if cfg.regime_top_k_neutral is not None:
                            current_top_k = int(cfg.regime_top_k_neutral)
                        if cfg.regime_exposure_neutral is not None:
                            current_exposure = float(cfg.regime_exposure_neutral)
                    # 하위 호환성 (2단계)
                    elif current_regime == "bull" and cfg.regime_top_k_bull is not None:
                        current_top_k = int(cfg.regime_top_k_bull)
                        if cfg.regime_exposure_bull is not None:
                            current_exposure = float(cfg.regime_exposure_bull)
                    elif current_regime == "bear" and cfg.regime_top_k_bear is not None:
                        current_top_k = int(cfg.regime_top_k_bear)
                        if cfg.regime_exposure_bear is not None:
                            current_exposure = float(cfg.regime_exposure_bear)
                else:
                    warns.append(f"[Stage5] {dt_ts.strftime('%Y-%m-%d')}: regime 데이터 없음, 기본값 사용")

            # [Stage13] selector 사용 (drop reason 추적 + fallback)
            # [Phase 8 Step 2 방안1] 스마트 버퍼링 옵션 추가
            g_sel, diagnostics = select_topk_with_fallback(
                g,
                ticker_col=ticker_col,
                score_col=score_col,
                top_k=current_top_k,
                buffer_k=int(cfg.buffer_k),
                prev_holdings=prev_holdings,
                group_col=cfg.group_col if cfg.diversify_enabled else None,
                max_names_per_group=cfg.max_names_per_group if cfg.diversify_enabled else None,
                required_cols=[ret_col],  # ret_col 필수
                filter_missing_price=True,
                filter_suspended=False,  # suspended 컬럼이 없을 수 있음
                smart_buffer_enabled=cfg.smart_buffer_enabled,  # [Phase 8 Step 2 방안1]
                smart_buffer_stability_threshold=cfg.smart_buffer_stability_threshold,  # [Phase 8 Step 2 방안1]
            )

            # [Stage13] diagnostics 저장
            diag_row = {
                "date": dt,
                "phase": phase,
                "top_k": current_top_k,
                "eligible_count": diagnostics["eligible_count"],
                "selected_count": diagnostics["selected_count"],  # K_eff
                "dropped_missing": diagnostics["dropped_missing"],
                "dropped_filter": diagnostics["dropped_filter"],
                "dropped_sectorcap": diagnostics["dropped_sectorcap"],
                "filled_from_next_rank": diagnostics["filled_from_next_rank"],
            }
            # drop_reasons를 JSON 문자열로 저장 (또는 개별 컬럼으로)
            if diagnostics["drop_reasons"]:
                diag_row["drop_reasons"] = str(diagnostics["drop_reasons"])
            else:
                diag_row["drop_reasons"] = None
            selection_diagnostics_rows.append(diag_row)
            
            # [Stage13] 런타임 프로파일 기록
            t_rebalance_end = time.time()
            rebalance_time = t_rebalance_end - t_rebalance_start
            runtime_profile_rows.append({
                "date": dt,
                "phase": phase,
                "rebalance_time_sec": rebalance_time,
                "k_eff": diagnostics["selected_count"],
                "top_k_used": current_top_k,
                "holding_days": int(cfg.holding_days),
                "eligible_count": diagnostics["eligible_count"],
            })
            
            # 매 10회마다 요약 로그
            if rebalance_count % 10 == 0:
                elapsed_phase = t_rebalance_end - t_phase_start
                avg_time = elapsed_phase / rebalance_count
                logger.info(f"[L7 Runtime] Phase '{phase}': {rebalance_count}/{len(rebalance_dates)} 완료, 평균 {avg_time:.3f}초/리밸런싱, 누적 {elapsed_phase:.1f}초")

            g_sel = g_sel.sort_values([score_col, ticker_col], ascending=[False, True]).reset_index(drop=True)

            scores = g_sel[score_col]
            w = _weights_from_scores(scores, cfg.weighting, cfg.softmax_temp)

            new_w = {t: float(wi) for t, wi in zip(g_sel[ticker_col].tolist(), w.tolist())}
            turnover_oneway = _compute_turnover_oneway(prev_w, new_w)

            gross_ret = float(np.dot(w, g_sel[ret_col].astype(float).to_numpy()))
            
            # [Phase 8 Step 2 방안1] 변동성 기반 exposure 조정
            volatility_adjustment = 1.0
            if cfg.volatility_adjustment_enabled and len(recent_returns_history) >= 2:
                recent_returns_array = np.array(recent_returns_history[-cfg.volatility_lookback_days:])
                volatility_adjustment = _calculate_volatility_adjustment(
                    recent_returns_array,
                    target_vol=cfg.target_volatility,
                    lookback_days=cfg.volatility_lookback_days,
                    max_mult=cfg.volatility_adjustment_max,
                    min_mult=cfg.volatility_adjustment_min,
                )
            
            # [Phase 8 Step 2 방안2] 국면별 리스크 스케일링 적용
            adjusted_exposure = _apply_risk_scaling(
                base_exposure=current_exposure,
                regime=current_regime,
                risk_scaling_enabled=cfg.risk_scaling_enabled,
                bear_multiplier=cfg.risk_scaling_bear_multiplier,
                neutral_multiplier=cfg.risk_scaling_neutral_multiplier,
                bull_multiplier=cfg.risk_scaling_bull_multiplier,
            )
            
            # [Stage5] exposure 적용 (변동성 조정 포함)
            final_exposure = adjusted_exposure * volatility_adjustment
            gross_ret = gross_ret * final_exposure
            
            # [Phase 8 Step 2] 수익률 히스토리 업데이트 (변동성 조정용)
            if cfg.volatility_adjustment_enabled:
                recent_returns_history.append(gross_ret)
                # 메모리 관리: lookback_days의 2배까지만 유지
                max_history = cfg.volatility_lookback_days * 2
                if len(recent_returns_history) > max_history:
                    recent_returns_history = recent_returns_history[-max_history:]
            
            # [개선안 1번] 거래비용 반영 (cost_bps=0.0 → 10.0)
            # 한국 증권거래 평균 비용: 10~15bps (0.1~0.15%)
            # cost_bps = 10.0 means 10 basis points = 0.1%
            
            # Position value 계산: 현재 보유 중인 포지션의 총 가치
            # position_value = sum(weights) = 1.0 (정규화된 포트폴리오)
            position_value = float(np.sum(w))  # 보통 1.0
            
            # [개선안 1번] Position value 기반 거래비용 계산
            # daily_trading_cost = position_value * cost_bps / 10000
            daily_trading_cost = position_value * float(cfg.cost_bps) / 10000.0
            
            # [개선안 1번] 포지션 변경 시에만 비용 발생하도록 조정
            # Turnover 기반 비용: 포지션 변경 비율에 비례하여 비용 발생
            # turnover_oneway는 포지션 변경 비율 (0.0~1.0)
            turnover_cost = float(turnover_oneway) * float(cfg.cost_bps) / 10000.0
            
            # 총 비용: 포지션 변경이 있을 때만 비용 발생
            # - turnover_oneway > 0: 포지션 변경 발생 → daily_trading_cost 적용
            # - turnover_oneway = 0: 포지션 변경 없음 → 비용 없음
            if turnover_oneway > 0:
                # 포지션 변경이 있을 때: position_value 기반 비용 적용
                total_cost = daily_trading_cost
            else:
                # 포지션 변경이 없을 때: 비용 없음 (보유 비용 없음)
                total_cost = 0.0
            
            # [개선안 1번] PnL에서 거래비용 차감
            net_ret = gross_ret - total_cost

            # [Stage 4] sector_name을 positions에 포함
            sector_col = cfg.group_col if cfg.diversify_enabled and cfg.group_col in g_sel.columns else None
            
            # [Stage13] k_eff, eligible_count, filled_count 계산
            k_eff = diagnostics["selected_count"]
            eligible_count = diagnostics["eligible_count"]
            filled_count = diagnostics["filled_from_next_rank"]
            
            for idx, (t, wi, sc, tr) in enumerate(zip(g_sel[ticker_col], w, g_sel[score_col], g_sel[ret_col])):
                pos_row = {
                    "date": dt,
                    "phase": phase,
                    "ticker": str(t),
                    "weight": float(wi),
                    "score": float(sc) if pd.notna(sc) else np.nan,
                    "ret_realized": float(tr),
                    "top_k": int(cfg.top_k),
                    "holding_days": int(cfg.holding_days),
                    "cost_bps": float(cfg.cost_bps),
                    "weighting": cfg.weighting,
                    "buffer_k": int(cfg.buffer_k),
                    # [Stage13] K_eff 및 관련 지표 추가
                    "k_eff": int(k_eff),
                    "eligible_count": int(eligible_count),
                    "filled_count": int(filled_count),
                }
                # [Stage 4] sector_name 추가
                if sector_col and sector_col in g_sel.columns:
                    row_sector = g_sel.iloc[idx][sector_col] if idx < len(g_sel) else None
                    pos_row[sector_col] = str(row_sector) if pd.notna(row_sector) else None
                positions_rows.append(pos_row)

            returns_row = {
                "date": dt,
                "phase": phase,
                "top_k": int(current_top_k),  # [Stage5] regime별 top_k
                "holding_days": int(cfg.holding_days),
                "cost_bps": float(cfg.cost_bps),
                "cost_bps_used": float(cost_bps_used),  # [Stage1] 실제 사용된 cost_bps
                "weighting": cfg.weighting,
                "buffer_k": int(cfg.buffer_k),
                "n_tickers": int(len(g_sel)),
                "gross_return": float(gross_ret),
                "net_return": float(net_ret),
                "turnover_oneway": float(turnover_oneway),
                "daily_trading_cost": float(daily_trading_cost),  # [개선안 1번] position_value 기반 비용
                "turnover_cost": float(turnover_cost),  # [개선안 1번] turnover 기반 비용
                "total_cost": float(total_cost),  # [개선안 1번] 실제 적용된 총 비용
            }
            
            # [Stage5] regime 및 exposure 기록
            # [Stage13] market_regime이 없으면 regime 컬럼을 생성하지 않음 (결측률 95%+ 방지)
            if regime_enabled_actual:
                returns_row["regime"] = str(current_regime) if current_regime else None
                returns_row["exposure"] = float(current_exposure)
            # [Phase 8 Step 2] 변동성 조정 및 리스크 스케일링 정보 기록
            if cfg.volatility_adjustment_enabled:
                returns_row["volatility_adjustment"] = float(volatility_adjustment)
            if cfg.risk_scaling_enabled or cfg.volatility_adjustment_enabled:
                returns_row["final_exposure"] = float(final_exposure)
            # else: regime_enabled_actual이 False면 regime/exposure 컬럼을 추가하지 않음
            
            returns_rows.append(returns_row)

            prev_w = new_w
            prev_holdings = g_sel[ticker_col].tolist()
        
        # Phase 종료 로그
        t_phase_end = time.time()
        phase_time = t_phase_end - t_phase_start
        logger.info(f"[L7 Runtime] Phase '{phase}' 완료: {rebalance_count}개 리밸런싱, 총 {phase_time:.1f}초, 평균 {phase_time/rebalance_count:.3f}초/리밸런싱")

    bt_positions = pd.DataFrame(positions_rows).sort_values(["phase", "date", "ticker"]).reset_index(drop=True)
    bt_returns = pd.DataFrame(returns_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    
    # [Stage13] bt_returns를 core와 diagnostics로 분리
    # 필수 컬럼: date, phase, net_return, gross_return, turnover_oneway 등 (validation 통과용)
    REQUIRED_CORE_COLS = [
        "date", "phase", "top_k", "holding_days", "cost_bps", "cost_bps_used",
        "weighting", "buffer_k", "n_tickers", "gross_return", "net_return",
        "turnover_oneway", "daily_trading_cost", "turnover_cost", "total_cost"
    ]
    
    # 진단 컬럼: regime, exposure 등 (결측 허용, 별도 저장)
    DIAGNOSTIC_COLS = ["regime", "exposure"]
    
    # core: 필수 컬럼만 포함 (validation 통과용)
    available_core_cols = [c for c in REQUIRED_CORE_COLS if c in bt_returns.columns]
    bt_returns_core = bt_returns[available_core_cols].copy()
    
    # diagnostics: 진단 컬럼만 포함 (결측 허용)
    available_diag_cols = [c for c in DIAGNOSTIC_COLS if c in bt_returns.columns]
    if available_diag_cols:
        bt_returns_diagnostics = bt_returns[["date", "phase"] + available_diag_cols].copy()
    else:
        # 진단 컬럼이 없으면 빈 DataFrame 생성
        bt_returns_diagnostics = pd.DataFrame(columns=["date", "phase"])
    
    # [Stage13] selection_diagnostics DataFrame 생성
    selection_diagnostics = pd.DataFrame(selection_diagnostics_rows).sort_values(["phase", "date"]).reset_index(drop=True)

    # equity curve
    eq_rows: List[dict] = []
    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        eq = 1.0
        peak = 1.0
        for dt, r in zip(g["date"], g["net_return"]):
            eq *= (1.0 + float(r))
            peak = max(peak, eq)
            dd = (eq / peak) - 1.0
            eq_rows.append({"date": dt, "phase": phase, "equity": float(eq), "drawdown": float(dd)})
    bt_equity_curve = pd.DataFrame(eq_rows).sort_values(["phase", "date"]).reset_index(drop=True)

    # metrics
    met_rows: List[dict] = []
    periods_per_year = 252.0 / float(cfg.holding_days) if cfg.holding_days > 0 else 12.6

    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        r_gross = g["gross_return"].astype(float).to_numpy()
        r_net = g["net_return"].astype(float).to_numpy()

        eq_g = float((1.0 + pd.Series(r_gross)).cumprod().iloc[-1]) if len(r_gross) else 1.0
        eq_n = float((1.0 + pd.Series(r_net)).cumprod().iloc[-1]) if len(r_net) else 1.0

        d0 = pd.to_datetime(g["date"].iloc[0]) if len(g) else pd.NaT
        d1 = pd.to_datetime(g["date"].iloc[-1]) if len(g) else pd.NaT
        # [오류 수정] timedelta64 호환성: pd.Timedelta로 변환하여 .days 사용
        years = max((pd.Timedelta(d1 - d0).days / 365.25) if pd.notna(d0) and pd.notna(d1) else 1e-9, 1e-9)

        # [오류 수정] 복소수 방지: eq_g가 음수이거나 0이면 -100% CAGR 처리
        # NaN 대신 -100%를 사용하여 최적화 알고리즘이 실패한 조합을 명확히 식별
        if eq_g <= 0:
            gross_cagr = -1.0  # -100% (완전 손실)
            if eq_g < 0:
                logger.warning(f"Phase {phase}: Portfolio equity is negative (eq_g={eq_g:.4f}). Setting CAGR=-100%")
        elif years > 0:
            try:
                gross_cagr_val = eq_g ** (1.0 / years) - 1.0
                # 복소수인 경우 실수부만 사용, 그래도 문제가 있으면 -100% 처리
                if isinstance(gross_cagr_val, complex):
                    gross_cagr = float(gross_cagr_val.real)
                    if np.isnan(gross_cagr) or np.isinf(gross_cagr):
                        gross_cagr = -1.0
                        logger.warning(f"Phase {phase}: Complex CAGR resulted in invalid value. Setting CAGR=-100%")
                else:
                    gross_cagr = float(gross_cagr_val)
                    if np.isnan(gross_cagr) or np.isinf(gross_cagr):
                        gross_cagr = -1.0
            except (ValueError, OverflowError, TypeError) as e:
                gross_cagr = -1.0
                logger.warning(f"Phase {phase}: CAGR calculation error ({type(e).__name__}: {e}). Setting CAGR=-100%")
        else:
            gross_cagr = -1.0
        
        if eq_n <= 0:
            net_cagr = -1.0  # -100% (완전 손실)
            if eq_n < 0:
                logger.warning(f"Phase {phase}: Portfolio net equity is negative (eq_n={eq_n:.4f}). Setting CAGR=-100%")
        elif years > 0:
            try:
                net_cagr_val = eq_n ** (1.0 / years) - 1.0
                # 복소수인 경우 실수부만 사용, 그래도 문제가 있으면 -100% 처리
                if isinstance(net_cagr_val, complex):
                    net_cagr = float(net_cagr_val.real)
                    if np.isnan(net_cagr) or np.isinf(net_cagr):
                        net_cagr = -1.0
                        logger.warning(f"Phase {phase}: Complex CAGR resulted in invalid value. Setting CAGR=-100%")
                else:
                    net_cagr = float(net_cagr_val)
                    if np.isnan(net_cagr) or np.isinf(net_cagr):
                        net_cagr = -1.0
            except (ValueError, OverflowError, TypeError) as e:
                net_cagr = -1.0
                logger.warning(f"Phase {phase}: Net CAGR calculation error ({type(e).__name__}: {e}). Setting CAGR=-100%")
        else:
            net_cagr = -1.0

        gross_vol = float(np.std(r_gross, ddof=1) * np.sqrt(periods_per_year)) if len(r_gross) > 1 else 0.0
        net_vol = float(np.std(r_net, ddof=1) * np.sqrt(periods_per_year)) if len(r_net) > 1 else 0.0

        gross_sharpe = float((np.mean(r_gross) / (np.std(r_gross, ddof=1) + 1e-12)) * np.sqrt(periods_per_year)) if len(r_gross) > 1 else 0.0
        net_sharpe = float((np.mean(r_net) / (np.std(r_net, ddof=1) + 1e-12)) * np.sqrt(periods_per_year)) if len(r_net) > 1 else 0.0

        mdd_g = _mdd(r_gross) if len(r_gross) else 0.0
        mdd_n = _mdd(r_net) if len(r_net) else 0.0
        
        # [오류 수정] 포트폴리오가 손실이면 MDD도 -100%로 설정
        if eq_g <= 0:
            mdd_g = -1.0
        if eq_n <= 0:
            mdd_n = -1.0

        # [Stage1] gross_minus_net_total_return_pct 계산
        gross_total_return_pct = float(eq_g - 1.0)
        net_total_return_pct = float(eq_n - 1.0)
        gross_minus_net_total_return_pct = float(gross_total_return_pct - net_total_return_pct)
        
        # [Stage1] avg_cost_pct 계산: 평균 비용을 퍼센트로 (total_cost의 평균)
        avg_cost_pct = float(g["total_cost"].mean() * 100.0) if "total_cost" in g.columns and len(g) > 0 else 0.0
        
        # [최종 수치셋] Calmar Ratio 계산: CAGR / |MDD|
        def _calculate_calmar_ratio(cagr: float, mdd: float) -> float:
            """Calmar Ratio = CAGR / |MDD|"""
            if mdd == 0:
                return float('inf') if cagr > 0 else 0.0
            abs_mdd = abs(mdd)
            if abs_mdd < 1e-9:  # MDD가 거의 0이면
                return float('inf') if cagr > 0 else 0.0
            return float(cagr / abs_mdd)
        
        gross_calmar = _calculate_calmar_ratio(gross_cagr, mdd_g)
        net_calmar = _calculate_calmar_ratio(net_cagr, mdd_n)
        
        # [최종 수치셋] Profit Factor 계산: 총 이익 / 총 손실
        def _calculate_profit_factor(returns: np.ndarray) -> float:
            """Profit Factor = sum(양수 수익) / abs(sum(음수 수익))"""
            profits = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            if losses == 0:
                return float('inf') if profits > 0 else 0.0
            return float(profits / losses)
        
        gross_profit_factor = _calculate_profit_factor(r_gross) if len(r_gross) > 0 else 0.0
        net_profit_factor = _calculate_profit_factor(r_net) if len(r_net) > 0 else 0.0
        
        # [최종 수치셋] Avg Trade Duration 계산: 평균 보유 일수
        # positions 데이터에서 각 종목의 보유 기간 계산
        avg_trade_duration = np.nan
        if len(bt_positions) > 0:
            phase_positions = bt_positions[bt_positions["phase"] == phase].copy()
            if len(phase_positions) > 0:
                phase_positions["date"] = pd.to_datetime(phase_positions["date"])
                phase_positions = phase_positions.sort_values(["ticker", "date"])
                
                # 각 종목별로 연속 보유 기간 계산
                durations = []
                for ticker, ticker_positions in phase_positions.groupby("ticker", sort=False):
                    ticker_positions = ticker_positions.sort_values("date")
                    if len(ticker_positions) > 1:
                        # 연속된 날짜 간격 계산
                        dates = ticker_positions["date"].values
                        for i in range(len(dates) - 1):
                            # [오류 수정] numpy.timedelta64는 .days 속성이 없으므로 pd.Timedelta로 변환
                            days_diff = pd.Timedelta(dates[i+1] - dates[i]).days
                            if days_diff <= cfg.holding_days * 2:  # 리밸런싱 주기 내 연속 보유
                                durations.append(days_diff)
                
                if len(durations) > 0:
                    avg_trade_duration = float(np.mean(durations))
                else:
                    # 보유 기간을 직접 계산할 수 없으면 holding_days를 기본값으로 사용
                    avg_trade_duration = float(cfg.holding_days)
        
        met_rows.append(
            {
                "phase": phase,
                "top_k": int(cfg.top_k),
                "holding_days": int(cfg.holding_days),
                "cost_bps": float(cfg.cost_bps),  # [Stage1] 하위 호환성 유지
                "cost_bps_used": float(cost_bps_used),  # [Stage1] 실제 사용된 cost_bps
                "cost_bps_config": float(cost_bps_config),  # [Stage1] config.yaml에서 읽은 원본 cost_bps
                "cost_bps_mismatch_flag": bool(cost_bps_mismatch_flag),  # [Stage1] 불일치 플래그
                "gross_minus_net_total_return_pct": float(gross_minus_net_total_return_pct),  # [Stage1] gross - net 차이 (퍼센트)
                "avg_cost_pct": float(avg_cost_pct),  # [Stage1] 평균 비용 (퍼센트)
                "buffer_k": int(cfg.buffer_k),
                "n_rebalances": int(len(g)),
                "gross_total_return": float(gross_total_return_pct),
                "net_total_return": float(net_total_return_pct),
                "gross_cagr": gross_cagr,
                "net_cagr": net_cagr,
                "gross_vol_ann": gross_vol,
                "net_vol_ann": net_vol,
                "gross_sharpe": gross_sharpe,
                "net_sharpe": net_sharpe,
                "gross_mdd": float(mdd_g),
                "net_mdd": float(mdd_n),
                "gross_hit_ratio": float((r_gross > 0).mean()) if len(r_gross) else np.nan,
                "net_hit_ratio": float((r_net > 0).mean()) if len(r_net) else np.nan,
                "avg_turnover_oneway": float(g["turnover_oneway"].mean()) if len(g) else np.nan,
                "avg_n_tickers": float(g["n_tickers"].mean()) if len(g) else np.nan,
                # [최종 수치셋] Calmar Ratio 추가
                "gross_calmar_ratio": float(gross_calmar) if not np.isinf(gross_calmar) else np.nan,
                "net_calmar_ratio": float(net_calmar) if not np.isinf(net_calmar) else np.nan,
                # [최종 수치셋] Profit Factor 추가
                "gross_profit_factor": float(gross_profit_factor) if not np.isinf(gross_profit_factor) else np.nan,
                "net_profit_factor": float(net_profit_factor) if not np.isinf(net_profit_factor) else np.nan,
                # [최종 수치셋] Avg Trade Duration 추가
                "avg_trade_duration": avg_trade_duration,
                "date_start": d0,
                "date_end": d1,
                "weighting": cfg.weighting,
            }
        )

    bt_metrics = pd.DataFrame(met_rows)
    
    # [최종 수치셋] 국면별 성과 계산 (bt_positions, bt_returns_core 생성 후)
    regime_metrics_rows: List[dict] = []
    if regime_enabled_actual and len(bt_returns_diagnostics) > 0 and "regime" in bt_returns_diagnostics.columns:
        # bt_returns와 bt_returns_diagnostics 병합
        bt_returns_with_regime = bt_returns_core.merge(
            bt_returns_diagnostics[["date", "phase", "regime"]],
            on=["date", "phase"],
            how="left"
        )
        
        # 국면별로 그룹화하여 성과 계산
        for phase, phase_data in bt_returns_with_regime.groupby("phase", sort=False):
            phase_data = phase_data.sort_values("date").reset_index(drop=True)
            
            for regime, regime_data in phase_data.groupby("regime", sort=False):
                if pd.isna(regime) or len(regime_data) == 0:
                    continue
                
                regime_data = regime_data.sort_values("date").reset_index(drop=True)
                r_net_regime = regime_data["net_return"].astype(float).to_numpy()
                
                if len(r_net_regime) == 0:
                    continue
                
                eq_n_regime = float((1.0 + pd.Series(r_net_regime)).cumprod().iloc[-1])
                d0_regime = pd.to_datetime(regime_data["date"].iloc[0])
                d1_regime = pd.to_datetime(regime_data["date"].iloc[-1])
                # [오류 수정] timedelta64 호환성: pd.Timedelta로 변환하여 .days 사용
                years_regime = max((pd.Timedelta(d1_regime - d0_regime).days / 365.25) if pd.notna(d0_regime) and pd.notna(d1_regime) else 1e-9, 1e-9)
                
                net_cagr_regime = float(eq_n_regime ** (1.0 / years_regime) - 1.0) if eq_n_regime > 0 and years_regime > 0 else -1.0
                net_vol_regime = float(np.std(r_net_regime, ddof=1) * np.sqrt(periods_per_year)) if len(r_net_regime) > 1 else 0.0
                net_sharpe_regime = float((np.mean(r_net_regime) / (np.std(r_net_regime, ddof=1) + 1e-12)) * np.sqrt(periods_per_year)) if len(r_net_regime) > 1 else 0.0
                
                mdd_regime = _mdd(r_net_regime) if len(r_net_regime) else 0.0
                if eq_n_regime <= 0:
                    mdd_regime = -1.0
                
                net_hit_ratio_regime = float((r_net_regime > 0).mean()) if len(r_net_regime) else np.nan
                net_total_return_regime = float(eq_n_regime - 1.0)
                
                regime_metrics_rows.append({
                    "phase": phase,
                    "regime": str(regime),
                    "n_rebalances": int(len(regime_data)),
                    "net_total_return": net_total_return_regime,
                    "net_cagr": net_cagr_regime,
                    "net_sharpe": net_sharpe_regime,
                    "net_mdd": float(mdd_regime),
                    "net_hit_ratio": net_hit_ratio_regime,
                    "date_start": d0_regime,
                    "date_end": d1_regime,
                })
    
    bt_regime_metrics = pd.DataFrame(regime_metrics_rows) if regime_metrics_rows else pd.DataFrame()

    quality = {
        "backtest": {
            "holding_days": int(cfg.holding_days),
            "top_k": int(cfg.top_k),
            "cost_bps": float(cfg.cost_bps),
            "buffer_k": int(cfg.buffer_k),
            "score_col_used": score_col,
            "ret_col_used": ret_col,
            "weighting": cfg.weighting,
            "softmax_temp": float(cfg.softmax_temp),
            "rows_positions": int(len(bt_positions)),
            "rows_returns": int(len(bt_returns)),
        }
    }

    # [Stage13] 런타임 프로파일 DataFrame 생성
    runtime_profile = pd.DataFrame(runtime_profile_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    
    # [Stage13] selection_diagnostics 반환
    # bt_returns_core를 bt_returns로 사용 (validation 통과용)
    # bt_returns_diagnostics는 별도로 저장
    # [최종 수치셋] bt_regime_metrics 추가 반환
    return bt_positions, bt_returns_core, bt_equity_curve, bt_metrics, quality, warns, selection_diagnostics, bt_returns_diagnostics, runtime_profile, bt_regime_metrics
