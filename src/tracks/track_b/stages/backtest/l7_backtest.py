# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/backtest/l7_backtest.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# [Stage13] selector import
from src.components.portfolio.selector import select_topk_with_fallback

# [bt20 프로페셔널] 적응형 리밸런싱 import
from src.features.adaptive_rebalancing_fixed import AdaptiveRebalancing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 20
    top_k: int = 20
    cost_bps: float = 10.0
    slippage_bps: float = 0.0  # [개선안 3번] 시장 임팩트/슬리피지 비용 (bps)
    score_col: str = "score_ens"
    ret_col: str = "true_short"
    weighting: str = "equal"  # equal | softmax
    softmax_temp: float = 1.0

    # [개선안 36번] 오버래핑 트랜치(필수): 월별 신규 포트 추가로 Rebalances 증가 → 타이밍 럭 제거
    overlapping_tranches_enabled: bool = False
    tranche_holding_days: int = 120  # 캘린더 day 기준 만기(예: 120일)
    tranche_max_active: int = 4  # 동시 보유 트랜치 수(예: 월별 4트랜치)
    tranche_allocation_mode: str = "fixed_equal"  # fixed_equal | active_equal

    # Turnover 완화 옵션
    # 이전 보유 종목이 상위(top_k + buffer_k) 안에 있으면 유지 후보로 인정
    # [리팩토링] config.yaml과 통일: 모든 전략에서 15 사용 (bt20_ens는 20)
    buffer_k: int = 15

    # [Phase 7 Step 2] Turnover 제어: 리밸런싱 주기 완화
    # rebalance_interval=1이면 모든 리밸런싱 실행, 2면 매 2번째만 실행 (빈도 50% 감소)
    # 목표: Avg Turnover ≤ 500% 달성
    # [리팩토링] config.yaml과 통일: 모든 전략에서 20 사용 (holding_days와 동일)
    rebalance_interval: int = 20  # 기본값 20 (월별 리밸런싱)

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
    regime_top_k_bull: Optional[
        int
    ] = None  # bull 시장에서 사용할 top_k (5단계 미지정 시 사용)
    regime_top_k_bear: Optional[
        int
    ] = None  # bear 시장에서 사용할 top_k (5단계 미지정 시 사용)
    regime_exposure_bull: Optional[
        float
    ] = None  # bull 시장에서 사용할 exposure (5단계 미지정 시 사용)
    regime_exposure_bear: Optional[
        float
    ] = None  # bear 시장에서 사용할 exposure (5단계 미지정 시 사용)

    # [Phase 8 Step 2 방안1] 리밸런싱 규칙/버퍼/노출 튜닝: Dev 붕괴 완화
    # [리팩토링] config.yaml과 통일: 모든 전략에서 True 사용
    smart_buffer_enabled: bool = True  # 스마트 버퍼링 활성화
    smart_buffer_stability_threshold: float = 0.7  # 보유 종목 안정성 임계값
    # [리팩토링] config.yaml과 통일: 모든 전략에서 True 사용
    volatility_adjustment_enabled: bool = True  # 변동성 기반 exposure 조정 활성화
    volatility_lookback_days: int = 60  # 변동성 계산 기간
    target_volatility: float = 0.15  # 목표 변동성 (15%)
    volatility_adjustment_max: float = 1.2  # 변동성 조정 최대 배수
    volatility_adjustment_min: float = 0.6  # 변동성 조정 최소 배수

    # [Phase 8 Step 2 방안2] 국면 필터/리스크 스케일링: Bear 구간 방어 강화
    # [리팩토링] config.yaml과 통일: 모든 전략에서 True 사용
    risk_scaling_enabled: bool = True  # 국면별 리스크 스케일링 활성화
    risk_scaling_bear_multiplier: float = 0.7  # Bear 구간 추가 포지션 축소 배수
    risk_scaling_neutral_multiplier: float = 0.9  # Neutral 구간 포지션 축소 배수
    risk_scaling_bull_multiplier: float = 1.0  # Bull 구간 포지션 축소 배수

    # [bt20 프로페셔널] 적응형 리밸런싱 설정
    adaptive_rebalancing_enabled: bool = False  # 적응형 리밸런싱 활성화
    signal_strength_thresholds: dict[str, float] = None  # 신호 강도 임계값
    rebalance_intervals: dict[str, int] = None  # 리밸런싱 간격 설정

    # [개선안: 일별 mark-to-market 백테스트] 리밸런싱 사이에도 일별 수익률 계산
    daily_backtest_enabled: bool = False  # 일별 백테스트 활성화 (월별 누적수익률 생성)


def _apply_dynamic_params_for_holding_days(
    cfg: BacktestConfig, strategy_name: str = None
) -> BacktestConfig:
    """
    [동적 기간 파라미터 적용] config.yaml의 전략별 설정 우선 적용, 동적 파라미터는 보조적으로 사용

    Args:
        cfg: 원본 BacktestConfig
        strategy_name: 전략 이름 (bt20_short, bt20_ens, bt120_long 등)

    Returns:
        업데이트된 BacktestConfig
    """
    try:
        # config.yaml 경로 찾기 (프로젝트 루트에서 찾기)
        current_file = Path(__file__)
        # C:\Users\seong\OneDrive\Desktop\bootcamp\000_code\src\tracks\track_b\stages\backtest\l7_backtest.py
        # 에서 C:\Users\seong\OneDrive\Desktop\bootcamp\000_code\configs\config.yaml로 이동
        config_path = (
            current_file.parent.parent.parent.parent.parent / "configs" / "config.yaml"
        )

        # 절대 경로로도 시도
        if not config_path.exists():
            config_path = Path(
                "C:/Users/seong/OneDrive/Desktop/bootcamp/000_code/configs/config.yaml"
            )
        if not config_path.exists():
            logger.warning(
                f"[동적 파라미터] config.yaml을 찾을 수 없습니다: {config_path}"
            )
            return cfg

        # config.yaml 로드
        with open(config_path, encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        # 우선순위 설정 확인 (기본: config 우선)
        priority_config = full_config.get("parameter_priority", "config_first")
        use_dynamic_fallback = priority_config == "dynamic_fallback"

        # holding_days에 따른 동적 파라미터 (fallback용)
        dynamic_params = full_config.get("holding_days_dynamic_params", {})
        holding_days = cfg.holding_days
        dynamic_available = holding_days in dynamic_params if dynamic_params else False

        if dynamic_available:
            dynamic_param_values = dynamic_params[holding_days]
            logger.info(
                f"[동적 파라미터] holding_days={holding_days} 파라미터 준비: {dynamic_param_values}"
            )
        else:
            dynamic_param_values = {}
            if not use_dynamic_fallback:
                logger.info(
                    f"[동적 파라미터] holding_days={holding_days}에 대한 동적 파라미터 없음 (config 우선 모드)"
                )

        # 파라미터 적용 로직
        if use_dynamic_fallback:
            # 동적 파라미터 우선 모드 (기존 방식)
            logger.info("[동적 파라미터] 동적 우선 모드: 동적 파라미터 우선 적용")
            updated_cfg = cfg.__class__(
                holding_days=cfg.holding_days,
                top_k=dynamic_param_values.get("top_k", cfg.top_k),
                cost_bps=dynamic_param_values.get("cost_bps", cfg.cost_bps),
                slippage_bps=dynamic_param_values.get("slippage_bps", cfg.slippage_bps),
                score_col=cfg.score_col,
                ret_col=cfg.ret_col,
                weighting=cfg.weighting,
                softmax_temp=cfg.softmax_temp,
                overlapping_tranches_enabled=cfg.overlapping_tranches_enabled,
                tranche_holding_days=dynamic_param_values.get(
                    "tranche_days", cfg.tranche_holding_days
                ),
                tranche_max_active=cfg.tranche_max_active,
                tranche_allocation_mode=cfg.tranche_allocation_mode,
                buffer_k=dynamic_param_values.get("buffer_k", cfg.buffer_k),
                rebalance_interval=dynamic_param_values.get(
                    "rebalance_interval", cfg.rebalance_interval
                ),
                diversify_enabled=cfg.diversify_enabled,
                group_col=cfg.group_col,
                max_names_per_group=cfg.max_names_per_group,
                regime_enabled=dynamic_param_values.get(
                    "regime_enabled", cfg.regime_enabled
                ),
                regime_top_k_bull_strong=cfg.regime_top_k_bull_strong,  # 유지
                regime_top_k_bull_weak=cfg.regime_top_k_bull_weak,  # 유지
                regime_top_k_bear_strong=cfg.regime_top_k_bear_strong,  # 유지
                regime_top_k_bear_weak=cfg.regime_top_k_bear_weak,  # 유지
                regime_top_k_neutral=cfg.regime_top_k_neutral,  # 유지
                regime_exposure_bull_strong=cfg.regime_exposure_bull_strong,  # 유지
                regime_exposure_bull_weak=cfg.regime_exposure_bull_weak,  # 유지
                regime_exposure_bear_strong=cfg.regime_exposure_bear_strong,  # 유지
                regime_exposure_bear_weak=cfg.regime_exposure_bear_weak,  # 유지
                regime_exposure_neutral=cfg.regime_exposure_neutral,  # 유지
                regime_top_k_bull=cfg.regime_top_k_bull,  # 유지
                regime_top_k_bear=cfg.regime_top_k_bear,  # 유지
                regime_exposure_bull=cfg.regime_exposure_bull,  # 유지
                regime_exposure_bear=cfg.regime_exposure_bear,  # 유지
                # 동적 파라미터 적용
                smart_buffer_enabled=cfg.smart_buffer_enabled,  # 유지
                smart_buffer_stability_threshold=cfg.smart_buffer_stability_threshold,  # 유지
                volatility_adjustment_enabled=cfg.volatility_adjustment_enabled,  # 유지
                volatility_lookback_days=cfg.volatility_lookback_days,  # 유지
                target_volatility=dynamic_param_values.get(
                    "target_vol", cfg.target_volatility
                ),  # 동적 적용
                volatility_adjustment_max=cfg.volatility_adjustment_max,  # 유지
                volatility_adjustment_min=cfg.volatility_adjustment_min,  # 유지
                # 동적 파라미터 적용
                risk_scaling_enabled=cfg.risk_scaling_enabled,  # 유지
                risk_scaling_bear_multiplier=cfg.risk_scaling_bear_multiplier,  # 유지
                risk_scaling_neutral_multiplier=cfg.risk_scaling_neutral_multiplier,  # 유지
                risk_scaling_bull_multiplier=cfg.risk_scaling_bull_multiplier,  # 유지
                # 유지
                adaptive_rebalancing_enabled=cfg.adaptive_rebalancing_enabled,
                signal_strength_thresholds=cfg.signal_strength_thresholds,
                rebalance_intervals=cfg.rebalance_intervals,
            )
        else:
            # config 우선 모드 (기본값): 전략별 설정 우선, 동적 파라미터 보조
            logger.info("[동적 파라미터] config 우선 모드: 전략별 설정 우선 적용")

            updated_cfg = cfg.__class__(
                # 기본 필드: 전략별 설정 유지
                holding_days=cfg.holding_days,
                top_k=cfg.top_k,  # 전략별 설정 유지
                cost_bps=cfg.cost_bps,  # 전략별 설정 유지
                slippage_bps=cfg.slippage_bps,  # 전략별 설정 유지
                score_col=cfg.score_col,
                ret_col=cfg.ret_col,
                weighting=cfg.weighting,
                softmax_temp=cfg.softmax_temp,
                # tranche 관련: 동적 파라미터 우선 (holding_days에 따라 필수)
                overlapping_tranches_enabled=cfg.overlapping_tranches_enabled,
                tranche_holding_days=dynamic_param_values.get("tranche_days")
                or cfg.tranche_holding_days,
                tranche_max_active=cfg.tranche_max_active,
                tranche_allocation_mode=cfg.tranche_allocation_mode,
                # 선택/리밸런싱: 동적 파라미터 우선 (holding_days 최적화)
                buffer_k=dynamic_param_values.get("buffer_k", cfg.buffer_k),
                rebalance_interval=dynamic_param_values.get(
                    "rebalance_interval", cfg.rebalance_interval
                ),
                # 분산투자: 전략별 설정 유지
                diversify_enabled=cfg.diversify_enabled,
                group_col=cfg.group_col,
                max_names_per_group=cfg.max_names_per_group,
                # regime: 동적 파라미터 우선 (시장 상황에 따라)
                regime_enabled=dynamic_param_values.get(
                    "regime_enabled", cfg.regime_enabled
                ),
                regime_top_k_bull_strong=cfg.regime_top_k_bull_strong,
                regime_top_k_bull_weak=cfg.regime_top_k_bull_weak,
                regime_top_k_bear_strong=cfg.regime_top_k_bear_strong,
                regime_top_k_bear_weak=cfg.regime_top_k_bear_weak,
                regime_top_k_neutral=cfg.regime_top_k_neutral,
                regime_exposure_bull_strong=cfg.regime_exposure_bull_strong,
                regime_exposure_bull_weak=cfg.regime_exposure_bull_weak,
                regime_exposure_bear_strong=cfg.regime_exposure_bear_strong,
                regime_exposure_bear_weak=cfg.regime_exposure_bear_weak,
                regime_exposure_neutral=cfg.regime_exposure_neutral,
                regime_top_k_bull=cfg.regime_top_k_bull,
                regime_top_k_bear=cfg.regime_top_k_bear,
                regime_exposure_bull=cfg.regime_exposure_bull,
                regime_exposure_bear=cfg.regime_exposure_bear,
                # 리스크 관리: 전략별 설정 유지 (중요 파라미터)
                smart_buffer_enabled=cfg.smart_buffer_enabled,
                smart_buffer_stability_threshold=cfg.smart_buffer_stability_threshold,
                volatility_adjustment_enabled=cfg.volatility_adjustment_enabled,
                volatility_lookback_days=cfg.volatility_lookback_days,
                target_volatility=cfg.target_volatility,  # 전략별 설정 유지 (중요!)
                volatility_adjustment_max=cfg.volatility_adjustment_max,
                volatility_adjustment_min=cfg.volatility_adjustment_min,
                risk_scaling_enabled=cfg.risk_scaling_enabled,
                risk_scaling_bear_multiplier=cfg.risk_scaling_bear_multiplier,
                risk_scaling_neutral_multiplier=cfg.risk_scaling_neutral_multiplier,
                risk_scaling_bull_multiplier=cfg.risk_scaling_bull_multiplier,
                adaptive_rebalancing_enabled=cfg.adaptive_rebalancing_enabled,
                signal_strength_thresholds=cfg.signal_strength_thresholds,
                rebalance_intervals=cfg.rebalance_intervals,
            )

        logger.info(
            f"[동적 파라미터] 파라미터 적용 완료: top_k={updated_cfg.top_k}, cost_bps={updated_cfg.cost_bps}, target_vol={updated_cfg.target_volatility}"
        )
        return updated_cfg

    except Exception as e:
        logger.error(f"[동적 파라미터] 적용 중 오류 발생: {e}")
        return cfg


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


def _pick_ret_col(df: pd.DataFrame, preferred: str, holding_days: int = 20) -> str:
    """[동적 Return 계산] holding_days에 따라 적절한 return 컬럼 선택"""
    # [실제 데이터 기반] L6 데이터의 실제 컬럼 구조에 맞게 수정

    # 실제 데이터에는 true_short, true_long 컬럼만 존재
    # holding_days에 따라 적절한 컬럼 선택:
    # - 20, 40, 60: 단기 return (true_short)
    # - 80, 100, 120: 장기 return (true_long)

    if holding_days <= 60:
        # 단기 전략: true_short 사용
        if "true_short" in df.columns:
            return "true_short"
    else:
        # 장기 전략: true_long 사용
        if "true_long" in df.columns:
            return "true_long"

    # 1. 우선적으로 preferred 컬럼 확인
    if preferred and preferred in df.columns:
        return preferred

    # 2. 일반적인 true 컬럼들 확인
    for c in ["true_short", "true_long", "y_true", "ret"]:
        if c in df.columns:
            return c

    # 3. ret_fwd 패턴의 컬럼들 확인 (만약 존재한다면)
    ret_cols = [col for col in df.columns if col.startswith("ret_fwd_")]
    if ret_cols:
        # holding_days에 가장 가까운 기간의 컬럼 선택
        best_match = min(
            ret_cols,
            key=lambda x: abs(int(x.split("_")[-1].replace("d", "")) - holding_days),
        )
        return best_match

    raise KeyError(
        f"return/true column not found for holding_days={holding_days}. Available: {list(df.columns)}"
    )


def _compute_turnover_oneway(
    prev_w: dict[str, float], new_w: dict[str, float]
) -> float:
    keys = set(prev_w) | set(new_w)
    s = 0.0
    for k in keys:
        s += abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0))
    return 0.5 * s


def _calculate_trading_cost(
    *,
    turnover_oneway: float,
    cost_bps: float,
    slippage_bps: float,
    exposure: float,
) -> dict[str, float]:
    """
    [개선안 1번][개선안 3번] 거래비용/슬리피지 계산 (턴오버 기반)

    기존 문제:
    - turnover_cost를 계산해놓고 실제 차감은 "리밸런싱 발생 시 고정 cost_bps"로 처리되어 과다 차감 가능

    정책:
    - 거래비용은 '거래된 금액'에 비례: traded_value = turnover_oneway * |exposure|
    - cost_bps(수수료/세금/스프레드) + slippage_bps(시장임팩트)를 합산 적용

    Args:
        turnover_oneway: 0~1 (one-way turnover)
        cost_bps: 기본 거래비용(bps)
        slippage_bps: 슬리피지 비용(bps)
        exposure: 포트폴리오 노출(레버리지 포함 가능)

    Returns:
        dict:
          - traded_value
          - cost_component
          - slippage_component
          - total_cost
    """
    tv = float(max(turnover_oneway, 0.0)) * float(abs(exposure))
    cb = float(max(cost_bps, 0.0))
    sb = float(max(slippage_bps, 0.0))
    cost_component = tv * cb / 10000.0
    slippage_component = tv * sb / 10000.0
    total_cost = cost_component + slippage_component
    return {
        "traded_value": float(tv),
        "cost_component": float(cost_component),
        "slippage_component": float(slippage_component),
        "total_cost": float(total_cost),
    }


def _mdd(rr: np.ndarray) -> float:
    """[최종 수치셋] MDD 계산 함수"""
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rr:
        eq *= 1.0 + float(r)
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


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """
    [개선안 34번] 안전한 상관계수 계산(결측/상수열 방지)
    """
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    m = aa.notna() & bb.notna()
    if int(m.sum()) < 3:
        return float("nan")
    aa = aa[m]
    bb = bb[m]
    if float(aa.std(ddof=1)) < 1e-12 or float(bb.std(ddof=1)) < 1e-12:
        return float("nan")
    return float(aa.corr(bb))


def _rank_ic(scores: pd.Series, rets: pd.Series) -> float:
    """
    [개선안 34번] Rank IC(Spearman): rank(score) vs rank(ret) Pearson corr
    """
    s = pd.to_numeric(scores, errors="coerce")
    r = pd.to_numeric(rets, errors="coerce")
    return _safe_corr(s.rank(method="average"), r.rank(method="average"))


def _long_short_alpha(scores: pd.Series, rets: pd.Series, k: int) -> float:
    """
    [개선안 34번] Long/Short Alpha = mean(top-k ret) - mean(bottom-k ret)
    """
    k = int(k)
    if k <= 0:
        return float("nan")
    df = pd.DataFrame(
        {
            "s": pd.to_numeric(scores, errors="coerce"),
            "r": pd.to_numeric(rets, errors="coerce"),
        }
    )
    df = df.dropna()
    if len(df) < max(2 * k, 6):
        return float("nan")
    df = df.sort_values("s", ascending=False).reset_index(drop=True)
    top = df.head(k)["r"].mean()
    bot = df.tail(k)["r"].mean()
    if pd.isna(top) or pd.isna(bot):
        return float("nan")
    return float(top - bot)


def _select_with_diversification(
    g_sorted: pd.DataFrame,
    *,
    ticker_col: str,
    top_k: int,
    buffer_k: int,
    prev_holdings: list[str],
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
    if (
        group_col is None
        or max_names_per_group is None
        or group_col not in g_sorted.columns
    ):
        return _select_with_buffer(
            g_sorted,
            ticker_col=ticker_col,
            top_k=top_k,
            buffer_k=buffer_k,
            prev_holdings=prev_holdings,
        )

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
    group_counts: dict[str, int] = {}

    # 1) keep 먼저 (업종 제약 고려)
    for t in keep:
        ticker_row = allow[allow[ticker_col].astype(str) == t]
        if len(ticker_row) > 0:
            sector = (
                str(ticker_row.iloc[0][group_col])
                if pd.notna(ticker_row.iloc[0][group_col])
                else "기타"
            )
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
    prev_holdings: list[str],
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
    prev_holdings: list[str],
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
    recent_window = (
        recent_returns[-lookback_days:]
        if len(recent_returns) >= lookback_days
        else recent_returns
    )
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


def _calculate_daily_portfolio_returns(
    *,
    rebalance_dates: list[pd.Timestamp],
    daily_prices: pd.DataFrame,
    positions_at_rebalance: dict[pd.Timestamp, dict[str, float]],
    cost_bps: float,
    slippage_bps: float,
    volatility_adjustment_enabled: bool,
    volatility_lookback_days: int,
    target_volatility: float,
    volatility_adjustment_max: float,
    volatility_adjustment_min: float,
    risk_scaling_enabled: bool,
    risk_scaling_bear_multiplier: float,
    risk_scaling_neutral_multiplier: float,
    risk_scaling_bull_multiplier: float,
    market_regime: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    [개선안: 일별 mark-to-market 백테스트] 리밸런싱 사이에도 일별 포트 수익률 계산

    Args:
        rebalance_dates: 리밸런싱 날짜 리스트
        daily_prices: 일별 가격 데이터 (date, ticker, close 컬럼)
        positions_at_rebalance: 리밸런싱 시점별 포지션 {date: {ticker: weight}}
        cost_bps: 거래비용
        slippage_bps: 슬리피지 비용
        volatility_adjustment_enabled: 변동성 조정 활성화
        기타 파라미터들...

    Returns:
        일별 포트 수익률 DataFrame (date, portfolio_return, turnover_cost 등)
    """
    daily_returns = []

    # 리밸런싱 날짜 정렬
    rebalance_dates = sorted(rebalance_dates)

    # 현재 보유 포지션 초기화 (첫 리밸런싱 전에는 현금)
    current_weights = {}  # {ticker: weight}

    # 전일 종가 저장 (일별 수익률 계산용)
    prev_close_prices = {}  # {ticker: prev_close}

    # 변동성 조정용 수익률 히스토리
    volatility_history = []

    # 모든 거래일에 대해 반복
    all_dates = sorted(daily_prices["date"].unique())

    for current_date in all_dates:
        date_prices = daily_prices[daily_prices["date"] == current_date].set_index(
            "ticker"
        )

        # 현재 날짜가 리밸런싱 날짜인지 확인
        is_rebalance_date = current_date in rebalance_dates

        if is_rebalance_date:
            # 리밸런싱 날짜: 새 포지션으로 교체 + 비용 발생
            new_weights = positions_at_rebalance.get(current_date, {})

            # 턴오버 계산 및 비용 적용
            turnover = _compute_turnover_oneway(current_weights, new_weights)
            cost_breakdown = _calculate_trading_cost(
                turnover_oneway=turnover,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                exposure=1.0,  # 기본 exposure
            )
            trading_cost = cost_breakdown["total_cost"]

            # 포지션 교체
            current_weights = new_weights.copy()

            # 리밸런싱 당일: 전일 종가 업데이트 (다음 날 수익률 계산용)
            for ticker in current_weights.keys():
                if ticker in date_prices.index:
                    prev_close_prices[ticker] = date_prices.loc[ticker, "close"]

            # 리밸런싱 당일 수익률은 비용만 반영 (가격 변동 없음 가정)
            portfolio_return = -trading_cost
            exposure_multiplier = 1.0
            regime_multiplier = 1.0

        else:
            # 리밸런싱 사이 날짜: 보유 포지션의 일별 수익률 계산
            if not current_weights:
                # 포지션 없는 경우 수익률 0
                portfolio_return = 0.0
                trading_cost = 0.0
                exposure_multiplier = 1.0
                regime_multiplier = 1.0
            else:
                # 각 종목의 일별 수익률 계산: (당일종가 / 전일종가) - 1
                ticker_returns = {}
                portfolio_gross_return = 0.0
                valid_positions = 0

                for ticker, weight in current_weights.items():
                    if ticker in date_prices.index and ticker in prev_close_prices:
                        current_close = date_prices.loc[ticker, "close"]
                        prev_close = prev_close_prices[ticker]

                        if prev_close > 0:  # 0으로 나누기 방지
                            daily_return = (current_close / prev_close) - 1
                            # 전일 종가 업데이트
                            prev_close_prices[ticker] = current_close
                        else:
                            daily_return = 0.0  # 비정상적인 가격 데이터
                    else:
                        # 가격 데이터 없는 경우 수익률 0 (결측 처리)
                        daily_return = 0.0

                    ticker_returns[ticker] = daily_return
                    portfolio_gross_return += weight * daily_return
                    if daily_return != 0.0:  # 실제 수익률이 있는 포지션 카운트
                        valid_positions += 1

                # 변동성 조정 적용
                exposure_multiplier = 1.0
                if volatility_adjustment_enabled and len(volatility_history) >= 2:
                    exposure_multiplier = _calculate_volatility_adjustment(
                        recent_returns=np.array(
                            volatility_history[-volatility_lookback_days:]
                        ),
                        target_vol=target_volatility,
                        lookback_days=volatility_lookback_days,
                        max_mult=volatility_adjustment_max,
                        min_mult=volatility_adjustment_min,
                    )

                # 리스크 스케일링 적용 (시장 국면 기반)
                regime_multiplier = 1.0
                if risk_scaling_enabled and market_regime is not None:
                    # 현재 국면 찾기
                    current_regime_data = market_regime[
                        market_regime["date"] <= current_date
                    ]
                    if len(current_regime_data) > 0:
                        current_regime = current_regime_data.iloc[-1]["regime"]
                        regime_multiplier = _apply_risk_scaling(
                            base_exposure=1.0,
                            regime=current_regime,
                            risk_scaling_enabled=True,
                            bear_multiplier=risk_scaling_bear_multiplier,
                            neutral_multiplier=risk_scaling_neutral_multiplier,
                            bull_multiplier=risk_scaling_bull_multiplier,
                        )

                final_exposure = exposure_multiplier * regime_multiplier
                portfolio_return = portfolio_gross_return * final_exposure
                trading_cost = 0.0  # 리밸런싱 사이에는 비용 없음

                # 가중치 드리프트 업데이트 (가격 변동으로 인한 비중 변화 반영)
                if abs(portfolio_gross_return) > 1e-10:  # 수치 안정성 체크
                    for ticker in list(current_weights.keys()):
                        if ticker in ticker_returns:
                            # 새로운 가중치 = 기존 가중치 * (1 + 종목수익률) / (1 + 포트수익률)
                            drift_multiplier = (1 + ticker_returns[ticker]) / (
                                1 + portfolio_gross_return
                            )
                            current_weights[ticker] *= drift_multiplier

                # 변동성 히스토리 업데이트
                if volatility_adjustment_enabled:
                    volatility_history.append(portfolio_return)
                    max_history = volatility_lookback_days * 2
                    if len(volatility_history) > max_history:
                        volatility_history = volatility_history[-max_history:]

        # 일별 결과 저장
        daily_returns.append(
            {
                "date": current_date,
                "portfolio_return": portfolio_return,
                "trading_cost": trading_cost,
                "is_rebalance_date": is_rebalance_date,
                "n_positions": len(current_weights),
                "valid_positions": (
                    valid_positions if "valid_positions" in locals() else 0
                ),
                "exposure_multiplier": exposure_multiplier,
                "regime_multiplier": regime_multiplier,
                "final_exposure": exposure_multiplier * regime_multiplier,
            }
        )

    return pd.DataFrame(daily_returns)


def _convert_daily_to_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    일별 수익률을 월별 누적수익률로 변환

    Args:
        daily_returns: 일별 포트 수익률 DataFrame (net_return 컬럼 사용)

    Returns:
        월별 누적수익률 DataFrame (year_month, monthly_return, cumulative_return 등)
    """
    # 날짜를 datetime으로 변환
    daily_returns = daily_returns.copy()
    daily_returns["date"] = pd.to_datetime(daily_returns["date"])
    daily_returns = daily_returns.sort_values("date")

    # 월별 그룹화
    daily_returns["year_month"] = daily_returns["date"].dt.to_period("M")

    monthly_returns = []
    cumulative_equity = 1.0

    for ym, group in daily_returns.groupby("year_month"):
        # 월별 누적수익률 계산: (1 + r1) * (1 + r2) * ... - 1
        # net_return 컬럼 사용 (비용 차감 후 순수익률)
        monthly_cumprod = (1 + group["net_return"]).cumprod()
        monthly_return = monthly_cumprod.iloc[-1] - 1

        # 누적 equity 업데이트
        cumulative_equity *= 1 + monthly_return

        monthly_returns.append(
            {
                "year_month": str(ym),
                "monthly_return": monthly_return,
                "cumulative_return": cumulative_equity - 1,
                "equity_value": cumulative_equity,
                "n_trading_days": len(group),
                "avg_daily_return": group["net_return"].mean(),
                "volatility_daily": group["net_return"].std(),
                "total_trading_cost": (
                    group["daily_trading_cost"].sum()
                    if "daily_trading_cost" in group.columns
                    else 0.0
                ),
                "n_rebalance_dates": (
                    group["is_rebalance_date"].sum()
                    if "is_rebalance_date" in group.columns
                    else 0
                ),
            }
        )

    return pd.DataFrame(monthly_returns)


def _run_daily_backtest(
    rebalance_scores: pd.DataFrame,
    daily_prices: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    phase_col: str = "phase",
    config_cost_bps: Optional[float] = None,
    market_regime: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """
    [개선안: 일별 mark-to-market 백테스트] 리밸런싱 사이에도 일별 포트 수익률 계산

    기존 run_backtest와 동일한 인터페이스 유지하되, 일별 수익률 계산 후 월별로 집계하여 반환
    """
    warns: list[str] = []

    # [동적 기간 파라미터 적용] holding_days에 따라 config.yaml에서 동적 파라미터 로드 및 적용
    strategy_name = None
    if cfg.score_col == "score_total_short":
        strategy_name = "bt20_short"
    elif cfg.score_col == "score_total_long":
        strategy_name = "bt120_long"
    elif cfg.score_col == "score_ens":
        strategy_name = "bt20_ens"

    cfg = _apply_dynamic_params_for_holding_days(cfg, strategy_name)

    # [Stage1] config_cost_bps 처리
    cost_bps_config = (
        float(config_cost_bps) if config_cost_bps is not None else float(cfg.cost_bps)
    )
    cost_bps_used = float(cfg.cost_bps)
    cost_bps_mismatch_flag = bool(abs(cost_bps_used - cost_bps_config) > 1e-6)

    if cost_bps_mismatch_flag:
        warns.append(
            f"[Stage1] cost_bps mismatch: config={cost_bps_config}, used={cost_bps_used}"
        )

    # [Stage5] 시장 국면(regime) 처리
    regime_dict: dict[pd.Timestamp, str] = {}
    if cfg.regime_enabled:
        if market_regime is None or len(market_regime) == 0:
            warns.append(
                "[Stage5] regime_enabled=True이지만 market_regime이 제공되지 않았습니다. regime 기능을 비활성화합니다."
            )
            regime_enabled_actual = False
        else:
            if (
                "date" not in market_regime.columns
                or "regime" not in market_regime.columns
            ):
                warns.append(
                    "[Stage5] market_regime에 'date' 또는 'regime' 컬럼이 없습니다. regime 기능을 비활성화합니다."
                )
                regime_enabled_actual = False
            else:
                regime_df = market_regime[["date", "regime"]].copy()
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                regime_dict = dict(zip(regime_df["date"], regime_df["regime"]))
                regime_enabled_actual = True
                warns.append(
                    f"[Stage5] 시장 국면 데이터 로드 완료: {len(regime_dict)}개 날짜"
                )
    else:
        regime_enabled_actual = False

    # 데이터 전처리
    for c in [date_col, ticker_col, phase_col]:
        if c not in rebalance_scores.columns:
            raise KeyError(f"rebalance_scores missing required column: {c}")

    score_col = _pick_score_col(rebalance_scores, cfg.score_col)
    ret_col = _pick_ret_col(rebalance_scores, cfg.ret_col, cfg.holding_days)

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

    # 일별 백테스트용 데이터 구조
    positions_rows: list[dict] = []
    daily_returns_rows: list[dict] = []
    selection_diagnostics_rows: list[dict] = []
    runtime_profile_rows: list[dict] = []

    # Phase별 처리
    for phase, dphase in df_sorted.groupby(phase_col, sort=False):
        t_phase_start = time.time()
        rebalance_dates_all = sorted(dphase[date_col].unique())

        # 리밸런싱 날짜 필터링 (기존 로직과 동일)
        rebalance_interval = int(getattr(cfg, "rebalance_interval", 1))
        if rebalance_interval > 1:
            rebalance_dates_filtered = rebalance_dates_all[::rebalance_interval]
            warns.append(
                f"[Phase 7 Step 2] rebalance_interval={rebalance_interval} 적용: {len(rebalance_dates_filtered)}/{len(rebalance_dates_all)}개 날짜 사용"
            )
        else:
            rebalance_dates_filtered = rebalance_dates_all

        # 리밸런싱 시점별 포지션 저장용
        positions_at_rebalance: dict[pd.Timestamp, dict[str, float]] = {}

        logger.info(
            f"[L7 일별 백테스트] Phase '{phase}' 시작 (리밸런싱 {len(rebalance_dates_filtered)}개)"
        )

        for dt, g in dphase.groupby(date_col, sort=True):
            dt_ts = pd.to_datetime(dt)

            # 리밸런싱 날짜가 아니면 스킵 (기존 로직 유지)
            if dt_ts not in rebalance_dates_filtered:
                continue

            g = g.sort_values(
                [score_col, ticker_col], ascending=[False, True]
            ).reset_index(drop=True)

            # [Stage5] 시장 국면에 따른 top_k 및 exposure 결정
            current_regime = None
            current_top_k = int(cfg.top_k)
            current_exposure = 1.0

            if regime_enabled_actual:
                regime_dates = sorted([d for d in regime_dict.keys() if d <= dt_ts])
                if len(regime_dates) > 0:
                    nearest_date = regime_dates[-1]
                    current_regime = regime_dict[nearest_date]

            # 종목 선택 (기존 로직과 동일)
            g_sel, diagnostics = select_topk_with_fallback(
                g,
                ticker_col=ticker_col,
                score_col=score_col,
                top_k=current_top_k,
                buffer_k=int(cfg.buffer_k),
                prev_holdings=[],  # 일별 백테스트에서는 이전 포지션 추적하지 않음
                group_col=cfg.group_col if cfg.diversify_enabled else None,
                max_names_per_group=(
                    cfg.max_names_per_group if cfg.diversify_enabled else None
                ),
                required_cols=[ret_col],
                filter_missing_price=True,
                filter_suspended=False,
                smart_buffer_enabled=cfg.smart_buffer_enabled,
                smart_buffer_stability_threshold=cfg.smart_buffer_stability_threshold,
            )

            g_sel = g_sel.sort_values(
                [score_col, ticker_col], ascending=[False, True]
            ).reset_index(drop=True)

            scores = g_sel[score_col]
            w = _weights_from_scores(scores, cfg.weighting, cfg.softmax_temp)

            # 리밸런싱 시점 포지션 저장
            new_w = {
                t: float(wi) for t, wi in zip(g_sel[ticker_col].tolist(), w.tolist())
            }
            positions_at_rebalance[dt_ts] = new_w

            # 포지션 데이터 저장 (기존 형식 유지)
            sector_col = (
                cfg.group_col
                if cfg.diversify_enabled and cfg.group_col in g_sel.columns
                else None
            )
            k_eff = diagnostics["selected_count"]
            eligible_count = diagnostics["eligible_count"]
            filled_count = diagnostics["filled_from_next_rank"]

            for idx, (t, wi, sc, tr) in enumerate(
                zip(g_sel[ticker_col], w, g_sel[score_col], g_sel[ret_col])
            ):
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
                    "k_eff": int(k_eff),
                    "eligible_count": int(eligible_count),
                    "filled_count": int(filled_count),
                }
                if sector_col and sector_col in g_sel.columns:
                    row_sector = (
                        g_sel.iloc[idx][sector_col] if idx < len(g_sel) else None
                    )
                    pos_row[sector_col] = (
                        str(row_sector) if pd.notna(row_sector) else None
                    )
                positions_rows.append(pos_row)

        # Phase 내 모든 리밸런싱 처리 완료 후 일별 포트 수익률 계산
        if positions_at_rebalance:
            # 해당 phase의 일별 가격 데이터 추출
            phase_date_range = daily_prices[
                (daily_prices["date"] >= min(positions_at_rebalance.keys()))
                & (daily_prices["date"] <= max(positions_at_rebalance.keys()))
            ].copy()

            if len(phase_date_range) > 0:
                # 일별 포트 수익률 계산
                daily_portfolio_returns = _calculate_daily_portfolio_returns(
                    rebalance_dates=list(positions_at_rebalance.keys()),
                    daily_prices=phase_date_range,
                    positions_at_rebalance=positions_at_rebalance,
                    cost_bps=float(cfg.cost_bps),
                    slippage_bps=float(getattr(cfg, "slippage_bps", 0.0)),
                    volatility_adjustment_enabled=cfg.volatility_adjustment_enabled,
                    volatility_lookback_days=cfg.volatility_lookback_days,
                    target_volatility=cfg.target_volatility,
                    volatility_adjustment_max=cfg.volatility_adjustment_max,
                    volatility_adjustment_min=cfg.volatility_adjustment_min,
                    risk_scaling_enabled=cfg.risk_scaling_enabled,
                    risk_scaling_bear_multiplier=cfg.risk_scaling_bear_multiplier,
                    risk_scaling_neutral_multiplier=cfg.risk_scaling_neutral_multiplier,
                    risk_scaling_bull_multiplier=cfg.risk_scaling_bull_multiplier,
                    market_regime=market_regime,
                )

                # 일별 결과를 returns_rows에 추가 (호환성을 위해)
                for _, row in daily_portfolio_returns.iterrows():
                    daily_returns_rows.append(
                        {
                            "date": row["date"],
                            "phase": phase,
                            "top_k": int(cfg.top_k),
                            "holding_days": int(cfg.holding_days),
                            "cost_bps": float(cfg.cost_bps),
                            "cost_bps_used": float(cost_bps_used),
                            "weighting": cfg.weighting,
                            "buffer_k": int(cfg.buffer_k),
                            "n_tickers": row["n_positions"],
                            "gross_return": row["portfolio_return"]
                            + row["trading_cost"],  # 총 수익률 + 비용 = 총 수익률
                            "net_return": row[
                                "portfolio_return"
                            ],  # 비용 차감 후 수익률
                            "turnover_oneway": 0.0,  # 일별 백테스트에서는 턴오버 개념 없음
                            "daily_trading_cost": row["trading_cost"],
                            "turnover_cost": 0.0,
                            "slippage_cost": 0.0,
                            "traded_value": 0.0,
                            "total_cost": row["trading_cost"],
                            "is_rebalance_date": row["is_rebalance_date"],
                            "exposure_multiplier": row["exposure_multiplier"],
                            "regime_multiplier": row["regime_multiplier"],
                            "final_exposure": row["final_exposure"],
                        }
                    )

                warns.append(
                    f"[일별 백테스트] Phase '{phase}': {len(daily_portfolio_returns)}일 처리 완료"
                )

        t_phase_end = time.time()
        phase_time = t_phase_end - t_phase_start
        logger.info(
            f"[L7 일별 백테스트] Phase '{phase}' 완료: {len(rebalance_dates_filtered)}개 리밸런싱, 총 {phase_time:.1f}초"
        )

    # 결과 데이터프레임 생성
    bt_positions = (
        pd.DataFrame(positions_rows)
        .sort_values(["phase", "date", "ticker"])
        .reset_index(drop=True)
    )
    bt_returns = (
        pd.DataFrame(daily_returns_rows)
        .sort_values(["phase", "date"])
        .reset_index(drop=True)
    )

    # 월별 누적수익률 DataFrame (별도 반환용)
    bt_monthly_returns = pd.DataFrame()
    if daily_returns_rows:
        daily_df = pd.DataFrame(daily_returns_rows)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        bt_monthly_returns = _convert_daily_to_monthly_returns(daily_df)

    # equity curve (기존 호환성 유지)
    eq_rows: list[dict] = []
    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        eq = 1.0
        for dt, r in zip(g["date"], g["net_return"]):
            eq *= 1.0 + float(r)
            eq_rows.append(
                {"date": dt, "phase": phase, "equity": float(eq), "drawdown": 0.0}
            )  # DD 계산 생략
    bt_equity_curve = (
        pd.DataFrame(eq_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    )

    # metrics 계산 (간소화)
    met_rows: list[dict] = []
    periods_per_year = 252.0  # 일별 데이터 기준

    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        r_net = g["net_return"].astype(float).to_numpy()

        eq_n = float((1.0 + pd.Series(r_net)).cumprod().iloc[-1]) if len(r_net) else 1.0
        years = len(g) / periods_per_year if len(g) > 0 else 0

        if eq_n <= 0:
            net_cagr = -1.0
        elif years > 0:
            net_cagr = eq_n ** (1.0 / years) - 1.0
        else:
            net_cagr = 0.0

        met_rows.append(
            {
                "phase": phase,
                "net_cagr": net_cagr,
                "net_total_return": eq_n - 1.0,
                "n_rebalances": (
                    len(g[g["is_rebalance_date"] == True])
                    if "is_rebalance_date" in g.columns
                    else len(g)
                ),
                "date_start": g["date"].min() if len(g) > 0 else None,
                "date_end": g["date"].max() if len(g) > 0 else None,
            }
        )

    bt_metrics = pd.DataFrame(met_rows)

    quality = {
        "backtest": {
            "daily_backtest_enabled": True,
            "holding_days": int(cfg.holding_days),
            "top_k": int(cfg.top_k),
            "cost_bps": float(cfg.cost_bps),
            "buffer_k": int(cfg.buffer_k),
            "score_col_used": score_col,
            "ret_col_used": ret_col,
            "weighting": cfg.weighting,
            "rows_positions": int(len(bt_positions)),
            "rows_returns": int(len(bt_returns)),
            "monthly_returns": (
                bt_monthly_returns.to_dict("records")
                if len(bt_monthly_returns) > 0
                else []
            ),
        }
    }

    # 호환성을 위해 기존 인터페이스 유지 (6개 값 반환)
    # 월별 누적수익률은 quality에 포함
    return bt_positions, bt_returns, bt_equity_curve, bt_metrics, quality, warns


def run_backtest(
    rebalance_scores: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    phase_col: str = "phase",
    config_cost_bps: Optional[
        float
    ] = None,  # [Stage1] config.yaml에서 읽은 원본 cost_bps 값
    market_regime: Optional[
        pd.DataFrame
    ] = None,  # [Stage5] 시장 국면 DataFrame (date, regime 컬럼)
    daily_prices: Optional[
        pd.DataFrame
    ] = None,  # [개선안: 일별 백테스트] 일별 가격 데이터 (date, ticker, close)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    warns: list[str] = []

    # [개선안: 일별 mark-to-market 백테스트] 옵션 활성화 시 별도 로직 사용
    if cfg.daily_backtest_enabled:
        if daily_prices is None:
            raise ValueError("[일별 백테스트] daily_prices 데이터가 필요합니다")
        return _run_daily_backtest(
            rebalance_scores=rebalance_scores,
            daily_prices=daily_prices,
            cfg=cfg,
            date_col=date_col,
            ticker_col=ticker_col,
            phase_col=phase_col,
            config_cost_bps=config_cost_bps,
            market_regime=market_regime,
        )

    # [동적 기간 파라미터 적용] holding_days에 따라 config.yaml에서 동적 파라미터 로드 및 적용
    # 전략 이름 추측 (score_col 기반)
    strategy_name = None
    if cfg.score_col == "score_total_short":
        strategy_name = "bt20_short"
    elif cfg.score_col == "score_total_long":
        strategy_name = "bt120_long"
    elif cfg.score_col == "score_ens":
        strategy_name = "bt20_ens"

    cfg = _apply_dynamic_params_for_holding_days(cfg, strategy_name)

    # [Stage1] config_cost_bps가 없으면 cfg.cost_bps를 사용 (하위 호환성)
    cost_bps_config = (
        float(config_cost_bps) if config_cost_bps is not None else float(cfg.cost_bps)
    )
    cost_bps_used = float(cfg.cost_bps)  # 실제 백테스트에 사용된 cost_bps
    cost_bps_mismatch_flag = bool(
        abs(cost_bps_used - cost_bps_config) > 1e-6
    )  # 부동소수점 오차 고려

    if cost_bps_mismatch_flag:
        warns.append(
            f"[Stage1] cost_bps mismatch: config={cost_bps_config}, used={cost_bps_used}"
        )

    # [Stage5] 시장 국면(regime) 처리
    regime_dict: dict[pd.Timestamp, str] = {}
    if cfg.regime_enabled:
        if market_regime is None or len(market_regime) == 0:
            warns.append(
                "[Stage5] regime_enabled=True이지만 market_regime이 제공되지 않았습니다. regime 기능을 비활성화합니다."
            )
            regime_enabled_actual = False
        else:
            # market_regime에서 date, regime 컬럼 추출
            if (
                "date" not in market_regime.columns
                or "regime" not in market_regime.columns
            ):
                warns.append(
                    "[Stage5] market_regime에 'date' 또는 'regime' 컬럼이 없습니다. regime 기능을 비활성화합니다."
                )
                regime_enabled_actual = False
            else:
                regime_df = market_regime[["date", "regime"]].copy()
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                regime_dict = dict(zip(regime_df["date"], regime_df["regime"]))
                regime_enabled_actual = True
                logger.info(
                    f"[Stage5] 시장 국면 데이터 로드 완료: {len(regime_dict)}개 날짜"
                )
    else:
        regime_enabled_actual = False

    for c in [date_col, ticker_col, phase_col]:
        if c not in rebalance_scores.columns:
            raise KeyError(f"rebalance_scores missing required column: {c}")

    score_col = _pick_score_col(rebalance_scores, cfg.score_col)
    ret_col = _pick_ret_col(rebalance_scores, cfg.ret_col, cfg.holding_days)
    print(
        f"[DEBUG] Strategy {cfg.score_col}: holding_days={cfg.holding_days}, selected ret_col={ret_col}"
    )

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

    positions_rows: list[dict] = []
    returns_rows: list[dict] = []
    # [Stage13] selection diagnostics 저장용
    selection_diagnostics_rows: list[dict] = []

    # [Stage13] 런타임 프로파일 저장용
    runtime_profile_rows: list[dict] = []
    t_load_inputs = time.time()
    logger.info(f"[L7 Runtime] 입력 로딩 완료: {time.time() - t_load_inputs:.3f}초")

    for phase, dphase in df_sorted.groupby(phase_col, sort=False):
        t_phase_start = time.time()
        rebalance_dates_all = sorted(dphase[date_col].unique())

        # [Phase 7 Step 2] Turnover 제어: 리밸런싱 주기 완화
        # [rebalance_interval 개선] L6R에서 이미 필터링했으므로, L7에서는 rebalance_interval=1로 고정
        # (L6R에서 rebalance_interval > 1일 때 이미 일별 데이터에서 필터링했으므로, L7에서는 추가 필터링 불필요)
        # [bt20 프로페셔널] 적응형 리밸런싱의 경우 L7에서 직접 필터링 적용
        rebalance_interval = (
            int(cfg.rebalance_interval) if hasattr(cfg, "rebalance_interval") else 1
        )

        # [bt20 프로페셔널] 적응형 리밸런싱 적용
        if cfg.adaptive_rebalancing_enabled:
            logger.info("[bt20 프로페셔널] 적응형 리밸런싱 적용 시작")

            # 적응형 리밸런싱 설정 구성
            adaptive_config = {
                "min_rebalance_days": (
                    cfg.rebalance_intervals.get("strong", 15)
                    if cfg.rebalance_intervals
                    else 15
                ),
                "max_rebalance_days": (
                    cfg.rebalance_intervals.get("weak", 25)
                    if cfg.rebalance_intervals
                    else 25
                ),
                "default_rebalance_days": 20,
                "signal_strength_threshold_high": (
                    cfg.signal_strength_thresholds.get("strong", 0.8)
                    if cfg.signal_strength_thresholds
                    else 0.8
                ),
                "signal_strength_threshold_low": (
                    cfg.signal_strength_thresholds.get("weak", 0.6)
                    if cfg.signal_strength_thresholds
                    else 0.6
                ),
                "ic_lookback_window": 60,
                "ic_min_periods": 20,
            }

            # 적응형 리밸런서 초기화
            adaptive_rebalancer = AdaptiveRebalancing(adaptive_config)

            # 적응형 리밸런싱 간격 계산
            intervals_df = adaptive_rebalancer.get_adaptive_rebalance_intervals(dphase)

            if not intervals_df.empty:
                # 적응형 간격에 따른 날짜 필터링
                filtered_dates = adaptive_rebalancer.filter_adaptive_dates(
                    rebalance_dates_all, intervals_df
                )

                rebalance_dates_filtered = sorted(filtered_dates)

                logger.info(
                    f"[bt20 프로페셔널] 적응형 리밸런싱 적용 완료: {len(rebalance_dates_filtered)}/{len(rebalance_dates_all)}개 날짜 선택"
                )
                logger.info(
                    f"[bt20 프로페셔널] 평균 리밸런싱 간격: {intervals_df['rebalance_interval'].mean():.1f}일"
                )

                # 간격 분포 로깅
                interval_counts = intervals_df["rebalance_interval"].value_counts()
                logger.info(f"[bt20 프로페셔널] 간격 분포: {interval_counts.to_dict()}")
            else:
                logger.warning(
                    "[bt20 프로페셔널] 적응형 간격 계산 실패, 기본 로직 사용"
                )
                rebalance_dates_filtered = rebalance_dates_all

        # 기존 로직 (모든 전략에서 동일하게 적용)
        # [bt20 프로페셔널] 적응형 리밸런싱 적용 (L6R 필터링 우회)
        if cfg.adaptive_rebalancing_enabled:
            # bt20_pro의 경우 L6R 필터링을 우회하고 L7에서 직접 모든 날짜 사용
            rebalance_dates_filtered = rebalance_dates_all
            logger.info(
                f"[bt20 프로페셔널] L6R 필터링 우회, 모든 날짜 사용: {len(rebalance_dates_all)}개 날짜"
            )

        # [rebalance_interval 개선] L6R에서 이미 필터링된 rebalance_scores를 사용하므로,
        # L7에서는 rebalance_interval 필터링을 건너뛰고 모든 날짜 사용
        # 단, rebalance_interval=1인 경우에만 L7에서 필터링 적용 (기존 동작 유지)
        elif rebalance_interval > 1:
            # L6R에서 이미 필터링되었으므로 L7에서는 추가 필터링 건너뛰기
            rebalance_dates_filtered = rebalance_dates_all
            logger.info(
                f"[Phase 7 Step 2] L6R에서 이미 rebalance_interval={rebalance_interval} 필터링 완료, L7에서는 추가 필터링 건너뛰기: {len(rebalance_dates_all)}개 날짜 사용"
            )
        elif rebalance_interval == 1:
            # 기존 로직: L7에서 필터링 (rebalance_interval=1이면 필터링 없음)
            rebalance_dates_filtered = rebalance_dates_all
        else:
            # 예외 처리
            rebalance_dates_filtered = rebalance_dates_all

        logger.info(
            f"[L7 Runtime] Phase '{phase}' 시작 (리밸런싱 {len(rebalance_dates_filtered)}개)"
        )
        prev_w: dict[str, float] = {}
        prev_holdings: list[str] = []

        rebalance_dates = sorted(dphase[date_col].unique())
        rebalance_count = 0

        # [Phase 8 Step 2 방안1] 변동성 기반 exposure 조정을 위한 수익률 히스토리 추적
        recent_returns_history: list[float] = []  # 최근 수익률 저장

        # [개선안 36번] 오버래핑 트랜치 모드: 트랜치 리스트를 유지
        tranches: list[
            dict
        ] = []  # each: {"open_date": pd.Timestamp, "w": Dict[str,float], "top_k": int}
        tranche_max_active = max(int(getattr(cfg, "tranche_max_active", 4)), 1)
        tranche_holding_days = max(int(getattr(cfg, "tranche_holding_days", 120)), 1)
        tranche_alloc_mode = (
            str(getattr(cfg, "tranche_allocation_mode", "fixed_equal")).strip().lower()
        )

        for dt, g in dphase.groupby(date_col, sort=True):
            t_rebalance_start = time.time()
            rebalance_count += 1
            g = g.sort_values(
                [score_col, ticker_col], ascending=[False, True]
            ).reset_index(drop=True)

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
                    warns.append(
                        f"[Stage5] {dt_ts.strftime('%Y-%m-%d')}: regime 데이터 없음, 기본값 사용"
                    )

            # [개선안 36번] 오버래핑 트랜치 모드:
            # - 매 리밸런싱 날짜(월별)마다 신규 트랜치 1개를 추가
            # - tranche_holding_days(캘린더 day) 지나면 트랜치 만기 제거
            # - 포트폴리오 가중치는 트랜치들의 평균(1/4 고정 배분)으로 계산
            if cfg.overlapping_tranches_enabled:
                dt_ts = pd.to_datetime(dt)
                # 만기 제거
                tranches = [
                    t
                    for t in tranches
                    if int((dt_ts - pd.to_datetime(t["open_date"])).days)
                    < tranche_holding_days
                ]
                # 신규 트랜치 생성(이전 트랜치 holdings는 유지되므로, 신규는 fresh selection)
                g_new_sel, diagnostics = select_topk_with_fallback(
                    g,
                    ticker_col=ticker_col,
                    score_col=score_col,
                    top_k=current_top_k,
                    buffer_k=0,  # 신규 트랜치는 버퍼링 없이 fresh 구성
                    prev_holdings=[],
                    group_col=cfg.group_col if cfg.diversify_enabled else None,
                    max_names_per_group=(
                        cfg.max_names_per_group if cfg.diversify_enabled else None
                    ),
                    required_cols=[ret_col],
                    filter_missing_price=True,
                    filter_suspended=False,
                    smart_buffer_enabled=False,
                    smart_buffer_stability_threshold=cfg.smart_buffer_stability_threshold,
                )
                g_new_sel = g_new_sel.sort_values(
                    [score_col, ticker_col], ascending=[False, True]
                ).reset_index(drop=True)
                w_new = _weights_from_scores(
                    g_new_sel[score_col], cfg.weighting, cfg.softmax_temp
                )
                tranche_w = {
                    t: float(wi)
                    for t, wi in zip(g_new_sel[ticker_col].tolist(), w_new.tolist())
                }
                tranches.append(
                    {"open_date": dt_ts, "w": tranche_w, "top_k": int(current_top_k)}
                )
                # 동시 보유 트랜치 수 제한(가장 오래된 것부터 제거)
                if len(tranches) > tranche_max_active:
                    tranches = sorted(
                        tranches, key=lambda x: pd.to_datetime(x["open_date"])
                    )[-tranche_max_active:]

                # 트랜치 가중치 집계(포트폴리오)
                if tranche_alloc_mode == "active_equal":
                    denom = float(max(len(tranches), 1))
                else:
                    # fixed_equal: 항상 1/N 배분(초기에는 일부 현금)
                    denom = float(tranche_max_active)
                agg_w: dict[str, float] = {}
                for t in tranches:
                    for k, v in (t.get("w") or {}).items():
                        agg_w[k] = agg_w.get(k, 0.0) + float(v) / denom

                # agg_w를 g 기준으로 slice
                tickers_today = set(g[ticker_col].astype(str).tolist())
                agg_w = {k: v for k, v in agg_w.items() if k in tickers_today}

                # selection diagnostics는 "신규 트랜치" 기준으로 기록(왜 K_eff가 떨어졌는지 추적 목적)
                g_sel = g_new_sel
                # portfolio weights array(선택된 종목 기준이 아닌, 전체 agg_w 기준)
                # -> 수익률/alpha 계산은 agg_w로 진행
                prev_holdings = g_sel[ticker_col].tolist()
                new_w = agg_w
                turnover_oneway = _compute_turnover_oneway(prev_w, new_w)

                # alpha quality는 후보 풀(g) 기준
                g_alpha = g.dropna(subset=[ret_col]).copy()
                ic = (
                    _safe_corr(g_alpha[score_col], g_alpha[ret_col])
                    if len(g_alpha)
                    else float("nan")
                )
                ric = (
                    _rank_ic(g_alpha[score_col], g_alpha[ret_col])
                    if len(g_alpha)
                    else float("nan")
                )
                ls_alpha = (
                    _long_short_alpha(
                        g_alpha[score_col], g_alpha[ret_col], k=int(current_top_k)
                    )
                    if len(g_alpha)
                    else float("nan")
                )

                # 포트폴리오 기간 수익률: agg_w * ret_col (월별 20일 fwd)
                g_ret_map = g[[ticker_col, ret_col]].copy()
                g_ret_map[ticker_col] = g_ret_map[ticker_col].astype(str)
                g_ret_map[ret_col] = pd.to_numeric(g_ret_map[ret_col], errors="coerce")
                ret_dict = dict(zip(g_ret_map[ticker_col], g_ret_map[ret_col]))
                gross_ret = float(
                    sum(
                        float(wi) * float(ret_dict.get(tk, 0.0) or 0.0)
                        for tk, wi in new_w.items()
                    )
                )

                # 변동성/리스크 스케일링은 포트폴리오 레벨에 적용
                volatility_adjustment = 1.0
                if (
                    cfg.volatility_adjustment_enabled
                    and len(recent_returns_history) >= 2
                ):
                    recent_returns_array = np.array(
                        recent_returns_history[-cfg.volatility_lookback_days :]
                    )
                    volatility_adjustment = _calculate_volatility_adjustment(
                        recent_returns_array,
                        target_vol=cfg.target_volatility,
                        lookback_days=cfg.volatility_lookback_days,
                        max_mult=cfg.volatility_adjustment_max,
                        min_mult=cfg.volatility_adjustment_min,
                    )
                adjusted_exposure = _apply_risk_scaling(
                    base_exposure=current_exposure,
                    regime=current_regime,
                    risk_scaling_enabled=cfg.risk_scaling_enabled,
                    bear_multiplier=cfg.risk_scaling_bear_multiplier,
                    neutral_multiplier=cfg.risk_scaling_neutral_multiplier,
                    bull_multiplier=cfg.risk_scaling_bull_multiplier,
                )
                final_exposure = adjusted_exposure * volatility_adjustment
                gross_ret = gross_ret * final_exposure
                if cfg.volatility_adjustment_enabled:
                    recent_returns_history.append(gross_ret)
                    max_history = cfg.volatility_lookback_days * 2
                    if len(recent_returns_history) > max_history:
                        recent_returns_history = recent_returns_history[-max_history:]

                cost_breakdown = _calculate_trading_cost(
                    turnover_oneway=float(turnover_oneway),
                    cost_bps=float(cfg.cost_bps),
                    slippage_bps=float(getattr(cfg, "slippage_bps", 0.0)),
                    exposure=float(final_exposure),
                )
                traded_value = float(cost_breakdown["traded_value"])
                turnover_cost = float(cost_breakdown["cost_component"])
                slippage_cost = float(cost_breakdown["slippage_component"])
                total_cost = float(cost_breakdown["total_cost"])
                daily_trading_cost = total_cost
                net_ret = gross_ret - total_cost

                # positions_rows: agg_w를 기록 (가시성 확보)
                k_eff = diagnostics["selected_count"]
                eligible_count = diagnostics["eligible_count"]
                filled_count = diagnostics["filled_from_next_rank"]

                # [Stage13] selection_diagnostics 기록 (트랜치 모드에서도 유지)
                selection_diagnostics_rows.append(
                    {
                        "date": dt,
                        "phase": phase,
                        "top_k": int(current_top_k),
                        "eligible_count": diagnostics.get("eligible_count"),
                        "selected_count": diagnostics.get("selected_count"),
                        "dropped_missing": diagnostics.get("dropped_missing"),
                        "dropped_filter": diagnostics.get("dropped_filter"),
                        "dropped_sectorcap": diagnostics.get("dropped_sectorcap"),
                        "filled_from_next_rank": diagnostics.get(
                            "filled_from_next_rank"
                        ),
                        "drop_reasons": (
                            str(diagnostics.get("drop_reasons"))
                            if diagnostics.get("drop_reasons")
                            else None
                        ),
                        "mode": "overlapping_tranches",
                    }
                )

                # [Stage13] runtime_profile 기록
                runtime_profile_rows.append(
                    {
                        "date": dt,
                        "phase": phase,
                        "rebalance_time_sec": float(time.time() - t_rebalance_start),
                        "k_eff": int(k_eff),
                        "top_k_used": int(current_top_k),
                        "holding_days": int(cfg.holding_days),
                        "eligible_count": int(eligible_count),
                        "mode": "overlapping_tranches",
                    }
                )
                for tk, wi in sorted(new_w.items()):
                    positions_rows.append(
                        {
                            "date": dt,
                            "phase": phase,
                            "ticker": str(tk),
                            "weight": float(wi),
                            "score": np.nan,
                            "ret_realized": float(ret_dict.get(tk, np.nan)),
                            "top_k": int(current_top_k),
                            "holding_days": int(cfg.holding_days),
                            "cost_bps": float(cfg.cost_bps),
                            "weighting": cfg.weighting,
                            "buffer_k": int(cfg.buffer_k),
                            "k_eff": int(k_eff),
                            "eligible_count": int(eligible_count),
                            "filled_count": int(filled_count),
                        }
                    )

                returns_row = {
                    "date": dt,
                    "phase": phase,
                    "top_k": int(current_top_k),
                    "holding_days": int(cfg.holding_days),
                    "cost_bps": float(cfg.cost_bps),
                    "cost_bps_used": float(cost_bps_used),
                    "weighting": cfg.weighting,
                    "buffer_k": int(cfg.buffer_k),
                    "n_tickers": int(len(new_w)),
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                    "turnover_oneway": float(turnover_oneway),
                    "ic": float(ic) if pd.notna(ic) else np.nan,
                    "rank_ic": float(ric) if pd.notna(ric) else np.nan,
                    "long_short_alpha": (
                        float(ls_alpha) if pd.notna(ls_alpha) else np.nan
                    ),
                    "daily_trading_cost": float(daily_trading_cost),
                    "turnover_cost": float(turnover_cost),
                    "slippage_cost": float(slippage_cost),
                    "traded_value": float(traded_value),
                    "total_cost": float(total_cost),
                    # 트랜치 상태
                    "tranche_active": int(len(tranches)),
                    "tranche_holding_days": int(tranche_holding_days),
                    "tranche_max_active": int(tranche_max_active),
                    "tranche_allocation_mode": str(tranche_alloc_mode),
                }
                if regime_enabled_actual:
                    returns_row["regime"] = (
                        str(current_regime) if current_regime else None
                    )
                    returns_row["exposure"] = float(current_exposure)
                if cfg.volatility_adjustment_enabled:
                    returns_row["volatility_adjustment"] = float(volatility_adjustment)
                if cfg.risk_scaling_enabled or cfg.volatility_adjustment_enabled:
                    returns_row["final_exposure"] = float(final_exposure)
                returns_rows.append(returns_row)

                prev_w = new_w
                continue  # next date

            # -----------------------
            # (기존) 단일 포트폴리오 모드
            # -----------------------
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
                max_names_per_group=(
                    cfg.max_names_per_group if cfg.diversify_enabled else None
                ),
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
            runtime_profile_rows.append(
                {
                    "date": dt,
                    "phase": phase,
                    "rebalance_time_sec": rebalance_time,
                    "k_eff": diagnostics["selected_count"],
                    "top_k_used": current_top_k,
                    "holding_days": int(cfg.holding_days),
                    "eligible_count": diagnostics["eligible_count"],
                }
            )

            # 매 10회마다 요약 로그
            if rebalance_count % 10 == 0:
                elapsed_phase = t_rebalance_end - t_phase_start
                avg_time = elapsed_phase / rebalance_count
                logger.info(
                    f"[L7 Runtime] Phase '{phase}': {rebalance_count}/{len(rebalance_dates)} 완료, 평균 {avg_time:.3f}초/리밸런싱, 누적 {elapsed_phase:.1f}초"
                )

            g_sel = g_sel.sort_values(
                [score_col, ticker_col], ascending=[False, True]
            ).reset_index(drop=True)

            scores = g_sel[score_col]
            w = _weights_from_scores(scores, cfg.weighting, cfg.softmax_temp)

            new_w = {
                t: float(wi) for t, wi in zip(g_sel[ticker_col].tolist(), w.tolist())
            }
            turnover_oneway = _compute_turnover_oneway(prev_w, new_w)

            # [개선안 34번] Alpha Quality (IC/Rank IC/Long-Short Alpha)
            # - selection 이전(후보 풀) 기준으로 계산: score가 미래수익을 얼마나 설명하는지
            g_alpha = g.dropna(subset=[ret_col]).copy()
            ic = (
                _safe_corr(g_alpha[score_col], g_alpha[ret_col])
                if len(g_alpha)
                else float("nan")
            )
            ric = (
                _rank_ic(g_alpha[score_col], g_alpha[ret_col])
                if len(g_alpha)
                else float("nan")
            )
            ls_alpha = (
                _long_short_alpha(
                    g_alpha[score_col], g_alpha[ret_col], k=int(current_top_k)
                )
                if len(g_alpha)
                else float("nan")
            )

            gross_ret = float(np.dot(w, g_sel[ret_col].astype(float).to_numpy()))

            # [Phase 8 Step 2 방안1] 변동성 기반 exposure 조정
            volatility_adjustment = 1.0
            if cfg.volatility_adjustment_enabled and len(recent_returns_history) >= 2:
                recent_returns_array = np.array(
                    recent_returns_history[-cfg.volatility_lookback_days :]
                )
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

            # [개선안 1번][개선안 3번] 거래비용/슬리피지 반영 (턴오버 기반)
            # - traded_value = turnover_oneway * |final_exposure|
            # - total_cost = traded_value * (cost_bps + slippage_bps) / 10000
            cost_breakdown = _calculate_trading_cost(
                turnover_oneway=float(turnover_oneway),
                cost_bps=float(cfg.cost_bps),
                slippage_bps=float(getattr(cfg, "slippage_bps", 0.0)),
                exposure=float(final_exposure),
            )
            traded_value = float(cost_breakdown["traded_value"])
            turnover_cost = float(cost_breakdown["cost_component"])
            slippage_cost = float(cost_breakdown["slippage_component"])
            total_cost = float(cost_breakdown["total_cost"])
            daily_trading_cost = total_cost  # 하위 호환 컬럼명 유지

            # PnL에서 비용 차감
            net_ret = gross_ret - total_cost

            # [Stage 4] sector_name을 positions에 포함
            sector_col = (
                cfg.group_col
                if cfg.diversify_enabled and cfg.group_col in g_sel.columns
                else None
            )

            # [Stage13] k_eff, eligible_count, filled_count 계산
            k_eff = diagnostics["selected_count"]
            eligible_count = diagnostics["eligible_count"]
            filled_count = diagnostics["filled_from_next_rank"]

            for idx, (t, wi, sc, tr) in enumerate(
                zip(g_sel[ticker_col], w, g_sel[score_col], g_sel[ret_col])
            ):
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
                    row_sector = (
                        g_sel.iloc[idx][sector_col] if idx < len(g_sel) else None
                    )
                    pos_row[sector_col] = (
                        str(row_sector) if pd.notna(row_sector) else None
                    )
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
                # [개선안 34번] Alpha Quality
                "ic": float(ic) if pd.notna(ic) else np.nan,
                "rank_ic": float(ric) if pd.notna(ric) else np.nan,
                "long_short_alpha": float(ls_alpha) if pd.notna(ls_alpha) else np.nan,
                "daily_trading_cost": float(
                    daily_trading_cost
                ),  # [개선안 1번][개선안 3번] 실제 차감된 총 비용(하위호환)
                "turnover_cost": float(
                    turnover_cost
                ),  # [개선안 1번] 수수료/세금/스프레드 비용(턴오버 기반)
                "slippage_cost": float(
                    slippage_cost
                ),  # [개선안 3번] 슬리피지 비용(턴오버 기반)
                "traded_value": float(
                    traded_value
                ),  # [개선안 1번] 거래된 비중(턴오버*노출)
                "total_cost": float(
                    total_cost
                ),  # [개선안 1번][개선안 3번] 실제 적용된 총 비용
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
        logger.info(
            f"[L7 Runtime] Phase '{phase}' 완료: {rebalance_count}개 리밸런싱, 총 {phase_time:.1f}초, 평균 {phase_time/rebalance_count:.3f}초/리밸런싱"
        )

    bt_positions = (
        pd.DataFrame(positions_rows)
        .sort_values(["phase", "date", "ticker"])
        .reset_index(drop=True)
    )
    bt_returns = (
        pd.DataFrame(returns_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    )

    # [Stage13] bt_returns를 core와 diagnostics로 분리
    # 필수 컬럼: date, phase, net_return, gross_return, turnover_oneway 등 (validation 통과용)
    REQUIRED_CORE_COLS = [
        "date",
        "phase",
        "top_k",
        "holding_days",
        "cost_bps",
        "cost_bps_used",
        "weighting",
        "buffer_k",
        "n_tickers",
        "gross_return",
        "net_return",
        "turnover_oneway",
        "daily_trading_cost",
        "turnover_cost",
        "total_cost",
        # [개선안 3번] 선택 컬럼(있으면 포함)
        "slippage_cost",
        "traded_value",
        # [개선안 34번] Alpha Quality
        "ic",
        "rank_ic",
        "long_short_alpha",
    ]

    # 진단 컬럼: regime, exposure 등 (결측 허용, 별도 저장)
    # [개선안 43번] 오버래핑 트랜치/리스크 스케일링 관련 컬럼이 Stage13 분리 과정에서 유실되지 않도록
    # diagnostics에 포함한다. (core는 최소 스키마 유지)
    DIAGNOSTIC_COLS = [
        "regime",
        "exposure",
        # [개선안 36번] 오버래핑 트랜치 상태
        "tranche_active",
        "tranche_holding_days",
        "tranche_max_active",
        "tranche_allocation_mode",
        "mode",
        # [Phase 8 Step 2] exposure 조정/리스크 스케일링 상태
        "volatility_adjustment",
        "final_exposure",
    ]

    # core: 필수 컬럼만 포함 (validation 통과용)
    available_core_cols = [c for c in REQUIRED_CORE_COLS if c in bt_returns.columns]
    bt_returns_core = bt_returns[available_core_cols].copy()

    # diagnostics: 진단 컬럼만 포함 (결측 허용)
    available_diag_cols = [c for c in DIAGNOSTIC_COLS if c in bt_returns.columns]
    if available_diag_cols:
        bt_returns_diagnostics = bt_returns[
            ["date", "phase"] + available_diag_cols
        ].copy()
    else:
        # 진단 컬럼이 없으면 빈 DataFrame 생성
        bt_returns_diagnostics = pd.DataFrame(columns=["date", "phase"])

    # [Stage13] selection_diagnostics DataFrame 생성
    # [Stage13] selection_diagnostics: empty-safe
    selection_diagnostics = pd.DataFrame(selection_diagnostics_rows)
    if (
        len(selection_diagnostics) > 0
        and "phase" in selection_diagnostics.columns
        and "date" in selection_diagnostics.columns
    ):
        selection_diagnostics = selection_diagnostics.sort_values(
            ["phase", "date"]
        ).reset_index(drop=True)
    else:
        selection_diagnostics = pd.DataFrame(columns=["date", "phase"])

    # equity curve
    eq_rows: list[dict] = []
    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        eq = 1.0
        peak = 1.0
        for dt, r in zip(g["date"], g["net_return"]):
            eq *= 1.0 + float(r)
            peak = max(peak, eq)
            dd = (eq / peak) - 1.0
            eq_rows.append(
                {"date": dt, "phase": phase, "equity": float(eq), "drawdown": float(dd)}
            )
    bt_equity_curve = (
        pd.DataFrame(eq_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    )

    # metrics
    met_rows: list[dict] = []
    periods_per_year = 252.0 / float(cfg.holding_days) if cfg.holding_days > 0 else 12.6

    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        r_gross = g["gross_return"].astype(float).to_numpy()
        r_net = g["net_return"].astype(float).to_numpy()

        # [개선안 34번] Alpha Quality aggregates
        ic_s = (
            pd.to_numeric(g.get("ic"), errors="coerce")
            if "ic" in g.columns
            else pd.Series([], dtype=float)
        )
        ric_s = (
            pd.to_numeric(g.get("rank_ic"), errors="coerce")
            if "rank_ic" in g.columns
            else pd.Series([], dtype=float)
        )
        ls_s = (
            pd.to_numeric(g.get("long_short_alpha"), errors="coerce")
            if "long_short_alpha" in g.columns
            else pd.Series([], dtype=float)
        )

        def _icir(x: pd.Series) -> float:
            x = x.dropna()
            if len(x) < 5:
                return float("nan")
            sd = float(x.std(ddof=1))
            if sd < 1e-12:
                return float("nan")
            return float((x.mean() / sd) * np.sqrt(periods_per_year))

        ic_mean = float(ic_s.dropna().mean()) if len(ic_s) else float("nan")
        ric_mean = float(ric_s.dropna().mean()) if len(ric_s) else float("nan")
        icir = _icir(ic_s) if len(ic_s) else float("nan")
        ricir = _icir(ric_s) if len(ric_s) else float("nan")
        ls_alpha_mean = float(ls_s.dropna().mean()) if len(ls_s) else float("nan")
        ls_alpha_ann = (
            float(ls_alpha_mean * periods_per_year)
            if pd.notna(ls_alpha_mean)
            else float("nan")
        )

        eq_g = (
            float((1.0 + pd.Series(r_gross)).cumprod().iloc[-1])
            if len(r_gross)
            else 1.0
        )
        eq_n = float((1.0 + pd.Series(r_net)).cumprod().iloc[-1]) if len(r_net) else 1.0

        d0 = pd.to_datetime(g["date"].iloc[0]) if len(g) else pd.NaT
        d1 = pd.to_datetime(g["date"].iloc[-1]) if len(g) else pd.NaT
        # [오류 수정] timedelta64 호환성: pd.Timedelta로 변환하여 .days 사용
        years = max(
            (
                (pd.Timedelta(d1 - d0).days / 365.25)
                if pd.notna(d0) and pd.notna(d1)
                else 1e-9
            ),
            1e-9,
        )

        # [오류 수정] 복소수 방지: eq_g가 음수이거나 0이면 -100% CAGR 처리
        # NaN 대신 -100%를 사용하여 최적화 알고리즘이 실패한 조합을 명확히 식별
        if eq_g <= 0:
            gross_cagr = -1.0  # -100% (완전 손실)
            if eq_g < 0:
                logger.warning(
                    f"Phase {phase}: Portfolio equity is negative (eq_g={eq_g:.4f}). Setting CAGR=-100%"
                )
        elif years > 0:
            try:
                gross_cagr_val = eq_g ** (1.0 / years) - 1.0
                # 복소수인 경우 실수부만 사용, 그래도 문제가 있으면 -100% 처리
                if isinstance(gross_cagr_val, complex):
                    gross_cagr = float(gross_cagr_val.real)
                    if np.isnan(gross_cagr) or np.isinf(gross_cagr):
                        gross_cagr = -1.0
                        logger.warning(
                            f"Phase {phase}: Complex CAGR resulted in invalid value. Setting CAGR=-100%"
                        )
                else:
                    gross_cagr = float(gross_cagr_val)
                    if np.isnan(gross_cagr) or np.isinf(gross_cagr):
                        gross_cagr = -1.0
            except (ValueError, OverflowError, TypeError) as e:
                gross_cagr = -1.0
                logger.warning(
                    f"Phase {phase}: CAGR calculation error ({type(e).__name__}: {e}). Setting CAGR=-100%"
                )
        else:
            gross_cagr = -1.0

        if eq_n <= 0:
            net_cagr = -1.0  # -100% (완전 손실)
            if eq_n < 0:
                logger.warning(
                    f"Phase {phase}: Portfolio net equity is negative (eq_n={eq_n:.4f}). Setting CAGR=-100%"
                )
        elif years > 0:
            try:
                net_cagr_val = eq_n ** (1.0 / years) - 1.0
                # 복소수인 경우 실수부만 사용, 그래도 문제가 있으면 -100% 처리
                if isinstance(net_cagr_val, complex):
                    net_cagr = float(net_cagr_val.real)
                    if np.isnan(net_cagr) or np.isinf(net_cagr):
                        net_cagr = -1.0
                        logger.warning(
                            f"Phase {phase}: Complex CAGR resulted in invalid value. Setting CAGR=-100%"
                        )
                else:
                    net_cagr = float(net_cagr_val)
                    if np.isnan(net_cagr) or np.isinf(net_cagr):
                        net_cagr = -1.0
            except (ValueError, OverflowError, TypeError) as e:
                net_cagr = -1.0
                logger.warning(
                    f"Phase {phase}: Net CAGR calculation error ({type(e).__name__}: {e}). Setting CAGR=-100%"
                )
        else:
            net_cagr = -1.0

        gross_vol = (
            float(np.std(r_gross, ddof=1) * np.sqrt(periods_per_year))
            if len(r_gross) > 1
            else 0.0
        )
        net_vol = (
            float(np.std(r_net, ddof=1) * np.sqrt(periods_per_year))
            if len(r_net) > 1
            else 0.0
        )

        gross_sharpe = (
            float(
                (np.mean(r_gross) / (np.std(r_gross, ddof=1) + 1e-12))
                * np.sqrt(periods_per_year)
            )
            if len(r_gross) > 1
            else 0.0
        )
        net_sharpe = (
            float(
                (np.mean(r_net) / (np.std(r_net, ddof=1) + 1e-12))
                * np.sqrt(periods_per_year)
            )
            if len(r_net) > 1
            else 0.0
        )

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
        gross_minus_net_total_return_pct = float(
            gross_total_return_pct - net_total_return_pct
        )

        # [Stage1] avg_cost_pct 계산: 평균 비용을 퍼센트로 (total_cost의 평균)
        avg_cost_pct = (
            float(g["total_cost"].mean() * 100.0)
            if "total_cost" in g.columns and len(g) > 0
            else 0.0
        )

        # [최종 수치셋] Calmar Ratio 계산: CAGR / |MDD|
        def _calculate_calmar_ratio(cagr: float, mdd: float) -> float:
            """Calmar Ratio = CAGR / |MDD|"""
            if mdd == 0:
                return float("inf") if cagr > 0 else 0.0
            abs_mdd = abs(mdd)
            if abs_mdd < 1e-9:  # MDD가 거의 0이면
                return float("inf") if cagr > 0 else 0.0
            return float(cagr / abs_mdd)

        gross_calmar = _calculate_calmar_ratio(gross_cagr, mdd_g)
        net_calmar = _calculate_calmar_ratio(net_cagr, mdd_n)

        # [최종 수치셋] Profit Factor 계산: 총 이익 / 총 손실
        def _calculate_profit_factor(returns: np.ndarray) -> float:
            """Profit Factor = sum(양수 수익) / abs(sum(음수 수익))"""
            profits = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            if losses == 0:
                return float("inf") if profits > 0 else 0.0
            return float(profits / losses)

        gross_profit_factor = (
            _calculate_profit_factor(r_gross) if len(r_gross) > 0 else 0.0
        )
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
                for ticker, ticker_positions in phase_positions.groupby(
                    "ticker", sort=False
                ):
                    ticker_positions = ticker_positions.sort_values("date")
                    if len(ticker_positions) > 1:
                        # 연속된 날짜 간격 계산
                        dates = ticker_positions["date"].values
                        for i in range(len(dates) - 1):
                            # [오류 수정] numpy.timedelta64는 .days 속성이 없으므로 pd.Timedelta로 변환
                            days_diff = pd.Timedelta(dates[i + 1] - dates[i]).days
                            if (
                                days_diff <= cfg.holding_days * 2
                            ):  # 리밸런싱 주기 내 연속 보유
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
                "cost_bps_config": float(
                    cost_bps_config
                ),  # [Stage1] config.yaml에서 읽은 원본 cost_bps
                "cost_bps_mismatch_flag": bool(
                    cost_bps_mismatch_flag
                ),  # [Stage1] 불일치 플래그
                "gross_minus_net_total_return_pct": float(
                    gross_minus_net_total_return_pct
                ),  # [Stage1] gross - net 차이 (퍼센트)
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
                "gross_hit_ratio": (
                    float((r_gross > 0).mean()) if len(r_gross) else np.nan
                ),
                "net_hit_ratio": float((r_net > 0).mean()) if len(r_net) else np.nan,
                "avg_turnover_oneway": (
                    float(g["turnover_oneway"].mean()) if len(g) else np.nan
                ),
                "avg_n_tickers": float(g["n_tickers"].mean()) if len(g) else np.nan,
                # [개선안 34번][최종 수치셋] Alpha Quality
                "ic": ic_mean,
                "rank_ic": ric_mean,
                "icir": icir,
                "rank_icir": ricir,
                "long_short_alpha": ls_alpha_mean,
                "long_short_alpha_ann": ls_alpha_ann,
                # [최종 수치셋] Calmar Ratio 추가
                "gross_calmar_ratio": (
                    float(gross_calmar) if not np.isinf(gross_calmar) else np.nan
                ),
                "net_calmar_ratio": (
                    float(net_calmar) if not np.isinf(net_calmar) else np.nan
                ),
                # [최종 수치셋] Profit Factor 추가
                "gross_profit_factor": (
                    float(gross_profit_factor)
                    if not np.isinf(gross_profit_factor)
                    else np.nan
                ),
                "net_profit_factor": (
                    float(net_profit_factor)
                    if not np.isinf(net_profit_factor)
                    else np.nan
                ),
                # [최종 수치셋] Avg Trade Duration 추가
                "avg_trade_duration": avg_trade_duration,
                "date_start": d0,
                "date_end": d1,
                "weighting": cfg.weighting,
            }
        )

    bt_metrics = pd.DataFrame(met_rows)

    # [최종 수치셋] 국면별 성과 계산 (bt_positions, bt_returns_core 생성 후)
    regime_metrics_rows: list[dict] = []
    if (
        regime_enabled_actual
        and len(bt_returns_diagnostics) > 0
        and "regime" in bt_returns_diagnostics.columns
    ):
        # bt_returns와 bt_returns_diagnostics 병합
        bt_returns_with_regime = bt_returns_core.merge(
            bt_returns_diagnostics[["date", "phase", "regime"]],
            on=["date", "phase"],
            how="left",
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
                years_regime = max(
                    (
                        (pd.Timedelta(d1_regime - d0_regime).days / 365.25)
                        if pd.notna(d0_regime) and pd.notna(d1_regime)
                        else 1e-9
                    ),
                    1e-9,
                )

                # [개선안 26번] Overflow 방지: regime 구간이 매우 짧을 때 (years_regime≈0) CAGR 계산이 폭주할 수 있음
                # - log 기반으로 계산하고, exp 입력을 clip하여 런타임 크래시를 방지
                net_cagr_regime = -1.0
                if eq_n_regime > 0 and years_regime > 0:
                    try:
                        rate = float(np.log(eq_n_regime) / years_regime)
                        # exp(20) ~ 4.85e8: 과도한 수치로 인한 Overflow 방지용 상한
                        rate_clip = float(np.clip(rate, -20.0, 20.0))
                        net_cagr_regime = float(np.exp(rate_clip) - 1.0)
                    except Exception:
                        net_cagr_regime = -1.0
                net_vol_regime = (
                    float(np.std(r_net_regime, ddof=1) * np.sqrt(periods_per_year))
                    if len(r_net_regime) > 1
                    else 0.0
                )
                net_sharpe_regime = (
                    float(
                        (np.mean(r_net_regime) / (np.std(r_net_regime, ddof=1) + 1e-12))
                        * np.sqrt(periods_per_year)
                    )
                    if len(r_net_regime) > 1
                    else 0.0
                )

                mdd_regime = _mdd(r_net_regime) if len(r_net_regime) else 0.0
                if eq_n_regime <= 0:
                    mdd_regime = -1.0

                net_hit_ratio_regime = (
                    float((r_net_regime > 0).mean()) if len(r_net_regime) else np.nan
                )
                net_total_return_regime = float(eq_n_regime - 1.0)

                regime_metrics_rows.append(
                    {
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
                    }
                )

    bt_regime_metrics = (
        pd.DataFrame(regime_metrics_rows) if regime_metrics_rows else pd.DataFrame()
    )

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
    # [Stage13] runtime_profile: empty-safe
    runtime_profile = pd.DataFrame(runtime_profile_rows)
    if (
        len(runtime_profile) > 0
        and "phase" in runtime_profile.columns
        and "date" in runtime_profile.columns
    ):
        runtime_profile = runtime_profile.sort_values(["phase", "date"]).reset_index(
            drop=True
        )
    else:
        runtime_profile = pd.DataFrame(columns=["date", "phase"])

    # [Stage13] selection_diagnostics 반환
    # bt_returns_core를 bt_returns로 사용 (validation 통과용)
    # bt_returns_diagnostics는 별도로 저장
    # [최종 수치셋] bt_regime_metrics 추가 반환
    # 호환성을 위해 기존 인터페이스 유지 (6개 값 반환)
    return bt_positions, bt_returns_core, bt_equity_curve, bt_metrics, quality, warns
