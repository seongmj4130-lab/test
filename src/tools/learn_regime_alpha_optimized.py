# -*- coding: utf-8 -*-
"""
[Phase 2] 국면별 α 자동 학습 스크립트 (최적화 버전)

목적: 국면(bull/neutral/bear)별로 듀얼호라이즌 결합 가중치 α를 데이터로부터 자동 학습

최적화:
- L6R 재실행 대신 ranking_short_daily와 ranking_long_daily를 직접 사용하여 score_ens 재계산
- 각 α 조합마다 L7 백테스트만 실행 (L6R 불필요)

주의사항:
- ⚠️ 누수 방지: α 학습은 Dev만 사용, Holdout 절대 금지
- ⚠️ 동일 기준 비교: 동일 cost_bps, holding_days, 리밸런싱 규칙 유지
"""
from __future__ import annotations

import sys
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import json
import copy

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists, save_artifact
from src.stages.backtest.regime_utils import map_regime_to_3level

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_required_artifacts(cfg: dict) -> dict:
    """필수 산출물 로드"""
    interim_dir = get_path(cfg, "data_interim")
    
    artifacts = {}
    required = [
        "cv_folds_short",
        "ranking_short_daily",
        "ranking_long_daily",
    ]
    
    for name in required:
        base = interim_dir / name
        if not artifact_exists(base):
            raise FileNotFoundError(f"Required artifact not found: {base}")
        artifacts[name] = load_artifact(base)
        logger.info(f"Loaded {name}: {len(artifacts[name]):,} rows")
    
    return artifacts


def load_market_regime(cfg: dict, rebalance_dates: pd.Series) -> pd.DataFrame:
    """시장 국면 데이터 로드 또는 생성"""
    interim_dir = get_path(cfg, "data_interim")
    
    # 기존 market_regime_daily가 있으면 로드
    base = interim_dir / "market_regime_daily"
    if artifact_exists(base):
        regime_df = load_artifact(base)
        logger.info(f"Loaded market_regime_daily: {len(regime_df):,} rows")
    else:
        # MIDTERM 기준 데이터 확인
        midterm_base = interim_dir / "ranking_20251223_164432" / "market_regime.parquet"
        if midterm_base.exists():
            regime_df = load_artifact(midterm_base.parent / "market_regime")
            logger.info(f"Loaded MIDTERM market_regime: {len(regime_df):,} rows")
        else:
            # 없으면 생성
            logger.info("Generating market_regime_daily...")
            from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime
            
            start_date = cfg["params"]["start_date"]
            end_date = cfg["params"]["end_date"]
            
            regime_cfg = cfg.get("l7", {}).get("regime", {})
            regime_df = build_market_regime(
                rebalance_dates=rebalance_dates,
                start_date=start_date,
                end_date=end_date,
                lookback_days=regime_cfg.get("lookback_days", 60),
                threshold_pct=regime_cfg.get("threshold_pct", 0.0),
                neutral_band=regime_cfg.get("neutral_band", 0.05),  # [Phase 2]
            )
    
    # 3단계 국면 컬럼 추가
    if "regime_3" not in regime_df.columns:
        from src.stages.backtest.regime_utils import apply_3level_regime
        regime_df = apply_3level_regime(regime_df, regime_col="regime", out_col="regime_3")
    
    return regime_df


def get_dev_rebalance_dates(cv_folds: pd.DataFrame) -> pd.Series:
    """Dev 구간의 리밸런싱 날짜 추출"""
    cv_folds = cv_folds.copy()
    cv_folds["test_end"] = pd.to_datetime(cv_folds["test_end"])
    
    # Dev 구간만 필터링
    dev_folds = cv_folds[cv_folds["segment"] == "dev"].copy()
    dev_dates = dev_folds["test_end"].unique()
    
    logger.info(f"Dev rebalance dates: {len(dev_dates)} dates")
    return pd.Series(sorted(dev_dates), name="date")


def compute_rebalance_scores_with_alpha(
    ranking_short: pd.DataFrame,
    ranking_long: pd.DataFrame,
    rebalance_dates: pd.Series,
    cv_folds: pd.DataFrame,
    market_regime_df: pd.DataFrame,
    regime_alpha: Dict[str, float],
    score_source: str = "score_total",
) -> pd.DataFrame:
    """
    ranking_short와 ranking_long을 사용하여 특정 α 조합으로 rebalance_scores 재계산
    
    이 함수는 L6R의 핵심 로직만 재현하여 효율적으로 score_ens를 계산합니다.
    """
    # 날짜/티커 정규화
    r_short = ranking_short.copy()
    r_long = ranking_long.copy()
    
    r_short["date"] = pd.to_datetime(r_short["date"])
    r_long["date"] = pd.to_datetime(r_long["date"])
    r_short["ticker"] = r_short["ticker"].astype(str).str.zfill(6)
    r_long["ticker"] = r_long["ticker"].astype(str).str.zfill(6)
    
    # 리밸런싱 날짜 매핑 (phase 포함)
    cv_folds = cv_folds.copy()
    cv_folds["test_end"] = pd.to_datetime(cv_folds["test_end"])
    cv_folds["phase"] = cv_folds["segment"].map({"dev": "dev", "holdout": "holdout"}).fillna("dev")
    
    rebal_map = cv_folds[["test_end", "phase"]].rename(columns={"test_end": "date"}).drop_duplicates("date")
    
    # 리밸런싱 날짜로 필터링
    r_short = r_short[r_short["date"].isin(rebalance_dates)].copy()
    r_long = r_long[r_long["date"].isin(rebalance_dates)].copy()
    
    # phase 부착
    r_short = r_short.merge(rebal_map, on="date", how="inner", validate="many_to_one")
    r_long = r_long.merge(rebal_map, on="date", how="inner", validate="many_to_one")
    
    # 단기/장기 랭킹 병합
    key = ["date", "ticker", "phase"]
    r = r_short[["date", "ticker", "phase", "score_total", "rank_total"]].merge(
        r_long[["date", "ticker", "phase", "score_total", "rank_total"]],
        on=key,
        how="outer",
        suffixes=("_short", "_long"),
        validate="one_to_one"
    )
    
    # 정규화: score_total 사용 (높을수록 좋음)
    if score_source == "rank_total":
        # rank_total을 사용: 낮은 rank가 좋으므로 -rank로 변환
        r["score_short_norm"] = -pd.to_numeric(r["rank_total_short"], errors="coerce").fillna(0)
        r["score_long_norm"] = -pd.to_numeric(r["rank_total_long"], errors="coerce").fillna(0)
    else:
        # score_total을 사용: 높은 score가 좋음
        r["score_short_norm"] = pd.to_numeric(r["score_total_short"], errors="coerce").fillna(0)
        r["score_long_norm"] = pd.to_numeric(r["score_total_long"], errors="coerce").fillna(0)
    
    # 시장 국면 데이터와 병합
    market_regime_df = market_regime_df.copy()
    if "date" in market_regime_df.columns:
        market_regime_df["date"] = pd.to_datetime(market_regime_df["date"])
    
    regime_col = "regime_3" if "regime_3" in market_regime_df.columns else "regime"
    if regime_col == "regime" and "regime" in market_regime_df.columns:
        market_regime_df["regime_3"] = market_regime_df["regime"].apply(map_regime_to_3level)
        regime_col = "regime_3"
    
    # 국면 데이터가 없으면 기본값 사용
    if regime_col not in market_regime_df.columns:
        r[regime_col] = "neutral"
    else:
        r = r.merge(
            market_regime_df[["date", regime_col]],
            on="date",
            how="left",
            validate="many_to_one"
        )
        # 결측값은 neutral로
        r[regime_col] = r[regime_col].fillna("neutral")
    
    # 국면별 α 적용
    def get_alpha_for_regime(regime: str) -> float:
        regime_lower = (regime or "").strip().lower()
        if regime_lower in regime_alpha:
            return float(regime_alpha[regime_lower])
        # 3단계로 변환
        r3 = map_regime_to_3level(regime_lower)
        return float(regime_alpha.get(r3, 0.5))
    
    r["alpha_short"] = r[regime_col].apply(get_alpha_for_regime)
    r["alpha_long"] = 1.0 - r["alpha_short"]
    
    # 결합: score_ens = α * score_short + (1-α) * score_long
    r["score_ens"] = (
        r["alpha_short"] * r["score_short_norm"] + r["alpha_long"] * r["score_long_norm"]
    )
    
    # 재랭킹
    r["score_total"] = r["score_ens"]
    r["rank_total"] = r.groupby(["date", "phase"], sort=False)["score_ens"].rank(
        method="first", ascending=False
    )
    
    # true_short 추가 (dataset_daily에서 가져와야 하지만, 여기서는 일단 생략)
    # 실제로는 rebalance_scores에 이미 있으므로 merge 필요
    
    # 필수 컬럼만 남기기
    result = r[["date", "ticker", "phase", "score_total", "rank_total", "score_ens"]].copy()
    
    return result


def evaluate_alpha_combination(
    cfg: dict,
    artifacts: dict,
    regime_alpha: Dict[str, float],
    dev_dates: pd.Series,
    market_regime_df: pd.DataFrame,
    existing_scores: Optional[pd.DataFrame] = None,
) -> Tuple[float, dict]:
    """
    특정 α 조합에 대한 Dev 성과 평가 (최적화 버전)
    
    Returns:
        (net_sharpe, metrics_dict)
    """
    try:
        # 기존 rebalance_scores에서 true_short 가져오기
        if existing_scores is None:
            interim_dir = get_path(cfg, "data_interim")
            base = interim_dir / "rebalance_scores"
            if artifact_exists(base):
                existing_scores = load_artifact(base)
            else:
                # MIDTERM 기준 데이터 확인
                midterm_base = interim_dir / "ranking_20251223_164432" / "rebalance_scores.parquet"
                if midterm_base.exists():
                    existing_scores = load_artifact(midterm_base.parent / "rebalance_scores")
        
        if existing_scores is None:
            logger.error("Cannot find existing rebalance_scores to extract true_short")
            return -999.0, {}
        
        # score_ens 재계산
        recomputed_scores = compute_rebalance_scores_with_alpha(
            ranking_short=artifacts["ranking_short_daily"],
            ranking_long=artifacts["ranking_long_daily"],
            rebalance_dates=dev_dates,
            cv_folds=artifacts["cv_folds_short"],
            market_regime_df=market_regime_df,
            regime_alpha=regime_alpha,
            score_source=cfg.get("l7", {}).get("ranking_score_source", "score_total"),
        )
        
        # true_short 병합 (기존 scores에서)
        existing_scores = existing_scores.copy()
        existing_scores["date"] = pd.to_datetime(existing_scores["date"])
        true_cols = ["date", "ticker", "phase", "true_short"]
        if "true_short" in existing_scores.columns:
            true_data = existing_scores[true_cols].drop_duplicates(["date", "ticker", "phase"])
            recomputed_scores = recomputed_scores.merge(
                true_data, on=["date", "ticker", "phase"], how="left", validate="many_to_one"
            )
        
        # Dev만 필터링
        dev_scores = recomputed_scores[
            (recomputed_scores["phase"] == "dev") & 
            (recomputed_scores["date"].isin(dev_dates))
        ].copy()
        
        if len(dev_scores) == 0:
            logger.warning("No dev scores found for this alpha combination")
            return -999.0, {}
        
        # true_short가 없으면 기본값 사용 (경고)
        if "true_short" not in dev_scores.columns or dev_scores["true_short"].isna().all():
            logger.warning("true_short not found, cannot compute accurate metrics")
            return -999.0, {}
        
        # L7 백테스트 실행 (Dev만)
        from src.tracks.track_b.stages.backtest.l7_backtest import run_backtest, BacktestConfig
        
        l7_cfg = cfg.get("l7", {})
        bt_config = BacktestConfig(
            holding_days=int(l7_cfg.get("holding_days", 20)),
            top_k=int(l7_cfg.get("top_k", 20)),
            cost_bps=float(l7_cfg.get("cost_bps", 10.0)),
            score_col="score_ens",
            ret_col="true_short",
            weighting=str(l7_cfg.get("weighting", "equal")),
            buffer_k=int(l7_cfg.get("buffer_k", 20)),
        )
        
        bt_result = run_backtest(
            rebalance_scores=dev_scores,
            cfg=bt_config,
            config_cost_bps=bt_config.cost_bps,
            market_regime=None,  # 이미 score_ens에 국면별 α가 반영됨
        )
        
        bt_metrics = bt_result[3]  # bt_metrics
        
        # Dev 성과 추출
        dev_metrics = bt_metrics[bt_metrics["phase"] == "dev"]
        if len(dev_metrics) == 0:
            logger.warning("No dev metrics found")
            return -999.0, {}
        
        net_sharpe = float(dev_metrics["net_sharpe"].iloc[0])
        metrics = {
            "net_sharpe": net_sharpe,
            "net_total_return": float(dev_metrics["net_total_return"].iloc[0]),
            "net_mdd": float(dev_metrics["net_mdd"].iloc[0]),
            "net_hit_ratio": float(dev_metrics["net_hit_ratio"].iloc[0]),
            "avg_turnover_oneway": float(dev_metrics["avg_turnover_oneway"].iloc[0]),
        }
        
        return net_sharpe, metrics
        
    except Exception as e:
        logger.error(f"Error evaluating alpha combination {regime_alpha}: {e}", exc_info=True)
        return -999.0, {}


def grid_search_regime_alpha(
    cfg: dict,
    artifacts: dict,
    dev_dates: pd.Series,
    market_regime_df: pd.DataFrame,
    alpha_candidates: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
) -> Tuple[Dict[str, float], List[dict]]:
    """
    그리드 서치를 통한 국면별 최적 α 학습
    
    Returns:
        (최적 regime_alpha 딕셔너리, 결과 요약 리스트)
    """
    logger.info("=" * 80)
    logger.info("국면별 α 그리드 서치 시작 (최적화 버전)")
    logger.info("=" * 80)
    logger.info(f"Alpha candidates: {alpha_candidates}")
    
    # 국면별 날짜 분류
    market_regime_df = market_regime_df.copy()
    if "date" in market_regime_df.columns:
        market_regime_df["date"] = pd.to_datetime(market_regime_df["date"])
    
    # 3단계 국면 컬럼 확인
    regime_col = "regime_3" if "regime_3" in market_regime_df.columns else "regime"
    if regime_col == "regime" and "regime" in market_regime_df.columns:
        market_regime_df["regime_3"] = market_regime_df["regime"].apply(map_regime_to_3level)
        regime_col = "regime_3"
    
    # Dev 날짜에 해당하는 국면 매핑
    if regime_col in market_regime_df.columns and "date" in market_regime_df.columns:
        regime_map = market_regime_df.set_index("date")[regime_col].to_dict()
        dev_regimes = {date: regime_map.get(date, "neutral") for date in dev_dates}
    else:
        # 국면 데이터가 없으면 모두 neutral
        logger.warning("Market regime data not available, using neutral for all dates")
        dev_regimes = {date: "neutral" for date in dev_dates}
    
    regime_counts = pd.Series(dev_regimes.values()).value_counts()
    logger.info(f"\nDev 국면 분포:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} dates ({count/len(dev_dates)*100:.1f}%)")
    
    # 그리드 서치: 각 국면별로 독립적으로 최적화
    optimal_alphas = {}
    results_summary = []
    
    regimes = ["bull", "neutral", "bear"]
    
    # 기존 rebalance_scores 로드 (true_short 추출용)
    interim_dir = get_path(cfg, "data_interim")
    existing_scores = None
    base = interim_dir / "rebalance_scores"
    if artifact_exists(base):
        existing_scores = load_artifact(base)
    else:
        midterm_base = interim_dir / "ranking_20251223_164432" / "rebalance_scores.parquet"
        if midterm_base.exists():
            existing_scores = load_artifact(midterm_base.parent / "rebalance_scores")
    
    for regime in regimes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Regime: {regime}")
        logger.info(f"{'='*80}")
        
        regime_dates = [d for d, r in dev_regimes.items() if r == regime]
        
        if len(regime_dates) == 0:
            logger.warning(f"No dates for regime {regime}, using default alpha 0.5")
            optimal_alphas[regime] = 0.5
            continue
        
        logger.info(f"Evaluating {len(alpha_candidates)} alpha values for {len(regime_dates)} dates...")
        
        best_sharpe = -999.0
        best_alpha = 0.5
        best_metrics = {}
        
        for alpha in alpha_candidates:
            logger.info(f"  Testing alpha={alpha:.1f}...")
            
            # 이 국면에만 해당 α 적용, 다른 국면은 기본값 사용
            test_regime_alpha = {
                "bull": optimal_alphas.get("bull", 0.5),
                "neutral": optimal_alphas.get("neutral", 0.5),
                "bear": optimal_alphas.get("bear", 0.5),
            }
            test_regime_alpha[regime] = alpha
            
            # 평가 (모든 Dev 날짜에 대해, 해당 국면 날짜만 이 α 사용)
            net_sharpe, metrics = evaluate_alpha_combination(
                cfg=cfg,
                artifacts=artifacts,
                regime_alpha=test_regime_alpha,
                dev_dates=pd.Series(dev_dates),  # 모든 Dev 날짜
                market_regime_df=market_regime_df,
                existing_scores=existing_scores,
            )
            
            logger.info(f"Sharpe: {net_sharpe:.4f}")
            results_summary.append({
                "regime": regime,
                "alpha": alpha,
                "net_sharpe": net_sharpe,
                **metrics,
            })
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_alpha = alpha
                best_metrics = metrics
        
        optimal_alphas[regime] = best_alpha
        logger.info(f"\n  ✅ Best alpha for {regime}: {best_alpha:.1f} (Sharpe: {best_sharpe:.4f})")
        logger.info(f"  Metrics: {best_metrics}")
    
    logger.info(f"\n{'='*80}")
    logger.info("최적 α 학습 완료")
    logger.info(f"{'='*80}")
    logger.info(f"Optimal regime_alpha:")
    for regime, alpha in optimal_alphas.items():
        logger.info(f"  {regime}: {alpha:.1f}")
    
    return optimal_alphas, results_summary


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="국면별 α 자동 학습 (최적화 버전)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="docs/regime_alpha_learning_results.json")
    args = parser.parse_args()
    
    # 설정 로드
    cfg = load_config(args.config)
    interim_dir = get_path(cfg, "data_interim")
    
    logger.info("=" * 80)
    logger.info("Phase 2: 국면별 α 자동 학습 (최적화 버전)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Interim dir: {interim_dir}")
    
    # 필수 산출물 로드
    logger.info("\n[1] 필수 산출물 로드")
    artifacts = load_required_artifacts(cfg)
    
    # Dev 리밸런싱 날짜 추출
    logger.info("\n[2] Dev 리밸런싱 날짜 추출")
    dev_dates = get_dev_rebalance_dates(artifacts["cv_folds_short"])
    
    # 시장 국면 데이터 로드
    logger.info("\n[3] 시장 국면 데이터 로드")
    market_regime_df = load_market_regime(cfg, dev_dates)
    
    # 그리드 서치 실행
    logger.info("\n[4] 그리드 서치 실행")
    optimal_alphas, results_summary = grid_search_regime_alpha(
        cfg=cfg,
        artifacts=artifacts,
        dev_dates=dev_dates,
        market_regime_df=market_regime_df,
    )
    
    # 결과 저장
    logger.info("\n[5] 결과 저장")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "optimal_regime_alpha": optimal_alphas,
        "results_summary": results_summary,
        "config_used": {
            "holding_days": cfg.get("l7", {}).get("holding_days", 20),
            "top_k": cfg.get("l7", {}).get("top_k", 20),
            "cost_bps": cfg.get("l7", {}).get("cost_bps", 10.0),
            "buffer_k": cfg.get("l7", {}).get("buffer_k", 20),
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Results saved to: {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("학습 완료!")
    logger.info("=" * 80)
    logger.info("\n다음 단계:")
    logger.info("1. 학습된 α 값을 config.yaml의 l6r.regime_alpha에 반영")
    logger.info("2. 전체 파이프라인 재실행 (Holdout 성과 검증)")
    logger.info("3. docs/midterm_vs_phases_metrics.md 업데이트")


if __name__ == "__main__":
    main()

