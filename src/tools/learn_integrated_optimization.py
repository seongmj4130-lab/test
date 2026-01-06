# -*- coding: utf-8 -*-
"""
[Phase 4.3] 통합 최적화 스크립트: L8 가중치 + L6R α 동시 최적화

목적: L8 feature_groups 가중치와 L6R regime_alpha를 동시에 최적화하여 성과 개선

전략:
- L8 가중치 조합을 몇 가지로 제한 (탐색 공간 축소)
- 각 가중치 조합에 대해 L6R α 최적화
- 최종적으로 최적 조합 선택

주의사항:
- ⚠️ 누수 방지: 최적화는 Dev만 사용, Holdout 절대 금지
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
import yaml
import shutil
from datetime import datetime

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
        "dataset_daily",  # L8 재실행에 필요
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
            end_date = cfg["params"].get("end_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
            
            regime_cfg = cfg.get("l7", {}).get("regime", {})
            regime_df = build_market_regime(
                rebalance_dates=rebalance_dates,
                start_date=start_date,
                end_date=end_date,
                lookback_days=regime_cfg.get("lookback_days", 60),
                threshold_pct=regime_cfg.get("threshold_pct", 0.0),
                neutral_band=regime_cfg.get("neutral_band", 0.05),
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


def generate_l8_weight_combinations() -> List[dict]:
    """
    L8 가중치 조합 생성
    
    Phase 4.1 기준:
    - short: technical=0.50, value=0.15, profitability=0.15, other=0.20
    - long: technical=0.20, value=0.40, profitability=0.40, other=0.00
    
    탐색 공간:
    - short technical: [0.40, 0.50, 0.60]
    - short value+profitability: [0.20, 0.30, 0.40] (합계)
    - long technical: [0.15, 0.20, 0.25]
    - long value+profitability: [0.50, 0.60, 0.70] (합계)
    """
    combinations = []
    
    # Phase 4.1 기준 조합 (기준선)
    combinations.append({
        "name": "phase4_1_baseline",
        "short": {
            "technical": 0.50,
            "value": 0.15,
            "profitability": 0.15,
            "other": 0.20,
        },
        "long": {
            "technical": 0.20,
            "value": 0.40,
            "profitability": 0.40,
            "other": 0.00,
        }
    })
    
    # 변형 조합들
    # 1. 단기 기술적 팩터 강화
    combinations.append({
        "name": "short_tech_high",
        "short": {
            "technical": 0.60,
            "value": 0.10,
            "profitability": 0.10,
            "other": 0.20,
        },
        "long": {
            "technical": 0.20,
            "value": 0.40,
            "profitability": 0.40,
            "other": 0.00,
        }
    })
    
    # 2. 단기 기술적 팩터 완화
    combinations.append({
        "name": "short_tech_low",
        "short": {
            "technical": 0.40,
            "value": 0.20,
            "profitability": 0.20,
            "other": 0.20,
        },
        "long": {
            "technical": 0.20,
            "value": 0.40,
            "profitability": 0.40,
            "other": 0.00,
        }
    })
    
    # 3. 장기 가치/수익성 강화
    combinations.append({
        "name": "long_value_high",
        "short": {
            "technical": 0.50,
            "value": 0.15,
            "profitability": 0.15,
            "other": 0.20,
        },
        "long": {
            "technical": 0.15,
            "value": 0.45,
            "profitability": 0.40,
            "other": 0.00,
        }
    })
    
    # 4. 장기 가치/수익성 완화
    combinations.append({
        "name": "long_value_low",
        "short": {
            "technical": 0.50,
            "value": 0.15,
            "profitability": 0.15,
            "other": 0.20,
        },
        "long": {
            "technical": 0.25,
            "value": 0.35,
            "profitability": 0.40,
            "other": 0.00,
        }
    })
    
    # 5. 균형 조합
    combinations.append({
        "name": "balanced",
        "short": {
            "technical": 0.45,
            "value": 0.20,
            "profitability": 0.20,
            "other": 0.15,
        },
        "long": {
            "technical": 0.20,
            "value": 0.40,
            "profitability": 0.40,
            "other": 0.00,
        }
    })
    
    return combinations


def update_feature_groups_config(
    config_path: Path,
    weights: dict,
    output_path: Path,
) -> None:
    """feature_groups config 파일 업데이트"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # target_weight 업데이트
    for group_name, target_weight in weights.items():
        if group_name in config.get("feature_groups", {}):
            config["feature_groups"][group_name]["target_weight"] = target_weight
    
    # 합계가 1.0이 되도록 정규화
    total = sum(
        g.get("target_weight", 0.0)
        for g in config.get("feature_groups", {}).values()
    )
    if total > 0:
        for group_name in config.get("feature_groups", {}):
            if group_name in weights:
                config["feature_groups"][group_name]["target_weight"] /= total
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    logger.info(f"Updated feature_groups config: {output_path}")


def run_l8_with_weights(
    cfg: dict,
    artifacts: dict,
    weight_combination: dict,
    temp_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    특정 가중치 조합으로 L8_short/L8_long 재실행
    
    Returns:
        (ranking_short_daily, ranking_long_daily)
    """
    from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
        run_L8_short_rank_engine,
        run_L8_long_rank_engine,
    )
    
    # 임시 config 파일 생성
    temp_short_config = temp_dir / "feature_groups_short_temp.yaml"
    temp_long_config = temp_dir / "feature_groups_long_temp.yaml"
    
    # 원본 config 파일 경로
    base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
    base_short_config = base_dir / cfg["l8_short"]["feature_groups_config"]
    base_long_config = base_dir / cfg["l8_long"]["feature_groups_config"]
    
    # 가중치 업데이트
    update_feature_groups_config(
        base_short_config,
        weight_combination["short"],
        temp_short_config,
    )
    update_feature_groups_config(
        base_long_config,
        weight_combination["long"],
        temp_long_config,
    )
    
    # 임시 config로 cfg 업데이트 (절대 경로 사용)
    temp_cfg = copy.deepcopy(cfg)
    # 절대 경로로 변환하여 사용
    try:
        temp_short_rel = str(temp_short_config.relative_to(base_dir))
    except ValueError:
        # 상대 경로 변환 실패 시 절대 경로 사용
        temp_short_rel = str(temp_short_config.resolve())
    
    try:
        temp_long_rel = str(temp_long_config.relative_to(base_dir))
    except ValueError:
        # 상대 경로 변환 실패 시 절대 경로 사용
        temp_long_rel = str(temp_long_config.resolve())
    
    temp_cfg["l8_short"]["feature_groups_config"] = temp_short_rel
    temp_cfg["l8_long"]["feature_groups_config"] = temp_long_rel
    
    # [Phase 4.5] feature_weights_config를 None으로 설정하여 feature_groups_config 사용 강제
    # feature_weights가 있으면 feature_groups_config가 무시되기 때문
    temp_cfg["l8_short"]["feature_weights_config"] = None
    temp_cfg["l8_long"]["feature_weights_config"] = None
    
    # L8_short 실행
    logger.info(f"Running L8_short with weights: {weight_combination['short']}")
    outputs_short, _ = run_L8_short_rank_engine(temp_cfg, artifacts, force=True)
    ranking_short = outputs_short["ranking_short_daily"]
    
    # L8_long 실행
    logger.info(f"Running L8_long with weights: {weight_combination['long']}")
    outputs_long, _ = run_L8_long_rank_engine(temp_cfg, artifacts, force=True)
    ranking_long = outputs_long["ranking_long_daily"]
    
    return ranking_short, ranking_long


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
        r["score_short_norm"] = -pd.to_numeric(r["rank_total_short"], errors="coerce").fillna(0)
        r["score_long_norm"] = -pd.to_numeric(r["rank_total_long"], errors="coerce").fillna(0)
    else:
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
    
    if regime_col not in market_regime_df.columns:
        r[regime_col] = "neutral"
    else:
        r = r.merge(
            market_regime_df[["date", regime_col]],
            on="date",
            how="left",
            validate="many_to_one"
        )
        r[regime_col] = r[regime_col].fillna("neutral")
    
    # 국면별 α 적용
    def get_alpha_for_regime(regime: str) -> float:
        regime_lower = (regime or "").strip().lower()
        if regime_lower in regime_alpha:
            return float(regime_alpha[regime_lower])
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
    
    result = r[["date", "ticker", "phase", "score_total", "rank_total", "score_ens"]].copy()
    
    return result


def evaluate_combination(
    cfg: dict,
    artifacts: dict,
    weight_combination: dict,
    regime_alpha: Dict[str, float],
    dev_dates: pd.Series,
    market_regime_df: pd.DataFrame,
    temp_dir: Path,
) -> Tuple[float, dict]:
    """
    특정 가중치 조합 + α 조합에 대한 Dev 성과 평가
    
    Returns:
        (net_sharpe, metrics_dict)
    """
    try:
        # L8 재실행 (가중치 조합 적용)
        ranking_short, ranking_long = run_l8_with_weights(
            cfg=cfg,
            artifacts=artifacts,
            weight_combination=weight_combination,
            temp_dir=temp_dir,
        )
        
        # rebalance_scores 재계산
        recomputed_scores = compute_rebalance_scores_with_alpha(
            ranking_short=ranking_short,
            ranking_long=ranking_long,
            rebalance_dates=dev_dates,
            cv_folds=artifacts["cv_folds_short"],
            market_regime_df=market_regime_df,
            regime_alpha=regime_alpha,
            score_source=cfg.get("l7", {}).get("ranking_score_source", "score_total"),
        )
        
        # true_short 또는 ret_fwd_20d 병합 (기존 dataset_daily에서)
        dataset = artifacts["dataset_daily"].copy()
        dataset["date"] = pd.to_datetime(dataset["date"])
        dataset["ticker"] = dataset["ticker"].astype(str).str.zfill(6)
        
        # ret_col 후보 순서대로 확인
        ret_col_candidates = ["true_short", "ret_fwd_20d", "y_true", "ret"]
        ret_col = None
        for col in ret_col_candidates:
            if col in dataset.columns:
                ret_col = col
                break
        
        if ret_col is None:
            logger.warning("No return column found in dataset_daily. Available columns: " + str(list(dataset.columns)))
            return -999.0, {}
        
        true_cols = ["date", "ticker", ret_col]
        true_data = dataset[true_cols].drop_duplicates(["date", "ticker"])
        # ret_col을 true_short로 이름 변경 (L7 백테스트 호환성)
        true_data = true_data.rename(columns={ret_col: "true_short"})
        recomputed_scores = recomputed_scores.merge(
            true_data, on=["date", "ticker"], how="left", validate="many_to_one"
        )
        
        # Dev만 필터링
        dev_scores = recomputed_scores[
            (recomputed_scores["phase"] == "dev") & 
            (recomputed_scores["date"].isin(dev_dates))
        ].copy()
        
        if len(dev_scores) == 0:
            logger.warning("No dev scores found for this combination")
            return -999.0, {}
        
        # true_short가 없으면 ret_fwd_20d 등을 확인
        if "true_short" not in dev_scores.columns:
            ret_col_candidates = ["ret_fwd_20d", "y_true", "ret"]
            for col in ret_col_candidates:
                if col in dev_scores.columns:
                    dev_scores["true_short"] = dev_scores[col]
                    logger.info(f"Using {col} as true_short for backtest")
                    break
        
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
            market_regime=None,
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
        logger.error(f"Error evaluating combination: {e}", exc_info=True)
        return -999.0, {}


def optimize_alpha_for_weights(
    cfg: dict,
    artifacts: dict,
    weight_combination: dict,
    dev_dates: pd.Series,
    market_regime_df: pd.DataFrame,
    temp_dir: Path,
    alpha_candidates: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
) -> Tuple[Dict[str, float], float, dict]:
    """
    특정 가중치 조합에 대해 L6R α 최적화
    
    Returns:
        (최적 regime_alpha, 최적 Sharpe, 최적 metrics)
    """
    logger.info(f"Optimizing alpha for weight combination: {weight_combination['name']}")
    
    # 국면별 날짜 분류
    market_regime_df = market_regime_df.copy()
    if "date" in market_regime_df.columns:
        market_regime_df["date"] = pd.to_datetime(market_regime_df["date"])
    
    regime_col = "regime_3" if "regime_3" in market_regime_df.columns else "regime"
    if regime_col == "regime" and "regime" in market_regime_df.columns:
        market_regime_df["regime_3"] = market_regime_df["regime"].apply(map_regime_to_3level)
        regime_col = "regime_3"
    
    if regime_col in market_regime_df.columns and "date" in market_regime_df.columns:
        regime_map = market_regime_df.set_index("date")[regime_col].to_dict()
        dev_regimes = {date: regime_map.get(date, "neutral") for date in dev_dates}
    else:
        dev_regimes = {date: "neutral" for date in dev_dates}
    
    # 국면별 독립 최적화
    optimal_alphas = {}
    regimes = ["bull", "neutral", "bear"]
    
    for regime in regimes:
        regime_dates = [d for d, r in dev_regimes.items() if r == regime]
        
        if len(regime_dates) == 0:
            logger.warning(f"No dates for regime {regime}, using default alpha 0.5")
            optimal_alphas[regime] = 0.5
            continue
        
        logger.info(f"  Regime {regime}: {len(regime_dates)} dates")
        
        best_sharpe = -999.0
        best_alpha = 0.5
        
        for alpha in alpha_candidates:
            test_regime_alpha = {
                "bull": optimal_alphas.get("bull", 0.5),
                "neutral": optimal_alphas.get("neutral", 0.5),
                "bear": optimal_alphas.get("bear", 0.5),
            }
            test_regime_alpha[regime] = alpha
            
            net_sharpe, _ = evaluate_combination(
                cfg=cfg,
                artifacts=artifacts,
                weight_combination=weight_combination,
                regime_alpha=test_regime_alpha,
                dev_dates=pd.Series(dev_dates),
                market_regime_df=market_regime_df,
                temp_dir=temp_dir,
            )
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_alpha = alpha
        
        optimal_alphas[regime] = best_alpha
        logger.info(f"    Best alpha for {regime}: {best_alpha:.1f} (Sharpe: {best_sharpe:.4f})")
    
    # 최적 α로 최종 평가
    final_sharpe, final_metrics = evaluate_combination(
        cfg=cfg,
        artifacts=artifacts,
        weight_combination=weight_combination,
        regime_alpha=optimal_alphas,
        dev_dates=pd.Series(dev_dates),
        market_regime_df=market_regime_df,
        temp_dir=temp_dir,
    )
    
    return optimal_alphas, final_sharpe, final_metrics


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4.3: 통합 최적화 (L8 가중치 + L6R α)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="docs/phase4_3_integrated_optimization_results.json")
    parser.add_argument("--temp-dir", type=str, default="data/interim/phase4_3_temp")
    args = parser.parse_args()
    
    # 설정 로드
    cfg = load_config(args.config)
    base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
    interim_dir = get_path(cfg, "data_interim")
    
    # temp_dir를 base_dir 내부에 생성 (상대 경로 변환을 위해)
    if Path(args.temp_dir).is_absolute():
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = base_dir / args.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Phase 4.3: 통합 최적화 (L8 가중치 + L6R α)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Temp dir: {temp_dir}")
    
    # 필수 산출물 로드
    logger.info("\n[1] 필수 산출물 로드")
    artifacts = load_required_artifacts(cfg)
    
    # Dev 리밸런싱 날짜 추출
    logger.info("\n[2] Dev 리밸런싱 날짜 추출")
    dev_dates = get_dev_rebalance_dates(artifacts["cv_folds_short"])
    
    # 시장 국면 데이터 로드
    logger.info("\n[3] 시장 국면 데이터 로드")
    market_regime_df = load_market_regime(cfg, dev_dates)
    
    # L8 가중치 조합 생성
    logger.info("\n[4] L8 가중치 조합 생성")
    weight_combinations = generate_l8_weight_combinations()
    logger.info(f"Generated {len(weight_combinations)} weight combinations")
    
    # 각 가중치 조합에 대해 α 최적화
    logger.info("\n[5] 통합 최적화 실행")
    results = []
    
    for i, weight_combo in enumerate(weight_combinations, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Weight Combination {i}/{len(weight_combinations)}: {weight_combo['name']}")
        logger.info(f"{'='*80}")
        
        optimal_alphas, best_sharpe, best_metrics = optimize_alpha_for_weights(
            cfg=cfg,
            artifacts=artifacts,
            weight_combination=weight_combo,
            dev_dates=dev_dates,
            market_regime_df=market_regime_df,
            temp_dir=temp_dir,
        )
        
        results.append({
            "weight_combination": weight_combo,
            "optimal_regime_alpha": optimal_alphas,
            "dev_sharpe": best_sharpe,
            "dev_metrics": best_metrics,
        })
        
        logger.info(f"\n  ✅ Best Sharpe: {best_sharpe:.4f}")
        logger.info(f"  Optimal alphas: {optimal_alphas}")
    
    # 최적 조합 선택
    logger.info("\n[6] 최적 조합 선택")
    best_result = max(results, key=lambda x: x["dev_sharpe"])
    
    logger.info(f"\n{'='*80}")
    logger.info("최적 조합:")
    logger.info(f"{'='*80}")
    logger.info(f"Weight Combination: {best_result['weight_combination']['name']}")
    logger.info(f"  Short weights: {best_result['weight_combination']['short']}")
    logger.info(f"  Long weights: {best_result['weight_combination']['long']}")
    logger.info(f"Optimal Alphas: {best_result['optimal_regime_alpha']}")
    logger.info(f"Dev Sharpe: {best_result['dev_sharpe']:.4f}")
    logger.info(f"Dev Metrics: {best_result['dev_metrics']}")
    
    # 결과 저장
    logger.info("\n[7] 결과 저장")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "optimization_date": datetime.now().isoformat(),
        "best_combination": best_result,
        "all_results": results,
        "config_used": {
            "holding_days": cfg.get("l7", {}).get("holding_days", 20),
            "top_k": cfg.get("l7", {}).get("top_k", 20),
            "cost_bps": cfg.get("l7", {}).get("cost_bps", 10.0),
            "buffer_k": cfg.get("l7", {}).get("buffer_k", 20),
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Results saved to: {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("통합 최적화 완료!")
    logger.info("=" * 80)
    logger.info("\n다음 단계:")
    logger.info("1. 최적 가중치를 feature_groups_short.yaml, feature_groups_long.yaml에 반영")
    logger.info("2. 최적 α 값을 config.yaml의 l6r.regime_alpha에 반영")
    logger.info("3. 전체 파이프라인 재실행 (Holdout 성과 검증)")
    logger.info("4. docs/phase_status.md 및 관련 문서 업데이트")


if __name__ == "__main__":
    main()

