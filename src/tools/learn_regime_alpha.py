"""
[Phase 2] 국면별 α 자동 학습 스크립트

목적: 국면(bull/neutral/bear)별로 듀얼호라이즌 결합 가중치 α를 데이터로부터 자동 학습

주의사항:
- ⚠️ 누수 방지: α 학습은 Dev만 사용, Holdout 절대 금지
- ⚠️ 동일 기준 비교: 동일 cost_bps, holding_days, 리밸런싱 규칙 유지
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.stages.backtest.regime_utils import map_regime_to_3level
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_required_artifacts(cfg: dict) -> dict:
    """필수 산출물 로드"""
    interim_dir = get_path(cfg, "data_interim")

    artifacts = {}
    required = [
        "dataset_daily",
        "cv_folds_short",
        "universe_k200_membership_monthly",
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
        return regime_df

    # 없으면 생성
    logger.info("Generating market_regime_daily...")
    from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime

    start_date = cfg["params"]["start_date"]
    end_date = cfg["params"]["end_date"]

    regime_df = build_market_regime(
        rebalance_dates=rebalance_dates,
        start_date=start_date,
        end_date=end_date,
        lookback_days=60,
        threshold_pct=0.0,
    )

    # 3단계 국면 컬럼 추가
    if "regime_3" not in regime_df.columns:
        from src.stages.backtest.regime_utils import apply_3level_regime

        regime_df = apply_3level_regime(
            regime_df, regime_col="regime", out_col="regime_3"
        )

    logger.info(f"Generated market_regime_daily: {len(regime_df):,} rows")
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


def evaluate_alpha_combination(
    cfg: dict,
    artifacts: dict,
    regime_alpha: dict[str, float],
    dev_dates: pd.Series,
    market_regime_df: pd.DataFrame,
) -> tuple[float, dict]:
    """
    특정 α 조합에 대한 Dev 성과 평가

    Returns:
        (net_sharpe, metrics_dict)
    """
    # L6R 실행을 위한 cfg 복사 및 regime_alpha 설정
    cfg_test = cfg.copy()
    if "l6r" not in cfg_test:
        cfg_test["l6r"] = {}
    cfg_test["l6r"]["regime_alpha"] = regime_alpha

    # L6R 실행 (임시 cfg 사용)
    from src.tracks.track_b.stages.modeling.l6r_ranking_scoring import (
        run_L6R_ranking_scoring,
    )

    try:
        l6r_outputs, l6r_warns = run_L6R_ranking_scoring(
            cfg=cfg_test,
            artifacts=artifacts,
            force=False,
        )
        rebalance_scores = l6r_outputs["rebalance_scores"]

        # Dev만 필터링
        rebalance_scores["date"] = pd.to_datetime(rebalance_scores["date"])
        dev_scores = rebalance_scores[
            (rebalance_scores["phase"] == "dev")
            & (rebalance_scores["date"].isin(dev_dates))
        ].copy()

        if len(dev_scores) == 0:
            logger.warning("No dev scores found for this alpha combination")
            return -999.0, {}

        # L7 백테스트 실행 (Dev만)
        from src.tracks.track_b.stages.backtest.l7_backtest import (
            BacktestConfig,
            run_backtest,
        )

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
            market_regime=market_regime_df,
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
        logger.error(
            f"Error evaluating alpha combination {regime_alpha}: {e}", exc_info=True
        )
        return -999.0, {}


def grid_search_regime_alpha(
    cfg: dict,
    artifacts: dict,
    dev_dates: pd.Series,
    market_regime_df: pd.DataFrame,
    alpha_candidates: list[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
) -> dict[str, float]:
    """
    그리드 서치를 통한 국면별 최적 α 학습

    Returns:
        최적 regime_alpha 딕셔너리: {"bull": 0.7, "neutral": 0.5, "bear": 0.3}
    """
    logger.info("=" * 80)
    logger.info("국면별 α 그리드 서치 시작")
    logger.info("=" * 80)
    logger.info(f"Alpha candidates: {alpha_candidates}")

    # 국면별 날짜 분류
    market_regime_df = market_regime_df.copy()
    market_regime_df["date"] = pd.to_datetime(market_regime_df["date"])

    # 3단계 국면 컬럼 확인
    regime_col = "regime_3" if "regime_3" in market_regime_df.columns else "regime"
    if regime_col == "regime":
        # 3단계로 변환
        market_regime_df["regime_3"] = market_regime_df["regime"].apply(
            map_regime_to_3level
        )
        regime_col = "regime_3"

    # Dev 날짜에 해당하는 국면 매핑
    regime_map = market_regime_df.set_index("date")[regime_col].to_dict()
    dev_regimes = {}
    for date in dev_dates:
        regime = regime_map.get(date, "neutral")
        dev_regimes[date] = regime

    regime_counts = pd.Series(dev_regimes.values()).value_counts()
    logger.info("\nDev 국면 분포:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} dates ({count/len(dev_dates)*100:.1f}%)")

    # 그리드 서치: 각 국면별로 독립적으로 최적화
    optimal_alphas = {}
    results_summary = []

    regimes = ["bull", "neutral", "bear"]

    for regime in regimes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Regime: {regime}")
        logger.info(f"{'='*80}")

        regime_dates = [d for d, r in dev_regimes.items() if r == regime]

        if len(regime_dates) == 0:
            logger.warning(f"No dates for regime {regime}, using default alpha 0.5")
            optimal_alphas[regime] = 0.5
            continue

        logger.info(
            f"Evaluating {len(alpha_candidates)} alpha values for {len(regime_dates)} dates..."
        )

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
                dev_dates=pd.Series(regime_dates),
                market_regime_df=market_regime_df,
            )

            logger.info(f"    Sharpe: {net_sharpe:.4f}")
            results_summary.append(
                {
                    "regime": regime,
                    "alpha": alpha,
                    "net_sharpe": net_sharpe,
                    **metrics,
                }
            )

            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_alpha = alpha
                best_metrics = metrics

        optimal_alphas[regime] = best_alpha
        logger.info(
            f"\n  Best alpha for {regime}: {best_alpha:.1f} (Sharpe: {best_sharpe:.4f})"
        )
        logger.info(f"  Metrics: {best_metrics}")

    logger.info(f"\n{'='*80}")
    logger.info("최적 α 학습 완료")
    logger.info(f"{'='*80}")
    logger.info("Optimal regime_alpha:")
    for regime, alpha in optimal_alphas.items():
        logger.info(f"  {regime}: {alpha:.1f}")

    return optimal_alphas, results_summary


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="국면별 α 자동 학습")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--output", type=str, default="docs/regime_alpha_learning_results.json"
    )
    args = parser.parse_args()

    # 설정 로드
    cfg = load_config(args.config)
    interim_dir = get_path(cfg, "data_interim")

    logger.info("=" * 80)
    logger.info("Phase 2: 국면별 α 자동 학습")
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
    import json

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "optimal_regime_alpha": optimal_alphas,
        "results_summary": results_summary,
        "config_used": {
            "holding_days": cfg.get("l7", {}).get("holding_days", 20),
            "top_k": cfg.get("l7", {}).get("top_k", 20),
            "cost_bps": cfg.get("l7", {}).get("cost_bps", 10.0),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
