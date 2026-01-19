# -*- coding: utf-8 -*-
"""
L5~L7 파이프라인 실행 스크립트 (재현성 검증용)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging

from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import load_artifact, save_artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_L5_train_models(cfg, artifacts, *, force=False):
    """L5: 앙상블 모델 학습"""
    logger.info("[L5] 시작")

    # 기존 데이터 로드
    interim_dir = get_path(cfg, "data_interim")
    dataset_daily = load_artifact(interim_dir / "dataset_daily.parquet")
    targets_and_folds = load_artifact(interim_dir / "targets_and_folds.parquet")

    # 모델 학습
    predictions_df = train_oos_predictions(
        dataset_daily=dataset_daily,
        targets_and_folds=targets_and_folds,
        cfg=cfg,
        force=force
    )

    logger.info(f"[L5] 완료: {len(predictions_df):,}행")
    return {"predictions_oos": predictions_df}, []

def run_L6_scoring(cfg, artifacts, *, force=False):
    """L6: 리밸런싱 스코어 생성"""
    logger.info("[L6] 시작")

    interim_dir = get_path(cfg, "data_interim")
    dataset_daily = load_artifact(interim_dir / "dataset_daily.parquet")
    predictions_df = artifacts["predictions_oos"]

    # 스코어 생성
    scores_df = build_rebalance_scores(
        dataset_daily=dataset_daily,
        predictions_df=predictions_df,
        cfg=cfg
    )

    logger.info(f"[L6] 완료: {len(scores_df):,}행")
    return {"scores_daily": scores_df}, []

def run_L7_backtest(cfg, artifacts, *, force=False):
    """L7: 백테스트 실행"""
    logger.info("[L7] 시작")

    interim_dir = get_path(cfg, "data_interim")
    scores_df = artifacts["scores_daily"]
    targets_and_folds = load_artifact(interim_dir / "targets_and_folds.parquet")
    dataset_daily = load_artifact(interim_dir / "dataset_daily.parquet")

    # 백테스트 설정
    bt_cfg = BacktestConfig(
        holding_days=cfg.get("l7", {}).get("holding_days", 20),
        top_k=cfg.get("l7", {}).get("top_k", 20),
        cost_bps=cfg.get("l7", {}).get("cost_bps", 0.0),
        score_col=cfg.get("l7", {}).get("score_col", "score_ens"),
        return_col=cfg.get("l7", {}).get("return_col", "true_short"),
        rebalance_interval=cfg.get("l7", {}).get("rebalance_interval", 20),
        smart_buffer_enabled=cfg.get("l7", {}).get("smart_buffer_enabled", True),
        volatility_adjustment_enabled=cfg.get("l7", {}).get("volatility_adjustment_enabled", True),
        volatility_lookback_days=cfg.get("l7", {}).get("volatility_lookback_days", 60),
    )

    # 백테스트 실행
    bt_results = run_backtest(
        scores_df=scores_df,
        targets_and_folds=targets_and_folds,
        dataset_daily=dataset_daily,
        bt_cfg=bt_cfg
    )

    logger.info(f"[L7] 완료: {len(bt_results):,}개 전략")
    return {"backtest_results": bt_results}, []

def main():
    """L5~L7 파이프라인 실행"""
    cfg = load_config('configs/config.yaml')

    print("🚀 L5~L7 파이프라인 실행 (재현성 검증)")
    print("="*60)

    artifacts = {}

    try:
        # L5: 모델 학습
        artifacts_l5, warnings_l5 = run_L5_train_models(cfg, artifacts, force=True)
        artifacts.update(artifacts_l5)

        # L6: 스코어 생성
        artifacts_l6, warnings_l6 = run_L6_scoring(cfg, artifacts, force=True)
        artifacts.update(artifacts_l6)

        # L7: 백테스트
        artifacts_l7, warnings_l7 = run_L7_backtest(cfg, artifacts, force=True)
        artifacts.update(artifacts_l7)

        print("✅ L5~L7 파이프라인 완료!")
        return True

    except Exception as e:
        print(f"❌ L5~L7 파이프라인 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
