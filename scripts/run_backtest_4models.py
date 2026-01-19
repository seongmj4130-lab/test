# -*- coding: utf-8 -*-
"""
4개 모델 백테스트 실행 스크립트

올바른 설정 (2026-01-07 수정):
- bt20_ens: holding_days=20, rebalance_interval=20 (단기 본질 유지)
- bt20_short: holding_days=20, rebalance_interval=20 (단기 본질 유지)
- bt120_ens: holding_days=20, rebalance_interval=20 (트랜치 추가 주기, 월별)
  - overlapping_tranches_enabled=true, tranche_holding_days=120, tranche_max_active=4
- bt120_long: holding_days=20, rebalance_interval=20 (트랜치 추가 주기, 월별)
  - overlapping_tranches_enabled=true, tranche_holding_days=120, tranche_max_active=4

⚠️ 중요: rebalance_interval=1이면 안 됨
- bt20_short: 20일 모멘텀 → 월 모멘텀으로 변질
- bt120_long: 트랜치 효과 소실 (매월 완전 교체)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact


def run_backtest_for_strategy(
    cfg: dict,
    strategy_config: dict,
    rebalance_scores: pd.DataFrame,
) -> pd.DataFrame:
    """특정 전략에 대한 백테스트 실행"""
    strategy_name = strategy_config['name']
    config_section = cfg.get(strategy_config['config_section'], {})

    logger.info(f"[L7] {strategy_name} 백테스트 실행")
    logger.info(f"  holding_days={config_section.get('holding_days')}, rebalance_interval={config_section.get('rebalance_interval')}")

    # L7: 백테스트
    # [수정] rebalance_interval 설정 추가 (기본값 1이 아닌 config에서 읽기)
    rebalance_interval = int(config_section.get("rebalance_interval", 1))

    # [수정] return_col도 config에서 읽기 (BT120은 true_short 사용)
    return_col = config_section.get("return_col", strategy_config['return_col'])

    # [수정] 오버래핑 트랜치 설정 (BT120 전략용)
    overlapping_tranches_enabled = config_section.get("overlapping_tranches_enabled", False)
    tranche_holding_days = int(config_section.get("tranche_holding_days", 120))
    tranche_max_active = int(config_section.get("tranche_max_active", 4))
    tranche_allocation_mode = config_section.get("tranche_allocation_mode", "fixed_equal")

    bt_cfg = BacktestConfig(
        holding_days=int(config_section.get("holding_days", 20)),
        top_k=int(config_section.get("top_k", 20)),
        cost_bps=float(config_section.get("cost_bps", 10.0)),
        score_col=strategy_config['score_col'],
        ret_col=return_col,  # [수정] config에서 읽은 값 사용
        weighting=config_section.get("weighting", "equal"),
        softmax_temp=float(config_section.get("softmax_temperature", 1.0)),
        buffer_k=int(config_section.get("buffer_k", 0)),
        rebalance_interval=rebalance_interval,  # [수정] rebalance_interval 추가
        overlapping_tranches_enabled=overlapping_tranches_enabled,  # [수정] 오버래핑 트랜치 설정
        tranche_holding_days=tranche_holding_days,
        tranche_max_active=tranche_max_active,
        tranche_allocation_mode=tranche_allocation_mode,
    )

    result = run_backtest(
        rebalance_scores=rebalance_scores,
        cfg=bt_cfg,
    )

    if len(result) >= 4:
        bt_metrics = result[3]
        bt_metrics['strategy'] = strategy_name
        return bt_metrics
    else:
        raise ValueError(f"Unexpected result length: {len(result)}")


def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info("[4개 모델 백테스트 실행]")
    logger.info("holding_days = rebalance_interval 설정")
    logger.info("=" * 80)
    logger.info(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 설정 로드
    cfg = load_config("configs/config.yaml")
    interim_dir = Path(get_path(cfg, "data_interim"))
    artifacts_dir = Path(get_path(cfg, "artifacts_reports"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    logger.info("[데이터 로드]")
    dataset_daily = load_artifact(interim_dir / "dataset_daily.parquet")
    cv_folds_short = load_artifact(interim_dir / "cv_folds_short.parquet")
    cv_folds_long = load_artifact(interim_dir / "cv_folds_long.parquet")
    universe_monthly = load_artifact(interim_dir / "universe_k200_membership_monthly.parquet")

    # L5: 모델 학습 (필요한 경우)
    pred_short_file = interim_dir / "pred_short_oos.parquet"
    pred_long_file = interim_dir / "pred_long_oos.parquet"

    if not artifact_exists(pred_short_file) or not artifact_exists(pred_long_file):
        logger.info("[L5] 모델 학습 시작 (예측 결과 파일이 없음)")

        l4 = cfg.get("l4", {}) or {}
        hs = int(l4.get("horizon_short", 20))
        hl = int(l4.get("horizon_long", 120))

        target_s = f"ret_fwd_{hs}d"
        target_l = f"ret_fwd_{hl}d"

        logger.info(f"[L5] Short 예측 학습 중... (target: {target_s})")
        pred_s, met_s, rep_s, w_s = train_oos_predictions(
            dataset_daily=dataset_daily,
            cv_folds=cv_folds_short,
            cfg=cfg,
            target_col=target_s,
            horizon=hs,
        )

        logger.info(f"[L5] Long 예측 학습 중... (target: {target_l})")
        pred_l, met_l, rep_l, w_l = train_oos_predictions(
            dataset_daily=dataset_daily,
            cv_folds=cv_folds_long,
            cfg=cfg,
            target_col=target_l,
            horizon=hl,
        )

        # 저장
        save_artifact(pred_s, pred_short_file)
        save_artifact(pred_l, pred_long_file)
        logger.info(f"[L5] 완료: pred_short_oos {len(pred_s):,}행, pred_long_oos {len(pred_l):,}행")
    else:
        logger.info("[L5] 기존 예측 결과 사용")
        pred_s = load_artifact(pred_short_file)
        pred_l = load_artifact(pred_long_file)

    # L6: 스코어 생성
    logger.info("[L6] 스코어 생성")
    l6_cfg = cfg.get("l6", {}) or {}
    rebalance_scores, summary, quality, warns = build_rebalance_scores(
        pred_short_oos=pred_s,
        pred_long_oos=pred_l,
        universe_k200_membership_monthly=universe_monthly,
        weight_short=float(l6_cfg.get("weight_short", 0.5)),
        weight_long=float(l6_cfg.get("weight_long", 0.5)),
    )
    logger.info(f"[L6] 완료: {len(rebalance_scores):,}행")

    # 4개 모델 백테스트 실행
    strategies = [
        {
            'name': 'bt20_ens',
            'config_section': 'l7_bt20_ens',
            'score_col': 'score_ens',
            'return_col': 'true_short',
        },
        {
            'name': 'bt20_short',
            'config_section': 'l7_bt20_short',
            'score_col': 'score_total_short',
            'return_col': 'true_short',
        },
        {
            'name': 'bt120_ens',
            'config_section': 'l7_bt120_ens',
            'score_col': 'score_ens',
            'return_col': 'true_short',  # [수정] 오버래핑 트랜치: 월별 PnL(20일 fwd)로 계산
        },
        {
            'name': 'bt120_long',
            'config_section': 'l7_bt120_long',
            'score_col': 'score_total_long',
            'return_col': 'true_short',  # [수정] 오버래핑 트랜치: 월별 PnL(20일 fwd)로 계산
        },
    ]

    all_results = {}

    for strategy in strategies:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"[{strategy['name']}] 시작")
            logger.info(f"{'='*80}")

            bt_metrics = run_backtest_for_strategy(
                cfg=cfg,
                strategy_config=strategy,
                rebalance_scores=rebalance_scores,
            )

            # 결과 저장
            output_file = interim_dir / f"bt_metrics_{strategy['name']}.parquet"
            save_artifact(bt_metrics, output_file)
            logger.info(f"[저장] {output_file}")

            all_results[strategy['name']] = bt_metrics

            # Holdout 구간 메트릭 출력
            holdout = bt_metrics[bt_metrics['phase'] == 'holdout']
            if len(holdout) > 0:
                logger.info(f"[{strategy['name']}] Holdout 성과:")
                logger.info(f"  Sharpe: {holdout['net_sharpe'].iloc[0]:.4f}")
                logger.info(f"  CAGR: {holdout['net_cagr'].iloc[0]:.4%}")
                logger.info(f"  MDD: {holdout['net_mdd'].iloc[0]:.4%}")
                logger.info(f"  Calmar: {holdout['net_calmar_ratio'].iloc[0]:.4f}")

        except Exception as e:
            logger.error(f"[{strategy['name']}] 백테스트 실패: {e}", exc_info=True)
            continue

    # 비교 리포트 생성
    logger.info("\n" + "=" * 80)
    logger.info("[최종 결과 비교]")
    logger.info("=" * 80)

    comparison_rows = []
    for strategy_name, bt_metrics in all_results.items():
        holdout = bt_metrics[bt_metrics['phase'] == 'holdout']
        if len(holdout) == 0:
            continue

        row = {
            'strategy': strategy_name,
            'holding_days': holdout['holding_days'].iloc[0] if 'holding_days' in holdout.columns else None,
            'net_sharpe': holdout['net_sharpe'].iloc[0],
            'net_cagr': holdout['net_cagr'].iloc[0],
            'net_mdd': holdout['net_mdd'].iloc[0],
            'net_calmar_ratio': holdout['net_calmar_ratio'].iloc[0],
        }
        comparison_rows.append(row)

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_file = artifacts_dir / "backtest_4models_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        logger.info(f"\n[비교 리포트] {comparison_file}")
        logger.info("\n" + comparison_df.to_string())

    logger.info("\n" + "=" * 80)
    logger.info("✅ 4개 모델 백테스트 완료")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
