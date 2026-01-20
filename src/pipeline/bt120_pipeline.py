"""
BT120 전체 파이프라인 실행 모듈

BT120(120일 보유 기간) 전략의 전체 파이프라인을 실행합니다.
- 데이터 로딩 (캐시 우선)
- 피처 생성/로딩
- 모델 학습/로딩
- 랭킹 생성
- 백테스트 실행
"""

import logging
from pathlib import Path

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact

logger = logging.getLogger(__name__)


def run_bt120_pipeline(
    config_path: str = "configs/config.yaml",
    strategy: str = "long",  # "long" (분리) or "ens" (통합)
    force_rebuild: bool = False,
) -> dict:
    """
    BT120 전체 파이프라인을 실행합니다.

    Args:
        config_path: 설정 파일 경로
        strategy: "long" (장기 랭킹만) or "ens" (통합 랭킹)
        force_rebuild: True면 캐시 무시하고 재계산

    Returns:
        dict: 파이프라인 실행 결과
        {
            "rebalance_scores": DataFrame,
            "bt_positions": DataFrame,
            "bt_returns": DataFrame,
            "bt_equity_curve": DataFrame,
            "bt_metrics": DataFrame,
            "config": dict,
            "artifacts_path": dict,
        }
    """
    logger.info("=" * 80)
    logger.info("BT120 파이프라인 실행 시작")
    logger.info(f"전략: {strategy}")
    logger.info("=" * 80)

    # 설정 로드
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)

    # BT120 설정 선택
    if strategy == "long":
        l7_config_key = "l7_bt120_long"
    elif strategy == "ens":
        l7_config_key = "l7_bt120_ens"
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'long' or 'ens'")

    l7_cfg = cfg.get(l7_config_key, {})
    if not l7_cfg:
        raise ValueError(f"Config key '{l7_config_key}' not found in config.yaml")

    # 1. 중간 산출물 로드 (캐시 우선)
    artifacts = {}
    artifacts_path = {}

    # L0: 유니버스
    logger.info("[L0] 유니버스 로드")
    uni_path = interim_dir / "universe_k200_membership_monthly"
    if artifact_exists(uni_path) and not force_rebuild:
        artifacts["universe_k200_membership_monthly"] = load_artifact(uni_path)
        artifacts_path["universe"] = str(uni_path)
        logger.info(
            f"  ✓ 캐시에서 로드: {len(artifacts['universe_k200_membership_monthly']):,}행"
        )
    else:
        logger.warning("  ✗ 유니버스 데이터가 없습니다. L0부터 실행이 필요합니다.")
        logger.warning("  python scripts/run_pipeline_l0_l7.py 를 먼저 실행하세요.")
        raise FileNotFoundError("universe_k200_membership_monthly not found")

    # L3: 패널 데이터
    logger.info("[L3] 패널 데이터 로드")
    panel_path = interim_dir / "panel_merged_daily"
    if artifact_exists(panel_path) and not force_rebuild:
        artifacts["panel_merged_daily"] = load_artifact(panel_path)
        artifacts_path["panel"] = str(panel_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['panel_merged_daily']):,}행")
    else:
        logger.warning("  ✗ 패널 데이터가 없습니다. L0~L3까지 실행이 필요합니다.")
        raise FileNotFoundError("panel_merged_daily not found")

    # L4: CV 분할 및 타겟
    logger.info("[L4] CV 분할 데이터 로드")
    dataset_path = interim_dir / "dataset_daily"
    cv_short_path = interim_dir / "cv_folds_short"
    cv_long_path = interim_dir / "cv_folds_long"

    if (
        artifact_exists(dataset_path)
        and artifact_exists(cv_short_path)
        and artifact_exists(cv_long_path)
        and not force_rebuild
    ):
        artifacts["dataset_daily"] = load_artifact(dataset_path)
        artifacts["cv_folds_short"] = load_artifact(cv_short_path)
        artifacts["cv_folds_long"] = load_artifact(cv_long_path)
        artifacts_path["dataset"] = str(dataset_path)
        artifacts_path["cv_short"] = str(cv_short_path)
        artifacts_path["cv_long"] = str(cv_long_path)
        logger.info(f"  ✓ 캐시에서 로드: dataset {len(artifacts['dataset_daily']):,}행")
    else:
        logger.warning("  ✗ CV 분할 데이터가 없습니다. L4까지 실행이 필요합니다.")
        raise FileNotFoundError("dataset_daily or cv_folds not found")

    # L5: 모델 예측
    logger.info("[L5] 모델 예측 로드")
    pred_short_path = interim_dir / "pred_short_oos"
    pred_long_path = interim_dir / "pred_long_oos"

    if (
        artifact_exists(pred_short_path)
        and artifact_exists(pred_long_path)
        and not force_rebuild
    ):
        artifacts["pred_short_oos"] = load_artifact(pred_short_path)
        artifacts["pred_long_oos"] = load_artifact(pred_long_path)
        artifacts_path["pred_short"] = str(pred_short_path)
        artifacts_path["pred_long"] = str(pred_long_path)
        logger.info(
            f"  ✓ 캐시에서 로드: pred_long {len(artifacts['pred_long_oos']):,}행"
        )
    else:
        logger.warning("  ✗ 모델 예측이 없습니다. L5까지 실행이 필요합니다.")
        raise FileNotFoundError("pred_short_oos or pred_long_oos not found")

    # L6: 스코어 생성
    logger.info("[L6] 스코어 생성")
    scores_path = interim_dir / "rebalance_scores"

    if artifact_exists(scores_path) and not force_rebuild:
        artifacts["rebalance_scores"] = load_artifact(scores_path)
        artifacts_path["scores"] = str(scores_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['rebalance_scores']):,}행")
    else:
        # L6 재실행
        logger.info("  → L6 재실행")
        from src.stages.modeling.l6_scoring import build_rebalance_scores

        l6 = cfg.get("l6", {}) or {}
        w_s = float(l6.get("weight_short", 0.5))
        w_l = float(l6.get("weight_long", 0.5))

        scores, summary, quality, warns = build_rebalance_scores(
            pred_short_oos=artifacts["pred_short_oos"],
            pred_long_oos=artifacts["pred_long_oos"],
            universe_k200_membership_monthly=artifacts[
                "universe_k200_membership_monthly"
            ],
            weight_short=w_s,
            weight_long=w_l,
        )
        artifacts["rebalance_scores"] = scores
        save_artifact(scores, scores_path, force=True)
        artifacts_path["scores"] = str(scores_path)
        logger.info(f"  ✓ 생성 완료: {len(scores):,}행")

    # L7: 백테스트 실행
    logger.info("[L7] 백테스트 실행")
    from src.tracks.track_b.stages.backtest.l7_backtest import (
        BacktestConfig,
        run_backtest,
    )

    # BacktestConfig 생성
    # [개선안 37번] l7_bt120_* 설정 전달 누락(regime/overlapping/slippage/diversify) 보완
    l7_regime = (
        (l7_cfg.get("regime", {}) or {})
        if isinstance(l7_cfg.get("regime", {}), dict)
        else {}
    )
    l7_div = (
        (l7_cfg.get("diversify", {}) or {})
        if isinstance(l7_cfg.get("diversify", {}), dict)
        else {}
    )
    bt_cfg = BacktestConfig(
        holding_days=int(l7_cfg.get("holding_days", 120)),
        top_k=int(l7_cfg.get("top_k", 15)),
        cost_bps=float(l7_cfg.get("cost_bps", 10.0)),
        slippage_bps=float(l7_cfg.get("slippage_bps", 0.0)),
        score_col=str(
            l7_cfg.get(
                "score_col", "score_total_long" if strategy == "long" else "score_ens"
            )
        ),
        ret_col=str(l7_cfg.get("return_col", "true_long")),
        weighting=str(l7_cfg.get("weighting", "equal")),
        softmax_temp=float(
            l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp", 1.0))
        ),
        buffer_k=int(l7_cfg.get("buffer_k", 15)),
        rebalance_interval=int(l7_cfg.get("rebalance_interval", 6)),
        smart_buffer_enabled=bool(l7_cfg.get("smart_buffer_enabled", True)),
        smart_buffer_stability_threshold=float(
            l7_cfg.get("smart_buffer_stability_threshold", 0.7)
        ),
        volatility_adjustment_enabled=bool(
            l7_cfg.get("volatility_adjustment_enabled", True)
        ),
        volatility_lookback_days=int(l7_cfg.get("volatility_lookback_days", 60)),
        target_volatility=float(l7_cfg.get("target_volatility", 0.15)),
        volatility_adjustment_max=float(l7_cfg.get("volatility_adjustment_max", 1.2)),
        volatility_adjustment_min=float(l7_cfg.get("volatility_adjustment_min", 0.6)),
        risk_scaling_enabled=bool(l7_cfg.get("risk_scaling_enabled", True)),
        risk_scaling_bear_multiplier=float(
            l7_cfg.get("risk_scaling_bear_multiplier", 0.7)
        ),
        risk_scaling_neutral_multiplier=float(
            l7_cfg.get("risk_scaling_neutral_multiplier", 0.9)
        ),
        risk_scaling_bull_multiplier=float(
            l7_cfg.get("risk_scaling_bull_multiplier", 1.0)
        ),
        overlapping_tranches_enabled=bool(
            l7_cfg.get("overlapping_tranches_enabled", False)
        ),
        tranche_holding_days=int(l7_cfg.get("tranche_holding_days", 120)),
        tranche_max_active=int(l7_cfg.get("tranche_max_active", 4)),
        tranche_allocation_mode=str(
            l7_cfg.get("tranche_allocation_mode", "fixed_equal")
        ),
        diversify_enabled=bool(l7_div.get("enabled", False)),
        group_col=str(l7_div.get("group_col", "sector_name")),
        max_names_per_group=int(l7_div.get("max_names_per_group", 4)),
        regime_enabled=bool(l7_regime.get("enabled", False)),
        regime_top_k_bull_strong=(
            int(l7_regime["top_k_bull_strong"])
            if "top_k_bull_strong" in l7_regime
            else None
        ),
        regime_top_k_bull_weak=(
            int(l7_regime["top_k_bull_weak"])
            if "top_k_bull_weak" in l7_regime
            else None
        ),
        regime_top_k_bear_strong=(
            int(l7_regime["top_k_bear_strong"])
            if "top_k_bear_strong" in l7_regime
            else None
        ),
        regime_top_k_bear_weak=(
            int(l7_regime["top_k_bear_weak"])
            if "top_k_bear_weak" in l7_regime
            else None
        ),
        regime_top_k_neutral=(
            int(l7_regime["top_k_neutral"]) if "top_k_neutral" in l7_regime else None
        ),
        regime_exposure_bull_strong=(
            float(l7_regime["exposure_bull_strong"])
            if "exposure_bull_strong" in l7_regime
            else None
        ),
        regime_exposure_bull_weak=(
            float(l7_regime["exposure_bull_weak"])
            if "exposure_bull_weak" in l7_regime
            else None
        ),
        regime_exposure_bear_strong=(
            float(l7_regime["exposure_bear_strong"])
            if "exposure_bear_strong" in l7_regime
            else None
        ),
        regime_exposure_bear_weak=(
            float(l7_regime["exposure_bear_weak"])
            if "exposure_bear_weak" in l7_regime
            else None
        ),
        regime_exposure_neutral=(
            float(l7_regime["exposure_neutral"])
            if "exposure_neutral" in l7_regime
            else None
        ),
        regime_top_k_bull=(
            int(l7_regime["top_k_bull"]) if "top_k_bull" in l7_regime else None
        ),
        regime_top_k_bear=(
            int(l7_regime["top_k_bear"]) if "top_k_bear" in l7_regime else None
        ),
        regime_exposure_bull=(
            float(l7_regime["exposure_bull"]) if "exposure_bull" in l7_regime else None
        ),
        regime_exposure_bear=(
            float(l7_regime["exposure_bear"]) if "exposure_bear" in l7_regime else None
        ),
    )

    # 시장 국면 데이터 (regime_enabled일 때)
    market_regime_df = None
    if l7_cfg.get("regime", {}).get("enabled", False):
        logger.info("  → 시장 국면 데이터 생성")
        from src.tracks.shared.stages.regime.l1d_market_regime import (
            build_market_regime,
        )

        rebalance_dates = artifacts["rebalance_scores"]["date"].unique()
        start_date = cfg["params"]["start_date"]
        end_date = cfg["params"]["end_date"]
        regime_cfg = l7_cfg.get("regime", {})

        market_regime_df = build_market_regime(
            rebalance_dates=rebalance_dates,
            start_date=start_date,
            end_date=end_date,
            lookback_days=regime_cfg.get("lookback_days", 60),
            threshold_pct=regime_cfg.get("threshold_pct", 0.0),
        )
        logger.info(f"  ✓ 시장 국면 데이터 생성: {len(market_regime_df):,}행")

    # 백테스트 실행
    result = run_backtest(
        rebalance_scores=artifacts["rebalance_scores"],
        cfg=bt_cfg,
        config_cost_bps=float(l7_cfg.get("cost_bps", 10.0)),
        market_regime=market_regime_df,
    )

    # 결과 처리
    if len(result) == 10:
        (
            bt_pos,
            bt_ret,
            bt_eq,
            bt_met,
            quality,
            warns,
            selection_diagnostics,
            bt_returns_diagnostics,
            runtime_profile,
            bt_regime_metrics,
        ) = result
    elif len(result) == 9:
        (
            bt_pos,
            bt_ret,
            bt_eq,
            bt_met,
            quality,
            warns,
            selection_diagnostics,
            bt_returns_diagnostics,
            runtime_profile,
        ) = result
        bt_regime_metrics = None
    elif len(result) == 6:
        bt_pos, bt_ret, bt_eq, bt_met, quality, warns = result
        selection_diagnostics = None
        bt_returns_diagnostics = None
        runtime_profile = None
        bt_regime_metrics = None
    else:
        raise ValueError(f"Unexpected return value count: {len(result)}")

    # 결과 저장
    suffix = "_bt120"
    save_artifact(bt_pos, interim_dir / f"bt_positions{suffix}", force=True)
    save_artifact(bt_ret, interim_dir / f"bt_returns{suffix}", force=True)
    save_artifact(bt_eq, interim_dir / f"bt_equity_curve{suffix}", force=True)
    save_artifact(bt_met, interim_dir / f"bt_metrics{suffix}", force=True)

    artifacts_path["bt_positions"] = str(interim_dir / f"bt_positions{suffix}")
    artifacts_path["bt_returns"] = str(interim_dir / f"bt_returns{suffix}")
    artifacts_path["bt_equity_curve"] = str(interim_dir / f"bt_equity_curve{suffix}")
    artifacts_path["bt_metrics"] = str(interim_dir / f"bt_metrics{suffix}")

    logger.info("=" * 80)
    logger.info("✅ BT120 파이프라인 실행 완료")
    logger.info("=" * 80)

    return {
        "rebalance_scores": artifacts["rebalance_scores"],
        "bt_positions": bt_pos,
        "bt_returns": bt_ret,
        "bt_equity_curve": bt_eq,
        "bt_metrics": bt_met,
        "config": l7_cfg,
        "artifacts_path": artifacts_path,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    strategy = sys.argv[1] if len(sys.argv) > 1 else "long"
    result = run_bt120_pipeline(strategy=strategy)
    print(f"\n✅ 완료: {len(result['bt_metrics'])}개 메트릭 생성")
