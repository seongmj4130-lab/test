"""
Track B: 투자 모델 서비스 모듈

UI에서 import 가능한 형태로 백테스트 실행 함수를 제공합니다.

[리팩토링 2단계] 함수/모듈화 - UI에서 import 가능한 형태
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src.tracks.shared.data_pipeline import prepare_common_data
from src.tracks.track_a.ranking_service import generate_rankings
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.tracks.track_b.stages.modeling.l6r_ranking_scoring import (
    run_L6R_ranking_scoring,
)
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact

logger = logging.getLogger(__name__)


def run_backtest_strategy(
    strategy: Literal["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"],
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    백테스트 전략을 실행하는 함수 (Track B 핵심 기능).

    [리팩토링 2단계] UI에서 import 가능한 형태로 모듈화

    Args:
        strategy: 전략 이름
            - "bt20_short": BT20 단기 랭킹
            - "bt20_ens": BT20 통합 랭킹
            - "bt120_long": BT120 장기 랭킹
            - "bt120_ens": BT120 통합 랭킹
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재계산

    Returns:
        dict: 백테스트 결과
        {
            "bt_positions": DataFrame,
            "bt_returns": DataFrame,
            "bt_equity_curve": DataFrame,
            "bt_metrics": DataFrame,
        }

    Example:
        >>> from src.tracks.track_b.backtest_service import run_backtest_strategy
        >>> results = run_backtest_strategy("bt20_short")
        >>> metrics = results["bt_metrics"]
    """
    logger.info(f"백테스트 전략 실행 시작: {strategy}")

    # 설정 로드
    cfg = load_config(config_path)

    # 전략별 설정 매핑
    strategy_config_map = {
        "bt20_short": "l7_bt20_short",
        "bt20_ens": "l7_bt20_ens",
        "bt120_long": "l7_bt120_long",
        "bt120_ens": "l7_bt120_ens",
    }

    l7_config_key = strategy_config_map.get(strategy)
    if not l7_config_key:
        raise ValueError(
            f"Unknown strategy: {strategy}. Use one of {list(strategy_config_map.keys())}"
        )

    l7_cfg = cfg.get(l7_config_key, {})
    if not l7_cfg:
        raise ValueError(f"Config key '{l7_config_key}' not found in config.yaml")

    # [개선안 24번] 전략별 rebalance_interval 반영 (bt20=20, bt120=120 등)
    # - L6R은 cfg['l7']['rebalance_interval']을 참조하므로, 전략별 설정으로 override한 cfg를 사용
    strategy_rebalance_interval = int(l7_cfg.get("rebalance_interval", 1) or 1)
    cfg_for_l6r = dict(cfg)
    cfg_for_l6r["l7"] = dict((cfg.get("l7", {}) if isinstance(cfg, dict) else {}) or {})
    cfg_for_l6r["l7"]["rebalance_interval"] = strategy_rebalance_interval

    # 공통 데이터 준비
    artifacts = prepare_common_data(
        config_path=config_path, force_rebuild=force_rebuild
    )

    # Track A 산출물 확인 (랭킹 데이터)
    interim_dir = Path(get_path(cfg, "data_interim"))
    ranking_short_path = interim_dir / "ranking_short_daily"
    ranking_long_path = interim_dir / "ranking_long_daily"

    if (
        not artifact_exists(ranking_short_path)
        or not artifact_exists(ranking_long_path)
        or force_rebuild
    ):
        logger.info("랭킹 데이터가 없습니다. Track A를 실행합니다...")
        rankings = generate_rankings(
            config_path=config_path, force_rebuild=force_rebuild
        )
        artifacts["ranking_short_daily"] = rankings["ranking_short_daily"]
        artifacts["ranking_long_daily"] = rankings["ranking_long_daily"]
    else:
        artifacts["ranking_short_daily"] = load_artifact(ranking_short_path)
        artifacts["ranking_long_daily"] = load_artifact(ranking_long_path)

    # L6R: 랭킹 스코어 변환
    logger.info("[L6R] 랭킹 스코어 변환")
    # [개선안 24번] interval별 캐시 분리 (bt120이 bt20 캐시를 공유하는 문제 방지)
    if strategy_rebalance_interval and strategy_rebalance_interval > 1:
        scores_path = (
            interim_dir
            / f"rebalance_scores_from_ranking_interval_{strategy_rebalance_interval}"
        )
    else:
        scores_path = interim_dir / "rebalance_scores_from_ranking"

    if artifact_exists(scores_path) and not force_rebuild:
        artifacts["rebalance_scores"] = load_artifact(scores_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['rebalance_scores']):,}행")
    else:
        logger.info("  → L6R 재실행")
        outputs, warns = run_L6R_ranking_scoring(
            cfg=cfg_for_l6r,
            artifacts={
                "ranking_short_daily": artifacts["ranking_short_daily"],
                "ranking_long_daily": artifacts["ranking_long_daily"],
                "dataset_daily": artifacts["dataset_daily"],
                "cv_folds_short": artifacts["cv_folds_short"],
                "universe_k200_membership_monthly": artifacts[
                    "universe_k200_membership_monthly"
                ],
                "ohlcv_daily": artifacts.get("ohlcv_daily"),  # 국면/interval 계산용
            },
            force=force_rebuild,
        )
        artifacts["rebalance_scores"] = outputs["rebalance_scores"]
        save_artifact(artifacts["rebalance_scores"], scores_path, force=True)
        logger.info(f"  ✓ 생성 완료: {len(artifacts['rebalance_scores']):,}행")

    # L7: 백테스트 실행
    logger.info("[L7] 백테스트 실행")
    regime_cfg = (
        l7_cfg.get("regime", {}) if isinstance(l7_cfg.get("regime", {}), dict) else {}
    )
    diversify_cfg = (
        l7_cfg.get("diversify", {})
        if isinstance(l7_cfg.get("diversify", {}), dict)
        else {}
    )
    bt_cfg = BacktestConfig(
        holding_days=int(l7_cfg.get("holding_days", 20)),
        top_k=int(l7_cfg.get("top_k", 12)),
        cost_bps=float(l7_cfg.get("cost_bps", 10.0)),
        slippage_bps=float(
            l7_cfg.get("slippage_bps", 0.0)
        ),  # [개선안 3번] 슬리피지 비용
        score_col=str(l7_cfg.get("score_col", "score_ens")),
        ret_col=str(l7_cfg.get("return_col", "true_short")),
        weighting=str(l7_cfg.get("weighting", "equal")),
        softmax_temp=float(
            l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp", 1.0))
        ),
        buffer_k=int(l7_cfg.get("buffer_k", 15)),
        rebalance_interval=int(l7_cfg.get("rebalance_interval", 1)),
        # [개선안 36번] 오버래핑 트랜치 옵션 매핑(필수 기능)
        overlapping_tranches_enabled=bool(
            l7_cfg.get("overlapping_tranches_enabled", False)
        ),
        tranche_holding_days=int(l7_cfg.get("tranche_holding_days", 120)),
        tranche_max_active=int(l7_cfg.get("tranche_max_active", 4)),
        tranche_allocation_mode=str(
            l7_cfg.get("tranche_allocation_mode", "fixed_equal")
        ),
        # [개선안 25번] config.yaml 옵션이 실제 백테스트에 반영되도록 매핑 보강
        diversify_enabled=bool(diversify_cfg.get("enabled", False)),
        group_col=str(diversify_cfg.get("group_col", "sector_name")),
        max_names_per_group=int(diversify_cfg.get("max_names_per_group", 4)),
        regime_enabled=bool(regime_cfg.get("enabled", False)),
        regime_top_k_bull_strong=(
            int(regime_cfg["top_k_bull_strong"])
            if "top_k_bull_strong" in regime_cfg
            and regime_cfg["top_k_bull_strong"] is not None
            else None
        ),
        regime_top_k_bull_weak=(
            int(regime_cfg["top_k_bull_weak"])
            if "top_k_bull_weak" in regime_cfg
            and regime_cfg["top_k_bull_weak"] is not None
            else None
        ),
        regime_top_k_bear_strong=(
            int(regime_cfg["top_k_bear_strong"])
            if "top_k_bear_strong" in regime_cfg
            and regime_cfg["top_k_bear_strong"] is not None
            else None
        ),
        regime_top_k_bear_weak=(
            int(regime_cfg["top_k_bear_weak"])
            if "top_k_bear_weak" in regime_cfg
            and regime_cfg["top_k_bear_weak"] is not None
            else None
        ),
        regime_top_k_neutral=(
            int(regime_cfg["top_k_neutral"])
            if "top_k_neutral" in regime_cfg and regime_cfg["top_k_neutral"] is not None
            else None
        ),
        regime_exposure_bull_strong=(
            float(regime_cfg["exposure_bull_strong"])
            if "exposure_bull_strong" in regime_cfg
            and regime_cfg["exposure_bull_strong"] is not None
            else None
        ),
        regime_exposure_bull_weak=(
            float(regime_cfg["exposure_bull_weak"])
            if "exposure_bull_weak" in regime_cfg
            and regime_cfg["exposure_bull_weak"] is not None
            else None
        ),
        regime_exposure_bear_strong=(
            float(regime_cfg["exposure_bear_strong"])
            if "exposure_bear_strong" in regime_cfg
            and regime_cfg["exposure_bear_strong"] is not None
            else None
        ),
        regime_exposure_bear_weak=(
            float(regime_cfg["exposure_bear_weak"])
            if "exposure_bear_weak" in regime_cfg
            and regime_cfg["exposure_bear_weak"] is not None
            else None
        ),
        regime_exposure_neutral=(
            float(regime_cfg["exposure_neutral"])
            if "exposure_neutral" in regime_cfg
            and regime_cfg["exposure_neutral"] is not None
            else None
        ),
        regime_top_k_bull=(
            int(regime_cfg["top_k_bull"])
            if "top_k_bull" in regime_cfg and regime_cfg["top_k_bull"] is not None
            else None
        ),
        regime_top_k_bear=(
            int(regime_cfg["top_k_bear"])
            if "top_k_bear" in regime_cfg and regime_cfg["top_k_bear"] is not None
            else None
        ),
        regime_exposure_bull=(
            float(regime_cfg["exposure_bull"])
            if "exposure_bull" in regime_cfg and regime_cfg["exposure_bull"] is not None
            else None
        ),
        regime_exposure_bear=(
            float(regime_cfg["exposure_bear"])
            if "exposure_bear" in regime_cfg and regime_cfg["exposure_bear"] is not None
            else None
        ),
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
        volatility_adjustment_min=float(l7_cfg.get("volatility_adjustment_min", 0.7)),
        risk_scaling_enabled=bool(l7_cfg.get("risk_scaling_enabled", True)),
        risk_scaling_bear_multiplier=float(
            l7_cfg.get("risk_scaling_bear_multiplier", 0.8)
        ),
        risk_scaling_neutral_multiplier=float(
            l7_cfg.get("risk_scaling_neutral_multiplier", 1.0)
        ),
        risk_scaling_bull_multiplier=float(
            l7_cfg.get("risk_scaling_bull_multiplier", 1.0)
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
        regime_cfg = l7_cfg.get("regime", {})

        # [개선안 23번] build_market_regime 시그니처 호환: start_date/end_date 제거, ohlcv_daily 기반으로 생성
        ohlcv_daily = artifacts.get("ohlcv_daily")
        if ohlcv_daily is None or (
            hasattr(ohlcv_daily, "__len__") and len(ohlcv_daily) == 0
        ):
            logger.warning(
                "  ⚠️ ohlcv_daily가 없어 시장 국면 생성을 건너뜁니다. (regime 비활성)"
            )
            market_regime_df = None
        else:
            # [개선안 27번] neutral_band 보정(전략별 의도 존중)
            # - neutral_band가 명시되면(0.0 포함) 그대로 사용 (bt20에서 의도적으로 neutral 제거 가능)
            # - neutral_band가 없고 threshold_pct=0.0이면 과민 분류(bull/bear 쏠림) 방지를 위해 0.05로 보정
            nb_raw = regime_cfg.get("neutral_band", None)
            if nb_raw is None:
                neutral_band = float(regime_cfg.get("threshold_pct", 0.05))
                if neutral_band <= 0:
                    neutral_band = 0.05
            else:
                neutral_band = float(nb_raw)
            market_regime_df = build_market_regime(
                rebalance_dates=rebalance_dates,
                ohlcv_daily=ohlcv_daily,
                lookback_days=int(regime_cfg.get("lookback_days", 60)),
                neutral_band=neutral_band,
                use_volume=bool(regime_cfg.get("use_volume", True)),
                use_volatility=bool(regime_cfg.get("use_volatility", True)),
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
    suffix = f"_{strategy}"
    save_artifact(bt_pos, interim_dir / f"bt_positions{suffix}", force=True)
    save_artifact(bt_ret, interim_dir / f"bt_returns{suffix}", force=True)
    save_artifact(bt_eq, interim_dir / f"bt_equity_curve{suffix}", force=True)
    save_artifact(bt_met, interim_dir / f"bt_metrics{suffix}", force=True)

    # [개선안 28번] 원인 진단 근거 저장 (옵션 산출물)
    if selection_diagnostics is not None:
        save_artifact(
            selection_diagnostics,
            interim_dir / f"bt_selection_diagnostics{suffix}",
            force=True,
        )
    if bt_returns_diagnostics is not None:
        save_artifact(
            bt_returns_diagnostics,
            interim_dir / f"bt_returns_diagnostics{suffix}",
            force=True,
        )
    if runtime_profile is not None:
        save_artifact(
            runtime_profile, interim_dir / f"bt_runtime_profile{suffix}", force=True
        )
    if bt_regime_metrics is not None:
        save_artifact(
            bt_regime_metrics, interim_dir / f"bt_regime_metrics{suffix}", force=True
        )

    logger.info("✅ 백테스트 전략 실행 완료")

    return {
        "bt_positions": bt_pos,
        "bt_returns": bt_ret,
        "bt_equity_curve": bt_eq,
        "bt_metrics": bt_met,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    strategy = sys.argv[1] if len(sys.argv) > 1 else "bt20_short"
    results = run_backtest_strategy(strategy=strategy)
    print(f"\n✅ 완료: {len(results['bt_metrics'])}개 메트릭 생성")
