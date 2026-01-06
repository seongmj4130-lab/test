# -*- coding: utf-8 -*-
"""
Track B: 투자 모델 서비스 모듈

UI에서 import 가능한 형태로 백테스트 실행 함수를 제공합니다.

[리팩토링 2단계] 함수/모듈화 - UI에서 import 가능한 형태
"""
from typing import Dict, Optional, Literal
from pathlib import Path
import pandas as pd
import logging

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists, save_artifact
from src.tracks.shared.data_pipeline import prepare_common_data
from src.tracks.track_a.ranking_service import generate_rankings
from src.tracks.track_b.stages.modeling.l6r_ranking_scoring import run_L6R_ranking_scoring
from src.tracks.track_b.stages.backtest.l7_backtest import run_backtest, BacktestConfig

logger = logging.getLogger(__name__)


def run_backtest_strategy(
    strategy: Literal["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"],
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
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
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(strategy_config_map.keys())}")
    
    l7_cfg = cfg.get(l7_config_key, {})
    if not l7_cfg:
        raise ValueError(f"Config key '{l7_config_key}' not found in config.yaml")
    
    # 공통 데이터 준비
    artifacts = prepare_common_data(config_path=config_path, force_rebuild=force_rebuild)
    
    # Track A 산출물 확인 (랭킹 데이터)
    interim_dir = Path(get_path(cfg, "data_interim"))
    ranking_short_path = interim_dir / "ranking_short_daily"
    ranking_long_path = interim_dir / "ranking_long_daily"
    
    if (not artifact_exists(ranking_short_path) or 
        not artifact_exists(ranking_long_path) or 
        force_rebuild):
        logger.info("랭킹 데이터가 없습니다. Track A를 실행합니다...")
        rankings = generate_rankings(config_path=config_path, force_rebuild=force_rebuild)
        artifacts["ranking_short_daily"] = rankings["ranking_short_daily"]
        artifacts["ranking_long_daily"] = rankings["ranking_long_daily"]
    else:
        artifacts["ranking_short_daily"] = load_artifact(ranking_short_path)
        artifacts["ranking_long_daily"] = load_artifact(ranking_long_path)
    
    # L6R: 랭킹 스코어 변환
    logger.info("[L6R] 랭킹 스코어 변환")
    scores_path = interim_dir / "rebalance_scores_from_ranking"
    
    if artifact_exists(scores_path) and not force_rebuild:
        artifacts["rebalance_scores"] = load_artifact(scores_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['rebalance_scores']):,}행")
    else:
        logger.info("  → L6R 재실행")
        outputs, warns = run_L6R_ranking_scoring(
            cfg=cfg,
            artifacts={
                "ranking_short_daily": artifacts["ranking_short_daily"],
                "ranking_long_daily": artifacts["ranking_long_daily"],
                "dataset_daily": artifacts["dataset_daily"],
                "cv_folds_short": artifacts["cv_folds_short"],
                "universe_k200_membership_monthly": artifacts["universe_k200_membership_monthly"],
            },
            force=force_rebuild,
        )
        artifacts["rebalance_scores"] = outputs["rebalance_scores"]
        save_artifact(artifacts["rebalance_scores"], scores_path, force=True)
        logger.info(f"  ✓ 생성 완료: {len(artifacts['rebalance_scores']):,}행")
    
    # L7: 백테스트 실행
    logger.info("[L7] 백테스트 실행")
    bt_cfg = BacktestConfig(
        holding_days=int(l7_cfg.get("holding_days", 20)),
        top_k=int(l7_cfg.get("top_k", 12)),
        cost_bps=float(l7_cfg.get("cost_bps", 10.0)),
        score_col=str(l7_cfg.get("score_col", "score_ens")),
        ret_col=str(l7_cfg.get("return_col", "true_short")),
        weighting=str(l7_cfg.get("weighting", "equal")),
        softmax_temp=float(l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp", 1.0))),
        buffer_k=int(l7_cfg.get("buffer_k", 15)),
        rebalance_interval=int(l7_cfg.get("rebalance_interval", 1)),
        smart_buffer_enabled=bool(l7_cfg.get("smart_buffer_enabled", True)),
        smart_buffer_stability_threshold=float(l7_cfg.get("smart_buffer_stability_threshold", 0.7)),
        volatility_adjustment_enabled=bool(l7_cfg.get("volatility_adjustment_enabled", True)),
        volatility_lookback_days=int(l7_cfg.get("volatility_lookback_days", 60)),
        target_volatility=float(l7_cfg.get("target_volatility", 0.15)),
        volatility_adjustment_max=float(l7_cfg.get("volatility_adjustment_max", 1.2)),
        volatility_adjustment_min=float(l7_cfg.get("volatility_adjustment_min", 0.7)),
        risk_scaling_enabled=bool(l7_cfg.get("risk_scaling_enabled", True)),
        risk_scaling_bear_multiplier=float(l7_cfg.get("risk_scaling_bear_multiplier", 0.8)),
        risk_scaling_neutral_multiplier=float(l7_cfg.get("risk_scaling_neutral_multiplier", 1.0)),
        risk_scaling_bull_multiplier=float(l7_cfg.get("risk_scaling_bull_multiplier", 1.0)),
    )
    
    # 시장 국면 데이터 (regime_enabled일 때)
    market_regime_df = None
    if l7_cfg.get("regime", {}).get("enabled", False):
        logger.info("  → 시장 국면 데이터 생성")
        from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime
        
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
        bt_pos, bt_ret, bt_eq, bt_met, quality, warns, selection_diagnostics, bt_returns_diagnostics, runtime_profile, bt_regime_metrics = result
    elif len(result) == 9:
        bt_pos, bt_ret, bt_eq, bt_met, quality, warns, selection_diagnostics, bt_returns_diagnostics, runtime_profile = result
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
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    strategy = sys.argv[1] if len(sys.argv) > 1 else "bt20_short"
    results = run_backtest_strategy(strategy=strategy)
    print(f"\n✅ 완료: {len(results['bt_metrics'])}개 메트릭 생성")

