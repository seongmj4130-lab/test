# -*- coding: utf-8 -*-
"""
Track B 전체 파이프라인 실행 모듈

Track B: 투자 모델 (Investment Model)
- 목적: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공
- L6R: 랭킹 스코어 변환
- L7: 백테스트 실행
"""
from pathlib import Path
from typing import Dict, Optional
import logging
import pandas as pd

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists, save_artifact

logger = logging.getLogger(__name__)


def run_track_b_pipeline(
    config_path: str = "configs/config.yaml",
    strategy: str = "bt20_short",  # "bt20_short", "bt20_ens", "bt120_long", "bt120_ens"
    force_rebuild: bool = False,
) -> Dict:
    """
    Track B 전체 파이프라인을 실행합니다.
    
    Track B는 랭킹을 기반으로 투자 모델을 실행하여 이용자에게 정보를 제공합니다.
    
    Args:
        config_path: 설정 파일 경로
        strategy: 투자 전략 ("bt20_short", "bt20_ens", "bt120_long", "bt120_ens")
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
    logger.info("Track B: 투자 모델 파이프라인 실행 시작")
    logger.info(f"전략: {strategy}")
    logger.info("=" * 80)
    
    # 설정 로드
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # 전략별 설정 매핑
    strategy_config_map = {
        "bt20_short": "l7_bt20_short",
        "bt20_pro": "l7_bt20_pro",      # [bt20 개선안 1] bt20 프로페셔널
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
    
    artifacts = {}
    artifacts_path = {}
    
    # 공통 데이터 확인
    logger.info("[공통 데이터 확인]")
    
    # L0: 유니버스
    uni_path = interim_dir / "universe_k200_membership_monthly"
    # [개선안 41번] force_rebuild 의미 정리:
    # - 입력(공통/Track A 캐시)은 항상 로드 가능해야 한다.
    # - force_rebuild는 Track B의 "출력(rebalance_scores/bt_*)"을 재생성할지 여부로만 사용한다.
    if artifact_exists(uni_path):
        artifacts["universe_k200_membership_monthly"] = load_artifact(uni_path)
        artifacts_path["universe"] = str(uni_path)
        logger.info(f"  ✓ 유니버스 로드: {len(artifacts['universe_k200_membership_monthly']):,}행")
    else:
        logger.warning("  ✗ 유니버스 데이터가 없습니다. L0부터 실행이 필요합니다.")
        raise FileNotFoundError("universe_k200_membership_monthly not found")
    
    # L4: CV 분할
    dataset_path = interim_dir / "dataset_daily"
    cv_short_path = interim_dir / "cv_folds_short"
    
    if (artifact_exists(dataset_path) and artifact_exists(cv_short_path)):
        artifacts["dataset_daily"] = load_artifact(dataset_path)
        artifacts["cv_folds_short"] = load_artifact(cv_short_path)
        artifacts_path["dataset"] = str(dataset_path)
        artifacts_path["cv_short"] = str(cv_short_path)
        logger.info(f"  ✓ 데이터셋 로드: {len(artifacts['dataset_daily']):,}행")
    else:
        logger.warning("  ✗ 데이터셋이 없습니다. L4까지 실행이 필요합니다.")
        raise FileNotFoundError("dataset_daily or cv_folds_short not found")
    
    # Track A 산출물 확인 (랭킹 데이터)
    logger.info("[Track A 산출물 확인]")
    ranking_short_path = interim_dir / "ranking_short_daily"
    ranking_long_path = interim_dir / "ranking_long_daily"
    
    if (artifact_exists(ranking_short_path) and artifact_exists(ranking_long_path)):
        artifacts["ranking_short_daily"] = load_artifact(ranking_short_path)
        artifacts["ranking_long_daily"] = load_artifact(ranking_long_path)
        artifacts_path["ranking_short"] = str(ranking_short_path)
        artifacts_path["ranking_long"] = str(ranking_long_path)
        logger.info(f"  ✓ 랭킹 데이터 로드: 단기 {len(artifacts['ranking_short_daily']):,}행, 장기 {len(artifacts['ranking_long_daily']):,}행")
    else:
        logger.warning("  ✗ 랭킹 데이터가 없습니다. Track A를 먼저 실행하세요.")
        logger.warning("  python -m src.pipeline.track_a_pipeline 를 먼저 실행하세요.")
        raise FileNotFoundError("ranking_short_daily or ranking_long_daily not found")
    
    # L6R: 랭킹 스코어 변환
    logger.info("[L6R] 랭킹 스코어 변환")
    from src.tracks.track_b.stages.modeling.l6r_ranking_scoring import run_L6R_ranking_scoring
    
    # [rebalance_interval 개선] 모델별 rebalance_interval을 캐시 키에 포함
    rebalance_interval = l7_cfg.get("rebalance_interval", 1)
    scores_path = interim_dir / f"rebalance_scores_from_ranking_interval_{rebalance_interval}"
    
    if artifact_exists(scores_path) and not force_rebuild:
        artifacts["rebalance_scores"] = load_artifact(scores_path)
        artifacts_path["scores"] = str(scores_path)
        logger.info(f"  ✓ 캐시에서 로드: {len(artifacts['rebalance_scores']):,}행 (rebalance_interval={rebalance_interval})")
    else:
        logger.info(f"  → L6R 재실행 (rebalance_interval={rebalance_interval})")
        # L6R 실행 (필요한 모든 artifacts 전달)
        # ohlcv_daily 로드 (시장 국면 분류용)
        ohlcv_path = interim_dir / "ohlcv_daily"
        ohlcv_daily = None
        if artifact_exists(ohlcv_path):
            ohlcv_daily = load_artifact(ohlcv_path)
        
        # [rebalance_interval 개선] l7_cfg를 cfg에 임시로 추가하여 L6R에서 읽을 수 있도록 함
        cfg_with_l7 = cfg.copy()
        cfg_with_l7["l7"] = l7_cfg
        
        outputs, warns = run_L6R_ranking_scoring(
            cfg=cfg_with_l7,  # l7_cfg 포함
            artifacts={
                "ranking_short_daily": artifacts["ranking_short_daily"],
                "ranking_long_daily": artifacts["ranking_long_daily"],
                "dataset_daily": artifacts["dataset_daily"],
                "cv_folds_short": artifacts["cv_folds_short"],
                "universe_k200_membership_monthly": artifacts["universe_k200_membership_monthly"],
                "ohlcv_daily": ohlcv_daily,
            },
            force=force_rebuild,
        )
        artifacts["rebalance_scores"] = outputs["rebalance_scores"]
        save_artifact(artifacts["rebalance_scores"], scores_path, force=True)
        artifacts_path["scores"] = str(scores_path)
        logger.info(f"  ✓ 생성 완료: {len(artifacts['rebalance_scores']):,}행 (rebalance_interval={rebalance_interval})")
    
    # L7: 백테스트 실행
    logger.info("[L7] 백테스트 실행")
    from src.tracks.track_b.stages.backtest.l7_backtest import run_backtest, BacktestConfig
    
    # BacktestConfig 생성
    # [리팩토링] config.yaml 우선 적용 구조로 변경
    # - config.yaml → BacktestConfig 기본값 순서로 적용
    # - .get() 기본값을 BacktestConfig 클래스 기본값으로 통일
    l7_regime = (l7_cfg.get("regime", {}) or {}) if isinstance(l7_cfg.get("regime", {}), dict) else {}
    l7_div = (l7_cfg.get("diversify", {}) or {}) if isinstance(l7_cfg.get("diversify", {}), dict) else {}
    
    # [리팩토링] BacktestConfig 기본값 가져오기 (config.yaml 우선 적용)
    default_cfg = BacktestConfig()
    
    bt_cfg = BacktestConfig(
        holding_days=int(l7_cfg.get("holding_days", default_cfg.holding_days)),
        top_k=int(l7_cfg.get("top_k", default_cfg.top_k)),
        cost_bps=float(l7_cfg.get("cost_bps", default_cfg.cost_bps)),
        slippage_bps=float(l7_cfg.get("slippage_bps", default_cfg.slippage_bps)),
        score_col=str(l7_cfg.get("score_col", default_cfg.score_col)),
        ret_col=str(l7_cfg.get("ret_col", default_cfg.ret_col)),
        weighting=str(l7_cfg.get("weighting", default_cfg.weighting)),
        softmax_temp=float(l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp", default_cfg.softmax_temp))),
        buffer_k=int(l7_cfg.get("buffer_k", default_cfg.buffer_k)),
        rebalance_interval=int(l7_cfg.get("rebalance_interval", default_cfg.rebalance_interval)),
        smart_buffer_enabled=bool(l7_cfg.get("smart_buffer_enabled", default_cfg.smart_buffer_enabled)),
        smart_buffer_stability_threshold=float(l7_cfg.get("smart_buffer_stability_threshold", default_cfg.smart_buffer_stability_threshold)),
        volatility_adjustment_enabled=bool(l7_cfg.get("volatility_adjustment_enabled", default_cfg.volatility_adjustment_enabled)),
        volatility_lookback_days=int(l7_cfg.get("volatility_lookback_days", default_cfg.volatility_lookback_days)),
        target_volatility=float(l7_cfg.get("target_volatility", default_cfg.target_volatility)),
        volatility_adjustment_max=float(l7_cfg.get("volatility_adjustment_max", default_cfg.volatility_adjustment_max)),
        volatility_adjustment_min=float(l7_cfg.get("volatility_adjustment_min", default_cfg.volatility_adjustment_min)),
        risk_scaling_enabled=bool(l7_cfg.get("risk_scaling_enabled", default_cfg.risk_scaling_enabled)),
        risk_scaling_bear_multiplier=float(l7_cfg.get("risk_scaling_bear_multiplier", default_cfg.risk_scaling_bear_multiplier)),
        # [bt20 프로페셔널] 적응형 리밸런싱 설정
        adaptive_rebalancing_enabled=bool(l7_cfg.get("adaptive_rebalancing_enabled", default_cfg.adaptive_rebalancing_enabled)),
        signal_strength_thresholds=l7_cfg.get("signal_strength_thresholds", default_cfg.signal_strength_thresholds),
        rebalance_intervals=l7_cfg.get("rebalance_intervals", default_cfg.rebalance_intervals),
        risk_scaling_neutral_multiplier=float(l7_cfg.get("risk_scaling_neutral_multiplier", default_cfg.risk_scaling_neutral_multiplier)),
        risk_scaling_bull_multiplier=float(l7_cfg.get("risk_scaling_bull_multiplier", default_cfg.risk_scaling_bull_multiplier)),
        overlapping_tranches_enabled=bool(l7_cfg.get("overlapping_tranches_enabled", default_cfg.overlapping_tranches_enabled)),
        tranche_holding_days=int(l7_cfg.get("tranche_holding_days", default_cfg.tranche_holding_days)),
        tranche_max_active=int(l7_cfg.get("tranche_max_active", default_cfg.tranche_max_active)),
        tranche_allocation_mode=str(l7_cfg.get("tranche_allocation_mode", default_cfg.tranche_allocation_mode)),
        diversify_enabled=bool(l7_div.get("enabled", default_cfg.diversify_enabled)),
        group_col=str(l7_div.get("group_col", default_cfg.group_col)),
        max_names_per_group=int(l7_div.get("max_names_per_group", default_cfg.max_names_per_group)),
        regime_enabled=bool(l7_regime.get("enabled", default_cfg.regime_enabled)),
        regime_top_k_bull_strong=(int(l7_regime["top_k_bull_strong"]) if "top_k_bull_strong" in l7_regime else default_cfg.regime_top_k_bull_strong),
        regime_top_k_bull_weak=(int(l7_regime["top_k_bull_weak"]) if "top_k_bull_weak" in l7_regime else default_cfg.regime_top_k_bull_weak),
        regime_top_k_bear_strong=(int(l7_regime["top_k_bear_strong"]) if "top_k_bear_strong" in l7_regime else default_cfg.regime_top_k_bear_strong),
        regime_top_k_bear_weak=(int(l7_regime["top_k_bear_weak"]) if "top_k_bear_weak" in l7_regime else default_cfg.regime_top_k_bear_weak),
        regime_top_k_neutral=(int(l7_regime["top_k_neutral"]) if "top_k_neutral" in l7_regime else default_cfg.regime_top_k_neutral),
        regime_exposure_bull_strong=(float(l7_regime["exposure_bull_strong"]) if "exposure_bull_strong" in l7_regime else default_cfg.regime_exposure_bull_strong),
        regime_exposure_bull_weak=(float(l7_regime["exposure_bull_weak"]) if "exposure_bull_weak" in l7_regime else default_cfg.regime_exposure_bull_weak),
        regime_exposure_bear_strong=(float(l7_regime["exposure_bear_strong"]) if "exposure_bear_strong" in l7_regime else default_cfg.regime_exposure_bear_strong),
        regime_exposure_bear_weak=(float(l7_regime["exposure_bear_weak"]) if "exposure_bear_weak" in l7_regime else default_cfg.regime_exposure_bear_weak),
        regime_exposure_neutral=(float(l7_regime["exposure_neutral"]) if "exposure_neutral" in l7_regime else default_cfg.regime_exposure_neutral),
        regime_top_k_bull=(int(l7_regime["top_k_bull"]) if "top_k_bull" in l7_regime else default_cfg.regime_top_k_bull),
        regime_top_k_bear=(int(l7_regime["top_k_bear"]) if "top_k_bear" in l7_regime else default_cfg.regime_top_k_bear),
        regime_exposure_bull=(float(l7_regime["exposure_bull"]) if "exposure_bull" in l7_regime else default_cfg.regime_exposure_bull),
        regime_exposure_bear=(float(l7_regime["exposure_bear"]) if "exposure_bear" in l7_regime else default_cfg.regime_exposure_bear),
    )
    
    # 시장 국면 데이터 (regime_enabled일 때)
    market_regime_df = None
    if l7_cfg.get("regime", {}).get("enabled", False):
        logger.info("  → 시장 국면 데이터 생성 (외부 API 없이 ohlcv_daily 사용)")
        from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime
        
        # ohlcv_daily 데이터 로드
        ohlcv_path = interim_dir / "ohlcv_daily"
        if not artifact_exists(ohlcv_path):
            logger.warning("  ⚠ ohlcv_daily가 없어 시장 국면 기능을 건너뜁니다.")
        else:
            ohlcv_daily = load_artifact(ohlcv_path)
            rebalance_dates = artifacts["rebalance_scores"]["date"].unique()
            regime_cfg = l7_cfg.get("regime", {})
            
            market_regime_df = build_market_regime(
                rebalance_dates=rebalance_dates,
                ohlcv_daily=ohlcv_daily,
                lookback_days=regime_cfg.get("lookback_days", 60),
                neutral_band=regime_cfg.get("neutral_band", 0.05),
                use_volume=regime_cfg.get("use_volume", True),
                use_volatility=regime_cfg.get("use_volatility", True),
            )
            logger.info(f"  ✓ 시장 국면 데이터 생성: {len(market_regime_df):,}행")
    
    # 백테스트 실행
    result = run_backtest(
        rebalance_scores=artifacts["rebalance_scores"],
        cfg=bt_cfg,
        config_cost_bps=bt_cfg.cost_bps,  # [리팩토링] BacktestConfig에서 가져옴 (config.yaml 우선 적용)
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

    # [개선안 43번] Stage13 진단/국면 아티팩트도 저장(설정 반영/검증용)
    # - bt_returns는 core/diagnostics로 분리되므로, diagnostics는 별도 파일로 저장해야 tranche/regime/exposure 등이 유실되지 않음
    if bt_returns_diagnostics is not None and isinstance(bt_returns_diagnostics, pd.DataFrame) and len(bt_returns_diagnostics) > 0:
        save_artifact(bt_returns_diagnostics, interim_dir / f"bt_returns_diagnostics{suffix}", force=True)
        artifacts_path["bt_returns_diagnostics"] = str(interim_dir / f"bt_returns_diagnostics{suffix}")
    if selection_diagnostics is not None and isinstance(selection_diagnostics, pd.DataFrame) and len(selection_diagnostics) > 0:
        save_artifact(selection_diagnostics, interim_dir / f"selection_diagnostics{suffix}", force=True)
        artifacts_path["selection_diagnostics"] = str(interim_dir / f"selection_diagnostics{suffix}")
    if runtime_profile is not None:
        # runtime_profile은 dict일 수 있어 csv/parquet로는 저장 불가 → json으로 저장
        try:
            import json
            rp_path = interim_dir / f"runtime_profile{suffix}.json"
            rp_path.write_text(json.dumps(runtime_profile, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts_path["runtime_profile"] = str(rp_path)
        except Exception:
            pass
    if bt_regime_metrics is not None and isinstance(bt_regime_metrics, pd.DataFrame) and len(bt_regime_metrics) > 0:
        save_artifact(bt_regime_metrics, interim_dir / f"bt_regime_metrics{suffix}", force=True)
        artifacts_path["bt_regime_metrics"] = str(interim_dir / f"bt_regime_metrics{suffix}")
    
    artifacts_path["bt_positions"] = str(interim_dir / f"bt_positions{suffix}")
    artifacts_path["bt_returns"] = str(interim_dir / f"bt_returns{suffix}")
    artifacts_path["bt_equity_curve"] = str(interim_dir / f"bt_equity_curve{suffix}")
    artifacts_path["bt_metrics"] = str(interim_dir / f"bt_metrics{suffix}")
    
    logger.info("=" * 80)
    logger.info("✅ Track B: 투자 모델 파이프라인 실행 완료")
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
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    strategy = sys.argv[1] if len(sys.argv) > 1 else "bt20_short"
    result = run_track_b_pipeline(strategy=strategy)
    print(f"\n✅ 완료: {len(result['bt_metrics'])}개 메트릭 생성")

