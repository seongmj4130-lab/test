# -*- coding: utf-8 -*-
"""
L0~L7 전체 파이프라인 실행 스크립트 (2015년 제거용)

config.yaml의 start_date가 2016-01-01로 설정되어 있으면
자동으로 2016년부터 데이터를 생성합니다.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

sys.stdout.reconfigure(encoding='utf-8')

import logging
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 각 스테이지 함수 직접 import
from src.stages.data.l0_universe import build_k200_membership_month_end
from src.stages.data.l1_ohlcv import download_ohlcv_panel
from src.stages.data.l2_fundamentals_dart import download_annual_fundamentals
from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.tracks.shared.stages.data.l3_panel_merge import build_panel_merged_daily
from src.tracks.shared.stages.data.l4_walkforward_split import build_targets_and_folds

# L7 백테스트
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.meta import build_meta, save_meta
from src.utils.quality import fundamental_coverage_report, walkforward_quality_report
from src.utils.validate import raise_if_invalid, validate_df


def run_L0_universe(cfg, artifacts=None, *, force=False):
    """L0: 유니버스 구성"""
    logger.info("[L0] 시작")
    p = cfg.get("params", {})
    df = build_k200_membership_month_end(
        start_date=p.get("start_date", "2016-01-01"),
        end_date=p.get("end_date", "2024-12-31"),
        index_code=p.get("index_code", "1028"),
        anchor_ticker=p.get("anchor_ticker", "005930"),
    )
    logger.info(f"[L0] 완료: {len(df):,}행")
    return {"universe_k200_membership_monthly": df}, []

def run_L1_base(cfg, artifacts, *, force=False):
    """L1: OHLCV 다운로드 + 기술적 지표 계산"""
    logger.info("[L1] 시작")
    p = cfg.get("params", {})
    uni = artifacts["universe_k200_membership_monthly"]
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())

    df = download_ohlcv_panel(
        tickers=tickers,
        start_date=p.get("start_date", "2016-01-01"),
        end_date=p.get("end_date", "2024-12-31"),
        calculate_technical_features=True,  # [FEATURESET_COMPLETE] 기술적 지표 자동 계산
    )

    # 기술적 지표가 포함되었는지 확인
    technical_cols = [c for c in df.columns if c in [
        "price_momentum_20d", "price_momentum_60d", "volatility_20d",
        "volatility_60d", "max_drawdown_60d", "volume_ratio", "momentum_reversal"
    ]]
    if technical_cols:
        logger.info(f"[L1] 기술적 지표 포함: {len(technical_cols)}개 ({', '.join(technical_cols[:5])}...)")

    logger.info(f"[L1] 완료: {len(df):,}행, {len(df.columns)}컬럼")
    return {"ohlcv_daily": df}, []

def run_L2_merge(cfg, artifacts, *, force=False):
    """L2: 재무 데이터 로드 (기존 데이터 사용, 새로 다운로드 안 함)"""
    logger.info("[L2] 시작 - 기존 DART 데이터 사용")

    # 기존 데이터 확인
    interim_dir = Path(get_path(cfg, "data_interim"))
    existing_file = interim_dir / "fundamentals_annual.parquet"

    if existing_file.exists():
        logger.info(f"[L2] 기존 데이터 로드: {existing_file}")
        df = load_artifact(existing_file)
        logger.info(f"[L2] 완료: {len(df):,}행 (기존 데이터 사용)")
        return {"fundamentals_annual": df}, []
    else:
        logger.warning("[L2] 기존 fundamentals_annual 데이터가 없습니다. L3에서 재무 데이터 없이 진행합니다.")
        # 빈 dict 반환 (L3에서 None으로 처리됨)
        return {}, []

def run_L3_features(cfg, artifacts, *, force=False):
    """L3: 패널 병합 (기존 데이터 있으면 사용)"""
    logger.info("[L3] 시작")

    # 기존 panel_merged_daily 확인
    interim_dir = Path(get_path(cfg, "data_interim"))
    existing_file = interim_dir / "panel_merged_daily.parquet"

    if existing_file.exists() and not force:
        logger.info(f"[L3] 기존 panel_merged_daily 사용: {existing_file}")
        df = load_artifact(existing_file)

        # 기술적 지표와 시가총액이 포함되어 있는지 확인
        technical_cols = [c for c in df.columns if c in [
            "price_momentum_20d", "volatility_20d", "market_cap"
        ]]
        if technical_cols:
            logger.info(f"[L3] 기존 데이터에 추가 피처 포함: {len(technical_cols)}개")
        else:
            logger.info("[L3] 기존 데이터 사용 (추가 피처는 OHLCV에서 병합)")
            # OHLCV의 기술적 지표를 병합
            ohlcv = artifacts.get("ohlcv_daily")
            if ohlcv is not None:
                technical_cols_ohlcv = [c for c in ohlcv.columns if c in [
                    "price_momentum_20d", "price_momentum_60d", "volatility_20d",
                    "volatility_60d", "max_drawdown_60d", "volume_ratio", "momentum_reversal"
                ]]
                if technical_cols_ohlcv:
                    df = df.merge(
                        ohlcv[["date", "ticker"] + technical_cols_ohlcv],
                        on=["date", "ticker"],
                        how="left",
                        suffixes=("", "_new")
                    )
                    # 기존 컬럼이 없으면 새 컬럼 사용
                    for col in technical_cols_ohlcv:
                        if col not in df.columns and f"{col}_new" in df.columns:
                            df[col] = df[f"{col}_new"]
                            df = df.drop(columns=[f"{col}_new"])
                    logger.info(f"[L3] 기술적 지표 병합 완료: {len(technical_cols_ohlcv)}개")

        logger.info(f"[L3] 완료: {len(df):,}행 (기존 데이터 사용)")
        return {"panel_merged_daily": df}, []

    # 새로 생성
    p = cfg.get("params", {})
    lag_days = int(p.get("fundamental_lag_days", 90))
    l3_cfg = cfg.get("l3", {}) or {}

    # fundamentals_annual이 없으면 None으로 전달 (L3에서 처리)
    fundamentals_annual = artifacts.get("fundamentals_annual")
    if fundamentals_annual is None:
        logger.warning("[L3] fundamentals_annual이 없습니다. OHLCV 데이터만 사용합니다.")

    df, warns = build_panel_merged_daily(
        ohlcv_daily=artifacts["ohlcv_daily"],
        fundamentals_annual=fundamentals_annual,
        universe_membership_monthly=artifacts.get("universe_k200_membership_monthly"),
        fundamental_lag_days=lag_days,
        filter_k200_members_only=bool(p.get("filter_k200_members_only", False)),
        fundamentals_effective_date_col=l3_cfg.get("fundamentals_effective_date_col", "effective_date"),
    )
    logger.info(f"[L3] 완료: {len(df):,}행")
    return {"panel_merged_daily": df}, warns

def run_L4_split(cfg, artifacts, *, force=False):
    """L4: CV 분할 및 타겟 생성"""
    logger.info("[L4] 시작")
    l4 = cfg.get("l4", {}) or {}
    panel = artifacts["panel_merged_daily"]

    df, cv_s, cv_l, warns = build_targets_and_folds(
        panel_merged_daily=panel,
        holdout_years=int(l4.get("holdout_years", 2)),
        step_days=int(l4.get("step_days", 20)),
        test_window_days=int(l4.get("test_window_days", 20)),
        embargo_days=int(l4.get("embargo_days", 20)),
        horizon_short=int(l4.get("horizon_short", 20)),
        horizon_long=int(l4.get("horizon_long", 120)),
        rolling_train_years_short=int(l4.get("rolling_train_years_short", 3)),
        rolling_train_years_long=int(l4.get("rolling_train_years_long", 5)),
        price_col=l4.get("price_col", None),
    )

    logger.info(f"[L4] 완료: dataset_daily {len(df):,}행, cv_folds_short {len(cv_s)}개, cv_folds_long {len(cv_l)}개")
    return {
        "dataset_daily": df,
        "cv_folds_short": cv_s,
        "cv_folds_long": cv_l,
    }, warns

def run_L5_modeling(cfg, artifacts, *, force=False):
    """L5: 모델 학습 및 예측"""
    logger.info("[L5] 시작")
    df = artifacts["dataset_daily"]
    cv_s = artifacts["cv_folds_short"]
    cv_l = artifacts["cv_folds_long"]

    l4 = cfg.get("l4", {}) or {}
    hs = int(l4.get("horizon_short", 20))
    hl = int(l4.get("horizon_long", 120))

    target_s = f"ret_fwd_{hs}d"
    target_l = f"ret_fwd_{hl}d"

    logger.info(f"[L5] Short 예측 학습 중... (target: {target_s})")
    pred_s, met_s, rep_s, w_s = train_oos_predictions(
        dataset_daily=df,
        cv_folds=cv_s,
        cfg=cfg,
        target_col=target_s,
        horizon=hs,
    )

    logger.info(f"[L5] Long 예측 학습 중... (target: {target_l})")
    pred_l, met_l, rep_l, w_l = train_oos_predictions(
        dataset_daily=df,
        cv_folds=cv_l,
        cfg=cfg,
        target_col=target_l,
        horizon=hl,
    )

    metrics = pd.concat([met_s, met_l], ignore_index=True)
    warns = (w_s or []) + (w_l or [])

    artifacts["_l5_report_short"] = rep_s
    artifacts["_l5_report_long"] = rep_l

    logger.info(f"[L5] 완료: pred_short_oos {len(pred_s):,}행, pred_long_oos {len(pred_l):,}행")
    return {
        "pred_short_oos": pred_s,
        "pred_long_oos": pred_l,
        "model_metrics": metrics,
    }, warns

def run_L6_scoring(cfg, artifacts, *, force=False):
    """L6: 스코어 생성"""
    logger.info("[L6] 시작")
    p = cfg.get("params", {}) or {}
    l6 = p.get("l6", {}) if isinstance(p.get("l6", {}), dict) else {}
    if not l6:
        l6 = cfg.get("l6", {}) or {}

    w_s = float(l6.get("weight_short", 0.5))
    w_l = float(l6.get("weight_long", 0.5))

    scores, summary, quality, warns = build_rebalance_scores(
        pred_short_oos=artifacts["pred_short_oos"],
        pred_long_oos=artifacts["pred_long_oos"],
        universe_k200_membership_monthly=artifacts.get("universe_k200_membership_monthly"),
        weight_short=w_s,
        weight_long=w_l,
    )

    artifacts["_l6_quality"] = {"scoring": quality} if isinstance(quality, dict) else {"scoring": quality}

    logger.info(f"[L6] 완료: rebalance_scores {len(scores):,}행")
    return {
        "rebalance_scores": scores,
        "rebalance_scores_summary": summary,
    }, warns

def run_L7_backtest(cfg, artifacts, *, force=False):
    """L7: 백테스트 실행"""
    logger.info("[L7] 시작")

    rebalance_scores = artifacts["rebalance_scores"]

    # L7 config
    p = cfg.get("params", {}) or {}
    l7 = p.get("l7", {}) if isinstance(p.get("l7", {}), dict) else {}
    if not l7:
        l7 = cfg.get("l7", {}) or {}

    # BacktestConfig 생성
    bt_cfg = BacktestConfig(
        holding_days=int(l7.get("holding_days", 20)),
        top_k=int(l7.get("top_k", 20)),
        cost_bps=float(l7.get("cost_bps", 10.0)),
        score_col=l7.get("score_col", "score_ens"),
        ret_col=l7.get("return_col", "true_short"),
        weighting=l7.get("weighting", "equal"),
        softmax_temp=float(l7.get("softmax_temp", 1.0)),
        buffer_k=int(l7.get("buffer_k", 0)),
    )

    result = run_backtest(
        rebalance_scores=rebalance_scores,
        cfg=bt_cfg,
    )

    # 반환값 처리 (10개, 9개 또는 6개)
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

    logger.info(f"[L7] 완료: bt_positions {len(bt_pos):,}행, bt_returns {len(bt_ret):,}행")

    outputs = {
        "bt_positions": bt_pos,
        "bt_returns": bt_ret,
        "bt_equity_curve": bt_eq,
        "bt_metrics": bt_met,
    }

    if selection_diagnostics is not None:
        outputs["selection_diagnostics"] = selection_diagnostics
    if bt_returns_diagnostics is not None:
        outputs["bt_returns_diagnostics"] = bt_returns_diagnostics
    if runtime_profile is not None:
        outputs["runtime_profile"] = runtime_profile
    if bt_regime_metrics is not None:
        outputs["bt_regime_metrics"] = bt_regime_metrics

    return outputs, warns

def main():
    print("=" * 80)
    print("L0~L7 전체 파이프라인 실행 (2015년 제거)")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 설정 로드
    cfg = load_config("configs/config.yaml")
    interim_dir = Path(get_path(cfg, "data_interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)

    save_formats = cfg.get("run", {}).get("save_formats", ["parquet", "csv"])
    fail_on_validation_error = bool(cfg.get("run", {}).get("fail_on_validation_error", True))
    write_meta = bool(cfg.get("run", {}).get("write_meta", True))
    force = True

    artifacts = {}
    target_stages = [
        ("L0", run_L0_universe),
        ("L1", run_L1_base),
        ("L2", run_L2_merge),
        ("L3", run_L3_features),
        ("L4", run_L4_split),
        ("L5", run_L5_modeling),
        ("L6", run_L6_scoring),
        ("L7", run_L7_backtest),
    ]

    # start_date 확인
    start_date = cfg.get("params", {}).get("start_date", "")
    logger.info(f"[설정] start_date: {start_date}")

    if start_date < "2016-01-01":
        logger.warning(f"[경고] start_date가 2016-01-01 이전입니다: {start_date}")
        logger.warning("2015년 데이터가 포함될 수 있습니다.")

    for stage_name, stage_func in target_stages:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"[{stage_name}] 시작")
            logger.info(f"{'='*80}")

            # 스테이지 실행
            outputs, stage_warnings = stage_func(cfg, artifacts, force=force)

            # L2는 fundamentals_annual이 없을 수 있으므로 예외 처리
            if not isinstance(outputs, dict):
                raise ValueError(f"{stage_name} must return dict[str, DataFrame].")

            # L2에서 fundamentals_annual이 없으면 스킵 (기존 panel_merged_daily에 재무 데이터 포함됨)
            if stage_name == "L2" and not outputs:
                logger.warning(f"[{stage_name}] 기존 데이터가 없습니다. L3에서 기존 panel_merged_daily를 사용합니다.")
                continue

            # 출력 저장
            for out_name, df in outputs.items():
                out_base = interim_dir / out_name

                # 저장
                save_artifact(df, out_base, force=force, formats=save_formats)
                artifacts[out_name] = df

                # 메타데이터 저장
                if write_meta:
                    quality = {}
                    if stage_name == "L3" and out_name == "panel_merged_daily":
                        quality["fundamental"] = fundamental_coverage_report(df)

                    if stage_name == "L4" and out_name == "dataset_daily":
                        quality["walkforward"] = walkforward_quality_report(
                            dataset_daily=df,
                            cv_folds_short=outputs.get("cv_folds_short", artifacts.get("cv_folds_short")),
                            cv_folds_long=outputs.get("cv_folds_long", artifacts.get("cv_folds_long")),
                            cfg=cfg,
                        )

                    meta = build_meta(
                        stage=f"{stage_name}:{out_name}",
                        run_id=f"remove_2015_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        df=df,
                        out_base_path=str(out_base),
                        quality=quality if quality else None,
                        warnings=stage_warnings if stage_warnings else None,
                        inputs={"prev_outputs": list(artifacts.keys())},
                        repo_dir=get_path(cfg, "base_dir"),
                    )
                    save_meta(out_base, meta, force=force)

                logger.info(f"[{stage_name}] {out_name} 저장 완료: {len(df):,}행, {len(df.columns)}컬럼")

            logger.info(f"[{stage_name}] 완료")

        except Exception as e:
            logger.error(f"[{stage_name}] 오류 발생: {e}", exc_info=True)
            raise

    logger.info(f"\n{'='*80}")
    logger.info("✅ 파이프라인 실행 완료")
    logger.info(f"{'='*80}")
    print("\n다음 단계: 2015년 데이터 제거 검증")
    print("  python scripts/remove_2015_data.py")

if __name__ == "__main__":
    main()
