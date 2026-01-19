# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/utils/artifact_contract.py
"""
아티팩트 계약 정의 (Artifact Contract)

각 Stage의 입력/출력 아티팩트에 대한 계약을 딕셔너리/상수로 정의합니다.
"""
from typing import Dict, List, Optional

# Stage별 필수 입력
REQUIRED_INPUTS: Dict[str, List[str]] = {
    "L0": [],
    "L1": ["universe_k200_membership_monthly"],
    "L1B": ["universe_k200_membership_monthly"],
    "L1D": ["rebalance_scores"],  # [Stage5] L6 이후에 실행되어야 함
    "L2": ["universe_k200_membership_monthly"],
    "L3": ["ohlcv_daily", "fundamentals_annual"],
    "L4": ["panel_merged_daily"],
    "L5": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L6": ["pred_short_oos", "pred_long_oos", "universe_k200_membership_monthly"],
    "L7": ["rebalance_scores"],
    "L7B": ["rebalance_scores"],
    "L7C": ["bt_returns", "rebalance_scores"],
    "L7D": ["bt_returns", "bt_equity_curve"],
    "L8": ["dataset_daily"],  # dataset_daily 또는 panel_merged_daily 둘 중 하나만 있으면 됨
    "L11": ["ranking_daily", "ohlcv_daily"],
}

# Stage별 옵션 입력
OPTIONAL_INPUTS: Dict[str, List[str]] = {
    "L7": ["market_regime"],  # L1D에서 생성, L5 이후 실행
    "L8": ["panel_merged_daily"],  # dataset_daily 또는 panel_merged_daily 둘 중 하나만 있으면 됨
}

# Stage별 필수 출력
REQUIRED_OUTPUTS: Dict[str, List[str]] = {
    "L0": ["universe_k200_membership_monthly"],
    "L1": ["ohlcv_daily"],
    "L1B": ["sector_map"],
    "L1D": ["market_regime"],
    "L2": ["fundamentals_annual"],
    "L3": ["panel_merged_daily"],
    "L4": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L5": ["pred_short_oos", "pred_long_oos", "model_metrics"],
    "L6": ["rebalance_scores", "rebalance_scores_summary"],
    "L7": [
        "bt_positions",
        "bt_returns",
        "bt_equity_curve",
        "bt_metrics",
        "selection_diagnostics",
        "bt_returns_diagnostics",
    ],
    "L7B": ["bt_sensitivity"],
    "L7C": ["bt_vs_benchmark", "bt_benchmark_compare", "bt_benchmark_returns"],
    "L7D": ["l7d_stability", "bt_yearly_metrics", "bt_rolling_sharpe", "bt_drawdown_events"],
    "L8": ["ranking_daily", "ranking_snapshot"],
    "L11": ["ui_top_bottom_daily", "ui_equity_curves", "ui_snapshot", "ui_metrics"],
}

# Stage별 옵션 출력
OPTIONAL_OUTPUTS: Dict[str, List[str]] = {
    # 현재 옵션 출력 없음
}

# 아티팩트별 필수 컬럼
REQUIRED_COLS_BY_OUTPUT: Dict[str, List[str]] = {
    "universe_k200_membership_monthly": ["date", "ticker"],
    "ohlcv_daily": ["date", "ticker"],
    "sector_map": ["date", "ticker", "sector_name"],
    "market_regime": ["date", "regime"],
    "fundamentals_annual": ["date", "ticker"],
    "panel_merged_daily": ["date", "ticker"],
    "dataset_daily": ["date", "ticker"],
    "cv_folds_short": ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"],
    "cv_folds_long": ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"],
    "pred_short_oos": ["date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"],
    "pred_long_oos": ["date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"],
    "model_metrics": ["horizon", "phase", "rmse"],
    "rebalance_scores": ["date", "ticker", "phase"],
    "rebalance_scores_summary": [
        "date",
        "phase",
        "n_tickers",
        "coverage_vs_universe_pct",
        "score_short_missing",
        "score_long_missing",
        "score_ens_missing",
    ],
    "bt_positions": ["date", "phase", "ticker"],
    "bt_returns": ["date", "phase"],
    "bt_equity_curve": ["date", "phase"],
    "bt_metrics": ["phase", "net_total_return", "net_sharpe", "net_mdd"],
    "bt_sensitivity": ["phase", "net_total_return", "net_sharpe", "net_mdd"],
    "bt_vs_benchmark": ["date", "phase", "bench_return", "excess_return"],
    "bt_benchmark_compare": ["phase", "tracking_error_ann", "information_ratio"],
    "bt_benchmark_returns": ["date", "phase", "bench_return"],
    "ranking_daily": ["date", "ticker", "score_total", "rank_total"],
    "ui_top_bottom_daily": ["date", "top_list", "bottom_list"],
    "ui_equity_curves": [
        "date",
        "strategy_ret",
        "bench_ret",
        "strategy_equity",
        "bench_equity",
        "excess_equity",
    ],
    "ui_snapshot": ["snapshot_date", "snapshot_type", "snapshot_rank", "ticker", "rank_total"],
    "ui_metrics": ["total_return", "cagr", "vol", "sharpe", "mdd"],
    "l7d_stability": [
        "phase",
        "year",
        "n_rebalances",
        "net_total_return",
        "net_vol_ann",
        "net_sharpe",
        "net_mdd",
        "net_hit_ratio",
        "date_start",
        "date_end",
        "net_return_col_used",
    ],
    "bt_yearly_metrics": [
        "phase",
        "year",
        "n_rebalances",
        "net_total_return",
        "net_vol_ann",
        "net_sharpe",
        "net_mdd",
        "net_hit_ratio",
        "date_start",
        "date_end",
        "net_return_col_used",
    ],
    "bt_rolling_sharpe": ["date", "phase", "net_rolling_sharpe"],
    "bt_drawdown_events": ["phase", "peak_date", "trough_date", "drawdown", "length_days"],
    "selection_diagnostics": ["date", "phase", "top_k", "eligible_count", "selected_count"],
    "bt_returns_diagnostics": ["date", "phase"],
}

# 아티팩트별 옵션 컬럼
OPTIONAL_COLS_BY_OUTPUT: Dict[str, List[str]] = {
    "rebalance_scores": ["score_short", "score_long", "score_ens"],
    "bt_positions": ["sector_name"],  # diversify.enabled일 때만
    "bt_returns": ["net_return", "gross_return"],
    "bt_equity_curve": ["net_equity", "gross_equity"],
    "bt_returns_diagnostics": ["regime", "exposure"],  # 결측 허용
}

# Track 구분
TRACK_BY_STAGE: Dict[str, str] = {
    "L0": "pipeline",
    "L1": "pipeline",
    "L1B": "pipeline",
    "L1D": "pipeline",
    "L2": "pipeline",
    "L3": "pipeline",
    "L4": "pipeline",
    "L5": "pipeline",
    "L6": "pipeline",
    "L7": "pipeline",
    "L7B": "pipeline",
    "L7C": "pipeline",
    "L7D": "pipeline",
    "L8": "ranking",
    "L11": "ranking",
}

# L2 재사용 규칙 (예외)
L2_REUSE_RULE = {
    "always_reuse": True,
    "skip_if_exists_exception": True,
    "storage_location": "base_interim_dir",  # run_tag 폴더 아님
    "artifact_name": "fundamentals_annual",
    "dart_api_call_forbidden": True,
}

# L8 입력 선택 규칙
L8_INPUT_RULE = {
    "required_one_of": ["dataset_daily", "panel_merged_daily"],
    "description": "dataset_daily 또는 panel_merged_daily 둘 중 하나만 있으면 됨",
}

# L11 입력 로드 규칙
L11_INPUT_RULE = {
    "ranking_daily": {
        "load_from": "baseline_tag",
        "fallback": "latest_in_base_interim_dir",
        "no_scan_error": True,
    },
    "ohlcv_daily": {
        "load_from": "base_interim_dir",
        "fallback": None,
    },
}

def get_stage_track(stage_name: str) -> str:
    """Stage의 Track 반환"""
    return TRACK_BY_STAGE.get(stage_name, "unknown")

def get_required_inputs(stage_name: str) -> List[str]:
    """Stage의 필수 입력 반환"""
    return REQUIRED_INPUTS.get(stage_name, [])

def get_optional_inputs(stage_name: str) -> List[str]:
    """Stage의 옵션 입력 반환"""
    return OPTIONAL_INPUTS.get(stage_name, [])

def get_required_outputs(stage_name: str) -> List[str]:
    """Stage의 필수 출력 반환"""
    return REQUIRED_OUTPUTS.get(stage_name, [])

def get_optional_outputs(stage_name: str) -> List[str]:
    """Stage의 옵션 출력 반환"""
    return OPTIONAL_OUTPUTS.get(stage_name, [])

def get_required_columns(artifact_name: str) -> List[str]:
    """아티팩트의 필수 컬럼 반환"""
    return REQUIRED_COLS_BY_OUTPUT.get(artifact_name, [])

def get_optional_columns(artifact_name: str) -> List[str]:
    """아티팩트의 옵션 컬럼 반환"""
    return OPTIONAL_COLS_BY_OUTPUT.get(artifact_name, [])

def is_pipeline_track(stage_name: str) -> bool:
    """Pipeline Track 여부 확인"""
    return get_stage_track(stage_name) == "pipeline"

def is_ranking_track(stage_name: str) -> bool:
    """Ranking Track 여부 확인"""
    return get_stage_track(stage_name) == "ranking"
