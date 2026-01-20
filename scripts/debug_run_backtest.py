"""run_backtest_4models.py 디버깅"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import load_artifact

# 설정 로드
cfg = load_config("configs/config.yaml")
interim_dir = Path(get_path(cfg, "data_interim"))

# 데이터 로드
dataset_daily = load_artifact(interim_dir / "dataset_daily.parquet")
cv_folds_short = load_artifact(interim_dir / "cv_folds_short.parquet")
cv_folds_long = load_artifact(interim_dir / "cv_folds_long.parquet")
universe_monthly = load_artifact(
    interim_dir / "universe_k200_membership_monthly.parquet"
)

# L6: 스코어 생성
l6_cfg = cfg.get("l6", {}) or {}
rebalance_scores, summary, quality, warns = build_rebalance_scores(
    pred_short_oos=load_artifact(interim_dir / "pred_short_oos.parquet"),
    pred_long_oos=load_artifact(interim_dir / "pred_long_oos.parquet"),
    universe_k200_membership_monthly=universe_monthly,
    weight_short=float(l6_cfg.get("weight_short", 0.5)),
    weight_long=float(l6_cfg.get("weight_long", 0.5)),
)

# bt20_ens 백테스트 실행
strategy_config = {
    "name": "bt20_ens",
    "config_section": "l7_bt20_ens",
    "score_col": "score_ens",
    "return_col": "true_short",
}
config_section = cfg.get("l7_bt20_ens", {})

bt_cfg = BacktestConfig(
    holding_days=int(config_section.get("holding_days", 20)),
    top_k=int(config_section.get("top_k", 20)),
    cost_bps=float(config_section.get("cost_bps", 10.0)),
    score_col=strategy_config["score_col"],
    ret_col=strategy_config["return_col"],
    weighting=config_section.get("weighting", "equal"),
    softmax_temp=float(config_section.get("softmax_temperature", 1.0)),
    buffer_k=int(config_section.get("buffer_k", 0)),
)

result = run_backtest(
    rebalance_scores=rebalance_scores,
    cfg=bt_cfg,
)

print(f"result 길이: {len(result)}")
if len(result) >= 4:
    bt_metrics = result[3]
    print(f"\nbt_metrics 타입: {type(bt_metrics)}")
    print(
        f"bt_metrics 길이: {len(bt_metrics) if hasattr(bt_metrics, '__len__') else 'N/A'}"
    )

    if isinstance(bt_metrics, pd.DataFrame):
        print(f"\nbt_metrics 컬럼: {bt_metrics.columns.tolist()}")
        print("\nbt_metrics 전체:")
        print(bt_metrics.to_string())

        holdout = bt_metrics[bt_metrics["phase"] == "holdout"]
        print("\nholdout 필터링 후:")
        print(holdout.to_string())
        if len(holdout) > 0:
            print(f"\nholdout 첫 번째 행의 net_sharpe: {holdout['net_sharpe'].iloc[0]}")
            print(f"holdout 첫 번째 행의 net_cagr: {holdout['net_cagr'].iloc[0]}")
            print(f"holdout 첫 번째 행의 net_mdd: {holdout['net_mdd'].iloc[0]}")
            print(
                f"holdout 첫 번째 행의 net_calmar_ratio: {holdout['net_calmar_ratio'].iloc[0]}"
            )
