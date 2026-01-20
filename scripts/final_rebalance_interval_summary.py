"""
rebalance_interval 적용 최종 결과 요약
"""

from pathlib import Path

import pandas as pd

data_dir = Path("data/interim")

print("=" * 80)
print("rebalance_interval 적용 최종 결과 요약")
print("=" * 80)

models = {
    "bt20_short": {"interval": 20, "expected": "약 123개"},
    "bt20_ens": {"interval": 20, "expected": "약 123개"},
    "bt120_long": {"interval": 120, "expected": "약 20개"},
    "bt120_ens": {"interval": 120, "expected": "약 20개"},
}

results = []

for model_name, info in models.items():
    # rebalance_scores 확인
    scores_path = (
        data_dir / f"rebalance_scores_from_ranking_interval_{info['interval']}.parquet"
    )
    if scores_path.exists():
        scores_df = pd.read_parquet(scores_path)
        scores_dates = scores_df["date"].nunique()
    else:
        scores_dates = 0

    # 백테스트 결과 확인
    returns_path = data_dir / f"bt_returns_{model_name}.parquet"
    metrics_path = data_dir / f"bt_metrics_{model_name}.parquet"

    if returns_path.exists() and metrics_path.exists():
        returns_df = pd.read_parquet(returns_path)
        metrics_df = pd.read_parquet(metrics_path)

        total_dates = returns_df["date"].nunique()
        dev_dates = returns_df[returns_df["phase"] == "dev"]["date"].nunique()
        holdout_dates = returns_df[returns_df["phase"] == "holdout"]["date"].nunique()

        # 지표 추출
        dev_metrics = (
            metrics_df[metrics_df["phase"] == "dev"].iloc[0]
            if len(metrics_df[metrics_df["phase"] == "dev"]) > 0
            else None
        )
        holdout_metrics = (
            metrics_df[metrics_df["phase"] == "holdout"].iloc[0]
            if len(metrics_df[metrics_df["phase"] == "holdout"]) > 0
            else None
        )

        results.append(
            {
                "model": model_name,
                "rebalance_interval": info["interval"],
                "rebalance_scores_dates": scores_dates,
                "bt_dates_total": total_dates,
                "bt_dates_dev": dev_dates,
                "bt_dates_holdout": holdout_dates,
                "dev_net_cagr": (
                    dev_metrics["net_cagr"] if dev_metrics is not None else None
                ),
                "dev_net_sharpe": (
                    dev_metrics["net_sharpe"] if dev_metrics is not None else None
                ),
                "dev_net_mdd": (
                    dev_metrics["net_mdd"] if dev_metrics is not None else None
                ),
                "holdout_net_cagr": (
                    holdout_metrics["net_cagr"] if holdout_metrics is not None else None
                ),
                "holdout_net_sharpe": (
                    holdout_metrics["net_sharpe"]
                    if holdout_metrics is not None
                    else None
                ),
                "holdout_net_mdd": (
                    holdout_metrics["net_mdd"] if holdout_metrics is not None else None
                ),
            }
        )

# 결과 테이블 출력
df_results = pd.DataFrame(results)

print("\n리밸런싱 날짜 수:")
print(
    df_results[
        [
            "model",
            "rebalance_interval",
            "rebalance_scores_dates",
            "bt_dates_total",
            "bt_dates_dev",
            "bt_dates_holdout",
        ]
    ].to_string(index=False)
)

print("\n\n백테스트 지표 (Dev):")
print(
    df_results[["model", "dev_net_cagr", "dev_net_sharpe", "dev_net_mdd"]].to_string(
        index=False
    )
)

print("\n\n백테스트 지표 (Holdout):")
print(
    df_results[
        ["model", "holdout_net_cagr", "holdout_net_sharpe", "holdout_net_mdd"]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("요약")
print("=" * 80)
for _, row in df_results.iterrows():
    print(f"\n{row['model']} (rebalance_interval={row['rebalance_interval']}):")
    print(f"  rebalance_scores: {row['rebalance_scores_dates']}개 날짜")
    print(
        f"  백테스트: {row['bt_dates_total']}개 리밸런싱 (dev: {row['bt_dates_dev']}, holdout: {row['bt_dates_holdout']})"
    )
    print(
        f"  Dev: CAGR={row['dev_net_cagr']:.4f}, Sharpe={row['dev_net_sharpe']:.4f}, MDD={row['dev_net_mdd']:.4f}"
    )
    print(
        f"  Holdout: CAGR={row['holdout_net_cagr']:.4f}, Sharpe={row['holdout_net_sharpe']:.4f}, MDD={row['holdout_net_mdd']:.4f}"
    )
