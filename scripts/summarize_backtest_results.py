"""백테스트 결과 요약"""
from pathlib import Path

import pandas as pd

base_dir = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")
strategies = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]

print("=" * 100)
print("백테스트 결과 요약 (backtest_settings_mapping.md 설정 적용 후)")
print("=" * 100)
print()

results = []
for strategy in strategies:
    metrics_path = base_dir / "data" / "interim" / f"bt_metrics_{strategy}.parquet"
    if not metrics_path.exists():
        continue

    df = pd.read_parquet(metrics_path)

    for phase in ["dev", "holdout"]:
        phase_df = df[df["phase"] == phase]
        if len(phase_df) == 0:
            continue

        row = phase_df.iloc[0]
        results.append(
            {
                "strategy": strategy,
                "phase": phase,
                "net_sharpe": row["net_sharpe"],
                "net_cagr": row["net_cagr"],
                "net_mdd": row["net_mdd"],
                "net_calmar_ratio": row["net_calmar_ratio"],
            }
        )

df_results = pd.DataFrame(results)

# Holdout 구간 결과
print("Holdout 구간 결과:")
print("-" * 100)
holdout = df_results[df_results["phase"] == "holdout"]
for _, row in holdout.iterrows():
    print(
        f"{row['strategy']:15s} | Sharpe: {row['net_sharpe']:7.4f} | CAGR: {row['net_cagr']:7.4f} | MDD: {row['net_mdd']:7.4f} | Calmar: {row['net_calmar_ratio']:7.4f}"
    )

print("\nDev 구간 결과:")
print("-" * 100)
dev = df_results[df_results["phase"] == "dev"]
for _, row in dev.iterrows():
    print(
        f"{row['strategy']:15s} | Sharpe: {row['net_sharpe']:7.4f} | CAGR: {row['net_cagr']:7.4f} | MDD: {row['net_mdd']:7.4f} | Calmar: {row['net_calmar_ratio']:7.4f}"
    )

# CSV로 저장
output_path = base_dir / "artifacts" / "reports" / "backtest_results_current.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(output_path, index=False)
print(f"\n결과 저장: {output_path}")
