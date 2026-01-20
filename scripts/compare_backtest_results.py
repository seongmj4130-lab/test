"""백테스트 결과 비교 (03_code vs 06_code22 vs 기대값)"""

from pathlib import Path

import pandas as pd

strategies = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]

# 기대값 (backtest_settings_mapping.md)
expected = {
    "bt20_ens": {
        "net_sharpe": 0.6826,
        "net_cagr": 0.1498,
        "net_mdd": -0.1098,
        "net_calmar_ratio": 1.3641,
    },
    "bt20_short": {
        "net_sharpe": 0.6464,
        "net_cagr": 0.1384,
        "net_mdd": -0.0909,
        "net_calmar_ratio": 1.5223,
    },
    "bt120_ens": {
        "net_sharpe": 0.6263,
        "net_cagr": 0.1166,
        "net_mdd": -0.0769,
        "net_calmar_ratio": 1.5156,
    },
    "bt120_long": {
        "net_sharpe": 0.6839,
        "net_cagr": 0.1360,
        "net_mdd": -0.0866,
        "net_calmar_ratio": 1.5700,
    },
}

# 03_code 결과
results_03 = {}
interim_dir_03 = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/data/interim")
for s in strategies:
    path = interim_dir_03 / f"bt_metrics_{s}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        holdout = df[df["phase"] == "holdout"]
        if len(holdout) > 0:
            results_03[s] = holdout.iloc[0].to_dict()

# 06_code22 결과
results_06 = {}
interim_dir_06 = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/06_code22/data/interim")
for s in strategies:
    path = interim_dir_06 / f"bt_metrics_{s}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        holdout = df[df["phase"] == "holdout"]
        if len(holdout) > 0:
            results_06[s] = holdout.iloc[0].to_dict()

print("=" * 100)
print("Holdout 구간 결과 비교 (03_code vs 06_code22 vs 기대값)")
print("=" * 100)

print(
    f"\n{'전략':<15} {'구분':<12} {'Net Sharpe':<12} {'Net CAGR':<12} {'Net MDD':<12} {'Calmar Ratio':<12}"
)
print("-" * 100)

for s in strategies:
    if s in expected:
        exp = expected[s]
        print(
            f"{s:<15} {'기대값':<12} {exp['net_sharpe']:>11.4f} {exp['net_cagr']:>11.2%} {exp['net_mdd']:>11.2%} {exp['net_calmar_ratio']:>11.4f}"
        )

    if s in results_03:
        r03 = results_03[s]
        print(
            f"{s:<15} {'03_code':<12} {r03['net_sharpe']:>11.4f} {r03['net_cagr']:>11.2%} {r03['net_mdd']:>11.2%} {r03['net_calmar_ratio']:>11.4f}"
        )

    if s in results_06:
        r06 = results_06[s]
        print(
            f"{s:<15} {'06_code22':<12} {r06['net_sharpe']:>11.4f} {r06['net_cagr']:>11.2%} {r06['net_mdd']:>11.2%} {r06['net_calmar_ratio']:>11.4f}"
        )

    print()

print("=" * 100)
print("차이 분석 (03_code - 기대값)")
print("=" * 100)
print(
    f"\n{'전략':<15} {'Net Sharpe':<12} {'Net CAGR':<12} {'Net MDD':<12} {'Calmar Ratio':<12}"
)
print("-" * 100)

for s in strategies:
    if s in results_03 and s in expected:
        r03 = results_03[s]
        exp = expected[s]
        sharpe_diff = r03["net_sharpe"] - exp["net_sharpe"]
        cagr_diff = r03["net_cagr"] - exp["net_cagr"]
        mdd_diff = r03["net_mdd"] - exp["net_mdd"]
        calmar_diff = r03["net_calmar_ratio"] - exp["net_calmar_ratio"]
        print(
            f"{s:<15} {sharpe_diff:>+11.4f} {cagr_diff:>+11.2%} {mdd_diff:>+11.2%} {calmar_diff:>+11.4f}"
        )

print("\n" + "=" * 100)
print("차이 분석 (06_code22 - 기대값)")
print("=" * 100)
print(
    f"\n{'전략':<15} {'Net Sharpe':<12} {'Net CAGR':<12} {'Net MDD':<12} {'Calmar Ratio':<12}"
)
print("-" * 100)

for s in strategies:
    if s in results_06 and s in expected:
        r06 = results_06[s]
        exp = expected[s]
        sharpe_diff = r06["net_sharpe"] - exp["net_sharpe"]
        cagr_diff = r06["net_cagr"] - exp["net_cagr"]
        mdd_diff = r06["net_mdd"] - exp["net_mdd"]
        calmar_diff = r06["net_calmar_ratio"] - exp["net_calmar_ratio"]
        print(
            f"{s:<15} {sharpe_diff:>+11.4f} {cagr_diff:>+11.2%} {mdd_diff:>+11.2%} {calmar_diff:>+11.4f}"
        )

print("\n" + "=" * 100)
print("차이 분석 (03_code - 06_code22)")
print("=" * 100)
print(
    f"\n{'전략':<15} {'Net Sharpe':<12} {'Net CAGR':<12} {'Net MDD':<12} {'Calmar Ratio':<12}"
)
print("-" * 100)

for s in strategies:
    if s in results_03 and s in results_06:
        r03 = results_03[s]
        r06 = results_06[s]
        sharpe_diff = r03["net_sharpe"] - r06["net_sharpe"]
        cagr_diff = r03["net_cagr"] - r06["net_cagr"]
        mdd_diff = r03["net_mdd"] - r06["net_mdd"]
        calmar_diff = r03["net_calmar_ratio"] - r06["net_calmar_ratio"]
        print(
            f"{s:<15} {sharpe_diff:>+11.4f} {cagr_diff:>+11.2%} {mdd_diff:>+11.2%} {calmar_diff:>+11.4f}"
        )
