import pandas as pd


def load_bt_metrics(strategy_name):
    """전략별 백테스트 메트릭스 로드"""
    file_path = f"data/interim/bt_metrics_{strategy_name}.csv"
    df = pd.read_csv(file_path)
    return df


def calculate_ic_metrics(df_metrics):
    """IC 관련 메트릭스 계산"""
    ic = df_metrics["ic"].iloc[0]
    rank_ic = df_metrics["rank_ic"].iloc[0]
    icir = df_metrics["icir"].iloc[0]
    rank_icir = df_metrics["rank_icir"].iloc[0]

    return {"ic": ic, "rank_ic": rank_ic, "icir": icir, "rank_icir": rank_icir}


# 전략 리스트
strategies = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]
strategy_names = {
    "bt20_short": "BT20 단기 (20일)",
    "bt20_ens": "BT20 앙상블 (20일)",
    "bt120_long": "BT120 장기 (120일)",
    "bt120_ens": "BT120 앙상블 (120일)",
}

print("=== Holdout 기간 IC 성과 지표 ===")

# 각 전략별 Holdout IC 메트릭스 계산
ic_results = []
for strategy in strategies:
    df = load_bt_metrics(strategy)
    holdout_data = df[df["phase"] == "holdout"]

    if len(holdout_data) > 0:
        ic_metrics = calculate_ic_metrics(holdout_data)

        result = {
            "strategy": strategy_names[strategy],
            "ic": ic_metrics["ic"],
            "rank_ic": ic_metrics["rank_ic"],
            "icir": ic_metrics["icir"],
            "rank_icir": ic_metrics["rank_icir"],
        }
        ic_results.append(result)

        print(f"{strategy_names[strategy]}:")
        print(".4f")
        print(".4f")
        print(".3f")
        print(".3f")
        print()

# DataFrame 생성
df_ic_results = pd.DataFrame(ic_results)

# CSV 및 Parquet 저장
df_ic_results.to_csv("data/holdout_ic_metrics.csv", index=False)
df_ic_results.to_parquet("data/holdout_ic_metrics.parquet", index=False)

print("IC 메트릭스 저장 완료:")
print("- data/holdout_ic_metrics.csv")
print("- data/holdout_ic_metrics.parquet")
print()

# Track A와 Track B 비교를 위한 종합 분석
print("=== Track A vs Track B 성과 비교 ===")

# Track A 현재 성과 (Hit Ratio)
track_a_hit_ratio = pd.read_csv("data/track_a_performance_metrics.csv")

print("Track A (랭킹 엔진) - Hit Ratio:")
for idx, row in track_a_hit_ratio.iterrows():
    print(f"  {row['metric']}: {row['value']} ({row['achievement']})")

print("\nTrack B (백테스트 전략) - IC 메트릭스:")
ic_summary = df_ic_results[["strategy", "rank_ic", "rank_icir"]].copy()
ic_summary["rank_ic"] = ic_summary["rank_ic"].round(4)
ic_summary["rank_icir"] = ic_summary["rank_icir"].round(3)
print(ic_summary.to_string(index=False))

print("\nTrack B (백테스트 전략) - Holdout 종합 성과:")
holdout_data = pd.read_csv("data/holdout_performance_metrics.csv")
holdout_summary = holdout_data[["strategy", "sharpe_ratio", "cagr", "hit_ratio"]].copy()
holdout_summary["sharpe_ratio"] = holdout_summary["sharpe_ratio"].round(3)
holdout_summary["cagr"] = (holdout_summary["cagr"] * 100).round(1).astype(str) + "%"
holdout_summary["hit_ratio"] = (holdout_summary["hit_ratio"] * 100).round(1).astype(
    str
) + "%"
print(holdout_summary.to_string(index=False))

# 최종 종합 보고
print("\n=== 최종 성과 비교표 ===")
print(
    "| 전략 | Hit Ratio (Track A) | Rank IC (Track B) | Rank ICIR (Track B) | Sharpe (Track B) |"
)
print(
    "|------|-------------------|------------------|-------------------|-----------------|"
)

for idx, row in holdout_data.iterrows():
    strategy = row["strategy"]
    hit_ratio = ".1f"
    rank_ic = ".4f"
    rank_icir = ".3f"
    sharpe = ".3f"

    print(f"| {strategy} | {hit_ratio} | {rank_ic} | {rank_icir} | {sharpe} |")

print("\n💡 분석:")
print("- Rank IC: 랭킹 스코어와 실제 수익률 간 상관관계 (양수일수록 좋음)")
print("- Rank ICIR: Rank IC의 정보 비율 (1.0 이상이면 우수)")
print("- Hit Ratio: 예측 정확도 (50% 이상이면 의미있음)")
print("- Track A의 Hit Ratio와 Track B의 Rank IC가 서로 검증됨")
