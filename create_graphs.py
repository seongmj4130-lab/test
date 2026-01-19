import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
df = pd.read_csv("data/ui_strategies_cumulative_comparison.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.set_index("month")

# 백분율로 변환
df_pct = df / 100

print("데이터 확인:")
print(df_pct.head())

# 색상 팔레트 설정
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

# 그래프 1: 20일 랭킹 비교
plt.figure(figsize=(14, 8))

# 20일 전략들 선택
strategies_20 = ["kospi200", "bt20_short_20", "bt120_long_20", "bt20_ens_20"]
labels_20 = ["KOSPI200", "BT20 단기 (20일)", "BT120 장기 (20일)", "BT20 앙상블 (20일)"]

for i, (strategy, label) in enumerate(zip(strategies_20, labels_20)):
    plt.plot(
        df_pct.index,
        df_pct[strategy] * 100,
        label=label,
        linewidth=2.5,
        color=colors[i],
        marker="o",
        markersize=4,
    )

plt.title("20일 랭킹 전략 비교 (2023-2024)", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("기간", fontsize=12)
plt.ylabel("누적 수익률 (%)", fontsize=12)
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# 첫 번째 그래프 저장
plt.savefig("20day_ranking_comparison.png", dpi=300, bbox_inches="tight")
print("20일 랭킹 그래프 저장 완료")

# 그래프 2: 40일 랭킹 비교
plt.figure(figsize=(14, 8))

# 40일 전략들 선택
strategies_40 = ["kospi200", "bt20_short_40", "bt120_long_40", "bt20_ens_40"]
labels_40 = ["KOSPI200", "BT20 단기 (40일)", "BT120 장기 (40일)", "BT20 앙상블 (40일)"]

for i, (strategy, label) in enumerate(zip(strategies_40, labels_40)):
    plt.plot(
        df_pct.index,
        df_pct[strategy] * 100,
        label=label,
        linewidth=2.5,
        color=colors[i],
        marker="s",
        markersize=4,
    )

plt.title("40일 랭킹 전략 비교 (2023-2024)", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("기간", fontsize=12)
plt.ylabel("누적 수익률 (%)", fontsize=12)
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# 두 번째 그래프 저장
plt.savefig("40day_ranking_comparison.png", dpi=300, bbox_inches="tight")
print("40일 랭킹 그래프 저장 완료")

print("그래프 생성 완료!")
print("생성된 파일:")
print("- 20day_ranking_comparison.png")
print("- 40day_ranking_comparison.png")
