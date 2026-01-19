import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
df = pd.read_csv("data/ui_strategies_cumulative_comparison.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.set_index("month")

# 백분율로 변환
df_pct = df / 100

print("3가지 전략 선택:")
strategies_3 = ["kospi200", "bt20_short_20", "bt120_long_20", "bt20_ens_20"]
labels_3 = ["KOSPI200", "BT20 단기 전략", "BT120 장기 전략", "BT20 앙상블 전략"]

print("전략들:", labels_3[1:])  # KOSPI200 제외하고 3가지 전략 표시

# 그래프 생성
plt.figure(figsize=(16, 9))

# 색상 팔레트 설정
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

for i, (strategy, label) in enumerate(zip(strategies_3, labels_3)):
    plt.plot(
        df_pct.index,
        df_pct[strategy] * 100,
        label=label,
        linewidth=3,
        color=colors[i],
        marker="o" if i < 2 else "s",
        markersize=5,
        alpha=0.8,
    )

plt.title(
    "3가지 주요 전략 누적 수익률 비교 (2023-2024)",
    fontsize=18,
    fontweight="bold",
    pad=20,
)
plt.xlabel("기간", fontsize=14)
plt.ylabel("누적 수익률 (%)", fontsize=14)
plt.legend(fontsize=12, loc="upper left", framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1)

# X축 레이블 회전 및 간격 조정
plt.xticks(rotation=45)
plt.xticks(df_pct.index[::2])  # 2개월 간격으로 표시

plt.tight_layout()

# 그래프 저장
plt.savefig("3_strategies_cumulative_returns.png", dpi=300, bbox_inches="tight")
print("3가지 전략 누적 수익률 그래프 저장 완료")

# 각 전략의 최종 수익률 출력
print("\n=== 각 전략 최종 누적 수익률 ===")
for strategy, label in zip(strategies_3, labels_3):
    final_return = df_pct[strategy].iloc[-1] * 100
    print(f"{label:15}: {final_return:6.1f}%")

print("\n그래프 생성 완료!")
print("생성된 파일: 3_strategies_cumulative_returns.png")
