import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
df = pd.read_csv('data/ui_strategies_cumulative_comparison.csv')
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)

# 20일과 40일 전략만 선택 (KOSPI200 포함)
strategies_20_40 = [
    'kospi200',
    'bt20_short_20', 'bt20_short_40',
    'bt120_long_20', 'bt120_long_40',
    'bt20_ens_20', 'bt20_ens_40'
]

# 색상 설정
colors = {
    'kospi200': '#2E86AB',
    'bt20_short_20': '#F24236',
    'bt20_short_40': '#F24236',
    'bt120_long_20': '#4CAF50',
    'bt120_long_40': '#4CAF50',
    'bt20_ens_20': '#FF9800',
    'bt20_ens_40': '#FF9800'
}

# 선 스타일 설정
linestyles = {
    'kospi200': '-',
    'bt20_short_20': '-',
    'bt20_short_40': '--',
    'bt120_long_20': '-',
    'bt120_long_40': '--',
    'bt20_ens_20': '-',
    'bt20_ens_40': '--'
}

# 라벨 설정
labels = {
    'kospi200': 'KOSPI200',
    'bt20_short_20': 'BT20 Short (20일)',
    'bt20_short_40': 'BT20 Short (40일)',
    'bt120_long_20': 'BT120 Long (20일)',
    'bt120_long_40': 'BT120 Long (40일)',
    'bt20_ens_20': 'BT20 Ensemble (20일)',
    'bt20_ens_40': 'BT20 Ensemble (40일)'
}

# 그래프 생성
plt.figure(figsize=(14, 8))

for strategy in strategies_20_40:
    if strategy in df.columns:
        plt.plot(df.index, df[strategy],
                color=colors[strategy],
                linestyle=linestyles[strategy],
                linewidth=2.5 if strategy == 'kospi200' else 2,
                label=labels[strategy],
                alpha=0.9)

# 그래프 스타일링
plt.title('전략별 누적수익률 비교 (20일 vs 40일)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('기간', fontsize=12)
plt.ylabel('누적수익률 (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

# x축 포맷팅
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# y축 포맷팅
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

# 배경색 설정
plt.gca().set_facecolor('#f8f9fa')

# 테두리 설정
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#cccccc')

plt.tight_layout()

# 그래프 저장
plt.savefig('data/strategy_comparison_20_40_days.png', dpi=300, bbox_inches='tight')
plt.savefig('data/strategy_comparison_20_40_days.pdf', bbox_inches='tight')

print("📊 그래프 생성 완료!")
print("💾 저장 위치: data/strategy_comparison_20_40_days.png")
print("💾 저장 위치: data/strategy_comparison_20_40_days.pdf")

# 전략별 최종 수익률 출력
print("\n🏆 전략별 최종 수익률 (2024년 12월):")
print("-" * 40)
for strategy in strategies_20_40:
    if strategy in df.columns:
        final_return = df[strategy].iloc[-1]
        print(f"{labels[strategy]:<20}: {final_return:>6.1f}%")

plt.show()
