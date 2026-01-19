import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 한글 폰트 설정 (Windows용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/dummy_excess_return_monthly_2023_2024_by_horizon_and_rank.csv')

# 날짜 형식 변환
df['month'] = pd.to_datetime(df['month'])
df = df.sort_values(['horizon_days', 'rank_strategy', 'month'])

# 모든 호라이즌 데이터 사용 (20, 40, 60, 80, 100, 120)
all_horizons = sorted(df['horizon_days'].unique())

# 2x3 그리드로 서브플롯 생성
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 전략별 색상 매핑
strategy_colors = {
    '단기랭킹': '#1f77b4',      # 파란색
    '장기랭킹': '#ff7f0e',      # 주황색
    '통합랭킹(5:5)': '#2ca02c'  # 초록색
}

# 각 호라이즌별로 그래프 그리기
for i, horizon in enumerate(all_horizons):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    # 해당 호라이즌 데이터 필터링
    horizon_data = df[df['horizon_days'] == horizon]

    # 각 전략별로 선 그래프
    for strategy in strategy_colors.keys():
        strategy_data = horizon_data[horizon_data['rank_strategy'] == strategy]
        if not strategy_data.empty:
            ax.plot(strategy_data['month'], strategy_data['excess_return_pct'],
                   label=strategy, color=strategy_colors[strategy], linewidth=2, marker='o', markersize=3)

    # 그래프 설정
    ax.set_title(f'{horizon}일 보유 기간', fontsize=14, fontweight='bold')
    ax.set_xlabel('기간', fontsize=10)
    ax.set_ylabel('초과 수익률 (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # x축 날짜 포맷팅
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

# 전체 제목
fig.suptitle('2023-2024 전략별 초과 수익률 비교 (20~120일 보유 기간)', fontsize=16, fontweight='bold', y=0.95)

# 레이아웃 조정
plt.tight_layout()
plt.savefig('data/excess_return_all_horizons_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print('그래프가 생성되었습니다: excess_return_all_horizons_comparison.png')

# 추가 분석: 각 호라이즌별 최종 수익률 요약
print("\n=== 각 호라이즌별 최종 수익률 요약 (2024-12) ===")
final_returns = df[df['month'] == '2024-12'].pivot_table(
    values='excess_return_pct',
    index='horizon_days',
    columns='rank_strategy'
).round(2)

print(final_returns)

# 최고 성과 전략 분석
print("\n=== 최고 성과 전략 분석 ===")
for horizon in all_horizons:
    horizon_data = final_returns.loc[horizon]
    best_strategy = horizon_data.idxmax()
    best_return = horizon_data.max()
    worst_strategy = horizon_data.idxmin()
    worst_return = horizon_data.min()

    print(f"{horizon}일: 최고 {best_strategy}({best_return}%), 최저 {worst_strategy}({worst_return}%)")