import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정 (Windows용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/dummy_excess_return_monthly_2023_2024_by_horizon_and_rank.csv')

# 날짜 형식 변환
df['month'] = pd.to_datetime(df['month'])
df = df.sort_values(['horizon_days', 'rank_strategy', 'month'])

# 20일과 40일 데이터만 필터링
df_filtered = df[df['horizon_days'].isin([20, 40])]

# 그래프 생성
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 전략별 색상 매핑
strategy_colors = {
    '단기랭킹': '#1f77b4',      # 파란색
    '장기랭킹': '#ff7f0e',      # 주황색
    '통합랭킹(5:5)': '#2ca02c'  # 초록색
}

# 각 호라이즌별로 그래프 그리기
for i, horizon in enumerate([20, 40]):
    ax = axes[i]

    # 해당 호라이즌 데이터 필터링
    horizon_data = df_filtered[df_filtered['horizon_days'] == horizon]

    # 각 전략별로 선 그래프
    for strategy in strategy_colors.keys():
        strategy_data = horizon_data[horizon_data['rank_strategy'] == strategy]
        if not strategy_data.empty:
            ax.plot(strategy_data['month'], strategy_data['excess_return_pct'],
                   label=strategy, color=strategy_colors[strategy], linewidth=2, marker='o')

    # 그래프 설정
    ax.set_title(f'{horizon}일 보유 기간 전략별 초과 수익률', fontsize=14, fontweight='bold')
    ax.set_xlabel('기간', fontsize=12)
    ax.set_ylabel('초과 수익률 (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # x축 날짜 포맷팅
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 전체 제목
fig.suptitle('2023-2024 전략별 초과 수익률 비교 (20일 vs 40일 보유)', fontsize=16, fontweight='bold', y=0.98)

# 레이아웃 조정
plt.tight_layout()
plt.savefig('data/excess_return_20_40_days_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print('그래프가 생성되었습니다: excess_return_20_40_days_comparison.png')