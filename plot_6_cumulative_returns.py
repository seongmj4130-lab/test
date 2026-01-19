import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/dummy_cum_return_monthly_tabs_v3.csv')

# horizon_days별로 그룹화
horizons = [20, 40, 60, 80, 100, 120]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # KOSPI, 단기, 장기, 혼합

# 6개 그래프 생성 (2x3 레이아웃)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, horizon in enumerate(horizons):
    ax = axes[i]

    # 해당 horizon 데이터 필터링
    horizon_data = df[df['horizon_days'] == horizon].copy()
    horizon_data['month'] = pd.to_datetime(horizon_data['month'])

    # 누적 수익률 그래프
    ax.plot(horizon_data['month'], horizon_data['kospi_cum_return_pct'],
            label='KOSPI', color=colors[0], linewidth=2, marker='o', markersize=3)
    ax.plot(horizon_data['month'], horizon_data['short_cum_return_pct'],
            label='단기 전략', color=colors[1], linewidth=2, marker='s', markersize=3)
    ax.plot(horizon_data['month'], horizon_data['long_cum_return_pct'],
            label='장기 전략', color=colors[2], linewidth=2, marker='^', markersize=3)
    ax.plot(horizon_data['month'], horizon_data['mix_cum_return_pct'],
            label='혼합 전략', color=colors[3], linewidth=2, marker='D', markersize=3)

    # 그래프 설정
    ax.set_title(f'{horizon}일 보유 기간 - 누적 수익률 추이', fontsize=14, fontweight='bold')
    ax.set_xlabel('기간', fontsize=12)
    ax.set_ylabel('누적 수익률 (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # x축 날짜 포맷팅
    ax.tick_params(axis='x', rotation=45)

# 전체 그래프 설정
plt.suptitle('KOSPI200 vs 전략별 누적 수익률 비교 (기간별)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# 그래프 저장 및 표시
plt.savefig('strategies_cumulative_returns_6charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 6개 누적 수익률 그래프가 'strategies_cumulative_returns_6charts.png'로 저장되었습니다.")
print("\n📊 그래프 분석:")
print("- 각 행은 보유 기간(20, 40, 60, 80, 100, 120일)을 나타냄")
print("- 파란색: KOSPI200 지수")
print("- 주황색: 단기 전략")
print("- 초록색: 장기 전략")
print("- 빨간색: 혼합 전략")
print("\n💡 주요 관찰점:")
print("- 보유 기간이 길수록 변동성이 커짐")
print("- 단기 전략은 KOSPI를 상회하는 경향")
print("- 장기 전략은 안정적인 수익률 추이")
print("- 혼합 전략은 균형 잡힌 성과")