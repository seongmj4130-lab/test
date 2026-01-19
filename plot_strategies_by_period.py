#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특정 기간 선택 시 3개 전략 + KOSPI200 비교 그래프
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import CheckButtons

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_strategies_by_selected_periods(selected_periods=['20', '60', '120']):
    """선택된 기간들에 대해 3개 전략 + KOSPI200 비교 그래프"""

    # 정정된 데이터 로드
    df = pd.read_csv('data/ui_strategies_cumulative_comparison_corrected.csv')
    df['month'] = pd.to_datetime(df['month'])

    print(f"=== 선택된 기간 {selected_periods}에 대한 전략 비교 그래프 생성 ===")

    # 전략 색상 설정
    strategy_colors = {
        'BT20 단기': '#1f77b4',
        'BT120 장기': '#ff7f0e',
        'BT20 앙상블': '#2ca02c',
        'KOSPI200': '#d62728'
    }

    # 선택된 기간 수에 따라 서브플롯 레이아웃 결정
    n_periods = len(selected_periods)
    if n_periods <= 3:
        nrows, ncols = 1, n_periods
        figsize = (6*n_periods, 6)
    elif n_periods <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 12)
    else:
        nrows, ncols = 2, 3  # 최대 6개로 제한
        selected_periods = selected_periods[:6]
        figsize = (18, 12)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_periods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'선택된 기간별 3개 전략 + KOSPI200 비교\n(월별 수익률 → cumprod(1+r) 누적 계산)',
                 fontsize=16, fontweight='bold', y=0.95)

    for idx, period in enumerate(selected_periods):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 각 전략별 라인 그래프
        for strategy_name, color in strategy_colors.items():
            if strategy_name == 'KOSPI200':
                col_name = 'kospi200'
                label = 'KOSPI200'
            elif strategy_name == 'BT20 단기':
                col_name = f'bt20_short_{period}'
                label = f'BT20 단기 ({period}일)'
            elif strategy_name == 'BT120 장기':
                col_name = f'bt120_long_{period}'
                label = f'BT120 장기 ({period}일)'
            else:  # BT20 앙상블
                col_name = f'bt20_ens_{period}'
                label = f'BT20 앙상블 ({period}일)'

            ax.plot(df['month'], df[col_name], label=label,
                   color=color, linewidth=2, marker='o', markersize=3)

        # 그래프 꾸미기
        ax.set_title(f'{period}일 보유 기간', fontsize=12, fontweight='bold')
        ax.set_ylabel('누적 수익률 (%)', fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        # X축 날짜 포맷팅
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # 0선 추가
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)

    # 남은 서브플롯 제거
    for idx in range(len(selected_periods), len(axes)):
        fig.delaxes(axes[idx])

    # 전체 레이아웃 조정
    plt.tight_layout()

    # 저장
    periods_str = '_'.join(selected_periods)
    output_path = f'strategies_comparison_selected_periods_{periods_str}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 선택된 기간 비교 그래프가 '{output_path}' 파일로 저장되었습니다.")

    # 선택된 기간들의 최종 성과 요약
    print("\n=== 선택된 기간별 최종 성과 요약 (2024년 12월) ===")
    final_row = df.iloc[-1]

    for period in selected_periods:
        print(f"\n📊 {period}일 보유 기간:")
        kospi_value = final_row['kospi200']

        for strategy_name in ['BT20 단기', 'BT120 장기', 'BT20 앙상블']:
            if strategy_name == 'BT20 단기':
                col_name = f'bt20_short_{period}'
            elif strategy_name == 'BT120 장기':
                col_name = f'bt120_long_{period}'
            else:
                col_name = f'bt20_ens_{period}'

            strategy_value = final_row[col_name]
            excess_return = strategy_value - kospi_value

            print(f"  {strategy_name}: {strategy_value:.1f}% (KOSPI200: {kospi_value:.1f}%, 초과: {excess_return:+.1f}%)")
def plot_all_periods_comparison():
    """모든 기간(20,40,60,80,100,120일)에 대해 3개 전략 + KOSPI200 비교"""

    selected_periods = ['20', '40', '60', '80', '100', '120']
    plot_strategies_by_selected_periods(selected_periods)

def plot_key_periods_comparison():
    """주요 기간(20,60,120일)만 비교"""

    selected_periods = ['20', '60', '120']
    plot_strategies_by_selected_periods(selected_periods)

def plot_short_term_comparison():
    """단기(20,40,60일) 비교"""

    selected_periods = ['20', '40', '60']
    plot_strategies_by_selected_periods(selected_periods)

def plot_long_term_comparison():
    """장기(80,100,120일) 비교"""

    selected_periods = ['80', '100', '120']
    plot_strategies_by_selected_periods(selected_periods)

if __name__ == "__main__":
    print("=== 기간별 전략 비교 그래프 생성 ===")
    print("1: 모든 기간(20,40,60,80,100,120일) 비교")
    print("2: 주요 기간(20,60,120일) 비교")
    print("3: 단기(20,40,60일) 비교")
    print("4: 장기(80,100,120일) 비교")

    choice = input("선택 (1-4, 기본값: 2): ").strip()

    if choice == '1':
        plot_all_periods_comparison()
    elif choice == '2':
        plot_key_periods_comparison()
    elif choice == '3':
        plot_short_term_comparison()
    elif choice == '4':
        plot_long_term_comparison()
    else:
        print("주요 기간(20,60,120일) 비교를 선택합니다.")
        plot_key_periods_comparison()