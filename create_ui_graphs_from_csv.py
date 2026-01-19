import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_ui_data():
    """UI용 월별 로그 수익률 데이터 로드"""

    df = pd.read_csv('data/ui_monthly_log_returns_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    print("📊 UI 데이터 로드 완료")
    print(f"   • 데이터 기간: {len(df)}개월")
    print(f"   • 컬럼 수: {len(df.columns)}개")
    print(f"   • 시작: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"   • 종료: {df['date'].max().strftime('%Y-%m-%d')}")

    return df

def show_data_columns(df):
    """데이터 컬럼 설명"""

    print("\n📋 사용된 수익률 데이터 설명")
    print("=" * 60)

    columns_description = {
        'date': '날짜 (YYYY-MM-DD)',
        'year_month': '연월 (YYYY-MM)',
        'kospi_tr_monthly_log_return': 'KOSPI TR 월별 로그 수익률 (%) - 배당 포함 총수익지수',
        'kospi_tr_cumulative_log_return': 'KOSPI TR 누적 로그 수익률 (%) - 2년 누적',
        'bt20_단기_monthly_log_return': 'BT20 단기 월별 로그 수익률 (%) - 20일 리밸런싱',
        'bt20_단기_cumulative_log_return': 'BT20 단기 누적 로그 수익률 (%) - 롱숏 전략',
        'bt20_앙상블_monthly_log_return': 'BT20 앙상블 월별 로그 수익률 (%) - 20일 리밸런싱',
        'bt20_앙상블_cumulative_log_return': 'BT20 앙상블 누적 로그 수익률 (%) - 롱온리 전략',
        'bt120_장기_monthly_log_return': 'BT120 장기 월별 로그 수익률 (%) - 120일 리밸런싱',
        'bt120_장기_cumulative_log_return': 'BT120 장기 누적 로그 수익률 (%) - 롱온리 전략',
        'bt120_앙상블_monthly_log_return': 'BT120 앙상블 월별 로그 수익률 (%) - 120일 리밸런싱',
        'bt120_앙상블_cumulative_log_return': 'BT120 앙상블 누적 로그 수익률 (%) - 롱온리 전략'
    }

    for col, desc in columns_description.items():
        if col in df.columns:
            print(f"• {col}: {desc}")
        else:
            print(f"• {col}: 컬럼 없음")

def create_kospi_tr_graphs(df):
    """KOSPI TR 그래프 생성"""

    print("\n📊 KOSPI TR 그래프 생성 중...")

    # 월별 수익률 바 차트
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(df['year_month'], df['kospi_tr_monthly_log_return'],
                  color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=0.5, width=0.6)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)

    # 값 표시
    for bar, value in zip(bars, df['kospi_tr_monthly_log_return']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
                height + (0.3 if height >= 0 else -0.8),
                '.1f', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8, fontweight='bold')

    ax.set_title('KOSPI TR 월별 로그 수익률 (UI 데이터 기반)', fontsize=14, fontweight='bold')
    ax.set_ylabel('월별 수익률 (%)')
    ax.set_xticks(range(0, len(df), 3))
    ax.set_xticklabels(df['year_month'][::3], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/ui_kospi_tr_monthly_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 누적 수익률 선 그래프
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df['year_month'], df['kospi_tr_cumulative_log_return'],
            color='#FF6B6B', linewidth=3, alpha=0.9, marker='o', markersize=4)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)

    # 시작점과 끝점 표시
    start_value = df['kospi_tr_cumulative_log_return'].iloc[0]
    end_value = df['kospi_tr_cumulative_log_return'].iloc[-1]

    ax.scatter([df['year_month'].iloc[0]], [start_value], color='green', s=100, zorder=5)
    ax.scatter([df['year_month'].iloc[-1]], [end_value], color='red', s=100, zorder=5)

    ax.text(df['year_month'].iloc[0], start_value + 1, '.1f',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')
    ax.text(df['year_month'].iloc[-1], end_value + 1, '.1f',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

    ax.set_title('KOSPI TR 누적 로그 수익률 (UI 데이터 기반)', fontsize=14, fontweight='bold')
    ax.set_ylabel('누적 수익률 (%)')
    ax.set_xticks(range(0, len(df), 3))
    ax.set_xticklabels(df['year_month'][::3], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/ui_kospi_tr_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ KOSPI TR 그래프 생성 완료")

def create_strategy_comparison_graph(df):
    """전략별 누적 수익률 비교 그래프"""

    print("📊 전략별 누적 수익률 비교 그래프 생성 중...")

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    strategies = ['kospi_tr', 'bt20_단기', 'bt20_앙상블', 'bt120_장기', 'bt120_앙상블']

    for i, strategy in enumerate(strategies):
        col_name = f'{strategy}_cumulative_log_return'
        if col_name in df.columns:
            ax.plot(df['year_month'], df[col_name],
                    color=colors[i], linewidth=2.5, alpha=0.9, label=strategy.upper())

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    ax.set_title('전략별 누적 로그 수익률 비교 (UI 데이터 기반)', fontsize=14, fontweight='bold')
    ax.set_ylabel('누적 수익률 (%)')
    ax.set_xticks(range(0, len(df), 3))
    ax.set_xticklabels(df['year_month'][::3], rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/ui_strategies_cumulative_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ 전략별 누적 수익률 비교 그래프 생성 완료")

def create_monthly_returns_heatmap(df):
    """월별 수익률 히트맵 생성"""

    print("📊 월별 수익률 히트맵 생성 중...")

    # 월별 수익률 데이터만 추출
    monthly_cols = [col for col in df.columns if 'monthly_log_return' in col]
    heatmap_data = df[monthly_cols].T

    # 컬럼 이름 정리
    strategy_names = ['KOSPI TR', 'BT20 단기', 'BT20 앙상블', 'BT120 장기', 'BT120 앙상블']
    heatmap_data.index = strategy_names

    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

    # 컬러바 추가
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('수익률 (%)', rotation=-90, va="bottom")

    # 레이블 설정
    ax.set_xticks(np.arange(len(df)))
    ax.set_yticks(np.arange(len(strategy_names)))
    ax.set_xticklabels(df['year_month'], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(strategy_names)

    # 값 표시
    for i in range(len(strategy_names)):
        for j in range(len(df)):
            value = heatmap_data.iloc[i, j]
            color = 'white' if abs(value) > 5 else 'black'
            ax.text(j, i, '.1f', ha="center", va="center",
                   color=color, fontsize=7, fontweight='bold')

    ax.set_title('월별 수익률 히트맵 (UI 데이터 기반)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/ui_monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ 월별 수익률 히트맵 생성 완료")

def create_performance_summary_table(df):
    """성과 요약 테이블 생성"""

    print("📊 성과 요약 테이블 생성 중...")

    # 각 전략별 최종 성과 계산
    summary_data = []

    strategies = {
        'kospi_tr': 'KOSPI TR',
        'bt20_단기': 'BT20 단기',
        'bt20_앙상블': 'BT20 앙상블',
        'bt120_장기': 'BT120 장기',
        'bt120_앙상블': 'BT120 앙상블'
    }

    for strategy_key, strategy_name in strategies.items():
        monthly_col = f'{strategy_key}_monthly_log_return'
        cumulative_col = f'{strategy_key}_cumulative_log_return'

        if monthly_col in df.columns and cumulative_col in df.columns:
            monthly_returns = df[monthly_col]
            final_cumulative = df[cumulative_col].iloc[-1]

            # 기본 통계
            avg_monthly = monthly_returns.mean()
            volatility = monthly_returns.std()
            max_return = monthly_returns.max()
            min_return = monthly_returns.min()
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)

            summary_data.append({
                '전략': strategy_name,
                '최종_누적_수익률': final_cumulative,
                '평균_월별_수익률': avg_monthly,
                '변동성': volatility,
                '최고_월별_수익률': max_return,
                '최저_월별_수익률': min_return,
                '양수_개월_수': positive_months,
                '총_개월_수': total_months,
                '양수_비율': positive_months / total_months * 100
            })

    summary_df = pd.DataFrame(summary_data)

    # CSV 저장
    summary_df.to_csv('results/ui_performance_summary.csv', index=False, encoding='utf-8-sig')

    print("✅ 성과 요약 테이블 생성: results/ui_performance_summary.csv")

    # 콘솔에 표시
    print("\n📈 전략별 성과 요약")
    print("=" * 90)
    print("전략".ljust(12), "최종 누적".rjust(10), "평균 월별".rjust(10), "변동성".rjust(8), "양수 비율".rjust(8))
    print("-" * 90)

    for _, row in summary_df.iterrows():
        strategy = row['전략']
        final_cum = f"{row['최종_누적_수익률']:.1f}%"
        avg_monthly = f"{row['평균_월별_수익률']:.2f}%"
        vol = f"{row['변동성']:.2f}%"
        pos_ratio = f"{row['양수_비율']:.1f}%"

        print(f"{strategy:<12} {final_cum:>10} {avg_monthly:>10} {vol:>8} {pos_ratio:>8}")

    return summary_df

def create_ui_graphs_summary():
    """UI 그래프 생성 요약"""

    summary_text = """
# UI 그래프 생성 결과

## 📊 생성된 그래프 파일들

### 1. 개별 KOSPI TR 그래프
- **ui_kospi_tr_monthly_returns.png**: 월별 로그 수익률 바 차트
- **ui_kospi_tr_cumulative_returns.png**: 누적 로그 수익률 선 그래프

### 2. 전략 비교 그래프
- **ui_strategies_cumulative_comparison.png**: 5개 전략 누적 수익률 비교
- **ui_monthly_returns_heatmap.png**: 월별 수익률 히트맵

### 3. 데이터 및 분석
- **ui_performance_summary.csv**: 전략별 성과 요약 테이블

## 📋 사용된 수익률 데이터 설명

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| kospi_tr_monthly_log_return | KOSPI TR 월별 로그 수익률 (배당 포함) | % |
| kospi_tr_cumulative_log_return | KOSPI TR 누적 로그 수익률 | % |
| bt20_단기_monthly_log_return | BT20 단기 전략 월별 로그 수익률 | % |
| bt20_단기_cumulative_log_return | BT20 단기 전략 누적 로그 수익률 | % |
| bt20_앙상블_monthly_log_return | BT20 앙상블 전략 월별 로그 수익률 | % |
| bt20_앙상블_cumulative_log_return | BT20 앙상블 전략 누적 로그 수익률 | % |
| bt120_장기_monthly_log_return | BT120 장기 전략 월별 로그 수익률 | % |
| bt120_장기_cumulative_log_return | BT120 장기 전략 누적 로그 수익률 | % |
| bt120_앙상블_monthly_log_return | BT120 앙상블 전략 월별 로그 수익률 | % |
| bt120_앙상블_cumulative_log_return | BT120 앙상블 전략 누적 로그 수익률 | % |

## 🎯 그래프 해석 가이드

### KOSPI TR 그래프
- **월별 그래프**: 단기 시장 변동성 및 상승/하락 패턴 파악
- **누적 그래프**: 2년간 장기 성과 추세 및 총 수익률

### 전략 비교 그래프
- **누적 비교**: 각 전략의 장기 성과 및 KOSPI TR 대비 초과 수익
- **히트맵**: 월별 성과 패턴 및 전략별 강점/약점 분석

### 성과 지표
- **양수 비율**: 전략의 일관성 (높을수록 좋음)
- **변동성**: 리스크 수준 (낮을수록 안정적)
- **최종 누적**: 2년간 총 성과
"""

    with open('results/ui_graphs_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print("✅ UI 그래프 생성 요약: results/ui_graphs_summary.md")

def main():
    """메인 실행 함수"""

    print("🎨 UI용 그래프 생성 시작")
    print("=" * 50)

    # 데이터 로드
    df = load_ui_data()

    # 데이터 컬럼 설명
    show_data_columns(df)

    # KOSPI TR 그래프 생성
    create_kospi_tr_graphs(df)

    # 전략 비교 그래프 생성
    create_strategy_comparison_graph(df)

    # 월별 수익률 히트맵 생성
    create_monthly_returns_heatmap(df)

    # 성과 요약 테이블 생성
    create_performance_summary_table(df)

    # 요약 문서 생성
    create_ui_graphs_summary()

    print("\n" + "=" * 50)
    print("🎯 UI 그래프 생성 완료!")
    print("=" * 50)

    print("\n📁 생성된 파일들:")
    print("   • results/ui_kospi_tr_monthly_returns.png")
    print("   • results/ui_kospi_tr_cumulative_returns.png")
    print("   • results/ui_strategies_cumulative_comparison.png")
    print("   • results/ui_monthly_returns_heatmap.png")
    print("   • results/ui_performance_summary.csv")
    print("   • results/ui_graphs_summary.md")

if __name__ == "__main__":
    main()
