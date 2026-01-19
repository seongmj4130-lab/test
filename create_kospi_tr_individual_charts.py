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

def create_kospi_tr_monthly_returns_chart():
    """KOSPI TR 월별 로그 수익률 그래프 생성"""

    print("📊 KOSPI TR 월별 로그 수익률 그래프 생성 중...")

    # UI 데이터 로드
    df = pd.read_csv('data/ui_monthly_log_returns_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 8))

    # 바 차트 생성
    bars = ax.bar(df['year_month'], df['kospi_tr_monthly_log_return'],
                  color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=0.5, width=0.6)

    # 0선 추가
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)

    # 값 표시
    for bar, value in zip(bars, df['kospi_tr_monthly_log_return']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
                height + (0.3 if height >= 0 else -0.8),
                '.1f', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')

    # 그래프 설정
    ax.set_title('KOSPI TR 월별 로그 수익률 (2023-2024)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('월별 로그 수익률 (%)', fontsize=12)
    ax.set_xlabel('기간', fontsize=12)

    # X축 레이블 설정 (3개월마다 표시)
    xticks = df['year_month'][::3]
    ax.set_xticks(range(0, len(df), 3))
    ax.set_xticklabels(xticks, rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')

    # 범례 추가
    ax.legend(['0% 기준선', 'KOSPI TR 월별 수익률'], loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/kospi_tr_monthly_log_returns_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ KOSPI TR 월별 로그 수익률 그래프 생성: results/kospi_tr_monthly_log_returns_chart.png")

    return df

def create_kospi_tr_cumulative_returns_chart():
    """KOSPI TR 누적 로그 수익률 그래프 생성"""

    print("📊 KOSPI TR 누적 로그 수익률 그래프 생성 중...")

    # UI 데이터 로드
    df = pd.read_csv('data/ui_monthly_log_returns_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 8))

    # 선 그래프 생성
    ax.plot(df['year_month'], df['kospi_tr_cumulative_log_return'],
            color='#FF6B6B', linewidth=3, alpha=0.9, marker='o', markersize=4)

    # 0선 추가
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

    # 그래프 설정
    ax.set_title('KOSPI TR 누적 로그 수익률 (2023-2024)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('누적 로그 수익률 (%)', fontsize=12)
    ax.set_xlabel('기간', fontsize=12)

    # X축 레이블 설정 (3개월마다 표시)
    xticks = df['year_month'][::3]
    ax.set_xticks(range(0, len(df), 3))
    ax.set_xticklabels(xticks, rotation=45, ha='right')

    ax.grid(True, alpha=0.3)

    # 범례 추가
    ax.legend(['KOSPI TR 누적 수익률', '시작점', '종료점'], loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/kospi_tr_cumulative_log_returns_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ KOSPI TR 누적 로그 수익률 그래프 생성: results/kospi_tr_cumulative_log_returns_chart.png")

    return df

def create_combined_kospi_tr_analysis():
    """KOSPI TR 월별 vs 누적 분석 그래프 생성"""

    print("📊 KOSPI TR 월별 vs 누적 종합 분석 그래프 생성 중...")

    # UI 데이터 로드
    df = pd.read_csv('data/ui_monthly_log_returns_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 두 개의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 상단: 월별 수익률 바 차트
    bars = ax1.bar(df['year_month'], df['kospi_tr_monthly_log_return'],
                   color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=0.5, width=0.6)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    ax1.set_title('KOSPI TR 월별 로그 수익률', fontsize=14, fontweight='bold')
    ax1.set_ylabel('월별 수익률 (%)', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # 하단: 누적 수익률 선 그래프
    ax2.plot(df['year_month'], df['kospi_tr_cumulative_log_return'],
            color='#FF6B6B', linewidth=3, alpha=0.9, marker='o', markersize=3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    ax2.set_title('KOSPI TR 누적 로그 수익률', fontsize=14, fontweight='bold')
    ax2.set_ylabel('누적 수익률 (%)', fontsize=11)
    ax2.set_xlabel('기간', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # X축 레이블 설정 (3개월마다)
    xticks = df['year_month'][::3]
    ax2.set_xticks(range(0, len(df), 3))
    ax2.set_xticklabels(xticks, rotation=45, ha='right')

    # 전체 제목
    fig.suptitle('KOSPI TR 로그 수익률 분석: 월별 vs 누적 (2023-2024)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('results/kospi_tr_monthly_vs_cumulative_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ KOSPI TR 월별 vs 누적 종합 분석 그래프 생성: results/kospi_tr_monthly_vs_cumulative_analysis.png")

    return df

def analyze_kospi_tr_performance():
    """KOSPI TR 성과 분석"""

    df = pd.read_csv('data/ui_monthly_log_returns_data.csv')

    monthly_returns = df['kospi_tr_monthly_log_return']
    cumulative_returns = df['kospi_tr_cumulative_log_return']

    analysis = {
        '총 기간': f"{len(df)}개월 (2023-01 ~ 2024-12)",
        '총 수익률': ".2f",
        '연평균 수익률': ".2f",
        '월평균 수익률': ".2f",
        '양수 월 수': f"{(monthly_returns > 0).sum()}개월",
        '음수 월 수': f"{(monthly_returns < 0).sum()}개월",
        '최고 월 수익률': ".2f",
        '최저 월 수익률': ".2f",
        '변동성 (월간)': ".2f",
        '변동성 (연간)': ".2f"
    }

    print("\n📊 KOSPI TR 성과 분석")
    print("-" * 40)
    for key, value in analysis.items():
        print(f"{key}: {value}")

    return analysis

def print_final_summary():
    """최종 요약 출력"""

    print("\n" + "="*80)
    print("🎯 KOSPI TR 로그 수익률 그래프 생성 완료")
    print("="*80)

    print("\n📈 생성된 그래프 파일들:")
    print("   1. results/kospi_tr_monthly_log_returns_chart.png")
    print("      - KOSPI TR 월별 로그 수익률 바 차트")
    print("      - 각 월의 수익률을 시각적으로 표시")
    print("      - 0% 기준선으로 플러스/마이너스 구분")

    print("\n   2. results/kospi_tr_cumulative_log_returns_chart.png")
    print("      - KOSPI TR 누적 로그 수익률 선 그래프")
    print("      - 2년간의 누적 성과 추이")
    print("      - 시작점과 종료점 강조 표시")

    print("\n   3. results/kospi_tr_monthly_vs_cumulative_analysis.png")
    print("      - 월별 vs 누적 수익률 종합 분석")
    print("      - 상단: 월별 바 차트")
    print("      - 하단: 누적 선 그래프")

    print("\n📊 데이터 특징:")
    print("   • 기간: 2023년 1월 ~ 2024년 12월 (24개월)")
    print("   • 배당 포함: KOSPI TR (총수익지수)")
    print("   • 로그 수익률: 복리 효과 반영")
    print("   • % 단위: 직관적인 해석 가능")

    print("\n💡 그래프 해석 포인트:")
    print("   • 월별 그래프: 단기 변동성과 시장 사이클 파악")
    print("   • 누적 그래프: 장기 성과 추세 및 총 수익률 확인")
    print("   • 2023년 하락 vs 2024년 회복 패턴 분석 가능")

def main():
    """메인 실행 함수"""

    # 월별 수익률 그래프 생성
    monthly_data = create_kospi_tr_monthly_returns_chart()

    # 누적 수익률 그래프 생성
    cumulative_data = create_kospi_tr_cumulative_returns_chart()

    # 종합 분석 그래프 생성
    combined_data = create_combined_kospi_tr_analysis()

    # 성과 분석
    analysis = analyze_kospi_tr_performance()

    # 최종 요약
    print_final_summary()

if __name__ == "__main__":
    main()
