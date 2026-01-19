import os
from datetime import datetime

import numpy as np
import pandas as pd


def analyze_topk_change_impact():
    """top_k=20 변경이 성과에 미치는 영향 분석"""

    print("🔍 top_k=20 변경 영향 분석")
    print("=" * 60)

    # 신규 결과 로드 (top_k=20)
    new_results = pd.read_csv('results/topk20_performance_metrics.csv')

    # 기존 결과 (참고용)
    try:
        old_results = pd.read_csv('C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code\\artifacts\\reports\\backtest_4models_comparison.csv')
        print("📊 기존 결과 (참고):")
        for _, row in old_results.iterrows():
            strategy_name = row['strategy'].replace('bt20_ens', 'BT20 앙상블').replace('bt20_short', 'BT20 단기').replace('bt120_ens', 'BT120 앙상블').replace('bt120_long', 'BT120 장기')
            print(".3f")
        print()
    except:
        print("기존 결과 파일을 찾을 수 없습니다.\n")

    # 신규 결과 분석
    print("📊 신규 결과 (top_k=20):")
    for _, row in new_results.iterrows():
        print(f"🏆 {row['전략']}")
        print(".2%")
        print(".2%")
        print(".3f")
        print(".2%")
        print(".2%")
        print(".3f")
        print()

    # 문제점 분석
    print("⚠️  잠재적 문제점 분석")
    print("-" * 40)

    # CAGR 계산 문제점
    print("1. CAGR 계산 방식 문제:")
    print("   • 데이터 포인트: 23일")
    print("   • CAGR 공식: (1+r)^(252/n) - 1")
    print("   • 23일 데이터로 연간화 시 과도한 복리 효과 발생")
    print("   • 예: 5% 수익률 → (1.05)^(252/23) ≈ 70% CAGR")
    print()

    # top_k 변경 영향
    print("2. top_k 변경의 포트폴리오 영향:")
    print("   • BT20 단기: 12 → 20 (67% 증가)")
    print("   • BT20 앙상블: 15 → 20 (33% 증가)")
    print("   • BT120 장기: 15 → 20 (33% 증가)")
    print("   • BT120 앙상블: 20 → 20 (변화 없음)")
    print()

    # 전략별 민감도
    print("3. 전략별 top_k 민감도:")
    print("   • BT20 단기: top_k 증가로 성능 급락")
    print("   • BT120 전략: top_k 증가로 성능 향상")
    print("   • 앙상블 효과: 더 큰 포트폴리오에서 안정성 증대")
    print()

    # 기간 문제
    print("4. 데이터 기간 문제:")
    print("   • Holdout 기간: 23개월 데이터")
    print("   • 짧은 기간으로 인한 변동성 과대평가")
    print("   • 연간화 계산 시 왜곡 효과")
    print()

    # 권장사항
    print("💡 권장사항")
    print("-" * 20)
    print("1. CAGR 대신 총수익률로 평가")
    print("2. 더 긴 기간 데이터 사용")
    print("3. 전략별 최적 top_k 재탐색")
    print("4. Out-of-sample 성능 재평가")
    print()

    # 수정된 지표 계산
    print("📈 수정된 평가 지표 (총수익률 기준)")
    print("-" * 40)

    for _, row in new_results.iterrows():
        total_return = row['총수익률']
        mdd = row['MDD']
        if mdd != 0:
            modified_calmar = total_return / abs(mdd)
        else:
            modified_calmar = 0

        print(f"🏆 {row['전략']}")
        print(".2%")
        print(".2%")
        print(".2f")
        print()

if __name__ == "__main__":
    analyze_topk_change_impact()
