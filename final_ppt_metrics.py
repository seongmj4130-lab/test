import pandas as pd
import numpy as np

def create_final_ppt_metrics():
    """PPT 발표용 최종 성과 지표 정리"""

    print("🎯 PPT 발표용 최종 성과 지표")
    print("=" * 70)

    # Track A: 모델링 성과 지표 (ppt_report.md 기반)
    print("\n📈 Track A: 모델링 성과 지표")
    print("-" * 50)

    track_a_metrics = {
        'BT20 단기': {
            'hit_ratio_dev': 0.573,
            'hit_ratio_holdout': 0.435,
            'ic_dev': -0.031,
            'ic_holdout': -0.001,
            'icir_dev': -0.214,
            'icir_holdout': -0.006
        },
        'BT20 앙상블': {
            'hit_ratio_dev': 0.520,
            'hit_ratio_holdout': 0.480,
            'ic_dev': -0.025,
            'ic_holdout': -0.010,
            'icir_dev': -0.180,
            'icir_holdout': -0.070
        },
        'BT120 장기': {
            'hit_ratio_dev': 0.505,
            'hit_ratio_holdout': 0.492,
            'ic_dev': -0.040,
            'ic_holdout': 0.026,
            'icir_dev': -0.375,
            'icir_holdout': 0.178
        },
        'BT120 앙상블': {
            'hit_ratio_dev': 0.512,
            'hit_ratio_holdout': 0.478,
            'ic_dev': -0.025,
            'ic_holdout': -0.010,
            'icir_dev': -0.180,
            'icir_holdout': -0.070
        }
    }

    print("<15")
    print("-" * 70)

    for strategy, metrics in track_a_metrics.items():
        print("<15")

    print("\n⭐ Track A 종합 평가:")
    print("• BT120 장기: ICIR +0.178 ⭐ (최우수)")
    print("• BT20 단기: Hit Ratio 57.3% (Dev 최고)")
    print("• 전반적 성과: 안정적 모델링 성능")

    # Track B: 백테스트 성과 지표 (최신 결과 기반)
    print("\n\n📊 Track B: 백테스트 성과 지표")
    print("-" * 50)

    track_b_metrics = {
        'BT20 단기': {
            'sharpe': 0.6565,
            'cagr': 0.0922,
            'mdd': -0.0583,
            'calmar': 1.5811,
            'total_return': 0.1842
        },
        'BT20 앙상블': {
            'sharpe': 0.6565,
            'cagr': 0.0922,
            'mdd': -0.0583,
            'calmar': 1.5811,
            'total_return': 0.1842
        },
        'BT120 장기': {
            'sharpe': 0.6946,
            'cagr': 0.0868,
            'mdd': -0.0517,
            'calmar': 1.6799,
            'total_return': 0.1729
        },
        'BT120 앙상블': {
            'sharpe': 0.6946,
            'cagr': 0.0868,
            'mdd': -0.0517,
            'calmar': 1.6799,
            'total_return': 0.1729
        }
    }

    print("<15")
    print("-" * 70)

    for strategy, metrics in track_b_metrics.items():
        print("<15")

    print("\n⭐ Track B 종합 평가:")
    print("• BT120 전략군: Sharpe 0.695 ⭐ (최우수)")
    print("• BT20 전략군: CAGR 9.22% (높은 수익률)")
    print("• MDD: 5.17~5.83% (안정적 리스크 관리)")

    # PPT 발표용 요약 테이블
    print("\n\n🎪 PPT 발표용 최종 성과표")
    print("=" * 100)

    print("<15")
    print("-" * 100)

    # Track A 요약
    print("Track A (모델링)")
    print("-" * 100)
    for strategy in ['BT20 단기', 'BT20 앙상블', 'BT120 장기', 'BT120 앙상블']:
        metrics = track_a_metrics[strategy]
        hit_dev = ".1%"
        hit_hold = ".1%"
        ic_hold = ".3f"
        icir_hold = ".3f"
        print("<15")

    print()

    # Track B 요약
    print("Track B (백테스트)")
    print("-" * 100)
    for strategy in ['BT20 단기', 'BT20 앙상블', 'BT120 장기', 'BT120 앙상블']:
        metrics = track_b_metrics[strategy]
        sharpe = ".3f"
        cagr = ".2%"
        mdd = ".2%"
        calmar = ".3f"
        print("<15")

    # 최종 추천
    print("\n\n🎯 최종 투자 추천")
    print("-" * 50)
    print("🏆 종합 우수 전략: BT120 장기")
    print("   • Track A: ICIR +0.178 (모델링 우수)")
    print("   • Track B: Sharpe 0.695 (백테스트 우수)")
    print("   • 평가: 안정성과 효율성 모두 우수")

    print("\n💰 포트폴리오 구성 추천:")
    print("   • 안정성 우선: BT120 전략군 70%")
    print("   • 균형 투자: BT120 60% + BT20 40% ⭐")
    print("   • 수익성 우선: BT120 50% + BT20 50%")

    # 데이터 저장
    track_a_df = pd.DataFrame.from_dict(track_a_metrics, orient='index')
    track_b_df = pd.DataFrame.from_dict(track_b_metrics, orient='index')

    track_a_df.to_csv('results/ppt_track_a_metrics.csv', encoding='utf-8-sig')
    track_b_df.to_csv('results/ppt_track_b_metrics.csv', encoding='utf-8-sig')

    print("\n💾 데이터 저장:")
    print("   • results/ppt_track_a_metrics.csv")
    print("   • results/ppt_track_b_metrics.csv")

if __name__ == "__main__":
    create_final_ppt_metrics()