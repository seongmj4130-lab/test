#!/usr/bin/env python3
"""
절대 수익률 중심 평가 전환 - 수익률을 메인 KPI로 사용하는 평가 시스템
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class AbsoluteReturnEvaluator:
    """절대 수익률 중심 평가 시스템"""

    def __init__(self):
        self.benchmark_data = self._load_benchmark_data()

    def _load_benchmark_data(self):
        """벤치마크 데이터 로드"""
        try:
            with open('configs/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('benchmark_data', {})
        except Exception as e:
            print(f"벤치마크 데이터 로드 실패: {e}")
            return {}

    def evaluate_strategies_absolute_return(self, results_df):
        """절대 수익률 중심 전략 평가"""

        print("🎯 절대 수익률 중심 평가 시스템")
        print("="*60)

        # 평가 가중치 설정 (수익률 중심)
        weights = {
            'cagr': 0.40,        # 절대 수익률 (가장 중요)
            'total_return': 0.25, # 총 수익률
            'sharpe': 0.15,      # 리스크 조정 수익률 (감소)
            'mdd': 0.10,         # 안정성 (감소)
            'calmar': 0.10       # Calmar 비율 (유지)
        }

        print("📊 평가 가중치 (수익률 중심):")
        for metric, weight in weights.items():
            print(f"  • {metric}: {weight:.0%}")
        print("\n🎯 전략별 절대 수익률 평가")
        print("-" * 60)

        evaluations = {}

        for strategy in ['bt20_short', 'bt20_ens', 'bt120_long']:
            strategy_data = results_df[results_df['strategy'] == strategy]

            if strategy_data.empty:
                continue

            # 최고 성과 케이스 선택 (수익률 기준)
            best_by_cagr = strategy_data.loc[strategy_data['cagr'].idxmax()]
            best_by_total_return = strategy_data.loc[strategy_data['total_return'].idxmax()]

            # 최종 평가 케이스 선택 (CAGR 우선)
            best_case = best_by_cagr

            # 벤치마크 대비 평가
            kospi_return = self.benchmark_data.get('kospi200', {}).get('annual_return_pct', 4.5)
            quant_avg_return = self.benchmark_data.get('quant_funds', {}).get('avg_annual_return', 6.5)

            evaluation = self._calculate_absolute_score(best_case, weights, kospi_return, quant_avg_return)

            evaluations[strategy] = {
                'best_case': best_case,
                'evaluation': evaluation
            }

            print(f"\n{strategy.upper()} (최적: {best_case['holding_days']}일)")
            print(f"  • CAGR: {best_case['cagr']:.2f}% (벤치마크: {kospi_return:.1f}%)")
            print(f"  • 총수익률: {best_case['total_return']:.2f}%")
            print(f"  • Sharpe: {best_case['sharpe']:.2f}")
            print(f"  • MDD: {best_case['mdd']:.1f}%")
            print(f"  • 종합점수: {evaluation['total_score']:.1f}점")
            print(f"  • 등급: {evaluation['grade']}")
            print(f"  • KOSPI 초과: {evaluation['excess_vs_kospi']:+.2f}%")
        # 전략 순위 결정
        self._rank_strategies_absolute_return(evaluations)

        return evaluations

    def _calculate_absolute_score(self, strategy_data, weights, kospi_return, quant_avg_return):
        """절대 수익률 기반 종합 점수 계산"""

        # 정규화된 지표 계산 (0-100 점수로 변환)
        cagr_score = self._normalize_cagr(strategy_data['cagr'])
        total_return_score = self._normalize_total_return(strategy_data['total_return'])
        sharpe_score = self._normalize_sharpe(strategy_data['sharpe'])
        mdd_score = self._normalize_mdd(strategy_data['mdd'])
        calmar_score = self._normalize_calmar(strategy_data['calmar'])

        # 가중 평균 점수
        total_score = (
            cagr_score * weights['cagr'] +
            total_return_score * weights['total_return'] +
            sharpe_score * weights['sharpe'] +
            mdd_score * weights['mdd'] +
            calmar_score * weights['calmar']
        )

        # 벤치마크 대비 성과
        excess_return_vs_kospi = strategy_data['cagr'] - kospi_return
        excess_return_vs_quant = strategy_data['cagr'] - quant_avg_return

        # 투자 등급 결정
        if strategy_data['cagr'] >= quant_avg_return:
            grade = "A"  # 퀀트 평균 이상
        elif strategy_data['cagr'] >= kospi_return:
            grade = "B"  # KOSPI 이상
        elif strategy_data['cagr'] >= kospi_return * 0.5:
            grade = "C"  # KOSPI 50% 이상
        else:
            grade = "D"  # 부진

        return {
            'total_score': total_score,
            'cagr_score': cagr_score,
            'excess_vs_kospi': excess_return_vs_kospi,
            'excess_vs_quant': excess_return_vs_quant,
            'grade': grade,
            'normalized_scores': {
                'cagr': cagr_score,
                'total_return': total_return_score,
                'sharpe': sharpe_score,
                'mdd': mdd_score,
                'calmar': calmar_score
            }
        }

    def _normalize_cagr(self, cagr):
        """CAGR 정규화 (0-100점)"""
        if cagr >= 12.0:  # 퀀트 상위권
            return 100
        elif cagr >= 6.5:  # 퀀트 평균
            return 75 + (cagr - 6.5) / (12.0 - 6.5) * 25
        elif cagr >= 4.5:  # KOSPI 수준
            return 50 + (cagr - 4.5) / (6.5 - 4.5) * 25
        elif cagr >= 0:
            return 25 + (cagr / 4.5) * 25
        else:
            return max(0, 25 + (cagr / 4.5) * 25)

    def _normalize_total_return(self, total_return):
        """총 수익률 정규화"""
        # 2년 기준으로 연환산
        if total_return >= 15.0:  # 연 7% 이상
            return 100
        elif total_return >= 9.2:  # KOSPI 수준
            return 75
        elif total_return >= 0:
            return 50 + (total_return / 9.2) * 25
        else:
            return max(0, 25 + (total_return / 9.2) * 25)

    def _normalize_sharpe(self, sharpe):
        """Sharpe 비율 정규화 (감소된 가중치)"""
        if sharpe >= 0.8:
            return 80  # 최대 80점 (수익률 중심)
        elif sharpe >= 0.4:
            return 40 + (sharpe - 0.4) / 0.4 * 40
        else:
            return max(0, (sharpe / 0.4) * 40)

    def _normalize_mdd(self, mdd):
        """MDD 정규화 (감소된 가중치)"""
        mdd_abs = abs(mdd)
        if mdd_abs <= 5.0:  # 매우 안정적
            return 60  # 최대 60점
        elif mdd_abs <= 12.0:  # KOSPI 수준
            return 40 + (12.0 - mdd_abs) / 7.0 * 20
        else:
            return max(0, 40 - (mdd_abs - 12.0) / 13.0 * 40)

    def _normalize_calmar(self, calmar):
        """Calmar 비율 정규화"""
        if calmar >= 2.0:
            return 70  # 최대 70점
        elif calmar >= 1.0:
            return 35 + (calmar - 1.0) / 1.0 * 35
        else:
            return max(0, (calmar / 1.0) * 35)

    def _rank_strategies_absolute_return(self, evaluations):
        """절대 수익률 기반 전략 순위 결정"""

        print("\n🏆 절대 수익률 기반 전략 순위")
        print("-" * 60)

        # 점수 기준 정렬
        ranked_strategies = sorted(
            evaluations.items(),
            key=lambda x: x[1]['evaluation']['total_score'],
            reverse=True
        )

        for rank, (strategy, data) in enumerate(ranked_strategies, 1):
            eval_data = data['evaluation']
            grade = eval_data['grade']

            grade_desc = {
                'A': '탁월 (퀀트 평균 이상)',
                'B': '우수 (KOSPI 이상)',
                'C': '보통 (KOSPI 50% 이상)',
                'D': '부진 (개선 필요)'
            }

            print(f"{rank}위: {strategy.upper()}")
            print(f"   종합점수: {eval_data['total_score']:.1f}점")
            print(f"   등급: {grade} - {grade_desc[grade]}")
            print(f"   KOSPI 초과: {eval_data['excess_vs_kospi']:+.2f}%")
            print(f"   퀀트 초과: {eval_data['excess_vs_quant']:+.2f}%")
    def create_absolute_return_report(self, evaluations):
        """절대 수익률 중심 평가 보고서 생성"""

        print("\n📋 절대 수익률 중심 평가 보고서")
        print("="*60)

        # 최고 전략 선정
        best_strategy = max(
            evaluations.items(),
            key=lambda x: x[1]['evaluation']['total_score']
        )[0]

        print("🎯 평가 결과 요약:")
        print("  • 메인 KPI: 절대 수익률 (CAGR)")
        print("  • 보조 KPI: 총 수익률, Sharpe, MDD, Calmar")
        print(f"  • 최고 전략: {best_strategy.upper()}")
        print("  • 평가 방식: 수익률 중심 가중치 적용")
        print("\n💡 투자 의사결정 가이드:")
        print("  • A등급: 적극 투자 추천")
        print("  • B등급: 보수적 투자 고려")
        print("  • C등급: 모니터링 후 결정")
        print("  • D등급: 전략 개선 필요")
        # 전략별 상세 권장사항
        print("\n🔧 전략별 권장사항:")
        for strategy, data in evaluations.items():
            grade = data['evaluation']['grade']
            cagr = data['best_case']['cagr']

            if grade == 'A':
                recommendation = "적극 투자 추천 - 안정적 수익 창출 가능"
            elif grade == 'B':
                recommendation = "보수적 투자 고려 - KOSPI 초과 가능성"
            elif grade == 'C':
                recommendation = "모니터링 후 결정 - 개선 여지 확인 필요"
            else:
                recommendation = "전략 개선 필요 - 현재 수익률 부진"

            print(f"  • {strategy.upper()}: {recommendation}")

        return best_strategy

def main():
    """메인 실행"""
    # 최신 백테스트 결과 로드
    results_path = "results/final_18_cases_backtest_report_20260114_030411.csv"
    if not Path(results_path).exists():
        print("❌ 백테스트 결과 파일을 찾을 수 없습니다.")
        return

    results_df = pd.read_csv(results_path)

    # 절대 수익률 중심 평가 실행
    evaluator = AbsoluteReturnEvaluator()
    evaluations = evaluator.evaluate_strategies_absolute_return(results_df)
    best_strategy = evaluator.create_absolute_return_report(evaluations)

    print(f"\n✅ 절대 수익률 중심 평가 완료! 최고 전략: {best_strategy.upper()}")

if __name__ == "__main__":
    main()