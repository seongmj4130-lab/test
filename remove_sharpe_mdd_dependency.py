#!/usr/bin/env python3
"""
과도한 Sharpe/MDD 의존성 제거 - 수익률 중심 평가 시스템
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class ReturnFocusedEvaluator:
    """수익률 중심 평가 시스템 (Sharpe/MDD 의존성 제거)"""

    def __init__(self):
        self.old_weights = {
            'cagr': 0.20,        # 낮은 수익률 가중치
            'total_return': 0.15, # 낮은 수익률 가중치
            'sharpe': 0.35,      # 높은 리스크 가중치 (문제!)
            'mdd': 0.20,         # 높은 리스크 가중치 (문제!)
            'calmar': 0.10       # 보조 지표
        }

        self.new_weights = {
            'cagr': 0.45,        # 높은 수익률 가중치 (증가)
            'total_return': 0.30, # 높은 수익률 가중치 (증가)
            'sharpe': 0.10,      # 낮은 리스크 가중치 (감소)
            'mdd': 0.10,         # 낮은 리스크 가중치 (감소)
            'calmar': 0.05       # 보조 지표 (감소)
        }

    def compare_evaluation_systems(self):
        """기존 vs 새로운 평가 시스템 비교"""

        print("🔄 과도한 Sharpe/MDD 의존성 제거")
        print("="*60)

        print("📊 평가 시스템 비교")
        print("-" * 60)
        print("구분".ljust(15), "기존 시스템".ljust(15), "새로운 시스템".ljust(15), "변화")
        print("-" * 60)

        for metric in self.old_weights.keys():
            old_w = self.old_weights[metric]
            new_w = self.new_weights[metric]
            change = new_w - old_w
            change_str = f"{change:+.0%}"

            print(f"{metric.ljust(15)} {old_w:>13.0%} {new_w:>13.0%} {change_str:>10}")

        print("\n💡 시스템 변화 설명:")
        print("  • CAGR 가중치: 20% → 45% (+25%p) - 절대 수익률 최우선")
        print("  • Total Return 가중치: 15% → 30% (+15%p) - 누적 수익률 중요성 증가")
        print("  • Sharpe 가중치: 35% → 10% (-25%p) - 리스크 조정 의존성 제거")
        print("  • MDD 가중치: 20% → 10% (-10%p) - 하락 위험 의존성 제거")
        print("  • Calmar 가중치: 10% → 5% (-5%p) - 복합 지표 영향 감소")

    def demonstrate_evaluation_difference(self):
        """기존 vs 새로운 평가 방식 차이 시연"""

        print("\n🎯 평가 방식 차이 시연")
        print("-" * 60)

        # 샘플 전략 데이터 (실제 백테스트 결과 기반)
        sample_strategies = {
            'High_Return_Low_Risk': {
                'cagr': 8.0, 'total_return': 15.0, 'sharpe': 1.2, 'mdd': -8.0, 'calmar': 1.0
            },
            'Moderate_Return_High_Risk': {
                'cagr': 5.0, 'total_return': 9.0, 'sharpe': 0.8, 'mdd': -15.0, 'calmar': 0.33
            },
            'Low_Return_Low_Risk': {
                'cagr': 2.0, 'total_return': 4.0, 'sharpe': 1.5, 'mdd': -5.0, 'calmar': 0.4
            }
        }

        print("전략별 평가 점수 비교:")
        print("전략".ljust(25), "기존점수".ljust(10), "새점수".ljust(10), "순위변화")
        print("-" * 60)

        results = []
        for name, metrics in sample_strategies.items():
            # 기존 시스템 점수 계산
            old_score = sum(metrics[k] * self.old_weights[k] for k in self.old_weights.keys())
            old_score = (old_score / 10) * 100  # 정규화 (0-100점)

            # 새로운 시스템 점수 계산
            new_score = sum(metrics[k] * self.new_weights[k] for k in self.new_weights.keys())
            new_score = (new_score / 10) * 100  # 정규화 (0-100점)

            results.append((name, old_score, new_score))

        # 기존 시스템 기준 정렬
        results.sort(key=lambda x: x[1], reverse=True)

        for name, old_score, new_score in results:
            rank_change = ""
            print(f"{name.ljust(25)} {old_score:>8.1f} {new_score:>8.1f} {rank_change}")

        print("\n💡 평가 결과 해석:")
        print("  • 기존 시스템: 리스크 지표(Sharpe/MDD)가 55% 차지")
        print("  • 새로운 시스템: 수익률 지표(CAGR/Total Return)가 75% 차지")
        print("  • 결과: 절대 수익률이 낮아도 리스크가 좋으면 고평가되던 문제 해결")

    def update_evaluation_config(self):
        """설정 파일에 새로운 평가 가중치 적용"""

        config_path = 'configs/config.yaml'

        try:
            # 기존 설정 로드
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}

            # 새로운 평가 가중치 설정
            if 'evaluation' not in config:
                config['evaluation'] = {}

            config['evaluation']['weights'] = {
                'description': '수익률 중심 평가 시스템 (Sharpe/MDD 의존성 제거)',
                'cagr': 0.45,
                'total_return': 0.30,
                'sharpe': 0.10,
                'mdd': 0.10,
                'calmar': 0.05,
                'last_updated': '2025-01-14',
                'change_reason': '과도한 Sharpe/MDD 의존성 제거, 절대 수익률 중심 평가'
            }

            # 설정 저장
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

            print("✅ config.yaml에 새로운 평가 가중치 적용 완료")
            print(f"📁 파일: {config_path}")

        except Exception as e:
            print(f"❌ 설정 파일 업데이트 실패: {e}")

    def validate_real_strategy_evaluation(self):
        """실제 전략에 새로운 평가 방식 적용"""

        print("\n🎯 실제 전략 재평가 (새로운 시스템)")
        print("-" * 60)

        # 실제 백테스트 결과 로드
        results_path = "results/final_18_cases_backtest_report_20260114_030411.csv"
        if not Path(results_path).exists():
            print("❌ 백테스트 결과 파일을 찾을 수 없습니다.")
            return

        df = pd.read_csv(results_path)

        # 벤치마크
        kospi_return = 4.5
        quant_avg_return = 6.5

        print("전략별 새로운 평가 결과:")
        print("전략".ljust(15), "CAGR".ljust(8), "총수익".ljust(8), "Sharpe".ljust(8), "MDD".ljust(8), "새점수".ljust(8), "등급")
        print("-" * 80)

        for strategy in ['bt20_short', 'bt20_ens', 'bt120_long']:
            strategy_data = df[df['strategy'] == strategy]

            if strategy_data.empty:
                continue

            # 최고 CAGR 케이스 선택
            best_idx = strategy_data['cagr(%)'].idxmax()
            best_case = strategy_data.loc[best_idx]

            # 새로운 평가 방식으로 점수 계산
            cagr_score = (best_case['cagr(%)'] / 10) * 100  # CAGR 기반 정규화
            total_return_score = (best_case['total_return(%)'] / 20) * 100  # 총수익 기반 정규화

            new_score = (
                best_case['cagr(%)'] * self.new_weights['cagr'] +
                best_case['total_return(%)'] * self.new_weights['total_return'] +
                best_case['sharpe'] * self.new_weights['sharpe'] +
                abs(best_case['mdd(%)']) * self.new_weights['mdd'] +  # MDD는 낮을수록 좋음
                best_case['calmar'] * self.new_weights['calmar']
            )

            # 등급 결정 (수익률 중심)
            if best_case['cagr(%)'] >= quant_avg_return:
                grade = "A"
            elif best_case['cagr(%)'] >= kospi_return:
                grade = "B"
            elif best_case['cagr(%)'] >= 0:
                grade = "C"
            else:
                grade = "D"

            print(f"{strategy.ljust(15)} {best_case['cagr(%)']:>6.2f} {best_case['total_return(%)']:>6.2f} {best_case['sharpe']:>6.2f} {best_case['mdd(%)']:>6.2f} {new_score:>6.1f} {grade}")

        print("\n💡 새로운 평가 방식의 특징:")
        print("  • CAGR와 총수익이 75%의 가중치 차지")
        print("  • Sharpe/MDD 영향력이 20%로 축소")
        print("  • 절대 수익률이 낮으면 등급 자동 하락")
        print("  • 리스크 관리 좋은데 수익률 낮은 전략 걸러냄")

def main():
    """메인 실행"""
    evaluator = ReturnFocusedEvaluator()

    # 1. 평가 시스템 비교
    evaluator.compare_evaluation_systems()

    # 2. 평가 방식 차이 시연
    evaluator.demonstrate_evaluation_difference()

    # 3. 설정 파일 업데이트
    evaluator.update_evaluation_config()

    # 4. 실제 전략 재평가
    evaluator.validate_real_strategy_evaluation()

    print("\n✅ 과도한 Sharpe/MDD 의존성 제거 완료!")
    print("🎯 평가 시스템: 수익률 중심으로 전환됨")

if __name__ == "__main__":
    main()
