# -*- coding: utf-8 -*-
"""
Hit Ratio 기준 모델 평가 및 최적화 방향 분석

목표: 과적합 없이 Hit Ratio 50% 이상 달성
"""

from pathlib import Path

import numpy as np
import pandas as pd


def analyze_hit_ratio_by_model():
    """모델별 Hit Ratio 분석"""
    base_dir = Path(__file__).parent.parent
    reports_dir = base_dir / 'artifacts' / 'reports'

    models = ['bt20_ens', 'bt20_short', 'bt120_ens', 'bt120_long']

    results = []

    for model in models:
        file_path = reports_dir / f'bt_metrics_{model}_optimized.csv'

        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)

        for phase in ['dev', 'holdout']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) == 0:
                continue

            row = phase_df.iloc[0]

            results.append({
                'model': model,
                'phase': phase,
                'net_hit_ratio': row.get('net_hit_ratio', 0),
                'gross_hit_ratio': row.get('gross_hit_ratio', 0),
                'net_sharpe': row.get('net_sharpe', 0),
                'net_cagr': row.get('net_cagr', 0),
                'net_mdd': row.get('net_mdd', 0),
            })

    return pd.DataFrame(results)

def generate_optimization_recommendations(results_df: pd.DataFrame):
    """Hit Ratio 최적화 권장사항 생성"""
    recommendations = []

    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]

        dev = model_df[model_df['phase'] == 'dev']
        holdout = model_df[model_df['phase'] == 'holdout']

        if len(dev) == 0 or len(holdout) == 0:
            continue

        dev_hr = dev['net_hit_ratio'].values[0]
        holdout_hr = holdout['net_hit_ratio'].values[0]

        # 과적합 체크 (Dev - Holdout 차이)
        overfitting = dev_hr - holdout_hr

        # 목표 달성 여부
        target_met = holdout_hr >= 0.50

        # 상태 평가
        if overfitting > 0.10:
            status = "⚠️ 과적합"
            priority = "높음"
        elif holdout_hr < 0.50:
            status = "❌ 목표 미달"
            priority = "높음"
        elif holdout_hr >= 0.50 and overfitting <= 0.10:
            status = "✅ 양호"
            priority = "낮음"
        else:
            status = "⚠️ 개선 필요"
            priority = "중간"

        recommendations.append({
            'model': model,
            'dev_hit_ratio': dev_hr,
            'holdout_hit_ratio': holdout_hr,
            'overfitting': overfitting,
            'target_met': target_met,
            'status': status,
            'priority': priority,
        })

    return pd.DataFrame(recommendations)

def main():
    print("="*80)
    print("Hit Ratio 기준 모델 평가 및 최적화 방향")
    print("="*80)

    # Hit Ratio 분석
    results_df = analyze_hit_ratio_by_model()

    if len(results_df) == 0:
        print("❌ 백테스트 결과 파일을 찾을 수 없습니다.")
        return

    print("\n## 1. 모델별 Hit Ratio 현황\n")
    print("| 모델 | 구간 | Net Hit Ratio | Gross Hit Ratio | Sharpe | CAGR | MDD |")
    print("|------|------|---------------|-----------------|--------|------|-----|")

    for _, row in results_df.iterrows():
        model = row['model']
        phase = row['phase']
        net_hr = row['net_hit_ratio']
        gross_hr = row['gross_hit_ratio']
        sharpe = row['net_sharpe']
        cagr = row['net_cagr']
        mdd = row['net_mdd']

        # 목표 달성 표시
        target_icon = "✅" if net_hr >= 0.50 else "❌"

        print(f"| {model:12s} | {phase:7s} | {net_hr:13.4f} {target_icon} | {gross_hr:15.4f} | {sharpe:6.4f} | {cagr:4.4f} | {mdd:5.4f} |")

    # 최적화 권장사항
    print("\n## 2. 과적합 및 목표 달성 분석\n")
    recommendations = generate_optimization_recommendations(results_df)

    print("| 모델 | Dev HR | Holdout HR | 과적합 | 목표달성 | 상태 | 우선순위 |")
    print("|------|--------|------------|--------|----------|------|----------|")

    for _, row in recommendations.iterrows():
        print(f"| {row['model']:12s} | {row['dev_hit_ratio']:6.4f} | {row['holdout_hit_ratio']:10.4f} | {row['overfitting']:6.4f} | {'✅' if row['target_met'] else '❌'} | {row['status']:10s} | {row['priority']:6s} |")

    # 최적화 방향 제시
    print("\n## 3. Hit Ratio 50% 이상 최적화 방향\n")

    for _, row in recommendations.iterrows():
        model = row['model']
        dev_hr = row['dev_hit_ratio']
        holdout_hr = row['holdout_hit_ratio']
        overfitting = row['overfitting']
        target_met = row['target_met']

        print(f"\n### {model.upper()}")

        if target_met and overfitting <= 0.10:
            print(f"✅ 목표 달성: Holdout Hit Ratio = {holdout_hr:.4f} (50% 이상)")
            print("   - 현재 상태 유지 권장")
        elif target_met and overfitting > 0.10:
            print(f"⚠️ 목표 달성했으나 과적합 존재: Holdout = {holdout_hr:.4f}, 과적합 = {overfitting:.4f}")
            print("   권장사항:")
            print("   1. 정규화 강화 (ridge_alpha 증가)")
            print("   2. 피쳐 선택 (IC 낮은 피쳐 제거)")
            print("   3. 교차 검증 개선 (embargo_days 증가)")
        elif not target_met and overfitting <= 0.10:
            print(f"❌ 목표 미달: Holdout Hit Ratio = {holdout_hr:.4f} (< 50%)")
            print("   권장사항:")
            print("   1. 피쳐 가중치 조정 (IC 높은 피쳐 강화)")
            print("   2. 모델 파라미터 튜닝 (ridge_alpha 조정)")
            print("   3. 피쳐 추가/개선 (모멘텀, 품질 지표)")
            gap = 0.50 - holdout_hr
            print(f"   4. 목표: {holdout_hr:.4f} → 0.50 (차이: {gap:.4f})")
        else:
            print(f"❌ 목표 미달 + 과적합: Holdout = {holdout_hr:.4f}, 과적합 = {overfitting:.4f}")
            print("   권장사항:")
            print("   1. 정규화 강화 (ridge_alpha 증가)")
            print("   2. 피쳐 선택 (IC 낮은 피쳐 제거)")
            print("   3. 교차 검증 개선 (embargo_days 증가)")
            print("   4. 피쳐 가중치 재조정")

    # 저장
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'artifacts' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'hit_ratio_analysis.csv', index=False, encoding='utf-8-sig')
    recommendations.to_csv(output_dir / 'hit_ratio_optimization_recommendations.csv', index=False, encoding='utf-8-sig')

    print(f"\n✅ 분석 결과 저장:")
    print(f"   - {output_dir / 'hit_ratio_analysis.csv'}")
    print(f"   - {output_dir / 'hit_ratio_optimization_recommendations.csv'}")

if __name__ == '__main__':
    main()
