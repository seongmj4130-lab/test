# -*- coding: utf-8 -*-
"""
BT20_SHORT 모델 Hit Ratio 최적화 스크립트

목표: 과적합 없이 Holdout Hit Ratio 50% 이상 달성
현재: Dev 57.32%, Holdout 43.48% (과적합 13.84%)

최적화 전략:
1. 정규화 강화 (ridge_alpha 증가)
2. 피쳐 가중치 재조정 (과적합 유발 피쳐 축소)
3. 교차 검증 개선 (embargo_days 증가)
"""

import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_current_weights():
    """현재 가중치 분석"""
    base_dir = Path(__file__).parent.parent
    
    weights_path = base_dir / 'configs' / 'feature_weights_short_ic_optimized.yaml'
    
    if not weights_path.exists():
        return None
    
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights_data = yaml.safe_load(f) or {}
        feature_weights = weights_data.get("feature_weights", {})
        group_weights = weights_data.get("group_weights", {})
    
    return feature_weights, group_weights

def suggest_optimized_weights():
    """과적합 감소를 위한 가중치 제안"""
    
    # 현재 가중치
    current_weights = {
        'value': 0.15,
        'profitability': 0.10,
        'technical': 0.60,  # 과적합 유발 가능성 높음
        'other': 0.10,
        'news': 0.05
    }
    
    # 최적화 제안: technical 축소, value/profitability 강화
    # 이유: technical 피쳐가 단기에서 과적합 유발 가능성
    optimized_weights = {
        'value': 0.20,  # 0.15 → 0.20 (재무 지표 강화)
        'profitability': 0.15,  # 0.10 → 0.15 (수익성 지표 강화)
        'technical': 0.50,  # 0.60 → 0.50 (과적합 유발 피쳐 축소)
        'other': 0.10,
        'news': 0.05
    }
    
    return current_weights, optimized_weights

def create_optimized_weights_file(weights: dict, output_path: Path):
    """최적화된 가중치 파일 생성"""
    feature_groups = {
        'value': ['equity', 'total_liabilities', 'net_income', 'debt_ratio', 'debt_ratio_sector_z'],
        'profitability': ['roe', 'roe_sector_z'],
        'technical': [
            'volatility_60d', 'volatility_20d', 'volatility', 'downside_volatility_60d',
            'price_momentum_60d', 'price_momentum_20d', 'price_momentum', 'momentum_rank',
            'momentum_3m', 'momentum_6m', 'momentum_reversal',
            'max_drawdown_60d', 'volume', 'volume_ratio', 'turnover',
            'close', 'high', 'low', 'open', 'ret_daily'
        ],
        'other': ['in_universe'],
        'news': []
    }
    
    feature_weights = {}
    for group_name, features in feature_groups.items():
        group_weight = weights.get(group_name, 0.0)
        n_features = len([f for f in features if f])
        
        if n_features > 0 and group_weight > 0:
            weight_per_feature = group_weight / n_features
            for feature in features:
                if feature:
                    feature_weights[feature] = weight_per_feature
    
    yaml_data = {
        'description': f"Hit Ratio 최적화 - 과적합 감소 (합={sum(weights.values()):.2f})",
        'feature_weights': feature_weights,
        'group_weights': weights,
        'metadata': {
            'total_weight': sum(weights.values()),
            'n_groups': len(weights),
            'n_features': len(feature_weights),
            'optimization_target': 'hit_ratio_50pct_no_overfitting'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return yaml_data

def main():
    base_dir = Path(__file__).parent.parent
    
    print("="*80)
    print("BT20_SHORT Hit Ratio 최적화 방안")
    print("="*80)
    
    print("\n현재 상태:")
    print("  Dev Hit Ratio: 57.32%")
    print("  Holdout Hit Ratio: 43.48%")
    print("  과적합: 13.84% (Dev - Holdout)")
    print("  목표: Holdout Hit Ratio ≥ 50%, 과적합 ≤ 10%")
    
    # 현재 가중치 분석
    current_feature_weights, current_group_weights = analyze_current_weights()
    if current_group_weights:
        print("\n현재 그룹 가중치:")
        for group, weight in current_group_weights.items():
            print(f"  {group}: {weight:.2f}")
    
    # 최적화 제안
    current, optimized = suggest_optimized_weights()
    
    print("\n최적화 제안:")
    print("| 그룹 | 현재 | 제안 | 변화 | 이유 |")
    print("|------|------|------|------|------|")
    
    for group in current.keys():
        curr = current[group]
        opt = optimized[group]
        change = opt - curr
        direction = "↑" if change > 0 else "↓" if change < 0 else "="
        
        if group == 'technical':
            reason = "과적합 유발 가능성 높음 → 축소"
        elif group in ['value', 'profitability']:
            reason = "안정적 예측력 → 강화"
        else:
            reason = "유지"
        
        print(f"| {group:15s} | {curr:4.2f} | {opt:4.2f} | {change:+5.2f} {direction} | {reason} |")
    
    # 최적화된 가중치 파일 생성
    output_path = base_dir / 'configs' / 'feature_weights_short_hitratio_optimized.yaml'
    create_optimized_weights_file(optimized, output_path)
    
    print(f"\n✅ 최적화된 가중치 파일 생성: {output_path}")
    
    # 추가 최적화 방안
    print("\n" + "="*80)
    print("추가 최적화 방안")
    print("="*80)
    
    print("\n1. 정규화 강화 (ridge_alpha 조정)")
    print("   현재: ridge_alpha = 0.5")
    print("   제안: ridge_alpha = 0.8 ~ 1.0 (과적합 감소)")
    
    print("\n2. 교차 검증 개선")
    print("   현재: embargo_days = 20")
    print("   제안: embargo_days = 30 (lookahead bias 추가 방지)")
    
    print("\n3. 피쳐 선택")
    print("   - IC < 0.01인 피쳐 제거")
    print("   - Rank IC 기준 필터링 활성화")
    
    print("\n4. 모델 파라미터 튜닝")
    print("   - top_k 조정 (12 → 10 또는 15)")
    print("   - buffer_k 조정 (15 → 20)")
    
    print("\n" + "="*80)
    print("적용 방법")
    print("="*80)
    print("\n1. config.yaml 수정:")
    print("   l5:")
    print("     feature_weights_config_short: configs/feature_weights_short_hitratio_optimized.yaml")
    print("     ridge_alpha: 0.8  # 0.5 → 0.8")
    print("\n2. L5 재학습 및 백테스트 실행")
    print("3. Holdout Hit Ratio 확인 (목표: ≥ 50%)")

if __name__ == '__main__':
    main()

