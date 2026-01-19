# -*- coding: utf-8 -*-
"""
피처 가중치 적용 방식 확인 스크립트

현재 피처셋이 그룹별로 가중치가 다르게 적용되는지 확인
"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config

def main():
    cfg = load_config("configs/config.yaml")
    
    print("="*80)
    print("피처 가중치 적용 방식 확인")
    print("="*80)
    
    # L8 설정 확인
    l8 = cfg.get("l8", {}) or {}
    l8_short = cfg.get("l8_short", {}) or {}
    l8_long = cfg.get("l8_long", {}) or {}
    
    print("\n[L8 공통 설정]")
    print(f"  feature_groups_config: {l8.get('feature_groups_config', 'N/A')}")
    print(f"  feature_weights_config: {l8.get('feature_weights_config', 'N/A')}")
    print(f"  regime_aware_weights_config: {l8.get('regime_aware_weights_config', 'N/A')}")
    
    print("\n[L8 단기 설정 (l8_short)]")
    print(f"  feature_groups_config: {l8_short.get('feature_groups_config', 'N/A')}")
    print(f"  feature_weights_config: {l8_short.get('feature_weights_config', 'N/A')}")
    
    print("\n[L8 장기 설정 (l8_long)]")
    print(f"  feature_groups_config: {l8_long.get('feature_groups_config', 'N/A')}")
    print(f"  feature_weights_config: {l8_long.get('feature_weights_config', 'N/A')}")
    
    # 가중치 파일 확인
    base_dir = Path(cfg.get("paths", {}).get("base_dir", PROJECT_ROOT))
    
    print("\n" + "="*80)
    print("가중치 파일 내용 확인")
    print("="*80)
    
    # 단기 가중치 파일
    short_weights_path = base_dir / l8_short.get("feature_weights_config", "")
    if short_weights_path.exists():
        print(f"\n[단기 가중치 파일] {short_weights_path}")
        with open(short_weights_path, 'r', encoding='utf-8') as f:
            short_weights = yaml.safe_load(f)
        
        if "feature_weights" in short_weights:
            print(f"  ✓ 피처별 가중치: {len(short_weights['feature_weights'])}개")
            # 그룹별로 분류
            if "group_weights" in short_weights:
                print(f"  ✓ 그룹별 가중치:")
                for group, weight in short_weights["group_weights"].items():
                    print(f"    - {group}: {weight:.3f}")
            
            # 피처별 가중치 샘플 (상위 10개)
            print(f"\n  피처별 가중치 샘플 (상위 10개):")
            sorted_weights = sorted(short_weights["feature_weights"].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
            for feat, weight in sorted_weights:
                print(f"    - {feat}: {weight:.4f}")
        else:
            print(f"  ⚠ feature_weights 키가 없습니다.")
    else:
        print(f"\n[단기 가중치 파일] 없음: {short_weights_path}")
    
    # 장기 가중치 파일
    long_weights_path = base_dir / l8_long.get("feature_weights_config", "")
    if long_weights_path.exists():
        print(f"\n[장기 가중치 파일] {long_weights_path}")
        with open(long_weights_path, 'r', encoding='utf-8') as f:
            long_weights = yaml.safe_load(f)
        
        if "feature_weights" in long_weights:
            print(f"  ✓ 피처별 가중치: {len(long_weights['feature_weights'])}개")
            if "group_weights" in long_weights:
                print(f"  ✓ 그룹별 가중치:")
                for group, weight in long_weights["group_weights"].items():
                    print(f"    - {group}: {weight:.3f}")
        else:
            print(f"  ⚠ feature_weights 키가 없습니다.")
    else:
        print(f"\n[장기 가중치 파일] 없음: {long_weights_path}")
    
    # 가중치 적용 우선순위 확인
    print("\n" + "="*80)
    print("가중치 적용 우선순위 (코드 기준)")
    print("="*80)
    print("""
    1순위: 국면별 가중치 (regime_aware_weights_config)
           - 시장 국면(Bull/Bear/Neutral)별로 다른 가중치 적용
           - 활성화 조건: regime_enabled=True AND regime_aware_weights_config 존재
    
    2순위: 피처별 가중치 (feature_weights_config의 feature_weights)
           - 각 피처마다 개별 가중치 적용
           - 현재 단기/장기 각각 다른 가중치 파일 사용
    
    3순위: 피처 그룹별 가중치 (feature_groups_config의 target_weight)
           - 피처를 그룹으로 묶어서 그룹별 가중치 적용
           - 그룹 내 피처는 균등 분배
    
    4순위: 균등 가중치
           - 모든 피처에 동일한 가중치 적용
    """)
    
    # 현재 적용 중인 가중치 방식 확인
    print("="*80)
    print("현재 적용 중인 가중치 방식")
    print("="*80)
    
    regime_enabled = cfg.get("l7", {}).get("regime", {}).get("enabled", False)
    
    if regime_enabled and l8.get("regime_aware_weights_config"):
        print("  → 1순위: 국면별 가중치 적용 중")
    elif l8_short.get("feature_weights_config") or l8_long.get("feature_weights_config"):
        print("  → 2순위: 피처별 가중치 적용 중")
        print(f"    - 단기: {l8_short.get('feature_weights_config', 'N/A')}")
        print(f"    - 장기: {l8_long.get('feature_weights_config', 'N/A')}")
    elif l8_short.get("feature_groups_config") or l8_long.get("feature_groups_config"):
        print("  → 3순위: 피처 그룹별 가중치 적용 중")
    else:
        print("  → 4순위: 균등 가중치 적용 중")

if __name__ == "__main__":
    main()

