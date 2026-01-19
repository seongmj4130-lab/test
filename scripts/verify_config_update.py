# -*- coding: utf-8 -*-
"""config.yaml 업데이트 확인"""
import yaml

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

print("=" * 80)
print("config.yaml 최적 가중치 설정 확인")
print("=" * 80)

print("\n[단기 랭킹 (l8_short)]")
print(f"  feature_groups_config: {cfg['l8_short']['feature_groups_config']}")

print("\n[장기 랭킹 (l8_long)]")
print(f"  feature_groups_config: {cfg['l8_long']['feature_groups_config']}")

print("\n" + "=" * 80)
print("✅ config.yaml 업데이트 완료")
print("=" * 80)
