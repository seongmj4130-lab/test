# -*- coding: utf-8 -*-
"""피쳐 가중치 적용 테스트"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from src.utils.config import load_config
from src.utils.io import load_artifact

# 설정 로드
cfg = load_config("configs/config.yaml")
l5 = cfg.get("l5", {})

print("="*80)
print("피쳐 가중치 적용 확인")
print("="*80)

# 가중치 파일 확인
base_dir = Path(cfg.get("paths", {}).get("base_dir", "."))

for horizon, config_key in [(20, "feature_weights_config_short"), (120, "feature_weights_config_long")]:
    print(f"\n[Horizon {horizon}일]")
    weights_config = l5.get(config_key)
    
    if not weights_config:
        print(f"  ❌ 설정 없음: {config_key}")
        continue
    
    weights_path = base_dir / weights_config
    print(f"  설정 경로: {weights_config}")
    print(f"  실제 경로: {weights_path}")
    print(f"  파일 존재: {weights_path.exists()}")
    
    if weights_path.exists():
        try:
            with open(weights_path, 'r', encoding='utf-8') as f:
                weights_data = yaml.safe_load(f) or {}
                feature_weights = weights_data.get("feature_weights", {})
                print(f"  ✅ 로드 성공: {len(feature_weights)}개 피쳐")
                
                # 상위 5개 피쳐 가중치 출력
                sorted_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
                print(f"  상위 5개 피쳐 가중치:")
                for feat, weight in sorted_weights[:5]:
                    print(f"    {feat}: {weight:.4f}")
        except Exception as e:
            print(f"  ❌ 로드 실패: {e}")

# 예측 결과 확인 (가중치 적용 여부)
print("\n" + "="*80)
print("예측 결과 확인")
print("="*80)

interim_dir = Path(cfg.get("paths", {}).get("data_interim", "data/interim"))
pred_short_file = interim_dir / "pred_short_oos.parquet"
pred_long_file = interim_dir / "pred_long_oos.parquet"

if pred_short_file.exists():
    pred_short = load_artifact(pred_short_file)
    print(f"\n[단기 예측 결과]")
    print(f"  행 수: {len(pred_short):,}")
    print(f"  날짜 범위: {pred_short['date'].min()} ~ {pred_short['date'].max()}")
    print(f"  Fold 수: {pred_short['fold_id'].nunique()}")
    
if pred_long_file.exists():
    pred_long = load_artifact(pred_long_file)
    print(f"\n[장기 예측 결과]")
    print(f"  행 수: {len(pred_long):,}")
    print(f"  날짜 범위: {pred_long['date'].min()} ~ {pred_long['date'].max()}")
    print(f"  Fold 수: {pred_long['fold_id'].nunique()}")

