# -*- coding: utf-8 -*-
"""피쳐 가중치 적용 검증"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from src.stages.modeling.l5_train_models import train_oos_predictions
from src.utils.config import load_config
from src.utils.io import load_artifact

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("="*80)
print("피쳐 가중치 적용 검증")
print("="*80)

cfg = load_config("configs/config.yaml")
interim_dir = Path(cfg.get("paths", {}).get("data_interim", "data/interim"))
dataset = load_artifact(Path(interim_dir) / "dataset_daily.parquet")
cv_short = load_artifact(Path(interim_dir) / "cv_folds_short.parquet")

print("\n[단기 모델 (20일) 학습 테스트]")
print("첫 번째 fold만 학습하여 가중치 적용 여부 확인...")

# 첫 번째 fold만 사용
cv_short_first = cv_short[cv_short['fold_id'] == cv_short['fold_id'].iloc[0]].copy()

pred, met, rep, warns = train_oos_predictions(
    dataset_daily=dataset,
    cv_folds=cv_short_first,
    cfg=cfg,
    target_col='ret_fwd_20d',
    horizon=20,
)

# 가중치 관련 경고 확인
weight_warns = [w for w in warns if '피쳐 가중치' in w or 'feature_weights' in w.lower()]

print(f"\n가중치 관련 메시지: {len(weight_warns)}개")
for w in weight_warns:
    print(f"  {w}")

if len(weight_warns) == 0:
    print("\n⚠️  경고: 가중치 관련 메시지가 없습니다.")
    print("  - 가중치 파일이 로드되지 않았을 수 있습니다.")
    print("  - config.yaml 설정을 확인하세요.")
else:
    print("\n✅ 가중치가 적용되었습니다!")
