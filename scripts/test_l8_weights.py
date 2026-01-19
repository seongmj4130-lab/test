# -*- coding: utf-8 -*-
"""
L8 랭킹 가중치 적용 테스트
"""
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
    run_L8_long_rank_engine,
    run_L8_short_rank_engine,
)
from src.utils.config import load_config
from src.utils.io import artifact_exists, load_artifact


def main():
    cfg = load_config("configs/config.yaml")
    interim_dir = PROJECT_ROOT / "data" / "interim"

    # 필요한 artifacts 로드
    artifacts = {
        "dataset_daily": load_artifact(interim_dir / "dataset_daily"),
        "panel_merged_daily": load_artifact(interim_dir / "panel_merged_daily"),
    }

    print("="*80)
    print("L8 단기 랭킹 생성 (피처 가중치 적용 확인)")
    print("="*80)

    # L8_short 실행
    outputs_short, warns_short = run_L8_short_rank_engine(
        cfg=cfg,
        artifacts=artifacts,
        force=True,  # 강제 재생성
    )

    print("\n[L8_short 경고 메시지]")
    for w in warns_short:
        if "가중치" in w or "weight" in w.lower():
            print(f"  {w}")

    print("\n" + "="*80)
    print("L8 장기 랭킹 생성 (피처 가중치 적용 확인)")
    print("="*80)

    # L8_long 실행
    outputs_long, warns_long = run_L8_long_rank_engine(
        cfg=cfg,
        artifacts=artifacts,
        force=True,  # 강제 재생성
    )

    print("\n[L8_long 경고 메시지]")
    for w in warns_long:
        if "가중치" in w or "weight" in w.lower():
            print(f"  {w}")

    # 가중치 파일 확인
    print("\n" + "="*80)
    print("가중치 파일 확인")
    print("="*80)

    l8_short = cfg.get("l8_short", {})
    l8_long = cfg.get("l8_long", {})

    base_dir = Path(cfg.get("paths", {}).get("base_dir", PROJECT_ROOT))

    short_weights_path = base_dir / l8_short.get("feature_weights_config", "")
    long_weights_path = base_dir / l8_long.get("feature_weights_config", "")

    print(f"\n[단기 가중치 파일]")
    print(f"  설정 경로: {l8_short.get('feature_weights_config', 'N/A')}")
    print(f"  실제 경로: {short_weights_path}")
    print(f"  존재 여부: {short_weights_path.exists()}")

    if short_weights_path.exists():
        with open(short_weights_path, 'r', encoding='utf-8') as f:
            short_weights = yaml.safe_load(f)
        print(f"  피처 수: {len(short_weights.get('feature_weights', {}))}")

    print(f"\n[장기 가중치 파일]")
    print(f"  설정 경로: {l8_long.get('feature_weights_config', 'N/A')}")
    print(f"  실제 경로: {long_weights_path}")
    print(f"  존재 여부: {long_weights_path.exists()}")

    if long_weights_path.exists():
        with open(long_weights_path, 'r', encoding='utf-8') as f:
            long_weights = yaml.safe_load(f)
        print(f"  피처 수: {len(long_weights.get('feature_weights', {}))}")

if __name__ == "__main__":
    main()

