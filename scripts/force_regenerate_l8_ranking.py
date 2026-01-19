# -*- coding: utf-8 -*-
"""
L8 단기/장기 랭킹 강제 재생성 스크립트
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.pipeline.track_a_pipeline import run_track_a_pipeline

def main():
    print("="*80)
    print("L8 단기/장기 랭킹 강제 재생성")
    print("="*80)
    
    # 캐시 파일 삭제
    interim_dir = PROJECT_ROOT / "data" / "interim"
    ranking_short_path = interim_dir / "ranking_short_daily"
    ranking_long_path = interim_dir / "ranking_long_daily"
    
    if ranking_short_path.exists():
        import shutil
        shutil.rmtree(ranking_short_path)
        print(f"✓ 기존 단기 랭킹 캐시 삭제: {ranking_short_path}")
    
    if ranking_long_path.exists():
        import shutil
        shutil.rmtree(ranking_long_path)
        print(f"✓ 기존 장기 랭킹 캐시 삭제: {ranking_long_path}")
    
    # L8 랭킹 재생성
    print("\n[L8 랭킹 재생성 시작]")
    result = run_track_a_pipeline(
        config_path="configs/config.yaml",
        force_rebuild=True,  # 강제 재생성
    )
    
    print("\n[재생성 완료]")
    print(f"  단기 랭킹: {len(result.get('ranking_short_daily', [])):,}행")
    print(f"  장기 랭킹: {len(result.get('ranking_long_daily', [])):,}행")

if __name__ == "__main__":
    main()

