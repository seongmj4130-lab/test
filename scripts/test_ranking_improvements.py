# -*- coding: utf-8 -*-
"""
Track A 랭킹산정 Hit Ratio 개선 테스트 스크립트

개선 방안:
1. 정규화 방법 변경 (percentile → zscore)
2. Sector-Relative 정규화 조정 (true → false)
3. 피처 가중치 극단화
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.ranking.score_engine import build_ranking_daily
from src.utils.config import load_config
from src.utils.io import artifact_exists, load_artifact


def test_normalization_methods():
    """정규화 방법별 Hit Ratio 비교"""
    print("="*80)
    print("정규화 방법별 Hit Ratio 비교 테스트")
    print("="*80)

    cfg = load_config("configs/config.yaml")
    interim_dir = PROJECT_ROOT / "data" / "interim"

    # 데이터 로드
    dataset_daily = load_artifact(interim_dir / "dataset_daily")

    # 정규화 방법별 테스트
    methods = ["percentile", "zscore"]
    results = {}

    for method in methods:
        print(f"\n[테스트] normalization_method = {method}")

        # 랭킹 생성
        ranking = build_ranking_daily(
            dataset_daily,
            feature_cols=None,
            feature_weights=None,  # 균등 가중치로 테스트
            feature_groups_config=None,
            normalization_method=method,
            date_col="date",
            universe_col="in_universe",
            sector_col="sector_name",
            use_sector_relative=True,
        )

        # Hit Ratio 계산 (간단 버전)
        # 실제 수익률과 비교
        if "ret_fwd_20d" in dataset_daily.columns:
            merged = ranking.merge(
                dataset_daily[["date", "ticker", "ret_fwd_20d"]],
                on=["date", "ticker"],
                how="inner"
            )

            # 방향 일치 계산
            merged["pred_direction"] = np.sign(merged["score_total"])
            merged["actual_direction"] = np.sign(merged["ret_fwd_20d"])
            merged["hit"] = (merged["pred_direction"] == merged["actual_direction"]).astype(int)

            hit_ratio = float(merged["hit"].mean())
            results[method] = hit_ratio
            print(f"  Hit Ratio: {hit_ratio:.2%}")

    print("\n" + "="*80)
    print("결과 비교")
    print("="*80)
    for method, hr in results.items():
        print(f"  {method}: {hr:.2%}")

    if len(results) == 2:
        diff = results["zscore"] - results["percentile"]
        print(f"\n  차이: {diff:+.2%}p")
        if diff > 0:
            print(f"  → zscore가 {diff:.2%}p 더 높음")
        else:
            print(f"  → percentile이 {abs(diff):.2%}p 더 높음")

def test_sector_relative():
    """Sector-Relative 정규화 영향 테스트"""
    print("\n" + "="*80)
    print("Sector-Relative 정규화 영향 테스트")
    print("="*80)

    cfg = load_config("configs/config.yaml")
    interim_dir = PROJECT_ROOT / "data" / "interim"

    dataset_daily = load_artifact(interim_dir / "dataset_daily")

    # Sector-Relative ON/OFF 비교
    settings = [
        {"use_sector_relative": True, "name": "Sector-Relative ON"},
        {"use_sector_relative": False, "name": "Sector-Relative OFF"},
    ]

    results = {}

    for setting in settings:
        print(f"\n[테스트] {setting['name']}")

        ranking = build_ranking_daily(
            dataset_daily,
            feature_cols=None,
            feature_weights=None,
            feature_groups_config=None,
            normalization_method="percentile",
            date_col="date",
            universe_col="in_universe",
            sector_col="sector_name",
            use_sector_relative=setting["use_sector_relative"],
        )

        if "ret_fwd_20d" in dataset_daily.columns:
            merged = ranking.merge(
                dataset_daily[["date", "ticker", "ret_fwd_20d"]],
                on=["date", "ticker"],
                how="inner"
            )

            merged["pred_direction"] = np.sign(merged["score_total"])
            merged["actual_direction"] = np.sign(merged["ret_fwd_20d"])
            merged["hit"] = (merged["pred_direction"] == merged["actual_direction"]).astype(int)

            hit_ratio = float(merged["hit"].mean())
            results[setting["name"]] = hit_ratio
            print(f"  Hit Ratio: {hit_ratio:.2%}")

    print("\n" + "="*80)
    print("결과 비교")
    print("="*80)
    for name, hr in results.items():
        print(f"  {name}: {hr:.2%}")

    if len(results) == 2:
        diff = results["Sector-Relative OFF"] - results["Sector-Relative ON"]
        print(f"\n  차이: {diff:+.2%}p")
        if diff > 0:
            print(f"  → Sector-Relative OFF가 {diff:.2%}p 더 높음")
        else:
            print(f"  → Sector-Relative ON이 {abs(diff):.2%}p 더 높음")

if __name__ == "__main__":
    print("Track A 랭킹산정 Hit Ratio 개선 테스트")
    print("="*80)

    # 테스트 1: 정규화 방법 비교
    try:
        test_normalization_methods()
    except Exception as e:
        print(f"\n[에러] 정규화 방법 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # 테스트 2: Sector-Relative 비교
    try:
        test_sector_relative()
    except Exception as e:
        print(f"\n[에러] Sector-Relative 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
