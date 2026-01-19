#!/usr/bin/env python3
"""
종목별 실제 피처 영향도를 다시 계산하는 스크립트
단기(short)와 장기(long) 각각 적용
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# 프로젝트 경로 설정
project_root = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/000_code")
sys.path.append(str(project_root))

from src.components.ranking.contribution_engine import (
    ContributionConfig,
    compute_group_contributions_for_day,
    load_group_map_from_yaml,
    pick_top_groups_per_row
)


def load_config() -> dict:
    """설정 파일 로드"""
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_feature_weights_and_groups(horizon: str, cfg: dict) -> tuple[Dict[str, float], Optional[Dict[str, List[str]]]]:
    """피처 가중치와 그룹 매핑 로드"""
    base_dir = Path(cfg.get("paths", {}).get("base_dir", "."))
    l8_key = f"l8_{horizon}"
    l8 = cfg.get(l8_key, {}) or {}

    # 피처 가중치 로드
    weights_rel = l8.get("feature_weights_config")
    if not weights_rel:
        raise ValueError(f"{l8_key}.feature_weights_config is missing in config.yaml")

    weights_path = base_dir / str(weights_rel)
    if not weights_path.exists():
        raise FileNotFoundError(f"feature_weights_config not found: {weights_path}")

    with open(weights_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    feature_weights = data.get("feature_weights", {}) or {}
    feature_weights = {str(k): float(v) for k, v in feature_weights.items()}

    # 그룹 매핑 로드
    groups_rel = l8.get("feature_groups_config")
    groups_path = (base_dir / str(groups_rel)) if groups_rel else None
    if groups_path is not None and not groups_path.exists():
        groups_path = None

    group_map = load_group_map_from_yaml(groups_path) if groups_path else None

    return feature_weights, group_map


def recalculate_feature_impact_for_date(date: str, horizon: str, cfg: dict) -> pd.DataFrame:
    """특정 날짜의 피처 영향도 재계산"""

    # 데이터 로드
    dataset_path = project_root / "data" / "interim" / "dataset_daily.parquet"
    ranking_path = project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20.csv"

    dataset_daily = pd.read_parquet(dataset_path)
    ranking_df = pd.read_csv(ranking_path)

    # 날짜 필터링
    date_ts = pd.Timestamp(date)
    day_df = dataset_daily[dataset_daily['date'] == date_ts].copy()
    day_ranking = ranking_df[ranking_df['date'] == date].copy()

    # 데이터 타입 통일
    day_ranking['date'] = pd.to_datetime(day_ranking['date'])
    day_ranking['ticker'] = day_ranking['ticker'].astype(str).str.zfill(6)
    day_df['ticker'] = day_df['ticker'].astype(str).str.zfill(6)

    if len(day_df) == 0:
        raise ValueError(f"dataset_daily에 date={date} 행이 없습니다.")
    if len(day_ranking) == 0:
        raise ValueError(f"ranking 데이터에 date={date} 행이 없습니다.")

    print(f"날짜 {date}: {len(day_df)}개 종목 데이터, {len(day_ranking)}개 랭킹 데이터")

    # 피처 가중치와 그룹 매핑 로드
    feature_weights, group_map = load_feature_weights_and_groups(horizon, cfg)

    # ContributionConfig 설정
    l8_key = f"l8_{horizon}"
    l8 = cfg.get(l8_key, {}) or {}
    contrib_cfg = ContributionConfig(
        normalization_method=l8.get("normalization_method", "percentile"),
        date_col="date",
        ticker_col="ticker",
        universe_col="in_universe",
        sector_col=l8.get("sector_col"),
        use_sector_relative=bool(l8.get("use_sector_relative", False)),
    )

    # 그룹 기여도 계산
    contrib = compute_group_contributions_for_day(
        day_df,
        feature_weights=feature_weights,
        group_map=group_map,
        cfg=contrib_cfg,
    )

    # Top 3 그룹 선정
    contrib = pick_top_groups_per_row(contrib, top_n=3)

    # 랭킹 데이터와 병합
    merged = day_ranking.merge(contrib, on=["date", "ticker"], how="left")

    return merged


def format_top_features(row: pd.Series) -> str:
    """Top 3 피처 포맷팅"""
    features = []
    for i in range(1, 4):
        name_col = f"group_top{i}_name"
        val_col = f"group_top{i}_value"

        if name_col in row.index and val_col in row.index:
            name = row[name_col]
            val = row[val_col]
            if pd.notna(name) and pd.notna(val):
                features.append(f"{name}({val:+.4f})")

    return ", ".join(features)


def recalculate_all_dates(horizon: str, sample_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """모든 날짜 또는 샘플 날짜들의 피처 영향도 재계산"""

    print(f"=== {horizon.upper()} 피처 영향도 재계산 시작 ===")

    cfg = load_config()
    ranking_path = project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20.csv"

    # 모든 날짜 가져오기
    ranking_df = pd.read_csv(ranking_path)
    all_dates = sorted(ranking_df['date'].unique())

    if sample_dates:
        dates_to_process = sample_dates
    else:
        dates_to_process = all_dates[:5]  # 샘플로 처음 5개 날짜만 처리

    print(f"처리할 날짜 수: {len(dates_to_process)}")

    all_results = []

    for date in dates_to_process:
        try:
            print(f"처리 중: {date}")
            result = recalculate_feature_impact_for_date(date, horizon, cfg)
            result['top_features_formatted'] = result.apply(format_top_features, axis=1)
            all_results.append(result)
        except Exception as e:
            print(f"오류 발생 ({date}): {e}")
            continue

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"=== {horizon.upper()} 재계산 완료: {len(combined)}개 행 ===")
        return combined
    else:
        raise ValueError("재계산된 데이터가 없습니다.")


def validate_results(results: pd.DataFrame, horizon: str):
    """결과 유효성 검증"""
    print("=== 유효성 검증 ===")

    # 각 날짜별로 다른 top3 그룹을 가지고 있는지 확인
    date_groups = results.groupby('date')['top_features_formatted'].nunique()

    print(f"날짜별 고유한 피처 그룹 패턴 수:")
    for date, count in date_groups.items():
        print(f"  {date}: {count}개 패턴")

    # 샘플로 몇 개 날짜의 결과를 출력
    sample_dates = results['date'].unique()[:3]
    for date in sample_dates:
        day_data = results[results['date'] == date]
        print(f"\n{date} 샘플 (상위 5개):")
        for _, row in day_data.head(5).iterrows():
            ticker = row['ticker']
            top_features = row['top_features_formatted']
            print(f"  {ticker}: {top_features}")


if __name__ == "__main__":
    # 샘플 날짜들로 테스트
    sample_dates = ['2023-01-02', '2023-01-03', '2023-01-04']

    print("종목별 실제 피처 영향도 재계산 시작")

    # 단기(short) 재계산
    try:
        short_results = recalculate_all_dates('short', sample_dates)
        validate_results(short_results, 'short')

        # 장기(long) 재계산
        long_results = recalculate_all_dates('long', sample_dates)
        validate_results(long_results, 'long')

        print("\n=== 재계산 성공! ===")
        print("단기 결과 샘플:")
        print(short_results.head())
        print("\n장기 결과 샘플:")
        print(long_results.head())

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()