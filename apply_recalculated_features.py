#!/usr/bin/env python3
"""
재계산된 피처 영향도를 원본 CSV 파일에 적용
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# 프로젝트 경로 설정
project_root = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/000_code")
sys.path.append(str(project_root))

from src.components.ranking.contribution_engine import (
    ContributionConfig,
    compute_group_contributions_for_day,
    load_group_map_from_yaml,
    pick_top_groups_per_row,
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


def recalculate_for_single_date(date: str, horizon: str) -> pd.DataFrame:
    """단일 날짜에 대한 피처 영향도 재계산"""
    cfg = load_config()

    # 데이터 로드
    dataset_path = project_root / "data" / "interim" / "dataset_daily.parquet"
    ranking_path = project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20.csv"

    dataset_daily = pd.read_parquet(dataset_path)
    ranking_df = pd.read_csv(ranking_path)

    # 날짜 필터링
    date_ts = pd.Timestamp(date)
    day_df = dataset_daily[dataset_daily['date'] == date_ts].copy()
    day_ranking = ranking_df[ranking_df['date'] == date].copy()

    if len(day_df) == 0:
        raise ValueError(f"dataset_daily에 date={date} 행이 없습니다.")
    if len(day_ranking) == 0:
        raise ValueError(f"ranking 데이터에 date={date} 행이 없습니다.")

    # 데이터 타입 통일
    day_ranking['date'] = pd.to_datetime(day_ranking['date'])
    day_ranking['ticker'] = day_ranking['ticker'].astype(str).str.zfill(6)
    day_df['ticker'] = day_df['ticker'].astype(str).str.zfill(6)

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

    # 필요한 컬럼들 추출 및 재구성
    result_cols = ['ranking', 'ticker', 'date', 'score_short', 'score_long', 'score_ens']

    # Top 3 그룹 정보 추가
    for i in range(1, 4):
        group_col = f'group_top{i}_name'
        value_col = f'group_top{i}_value'

        if group_col in merged.columns and value_col in merged.columns:
            merged[f'top{i}_feature_group'] = merged[group_col]
        else:
            merged[f'top{i}_feature_group'] = 'unknown'

    # 피처 정보는 contribution_engine에서 제공되지 않으므로 빈 값으로 설정
    for i in range(1, 4):
        merged[f'top{i}_features'] = ''

    result_cols.extend(['top1_feature_group', 'top2_feature_group', 'top3_feature_group',
                       'top1_features', 'top2_features', 'top3_features', 'original_ranking'])

    return merged[result_cols]


def update_csv_file(horizon: str, max_dates: Optional[int] = None):
    """CSV 파일을 재계산된 데이터로 업데이트"""

    print(f"=== {horizon.upper()} CSV 파일 업데이트 시작 ===")

    # 원본 파일 로드 (없으면 새로 생성)
    csv_path = project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20_formatted.csv"

    # ranking 데이터 로드해서 기본 구조 생성
    ranking_path = project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20.csv"
    ranking_df = pd.read_csv(ranking_path)

    # 티커 포맷팅 및 종목명 추가
    ticker_to_name = {
        '005930': '삼성전자', '000660': 'SK하이닉스', '035420': 'NAVER', '034730': 'SK텔레콤',
        '005380': '현대차', '000270': '기아', '035720': '카카오', '005490': 'POSCO홀딩스',
        '051910': 'LG화학', '012330': '현대모비스', '055550': '신한지주', '032830': '삼성생명',
        '003550': 'LG', '006400': '삼성SDI', '086790': '하나금융지주', '138040': '메리츠금융지주',
        '036570': '엔씨소프트', '000810': '삼성화재', '009150': '삼성전기', '034730': 'SK',
        '352820': '하이브', '011200': 'HMM', '010130': '고려아연', '009830': '한화솔루션',
        '241560': '두산밥캣', '137310': '에스디바이오센서', '003240': '태광산업'
    }

    # 기본 데이터프레임 생성
    original_df = ranking_df.copy()
    original_df['ticker_formatted'] = original_df['ticker'].astype(str).str.zfill(6)
    original_df['company_name'] = original_df['ticker_formatted'].map(ticker_to_name).fillna('Unknown')

    # 모든 날짜 가져오기
    all_dates = sorted(original_df['date'].unique())
    if max_dates:
        all_dates = all_dates[:max_dates]

    print(f"처리할 날짜 수: {len(all_dates)}")

    updated_rows = []

    for date in all_dates:
        try:
            print(f"처리 중: {date}")
            recalculated = recalculate_for_single_date(date, horizon)

            # 해당 날짜의 원본 데이터를 재계산된 데이터로 교체
            date_mask = original_df['date'] == date
            original_for_date = original_df[date_mask].copy()

            # 재계산된 데이터로 업데이트
            for _, recalc_row in recalculated.iterrows():
                ticker = recalc_row['ticker']
                ticker_mask = original_for_date['ticker_formatted'] == ticker

                if ticker_mask.any():
                    # 기존 행 업데이트
                    for col in ['top1_feature_group', 'top2_feature_group', 'top3_feature_group',
                              'top1_features', 'top2_features', 'top3_features']:
                        if col in recalc_row.index:
                            original_for_date.loc[ticker_mask, col] = recalc_row[col]

            updated_rows.append(original_for_date)

        except Exception as e:
            print(f"오류 발생 ({date}): {e}")
            # 오류 발생 시 기본 값 설정
            date_data = original_df[original_df['date'] == date].copy()
            for col in ['top1_feature_group', 'top2_feature_group', 'top3_feature_group',
                       'top1_features', 'top2_features', 'top3_features']:
                if col not in date_data.columns:
                    date_data[col] = ''
            updated_rows.append(date_data)
            continue

    # 업데이트된 데이터 결합
    if updated_rows:
        final_df = pd.concat(updated_rows, ignore_index=True)

        # 정렬 (원본과 동일하게)
        final_df = final_df.sort_values(['date', 'ranking']).reset_index(drop=True)

        # 컬럼 순서 조정
        cols = ['ranking', 'ticker_formatted', 'company_name', 'date', 'score_short', 'score_long', 'score_ens',
                'top1_feature_group', 'top2_feature_group', 'top3_feature_group',
                'top1_features', 'top2_features', 'top3_features', 'original_ranking']
        final_df = final_df[cols]

        # 파일 저장
        output_path = csv_path
        final_df.to_csv(output_path, index=False)
        print(f"=== {horizon.upper()} CSV 업데이트 완료: {output_path} ===")

        return final_df
    else:
        raise ValueError("업데이트된 데이터가 없습니다.")


def validate_updated_file(horizon: str):
    """업데이트된 파일 유효성 검증"""
    print(f"=== {horizon.upper()} 파일 유효성 검증 ===")

    csv_path = project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20_formatted.csv"
    df = pd.read_csv(csv_path)

    # 샘플 날짜로 검증
    sample_dates = df['date'].unique()[:3]

    for date in sample_dates:
        day_data = df[df['date'] == date]
        unique_patterns = day_data['top1_feature_group'].astype(str) + ',' + \
                         day_data['top2_feature_group'].astype(str) + ',' + \
                         day_data['top3_feature_group'].astype(str)

        unique_count = unique_patterns.nunique()
        print(f"{date}: {unique_count}개 고유 패턴")

        # 샘플 출력
        print(f"  샘플 (상위 3개):")
        for _, row in day_data.head(3).iterrows():
            ticker = row['company_name']
            pattern = f"{row['top1_feature_group']},{row['top2_feature_group']},{row['top3_feature_group']}"
            print(f"    {ticker}: {pattern}")


if __name__ == "__main__":
    print("CSV 파일 피처 영향도 업데이트 시작")

    # 샘플로 일부 날짜만 처리 (테스트용)
    max_dates = 5  # None으로 설정하면 전체 처리

    try:
        # 단기(short) 업데이트
        updated_short = update_csv_file('short', max_dates)
        validate_updated_file('short')

        # 장기(long) 업데이트
        updated_long = update_csv_file('long', max_dates)
        validate_updated_file('long')

        print("\n=== CSV 파일 업데이트 성공! ===")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
