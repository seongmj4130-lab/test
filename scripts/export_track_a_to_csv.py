# -*- coding: utf-8 -*-
"""
Track A 산출물을 CSV로 저장
- 날짜 범위: 2023-01-01 ~ 2024-12-31
- 컬럼: 날짜, 종목명(티커), 스코어, top3 영향 팩터셋(절댓값)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.components.ranking.score_engine import (
    normalize_feature_cross_sectional,
    _pick_feature_cols,
)
from src.utils.feature_groups import get_feature_groups, load_feature_groups


# 팩터 그룹 한글명 매핑
FACTOR_GROUP_NAMES = {
    "technical": "기술적분석",
    "value": "가치",
    "profitability": "수익성",
    "news": "뉴스",
    "other": "기타",
    "esg": "ESG",
}


def load_feature_weights(weights_config_path: Path) -> Dict[str, float]:
    """피처 가중치 파일 로드"""
    if not weights_config_path.exists():
        print(f"경고: 가중치 파일을 찾을 수 없습니다: {weights_config_path}")
        return {}
    
    with open(weights_config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    return data.get("feature_weights", {})


def calculate_feature_contributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    feature_weights: Dict[str, float],
    normalization_method: str = "percentile",
    sector_col: str = None,
    use_sector_relative: bool = True,
) -> pd.DataFrame:
    """
    각 종목/날짜별로 팩터 기여도 계산
    
    Returns:
        원본 df에 각 팩터의 기여도 컬럼이 추가된 DataFrame
    """
    out = df.copy()
    
    # sector-relative 정규화 사용 여부 결정
    actual_sector_col = None
    if use_sector_relative and sector_col and sector_col in out.columns:
        if out[sector_col].notna().sum() > 0:
            actual_sector_col = sector_col
    
    # 각 팩터의 정규화된 값 계산
    normalized_features = {}
    for feat in feature_cols:
        if feat not in out.columns:
            continue
        
        normalized = normalize_feature_cross_sectional(
            out,
            feat,
            "date",
            method=normalization_method,
            sector_col=actual_sector_col,
        )
        normalized_features[feat] = normalized
    
    # 각 팩터의 기여도 계산 (정규화된 값 × 가중치)
    contribution_cols = {}
    for feat in normalized_features.keys():
        weight = feature_weights.get(feat, 0.0)
        contribution = normalized_features[feat] * weight
        contribution_cols[feat] = contribution
        out[f"contrib_{feat}"] = contribution
    
    return out


def get_feature_to_group_mapping(
    feature_groups_config: Path,
) -> Dict[str, str]:
    """
    피처명을 그룹명으로 매핑하는 딕셔너리 생성
    
    Returns:
        {피처명: 그룹명} 딕셔너리
    """
    if not feature_groups_config.exists():
        return {}
    
    cfg_groups = load_feature_groups(feature_groups_config)
    feature_groups = get_feature_groups(cfg_groups)
    
    mapping = {}
    for group_name, features in feature_groups.items():
        for feat in features:
            mapping[str(feat)] = group_name
    
    return mapping


def get_top3_factor_groups(
    row: pd.Series,
    feature_cols: List[str],
    feature_to_group: Dict[str, str],
    prefix: str = "contrib_",
) -> Tuple[str, str, str]:
    """
    한 행에서 절댓값 기준 top3 팩터 그룹 추출 (한글명)
    
    Returns:
        (top1, top2, top3) 튜플 (팩터 그룹 한글명)
    """
    contributions = {}
    for feat in feature_cols:
        col = f"{prefix}{feat}"
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                # 그룹명으로 변환
                group_name = feature_to_group.get(feat, "other")
                # 그룹별 기여도 합산 (같은 그룹에 속한 여러 피처의 기여도 합산)
                if group_name not in contributions:
                    contributions[group_name] = 0.0
                contributions[group_name] += abs(val)
    
    if len(contributions) == 0:
        return ("", "", "")
    
    # 절댓값 기준 내림차순 정렬
    sorted_groups = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    # 한글명으로 변환
    top3 = []
    for group_name, _ in sorted_groups[:3]:
        korean_name = FACTOR_GROUP_NAMES.get(group_name, group_name)
        top3.append(korean_name)
    
    # 부족한 경우 빈 문자열로 채움
    while len(top3) < 3:
        top3.append("")
    
    return tuple(top3[:3])


def get_stock_names(tickers: List[str]) -> Dict[str, str]:
    """
    티커 리스트로부터 종목명 딕셔너리 생성
    
    Returns:
        {티커: 종목명} 딕셔너리
    """
    try:
        import pykrx.stock as stock
    except ImportError:
        print("경고: pykrx가 설치되어 있지 않습니다. 종목명 없이 티커만 표시됩니다.")
        return {ticker: "" for ticker in tickers}
    
    stock_names = {}
    unique_tickers = sorted(set(tickers))
    
    print(f"  - 종목명 조회 중 ({len(unique_tickers)}개 티커)...")
    for i, ticker in enumerate(unique_tickers):
        try:
            name = stock.get_market_ticker_name(ticker)
            stock_names[ticker] = name if name else ""
        except Exception as e:
            stock_names[ticker] = ""
            if (i + 1) % 50 == 0:
                print(f"    진행: {i+1}/{len(unique_tickers)}")
    
    print(f"  - 종목명 조회 완료: {sum(1 for v in stock_names.values() if v)}개 성공")
    return stock_names


def export_track_a_to_csv(
    ranking_file: str = "data/interim/ranking_short_daily.parquet",
    dataset_file: str = "data/interim/dataset_daily.parquet",
    weights_config: str = "configs/feature_weights_short_hitratio_optimized.yaml",
    groups_config: str = "configs/feature_groups_short.yaml",
    output_file: str = "data/processed/track_a_output_2023_2024.csv",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    normalization_method: str = "percentile",
):
    """
    Track A 산출물을 CSV로 저장
    
    Args:
        ranking_file: ranking_short_daily.parquet 경로
        dataset_file: dataset_daily.parquet 경로 (피처 데이터)
        weights_config: 피처 가중치 설정 파일 경로
        output_file: 출력 CSV 파일 경로
        start_date: 시작 날짜
        end_date: 종료 날짜
        normalization_method: 정규화 방법
    """
    project_root = Path(__file__).resolve().parent.parent
    ranking_path = project_root / ranking_file
    dataset_path = project_root / dataset_file
    weights_path = project_root / weights_config
    groups_path = project_root / groups_config
    output_path = project_root / output_file
    
    print(f"[1/5] 데이터 로드 중...")
    print(f"  - Ranking: {ranking_path}")
    print(f"  - Dataset: {dataset_path}")
    
    # 랭킹 데이터 로드
    ranking_df = pd.read_parquet(ranking_path)
    ranking_df["date"] = pd.to_datetime(ranking_df["date"])
    
    # 날짜 필터링
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    ranking_df = ranking_df[
        (ranking_df["date"] >= start_dt) & (ranking_df["date"] <= end_dt)
    ].copy()
    
    print(f"  - 랭킹 데이터: {len(ranking_df):,}행, {ranking_df['date'].nunique()}개 날짜")
    
    # 원본 데이터 로드 (피처 포함)
    dataset_df = pd.read_parquet(dataset_path)
    dataset_df["date"] = pd.to_datetime(dataset_df["date"])
    
    # 날짜 필터링
    dataset_df = dataset_df[
        (dataset_df["date"] >= start_dt) & (dataset_df["date"] <= end_dt)
    ].copy()
    
    print(f"  - 원본 데이터: {len(dataset_df):,}행")
    
    # 랭킹과 원본 데이터 병합
    print(f"[2/5] 데이터 병합 중...")
    merged_df = ranking_df.merge(
        dataset_df,
        on=["date", "ticker"],
        how="inner",
    )
    print(f"  - 병합 결과: {len(merged_df):,}행")
    
    # 피처 컬럼 선택
    print(f"[3/5] 피처 가중치 로드 및 기여도 계산 중...")
    feature_cols = _pick_feature_cols(merged_df)
    print(f"  - 사용 피처: {len(feature_cols)}개")
    
    # 피처 가중치 로드
    feature_weights = load_feature_weights(weights_path)
    print(f"  - 가중치 로드: {len(feature_weights)}개")
    
    # 가중치가 없는 피처는 제외
    feature_cols_with_weights = [f for f in feature_cols if f in feature_weights and feature_weights[f] != 0]
    print(f"  - 가중치가 있는 피처: {len(feature_cols_with_weights)}개")
    
    if len(feature_cols_with_weights) == 0:
        raise ValueError("가중치가 있는 피처가 없습니다.")
    
    # sector_col 확인
    sector_col = None
    if "sector_name" in merged_df.columns:
        if merged_df["sector_name"].notna().sum() > 0:
            sector_col = "sector_name"
    
    # 팩터 기여도 계산
    merged_with_contrib = calculate_feature_contributions(
        merged_df,
        feature_cols_with_weights,
        feature_weights,
        normalization_method=normalization_method,
        sector_col=sector_col,
        use_sector_relative=True,
    )
    
    # 피처-그룹 매핑 로드
    print(f"[4/6] 피처 그룹 매핑 로드 중...")
    feature_to_group = get_feature_to_group_mapping(groups_path)
    print(f"  - 그룹 매핑: {len(feature_to_group)}개 피처")
    
    print(f"[5/6] Top3 팩터 그룹 추출 중...")
    # Top3 팩터 그룹 추출 (한글명)
    top3_groups = merged_with_contrib.apply(
        lambda row: get_top3_factor_groups(row, feature_cols_with_weights, feature_to_group),
        axis=1,
    )
    
    # 종목명 조회
    print(f"[6/6] 종목명 조회 중...")
    unique_tickers = merged_with_contrib["ticker"].unique().tolist()
    stock_names = get_stock_names(unique_tickers)
    
    # 종목명과 티커 결합
    merged_with_contrib["stock_name"] = merged_with_contrib["ticker"].map(stock_names)
    merged_with_contrib["종목명_티커"] = merged_with_contrib.apply(
        lambda row: f"{row['stock_name']}({row['ticker']})" if row['stock_name'] else row['ticker'],
        axis=1,
    )
    
    # 결과 DataFrame 구성
    result_df = pd.DataFrame({
        "날짜": merged_with_contrib["date"].dt.strftime("%Y-%m-%d"),
        "종목명(티커)": merged_with_contrib["종목명_티커"],
        "스코어": merged_with_contrib["score_total"],
        "Top1_팩터그룹": [f[0] for f in top3_groups],
        "Top2_팩터그룹": [f[1] for f in top3_groups],
        "Top3_팩터그룹": [f[2] for f in top3_groups],
    })
    
    # Top3 팩터셋을 하나의 컬럼으로 합치기 (예: "팩터1|팩터2|팩터3")
    result_df["Top3_영향_팩터셋"] = result_df.apply(
        lambda row: "|".join([f for f in [row["Top1_팩터그룹"], row["Top2_팩터그룹"], row["Top3_팩터그룹"]] if f]),
        axis=1,
    )
    
    # 최종 컬럼 선택
    final_df = result_df[["날짜", "종목명(티커)", "스코어", "Top3_영향_팩터셋"]].copy()
    
    # 정렬 (날짜, 스코어 내림차순)
    final_df = final_df.sort_values(["날짜", "스코어"], ascending=[True, False])
    
    print(f"[7/7] CSV 저장 중...")
    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 완료!")
    print(f"  - 출력 파일: {output_path}")
    print(f"  - 총 행 수: {len(final_df):,}")
    print(f"  - 날짜 범위: {final_df['날짜'].min()} ~ {final_df['날짜'].max()}")
    print(f"  - 종목 수: {final_df['종목명(티커)'].nunique()}개")
    
    # 샘플 출력
    print(f"\n📊 샘플 데이터 (상위 10행):")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Track A 산출물을 CSV로 저장")
    parser.add_argument("--ranking", type=str, default="data/interim/ranking_short_daily.parquet",
                       help="랭킹 파일 경로")
    parser.add_argument("--dataset", type=str, default="data/interim/dataset_daily.parquet",
                       help="원본 데이터 파일 경로")
    parser.add_argument("--weights", type=str, default="configs/feature_weights_short_hitratio_optimized.yaml",
                       help="피처 가중치 설정 파일 경로")
    parser.add_argument("--groups", type=str, default="configs/feature_groups_short.yaml",
                       help="피처 그룹 설정 파일 경로")
    parser.add_argument("--output", type=str, default="data/processed/track_a_output_2023_2024.csv",
                       help="출력 CSV 파일 경로")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                       help="종료 날짜 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    export_track_a_to_csv(
        ranking_file=args.ranking,
        dataset_file=args.dataset,
        weights_config=args.weights,
        groups_config=args.groups,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
    )

