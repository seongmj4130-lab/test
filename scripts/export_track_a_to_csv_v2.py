"""
Track A 산출물을 CSV로 저장 (단기, 장기, 통합)
- 날짜 범위: 2023-01-01 ~ 2024-12-31
- 컬럼: 날짜, 종목명(티커), 스코어, top3 영향 팩터셋(절댓값)
- CSV 상단에 팩터셋 정보 메타데이터 포함
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.components.ranking.score_engine import (
    _pick_feature_cols,
    normalize_feature_cross_sectional,
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


def load_feature_weights(weights_config_path: Path) -> dict[str, float]:
    """피처 가중치 파일 로드"""
    if not weights_config_path.exists():
        print(f"경고: 가중치 파일을 찾을 수 없습니다: {weights_config_path}")
        return {}

    with open(weights_config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("feature_weights", {})


def get_feature_groups_info(feature_groups_config: Path) -> dict[str, list[str]]:
    """
    팩터 그룹별 피처 목록 반환

    Returns:
        {그룹명: [피처리스트]} 딕셔너리
    """
    if not feature_groups_config.exists():
        return {}

    cfg_groups = load_feature_groups(feature_groups_config)
    feature_groups = get_feature_groups(cfg_groups)

    return feature_groups


def generate_metadata_header(feature_groups: dict[str, list[str]]) -> list[str]:
    """
    CSV 상단에 추가할 메타데이터 헤더 생성

    Returns:
        메타데이터 라인 리스트
    """
    lines = [
        "# Track A 랭킹 결과",
        f"# 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# 팩터 그룹별 포함 피처 목록:",
        "#",
    ]

    # 그룹별로 정렬하여 출력
    sorted_groups = sorted(feature_groups.items())

    for group_name, features in sorted_groups:
        korean_name = FACTOR_GROUP_NAMES.get(group_name, group_name)
        lines.append(f"# [{korean_name} ({group_name})]")
        for feat in sorted(features):
            lines.append(f"#   - {feat}")
        lines.append("#")

    return lines


def calculate_feature_contributions(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_weights: dict[str, float],
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
) -> dict[str, str]:
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
    feature_cols: list[str],
    feature_to_group: dict[str, str],
    prefix: str = "contrib_",
) -> tuple[str, str, str]:
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


def get_stock_names(tickers: list[str]) -> dict[str, str]:
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
        except Exception:
            stock_names[ticker] = ""
        if (i + 1) % 50 == 0:
            print(f"    진행: {i+1}/{len(unique_tickers)}")

    print(f"  - 종목명 조회 완료: {sum(1 for v in stock_names.values() if v)}개 성공")
    return stock_names


def save_csv_with_metadata(
    df: pd.DataFrame,
    output_path: Path,
    metadata_lines: list[str],
):
    """
    메타데이터 헤더와 함께 CSV 저장
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        # 메타데이터 쓰기
        for line in metadata_lines:
            f.write(line + "\n")

        # 실제 데이터 쓰기
        df.to_csv(f, index=False, lineterminator="\n")


def process_ranking_data(
    ranking_file: str,
    dataset_file: str,
    weights_config: str,
    groups_config: str,
    start_date: str,
    end_date: str,
    normalization_method: str = "percentile",
    horizon_name: str = "",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    랭킹 데이터 처리

    Returns:
        (결과 DataFrame, 팩터 그룹 정보)
    """
    project_root = Path(__file__).resolve().parent.parent
    ranking_path = project_root / ranking_file
    dataset_path = project_root / dataset_file
    weights_path = project_root / weights_config
    groups_path = project_root / groups_config

    print(f"\n[{horizon_name}] 데이터 로드 중...")
    print(f"  - Ranking: {ranking_path.name}")
    print(f"  - Dataset: {dataset_path.name}")

    # 랭킹 데이터 로드
    ranking_df = pd.read_parquet(ranking_path)
    ranking_df["date"] = pd.to_datetime(ranking_df["date"])

    # 날짜 필터링
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    ranking_df = ranking_df[
        (ranking_df["date"] >= start_dt) & (ranking_df["date"] <= end_dt)
    ].copy()

    print(
        f"  - 랭킹 데이터: {len(ranking_df):,}행, {ranking_df['date'].nunique()}개 날짜"
    )

    # 원본 데이터 로드 (피처 포함)
    dataset_df = pd.read_parquet(dataset_path)
    dataset_df["date"] = pd.to_datetime(dataset_df["date"])

    # 날짜 필터링
    dataset_df = dataset_df[
        (dataset_df["date"] >= start_dt) & (dataset_df["date"] <= end_dt)
    ].copy()

    print(f"  - 원본 데이터: {len(dataset_df):,}행")

    # 랭킹과 원본 데이터 병합
    print("  - 데이터 병합 중...")
    merged_df = ranking_df.merge(
        dataset_df,
        on=["date", "ticker"],
        how="inner",
    )
    print(f"  - 병합 결과: {len(merged_df):,}행")

    # 피처 컬럼 선택
    print("  - 피처 가중치 로드 및 기여도 계산 중...")
    feature_cols = _pick_feature_cols(merged_df)
    print(f"  - 사용 피처: {len(feature_cols)}개")

    # 피처 가중치 로드
    feature_weights = load_feature_weights(weights_path)
    print(f"  - 가중치 로드: {len(feature_weights)}개")

    # 가중치가 없는 피처는 제외
    feature_cols_with_weights = [
        f for f in feature_cols if f in feature_weights and feature_weights[f] != 0
    ]
    print(f"  - 가중치가 있는 피처: {len(feature_cols_with_weights)}개")

    if len(feature_cols_with_weights) == 0:
        raise ValueError("가중치가 있는 피처가 없습니다.")

    # 팩터 그룹 정보 로드
    feature_groups = get_feature_groups_info(groups_path)

    # 피처-그룹 매핑 로드
    feature_to_group = get_feature_to_group_mapping(groups_path)
    print(f"  - 그룹 매핑: {len(feature_to_group)}개 피처")

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

    print("  - Top3 팩터 그룹 추출 중...")
    # Top3 팩터 그룹 추출 (한글명)
    top3_groups = merged_with_contrib.apply(
        lambda row: get_top3_factor_groups(
            row, feature_cols_with_weights, feature_to_group
        ),
        axis=1,
    )

    # 결과 DataFrame 구성
    result_df = pd.DataFrame(
        {
            "날짜": merged_with_contrib["date"].dt.strftime("%Y-%m-%d"),
            "종목명(티커)": merged_with_contrib["ticker"],  # 나중에 종목명 추가
            "스코어": merged_with_contrib["score_total"],
            "Top1_팩터그룹": [f[0] for f in top3_groups],
            "Top2_팩터그룹": [f[1] for f in top3_groups],
            "Top3_팩터그룹": [f[2] for f in top3_groups],
        }
    )

    # Top3 팩터셋을 하나의 컬럼으로 합치기
    result_df["Top3_영향_팩터셋"] = result_df.apply(
        lambda row: "|".join(
            [
                f
                for f in [
                    row["Top1_팩터그룹"],
                    row["Top2_팩터그룹"],
                    row["Top3_팩터그룹"],
                ]
                if f
            ]
        ),
        axis=1,
    )

    # 최종 컬럼 선택
    final_df = result_df[["날짜", "종목명(티커)", "스코어", "Top3_영향_팩터셋"]].copy()

    # 정렬 (날짜, 스코어 내림차순)
    final_df = final_df.sort_values(["날짜", "스코어"], ascending=[True, False])

    return final_df, feature_groups


def export_track_a_to_csv(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    normalization_method: str = "percentile",
    short_ranking_file: str = "data/interim/ranking_short_daily.parquet",
    long_ranking_file: str = "data/interim/ranking_long_daily.parquet",
    dataset_file: str = "data/interim/dataset_daily.parquet",
    short_weights_config: str = "configs/feature_weights_short_hitratio_optimized.yaml",
    long_weights_config: str = "configs/feature_weights_long_ic_optimized.yaml",
    short_groups_config: str = "configs/feature_groups_short.yaml",
    long_groups_config: str = "configs/feature_groups_long.yaml",
    output_dir: str = "data/processed",
):
    """
    Track A 산출물을 CSV로 저장 (단기, 장기, 통합)
    """
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / output_dir

    print("=" * 60)
    print("Track A 랭킹 결과 CSV 생성")
    print("=" * 60)

    # 종목명 조회 (한 번만)
    print("\n[공통] 종목명 조회 중...")
    # 먼저 티커 목록 확인
    ranking_short = pd.read_parquet(project_root / short_ranking_file)
    unique_tickers = ranking_short["ticker"].unique().tolist()
    stock_names = get_stock_names(unique_tickers)

    # 단기 처리
    short_df, short_groups = process_ranking_data(
        ranking_file=short_ranking_file,
        dataset_file=dataset_file,
        weights_config=short_weights_config,
        groups_config=short_groups_config,
        start_date=start_date,
        end_date=end_date,
        normalization_method=normalization_method,
        horizon_name="단기",
    )

    # 종목명 추가
    short_df["종목명(티커)"] = short_df["종목명(티커)"].map(
        lambda ticker: (
            f"{stock_names.get(ticker, '')}({ticker})"
            if stock_names.get(ticker)
            else ticker
        )
    )

    # 장기 처리
    long_df, long_groups = process_ranking_data(
        ranking_file=long_ranking_file,
        dataset_file=dataset_file,
        weights_config=long_weights_config,
        groups_config=long_groups_config,
        start_date=start_date,
        end_date=end_date,
        normalization_method=normalization_method,
        horizon_name="장기",
    )

    # 종목명 추가
    long_df["종목명(티커)"] = long_df["종목명(티커)"].map(
        lambda ticker: (
            f"{stock_names.get(ticker, '')}({ticker})"
            if stock_names.get(ticker)
            else ticker
        )
    )

    # 통합 처리 (단기 0.5, 장기 0.5)
    print("\n[통합] 단기 0.5 + 장기 0.5 비율로 통합 중...")

    # 날짜, 티커 기준으로 병합
    short_for_merge = short_df.copy()
    short_for_merge["ticker"] = (
        short_for_merge["종목명(티커)"]
        .str.extract(r"\((\d+)\)")
        .fillna(short_for_merge["종목명(티커)"])
    )

    long_for_merge = long_df.copy()
    long_for_merge["ticker"] = (
        long_for_merge["종목명(티커)"]
        .str.extract(r"\((\d+)\)")
        .fillna(long_for_merge["종목명(티커)"])
    )

    merged_combined = short_for_merge[["날짜", "ticker", "스코어"]].merge(
        long_for_merge[["날짜", "ticker", "스코어"]],
        on=["날짜", "ticker"],
        how="inner",
        suffixes=("_short", "_long"),
    )

    # 통합 스코어 계산 (단기 0.5, 장기 0.5)
    merged_combined["스코어"] = (
        0.5 * merged_combined["스코어_short"] + 0.5 * merged_combined["스코어_long"]
    )

    # Top3 팩터셋도 통합 (단기와 장기 각각의 Top3를 합쳐서 상위 3개 선택)
    # 일단 단기 Top3를 기본으로 사용 (간단한 방법)
    short_top3 = short_for_merge.set_index(["날짜", "ticker"])["Top3_영향_팩터셋"]
    long_top3 = long_for_merge.set_index(["날짜", "ticker"])["Top3_영향_팩터셋"]

    # 통합 팩터셋: 단기와 장기를 합쳐서 고유값 추출
    combined_df = merged_combined[["날짜", "ticker", "스코어"]].copy()
    combined_df["종목명(티커)"] = (
        short_for_merge.set_index(["날짜", "ticker"])
        .loc[list(zip(combined_df["날짜"], combined_df["ticker"]))]["종목명(티커)"]
        .values
    )

    # 단기 Top3 팩터셋을 기본으로 사용 (통합 팩터셋 계산은 복잡하므로)
    combined_df["Top3_영향_팩터셋"] = (
        short_for_merge.set_index(["날짜", "ticker"])
        .loc[list(zip(combined_df["날짜"], combined_df["ticker"]))]["Top3_영향_팩터셋"]
        .values
    )

    combined_df = combined_df[
        ["날짜", "종목명(티커)", "스코어", "Top3_영향_팩터셋"]
    ].copy()
    combined_df = combined_df.sort_values(["날짜", "스코어"], ascending=[True, False])

    # 통합용 팩터 그룹 (단기 + 장기 통합)
    combined_groups = {**short_groups, **long_groups}
    # 중복 제거
    for group_name in combined_groups:
        combined_groups[group_name] = sorted(list(set(combined_groups[group_name])))

    # CSV 저장
    print("\n[저장] CSV 파일 생성 중...")

    # 단기 CSV
    short_metadata = generate_metadata_header(short_groups)
    short_metadata.insert(2, "# 랭킹 유형: 단기 (Short-term)")
    short_output = output_path / "track_a_short_2023_2024.csv"
    save_csv_with_metadata(short_df, short_output, short_metadata)
    print(f"  ✅ 단기: {short_output}")
    print(f"     - 행 수: {len(short_df):,}")

    # 장기 CSV
    long_metadata = generate_metadata_header(long_groups)
    long_metadata.insert(2, "# 랭킹 유형: 장기 (Long-term)")
    long_output = output_path / "track_a_long_2023_2024.csv"
    save_csv_with_metadata(long_df, long_output, long_metadata)
    print(f"  ✅ 장기: {long_output}")
    print(f"     - 행 수: {len(long_df):,}")

    # 통합 CSV
    combined_metadata = generate_metadata_header(combined_groups)
    combined_metadata.insert(2, "# 랭킹 유형: 통합 (Combined: 단기 0.5 + 장기 0.5)")
    combined_output = output_path / "track_a_combined_2023_2024.csv"
    save_csv_with_metadata(combined_df, combined_output, combined_metadata)
    print(f"  ✅ 통합: {combined_output}")
    print(f"     - 행 수: {len(combined_df):,}")

    print("\n✅ 모든 파일 생성 완료!")
    print(f"  - 날짜 범위: {start_date} ~ {end_date}")
    print(f"  - 종목 수: {len(unique_tickers)}개")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Track A 산출물을 CSV로 저장 (단기, 장기, 통합)"
    )
    parser.add_argument(
        "--start-date", type=str, default="2023-01-01", help="시작 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-12-31", help="종료 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="출력 디렉토리"
    )

    args = parser.parse_args()

    export_track_a_to_csv(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
