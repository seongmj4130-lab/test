#!/usr/bin/env python3
"""
Holdout 전체 기간 일별 랭킹 top20 생성 (단기/장기 모두)
Unknown 종목들은 별도 목록화
"""

import sys
from pathlib import Path
from typing import Optional

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
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_weights_and_groups(
    horizon: str, cfg: dict
) -> tuple[dict[str, float], Optional[dict[str, list[str]]]]:
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

    with open(weights_path, encoding="utf-8") as f:
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


def recalculate_for_single_date(date: str, horizon: str, cfg: dict) -> pd.DataFrame:
    """단일 날짜에 대한 피처 영향도 재계산"""
    # 데이터 로드
    dataset_path = project_root / "data" / "interim" / "dataset_daily.parquet"
    ranking_path = (
        project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20.csv"
    )

    dataset_daily = pd.read_parquet(dataset_path)
    ranking_df = pd.read_csv(ranking_path)

    # 날짜 필터링
    date_ts = pd.Timestamp(date)
    day_df = dataset_daily[dataset_daily["date"] == date_ts].copy()
    day_ranking = ranking_df[ranking_df["date"] == date].copy()

    if len(day_df) == 0:
        print(f"Warning: dataset_daily에 date={date} 행이 없습니다. 건너뜀")
        return pd.DataFrame()
    if len(day_ranking) == 0:
        print(f"Warning: ranking 데이터에 date={date} 행이 없습니다. 건너뜀")
        return pd.DataFrame()

    # 데이터 타입 통일
    day_ranking["date"] = pd.to_datetime(day_ranking["date"])
    day_ranking["ticker"] = day_ranking["ticker"].astype(str).str.zfill(6)
    day_df["ticker"] = day_df["ticker"].astype(str).str.zfill(6)

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

    # 필요한 컬럼들만 선택
    result_cols = [
        "ranking",
        "ticker",
        "date",
        "score_short",
        "score_long",
        "score_ens",
    ]

    # Top 3 그룹 정보 추가
    for i in range(1, 4):
        group_col = f"group_top{i}_name"
        if group_col in merged.columns:
            merged[f"top{i}_feature_group"] = merged[group_col]
        else:
            merged[f"top{i}_feature_group"] = "unknown"

    result_cols.extend(
        [
            "top1_feature_group",
            "top2_feature_group",
            "top3_feature_group",
            "original_ranking",
        ]
    )

    return merged[result_cols]


def generate_full_holdout_rankings(
    horizon: str, max_dates: Optional[int] = None
) -> pd.DataFrame:
    """Holdout 전체 기간 랭킹 생성"""

    print(f"=== {horizon.upper()} Holdout 전체 기간 랭킹 생성 시작 ===")

    cfg = load_config()

    # 원본 랭킹 데이터에서 모든 날짜 가져오기
    ranking_path = (
        project_root / "data" / f"daily_all_business_days_{horizon}_ranking_top20.csv"
    )
    ranking_df = pd.read_csv(ranking_path)

    all_dates = sorted(ranking_df["date"].unique())

    # Holdout 기간으로 필터링 (2023-01-01 이후)
    holdout_dates = [d for d in all_dates if d >= "2023-01-01"]

    if max_dates:
        holdout_dates = holdout_dates[:max_dates]

    print(f"처리할 Holdout 날짜 수: {len(holdout_dates)} (전체: {len(all_dates)})")

    all_results = []

    for i, date in enumerate(holdout_dates):
        if (i + 1) % 50 == 0:
            print(f"진행 중: {i+1}/{len(holdout_dates)} 날짜 처리 완료")

        try:
            result = recalculate_for_single_date(date, horizon, cfg)
            if not result.empty:
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


def create_formatted_rankings(horizon: str, data: pd.DataFrame) -> pd.DataFrame:
    """최종 포맷팅된 랭킹 데이터 생성"""

    # 티커 포맷팅 및 종목명 추가
    ticker_to_name = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "035420": "NAVER",
        "034730": "SK텔레콤",
        "005380": "현대차",
        "000270": "기아",
        "035720": "카카오",
        "005490": "POSCO홀딩스",
        "051910": "LG화학",
        "012330": "현대모비스",
        "055550": "신한지주",
        "032830": "삼성생명",
        "003550": "LG",
        "006400": "삼성SDI",
        "086790": "하나금융지주",
        "138040": "메리츠금융지주",
        "036570": "엔씨소프트",
        "000810": "삼성화재",
        "009150": "삼성전기",
        "034730": "SK",
        "352820": "하이브",
        "011200": "HMM",
        "010130": "고려아연",
        "009830": "한화솔루션",
        "241560": "두산밥캣",
        "137310": "에스디바이오센서",
        "003240": "태광산업",
    }

    # 포맷팅된 데이터프레임 생성
    formatted = pd.DataFrame(
        {
            "랭킹": data["ranking"],
            "종목명(ticker)": data["ticker"].apply(
                lambda x: f"{ticker_to_name.get(x, 'Unknown')}({x})"
            ),
            "날짜": data["date"],
            "score": data["score_ens"],
            "top3 피쳐그룹": data["top1_feature_group"].astype(str)
            + ","
            + data["top2_feature_group"].astype(str)
            + ","
            + data["top3_feature_group"].astype(str),
        }
    )

    return formatted


def extract_unknown_stocks(horizon: str, data: pd.DataFrame) -> pd.DataFrame:
    """Unknown 종목 추출"""
    unknown_data = data[data["종목명(ticker)"].str.startswith("Unknown")]
    unknown_stocks = unknown_data[["종목명(ticker)"]].drop_duplicates()
    unknown_stocks["전략"] = horizon.upper()

    return unknown_stocks


def save_results(
    horizon: str, formatted_data: pd.DataFrame, unknown_stocks: pd.DataFrame
):
    """결과 파일들 저장"""

    # 포맷팅된 랭킹 데이터 저장
    csv_path = project_root / "data" / f"holdout_daily_ranking_{horizon}_top20.csv"
    parquet_path = (
        project_root / "data" / f"holdout_daily_ranking_{horizon}_top20.parquet"
    )

    formatted_data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    formatted_data.to_parquet(parquet_path, index=False)

    print(f"=== {horizon.upper()} 파일 저장 완료 ===")
    print(f"CSV: {csv_path} ({len(formatted_data)}행)")
    print(f"Parquet: {parquet_path}")

    # Unknown 종목 목록 저장
    unknown_path = project_root / "data" / f"unknown_stocks_{horizon}.csv"
    unknown_stocks.to_csv(unknown_path, index=False, encoding="utf-8-sig")
    print(f"Unknown 종목: {unknown_path} ({len(unknown_stocks)}개)")


def main():
    """메인 실행 함수"""
    print("Holdout 전체 기간 일별 랭킹 Top20 생성 시작")

    # 처리할 최대 날짜 수 (테스트용, None으로 하면 전체)
    max_dates = None  # 전체 기간 처리

    try:
        # 장기 데이터 생성
        print("\n1. 장기 전략 처리 중...")
        long_data = generate_full_holdout_rankings("long", max_dates)
        long_formatted = create_formatted_rankings("long", long_data)
        long_unknown = extract_unknown_stocks("long", long_formatted)
        save_results("long", long_formatted, long_unknown)

        # 단기 데이터 생성
        print("\n2. 단기 전략 처리 중...")
        short_data = generate_full_holdout_rankings("short", max_dates)
        short_formatted = create_formatted_rankings("short", short_data)
        short_unknown = extract_unknown_stocks("short", short_formatted)
        save_results("short", short_formatted, short_unknown)

        # 전체 Unknown 목록 생성
        print("\n3. 전체 Unknown 종목 목록 생성...")
        all_unknown = pd.concat([long_unknown, short_unknown]).drop_duplicates()
        all_unknown_path = (
            project_root / "data" / "all_unknown_stocks_for_hardcoding.csv"
        )
        all_unknown.to_csv(all_unknown_path, index=False, encoding="utf-8-sig")
        print(f"전체 Unknown 목록: {all_unknown_path} ({len(all_unknown)}개)")

        print("\n=== Holdout 랭킹 생성 성공! ===")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
