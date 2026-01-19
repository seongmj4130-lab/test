# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/ranking/ranking_explainability.py
"""
[Stage9] Ranking 설명가능성 확장
- 피처/팩터 그룹별 기여도 계산
- 상위 기여 피처 추출
- ranking_daily에 설명가능성 컬럼 추가
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ranking 모듈 import를 위한 경로 추가
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.components.ranking.score_engine import (
    _pick_feature_cols,
    normalize_feature_cross_sectional,
)
from src.utils.feature_groups import (
    get_feature_groups,  # [개선안 14번] feature_groups.yaml 스키마 파싱 정정
)
from src.utils.feature_groups import load_feature_groups, map_features_to_groups


def calculate_feature_contributions(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_weights: dict[str, float],
    normalized_features: dict[str, pd.Series],
    score_total: pd.Series,
) -> pd.DataFrame:
    """
    피처별 기여도 계산

    Args:
        df: 입력 DataFrame
        feature_cols: 피처 컬럼 리스트
        feature_weights: 피처별 가중치 딕셔너리
        normalized_features: 정규화된 피처 값 딕셔너리 (feature_col -> Series)
        score_total: 총점 Series

    Returns:
        기여도가 추가된 DataFrame
    """
    out = df.copy()

    # 피처별 기여도 계산 (가중치 * 정규화값)
    for feat in feature_cols:
        if feat in normalized_features and feat in feature_weights:
            weight = feature_weights[feat]
            normalized = normalized_features[feat].fillna(0.0)
            contrib = weight * normalized
            out[f"contrib_{feat}"] = contrib

    return out


def calculate_group_contributions(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_weights: dict[str, float],
    normalized_features: dict[str, pd.Series],
    feature_groups_config: Optional[Path] = None,
) -> pd.DataFrame:
    """
    그룹별 기여도 계산

    Args:
        df: 입력 DataFrame
        feature_cols: 피처 컬럼 리스트
        feature_weights: 피처별 가중치 딕셔너리
        normalized_features: 정규화된 피처 값 딕셔너리
        feature_groups_config: 피처 그룹 설정 파일 경로

    Returns:
        그룹별 기여도 컬럼이 추가된 DataFrame
    """
    out = df.copy()

    # 그룹 매핑 로드
    if feature_groups_config and Path(feature_groups_config).exists():
        config = load_feature_groups(feature_groups_config)
        groups_map = map_features_to_groups(feature_cols, config)

        # 그룹명을 표준 이름으로 매핑 (feature_groups.yaml의 그룹명 사용)
        # 예: value -> fundamental, profitability -> fundamental, technical -> price 등
        standard_groups_map = {
            "core": [],
            "price": [],
            "fundamental": [],
            "sector_adj": [],
            "other": [],
        }

        # 그룹 매핑 규칙
        for group_name, group_features in groups_map.items():
            group_lower = group_name.lower()
            if any(
                x in group_lower
                for x in ["value", "profitability", "fundamental", "roe", "debt"]
            ):
                standard_groups_map["fundamental"].extend(group_features)
            elif any(x in group_lower for x in ["sector", "relative"]):
                standard_groups_map["sector_adj"].extend(group_features)
            elif any(
                x in group_lower for x in ["technical", "momentum", "volume", "price"]
            ):
                standard_groups_map["price"].extend(group_features)
            elif any(x in group_lower for x in ["core", "base"]):
                standard_groups_map["core"].extend(group_features)
            else:
                standard_groups_map["other"].extend(group_features)

        groups_map = standard_groups_map
    else:
        # 기본 그룹 매핑 (피처명 기반 추론)
        groups_map = {
            "core": [],
            "price": [],
            "fundamental": [],
            "sector_adj": [],
            "other": [],
        }

        for feat in feature_cols:
            feat_lower = feat.lower()
            if any(
                x in feat_lower
                for x in ["roe", "debt", "equity", "income", "asset", "net_income"]
            ):
                groups_map["fundamental"].append(feat)
            elif any(x in feat_lower for x in ["sector", "relative", "_sector_", "_z"]):
                groups_map["sector_adj"].append(feat)
            elif any(
                x in feat_lower
                for x in ["momentum", "volume", "volatility", "price", "volume_ratio"]
            ):
                groups_map["price"].append(feat)
            elif any(x in feat_lower for x in ["core", "base"]):
                groups_map["core"].append(feat)
            else:
                groups_map["other"].append(feat)

    # 그룹별 기여도 계산
    group_contributions = {}
    for group_name, group_features in groups_map.items():
        contrib = pd.Series(0.0, index=out.index)
        for feat in group_features:
            if feat in normalized_features and feat in feature_weights:
                weight = feature_weights[feat]
                normalized = normalized_features[feat].fillna(0.0)
                contrib += weight * normalized
        group_contributions[f"contrib_{group_name}"] = contrib

    # 그룹별 기여도 컬럼 추가
    for col_name, contrib_series in group_contributions.items():
        out[col_name] = contrib_series

    return out


def extract_top_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_weights: dict[str, float],
    normalized_features: dict[str, pd.Series],
    top_k: int = 5,
) -> pd.Series:
    """
    상위 기여 피처 추출

    Args:
        df: 입력 DataFrame
        feature_cols: 피처 컬럼 리스트
        feature_weights: 피처별 가중치 딕셔너리
        normalized_features: 정규화된 피처 값 딕셔너리
        top_k: 상위 K개 피처 추출

    Returns:
        "featureA:0.12;featureB:0.08;..." 형식의 문자열 Series
    """
    top_features_list = []

    for idx in df.index:
        # 각 피처의 기여도 계산
        contribs = {}
        for feat in feature_cols:
            if feat in normalized_features and feat in feature_weights:
                weight = feature_weights[feat]
                normalized_val = (
                    normalized_features[feat].loc[idx]
                    if idx in normalized_features[feat].index
                    else 0.0
                )
                contrib = weight * (normalized_val if pd.notna(normalized_val) else 0.0)
                contribs[feat] = contrib

        # 절댓값 기준 상위 K개 선택
        sorted_contribs = sorted(
            contribs.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_k]

        # 문자열 형식으로 변환
        top_features_str = ";".join(
            [
                f"{feat}:{contrib:.4f}"
                for feat, contrib in sorted_contribs
                if abs(contrib) > 1e-6
            ]
        )
        top_features_list.append(top_features_str if top_features_str else "")

    return pd.Series(top_features_list, index=df.index)


def build_ranking_with_explainability(
    ranking_daily: pd.DataFrame,
    input_df: pd.DataFrame,
    feature_groups_config: Optional[Path] = None,
    normalization_method: str = "percentile",
    sector_col: Optional[str] = None,
    use_sector_relative: bool = True,
    top_k_features: int = 5,
) -> pd.DataFrame:
    """
    ranking_daily에 설명가능성 컬럼 추가

    Args:
        ranking_daily: Stage8의 ranking_daily (date, ticker, score_total, rank_total 포함)
        input_df: 원본 입력 데이터 (피처 포함)
        feature_groups_config: 피처 그룹 설정 파일 경로
        normalization_method: 정규화 방법
        sector_col: 섹터 컬럼명
        use_sector_relative: sector-relative 정규화 사용 여부
        top_k_features: 상위 기여 피처 개수

    Returns:
        설명가능성 컬럼이 추가된 ranking_daily
    """
    # ranking_daily와 input_df 병합
    merged = ranking_daily.merge(
        input_df,
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_input"),
    )

    # 피처 컬럼 선택
    feature_cols = _pick_feature_cols(input_df)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in input_df")

    # sector-relative 정규화 사용 여부 결정
    actual_sector_col = None
    if use_sector_relative and sector_col and sector_col in merged.columns:
        if merged[sector_col].notna().sum() > 0:
            actual_sector_col = sector_col

    # 피처별 정규화 (재계산)
    normalized_features = {}
    for feat in feature_cols:
        if feat not in merged.columns:
            continue
        normalized_features[feat] = normalize_feature_cross_sectional(
            merged,
            feat,
            "date",
            method=normalization_method,
            sector_col=actual_sector_col,
        )

    # 가중치 결정 (build_score_total과 동일한 로직)
    if feature_groups_config and Path(feature_groups_config).exists():
        # [개선안 14번] load_feature_groups()는 전체 config(dict)를 반환하므로 그룹->피처 리스트만 추출해야 한다.
        feature_groups = get_feature_groups(load_feature_groups(feature_groups_config))
        group_weights = {}

        group_names = set()
        for feat in feature_cols:
            for group_name, group_features in feature_groups.items():
                if feat in set(map(str, group_features)):
                    group_names.add(group_name)
                    break

        if len(group_names) > 0:
            weight_per_group = 1.0 / len(group_names)
            for group_name in group_names:
                group_weights[group_name] = weight_per_group

        feature_weights = {}
        for feat in feature_cols:
            for group_name, group_features in feature_groups.items():
                gf = set(map(str, group_features))
                if feat in gf:
                    n_features_in_group = len([f for f in feature_cols if f in gf])
                    if n_features_in_group > 0:
                        feature_weights[feat] = (
                            group_weights.get(group_name, 0.0) / n_features_in_group
                        )
                    break
            if feat not in feature_weights:
                feature_weights[feat] = 1.0 / len(feature_cols)
    else:
        feature_weights = {
            feat: 1.0 / len(normalized_features) for feat in normalized_features.keys()
        }

    # 가중치 정규화
    total_weight = sum(
        feature_weights.get(feat, 0.0) for feat in normalized_features.keys()
    )
    if total_weight > 1e-8:
        feature_weights = {
            feat: w / total_weight
            for feat, w in feature_weights.items()
            if feat in normalized_features
        }

    # 그룹별 기여도 계산
    result = calculate_group_contributions(
        merged,
        feature_cols,
        feature_weights,
        normalized_features,
        feature_groups_config,
    )

    # 상위 기여 피처 추출
    top_features = extract_top_features(
        merged,
        feature_cols,
        feature_weights,
        normalized_features,
        top_k=top_k_features,
    )
    result["top_features"] = top_features

    # 최종 컬럼 선택 (ranking_daily 기본 컬럼 + 설명가능성 컬럼)
    output_cols = ["date", "ticker", "score_total", "rank_total"]
    if "in_universe" in result.columns:
        output_cols.append("in_universe")
    if sector_col and sector_col in result.columns:
        output_cols.append(sector_col)

    # 그룹별 기여도 컬럼 추가
    group_cols = [col for col in result.columns if col.startswith("contrib_")]
    output_cols.extend(sorted(group_cols))

    # top_features 추가
    output_cols.append("top_features")

    return result[output_cols].copy()


def make_snapshot_with_explainability(
    ranking_daily: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    최신일 기준 Top/Bottom K 스냅샷 생성 (top_features 포함)

    Args:
        ranking_daily: 설명가능성 컬럼이 포함된 ranking_daily
        top_k: 상위/하위 K개

    Returns:
        스냅샷 DataFrame
    """
    if len(ranking_daily) == 0:
        return pd.DataFrame()

    # 최신일 추출
    latest_date = ranking_daily["date"].max()
    snapshot_df = ranking_daily[ranking_daily["date"] == latest_date].copy()

    # in_universe=True만 필터링
    if "in_universe" in snapshot_df.columns:
        snapshot_df = snapshot_df[snapshot_df["in_universe"]].copy()

    # rank_total 기준 정렬
    snapshot_df = snapshot_df[snapshot_df["rank_total"].notna()].copy()

    # Top K / Bottom K
    top_k_df = snapshot_df.nsmallest(top_k, "rank_total").copy()
    top_k_df["snapshot_type"] = "top10"

    bottom_k_df = snapshot_df.nlargest(top_k, "rank_total").copy()
    bottom_k_df["snapshot_type"] = "bottom10"

    # 컬럼 선택
    output_cols = [
        "snapshot_date",
        "snapshot_type",
        "ticker",
        "score_total",
        "rank_total",
    ]
    if "top_features" in snapshot_df.columns:
        output_cols.append("top_features")

    # 그룹별 기여도 컬럼 추가
    contrib_cols = [col for col in snapshot_df.columns if col.startswith("contrib_")]
    output_cols.extend(sorted(contrib_cols))

    result = pd.concat([top_k_df, bottom_k_df], ignore_index=True)
    result["snapshot_date"] = latest_date

    # 컬럼 순서 조정
    available_cols = [col for col in output_cols if col in result.columns]
    result = result[available_cols].copy()

    return result
