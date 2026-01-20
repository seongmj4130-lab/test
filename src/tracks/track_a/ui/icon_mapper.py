# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/ui/icon_mapper.py
"""
UI 아이콘 매핑 유틸리티

그룹별 기여도(contrib_*)와 Top Features를 UI 친화적인 아이콘 형식으로 변환
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

# 기본 아이콘 매핑 (설정 파일이 없을 때 사용)
DEFAULT_GROUP_ICONS = {
    "fundamental": {
        "icon": "💰",
        "label": "재무",
        "description": "재무 지표가 높은 순위에 기여",
        "color": "#4CAF50",
    },
    "price": {
        "icon": "📈",
        "label": "가격",
        "description": "가격/기술 지표가 높은 순위에 기여",
        "color": "#2196F3",
    },
    "sector_adj": {
        "icon": "🏢",
        "label": "섹터",
        "description": "섹터 상대 성과가 높은 순위에 기여",
        "color": "#FF9800",
    },
    "core": {
        "icon": "⭐",
        "label": "핵심",
        "description": "핵심 지표가 높은 순위에 기여",
        "color": "#9C27B0",
    },
    "other": {
        "icon": "📊",
        "label": "기타",
        "description": "기타 지표가 높은 순위에 기여",
        "color": "#666666",
    },
}

DEFAULT_FEATURE_ICONS = {
    "roe": {"icon": "📊", "label": "ROE", "description": "자기자본이익률"},
    "debt_ratio": {"icon": "💳", "label": "부채비율", "description": "재무 안정성"},
    "net_income": {"icon": "💰", "label": "순이익", "description": "수익성"},
    "equity": {"icon": "🏦", "label": "자본", "description": "자본 규모"},
    "total_liabilities": {"icon": "📋", "label": "총부채", "description": "부채 규모"},
    "momentum": {"icon": "📈", "label": "모멘텀", "description": "추세 강도"},
    "volume": {"icon": "📊", "label": "거래량", "description": "유동성"},
    "volatility": {"icon": "📉", "label": "변동성", "description": "가격 변동성"},
    "sector_relative": {
        "icon": "🏢",
        "label": "섹터 상대",
        "description": "업종 대비 성과",
    },
}


def load_icon_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """아이콘 설정 파일 로드"""
    if config_path is None:
        # 기본 경로: configs/ui_icons.yaml
        config_path = (
            Path(__file__).parent.parent.parent.parent / "configs" / "ui_icons.yaml"
        )

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config

    # 기본값 반환
    return {"groups": DEFAULT_GROUP_ICONS, "features": DEFAULT_FEATURE_ICONS}


def map_contributions_to_icons(
    contrib_dict: dict[str, float],
    config: Optional[dict[str, Any]] = None,
    top_k: int = 3,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """
    contrib_* 딕셔너리를 아이콘 리스트로 변환

    Args:
        contrib_dict: {"contrib_fundamental": 0.35, "contrib_price": 0.28, ...}
                     또는 {"fundamental": 0.35, "price": 0.28, ...}
        config: 아이콘 설정 딕셔너리 (None이면 기본값 사용)
        top_k: 상위 K개 그룹만 반환
        threshold: 기여도 임계값 (이 값 미만은 제외)

    Returns:
        [{"icon": "💰", "label": "재무", "value": 0.35, "description": "...", "color": "#4CAF50"}, ...]
    """
    if config is None:
        config = load_icon_config()

    group_icons_config = config.get("groups", DEFAULT_GROUP_ICONS)

    # contrib_ 접두사 제거
    normalized_dict = {}
    for key, value in contrib_dict.items():
        if value is None or pd.isna(value):
            continue
        group_key = key.replace("contrib_", "")
        normalized_dict[group_key] = float(value)

    # 기여도 절댓값 기준 정렬
    sorted_contribs = sorted(
        normalized_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )

    icons = []
    for group_name, contrib_value in sorted_contribs[:top_k]:
        if abs(contrib_value) < threshold:
            continue

        icon_info = group_icons_config.get(
            group_name, DEFAULT_GROUP_ICONS.get("other", {})
        )

        icons.append(
            {
                "icon": icon_info.get("icon", "📊"),
                "label": icon_info.get("label", group_name),
                "value": contrib_value,
                "description": icon_info.get("description", ""),
                "color": icon_info.get("color", "#666666"),
            }
        )

    return icons


def parse_top_features(
    top_features_str: str,
    config: Optional[dict[str, Any]] = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    top_features 문자열 파싱

    Args:
        top_features_str: "roe:0.12;debt_ratio:0.08;..." 형식의 문자열
        config: 아이콘 설정 딕셔너리 (None이면 기본값 사용)
        top_k: 상위 K개 피처만 반환

    Returns:
        [{"feature": "roe", "value": 0.12, "icon": "📊", "label": "ROE", "description": "..."}, ...]
    """
    if config is None:
        config = load_icon_config()

    feature_icons_config = config.get("features", DEFAULT_FEATURE_ICONS)

    if not top_features_str or pd.isna(top_features_str):
        return []

    features = []
    for item in str(top_features_str).split(";"):
        if ":" not in item:
            continue

        parts = item.split(":", 1)
        if len(parts) != 2:
            continue

        feat, val_str = parts
        feat = feat.strip()

        try:
            val = float(val_str.strip())
            feat_info = feature_icons_config.get(
                feat, {"icon": "📊", "label": feat, "description": ""}
            )

            features.append(
                {
                    "feature": feat,
                    "value": val,
                    "icon": feat_info.get("icon", "📊"),
                    "label": feat_info.get("label", feat),
                    "description": feat_info.get("description", ""),
                }
            )
        except (ValueError, AttributeError):
            continue

    # 절댓값 기준 정렬 후 상위 K개 반환
    features.sort(key=lambda x: abs(x["value"]), reverse=True)
    return features[:top_k]


def enrich_ranking_with_icons(
    ranking_row: pd.Series,
    config: Optional[dict[str, Any]] = None,
    group_top_k: int = 3,
    feature_top_k: int = 3,
) -> dict[str, Any]:
    """
    ranking_daily의 한 행을 UI 친화적인 형식으로 변환

    Args:
        ranking_row: ranking_daily의 한 행 (contrib_*, top_features 포함)
        config: 아이콘 설정 딕셔너리
        group_top_k: 그룹 아이콘 상위 K개
        feature_top_k: 피처 아이콘 상위 K개

    Returns:
        {
            "ticker": "005930",
            "rank": 8,
            "score": 0.8275,
            "group_icons": [...],
            "feature_icons": [...]
        }
    """
    if config is None:
        config = load_icon_config()

    # 그룹별 기여도 수집
    contrib_dict = {}
    for col in ranking_row.index:
        if col.startswith("contrib_"):
            contrib_dict[col] = ranking_row[col]

    group_icons = map_contributions_to_icons(
        contrib_dict, config=config, top_k=group_top_k
    )

    # Top features 파싱
    top_features_str = ranking_row.get("top_features", "")
    feature_icons = parse_top_features(
        top_features_str, config=config, top_k=feature_top_k
    )

    return {
        "ticker": str(ranking_row.get("ticker", "")),
        "rank": int(ranking_row.get("rank_total", 0)),
        "score": float(ranking_row.get("score_total", 0.0)),
        "group_icons": group_icons,
        "feature_icons": feature_icons,
    }


if __name__ == "__main__":
    # 테스트
    from pathlib import Path

    # 예시 데이터
    contrib_dict = {
        "contrib_fundamental": 0.35,
        "contrib_price": 0.28,
        "contrib_sector_adj": 0.15,
    }

    print("그룹별 아이콘 매핑 테스트:")
    icons = map_contributions_to_icons(contrib_dict)
    for icon in icons:
        print(f"  {icon['icon']} {icon['label']}: {icon['value']:.2f}")

    print("\nTop Features 파싱 테스트:")
    top_features_str = "roe:0.12;debt_ratio:0.08;net_income:0.05"
    features = parse_top_features(top_features_str)
    for feat in features:
        print(f"  {feat['icon']} {feat['label']}: {feat['value']:.2f}")
