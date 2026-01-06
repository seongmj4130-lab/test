# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/utils/feature_groups.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml
import pandas as pd
import numpy as np

def load_feature_groups(config_path: Optional[Path] = None) -> Dict:
    """
    feature_groups.yaml 파일을 로드하여 피처 그룹 설정을 반환
    
    Args:
        config_path: YAML 파일 경로 (None이면 configs/feature_groups.yaml 사용)
    
    Returns:
        피처 그룹 설정 딕셔너리
    """
    if config_path is None:
        # 프로젝트 루트 기준으로 configs/feature_groups.yaml 찾기
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        config_path = project_root / "configs" / "feature_groups.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"feature_groups.yaml 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}

def get_feature_groups(config: Dict) -> Dict[str, List[str]]:
    """
    피처 그룹 설정에서 그룹명 -> 피처 리스트 매핑 추출
    
    Args:
        config: load_feature_groups()로 로드한 설정
    
    Returns:
        {group_name: [feature1, feature2, ...]} 딕셔너리
    """
    groups = config.get("feature_groups", {})
    result = {}
    
    for group_name, group_config in groups.items():
        if isinstance(group_config, dict) and "features" in group_config:
            features = group_config["features"]
            if isinstance(features, list):
                result[group_name] = [str(f) for f in features]
    
    return result

def get_group_target_weights(config: Dict) -> Dict[str, float]:
    """
    피처 그룹 설정에서 그룹명 -> 목표 가중치 매핑 추출
    
    Args:
        config: load_feature_groups()로 로드한 설정
    
    Returns:
        {group_name: target_weight} 딕셔너리
    """
    groups = config.get("feature_groups", {})
    result = {}
    
    for group_name, group_config in groups.items():
        if isinstance(group_config, dict) and "target_weight" in group_config:
            weight = float(group_config["target_weight"])
            result[group_name] = weight
    
    return result

def map_features_to_groups(feature_cols: List[str], config: Dict) -> Dict[str, List[str]]:
    """
    실제 사용 가능한 피처 목록을 그룹별로 분류
    
    Args:
        feature_cols: 실제 사용 가능한 피처 컬럼 리스트
        config: load_feature_groups()로 로드한 설정
    
    Returns:
        {group_name: [available_feature1, ...]} 딕셔너리 (존재하는 피처만 포함)
    """
    groups = get_feature_groups(config)
    feature_set = set(feature_cols)
    result = {}
    
    for group_name, group_features in groups.items():
        available_features = [f for f in group_features if f in feature_set]
        if available_features:
            result[group_name] = available_features
    
    return result

def calculate_feature_group_balance(
    feature_cols: List[str],
    feature_importance: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    피처 그룹 밸런싱 정보를 계산하여 DataFrame으로 반환
    
    Args:
        feature_cols: 실제 사용 가능한 피처 컬럼 리스트
        feature_importance: 피처별 중요도 (선택적, {feature_name: importance_score})
        config: load_feature_groups()로 로드한 설정 (None이면 자동 로드)
    
    Returns:
        DataFrame with columns: [group_name, n_features, features_list, target_weight, actual_weight, balance_ratio]
    """
    if config is None:
        config = load_feature_groups()
    
    groups_map = map_features_to_groups(feature_cols, config)
    target_weights = get_group_target_weights(config)
    
    # 전체 피처 수
    total_features = len(feature_cols)
    
    # 그룹별 실제 가중치 계산
    rows = []
    for group_name, group_features in groups_map.items():
        n_features = len(group_features)
        actual_weight = n_features / total_features if total_features > 0 else 0.0
        target_weight = target_weights.get(group_name, 0.0)
        balance_ratio = actual_weight / target_weight if target_weight > 0 else 0.0
        
        rows.append({
            "group_name": group_name,
            "n_features": n_features,
            "features_list": ",".join(sorted(group_features)),
            "target_weight": target_weight,
            "actual_weight": actual_weight,
            "balance_ratio": balance_ratio,
        })
    
    # 그룹에 속하지 않은 피처들
    grouped_features = set()
    for features in groups_map.values():
        grouped_features.update(features)
    ungrouped_features = [f for f in feature_cols if f not in grouped_features]
    
    if ungrouped_features:
        n_ungrouped = len(ungrouped_features)
        actual_weight_ungrouped = n_ungrouped / total_features if total_features > 0 else 0.0
        rows.append({
            "group_name": "ungrouped",
            "n_features": n_ungrouped,
            "features_list": ",".join(sorted(ungrouped_features)),
            "target_weight": 0.0,
            "actual_weight": actual_weight_ungrouped,
            "balance_ratio": 0.0,
        })
    
    df = pd.DataFrame(rows)
    
    # feature_importance가 있으면 그룹별 평균 중요도 추가
    if feature_importance:
        group_importance = {}
        for group_name, group_features in groups_map.items():
            importances = [feature_importance.get(f, 0.0) for f in group_features]
            group_importance[group_name] = np.mean(importances) if importances else 0.0
        
        if ungrouped_features:
            importances_ungrouped = [feature_importance.get(f, 0.0) for f in ungrouped_features]
            group_importance["ungrouped"] = np.mean(importances_ungrouped) if importances_ungrouped else 0.0
        
        df["avg_importance"] = df["group_name"].map(group_importance).fillna(0.0)
    
    return df







