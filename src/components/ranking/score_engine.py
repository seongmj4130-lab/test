# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/components/ranking/score_engine.py
"""
[Stage7] Ranking 엔진: score_total 및 rank_total 생성

기능:
- 피처별 날짜별 cross-sectional 정규화 (percentile 또는 zscore)
- 피처 그룹별 가중치 합산으로 score_total 생성
- in_universe=True 대상으로 rank_total 1~N 생성
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Literal
from pathlib import Path

from src.utils.feature_groups import (
    load_feature_groups,
    get_feature_groups,  # [개선안 14번] feature_groups.yaml 파싱 버그 수정: 그룹->피처 리스트 추출
    calculate_feature_group_balance,
)

def _pick_feature_cols(df: pd.DataFrame) -> List[str]:
    """피처 컬럼 선택 (식별자/타겟 제외)"""
    exclude = {
        "date", "ticker",
        "ret_fwd_20d", "ret_fwd_120d",
        "split", "phase", "segment", "fold_id",
        "in_universe", "ym", "corp_code",
        "open", "high", "low", "close", "volume",  # OHLCV는 피처로 사용하지 않음
    }
    
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            cols.append(c)
    
    if len(cols) == 0:
        raise ValueError("No numeric feature columns found after excluding identifiers/targets.")
    
    return sorted(cols)

def normalize_feature_cross_sectional(
    df: pd.DataFrame,
    feature_col: str,
    date_col: str,
    method: Literal["percentile", "zscore"] = "percentile",
    sector_col: Optional[str] = None,
) -> pd.Series:
    """
    날짜별 cross-sectional 정규화 (sector-relative 지원)
    
    Args:
        df: 입력 DataFrame (date, feature_col 포함, 선택적으로 sector_col)
        feature_col: 정규화할 피처 컬럼명
        date_col: 날짜 컬럼명
        method: "percentile" 또는 "zscore"
        sector_col: 섹터 컬럼명 (None이면 전체 시장 기준, 있으면 섹터별 정규화)
    
    Returns:
        정규화된 값 Series (인덱스는 df와 동일)
    """
    if feature_col not in df.columns:
        raise KeyError(f"Feature column not found: {feature_col}")
    if date_col not in df.columns:
        raise KeyError(f"Date column not found: {date_col}")
    
    # [Stage8] sector-relative 정규화가 가능한지 확인
    use_sector_relative = (
        sector_col is not None
        and sector_col in df.columns
        and df[sector_col].notna().sum() > 0
    )
    
    result = pd.Series(index=df.index, dtype=float)
    
    if use_sector_relative:
        # [Stage8] 섹터별 정규화: 같은 date, 같은 sector 내에서 정규화
        for (date, sector), group in df.groupby([date_col, sector_col], sort=False):
            values = group[feature_col].values
            
            if method == "percentile":
                # Percentile rank (0~1)
                ranks = pd.Series(values).rank(pct=True, method="first")
                normalized = ranks.values
            elif method == "zscore":
                # Z-score 정규화
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > 1e-8:
                    normalized = (values - mean_val) / std_val
                else:
                    normalized = np.zeros_like(values)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            result.loc[group.index] = normalized
    else:
        # 전체 시장 기준 정규화 (기존 로직)
        for date, group in df.groupby(date_col, sort=False):
            values = group[feature_col].values
            
            if method == "percentile":
                # Percentile rank (0~1)
                ranks = pd.Series(values).rank(pct=True, method="first")
                normalized = ranks.values
            elif method == "zscore":
                # Z-score 정규화
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > 1e-8:
                    normalized = (values - mean_val) / std_val
                else:
                    normalized = np.zeros_like(values)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            result.loc[group.index] = normalized
    
    return result

def build_score_total(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    feature_weights: Optional[Dict[str, float]] = None,
    feature_groups_config: Optional[Path] = None,
    normalization_method: Literal["percentile", "zscore"] = "percentile",
    date_col: str = "date",
    sector_col: Optional[str] = None,
    use_sector_relative: bool = True,
    market_regime_df: Optional[pd.DataFrame] = None,  # [국면별 전략] 시장 국면 DataFrame (date, regime)
    regime_weights_config: Optional[Dict[str, Dict[str, float]]] = None,  # [국면별 전략] 국면별 가중치
) -> pd.DataFrame:
    """
    score_total 생성
    
    Args:
        df: 입력 DataFrame (date, ticker, 피처들 포함)
        feature_cols: 사용할 피처 컬럼 리스트 (None이면 자동 선택)
        feature_weights: 피처별 가중치 딕셔너리 (None이면 균등 가중치)
        feature_groups_config: 피처 그룹 설정 파일 경로 (None이면 그룹별 균등 가중치)
        normalization_method: 정규화 방법 ("percentile" 또는 "zscore")
        date_col: 날짜 컬럼명
    
    Returns:
        df에 score_total 컬럼이 추가된 DataFrame
    """
    out = df.copy()
    
    # 피처 컬럼 선택
    if feature_cols is None:
        feature_cols = _pick_feature_cols(df)
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns available for scoring.")
    
    # [Stage8] sector-relative 정규화 사용 여부 결정
    actual_sector_col = None
    if use_sector_relative and sector_col and sector_col in out.columns:
        if out[sector_col].notna().sum() > 0:
            actual_sector_col = sector_col
    
    # 피처별 정규화
    normalized_features = {}
    for feat in feature_cols:
        if feat not in out.columns:
            continue
        normalized_features[feat] = normalize_feature_cross_sectional(
            out, feat, date_col, method=normalization_method, sector_col=actual_sector_col
        )
    
    if len(normalized_features) == 0:
        raise ValueError("No valid features found after normalization.")
    
    # 가중치 결정
    if feature_groups_config is not None and Path(feature_groups_config).exists():
        # [개선안 14번] feature_groups.yaml 스키마는 최상위에 feature_groups/balancing 키가 있음.
        # 기존 구현은 load_feature_groups() 반환(dict)의 최상위 키를 그대로 iterate하여 그룹 가중치가 사실상 무시되는 버그가 있었다.
        cfg_groups = load_feature_groups(feature_groups_config)
        feature_groups = get_feature_groups(cfg_groups)
        group_weights = {}
        
        # 그룹별 균등 가중치 (그룹 합 = 1)
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
        
        # 피처별 가중치 계산 (그룹 내 균등)
        feature_weights = {}
        for feat in feature_cols:
            for group_name, group_features in feature_groups.items():
                gf = set(map(str, group_features))
                if feat in gf:
                    n_features_in_group = len([f for f in feature_cols if f in gf])
                    if n_features_in_group > 0:
                        feature_weights[feat] = group_weights.get(group_name, 0.0) / n_features_in_group
                    break
            if feat not in feature_weights:
                # 그룹에 속하지 않은 피처는 균등 가중치
                feature_weights[feat] = 1.0 / len(feature_cols)
    
    elif feature_weights is not None:
        # 사용자 지정 가중치 사용
        pass
    else:
        # 균등 가중치
        feature_weights = {feat: 1.0 / len(normalized_features) for feat in normalized_features.keys()}
    
    # [국면별 전략] 국면별 가중치 적용
    use_regime_weights = (
        market_regime_df is not None
        and regime_weights_config is not None
        and len(regime_weights_config) > 0
        and "regime" in market_regime_df.columns
    )
    
    if use_regime_weights:
        # 날짜별로 국면에 맞는 가중치 사용
        score_total = pd.Series(0.0, index=out.index)
        
        for date, group in out.groupby(date_col, sort=False):
            # 해당 날짜의 국면 조회
            regime_row = market_regime_df[market_regime_df[date_col] == date]
            
            if len(regime_row) > 0:
                regime = regime_row.iloc[0]["regime"]
                
                # [국면 세분화] 5단계 국면별 가중치 선택
                if regime in regime_weights_config:
                    date_weights = regime_weights_config[regime]
                elif regime == "bull_strong" and "bull" in regime_weights_config:
                    date_weights = regime_weights_config["bull"]
                elif regime == "bull_weak" and "bull" in regime_weights_config:
                    date_weights = regime_weights_config["bull"]
                elif regime == "bear_strong" and "bear" in regime_weights_config:
                    date_weights = regime_weights_config["bear"]
                elif regime == "bear_weak" and "bear" in regime_weights_config:
                    date_weights = regime_weights_config["bear"]
                elif regime == "neutral":
                    # neutral은 bull/bear 평균 또는 기본 가중치
                    if "bull" in regime_weights_config and "bear" in regime_weights_config:
                        # bull과 bear 가중치의 평균
                        bull_weights = regime_weights_config["bull"]
                        bear_weights = regime_weights_config["bear"]
                        common_features = set(bull_weights.keys()) & set(bear_weights.keys())
                        if len(common_features) > 0:
                            date_weights = {f: (bull_weights.get(f, 0) + bear_weights.get(f, 0)) / 2.0 for f in common_features}
                        else:
                            date_weights = feature_weights
                    else:
                        date_weights = feature_weights
                else:
                    date_weights = feature_weights  # 기본 가중치 사용
            else:
                date_weights = feature_weights  # 국면 정보 없으면 기본 가중치
            
            # 가중치 정규화 (합 = 1)
            total_weight = sum(date_weights.get(feat, 0.0) for feat in normalized_features.keys())
            if total_weight > 1e-8:
                date_weights = {feat: w / total_weight for feat, w in date_weights.items() if feat in normalized_features}
            else:
                date_weights = {feat: 1.0 / len(normalized_features) for feat in normalized_features.keys()}
            
            # 해당 날짜의 score_total 계산
            for feat, normalized_values in normalized_features.items():
                weight = date_weights.get(feat, 0.0)
                score_total.loc[group.index] += weight * normalized_values.loc[group.index].fillna(0.0)
    else:
        # 기본 가중치 사용 (기존 로직)
        # 가중치 정규화 (합 = 1)
        total_weight = sum(feature_weights.get(feat, 0.0) for feat in normalized_features.keys())
        if total_weight > 1e-8:
            feature_weights = {feat: w / total_weight for feat, w in feature_weights.items() if feat in normalized_features}
        
        # score_total 계산
        score_total = pd.Series(0.0, index=out.index)
        for feat, normalized_values in normalized_features.items():
            weight = feature_weights.get(feat, 0.0)
            score_total += weight * normalized_values.fillna(0.0)
    
    out["score_total"] = score_total
    
    # [Stage8] sector_col이 있으면 유지 (build_ranking_daily에서 사용)
    # build_score_total은 입력 DataFrame의 모든 컬럼을 유지하므로
    # sector_col이 이미 out에 있으면 그대로 유지됨
    
    return out

def build_rank_total(
    df: pd.DataFrame,
    score_col: str = "score_total",
    date_col: str = "date",
    universe_col: str = "in_universe",
) -> pd.DataFrame:
    """
    rank_total 생성 (in_universe=True 대상으로만)
    
    Args:
        df: 입력 DataFrame (score_total, in_universe 포함)
        score_col: 점수 컬럼명
        date_col: 날짜 컬럼명
        universe_col: 유니버스 필터 컬럼명
    
    Returns:
        df에 rank_total 컬럼이 추가된 DataFrame
    """
    out = df.copy()
    
    if score_col not in out.columns:
        raise KeyError(f"Score column not found: {score_col}")
    if date_col not in out.columns:
        raise KeyError(f"Date column not found: {date_col}")
    
    # in_universe 필터링
    if universe_col in out.columns:
        universe_mask = out[universe_col].fillna(False).astype(bool)
    else:
        universe_mask = pd.Series(True, index=out.index)
    
    # 날짜별 랭킹 (높은 점수 = 낮은 랭크 = 상위)
    rank_total = pd.Series(np.nan, index=out.index, dtype=float)
    
    for date, group in out.groupby(date_col, sort=False):
        group_mask = universe_mask.loc[group.index]
        universe_group = group.loc[group_mask]
        
        if len(universe_group) == 0:
            continue
        
        # 랭킹 계산 (높은 점수 = 낮은 랭크)
        ranks = universe_group[score_col].rank(ascending=False, method="first")
        rank_total.loc[universe_group.index] = ranks.values
    
    out["rank_total"] = rank_total
    
    return out

def build_ranking_daily(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    feature_weights: Optional[Dict[str, float]] = None,
    feature_groups_config: Optional[Path] = None,
    normalization_method: Literal["percentile", "zscore"] = "percentile",
    date_col: str = "date",
    universe_col: str = "in_universe",
    sector_col: Optional[str] = None,
    use_sector_relative: bool = True,
    market_regime_df: Optional[pd.DataFrame] = None,  # [국면별 전략] 시장 국면 DataFrame
    regime_weights_config: Optional[Dict[str, Dict[str, float]]] = None,  # [국면별 전략] 국면별 가중치
) -> pd.DataFrame:
    """
    ranking_daily 생성 (score_total + rank_total)
    
    Args:
        df: 입력 DataFrame (date, ticker, 피처들, in_universe 포함)
        feature_cols: 사용할 피처 컬럼 리스트
        feature_weights: 피처별 가중치 딕셔너리
        feature_groups_config: 피처 그룹 설정 파일 경로
        normalization_method: 정규화 방법
        date_col: 날짜 컬럼명
        universe_col: 유니버스 필터 컬럼명
    
    Returns:
        ranking_daily DataFrame (date, ticker, score_total, rank_total 포함)
    """
    # score_total 생성
    df_scored = build_score_total(
        df,
        feature_cols=feature_cols,
        feature_weights=feature_weights,
        feature_groups_config=feature_groups_config,
        normalization_method=normalization_method,
        date_col=date_col,
        sector_col=sector_col,
        use_sector_relative=use_sector_relative,
        market_regime_df=market_regime_df,  # [국면별 전략]
        regime_weights_config=regime_weights_config,  # [국면별 전략]
    )
    
    # rank_total 생성
    df_ranked = build_rank_total(
        df_scored,
        score_col="score_total",
        date_col=date_col,
        universe_col=universe_col,
    )
    
    # 최종 컬럼 선택
    output_cols = [date_col, "ticker", "score_total", "rank_total"]
    if universe_col in df_ranked.columns:
        output_cols.append(universe_col)
    
    # [Stage8] sector_col 포함 (있는 경우)
    if sector_col and sector_col in df_ranked.columns:
        output_cols.append(sector_col)
    
    result = df_ranked[output_cols].copy()
    
    return result
