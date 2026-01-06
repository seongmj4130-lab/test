# -*- coding: utf-8 -*-
# [Dual Horizon 전략] L8_short/L8_long 분리 실행
"""
[L8_short/L8_long] 단기/장기 랭킹 분리 생성

L8_short: 모멘텀 중심 (20영업일 내 상대강도/추세/리버설 포착)
L8_long: 가치 중심 (120영업일 관점의 저평가/퀄리티+리스크 안정)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path

# ranking 모듈 import를 위한 경로 추가
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.components.ranking.score_engine import build_ranking_daily, _pick_feature_cols
from src.components.ranking.regime_strategy import load_regime_weights

def run_L8_short_rank_engine(
    cfg: dict,
    artifacts: dict,
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [L8_short] 단기 모멘텀 중심 랭킹 생성
    
    Returns:
        (outputs, warnings) 튜플
        - outputs: {"ranking_short_daily": DataFrame}
        - warnings: 경고 메시지 리스트
    """
    warns: list[str] = []
    
    # 설정 읽기
    l8_short = cfg.get("l8_short", {}) or {}
    normalization_method = l8_short.get("normalization_method", "percentile")
    feature_groups_config = l8_short.get("feature_groups_config", "configs/feature_groups_short.yaml")
    feature_weights_config = l8_short.get("feature_weights_config", "configs/feature_weights_short.yaml")
    use_sector_relative = l8_short.get("use_sector_relative", True)
    sector_col = l8_short.get("sector_col", "sector_name")
    
    # 입력 데이터 확인
    dataset_daily = artifacts.get("dataset_daily")
    panel_merged_daily = artifacts.get("panel_merged_daily")
    
    if dataset_daily is not None and len(dataset_daily) > 0:
        input_df = dataset_daily.copy()
        input_source = "dataset_daily"
        
        if sector_col not in input_df.columns and panel_merged_daily is not None:
            if sector_col in panel_merged_daily.columns:
                sector_info = panel_merged_daily[["date", "ticker", sector_col]].drop_duplicates(["date", "ticker"])
                input_df = input_df.merge(sector_info, on=["date", "ticker"], how="left")
                warns.append(f"[L8_short] {sector_col}을 panel_merged_daily에서 병합")
    elif panel_merged_daily is not None and len(panel_merged_daily) > 0:
        input_df = panel_merged_daily.copy()
        input_source = "panel_merged_daily"
    else:
        raise ValueError(
            "L8_short requires 'dataset_daily' or 'panel_merged_daily' in artifacts. "
            f"Available keys: {list(artifacts.keys())}"
        )
    
    # 필수 컬럼 확인
    required_cols = ["date", "ticker"]
    missing_cols = [c for c in required_cols if c not in input_df.columns]
    if missing_cols:
        raise KeyError(f"Input DataFrame missing required columns: {missing_cols}")
    
    # 날짜/티커 정규화
    input_df["date"] = pd.to_datetime(input_df["date"], errors="raise")
    input_df["ticker"] = input_df["ticker"].astype(str).str.zfill(6)
    
    # in_universe 확인
    if "in_universe" not in input_df.columns:
        warns.append("[L8_short] 'in_universe' column not found. All rows will be treated as in_universe=True.")
        input_df["in_universe"] = True
    else:
        input_df["in_universe"] = input_df["in_universe"].fillna(False).astype(bool)
    
    # 피처 가중치 파일 로드
    base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
    feature_weights = None
    if feature_weights_config:
        weights_path = base_dir / feature_weights_config
        if weights_path.exists():
            try:
                import yaml
                with open(weights_path, 'r', encoding='utf-8') as f:
                    weights_data = yaml.safe_load(f)
                feature_weights = weights_data.get("feature_weights", {})
                warns.append(f"[L8_short] 피처 가중치 로드 완료: {len(feature_weights)}개 피처")
            except Exception as e:
                warns.append(f"[L8_short] 피처 가중치 로드 실패: {e}. feature_groups 사용.")
    
    # feature_groups 설정 파일 경로
    feature_groups_path = None
    if feature_groups_config:
        feature_groups_path = base_dir / feature_groups_config
        if not feature_groups_path.exists():
            warns.append(f"[L8_short] Feature groups config not found: {feature_groups_path}. Using equal weights.")
            feature_groups_path = None
    
    # sector_name 확인
    actual_sector_col = None
    if use_sector_relative and sector_col in input_df.columns:
        if input_df[sector_col].notna().sum() > 0:
            actual_sector_col = sector_col
            warns.append(f"[L8_short] sector-relative 정규화 사용: {sector_col}")
        else:
            warns.append(f"[L8_short] {sector_col} 컬럼이 모두 NaN이어서 전체 시장 기준 정규화 사용")
    
    # ranking_short_daily 생성
    try:
        use_feature_groups = feature_groups_path if feature_weights is None else None
        
        ranking_short_daily = build_ranking_daily(
            input_df,
            feature_cols=None,  # 자동 선택
            feature_weights=feature_weights,  # 명시적 가중치 우선
            feature_groups_config=use_feature_groups,  # 가중치가 없을 때만 사용
            normalization_method=normalization_method,
            date_col="date",
            universe_col="in_universe",
            sector_col=actual_sector_col,
            use_sector_relative=use_sector_relative,
            market_regime_df=None,  # 단기 랭킹은 국면별 가중치 미사용
            regime_weights_config=None,
        )
        
        # sector_name 추가
        if actual_sector_col and actual_sector_col in input_df.columns:
            sector_info = input_df[["date", "ticker", actual_sector_col]].drop_duplicates(["date", "ticker"])
            ranking_short_daily = ranking_short_daily.merge(sector_info, on=["date", "ticker"], how="left")
            warns.append(f"[L8_short] {actual_sector_col}을 ranking_short_daily에 병합 완료")
    except Exception as e:
        raise RuntimeError(f"Failed to build ranking_short_daily: {e}") from e
    
    # 검증
    if len(ranking_short_daily) == 0:
        raise ValueError("ranking_short_daily is empty after processing.")
    
    # 최종 컬럼 정리
    output_cols = ["date", "ticker", "score_total", "rank_total"]
    if "in_universe" in ranking_short_daily.columns:
        output_cols.append("in_universe")
    if actual_sector_col and actual_sector_col in ranking_short_daily.columns:
        output_cols.append(actual_sector_col)
    
    ranking_short_daily_final = ranking_short_daily[output_cols].copy()
    
    # 중복 키 확인
    dup = ranking_short_daily_final.duplicated(subset=["date", "ticker"]).sum()
    if dup > 0:
        raise ValueError(f"ranking_short_daily has duplicate (date, ticker) keys: {dup}")
    
    warns.append(f"[L8_short] 완료: {len(ranking_short_daily_final):,}행, {ranking_short_daily_final['date'].nunique()}개 날짜")
    
    return {
        "ranking_short_daily": ranking_short_daily_final,
    }, warns


def run_L8_long_rank_engine(
    cfg: dict,
    artifacts: dict,
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [L8_long] 장기 가치 중심 랭킹 생성
    
    Returns:
        (outputs, warnings) 튜플
        - outputs: {"ranking_long_daily": DataFrame}
        - warnings: 경고 메시지 리스트
    """
    warns: list[str] = []
    
    # 설정 읽기
    l8_long = cfg.get("l8_long", {}) or {}
    normalization_method = l8_long.get("normalization_method", "percentile")
    feature_groups_config = l8_long.get("feature_groups_config", "configs/feature_groups_long.yaml")
    feature_weights_config = l8_long.get("feature_weights_config", "configs/feature_weights_long.yaml")
    use_sector_relative = l8_long.get("use_sector_relative", True)
    sector_col = l8_long.get("sector_col", "sector_name")
    
    # 입력 데이터 확인
    dataset_daily = artifacts.get("dataset_daily")
    panel_merged_daily = artifacts.get("panel_merged_daily")
    
    if dataset_daily is not None and len(dataset_daily) > 0:
        input_df = dataset_daily.copy()
        input_source = "dataset_daily"
        
        if sector_col not in input_df.columns and panel_merged_daily is not None:
            if sector_col in panel_merged_daily.columns:
                sector_info = panel_merged_daily[["date", "ticker", sector_col]].drop_duplicates(["date", "ticker"])
                input_df = input_df.merge(sector_info, on=["date", "ticker"], how="left")
                warns.append(f"[L8_long] {sector_col}을 panel_merged_daily에서 병합")
    elif panel_merged_daily is not None and len(panel_merged_daily) > 0:
        input_df = panel_merged_daily.copy()
        input_source = "panel_merged_daily"
    else:
        raise ValueError(
            "L8_long requires 'dataset_daily' or 'panel_merged_daily' in artifacts. "
            f"Available keys: {list(artifacts.keys())}"
        )
    
    # 필수 컬럼 확인
    required_cols = ["date", "ticker"]
    missing_cols = [c for c in required_cols if c not in input_df.columns]
    if missing_cols:
        raise KeyError(f"Input DataFrame missing required columns: {missing_cols}")
    
    # 날짜/티커 정규화
    input_df["date"] = pd.to_datetime(input_df["date"], errors="raise")
    input_df["ticker"] = input_df["ticker"].astype(str).str.zfill(6)
    
    # in_universe 확인
    if "in_universe" not in input_df.columns:
        warns.append("[L8_long] 'in_universe' column not found. All rows will be treated as in_universe=True.")
        input_df["in_universe"] = True
    else:
        input_df["in_universe"] = input_df["in_universe"].fillna(False).astype(bool)
    
    # 피처 가중치 파일 로드
    base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
    feature_weights = None
    if feature_weights_config:
        weights_path = base_dir / feature_weights_config
        if weights_path.exists():
            try:
                import yaml
                with open(weights_path, 'r', encoding='utf-8') as f:
                    weights_data = yaml.safe_load(f)
                feature_weights = weights_data.get("feature_weights", {})
                warns.append(f"[L8_long] 피처 가중치 로드 완료: {len(feature_weights)}개 피처")
            except Exception as e:
                warns.append(f"[L8_long] 피처 가중치 로드 실패: {e}. feature_groups 사용.")
    
    # feature_groups 설정 파일 경로
    feature_groups_path = None
    if feature_groups_config:
        feature_groups_path = base_dir / feature_groups_config
        if not feature_groups_path.exists():
            warns.append(f"[L8_long] Feature groups config not found: {feature_groups_path}. Using equal weights.")
            feature_groups_path = None
    
    # sector_name 확인
    actual_sector_col = None
    if use_sector_relative and sector_col in input_df.columns:
        if input_df[sector_col].notna().sum() > 0:
            actual_sector_col = sector_col
            warns.append(f"[L8_long] sector-relative 정규화 사용: {sector_col}")
        else:
            warns.append(f"[L8_long] {sector_col} 컬럼이 모두 NaN이어서 전체 시장 기준 정규화 사용")
    
    # ranking_long_daily 생성
    try:
        use_feature_groups = feature_groups_path if feature_weights is None else None
        
        ranking_long_daily = build_ranking_daily(
            input_df,
            feature_cols=None,  # 자동 선택
            feature_weights=feature_weights,  # 명시적 가중치 우선
            feature_groups_config=use_feature_groups,  # 가중치가 없을 때만 사용
            normalization_method=normalization_method,
            date_col="date",
            universe_col="in_universe",
            sector_col=actual_sector_col,
            use_sector_relative=use_sector_relative,
            market_regime_df=None,  # 장기 랭킹은 국면별 가중치 미사용
            regime_weights_config=None,
        )
        
        # sector_name 추가
        if actual_sector_col and actual_sector_col in input_df.columns:
            sector_info = input_df[["date", "ticker", actual_sector_col]].drop_duplicates(["date", "ticker"])
            ranking_long_daily = ranking_long_daily.merge(sector_info, on=["date", "ticker"], how="left")
            warns.append(f"[L8_long] {actual_sector_col}을 ranking_long_daily에 병합 완료")
    except Exception as e:
        raise RuntimeError(f"Failed to build ranking_long_daily: {e}") from e
    
    # 검증
    if len(ranking_long_daily) == 0:
        raise ValueError("ranking_long_daily is empty after processing.")
    
    # 최종 컬럼 정리
    output_cols = ["date", "ticker", "score_total", "rank_total"]
    if "in_universe" in ranking_long_daily.columns:
        output_cols.append("in_universe")
    if actual_sector_col and actual_sector_col in ranking_long_daily.columns:
        output_cols.append(actual_sector_col)
    
    ranking_long_daily_final = ranking_long_daily[output_cols].copy()
    
    # 중복 키 확인
    dup = ranking_long_daily_final.duplicated(subset=["date", "ticker"]).sum()
    if dup > 0:
        raise ValueError(f"ranking_long_daily has duplicate (date, ticker) keys: {dup}")
    
    warns.append(f"[L8_long] 완료: {len(ranking_long_daily_final):,}행, {ranking_long_daily_final['date'].nunique()}개 날짜")
    
    return {
        "ranking_long_daily": ranking_long_daily_final,
    }, warns








