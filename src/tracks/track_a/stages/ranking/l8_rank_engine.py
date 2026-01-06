# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/ranking/l8_rank_engine.py
"""
[Stage7] Ranking 엔진 실행 스테이지

입력:
- dataset_daily 또는 panel_merged_daily
- fundamentals_annual (루트 재사용)

출력:
- ranking_daily (date, ticker, score_total, rank_total, in_universe)
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

def run_L8_rank_engine(
    cfg: dict,
    artifacts: dict,
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [Stage7] Ranking 엔진 실행
    
    Args:
        cfg: 설정 딕셔너리
        artifacts: 이전 스테이지 산출물 딕셔너리
        force: 강제 재생성 플래그
    
    Returns:
        (outputs, warnings) 튜플
        - outputs: {"ranking_daily": DataFrame}
        - warnings: 경고 메시지 리스트
    """
    warns: list[str] = []
    
    # 설정 읽기
    l8 = cfg.get("l8", {}) or cfg.get("params", {}).get("l8", {})
    normalization_method = l8.get("normalization_method", "percentile")  # "percentile" | "zscore"
    feature_groups_config = l8.get("feature_groups_config", None)
    feature_weights_config = l8.get("feature_weights_config", None)  # [IC 최적화] 최적 가중치 파일 경로
    use_sector_relative = l8.get("use_sector_relative", True)  # [Stage8] sector-relative 정규화 사용 여부
    sector_col = l8.get("sector_col", "sector_name")  # [Stage8] 섹터 컬럼명
    enforce_group_balance = l8.get("enforce_group_balance", True)  # [Stage8] 그룹별 밸런스 강제 여부
    
    # 입력 데이터 확인
    dataset_daily = artifacts.get("dataset_daily")
    panel_merged_daily = artifacts.get("panel_merged_daily")
    
    # [Stage8] panel_merged_daily가 없으면 base_interim_dir에서 로드 시도
    if panel_merged_daily is None:
        from src.utils.config import get_path
        from src.utils.io import load_artifact, artifact_exists
        base_interim_dir = Path(get_path(cfg, "data_interim"))
        panel_merged_path = base_interim_dir / "panel_merged_daily.parquet"
        if artifact_exists(panel_merged_path):
            panel_merged_daily = load_artifact(panel_merged_path)
            warns.append(f"[Stage8] panel_merged_daily를 base_interim_dir에서 로드: {panel_merged_path}")
    
    if dataset_daily is not None and len(dataset_daily) > 0:
        input_df = dataset_daily.copy()
        input_source = "dataset_daily"
        
        # [Stage8] dataset_daily에 sector_name이 없으면 panel_merged_daily에서 가져오기
        if sector_col not in input_df.columns and panel_merged_daily is not None:
            if sector_col in panel_merged_daily.columns:
                sector_info = panel_merged_daily[["date", "ticker", sector_col]].drop_duplicates(["date", "ticker"])
                input_df = input_df.merge(sector_info, on=["date", "ticker"], how="left")
                warns.append(f"[Stage8] {sector_col}을 panel_merged_daily에서 병합")
    elif panel_merged_daily is not None and len(panel_merged_daily) > 0:
        input_df = panel_merged_daily.copy()
        input_source = "panel_merged_daily"
    else:
        raise ValueError(
            "L8 requires 'dataset_daily' or 'panel_merged_daily' in artifacts. "
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
        warns.append("[L8] 'in_universe' column not found. All rows will be treated as in_universe=True.")
        input_df["in_universe"] = True
    else:
        input_df["in_universe"] = input_df["in_universe"].fillna(False).astype(bool)
    
    # 피처 그룹 설정 파일 경로
    feature_groups_path = None
    if feature_groups_config:
        base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
        feature_groups_path = base_dir / feature_groups_config
        if not feature_groups_path.exists():
            warns.append(f"[L8] Feature groups config not found: {feature_groups_path}. Using equal weights.")
            feature_groups_path = None
    
    # [IC 최적화] 최적 가중치 파일 로드
    optimal_feature_weights = None
    regime_weights_config = None  # [국면별 전략] 국면별 가중치
    market_regime_df = None  # [국면별 전략] 시장 국면 DataFrame
    
    # [국면별 전략] 국면별 가중치 설정 확인
    regime_aware_config = l8.get("regime_aware_weights_config", None)
    regime_enabled = cfg.get("l7", {}).get("regime", {}).get("enabled", False)
    
    if regime_aware_config and regime_enabled:
        base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
        regime_weights_config = load_regime_weights(
            config_path=regime_aware_config,
            base_dir=base_dir,
        )
        if len(regime_weights_config) > 0:
            warns.append(f"[L8 국면별 전략] 국면별 가중치 로드 완료: {list(regime_weights_config.keys())}")
            
            # 시장 국면 데이터 생성 (rebalance_scores에서 가져오거나 새로 계산)
            from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime
            date_col_actual = "date"  # build_market_regime은 "date" 컬럼을 사용
            dates = input_df[date_col_actual].unique()
            start_date = str(input_df[date_col_actual].min().date())
            end_date = str(input_df[date_col_actual].max().date())
            
            try:
                market_regime_df = build_market_regime(
                    rebalance_dates=dates,
                    start_date=start_date,
                    end_date=end_date,
                    lookback_days=cfg.get("l7", {}).get("regime", {}).get("lookback_days", 60),
                    threshold_pct=cfg.get("l7", {}).get("regime", {}).get("threshold_pct", 0.0),
                )
                warns.append(f"[L8 국면별 전략] 시장 국면 데이터 생성 완료: {len(market_regime_df)}개 날짜")
                # [국면 세분화] 5단계 국면 분포 출력
                regime_counts = market_regime_df["regime"].value_counts()
                warns.append(f"[L8 국면별 전략] 국면 분포: {dict(regime_counts)}")
            except Exception as e:
                warns.append(f"[L8 국면별 전략] 시장 국면 데이터 생성 실패: {e}. 국면별 가중치 비활성화.")
                regime_weights_config = None
                market_regime_df = None
    
    # [IC 최적화] 기본 최적 가중치 파일 로드 (국면별 가중치가 없을 때만)
    if optimal_feature_weights is None and regime_weights_config is None:
        if feature_weights_config:
            base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
            weights_path = base_dir / feature_weights_config
            if weights_path.exists():
                try:
                    import yaml
                    with open(weights_path, 'r', encoding='utf-8') as f:
                        weights_data = yaml.safe_load(f)
                    optimal_feature_weights = weights_data.get("feature_weights", {})
                    warns.append(f"[L8 IC 최적화] 최적 가중치 로드 완료: {len(optimal_feature_weights)}개 피처 ({weights_path})")
                except Exception as e:
                    warns.append(f"[L8 IC 최적화] 최적 가중치 로드 실패: {e}. feature_groups 사용.")
            else:
                warns.append(f"[L8 IC 최적화] 최적 가중치 파일이 없습니다: {weights_path}. feature_groups 사용.")
    
    # [Stage8] sector_name 확인
    actual_sector_col = None
    if use_sector_relative and sector_col in input_df.columns:
        if input_df[sector_col].notna().sum() > 0:
            actual_sector_col = sector_col
            warns.append(f"[Stage8] sector-relative 정규화 사용: {sector_col}")
        else:
            warns.append(f"[Stage8] {sector_col} 컬럼이 모두 NaN이어서 전체 시장 기준 정규화 사용")
    else:
        if use_sector_relative:
            warns.append(f"[Stage8] {sector_col} 컬럼이 없어서 전체 시장 기준 정규화 사용")
    
    # [Stage8] 그룹별 피처 밸런싱 강제
    if enforce_group_balance and feature_groups_path and feature_groups_path.exists():
        from src.utils.feature_groups import load_feature_groups, map_features_to_groups, get_group_target_weights
        
        feature_cols_auto = _pick_feature_cols(input_df)
        config = load_feature_groups(feature_groups_path)
        groups_map = map_features_to_groups(feature_cols_auto, config)
        target_weights = get_group_target_weights(config)
        
        # 그룹별 피처 개수 확인 및 경고
        for group_name, group_features in groups_map.items():
            n_features = len(group_features)
            target_weight = target_weights.get(group_name, 0.0)
            warns.append(f"[Stage8] 그룹 '{group_name}': {n_features}개 피처, 목표 가중치 {target_weight:.2%}")
        
        # 그룹에 속하지 않은 피처 확인
        grouped_features = set()
        for features in groups_map.values():
            grouped_features.update(features)
        ungrouped = [f for f in feature_cols_auto if f not in grouped_features]
        if ungrouped:
            warns.append(f"[Stage8] 그룹에 속하지 않은 피처 {len(ungrouped)}개: {', '.join(ungrouped[:5])}...")
    
    # ranking_daily 생성
    try:
        # [IC 최적화] 최적 가중치가 있으면 우선 사용, 없으면 feature_groups 사용
        use_feature_groups = feature_groups_path if optimal_feature_weights is None else None
        
        ranking_daily = build_ranking_daily(
            input_df,
            feature_cols=None,  # 자동 선택
            feature_weights=optimal_feature_weights,  # [IC 최적화] 최적 가중치 또는 None
            feature_groups_config=use_feature_groups,  # 최적 가중치가 없을 때만 사용
            normalization_method=normalization_method,
            date_col="date",
            universe_col="in_universe",
            sector_col=actual_sector_col,  # [Stage8] sector-relative 정규화
            use_sector_relative=use_sector_relative,  # [Stage8]
            market_regime_df=market_regime_df,  # [국면별 전략]
            regime_weights_config=regime_weights_config,  # [국면별 전략]
        )
        
        # [Stage8] ranking_daily에 sector_name 추가 (input_df에 있으면)
        if actual_sector_col:
            if actual_sector_col in ranking_daily.columns:
                # 이미 ranking_daily에 포함되어 있음
                warns.append(f"[Stage8] {actual_sector_col}이 이미 ranking_daily에 포함됨: {ranking_daily[actual_sector_col].notna().sum():,} / {len(ranking_daily):,} 행")
            elif actual_sector_col in input_df.columns:
                # ranking_daily와 input_df를 merge하여 sector_name 추가
                sector_info = input_df[["date", "ticker", actual_sector_col]].drop_duplicates(["date", "ticker"])
                ranking_daily = ranking_daily.merge(sector_info, on=["date", "ticker"], how="left")
                warns.append(f"[Stage8] {actual_sector_col}을 ranking_daily에 병합 완료: {ranking_daily[actual_sector_col].notna().sum():,} / {len(ranking_daily):,} 행")
            else:
                warns.append(f"[Stage8] {actual_sector_col}이 input_df에 없어 ranking_daily에 포함하지 않음")
    except Exception as e:
        raise RuntimeError(f"Failed to build ranking_daily: {e}") from e
    
    # 검증
    if len(ranking_daily) == 0:
        raise ValueError("ranking_daily is empty after processing.")
    
    # 날짜별 coverage 확인
    coverage_by_date = (
        ranking_daily.groupby("date", sort=False)
        .agg({
            "in_universe": "sum",
            "rank_total": lambda x: x.notna().sum(),
        })
        .rename(columns={"in_universe": "n_universe", "rank_total": "n_ranked"})
    )
    coverage_by_date["coverage_pct"] = (
        coverage_by_date["n_ranked"] / coverage_by_date["n_universe"].replace(0, np.nan)
    ).fillna(0.0)
    
    coverage_mean = coverage_by_date["coverage_pct"].mean()
    if coverage_mean < 0.95:
        warns.append(
            f"[L8] Low universe coverage: mean={coverage_mean:.1%} (expected >= 95%). "
            f"Some dates may have missing rankings."
        )
    
    # 날짜별 랭킹 연속성 확인
    for date, group in ranking_daily.groupby("date", sort=False):
        universe_group = group[group["in_universe"]].copy()
        if len(universe_group) == 0:
            continue
        
        ranks = universe_group["rank_total"].dropna().sort_values()
        if len(ranks) == 0:
            continue
        
        expected_ranks = pd.Series(range(1, len(ranks) + 1), dtype=float)
        if not ranks.equals(expected_ranks):
            # 중복 또는 불연속 확인
            if ranks.duplicated().any():
                warns.append(f"[L8] Duplicate ranks found on {date}. Ranks: {ranks.tolist()[:10]}...")
            if ranks.min() != 1 or ranks.max() != len(ranks):
                warns.append(
                    f"[L8] Non-contiguous ranks on {date}: min={ranks.min()}, max={ranks.max()}, "
                    f"expected 1..{len(ranks)}"
                )
    
    # 최종 컬럼 정리
    output_cols = ["date", "ticker", "score_total", "rank_total"]
    if "in_universe" in ranking_daily.columns:
        output_cols.append("in_universe")
    
    # [Stage8] sector_name 포함 (있는 경우)
    if actual_sector_col and actual_sector_col in ranking_daily.columns:
        output_cols.append(actual_sector_col)
        warns.append(f"[Stage8] {actual_sector_col} 컬럼을 ranking_daily에 포함")
    
    ranking_daily_final = ranking_daily[output_cols].copy()
    
    # 중복 키 확인
    dup = ranking_daily_final.duplicated(subset=["date", "ticker"]).sum()
    if dup > 0:
        raise ValueError(f"ranking_daily has duplicate (date, ticker) keys: {dup}")
    
    # ranking_snapshot 생성 (Top20/Bottom20)
    snapshot_data = []
    for date, group in ranking_daily_final.groupby("date", sort=False):
        universe_group = group[group["in_universe"]].copy() if "in_universe" in group.columns else group.copy()
        universe_group = universe_group[universe_group["rank_total"].notna()].copy()
        
        if len(universe_group) == 0:
            continue
        
        # Top 20
        top20 = universe_group.nsmallest(20, "rank_total")[["ticker", "score_total", "rank_total"]].copy()
        top20["snapshot_type"] = "top20"
        top20["date"] = date
        
        # Bottom 20
        bottom20 = universe_group.nlargest(20, "rank_total")[["ticker", "score_total", "rank_total"]].copy()
        bottom20["snapshot_type"] = "bottom20"
        bottom20["date"] = date
        
        snapshot_data.append(top20)
        snapshot_data.append(bottom20)
    
    if snapshot_data:
        ranking_snapshot = pd.concat(snapshot_data, ignore_index=True)
        ranking_snapshot = ranking_snapshot[["date", "snapshot_type", "ticker", "score_total", "rank_total"]].copy()
    else:
        ranking_snapshot = pd.DataFrame(columns=["date", "snapshot_type", "ticker", "score_total", "rank_total"])
    
    return {
        "ranking_daily": ranking_daily_final,
        "ranking_snapshot": ranking_snapshot,
    }, warns
