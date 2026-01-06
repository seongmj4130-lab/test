# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tracks/shared/stages/data/l3_feature_engineering.py
"""
[L3] 현재 데이터 기반 팩터 엔지니어링
- Momentum: 가격 모멘텀 (20일, 60일 수익률)
- Volatility: 변동성 (20일, 60일 수익률 표준편차)
- Liquidity: 유동성 (volume_ratio, turnover)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_based_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    *,
    momentum_windows: list[int] = [20, 60],
    volatility_windows: list[int] = [20, 60],
    volume_ratio_window: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    """
    현재 데이터(가격/거래량)로 계산 가능한 팩터 추가
    
    Args:
        df: panel_merged_daily (date, ticker, close, volume 등 포함)
        price_col: 가격 컬럼명 (기본: "close")
        volume_col: 거래량 컬럼명 (기본: "volume")
        momentum_windows: 모멘텀 계산 기간 리스트 (기본: [20, 60])
        volatility_windows: 변동성 계산 기간 리스트 (기본: [20, 60])
        volume_ratio_window: 거래량 비율 계산 기간 (기본: 60)
    
    Returns:
        (df_with_features, warnings) 튜플
    """
    warns: list[str] = []
    out = df.copy()
    
    # 필수 컬럼 확인
    if price_col not in out.columns:
        warns.append(f"[L3 Features] {price_col} 컬럼이 없어 가격 기반 팩터를 계산하지 않습니다.")
        return out, warns
    
    if volume_col not in out.columns:
        warns.append(f"[L3 Features] {volume_col} 컬럼이 없어 거래량 기반 팩터를 계산하지 않습니다.")
        # 가격 기반만 계산
        volume_col = None
    
    # ticker-date 정렬 (shift 안정성)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # 가격을 숫자로 변환
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    
    # -------------------------
    # 1) Momentum (가격 모멘텀)
    # -------------------------
    for window in momentum_windows:
        col_name = f"price_momentum_{window}d"
        # window일 전 가격 대비 현재 수익률
        out[col_name] = out.groupby("ticker", group_keys=False)[price_col].apply(
            lambda x: (x / x.shift(window) - 1.0) * 100.0
        )
        warns.append(f"[L3 Features] {col_name} 생성 완료")
    
    # 기본 momentum (20일)을 price_momentum으로도 저장 (feature_groups.yaml 호환)
    if f"price_momentum_{momentum_windows[0]}d" in out.columns:
        out["price_momentum"] = out[f"price_momentum_{momentum_windows[0]}d"]
    
    # -------------------------
    # 2) Volatility (변동성)
    # -------------------------
    # 일일 수익률 계산
    out["ret_daily"] = out.groupby("ticker", group_keys=False)[price_col].apply(
        lambda x: x.pct_change()
    )
    
    for window in volatility_windows:
        col_name = f"volatility_{window}d"
        # window일 수익률 표준편차 (연율화: sqrt(252))
        out[col_name] = out.groupby("ticker", group_keys=False)["ret_daily"].apply(
            lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).std() * np.sqrt(252) * 100.0
        )
        warns.append(f"[L3 Features] {col_name} 생성 완료")
    
    # 기본 volatility (20일)을 volatility로도 저장 (feature_groups.yaml 호환)
    if f"volatility_{volatility_windows[0]}d" in out.columns:
        out["volatility"] = out[f"volatility_{volatility_windows[0]}d"]
    
    # -------------------------
    # 3) Liquidity (유동성)
    # -------------------------
    if volume_col is not None:
        out[volume_col] = pd.to_numeric(out[volume_col], errors="coerce")
        
        # volume_ratio: 현재 거래량 / 평균 거래량 (과거 N일)
        volume_ratio_col = "volume_ratio"
        out[volume_ratio_col] = out.groupby("ticker", group_keys=False)[volume_col].apply(
            lambda x: x / x.rolling(window=volume_ratio_window, min_periods=max(1, volume_ratio_window // 2)).mean()
        )
        warns.append(f"[L3 Features] {volume_ratio_col} 생성 완료")
        
        # turnover: 거래량 * 가격 (간단 버전, 시가총액 대신 가격 사용)
        # 실제로는 시가총액이 필요하지만, 현재 데이터로는 가격 * 거래량으로 근사
        turnover_col = "turnover"
        out[turnover_col] = out[volume_col] * out[price_col]
        warns.append(f"[L3 Features] {turnover_col} 생성 완료 (가격*거래량 근사)")
    
    # -------------------------
    # 4) [팩터 정교화] 모멘텀 반전 신호
    # -------------------------
    # 모멘텀이 과열 구간(상위 10%)이면 반전 신호
    if f"price_momentum_{momentum_windows[0]}d" in out.columns:
        momentum_col = f"price_momentum_{momentum_windows[0]}d"
        
        # 일별 cross-sectional rank 계산
        out["momentum_rank"] = out.groupby("date", group_keys=False)[momentum_col].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )
        
        # 반전 신호: 상위 10%는 음수, 나머지는 원래 값
        out["momentum_reversal"] = out[momentum_col].copy()
        overheat_mask = out["momentum_rank"] > 0.9
        out.loc[overheat_mask, "momentum_reversal"] = -out.loc[overheat_mask, momentum_col]
        
        warns.append(f"[L3 Features] momentum_reversal 생성 완료 (과열 구간 반전)")
    
    # -------------------------
    # 5) [팩터 정교화] 리스크 팩터
    # -------------------------
    # Max Drawdown (60일)
    if "ret_daily" in out.columns:
        # 누적 수익률 계산
        out["cumret"] = out.groupby("ticker", group_keys=False)["ret_daily"].apply(
            lambda x: (1 + x).cumprod()
        )
        
        # Rolling Max 계산
        out["rolling_max"] = out.groupby("ticker", group_keys=False)["cumret"].apply(
            lambda x: x.rolling(window=60, min_periods=20).max()
        )
        
        # Drawdown 계산: (현재 - 최고점) / 최고점
        out["drawdown"] = (out["cumret"] - out["rolling_max"]) / out["rolling_max"] * 100.0
        
        # Max Drawdown (60일)
        out["max_drawdown_60d"] = out.groupby("ticker", group_keys=False)["drawdown"].apply(
            lambda x: x.rolling(window=60, min_periods=20).min()
        )
        warns.append(f"[L3 Features] max_drawdown_60d 생성 완료")
        
        # Downside Volatility (하방 변동성)
        # 음수 수익률만 고려한 변동성
        out["ret_negative"] = out["ret_daily"].copy()
        out.loc[out["ret_negative"] > 0, "ret_negative"] = 0.0
        
        out["downside_volatility_60d"] = out.groupby("ticker", group_keys=False)["ret_negative"].apply(
            lambda x: x.rolling(window=60, min_periods=20).std() * np.sqrt(252) * 100.0
        )
        warns.append(f"[L3 Features] downside_volatility_60d 생성 완료")
        
        # 중간 계산 컬럼 제거
        out = out.drop(columns=["cumret", "rolling_max", "drawdown", "ret_negative"], errors="ignore")
    
    # -------------------------
    # 6) 정리: ret_daily는 중간 계산용이므로 제거 (필요시 주석 해제)
    # -------------------------
    # out = out.drop(columns=["ret_daily"], errors="ignore")
    
    # ticker-date 정렬 유지
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    return out, warns

