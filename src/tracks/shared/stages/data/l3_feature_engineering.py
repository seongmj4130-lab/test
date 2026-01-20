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
        warns.append(
            f"[L3 Features] {price_col} 컬럼이 없어 가격 기반 팩터를 계산하지 않습니다."
        )
        return out, warns

    if volume_col not in out.columns:
        warns.append(
            f"[L3 Features] {volume_col} 컬럼이 없어 거래량 기반 팩터를 계산하지 않습니다."
        )
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
            lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).std()
            * np.sqrt(252)
            * 100.0
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
        out[volume_ratio_col] = out.groupby("ticker", group_keys=False)[
            volume_col
        ].apply(
            lambda x: x
            / x.rolling(
                window=volume_ratio_window, min_periods=max(1, volume_ratio_window // 2)
            ).mean()
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
        out["momentum_rank"] = out.groupby("date", group_keys=False)[
            momentum_col
        ].transform(lambda x: x.rank(pct=True, na_option="keep"))

        # 반전 신호: 상위 10%는 음수, 나머지는 원래 값
        out["momentum_reversal"] = out[momentum_col].copy()
        overheat_mask = out["momentum_rank"] > 0.9
        out.loc[overheat_mask, "momentum_reversal"] = -out.loc[
            overheat_mask, momentum_col
        ]

        warns.append("[L3 Features] momentum_reversal 생성 완료 (과열 구간 반전)")

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
        out["drawdown"] = (
            (out["cumret"] - out["rolling_max"]) / out["rolling_max"] * 100.0
        )

        # Max Drawdown (60일)
        out["max_drawdown_60d"] = out.groupby("ticker", group_keys=False)[
            "drawdown"
        ].apply(lambda x: x.rolling(window=60, min_periods=20).min())
        warns.append("[L3 Features] max_drawdown_60d 생성 완료")

        # Downside Volatility (하방 변동성)
        # 음수 수익률만 고려한 변동성
        out["ret_negative"] = out["ret_daily"].copy()
        out.loc[out["ret_negative"] > 0, "ret_negative"] = 0.0

        out["downside_volatility_60d"] = out.groupby("ticker", group_keys=False)[
            "ret_negative"
        ].apply(
            lambda x: x.rolling(window=60, min_periods=20).std() * np.sqrt(252) * 100.0
        )
        warns.append("[L3 Features] downside_volatility_60d 생성 완료")

        # 중간 계산 컬럼 제거
        out = out.drop(
            columns=["cumret", "rolling_max", "drawdown", "ret_negative"],
            errors="ignore",
        )

    # -------------------------
    # 6) 데이터 클리닝 및 검증
    # -------------------------
    out, cleaning_warns = clean_feature_data(out)
    warns.extend(cleaning_warns)

    # -------------------------
    # 7) 정리: ret_daily는 중간 계산용이므로 제거 (필요시 주석 해제)
    # -------------------------
    # out = out.drop(columns=["ret_daily"], errors="ignore")

    # ticker-date 정렬 유지
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    return out, warns


def clean_feature_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    피처 데이터 클리닝: 결측치 보간, 이상치 처리, 데이터 검증

    Args:
        df: 피처가 추가된 데이터프레임

    Returns:
        (cleaned_df, warnings) 튜플
    """
    warns: list[str] = []
    out = df.copy()

    # 피처 컬럼들 식별 (수치형 피처만 대상)
    numeric_cols = []
    for col in out.columns:
        if col not in [
            "date",
            "ticker",
            "in_universe",
            "sector_name",
        ] and pd.api.types.is_numeric_dtype(out[col]):
            numeric_cols.append(col)

    warns.append(f"[L3 Cleaning] {len(numeric_cols)}개 수치 피처 대상 클리닝 진행")

    # 1. 결측치 처리
    missing_stats = {}
    for col in numeric_cols:
        missing_count = out[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(out) * 100
            missing_stats[col] = missing_pct

            # 결측치 보간 (ticker 그룹 내 forward fill + backward fill)
            out[col] = out.groupby("ticker")[col].transform(
                lambda x: x.fillna(method="ffill").fillna(method="bfill")
            )

            # 그래도 결측치가 남았다면 median으로 채움
            remaining_missing = out[col].isnull().sum()
            if remaining_missing > 0:
                median_val = out[col].median()
                out[col] = out[col].fillna(median_val)
                warns.append(
                    f"[L3 Cleaning] {col}: {missing_pct:.1f}% 결측치 → median({median_val:.3f}) 보간"
                )

    # 2. 이상치 처리 (99% 분위수 기반 클리핑)
    outlier_stats = {}
    for col in numeric_cols:
        if out[col].notna().sum() > 100:  # 충분한 데이터가 있는 경우만
            q99 = out[col].quantile(0.99)
            outlier_count = (out[col] > q99).sum()
            if outlier_count > 0:
                outlier_pct = outlier_count / len(out) * 100
                if outlier_pct > 0.1:  # 0.1% 초과하는 경우만 클리핑
                    # 클리핑 적용
                    out[col] = out[col].clip(upper=q99)
                    outlier_stats[col] = outlier_pct
                    warns.append(
                        f"[L3 Cleaning] {col}: {outlier_pct:.1f}% 이상치 → 99% 분위수({q99:.3f}) 클리핑"
                    )

    # 3. 데이터 검증 요약
    total_missing_handled = len(missing_stats)
    total_outliers_handled = len(outlier_stats)

    if total_missing_handled > 0:
        warns.append(f"[L3 Cleaning] 결측치 처리 완료: {total_missing_handled}개 피처")
    else:
        warns.append("[L3 Cleaning] 결측치 없음 ✓")

    if total_outliers_handled > 0:
        warns.append(f"[L3 Cleaning] 이상치 처리 완료: {total_outliers_handled}개 피처")
    else:
        warns.append("[L3 Cleaning] 이상치 적음 ✓")

    # 4. 최종 검증
    final_shape = out.shape
    warns.append(
        f"[L3 Cleaning] 최종 데이터: {final_shape[0]:,}행 × {final_shape[1]}열"
    )

    return out, warns
