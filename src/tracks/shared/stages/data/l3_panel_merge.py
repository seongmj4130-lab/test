# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l3_panel_merge.py
from __future__ import annotations

import numpy as np
import pandas as pd

def build_panel_merged_daily(
    *,
    ohlcv_daily: pd.DataFrame,
    fundamentals_annual: pd.DataFrame,
    universe_membership_monthly: pd.DataFrame | None = None,
    fundamental_lag_days: int = 90,
    filter_k200_members_only: bool = False,
    # [Stage 1] 추가 파라미터
    fundamentals_effective_date_col: str = "effective_date",
    # [Stage 4] 추가 파라미터
    sector_map: pd.DataFrame | None = None,
    # [L1B] pykrx 재무데이터 추가
    pykrx_fundamentals_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    ohlcv_daily(date,ticker,OHLCV...) + fundamentals_annual(date,ticker,...)를
    fundamental_lag_days 만큼 지연시킨 effective_date 기준으로 asof merge하여
    panel_merged_daily를 생성한다.

    핵심:
    - merge_asof는 left_on 키(date)가 "전역적으로" 정렬돼 있어야 해서
      merge 직전 정렬을 반드시 ['date','ticker']로 맞춘다.
    """
    warns: list[str] = []

    # -------------------------
    # 0) 기본 방어
    # -------------------------
    if ohlcv_daily is None or ohlcv_daily.empty:
        raise ValueError("ohlcv_daily가 비었습니다.")
    if fundamentals_annual is None or fundamentals_annual.empty:
        warns.append("fundamentals_annual이 비었습니다. 머지는 되지만 재무컬럼은 대부분 NaN이 됩니다.")

    o = ohlcv_daily.copy()
    f = fundamentals_annual.copy() if fundamentals_annual is not None else pd.DataFrame()

    # -------------------------
    # 1) 키 표준화
    # -------------------------
    if "date" not in o.columns or "ticker" not in o.columns:
        raise ValueError(f"ohlcv_daily에 date/ticker가 없습니다: {list(o.columns)}")

    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    o["ticker"] = o["ticker"].astype(str).str.zfill(6)
    o = o.dropna(subset=["date", "ticker"])

    if not f.empty:
        if "date" not in f.columns or "ticker" not in f.columns:
            raise ValueError(f"fundamentals_annual에 date/ticker가 없습니다: {list(f.columns)}")

        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f["ticker"] = f["ticker"].astype(str).str.zfill(6)
        f = f.dropna(subset=["date", "ticker"])

        # (ticker,date) 중복 방지
        dup = f.duplicated(["ticker", "date"])
        if dup.any():
            ndup = int(dup.sum())
            warns.append(f"fundamentals_annual duplicate (ticker,date)={ndup} -> keep='last'로 제거")
            f = (
                f.sort_values(["ticker", "date"], kind="mergesort")
                 .drop_duplicates(["ticker", "date"], keep="last")
            )

        # [Stage 1] effective_date 처리
        # Stage1부터는 fundamentals_annual에 이미 effective_date가 포함되어 있으면 우선 사용
        if fundamentals_effective_date_col in f.columns:
            # 이미 effective_date가 있으면 그대로 사용
            if f[fundamentals_effective_date_col].isna().all():
                # 모두 NaN이면 fallback으로 계산
                warns.append(f"[L3] {fundamentals_effective_date_col} exists but all NaN -> using fallback calculation")
                f["effective_date"] = f["date"] + pd.to_timedelta(int(fundamental_lag_days), unit="D")
            else:
                # 일부라도 있으면 사용 (NaN은 fallback으로 보완)
                f["effective_date"] = f[fundamentals_effective_date_col].fillna(
                    f["date"] + pd.to_timedelta(int(fundamental_lag_days), unit="D")
                )
                warns.append(f"[L3] Using {fundamentals_effective_date_col} from fundamentals_annual (Stage 1)")
        else:
            # 기존 방식: date + lag_days
            f["effective_date"] = f["date"] + pd.to_timedelta(int(fundamental_lag_days), unit="D")
            warns.append(f"[L3] {fundamentals_effective_date_col} not found -> using date + {fundamental_lag_days} days (legacy)")

    # -------------------------
    # 2) (옵션) K200 멤버만 필터
    # -------------------------
    # 지금은 기본 False로 두고, 필요 시 확장.
    # (월말 멤버십을 일별로 정교하게 매핑하려면 별도 정책 정의가 필요)
    if filter_k200_members_only and universe_membership_monthly is not None and not universe_membership_monthly.empty:
        if "date" in universe_membership_monthly.columns and "ticker" in universe_membership_monthly.columns:
            u = universe_membership_monthly.copy()
            u["date"] = pd.to_datetime(u["date"], errors="coerce")
            u["ticker"] = u["ticker"].astype(str).str.zfill(6)
            u = u.dropna(subset=["date", "ticker"])

            # 가장 단순한 정책: 월말 멤버십 테이블에 존재하는 ticker만 남김(기간 전체)
            valid_tickers = set(u["ticker"].unique().tolist())
            before = len(o)
            o = o[o["ticker"].isin(valid_tickers)].copy()
            after = len(o)
            warns.append(f"filter_k200_members_only 적용: rows {before} -> {after}")
        else:
            warns.append("filter_k200_members_only 요청했으나 universe_membership_monthly 스키마가 달라 스킵")

    # -------------------------
    # 3) merge_asof (핵심 수정 포인트)
    #    - left: ['date','ticker'] 정렬
    #    - right: ['effective_date','ticker'] 정렬
    # -------------------------
    if f.empty:
        merged = o
    else:
        o_sorted = (
            o.sort_values(["date", "ticker"], kind="mergesort")
             .reset_index(drop=True)
        )
        f_sorted = (
            f.sort_values(["effective_date", "ticker"], kind="mergesort")
             .reset_index(drop=True)
        )

        # right의 원래 date는 남기면 혼동이 생기니 제거(필요 시 fiscal_year 같은 컬럼으로 따로 남길 것)
        f_join = f_sorted.drop(columns=["date"], errors="ignore")

        merged = pd.merge_asof(
            o_sorted,
            f_join,
            left_on="date",
            right_on="effective_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )

        # 정리
        merged = merged.drop(columns=["effective_date"], errors="ignore")

    # -------------------------
    # (P0-2) 월별 유니버스 멤버십 매핑: in_universe 생성 + (옵션) 필터링
    # -------------------------
    def _normalize_ticker(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.upper()
        s = s.str.replace(r"[^0-9]", "", regex=True)
        return s.str.zfill(6)

    def _to_dt(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, errors="coerce")

    if "ticker" in merged.columns:
        merged["ticker"] = _normalize_ticker(merged["ticker"])
    if "date" in merged.columns:
        merged["date"] = _to_dt(merged["date"])
    
    if universe_membership_monthly is not None and len(universe_membership_monthly) > 0:
        un = universe_membership_monthly.copy()
        # universe 쪽 date/ym 정리
        if "ym" not in un.columns:
            if "date" not in un.columns:
                raise KeyError("universe_membership_monthly must have 'date' or 'ym'")
            un["date"] = _to_dt(un["date"])
            un["ym"] = un["date"].dt.to_period("M").astype(str)
        else:
            un["ym"] = un["ym"].astype(str)

        if "ticker" not in un.columns:
            raise KeyError("universe_membership_monthly missing 'ticker'")
        un["ticker"] = _normalize_ticker(un["ticker"])
        un_key = un[["ym", "ticker"]].drop_duplicates()

        merged["ym"] = merged["date"].dt.to_period("M").astype(str)
        merged = merged.merge(un_key, on=["ym", "ticker"], how="left", indicator=True)
        merged["in_universe"] = merged["_merge"].eq("both")
        merged.drop(columns=["_merge"], inplace=True)
    
        # 월별 멤버십 누락 체크(데이터 자체 문제)
        ym_in_data = set(merged["ym"].dropna().unique().tolist())
        ym_in_univ = set(un_key["ym"].dropna().unique().tolist())
        missing_ym = sorted(ym_in_data - ym_in_univ)
        if missing_ym:
            warns.append(f"[L3] universe_membership_monthly missing months: {missing_ym[:10]} (total={len(missing_ym)})")

    else:
        merged["in_universe"] = True
        warns.append("[L3] universe_membership_monthly is empty -> in_universe forced True (NOT OK for K200 strategy)")
    
    if filter_k200_members_only:
        before = len(merged)
        merged = merged.loc[merged["in_universe"]].copy()
        after = len(merged)
        warns.append(f"[L3] filter_k200_members_only=True applied: {before} -> {after} rows")

    # -------------------------
    # [Stage 4] sector_map merge (sector_name 추가)
    # -------------------------
    if sector_map is not None and not sector_map.empty:
        if "date" in sector_map.columns and "ticker" in sector_map.columns and "sector_name" in sector_map.columns:
            sector = sector_map.copy()
            sector["date"] = pd.to_datetime(sector["date"], errors="coerce")
            sector["ticker"] = sector["ticker"].astype(str).str.zfill(6)
            sector = sector.dropna(subset=["date", "ticker", "sector_name"])
            
            if not sector.empty:
                # 날짜별로 가장 최근 업종 정보를 사용 (asof merge)
                merged_sorted = merged.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
                sector_sorted = sector.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
                
                merged = pd.merge_asof(
                    merged_sorted,
                    sector_sorted[["date", "ticker", "sector_name"]],
                    left_on="date",
                    right_on="date",
                    by="ticker",
                    direction="backward",
                    allow_exact_matches=True,
                )
                warns.append("[Stage 4] sector_name merged to panel_merged_daily")
            else:
                warns.append("[Stage 4] sector_map이 비어있어 sector_name merge를 건너뜁니다.")
        else:
            warns.append("[Stage 4] sector_map에 필요한 컬럼(date, ticker, sector_name)이 없어 merge를 건너뜁니다.")
    else:
        warns.append("[Stage 4] sector_map이 제공되지 않아 sector_name을 추가하지 않습니다.")

    # -------------------------
    # [L1B] pykrx 재무데이터 병합 (PER, PBR, EPS, BPS, DIV, market_cap)
    # -------------------------
    if pykrx_fundamentals_daily is not None and not pykrx_fundamentals_daily.empty:
        pykrx = pykrx_fundamentals_daily.copy()
        
        # 키 표준화
        if "date" not in pykrx.columns or "ticker" not in pykrx.columns:
            warns.append("[L1B] pykrx_fundamentals_daily에 date/ticker가 없어 병합을 건너뜁니다.")
        else:
            pykrx["date"] = pd.to_datetime(pykrx["date"], errors="coerce")
            pykrx["ticker"] = pykrx["ticker"].astype(str).str.zfill(6)
            pykrx = pykrx.dropna(subset=["date", "ticker"])
            
            if not pykrx.empty:
                # date, ticker 기준으로 병합
                merged_sorted = merged.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
                pykrx_sorted = pykrx.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
                
                # 중복 제거 (같은 date, ticker에 여러 행이 있을 수 있음)
                pykrx_sorted = pykrx_sorted.drop_duplicates(subset=["date", "ticker"], keep="last")
                
                merged = merged_sorted.merge(
                    pykrx_sorted,
                    on=["date", "ticker"],
                    how="left"
                )
                
                pykrx_cols = ["PER", "PBR", "EPS", "BPS", "DIV", "market_cap"]
                added_cols = [c for c in pykrx_cols if c in merged.columns]
                warns.append(f"[L1B] pykrx 재무데이터 병합 완료: {len(added_cols)}개 컬럼 추가 ({', '.join(added_cols)})")
            else:
                warns.append("[L1B] pykrx_fundamentals_daily가 비어있어 병합을 건너뜁니다.")
    else:
        warns.append("[L1B] pykrx_fundamentals_daily가 제공되지 않아 pykrx 재무데이터를 추가하지 않습니다.")

    # -------------------------
    # [L1B] 회전율 계산 (turnover_ratio = 거래대금 / 시가총액)
    # -------------------------
    if "market_cap" in merged.columns:
        # 거래대금 계산: value 컬럼이 있으면 사용, 없으면 volume * close로 계산
        if "value" in merged.columns:
            trading_value = merged["value"]
        elif "volume" in merged.columns and "close" in merged.columns:
            # volume * close로 거래대금 추정
            trading_value = merged["volume"] * merged["close"]
            warns.append("[L1B] value 컬럼이 없어 volume * close로 거래대금 계산")
        else:
            trading_value = None
        
        if trading_value is not None:
            merged["turnover_ratio"] = trading_value / merged["market_cap"]
            # 0으로 나누기 방지
            merged["turnover_ratio"] = merged["turnover_ratio"].replace([np.inf, -np.inf], np.nan)
            warns.append("[L1B] turnover_ratio 계산 완료 (거래대금/시가총액)")
        else:
            warns.append("[L1B] turnover_ratio 계산 불가 (거래대금 또는 시가총액 없음)")
    else:
        warns.append("[L1B] turnover_ratio 계산 불가 (market_cap 없음)")

    # -------------------------
    # [Stage6] 업종 내 상대화(z-score) 피처 생성: debt_ratio_sector_z, roe_sector_z
    # -------------------------
    if "sector_name" in merged.columns and not merged.empty:
        # debt_ratio, roe가 있는 경우에만 z-score 계산
        for base_col, z_col in [("debt_ratio", "debt_ratio_sector_z"), ("roe", "roe_sector_z")]:
            if base_col in merged.columns:
                # 날짜별, 업종별로 그룹화하여 z-score 계산
                # groupby().transform()을 사용하여 인덱스 유지
                def calc_sector_z(series):
                    """업종 내 z-score 계산 (표준편차 0/표본 부족이면 NaN 유지)"""
                    values = series.dropna()
                    if len(values) < 2:  # 표본 부족 (최소 2개 필요)
                        return pd.Series([np.nan] * len(series), index=series.index)
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val == 0 or np.isnan(std_val):  # 표준편차 0
                        return pd.Series([np.nan] * len(series), index=series.index)
                    # 원본 series의 NaN은 그대로 유지하고, 유효한 값만 z-score 계산
                    result = (series - mean_val) / std_val
                    result[series.isna()] = np.nan  # 원본이 NaN이면 결과도 NaN
                    return result
                
                # transform을 사용하여 원본 인덱스 유지
                merged[z_col] = merged.groupby(["date", "sector_name"], group_keys=False)[base_col].transform(calc_sector_z)
                
                # z-score가 계산된 행 수 확인
                z_count = merged[z_col].notna().sum()
                warns.append(f"[Stage6] {z_col} 생성 완료: {z_count:,} / {len(merged):,} 행 (NaN={merged[z_col].isna().sum():,})")
            else:
                warns.append(f"[Stage6] {base_col} 컬럼이 없어 {base_col}_sector_z를 생성하지 않습니다.")
    else:
        warns.append("[Stage6] sector_name이 없어 업종 내 z-score 피처를 생성하지 않습니다.")

    # -------------------------
    # [개선안: 현재 데이터 기반 팩터 추가] Momentum, Volatility, Liquidity
    # -------------------------
    try:
        from src.tracks.shared.stages.data.l3_feature_engineering import add_price_based_features
        merged, feat_warns = add_price_based_features(
            merged,
            price_col="close",
            volume_col="volume",
            momentum_windows=[20, 60],
            volatility_windows=[20, 60],
            volume_ratio_window=60,
        )
        warns.extend(feat_warns or [])
    except Exception as e:
        warns.append(f"[L3 Features] 팩터 추가 실패 (계속 진행): {type(e).__name__}: {e}")
    
    # -------------------------
    # 4) 후처리: downstream 편의 위해 ticker-date 정렬로 되돌림
    # -------------------------
    merged = merged.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    return merged, warns
