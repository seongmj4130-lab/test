# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l1_ohlcv.py
# [FEATURESET_COMPLETE] 기술적 지표 계산 통합
from __future__ import annotations

import pandas as pd

def _require_pykrx():
    try:
        from pykrx import stock
        return stock
    except Exception as e:
        raise ImportError("pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요.") from e

def _to_yyyymmdd(s: str) -> str:
    return pd.to_datetime(s).strftime("%Y%m%d")

def download_ohlcv_panel(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    calculate_technical_features: bool = True,
) -> pd.DataFrame:
    """
    OHLCV 데이터 다운로드 및 기술적 지표 계산
    
    Args:
        tickers: 종목코드 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        calculate_technical_features: 기술적 지표 계산 여부 (기본값: True)
    
    Returns:
        date, ticker, open, high, low, close, volume, value 및 기술적 지표 컬럼 포함
    """
    stock = _require_pykrx()

    s = _to_yyyymmdd(start_date)
    e = _to_yyyymmdd(end_date)

    frames = []
    for t in sorted(set(tickers)):
        o = stock.get_market_ohlcv_by_date(s, e, t)
        if o is None or len(o) == 0:
            # 상폐/휴면 등으로 비어있을 수 있음 -> 이후 validate에서 걸러도 됨
            continue

        df = o.copy()
        df = df.reset_index()
        # 컬럼명이 한글일 수 있음(시가/고가/저가/종가/거래량/거래대금)
        rename = {
            "날짜": "date",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "value",
        }
        df = df.rename(columns=rename)

        # pykrx가 인덱스를 날짜로 주는 경우가 많아서 'date'가 없을 수도 있음
        if "date" not in df.columns:
            # reset_index 후 첫 컬럼이 날짜일 가능성
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["ticker"] = str(t).zfill(6)

        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        if "value" in df.columns:
            keep.append("value")
        df = df[keep]

        frames.append(df)

    if not frames:
        raise RuntimeError("OHLCV 다운로드 결과가 비었습니다. tickers/start/end 확인 필요")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # [FEATURESET_COMPLETE] 기술적 지표 계산
    if calculate_technical_features:
        try:
            from src.tracks.shared.stages.data.l1_technical_features import calculate_technical_features
            out = calculate_technical_features(out)
        except ImportError as e:
            import warnings
            warnings.warn(f"기술적 지표 계산 모듈을 불러올 수 없습니다: {e}. 기술적 지표 없이 진행합니다.")
        except Exception as e:
            import warnings
            warnings.warn(f"기술적 지표 계산 중 오류 발생: {e}. 기술적 지표 없이 진행합니다.")
    
    return out
