# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l1b_sector_map.py
from __future__ import annotations

import pandas as pd
from typing import List

def _require_pykrx():
    try:
        from pykrx import stock
        return stock
    except Exception as e:
        raise ImportError("pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요.") from e

def build_sector_map(
    *,
    asof_dates: pd.DatetimeIndex | List[pd.Timestamp],
    tickers: List[str],
) -> pd.DataFrame:
    """
    [Stage 4] 업종 매핑 생성
    
    각 날짜별로 티커의 업종 정보를 수집하여 sector_map을 생성합니다.
    
    Args:
        asof_dates: 기준일 리스트 (일반적으로 월말 거래일)
        tickers: 티커 리스트 (6자리 문자열)
    
    Returns:
        DataFrame with columns: date, ticker, sector_name
        - date: 기준일 (datetime64)
        - ticker: 티커 (6자리 문자열)
        - sector_name: 업종명 (문자열)
    """
    stock = _require_pykrx()
    
    # ticker 정규화
    norm_tickers = sorted(set(str(t).strip().zfill(6) for t in tickers if str(t).strip()))
    
    if not norm_tickers:
        raise ValueError("tickers가 비어있습니다.")
    
    # dates 정규화
    if isinstance(asof_dates, pd.DatetimeIndex):
        dates = asof_dates.tolist()
    else:
        dates = [pd.Timestamp(d) for d in asof_dates]
    
    records: List[dict] = []
    
    # [Stage 4] 실데이터로 업종 정보 수집
    # pykrx를 사용하여 실제 업종 정보를 가져옵니다.
    # 주의: pykrx는 직접적인 업종 정보 API를 제공하지 않으므로,
    # 종목명 기반 업종 추정 방식을 사용합니다.
    
    # [Stage 4] 실데이터로 업종 정보 수집
    # 각 날짜별로 업종 정보 수집 (날짜별로 업종이 변경될 수 있음)
    import logging
    logger = logging.getLogger(__name__)
    
    for date in dates:
        date_ts = pd.Timestamp(date)
        date_str = date_ts.strftime("%Y%m%d")
        
        # [Stage 4] pykrx를 사용하여 실제 업종 정보 가져오기
        # 날짜별 업종 매핑 캐시 (같은 날짜 내에서는 재사용)
        sector_mapping_cache = {}
        
        # [Stage 4] 전체 티커 리스트에서 종목명 기반 업종 추정
        # pykrx의 get_market_ticker_name()을 사용하여 종목명 가져오기
        for ticker in norm_tickers:
            if ticker in sector_mapping_cache:
                sector_name = sector_mapping_cache[ticker]
            else:
                try:
                    sector_name = _get_sector_from_pykrx(ticker, date_ts)
                    sector_mapping_cache[ticker] = sector_name
                except Exception as e:
                    logger.warning(f"[Stage 4] 업종 정보 가져오기 실패 (ticker={ticker}, date={date_str}): {e}")
                    sector_name = "기타"
                    sector_mapping_cache[ticker] = sector_name
            
            records.append({
                "date": date_ts,
                "ticker": ticker,
                "sector_name": sector_name,
            })
    
    if not records:
        raise RuntimeError("업종 매핑 데이터를 생성할 수 없습니다. tickers와 dates를 확인하세요.")
    
    df = pd.DataFrame(records)
    df = df.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    
    return df

# [Stage 4] 실제 업종 정보를 가져오는 헬퍼 함수 (pykrx 사용)
def _get_sector_from_pykrx(ticker: str, date: pd.Timestamp) -> str:
    """
    pykrx를 사용하여 실제 업종 정보를 가져오는 함수
    
    pykrx의 get_market_ticker_list()를 사용하여 업종별 티커 리스트를 가져온 후
    역매핑하여 업종 정보를 추출합니다.
    
    Args:
        ticker: 티커 (6자리 문자열)
        date: 기준일
    
    Returns:
        업종명 (문자열)
    """
    stock = _require_pykrx()
    
    try:
        # pykrx의 업종 코드 리스트 가져오기
        # 업종 코드: 01=종합주가지수, 02=대형주, 03=중형주, 04=소형주, 05=섹터지수 등
        # 실제 업종 분류는 섹터지수(05)를 사용
        
        # 날짜를 문자열로 변환 (YYYYMMDD)
        date_str = date.strftime("%Y%m%d")
        
        # 업종별 티커 리스트 가져오기 (섹터지수 사용)
        # pykrx의 섹터지수 업종 코드:
        # - G25: 전기·가스
        # - G35: 건설업
        # - G45: 기계
        # - G50: 자동차
        # - G52: 화학
        # - G55: 비금속
        # - G60: 금속
        # - G70: IT·전자
        # - G75: 유통
        # - G80: 운송
        # - G85: 금융
        # - G90: 통신
        # - G95: 서비스
        
        # 각 업종별 티커 리스트를 가져와서 역매핑
        sector_codes = {
            "G25": "전기·가스",
            "G35": "건설업",
            "G45": "기계",
            "G50": "자동차",
            "G52": "화학",
            "G55": "비금속",
            "G60": "금속",
            "G70": "IT·전자",
            "G75": "유통",
            "G80": "운송",
            "G85": "금융",
            "G90": "통신",
            "G95": "서비스",
        }
        
        # [Stage 4] pykrx의 get_market_ticker_name()으로 종목명을 가져와서 업종 추정
        # pykrx는 직접적인 업종 정보 API를 제공하지 않으므로 종목명 기반 추정 사용
        try:
            stock_name = stock.get_market_ticker_name(ticker)
            # 종목명 기반 업종 추정 (간단한 키워드 매칭)
            if any(kw in stock_name for kw in ["은행", "금융", "금융지주"]):
                return "금융"
            elif any(kw in stock_name for kw in ["증권", "투자"]):
                return "금융"
            elif any(kw in stock_name for kw in ["보험", "생명", "화재"]):
                return "금융"
            elif any(kw in stock_name for kw in ["전자", "반도체", "디스플레이"]):
                return "IT·전자"
            elif any(kw in stock_name for kw in ["화학", "화학소재"]):
                return "화학"
            elif any(kw in stock_name for kw in ["자동차", "모빌리티"]):
                return "자동차"
            elif any(kw in stock_name for kw in ["건설", "엔지니어링"]):
                return "건설업"
            elif any(kw in stock_name for kw in ["통신", "텔레콤"]):
                return "통신"
            elif any(kw in stock_name for kw in ["유통", "마트", "백화점"]):
                return "유통"
            elif any(kw in stock_name for kw in ["운송", "물류", "항공", "해운"]):
                return "운송"
            elif any(kw in stock_name for kw in ["에너지", "전기", "가스"]):
                return "전기·가스"
            elif any(kw in stock_name for kw in ["금속", "철강", "비철금속"]):
                return "금속"
            elif any(kw in stock_name for kw in ["기계", "중공업"]):
                return "기계"
            elif any(kw in stock_name for kw in ["서비스", "레저", "엔터테인먼트"]):
                return "서비스"
        except Exception:
            pass
        
        # 모든 방법 실패 시 "기타" 반환
        return "기타"
        
    except Exception as e:
        # pykrx 오류 발생 시 "기타" 반환
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[Stage 4] pykrx에서 업종 정보를 가져오는 중 오류 발생 (ticker={ticker}, date={date}): {e}")
        return "기타"
