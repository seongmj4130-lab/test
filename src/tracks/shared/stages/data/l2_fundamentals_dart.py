# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l2_fundamentals_dart.py
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _require_opendart():
    try:
        import OpenDartReader
        return OpenDartReader
    except Exception as e:
        raise ImportError(
            "OpenDartReader가 필요합니다. `pip install OpenDartReader` 후 재실행하세요."
        ) from e

def _to_float_safe(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    if s in {"", "-", "nan", "NaN", "None"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _pick_amount(df: pd.DataFrame, names: list[str]) -> float | None:
    """
    OpenDartReader 재무 DF에서 account_nm 기반으로 금액을 뽑는다.
    - 공시 스키마 차이를 고려해 amount 컬럼을 방어적으로 선택한다.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "account_nm" not in df.columns:
        return None

    # amount 컬럼 후보
    cand_cols = [c for c in ["thstrm_amount", "thstrm_amount "] if c in df.columns]
    if not cand_cols:
        # 확실하지 않지만 amount 유사 컬럼 탐색
        num_like = [c for c in df.columns if "amount" in str(c).lower()]
        cand_cols = num_like[:1]

    if not cand_cols:
        return None

    col = cand_cols[0]
    s = df[df["account_nm"].isin(names)][col]
    if s.empty:
        return None

    return _to_float_safe(s.iloc[0])

def _load_corp_map(dart) -> dict[str, str]:
    """
    stock_code(6) -> corp_code(8) 매핑 생성
    """
    corp = getattr(dart, "corp_codes", None)
    corp_df = corp() if callable(corp) else corp
    if corp_df is None or not isinstance(corp_df, pd.DataFrame):
        raise RuntimeError("OpenDartReader corp_codes 로드 실패(버전/환경 확인 필요)")

    if "stock_code" not in corp_df.columns or "corp_code" not in corp_df.columns:
        raise RuntimeError(f"corp_codes 스키마 불일치: {list(corp_df.columns)}")

    c = corp_df.copy()
    c = c[c["stock_code"].notna() & (c["stock_code"].astype(str).str.strip() != "")]
    c["stock_code"] = c["stock_code"].astype(str).str.zfill(6)
    c["corp_code"] = c["corp_code"].astype(str).str.zfill(8)

    return dict(zip(c["stock_code"], c["corp_code"]))

def _call_finstate_safely(dart, corp_code: str, year: int, *, reprt_code: str, fs_div: str | None):
    """
    - OpenDartReader가 '조회 없음'을 dict(status=013)로 반환/출력하는 케이스 방어
    - stdout(불필요 출력) 억제
    - return: (DataFrame|None, status_code|None, message|None)
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            # fs_div 지원 여부가 버전마다 다를 수 있으므로 TypeError 방어
            if fs_div is not None:
                try:
                    res = dart.finstate(corp_code, year, reprt_code=reprt_code, fs_div=fs_div)
                except TypeError:
                    res = dart.finstate(corp_code, year, reprt_code=reprt_code)
            else:
                res = dart.finstate(corp_code, year, reprt_code=reprt_code)
    except Exception as e:
        return None, "EXC", str(e)

    # 케이스 1) dict 반환 (예: {'status':'013','message':'조회된 데이타가 없습니다.'})
    if isinstance(res, dict):
        status = str(res.get("status", "")).strip()
        msg = str(res.get("message", "")).strip()
        return None, status or "DICT", msg or None

    # 케이스 2) DataFrame 반환
    if isinstance(res, pd.DataFrame):
        if res.empty:
            return None, "EMPTY", "empty dataframe"
        return res, None, None

    # 그 외 타입
    return None, "UNKNOWN", f"unexpected type: {type(res)}"

def download_annual_fundamentals(
    *,
    tickers: list[str],
    start_year: int,
    end_year: int,
    api_key: str | None = None,
    api_keys: list[str] | None = None,  # [개선안] 여러 API 키 지원
    sleep_sec: float = 0.2,
    # --- 추가 옵션 (기본값 유지: 기존 호출 깨지지 않음) ---
    reprt_code: str = "11011",           # 사업보고서
    fs_div_order: tuple[str, ...] = ("CFS", "OFS"),  # 연결 우선 -> 개별 fallback
    log_every: int = 100,                # 진행 로그 빈도
    min_success_ratio: float = 0.03,     # 성공률이 너무 낮으면 버그로 판단
    # --- Stage 1 추가 옵션 ---
    disclosure_lag_days: int = 1,        # 공시 접수일 이후 지연일수
    fallback_lag_days: int = 90,         # 접수일 없을 때 fallback 지연일수
) -> pd.DataFrame:
    """
    [Stage 1 업데이트] [개선안] 여러 API 키 지원
    출력:
      - date(YYYY-12-31), ticker(6), corp_code(8),
        report_rcept_date (접수일, 가능한 경우),
        effective_date (유효일),
        lag_source ("rcept_date" | "year_end_fallback"),
        net_income, total_liabilities, equity, debt_ratio, roe

    정책:
      - api_key 또는 api_keys 중 하나는 필수
      - api_keys가 제공되면 순환 사용 (rate limit 회피)
      - ticker->corp_code 매핑 실패는 warnings로 누적(하지만 전체가 실패면 예외)
      - DART 조회 결과가 dict(status=013 등)인 케이스를 안전하게 처리
      - 성공률이 극단적으로 낮으면(대부분 013) "파라미터/매핑 오류"로 보고 실패
      - [Stage 1] report_rcept_date 추출 시도 (finstate DataFrame 또는 list() 메서드)
      - [Stage 1] effective_date = report_rcept_date + disclosure_lag_days (있으면)
                   또는 year_end + fallback_lag_days (fallback)
    """
    # [개선안] 여러 API 키 지원: api_keys 우선, 없으면 api_key, 없으면 환경변수
    if api_keys is None or len(api_keys) == 0:
        if api_key is None:
            api_key = os.getenv("DART_API_KEY")
        api_key = (api_key or "").strip()
        if api_key:
            api_keys = [api_key]
        else:
            # 환경변수에서 여러 키를 쉼표로 구분하여 받을 수 있음
            env_keys = os.getenv("DART_API_KEYS", "")
            if env_keys:
                api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            else:
                api_keys = []

    if not api_keys or len(api_keys) == 0:
        raise RuntimeError(
            "DART_API_KEY가 없습니다. api_key, api_keys 파라미터 또는 환경변수 DART_API_KEY/DART_API_KEYS를 설정해야 L2를 진행할 수 있습니다."
        )

    # API 키 정리
    api_keys = [k.strip() for k in api_keys if k.strip()]
    if len(api_keys) == 0:
        raise RuntimeError("유효한 DART_API_KEY가 없습니다.")

    logger.info(f"[L2] API 키 {len(api_keys)}개 사용: {api_keys[0][:10]}..." + (f", {api_keys[1][:10]}..." if len(api_keys) > 1 else ""))

    OpenDartReader = _require_opendart()
    # 첫 번째 키로 corp_map 로드 (corp_map은 키와 무관)
    dart = OpenDartReader(api_keys[0])
    corp_map = _load_corp_map(dart)

    # API 키 순환을 위한 인덱스
    api_key_idx = 0

    # ticker 정규화
    norm_tickers = sorted({str(t).strip().zfill(6) for t in tickers if str(t).strip() != ""})

    # 매핑 성공률 체크(초기 버그 탐지)
    mapped = [t for t in norm_tickers if t in corp_map]
    map_ratio = (len(mapped) / max(len(norm_tickers), 1))
    logger.info(f"[L2] corp_code mapping: {len(mapped)}/{len(norm_tickers)} ({map_ratio:.1%})")

    if len(mapped) == 0:
        raise RuntimeError(
            "[L2] ticker->corp_code 매핑이 0건입니다. "
            "ticker 포맷(6자리), corp_codes 로드, 유니버스 생성 로직을 확인하세요."
        )

    records: list[dict[str, Any]] = []

    # 통계
    req_cnt = 0
    ok_cnt = 0
    no_data_cnt = 0
    map_miss_cnt = 0
    exc_cnt = 0

    for t in norm_tickers:
        corp_code = corp_map.get(t)
        if not corp_code:
            map_miss_cnt += 1
            continue

        for y in range(start_year, end_year + 1):
            req_cnt += 1

            fs = None
            status = None
            msg = None

            # [개선안] API 키 순환 사용 (rate limit 회피)
            current_api_key = api_keys[api_key_idx % len(api_keys)]
            dart = OpenDartReader(current_api_key)
            api_key_idx += 1

            # CFS -> OFS 순서로 시도
            for fs_div in fs_div_order:
                fs, status, msg = _call_finstate_safely(
                    dart, corp_code, y, reprt_code=reprt_code, fs_div=fs_div
                )
                if fs is not None:
                    break

            if fs is None:
                # status가 013/EMPTY 등: 데이터 없음으로 처리
                if status in {"013", "EMPTY"}:
                    no_data_cnt += 1
                elif status == "EXC":
                    exc_cnt += 1
                else:
                    # UNKNOWN/DICT 등도 no-data로 보되, 카운팅은 별도로
                    no_data_cnt += 1

                # 여기서 row를 굳이 만들면 "df는 비지 않지만 값은 전부 None"이 되어 검증이 약해짐.
                # → fundamentals는 '있는 것만' 적재하고, 없는 것은 merge에서 NaN으로 남기는 게 정상.
                if req_cnt % log_every == 0:
                    logger.info(
                        f"[L2] progress req={req_cnt} ok={ok_cnt} no_data={no_data_cnt} map_miss={map_miss_cnt} exc={exc_cnt}"
                    )
                time.sleep(sleep_sec)
                continue

            # 정상 DF를 받았을 때만 파싱
            ok_cnt += 1

            # [Stage 1] report_rcept_date 추출 (접수일)
            report_rcept_date = None
            lag_source = "year_end_fallback"

            # 방법 1: finstate DataFrame에서 접수일 컬럼 찾기
            rcept_cols = [c for c in fs.columns if "rcept" in str(c).lower() or "접수" in str(c)]
            if rcept_cols:
                try:
                    # 각 행에서 접수일 추출 시도 (첫 번째 유효한 값 사용)
                    for col in rcept_cols:
                        rcept_vals = fs[col].dropna()
                        if len(rcept_vals) > 0:
                            rcept_val = rcept_vals.iloc[0]
                            if pd.notna(rcept_val) and str(rcept_val).strip() != "":
                                # YYYYMMDD 형식으로 파싱 시도
                                rcept_str = str(rcept_val).strip().replace("-", "").replace("/", "")
                                if len(rcept_str) == 8 and rcept_str.isdigit():
                                    report_rcept_date = pd.to_datetime(rcept_str, format="%Y%m%d", errors="coerce")
                                    if pd.notna(report_rcept_date):
                                        lag_source = "rcept_date"
                                        break
                except Exception:
                    pass

            # 방법 2: OpenDartReader의 list() 메서드로 공시 목록에서 접수일 가져오기 (fallback)
            # 주의: 추가 API 호출이므로 성능 고려 필요
            if lag_source != "rcept_date":
                try:
                    buf2 = io.StringIO()
                    with redirect_stdout(buf2):
                        # reprt_code에 해당하는 공시 목록 조회
                        list_res = dart.list(corp_code, kind="A", start=f"{y}0101", end=f"{y}1231")
                        if isinstance(list_res, pd.DataFrame) and not list_res.empty:
                            # reprt_code와 일치하는 공시 찾기
                            if "reprt_code" in list_res.columns:
                                matching = list_res[list_res["reprt_code"] == reprt_code]
                            else:
                                matching = list_res

                            if not matching.empty and "rcept_dt" in matching.columns:
                                rcept_val = matching["rcept_dt"].iloc[0]
                                if pd.notna(rcept_val) and str(rcept_val).strip() != "":
                                    rcept_str = str(rcept_val).strip().replace("-", "").replace("/", "")
                                    if len(rcept_str) == 8 and rcept_str.isdigit():
                                        report_rcept_date = pd.to_datetime(rcept_str, format="%Y%m%d", errors="coerce")
                                        if pd.notna(report_rcept_date):
                                            lag_source = "rcept_date"
                except Exception:
                    pass  # list() 실패해도 계속 진행 (fallback 사용)

            # year_end 날짜 (기본값)
            year_end = pd.to_datetime(f"{y}-12-31")

            # effective_date 계산
            if lag_source == "rcept_date" and report_rcept_date is not None:
                effective_date = report_rcept_date + pd.Timedelta(days=disclosure_lag_days)
            else:
                # fallback: year_end + fallback_lag_days
                effective_date = year_end + pd.Timedelta(days=fallback_lag_days)
                report_rcept_date = None  # 접수일을 못 구한 경우 None으로 기록

            net_income = _pick_amount(
                fs,
                [
                    "당기순이익",
                    "당기순이익(손실)",
                    "지배기업소유주지분에 대한 당기순이익",
                    "지배기업소유주지분당기순이익",
                ],
            )
            total_liab = _pick_amount(fs, ["부채총계", "부채총액"])
            equity = _pick_amount(
                fs,
                [
                    "자본총계",
                    "자본총액",
                    "자본총계(지배기업 소유주지분)",
                    "지배기업소유주지분",
                ],
            )

            debt_ratio = None
            roe = None
            if (total_liab is not None) and (equity is not None) and (equity != 0):
                debt_ratio = (total_liab / equity) * 100.0
            if (net_income is not None) and (equity is not None) and (equity != 0):
                roe = (net_income / equity) * 100.0

            records.append(
                {
                    "date": f"{y}-12-31",  # year_end (기존 호환성 유지)
                    "ticker": t,
                    "corp_code": str(corp_code).zfill(8),
                    "report_rcept_date": report_rcept_date,  # [Stage 1] 접수일
                    "effective_date": effective_date,  # [Stage 1] 유효일
                    "lag_source": lag_source,  # [Stage 1] 지연 소스
                    "net_income": net_income,
                    "total_liabilities": total_liab,
                    "equity": equity,
                    "debt_ratio": debt_ratio,
                    "roe": roe,
                }
            )

            if req_cnt % log_every == 0:
                logger.info(
                    f"[L2] progress req={req_cnt} ok={ok_cnt} no_data={no_data_cnt} map_miss={map_miss_cnt} exc={exc_cnt}"
                )

            time.sleep(sleep_sec)

    df = pd.DataFrame(records)

    # 최종 품질 체크(너가 원한 '검증을 강제'하는 부분)
    if req_cnt == 0:
        raise RuntimeError("[L2] 요청 건수가 0입니다. tickers/start_year/end_year 입력을 확인하세요.")

    success_ratio = ok_cnt / max(req_cnt, 1)
    logger.info(
        f"[L2] done req={req_cnt} ok={ok_cnt} no_data={no_data_cnt} map_miss={map_miss_cnt} exc={exc_cnt} success_ratio={success_ratio:.2%}"
    )

    if df.empty:
        raise RuntimeError(
            "[L2] DART 재무 수집 결과가 비었습니다. "
            f"(req={req_cnt}, ok={ok_cnt}, no_data={no_data_cnt}, map_miss={map_miss_cnt}, exc={exc_cnt}) "
            "corp_code 매핑/파라미터(reprt_code/fs_div)/API키/호출 제한을 확인하세요."
        )

    # “대부분 013”이면 버그 가능성이 높으니 강제 실패(조용히 다음 단계로 못 넘어가게)
    if success_ratio < min_success_ratio:
        raise RuntimeError(
            "[L2] DART 조회 성공률이 비정상적으로 낮습니다. "
            f"success_ratio={success_ratio:.2%} (min={min_success_ratio:.2%}). "
            "corp_code 사용 여부, finstate 파라미터(reprt_code/fs_div), ticker 포맷(zfill) 문제를 점검하세요."
        )

    # 타입 정리
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)

    # [Stage 1] effective_date, report_rcept_date 타입 정리
    if "effective_date" in df.columns:
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    if "report_rcept_date" in df.columns:
        df["report_rcept_date"] = pd.to_datetime(df["report_rcept_date"], errors="coerce")
    if "lag_source" in df.columns:
        df["lag_source"] = df["lag_source"].astype(str)

    return df
