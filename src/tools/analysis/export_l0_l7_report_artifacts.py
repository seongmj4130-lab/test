# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/export_l0_l7_report_artifacts.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

# ----------------------------
# 0) 기본 설정
# ----------------------------
DEFAULT_BASE = Path(r"C:/Users/seong/OneDrive/바탕 화면/bootcamp/03_code")
DEFAULT_INTERIM = DEFAULT_BASE / "data" / "interim"
DEFAULT_OUTDIR = DEFAULT_BASE / "data" / "final_report" / "l0_l7_exports"

# L0~L7 핵심 산출물 (요청 범위 그대로)
ARTIFACTS_L0_L7 = [
    # L0
    "universe_k200_membership_monthly",
    # L1
    "ohlcv_daily",
    # L2
    "fundamentals_annual",
    # L3
    "panel_merged_daily",
    # L4
    "dataset_daily",
    "cv_folds_short",
    "cv_folds_long",
    # L5
    "pred_short_oos",
    "pred_long_oos",
    "model_metrics",
    # L6
    "rebalance_scores",
    "rebalance_scores_summary",
    # L7
    "bt_positions",
    "bt_returns",
    "bt_equity_curve",
    "bt_metrics",
]

# ----------------------------
# 1) 유틸
# ----------------------------
def _set_pandas_print_options():
    # "생략 없이 프린트"를 가능하게 하는 옵션(콘솔/파일 모두)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

def _load_artifact(interim_dir: Path, name: str) -> pd.DataFrame:
    """
    run_all.py 저장 규칙(주로 parquet)을 기준으로 로드.
    - 우선: <name>.parquet
    - fallback: <name>.csv
    """
    p_parquet = interim_dir / f"{name}.parquet"
    p_csv = interim_dir / f"{name}.csv"

    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        # 날짜 컬럼 자동 파싱은 프로젝트마다 다르므로 최소 로드만 수행
        return pd.read_csv(p_csv)

    raise FileNotFoundError(f"[MISS] artifact not found: {name} (expected {p_parquet} or {p_csv})")

def _transform_to_nx1(df: pd.DataFrame) -> pd.DataFrame:
    """
    [핵심 수정] 다중 컬럼 DataFrame을 'n행 1열' 형태로 변환.
    각 행의 데이터를 'Key: Value | Key: Value' 형태의 단일 문자열로 병합.
    """
    if df.empty:
        return pd.DataFrame(columns=["row_data"])

    # lambda 함수: 한 행(row)을 받아서 문자열로 결합
    def row_to_string(row):
        # 값이 None/NaN인 경우도 문자열로 처리
        items = [f"{k}: {v}" for k, v in row.items()]
        return " | ".join(items)

    # 행별 적용 (axis=1)
    # 결과는 Series 형태가 되므로 to_frame()으로 DataFrame 변환
    series_nx1 = df.apply(row_to_string, axis=1)
    
    return series_nx1.to_frame(name="row_data")

def _save_both(df: pd.DataFrame, out_base: Path):
    """
    out_base가 '.../artifact_name' 형태라고 가정.
    - parquet: .../artifact_name.parquet
    - csv    : .../artifact_name.csv
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)

    p_parquet = out_base.with_suffix(".parquet")
    p_csv = out_base.with_suffix(".csv")

    # parquet
    df.to_parquet(p_parquet, index=False)

    # csv (윈도/엑셀 호환 위해 utf-8-sig)
    df.to_csv(p_csv, index=False, encoding="utf-8-sig")

    return p_parquet, p_csv

def _dump_full_to_txt(df: pd.DataFrame, txt_path: Path, chunk_rows: int = 2000):
    """
    "생략 없이 출력" 요구를 충족하기 위해
    df 전체를 .txt로 덤프(행이 많아도 chunk로 나눠 메모리 폭발 방지).
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(df.shape[0])

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"[FULL DUMP - Nx1 Format]\nshape={df.shape}\ncolumns={list(df.columns)}\n\n")
        if n == 0:
            f.write("(empty)\n")
            return

        # chunk print
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            part = df.iloc[start:end]
            f.write(f"\n--- rows {start} ~ {end-1} ---\n")
            # index=False로 설정하여 순수 데이터만 출력 (row_data 컬럼 내용)
            f.write(part.to_string(index=False, header=False)) 
            f.write("\n")

# ----------------------------
# 2) 메인
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    parser.add_argument("--interim", type=str, default=str(DEFAULT_INTERIM))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--dump-txt", action="store_true", help="각 artifact를 txt로 '생략 없이' 덤프")
    parser.add_argument("--dump-chunk-rows", type=int, default=2000)
    args = parser.parse_args()

    base_dir = Path(args.base)
    interim_dir = Path(args.interim)
    out_dir = Path(args.outdir)

    if not interim_dir.exists():
        raise FileNotFoundError(f"interim dir not found: {interim_dir}")

    _set_pandas_print_options()

    print("=== EXPORT L0~L7 ARTIFACTS (Format: n-rows x 1-col) ===")
    print("BASE    :", base_dir)
    print("INTERIM :", interim_dir)
    print("OUTDIR  :", out_dir)
    print("DUMP_TXT:", bool(args.dump_txt))
    print()

    ok = []
    fail = []

    for name in ARTIFACTS_L0_L7:
        try:
            # 1. 로드
            df_origin = _load_artifact(interim_dir, name)
            
            # 2. n행 1열 변환
            df_nx1 = _transform_to_nx1(df_origin)

            out_base = out_dir / name

            # 3. 저장 (변환된 df_nx1 저장)
            p_parquet, p_csv = _save_both(df_nx1, out_base)

            # 콘솔 정보 출력
            print(f"[OK] {name}")
            print(f"  original shape : {df_origin.shape}")
            print(f"  nx1 shape      : {df_nx1.shape}")
            print(f"  saved          : {p_parquet}")
            print(f"                   {p_csv}")

            # 4. (옵션) TXT 덤프
            if args.dump_txt:
                txt_path = out_dir / "_full_print" / f"{name}.txt"
                _dump_full_to_txt(df_nx1, txt_path, chunk_rows=int(args.dump_chunk_rows))
                print(f"  fulltxt        : {txt_path}")

            print()
            ok.append(name)

        except Exception as e:
            print(f"[FAIL] {name} -> {e}")
            fail.append((name, str(e)))

    print("=== SUMMARY ===")
    print("OK  :", len(ok), "/", len(ARTIFACTS_L0_L7))
    if fail:
        print("FAIL:", len(fail))
        for n, msg in fail:
            print(f" - {n}: {msg}")

    # 실패가 있으면 exit code 1
    if fail:
        sys.exit(1)

if __name__ == "__main__":
    main()
