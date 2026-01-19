# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/report_extract_core_data_full.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

BASE = Path(r"C:/Users/seong/OneDrive/바탕 화면/bootcamp/03_code")
INTERIM_DIR = BASE / "data" / "report_core_L0_L7"
OUT_DIR = BASE / "data" / "report_core_L0_L7"
OUT_EXTRACTS = OUT_DIR / "extracts_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_EXTRACTS.mkdir(parents=True, exist_ok=True)

MANIFEST_PQ = OUT_DIR / "_MANIFEST_core_extracts.parquet"  # (artifact_core_name, columns) 기대

def _read_manifest() -> pd.DataFrame:
    if not MANIFEST_PQ.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PQ}\n"
            f"-> 먼저 A~G core 목록을 생성하는 스크립트를 실행해 manifest를 만들어야 합니다."
        )
    m = pd.read_parquet(MANIFEST_PQ).copy()

    # 허용 컬럼명 유연 처리
    if "artifact_core_name" not in m.columns:
        # 흔한 대체 키들
        for cand in ["artifact", "name", "artifact_name", "out_name"]:
            if cand in m.columns:
                m = m.rename(columns={cand: "artifact_core_name"})
                break
    if "columns" not in m.columns:
        for cand in ["cols", "col_list", "core_columns"]:
            if cand in m.columns:
                m = m.rename(columns={cand: "columns"})
                break

    if "artifact_core_name" not in m.columns or "columns" not in m.columns:
        raise KeyError(f"manifest schema mismatch. got={m.columns.tolist()}")

    m["artifact_core_name"] = m["artifact_core_name"].astype(str)
    m["columns"] = m["columns"].astype(str)
    return m

def _parse_columns(s: str) -> List[str]:
    """
    manifest의 columns가
    - "a,b,c" 또는
    - "['a','b','c']" 같은 형태일 수 있어 둘 다 처리
    """
    s = (s or "").strip()
    if not s:
        return []
    # list-like
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = eval(s, {"__builtins__": {}})  # 안전 최소화
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    # comma-separated
    return [x.strip() for x in s.split(",") if x.strip()]

def _load_interim_artifact(name: str) -> Tuple[pd.DataFrame, Path]:
    # 프로젝트에서 save_artifact가 보통 name.parquet를 만들었으므로 우선 그 규칙을 사용
    p1 = INTERIM_DIR / f"{name}.parquet"
    if p1.exists():
        return pd.read_parquet(p1), p1

    # 혹시 name만 폴더로 저장한 케이스 대비(드문 경우)
    p2 = INTERIM_DIR / name / f"{name}.parquet"
    if p2.exists():
        return pd.read_parquet(p2), p2

    raise FileNotFoundError(f"interim artifact parquet not found for '{name}': tried {p1} and {p2}")

def _save_dual(df: pd.DataFrame, out_base: Path) -> Tuple[Path, Path]:
    pq = out_base.with_suffix(".parquet")
    csv = out_base.with_suffix(".csv")
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    return pq, csv

def _flatten_to_1col(df: pd.DataFrame, artifact_name: str) -> pd.DataFrame:
    """
    (아주 큼) 모든 셀을 1컬럼으로 펼침:
      data = "{artifact}.{col} : {value}"
    """
    tmp = df.copy()
    tmp = tmp.reset_index(drop=True)

    # 문자열 변환(NA 포함)
    # stack은 MultiIndex(행,열) -> Series로 변환
    s = tmp.astype(object).stack(dropna=False)

    # s.index = (row_idx, col_name)
    # 값이 너무 길어도 '생략 없이' 저장되도록 그대로 문자열화
    lines = []
    for (r, c), v in s.items():
        lines.append(f"{artifact_name}.{c} : {v}")

    return pd.DataFrame({"data": lines})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flatten-1col", action="store_true", help="모든 셀을 n행1열(data)로 완전 펼침 저장(매우 큼)")
    args = ap.parse_args()

    m = _read_manifest()

    summary_rows: List[Dict[str, Any]] = []
    flat_parts: List[pd.DataFrame] = []

    for _, row in m.iterrows():
        name = str(row["artifact_core_name"])
        col_list = _parse_columns(str(row["columns"]))

        df, src_path = _load_interim_artifact(name)

        # 존재 컬럼만 추출(없는 컬럼은 기록만)
        exist_cols = [c for c in col_list if c in df.columns]
        missing_cols = [c for c in col_list if c not in df.columns]

        if exist_cols:
            out_df = df[exist_cols].copy()
        else:
            # 컬럼이 하나도 안 맞으면 원본을 그대로 저장(완전 실패 방지)
            out_df = df.copy()

        out_base = OUT_EXTRACTS / f"{name}__core_full"
        pq_path, csv_path = _save_dual(out_df, out_base)

        summary_rows.append(
            {
                "artifact": name,
                "src_parquet": str(src_path),
                "rows": int(out_df.shape[0]),
                "cols_saved": int(out_df.shape[1]),
                "saved_parquet": str(pq_path),
                "saved_csv": str(csv_path),
                "requested_cols": ",".join(col_list),
                "missing_cols": ",".join(missing_cols),
            }
        )

        if args.flatten_1col:
            flat_parts.append(_flatten_to_1col(out_df, artifact_name=name))

        print(f"[OK] {name}: saved rows={out_df.shape[0]:,} cols={out_df.shape[1]} -> {pq_path.name}, {csv_path.name}")
        if missing_cols:
            print(f"     [WARN] missing_cols({len(missing_cols)}): {missing_cols[:10]}{' ...' if len(missing_cols)>10 else ''}")

    # 요약 manifest 저장
    summary = pd.DataFrame(summary_rows)
    _save_dual(summary, OUT_DIR / "_SUMMARY_core_full_extracts")

    # 펼침 파일 저장(옵션)
    if args.flatten_1col:
        flat = pd.concat(flat_parts, ignore_index=True) if flat_parts else pd.DataFrame({"data": []})
        _save_dual(flat, OUT_DIR / "_CORE_DATA_flatten_1col_full")
        print(f"[DONE] flatten saved rows={len(flat):,} -> _CORE_DATA_flatten_1col_full.(parquet/csv)")

    print("\n[DONE] summary saved -> _SUMMARY_core_full_extracts.(parquet/csv)")

if __name__ == "__main__":
    main()
