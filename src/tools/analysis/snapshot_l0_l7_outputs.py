# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/snapshot_l0_l7_outputs.py
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.config import load_config, get_path

# -----------------------------
# Config
# -----------------------------
DEFAULT_ARTIFACTS = [
    # L0~L4
    "universe_k200_membership_monthly",
    "ohlcv_daily",
    "fundamentals_annual",
    "panel_merged_daily",
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
    # L7B/L7C/L7D extensions
    "bt_sensitivity_metrics",
    "bt_vs_benchmark",
    "bt_benchmark_returns",
    "bt_benchmark_compare",
    "bt_yearly_metrics",
    "bt_drawdown_events",
    "bt_rolling_sharpe",
]

EXPORT_EXTS = [".parquet", ".csv"]

@dataclass
class ArtifactRecord:
    name: str
    src_base: str
    dst_base: str
    has_parquet: bool
    has_csv: bool
    has_meta: bool
    parquet_bytes: int
    csv_bytes: int
    meta_bytes: int
    meta_stage: str
    meta_run_id: str
    meta_n_rows: int
    meta_n_cols: int

def _root() -> Path:
    # .../03_code/src/stages/snapshot_l0_l7_outputs.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]

def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _discover_artifacts_from_meta(interim: Path) -> List[str]:
    names = []
    for mp in sorted(interim.glob("*__meta.json")):
        # e.g., pred_short_oos__meta.json -> pred_short_oos
        stem = mp.name.replace("__meta.json", "")
        if stem:
            names.append(stem)
    return sorted(set(names))

def _file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0

def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return True

def _parse_meta(meta_path: Path) -> Tuple[str, str, int, int]:
    """
    meta JSON 구조는 utils.meta.build_meta() 결과를 따른다고 가정.
    최소한 stage/run_id/df_shape(혹은 n_rows/n_cols)를 안전하게 읽는다.
    """
    m = _read_json(meta_path)

    stage = str(m.get("stage", ""))
    run_id = str(m.get("run_id", ""))

    # build_meta에서 df info가 어떤 키로 저장되든, 아래 순서로 우선 탐색
    n_rows = -1
    n_cols = -1

    # 1) df_shape
    if isinstance(m.get("df_shape", None), (list, tuple)) and len(m["df_shape"]) == 2:
        n_rows = int(m["df_shape"][0])
        n_cols = int(m["df_shape"][1])

    # 2) df / summary 내부
    if (n_rows < 0 or n_cols < 0) and isinstance(m.get("df", None), dict):
        d = m["df"]
        if "n_rows" in d:
            n_rows = int(d["n_rows"])
        if "n_cols" in d:
            n_cols = int(d["n_cols"])

    # 3) fallback
    if n_rows < 0:
        n_rows = int(m.get("n_rows", -1))
    if n_cols < 0:
        n_cols = int(m.get("n_cols", -1))

    return stage, run_id, n_rows, n_cols

def snapshot(
    *,
    root: Path,
    interim: Path,
    out_dir: Path,
    include_discovered: bool,
) -> List[ArtifactRecord]:
    _safe_mkdir(out_dir)

    # export 대상 artifact 목록 확정
    names = list(DEFAULT_ARTIFACTS)
    if include_discovered:
        names += _discover_artifacts_from_meta(interim)
    names = sorted(set(names))

    records: List[ArtifactRecord] = []

    for name in names:
        src_base = interim / name
        dst_base = out_dir / name

        src_parquet = src_base.with_suffix(".parquet")
        src_csv = src_base.with_suffix(".csv")
        src_meta = interim / f"{name}__meta.json"

        dst_parquet = dst_base.with_suffix(".parquet")
        dst_csv = dst_base.with_suffix(".csv")
        dst_meta = out_dir / f"{name}__meta.json"

        has_parquet = _copy_if_exists(src_parquet, dst_parquet)
        has_csv = _copy_if_exists(src_csv, dst_csv)
        has_meta = _copy_if_exists(src_meta, dst_meta)

        parquet_bytes = _file_size(dst_parquet) if has_parquet else 0
        csv_bytes = _file_size(dst_csv) if has_csv else 0
        meta_bytes = _file_size(dst_meta) if has_meta else 0

        meta_stage = ""
        meta_run_id = ""
        meta_n_rows = -1
        meta_n_cols = -1
        if has_meta:
            meta_stage, meta_run_id, meta_n_rows, meta_n_cols = _parse_meta(dst_meta)

        # parquet/csv/meta 중 하나도 없으면 기록은 남기되, 존재 여부로 확인 가능하게 한다.
        rec = ArtifactRecord(
            name=name,
            src_base=str(src_base),
            dst_base=str(dst_base),
            has_parquet=has_parquet,
            has_csv=has_csv,
            has_meta=has_meta,
            parquet_bytes=parquet_bytes,
            csv_bytes=csv_bytes,
            meta_bytes=meta_bytes,
            meta_stage=meta_stage,
            meta_run_id=meta_run_id,
            meta_n_rows=meta_n_rows,
            meta_n_cols=meta_n_cols,
        )
        records.append(rec)

    return records

def write_manifest(out_dir: Path, records: List[ArtifactRecord]) -> None:
    # CSV
    manifest_csv = out_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            "has_parquet",
            "has_csv",
            "has_meta",
            "parquet_bytes",
            "csv_bytes",
            "meta_bytes",
            "meta_stage",
            "meta_run_id",
            "meta_n_rows",
            "meta_n_cols",
        ])
        for r in records:
            w.writerow([
                r.name,
                int(r.has_parquet),
                int(r.has_csv),
                int(r.has_meta),
                r.parquet_bytes,
                r.csv_bytes,
                r.meta_bytes,
                r.meta_stage,
                r.meta_run_id,
                r.meta_n_rows,
                r.meta_n_cols,
            ])

    # JSON
    manifest_json = out_dir / "manifest.json"
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": [r.__dict__ for r in records],
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # README
    readme = out_dir / "README.txt"
    n_total = len(records)
    n_ok = sum(1 for r in records if r.has_parquet and r.has_csv and r.has_meta)
    with readme.open("w", encoding="utf-8") as f:
        f.write("Snapshot created.\n")
        f.write(f"- total records: {n_total}\n")
        f.write(f"- fully packaged (parquet+csv+meta): {n_ok}\n")
        f.write("- files:\n")
        f.write("  - *.parquet / *.csv per artifact (if existed in interim)\n")
        f.write("  - *__meta.json per artifact (if existed in interim)\n")
        f.write("  - manifest.csv / manifest.json\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--include-discovered", action="store_true")
    args = parser.parse_args()

    root = _root()
    cfg_path = _cfg_path(root)
    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")

    tag = args.tag.strip()
    if not tag:
        tag = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")

    out_dir = root / "data" / "snapshots" / tag
    _safe_mkdir(out_dir)

    print("=== SNAPSHOT RUNNER ===")
    print("ROOT  :", root)
    print("CFG   :", cfg_path)
    print("INTERIM:", interim)
    print("OUT   :", out_dir)
    print("include_discovered:", bool(args.include_discovered))

    # config도 같이 복사(재현성)
    _copy_if_exists(cfg_path, out_dir / "config.yaml")

    records = snapshot(
        root=root,
        interim=Path(interim),
        out_dir=out_dir,
        include_discovered=bool(args.include_discovered),
    )
    write_manifest(out_dir, records)

    # 요약 출력
    n_total = len(records)
    n_meta = sum(1 for r in records if r.has_meta)
    n_parq = sum(1 for r in records if r.has_parquet)
    n_csv = sum(1 for r in records if r.has_csv)
    n_full = sum(1 for r in records if r.has_parquet and r.has_csv and r.has_meta)

    print("\n=== SUMMARY ===")
    print(f"records: {n_total}")
    print(f"has_meta: {n_meta} / has_parquet: {n_parq} / has_csv: {n_csv}")
    print(f"fully packaged (parquet+csv+meta): {n_full}")
    print("✅ Snapshot completed.")

if __name__ == "__main__":
    main()
