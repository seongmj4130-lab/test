# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/combine_snapshot_outputs_to_one.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.utils.config import load_config, get_path
from src.utils.io import save_artifact

def _project_root() -> Path:
    # .../03_code/src/stages/xxx.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]

def _load_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_artifact(snapshot_dir: Path, name: str) -> pd.DataFrame:
    p_parq = snapshot_dir / f"{name}.parquet"
    p_csv = snapshot_dir / f"{name}.csv"

    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv, low_memory=False)
    raise FileNotFoundError(f"Missing data for artifact='{name}' in snapshot_dir={snapshot_dir}")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return out

def _get_snapshots_dir(cfg: dict, root: Path, snapshots_dir_arg: str = "") -> Path:
    """
    우선순위:
    1) --snapshots-dir 인자
    2) config의 paths.data_snapshots
    3) config의 paths.base_dir / data / snapshots
    4) project root / data / snapshots
    """
    if snapshots_dir_arg and snapshots_dir_arg.strip():
        return Path(snapshots_dir_arg).expanduser().resolve()

    # 2) config에 정의된 경우
    try:
        return get_path(cfg, "data_snapshots")
    except KeyError:
        pass

    # 3) base_dir 기반 폴백
    try:
        base_dir = get_path(cfg, "base_dir")
    except KeyError:
        base_dir = root

    cand = base_dir / "data" / "snapshots"
    return cand

def main():
    parser = argparse.ArgumentParser(description="Combine snapshot outputs into ONE table (parquet + csv).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--tag", type=str, required=True, help="snapshot tag folder name (e.g., baseline_after_L7BCD)")
    parser.add_argument("--out-name", type=str, default="", help="base output name (no extension). default=combined__<tag>")
    parser.add_argument("--out-dir", type=str, default="", help="optional override output directory")
    parser.add_argument("--snapshots-dir", type=str, default="", help="optional override snapshots base dir")
    parser.add_argument("--include-meta-cols", action="store_true", help="attach meta.stage/meta.run_id as columns")
    args = parser.parse_args()

    root = _project_root()
    cfg_path = (root / args.config).resolve()
    cfg = load_config(str(cfg_path))

    snapshots_dir = _get_snapshots_dir(cfg, root, args.snapshots_dir)
    snapshot_dir = snapshots_dir / args.tag
    if not snapshot_dir.exists():
        # 마지막 폴백: ROOT/data/snapshots/<tag>
        alt = root / "data" / "snapshots" / args.tag
        if alt.exists():
            snapshot_dir = alt
        else:
            raise FileNotFoundError(f"Snapshot folder not found: {snapshot_dir} (also tried: {alt})")

    out_name = args.out_name.strip() or f"combined__{args.tag}"
    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = snapshot_dir  # snapshot 폴더 안에 저장(가장 안전)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== COMBINE SNAPSHOT OUTPUTS ===")
    print(f"ROOT       : {root}")
    print(f"CFG        : {cfg_path}")
    print(f"SNAPSHOT   : {snapshot_dir}")
    print(f"OUT_DIR    : {out_dir}")
    print(f"OUT_NAME   : {out_name}")
    print(f"include_meta_cols: {bool(args.include_meta_cols)}")

    meta_files = sorted(snapshot_dir.glob("*__meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta files found in snapshot: {snapshot_dir}")

    dfs: List[pd.DataFrame] = []
    manifest_rows: List[dict] = []

    for mp in meta_files:
        name = mp.name.replace("__meta.json", "")
        meta = _load_meta(mp)

        df = _read_artifact(snapshot_dir, name)
        df = _normalize_df(df)

        df.insert(0, "__artifact", name)
        df.insert(1, "__snapshot_tag", args.tag)

        if args.include_meta_cols:
            df.insert(2, "__meta_stage", meta.get("stage", None))
            df.insert(3, "__meta_run_id", meta.get("run_id", None))

        dfs.append(df)

        manifest_rows.append({
            "artifact": name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "has_date": bool("date" in df.columns),
            "meta_stage": meta.get("stage", None),
            "meta_run_id": meta.get("run_id", None),
        })

        print(f"- loaded: {name:30s} shape={df.shape}")

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows).sort_values(["artifact"]).reset_index(drop=True)

    out_base = out_dir / out_name
    save_artifact(combined, out_base, force=True, formats=["parquet", "csv"])

    man_base = out_dir / f"{out_name}__manifest"
    save_artifact(manifest, man_base, force=True, formats=["parquet", "csv"])

    print("\n✅ DONE")
    print(f"- combined saved: {out_base}.parquet / {out_base}.csv")
    print(f"- manifest saved: {man_base}.parquet / {man_base}.csv")
    print(f"- combined shape: {combined.shape}")
    print(f"- manifest shape: {manifest.shape}")

if __name__ == "__main__":
    main()
