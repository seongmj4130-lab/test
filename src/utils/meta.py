# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/utils/meta.py
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

KST = timezone(timedelta(hours=9))


def _now_kst_iso() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S%z")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _missing_topn(df: pd.DataFrame, n: int = 20) -> dict[str, float]:
    if df.empty:
        return {}
    miss = (df.isna().mean() * 100.0).sort_values(ascending=False).head(n)
    return {k: round(float(v), 4) for k, v in miss.items()}


def _df_signature(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("date", "ticker"),
) -> dict[str, Any]:
    sig: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
    }

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        sig["date_min"] = None if d.isna().all() else str(d.min().date())
        sig["date_max"] = None if d.isna().all() else str(d.max().date())

    if "ticker" in df.columns:
        sig["ticker_nunique"] = int(pd.Series(df["ticker"]).nunique(dropna=True))

    if all(c in df.columns for c in key_cols):
        dup = int(df.duplicated(subset=list(key_cols)).sum())
        sig["key_dup_count"] = dup
    else:
        sig["key_dup_count"] = None

    return sig


def get_git_commit(repo_dir: Union[str, Path]) -> Optional[str]:
    """가능하면 git commit을 meta에 남김(없어도 OK)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def build_meta(
    *,
    stage: str,
    run_id: str,
    df: pd.DataFrame,
    out_base_path: Union[str, Path],
    inputs: Optional[dict[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
    schema_version: Optional[str] = None,
    repo_dir: Optional[Union[str, Path]] = None,
    extra: Optional[dict[str, Any]] = None,
    quality: dict | None = None,  # ✅ 추가
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "stage": stage,
        "run_id": run_id,
        "created_at_kst": _now_kst_iso(),
        "out_base": str(Path(out_base_path)),
        "data_signature": _df_signature(df),
        "missing_top20_pct": _missing_topn(df, 20),
        "inputs": inputs or {},
        "warnings": list(warnings or []),
        "schema_version": schema_version,
    }
    if quality is not None:  # ✅ 추가
        meta["quality"] = quality  # ✅ 추가

    if repo_dir is not None:
        meta["git_commit"] = get_git_commit(repo_dir)

    if extra:
        meta["extra"] = extra

    return meta


def save_meta(
    out_base_path: Union[str, Path], meta: dict[str, Any], *, force: bool = True
) -> Path:
    out_base = Path(out_base_path)
    meta_path = out_base.parent / f"{out_base.name}__meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if meta_path.exists() and (not force):
        raise FileExistsError(f"Already exists: {meta_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta_path
