# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/validate_l7b_l7c_l7d_outputs.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _artifact_base(interim: Path, name: str) -> Path:
    return interim / name


def _fail(msg: str) -> None:
    raise SystemExit(msg)


def _check_numeric_finite(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if (~np.isfinite(s.to_numpy())).any():
            _fail(f"[FAIL] {label}: non-finite values found in numeric col '{c}'")


def _check_dupes(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if not all(k in df.columns for k in keys):
        _fail(f"[FAIL] {label}: duplicate key columns not found: {keys}")
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(10)
        _fail(f"[FAIL] {label}: duplicates on keys={keys}. sample:\n{sample}")


def _infer_rolling_keys(df: pd.DataFrame) -> list[str]:
    """
    bt_rolling_sharpe 같은 다차원 결과는 (date, phase)만으로 유니크가 아닐 수 있으므로,
    window/series 계열 컬럼을 찾아 유니크 키에 포함한다.
    """
    base = ["date", "phase"]

    candidates = [
        "window_days",
        "window",
        "lookback_days",
        "lookback",
        "rolling_window",
        "window_months",
        "series",
        "kind",
        "metric",
        "return_col",
        "return_col_used",
        "net_return_col_used",
    ]

    extras = [c for c in candidates if c in df.columns]

    # 최소 1개 이상은 있어야 (date,phase) 중복이 정당화됨.
    if len(extras) == 0:
        return base

    # extras 전부 포함(보수적으로 유니크 보장)
    return base + extras


def main():
    print("=== L7B/L7C/L7D Validation Runner ===")
    root = _root()
    cfg_path = _cfg_path(root)

    print("ROOT  :", root)
    print("CFG   :", cfg_path)

    if not cfg_path.exists():
        _fail(f"[FAIL] config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    meta_files = sorted(interim.glob("*__meta.json"))
    if not meta_files:
        _fail(f"[FAIL] no meta files found in: {interim}")

    discovered = []
    for mp in meta_files:
        m = _load_meta(mp)
        stage = str(m.get("stage", ""))
        if (
            stage.startswith("L7B:")
            or stage.startswith("L7C:")
            or stage.startswith("L7D:")
        ):
            out_name = mp.name.replace("__meta.json", "")
            discovered.append((out_name, stage, mp))

    if not discovered:
        _fail("[FAIL] no L7B/L7C/L7D meta found. did you run those stages?")

    print("\n=== Discovered outputs (from meta.stage) ===")
    for out_name, stage, mp in discovered:
        print(f"- {stage} -> {out_name} (meta={mp.name})")

    # L7D yearly metrics는 스키마를 엄격히 고정
    strict_required = {
        "bt_yearly_metrics": [
            "phase",
            "year",
            "n_rebalances",
            "net_total_return",
            "net_vol_ann",
            "net_sharpe",
            "net_mdd",
            "net_hit_ratio",
            "date_start",
            "date_end",
            "net_return_col_used",
        ]
    }

    print("\n=== Artifact existence & schema checks ===")
    for out_name, stage, mp in discovered:
        base = _artifact_base(interim, out_name)
        if not artifact_exists(base):
            _fail(f"[FAIL] artifact missing for {stage}: {base}")

        df = load_artifact(base)
        if not isinstance(df, pd.DataFrame):
            _fail(f"[FAIL] {stage}:{out_name} is not a DataFrame")
        if df.shape[0] == 0:
            _fail(f"[FAIL] {stage}:{out_name} has 0 rows")

        # strict schema check
        if out_name in strict_required:
            need = strict_required[out_name]
            miss = [c for c in need if c not in df.columns]
            if miss:
                _fail(
                    f"[FAIL] {stage}:{out_name} missing required cols: {miss}\n"
                    f"cols={list(df.columns)}"
                )

        # date parsing sanity
        for dc in ["date", "date_start", "date_end"]:
            if dc in df.columns:
                d = pd.to_datetime(df[dc], errors="coerce")
                if d.isna().any():
                    _fail(
                        f"[FAIL] {stage}:{out_name} has invalid '{dc}' values (NaT present)"
                    )

        # duplicate checks (output별로 키를 다르게 적용)
        if out_name == "bt_rolling_sharpe":
            keys = _infer_rolling_keys(df)
            if keys == ["date", "phase"]:
                # window/series 계열 컬럼이 없는데 (date,phase) 중복이면 진짜 문제로 취급
                _check_dupes(df, ["date", "phase"], f"{stage}:{out_name}")
            else:
                _check_dupes(df, keys, f"{stage}:{out_name}")

        else:
            # 일반 규칙: 가능하면 가장 세밀한 키로 검증
            if all(c in df.columns for c in ["date", "ticker", "phase"]):
                _check_dupes(df, ["date", "ticker", "phase"], f"{stage}:{out_name}")
            elif all(c in df.columns for c in ["date", "phase"]):
                _check_dupes(df, ["date", "phase"], f"{stage}:{out_name}")
            elif all(c in df.columns for c in ["phase", "year"]):
                _check_dupes(df, ["phase", "year"], f"{stage}:{out_name}")

        # numeric finite check
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        _check_numeric_finite(df, numeric_cols, f"{stage}:{out_name}")

        print(f"[PASS] {stage}:{out_name} shape={df.shape}")

    if artifact_exists(interim / "bt_yearly_metrics"):
        y = load_artifact(interim / "bt_yearly_metrics")
        years = sorted(pd.Series(y["year"]).dropna().astype(int).unique().tolist())
        print("\n=== Yearly metrics summary ===")
        print("years:", years)
        print("phases:", sorted(pd.Series(y["phase"]).astype(str).unique().tolist()))

    print("\n✅ L7B/L7C/L7D VALIDATION COMPLETE: All critical checks passed.")
    print(
        "➡️ Next: run full audit (L0~L7 + extensions) and then final reporting tables."
    )


if __name__ == "__main__":
    main()
