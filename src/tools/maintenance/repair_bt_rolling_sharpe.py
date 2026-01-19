# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/maintenance/repair_bt_rolling_sharpe.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.meta import build_meta, save_meta


def _root() -> Path:
    # .../03_code/src/stages/repair_bt_rolling_sharpe.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]

def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"

def _infer_keys(df: pd.DataFrame) -> list[str]:
    # validator가 현재 date/phase/net_return_col_used 까지 잡고 있으므로 동일하게 사용
    keys = ["date", "phase"]
    if "net_return_col_used" in df.columns:
        keys.append("net_return_col_used")

    # 만약 실제로 window/series 컬럼이 존재한다면 키에 포함(있으면 더 안전)
    extra_candidates = [
        "window_days", "window", "lookback_days", "lookback",
        "rolling_window", "series", "kind", "metric", "return_col"
    ]
    for c in extra_candidates:
        if c in df.columns and c not in keys:
            keys.append(c)

    return keys

def _check_conflicting_duplicates(df: pd.DataFrame, keys: list[str]) -> tuple[bool, pd.DataFrame]:
    """
    keys가 동일한데 다른 값(충돌)이 있는지 검사.
    - 충돌이 없으면 (True, empty)
    - 충돌이 있으면 (False, sample_df)
    """
    dup_mask = df.duplicated(keys, keep=False)
    if not dup_mask.any():
        return True, pd.DataFrame()

    ddup = df.loc[dup_mask].copy()

    non_keys = [c for c in df.columns if c not in keys]

    # 각 key 그룹에서 non-key 컬럼의 nunique가 1을 초과하면 "충돌"
    nunique = ddup.groupby(keys, dropna=False)[non_keys].nunique(dropna=False)
    conflict_groups = (nunique > 1).any(axis=1)
    if conflict_groups.any():
        bad_keys = conflict_groups[conflict_groups].index.to_frame(index=False)
        sample_keys = bad_keys.head(5)
        sample = ddup.merge(sample_keys, on=keys, how="inner").head(30)
        return False, sample

    return True, pd.DataFrame()

def main():
    print("=== REPAIR bt_rolling_sharpe (dedup) ===")
    root = _root()
    cfg_path = _cfg_path(root)

    print("ROOT :", root)
    print("CFG  :", cfg_path)

    if not cfg_path.exists():
        raise SystemExit(f"[FAIL] config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    base = interim / "bt_rolling_sharpe"
    if not artifact_exists(base):
        raise SystemExit(f"[FAIL] artifact missing: {base}")

    df = load_artifact(base)
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise SystemExit(f"[FAIL] invalid DataFrame loaded: shape={getattr(df,'shape',None)}")

    # 타입 정리
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise SystemExit("[FAIL] bt_rolling_sharpe has invalid 'date' (NaT)")
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str)

    keys = _infer_keys(df)

    # 중복 현황
    before_rows = int(df.shape[0])
    before_unique_keys = int(df[keys].drop_duplicates().shape[0])
    dup_cnt = int(df.duplicated(keys, keep=False).sum())

    print(f"keys used: {keys}")
    print(f"rows(before)={before_rows}, unique_keys(before)={before_unique_keys}, dup_rows={dup_cnt}")

    # 충돌 여부 확인 (같은 key인데 값이 다르면 여기서 FAIL)
    ok, sample = _check_conflicting_duplicates(df, keys)
    if not ok:
        print("\n[FAIL] Found conflicting duplicates (same keys, different values). Sample:")
        print(sample)
        raise SystemExit("[FAIL] cannot auto-dedup safely. Fix L7D generation logic first.")

    # 안전한 dedup: "완전히 동일한 행" 제거 → 그 다음에도 key 중복 있으면 key 기준 집계
    df2 = df.drop_duplicates().copy()

    if df2.duplicated(keys).any():
        # key가 같고 값 충돌은 없다고 확인됐으므로, key 기준으로 안전 집계 가능
        non_keys = [c for c in df2.columns if c not in keys]
        num_cols = [c for c in non_keys if pd.api.types.is_numeric_dtype(df2[c])]
        other_cols = [c for c in non_keys if c not in num_cols]

        agg = {c: "mean" for c in num_cols}
        agg.update({c: "first" for c in other_cols})

        df2 = df2.groupby(keys, as_index=False, dropna=False).agg(agg)

    # 최종 검증
    after_rows = int(df2.shape[0])
    after_unique_keys = int(df2[keys].drop_duplicates().shape[0])
    if after_rows != after_unique_keys:
        raise SystemExit(f"[FAIL] dedup failed: rows(after)={after_rows} != unique_keys(after)={after_unique_keys}")

    print(f"rows(after)={after_rows}, unique_keys(after)={after_unique_keys}")

    # 저장 (기존 bt_rolling_sharpe 덮어쓰기)
    save_formats = cfg.get("run", {}).get("save_formats", ["parquet", "csv"])
    save_artifact(df2, base, force=True, formats=save_formats)

    meta = build_meta(
        stage="L7D:bt_rolling_sharpe",
        run_id="repair_bt_rolling_sharpe_dedup",
        df=df2,
        out_base_path=base,
        warnings=[f"dedup applied: rows {before_rows} -> {after_rows} on keys={keys}"],
        inputs={"source": "bt_rolling_sharpe (existing)"},
        repo_dir=get_path(cfg, "base_dir"),
        quality={"repair": {
            "keys": keys,
            "rows_before": before_rows,
            "rows_after": after_rows,
            "dup_rows_before": dup_cnt,
        }},
    )
    save_meta(base, meta, force=True)

    print("✅ REPAIR COMPLETE: bt_rolling_sharpe dedup saved with updated meta.")

if __name__ == "__main__":
    main()
