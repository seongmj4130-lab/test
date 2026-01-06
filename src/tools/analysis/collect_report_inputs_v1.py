# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/collect_report_inputs_v1.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import pandas as pd

try:
    from src.utils.config import load_config
except Exception:
    load_config = None

def _print_block(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def _project_root_from_this_file() -> Path:
    # .../03_code/src/stages/collect_report_inputs_v1.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]

def _get_nested(cfg: dict, key_path: str, default=None):
    cur: Any = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _first_non_none(cfg: dict, paths: List[str], default=None):
    for p in paths:
        v = _get_nested(cfg, p, None)
        if v is not None:
            return v
    return default

def _safe_cast_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    ArrowTypeError 방지:
    - object dtype 컬럼은 전부 string으로 변환 (숫자/None/문자 혼합 대비)
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("string")
    return out

def _safe_save_one_table(df: pd.DataFrame, out_base: Path, formats: List[str]):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out = _safe_cast_for_parquet(df)

    if "parquet" in formats:
        out.to_parquet(str(out_base) + ".parquet", index=False)
    if "csv" in formats:
        out.to_csv(str(out_base) + ".csv", index=False, encoding="utf-8-sig")

def _unique_name(base: str, existing: set[str]) -> str:
    name = base
    i = 1
    while name in existing:
        name = f"{base}_{i}"
        i += 1
    existing.add(name)
    return name

def _to_kv(
    df: pd.DataFrame,
    *,
    section: str,
    entity_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    여러 요약 테이블을 '단일 파일'로 합치기 위한 KV(long-form) 변환

    반환 스키마:
      section, entity, metric, value, value_num

    ✅ pandas.melt 충돌 방지:
    - 원본 df에 'value' / 'metric' 컬럼이 이미 있어도 동작하도록
      var_name/value_name을 충돌 없는 임시 이름으로 생성한다.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["section", "entity", "metric", "value", "value_num"])

    dfx = df.copy()

    # entity 구성
    if entity_cols:
        use = [c for c in entity_cols if c in dfx.columns]
        if use:
            ent = []
            for _, r in dfx[use].iterrows():
                parts = []
                for c in use:
                    v = r[c]
                    if pd.isna(v):
                        continue
                    parts.append(f"{c}={v}")
                ent.append("|".join(parts) if parts else "row")
        else:
            ent = [f"row_{i}" for i in range(len(dfx))]
    else:
        ent = [f"row_{i}" for i in range(len(dfx))]

    existing = set(map(str, dfx.columns.tolist()))
    entity_tmp = _unique_name("__entity__", existing)
    metric_tmp = _unique_name("__metric__", existing)
    value_tmp = _unique_name("__value__", existing)

    dfx[entity_tmp] = ent

    id_vars = [entity_tmp]
    value_vars = [c for c in dfx.columns if c not in id_vars]

    long = dfx.melt(id_vars=id_vars, value_vars=value_vars, var_name=metric_tmp, value_name=value_tmp)

    long = long.rename(
        columns={
            entity_tmp: "entity",
            metric_tmp: "metric",
            value_tmp: "value",
        }
    )
    long.insert(0, "section", section)

    # value: 무조건 string, value_num: 숫자 파생(가능하면)
    long["value"] = long["value"].astype("string")
    long["value_num"] = pd.to_numeric(long["value"], errors="coerce")

    return long[["section", "entity", "metric", "value", "value_num"]]

def _load_combined(snapshot_dir: Path, tag: str) -> pd.DataFrame:
    p_parq = snapshot_dir / f"combined__{tag}.parquet"
    p_csv = snapshot_dir / f"combined__{tag}.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv, low_memory=False)
    raise FileNotFoundError(f"combined__{tag}.parquet/csv not found in: {snapshot_dir}")

def _pick_artifact_df(combined: pd.DataFrame, artifact_name: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if "__artifact" not in combined.columns:
        return pd.DataFrame()
    sub = combined.loc[combined["__artifact"].astype(str) == str(artifact_name)].copy()
    if cols:
        keep = [c for c in cols if c in sub.columns]
        if keep:
            sub = sub[keep].copy()
    return sub

def _spearman_ic_one(g: pd.DataFrame, y_true: str, y_pred: str) -> float:
    x = g[[y_true, y_pred]].dropna()
    if len(x) < 2:
        return np.nan
    r1 = x[y_true].rank(method="average")
    r2 = x[y_pred].rank(method="average")
    v = np.corrcoef(r1.to_numpy(), r2.to_numpy())[0, 1]
    return float(v) if np.isfinite(v) else np.nan

def _top_bottom_spread_one(g: pd.DataFrame, y_true: str, y_pred: str, q: float = 0.2) -> float:
    x = g[[y_true, y_pred]].dropna()
    if len(x) < 10:
        return np.nan
    p = pd.to_numeric(x[y_pred], errors="coerce").to_numpy()
    if np.all(np.isnan(p)):
        return np.nan
    hi = np.nanquantile(p, 1.0 - q)
    lo = np.nanquantile(p, q)
    top = pd.to_numeric(x.loc[pd.to_numeric(x[y_pred], errors="coerce") >= hi, y_true], errors="coerce").mean()
    bot = pd.to_numeric(x.loc[pd.to_numeric(x[y_pred], errors="coerce") <= lo, y_true], errors="coerce").mean()
    return float(top - bot) if np.isfinite(top) and np.isfinite(bot) else np.nan

def _ic_spread_summary(pred: pd.DataFrame, name: str) -> pd.DataFrame:
    need = {"date", "phase", "y_true", "y_pred"}
    if pred is None or pred.empty or not need.issubset(set(pred.columns)):
        return pd.DataFrame()

    df = pred[["date", "phase", "y_true", "y_pred"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["phase"] = df["phase"].astype(str)

    rows = []
    for (ph, dt), g in df.groupby(["phase", "date"], sort=True):
        ic = _spearman_ic_one(g, "y_true", "y_pred")
        sp = _top_bottom_spread_one(g, "y_true", "y_pred", q=0.2)
        rows.append({"artifact": name, "phase": ph, "date": dt, "ic_rank": ic, "spread_q20": sp})

    daily = pd.DataFrame(rows)

    out = []
    for ph, g in daily.groupby("phase", sort=False):
        ic = g["ic_rank"].astype(float)
        sp = g["spread_q20"].astype(float)

        ic_mean = float(np.nanmean(ic.to_numpy())) if len(ic) else np.nan
        ic_std = float(np.nanstd(ic.to_numpy(), ddof=1)) if (ic.notna().sum() >= 2) else np.nan
        ic_ir = float(ic_mean / (ic_std + 1e-12)) if np.isfinite(ic_std) else np.nan
        ic_pos = float((ic > 0).mean()) if len(ic) else np.nan

        sp_mean = float(np.nanmean(sp.to_numpy())) if len(sp) else np.nan

        out.append(
            {
                "artifact": name,
                "phase": ph,
                "n_dates": int(g["date"].nunique()),
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
                "ic_pos_ratio": ic_pos,
                "spread_mean_q20": sp_mean,
            }
        )

    return pd.DataFrame(out).sort_values(["artifact", "phase"]).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="REPORT PACK COLLECTOR (one file)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--formats", type=str, default="parquet,csv")
    args = parser.parse_args()

    root = _project_root_from_this_file()
    cfg_path = (root / args.config).resolve()
    snapshot_dir = root / "data" / "snapshots" / args.tag
    out_dir = snapshot_dir / "report_pack"
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]

    _print_block("🚩 REPORT PACK COLLECTOR (1차 보고서 입력 수집 | ONE FILE)")
    print(f"ROOT       : {root}")
    print(f"CFG        : {cfg_path}")
    print(f"SNAPSHOT   : {snapshot_dir}")
    print(f"OUT_DIR    : {out_dir}")
    print(f"FORMATS    : {formats}")

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot folder not found: {snapshot_dir}")

    # -------------------------------------------------------------------------
    # [0] CONFIG SUMMARY  (✅ params.* 우선으로 읽도록 수정)
    # -------------------------------------------------------------------------
    _print_block("[0] CONFIG SUMMARY")

    cfg = {}
    if load_config is not None and cfg_path.exists():
        try:
            cfg = load_config(str(cfg_path))
        except Exception:
            cfg = {}

    # key: 출력용 라벨, paths: 탐색 경로 우선순위
    cfg_map = [
        ("params.start_date", ["params.start_date"]),
        ("params.end_date", ["params.end_date"]),

        ("params.holdout_years", ["params.holdout_years", "l4.holdout_years"]),
        ("params.step_days", ["params.step_days", "l4.step_days"]),
        ("params.test_window_days", ["params.test_window_days", "l4.test_window_days"]),
        ("params.embargo_days", ["params.embargo_days", "l4.embargo_days"]),

        ("params.horizon_short", ["params.horizon_short", "l4.horizon_short"]),
        ("params.horizon_long", ["params.horizon_long", "l4.horizon_long"]),

        ("params.rolling_train_years_short", ["params.rolling_train_years_short", "l4.rolling_train_years_short"]),
        ("params.rolling_train_years_long", ["params.rolling_train_years_long", "l4.rolling_train_years_long"]),

        ("params.l6.weight_short", ["params.l6.weight_short", "l6.weight_short"]),
        ("params.l6.weight_long", ["params.l6.weight_long", "l6.weight_long"]),

        # l7는 네 YAML이 params.l6.l7로 들어가있어서 둘 다 탐색
        ("params.l7.holding_days", ["params.l7.holding_days", "params.l6.l7.holding_days", "l7.holding_days"]),
        ("params.l7.top_k", ["params.l7.top_k", "params.l6.l7.top_k", "l7.top_k"]),
        ("params.l7.cost_bps", ["params.l7.cost_bps", "params.l6.l7.cost_bps", "l7.cost_bps"]),
        ("params.l7.weighting", ["params.l7.weighting", "params.l6.l7.weighting", "l7.weighting"]),
        ("params.l7.softmax_temp", ["params.l7.softmax_temp", "params.l6.l7.softmax_temp", "l7.softmax_temp"]),
        ("params.l7.score_col", ["params.l7.score_col", "params.l6.l7.score_col", "l7.score_col"]),
        ("params.l7.return_col", ["params.l7.return_col", "params.l6.l7.return_col", "l7.return_col"]),
    ]

    cfg_rows = [{"key": k, "value": _first_non_none(cfg, paths, None)} for k, paths in cfg_map]
    cfg_df = pd.DataFrame(cfg_rows, columns=["key", "value"])
    print(cfg_df.to_string(index=False) if len(cfg_df) else "(config empty)")

    # -------------------------------------------------------------------------
    # [1] LOAD COMBINED
    # -------------------------------------------------------------------------
    _print_block("[1] LOAD COMBINED SNAPSHOT")
    combined = _load_combined(snapshot_dir, args.tag)
    print(f"✅ combined loaded: shape={combined.shape}")
    if "__artifact" not in combined.columns:
        raise KeyError("combined file must contain '__artifact' column")

    inv = (
        combined["__artifact"]
        .astype("string")
        .fillna("NA")
        .value_counts(dropna=False)
        .rename_axis("artifact")
        .reset_index(name="rows")
        .sort_values("artifact")
        .reset_index(drop=True)
    )
    print(f"- artifacts: {len(inv)}")
    print(inv.head(30).to_string(index=False))

    # -------------------------------------------------------------------------
    # [2] BUILD SUMMARIES
    # -------------------------------------------------------------------------
    _print_block("[2] BUILD SUMMARIES")

    # Universe monthly
    univ = _pick_artifact_df(combined, "universe_k200_membership_monthly", cols=["date", "ticker"])
    univ_monthly = pd.DataFrame()
    univ_summary = pd.DataFrame()
    if not univ.empty and {"date", "ticker"}.issubset(univ.columns):
        u = univ.copy()
        u["date"] = pd.to_datetime(u["date"], errors="coerce")
        u = u.dropna(subset=["date"])
        u["ticker"] = u["ticker"].astype(str).str.zfill(6)
        u["ym"] = u["date"].dt.to_period("M").astype(str)
        univ_monthly = u.groupby("ym", sort=True)["ticker"].nunique().rename("universe_n_tickers").reset_index()
        univ_summary = pd.DataFrame([{
            "months": int(len(univ_monthly)),
            "ym_min": univ_monthly["ym"].min() if len(univ_monthly) else None,
            "ym_max": univ_monthly["ym"].max() if len(univ_monthly) else None,
            "tickers_mean": float(univ_monthly["universe_n_tickers"].mean()) if len(univ_monthly) else np.nan,
            "tickers_min": int(univ_monthly["universe_n_tickers"].min()) if len(univ_monthly) else np.nan,
            "tickers_max": int(univ_monthly["universe_n_tickers"].max()) if len(univ_monthly) else np.nan,
        }])

    def cv_sum(name: str) -> pd.DataFrame:
        df = _pick_artifact_df(combined, name)
        if df.empty:
            return pd.DataFrame([{"artifact": name, "status": "MISSING"}])
        cols = [c for c in ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"] if c in df.columns]
        d = df[cols].copy() if cols else df.copy()
        for c in ["train_start", "train_end", "test_start", "test_end"]:
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        segs = ",".join(sorted(d["segment"].astype(str).dropna().unique().tolist())) if "segment" in d.columns else "NA"
        return pd.DataFrame([{
            "artifact": name,
            "status": "OK",
            "n_folds": int(len(d)),
            "segments": segs,
            "train_start_min": d["train_start"].min() if "train_start" in d.columns else None,
            "test_end_max": d["test_end"].max() if "test_end" in d.columns else None,
        }])

    cv_summary = pd.concat([cv_sum("cv_folds_short"), cv_sum("cv_folds_long")], ignore_index=True)

    # Model metrics summary
    mm = _pick_artifact_df(combined, "model_metrics")
    mm_sum = pd.DataFrame()
    if not mm.empty:
        keep = [c for c in ["horizon", "phase", "model", "rmse", "mae", "ic_rank", "hit_ratio", "n_train", "n_test", "n_features"] if c in mm.columns]
        if keep:
            m = mm[keep].copy()
            for c in ["rmse", "mae", "ic_rank", "hit_ratio", "n_train", "n_test", "n_features"]:
                if c in m.columns:
                    m[c] = pd.to_numeric(m[c], errors="coerce")
            gcols = [c for c in ["horizon", "phase", "model"] if c in m.columns]
            agg = {c: "mean" for c in ["rmse", "mae", "ic_rank", "hit_ratio", "n_train", "n_test", "n_features"] if c in m.columns}
            if gcols and agg:
                mm_sum = m.groupby(gcols, sort=False).agg(agg).reset_index()

    # IC / Spread summary (pred_short_oos, pred_long_oos)
    pred_s = _pick_artifact_df(combined, "pred_short_oos", cols=["date", "phase", "y_true", "y_pred"])
    pred_l = _pick_artifact_df(combined, "pred_long_oos", cols=["date", "phase", "y_true", "y_pred"])
    ic_sum = pd.concat(
        [_ic_spread_summary(pred_s, "pred_short_oos"), _ic_spread_summary(pred_l, "pred_long_oos")],
        ignore_index=True
    )

    # 기타 성과 테이블(있으면 포함)
    bt_metrics = _pick_artifact_df(combined, "bt_metrics")
    bench_metrics = _pick_artifact_df(combined, "bt_benchmark_compare")  # 네 snapshot에 이름이 이거로 보임
    roll_sharpe = _pick_artifact_df(combined, "bt_rolling_sharpe")
    dd_events = _pick_artifact_df(combined, "bt_drawdown_events")
    yearly = _pick_artifact_df(combined, "bt_yearly_metrics")

    roll_stats = pd.DataFrame()
    if not roll_sharpe.empty:
        num_cols = [c for c in ["net_rolling_n", "net_rolling_mean", "net_rolling_vol_ann", "net_rolling_sharpe"] if c in roll_sharpe.columns]
        if num_cols:
            desc = roll_sharpe[num_cols].apply(pd.to_numeric, errors="coerce").describe().T[["mean", "min", "max"]]
            roll_stats = desc.reset_index().rename(columns={"index": "metric"})  # ✅ 'metric' 있어도 _to_kv에서 충돌 회피

    dd_top10 = pd.DataFrame()
    if not dd_events.empty and "drawdown" in dd_events.columns:
        d = dd_events.copy()
        d["drawdown"] = pd.to_numeric(d["drawdown"], errors="coerce")
        dd_top10 = d.sort_values("drawdown").head(10)

    # -------------------------------------------------------------------------
    # [3] MERGE INTO ONE TABLE (KV)
    # -------------------------------------------------------------------------
    _print_block("[3] MERGE INTO ONE TABLE")

    parts = []
    parts.append(_to_kv(cfg_df, section="config", entity_cols=["key"]))
    parts.append(_to_kv(inv, section="inventory", entity_cols=["artifact"]))
    parts.append(_to_kv(univ_summary, section="universe_summary"))
    parts.append(_to_kv(univ_monthly, section="universe_monthly_counts", entity_cols=["ym"]))
    parts.append(_to_kv(cv_summary, section="cv_summary", entity_cols=["artifact"]))
    parts.append(_to_kv(mm_sum, section="model_metrics_summary", entity_cols=[c for c in ["horizon", "phase", "model"] if c in mm_sum.columns]))
    parts.append(_to_kv(ic_sum, section="ic_spread_summary", entity_cols=[c for c in ["artifact", "phase"] if c in ic_sum.columns]))
    parts.append(_to_kv(bt_metrics, section="bt_metrics", entity_cols=[c for c in ["phase"] if c in bt_metrics.columns]))
    parts.append(_to_kv(bench_metrics, section="benchmark_metrics", entity_cols=[c for c in ["phase"] if c in bench_metrics.columns]))
    parts.append(_to_kv(roll_stats, section="rolling_sharpe_stats", entity_cols=["metric"] if "metric" in roll_stats.columns else None))
    parts.append(_to_kv(dd_top10, section="drawdown_top10", entity_cols=[c for c in ["phase", "peak_date", "trough_date"] if c in dd_top10.columns]))
    parts.append(_to_kv(yearly, section="yearly_metrics", entity_cols=[c for c in ["phase", "year"] if c in yearly.columns]))

    all_kv = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    all_kv = all_kv.sort_values(["section", "entity", "metric"], kind="mergesort").reset_index(drop=True)

    print(f"- total rows (kv): {len(all_kv):,}")
    print("- sections:")
    print(all_kv["section"].value_counts().sort_index().to_string())

    for sec in ["ic_spread_summary", "bt_metrics", "benchmark_metrics"]:
        ss = all_kv.loc[all_kv["section"] == sec].head(30)
        if len(ss):
            print("\n" + f"[preview] {sec}")
            print(ss.to_string(index=False))

    # -------------------------------------------------------------------------
    # [4] SAVE (ONE FILE)
    # -------------------------------------------------------------------------
    _print_block("[4] SAVE (ONE FILE)")
    out_base = out_dir / f"reportpack__{args.tag}__all"
    _safe_save_one_table(all_kv, out_base, formats=formats)
    print(f"✅ Saved:\n- {out_base}.parquet\n- {out_base}.csv")

if __name__ == "__main__":
    main()
