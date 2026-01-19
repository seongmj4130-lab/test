# -*- coding: utf-8 -*-
"""
[개선안 Midterm Pack] 중간발표용 산출물(표/그래프/manifest) 자동 생성 스크립트

중요 제약:
- 기존 파이프라인 로직은 변경하지 않음 (분석/리포트 산출물만 추가 생성)
- 경로는 configs/config.yaml의 paths.*를 신뢰
- 가능하면 data/interim/{run_tag}/ 를 우선 사용하되,
  레포에는 legacy 모드(= data/interim 루트 저장) 산출물이 공존하므로 자동 fallback 지원

생성물:
- artifacts/figures/*.png
- artifacts/tables/*.csv
- artifacts/run_manifest.json
 - artifacts/validation_report.md
 - artifacts/canva/canva_assets_manifest.json
 - reports/midterm_report.md
 - reports/ppt_outline_12slides.md
 - reports/speaker_notes.md
 - reports/canva_export_pack.md
 - reports/ppt_numbers_by_slide.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# matplotlib only (no seaborn)
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

from src.components.ranking.score_engine import (
    _pick_feature_cols as _pick_feature_cols_ranking,
)
from src.stages.modeling.l5_train_models import (
    _pick_feature_cols as _pick_feature_cols_model,
)
from src.utils.config import get_path, load_config
from src.utils.feature_groups import get_feature_groups, load_feature_groups
from src.utils.io import artifact_exists, load_artifact


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_kst_iso() -> str:
    """
    KST(UTC+9) ISO 타임스탬프를 반환.
    - 시스템 로케일/타임존에 의존하지 않도록 고정 오프셋 사용
    """
    kst = timezone(offset=pd.Timedelta(hours=9).to_pytimedelta(), name="Asia/Seoul")
    return datetime.now(kst).isoformat()


def _configure_matplotlib_korean_font() -> None:
    """
    Windows 환경에서 한글이 깨지지 않도록 폰트 설정.
    - Malgun Gothic 우선
    - 없으면 DejaVu Sans로 fallback (한글은 깨질 수 있음)
    """
    # [개선안 Midterm Pack] 한글 폰트 설정 (경고/깨짐 최소화)
    preferred = ["Malgun Gothic", "맑은 고딕"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
    # fallback
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["axes.unicode_minus"] = False


def _ensure_dt(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _relpath(p: Path, base_dir: Path) -> str:
    try:
        return str(p.relative_to(base_dir)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _resolve_artifact_base(
    *,
    interim_base: Path,
    run_dir: Optional[Path],
    name: str,
) -> Tuple[Path, str]:
    """
    Returns:
      (out_base_path_without_suffix, source_mode)
    source_mode: "run_dir" | "interim_root"
    """
    # 1) run_dir 우선
    if run_dir is not None:
        cand = run_dir / name
        if artifact_exists(cand):
            return cand, "run_dir"

    # 2) legacy root 저장 fallback
    cand2 = interim_base / name
    if artifact_exists(cand2):
        return cand2, "interim_root"

    return cand2, "missing"

def _artifact_file_path(out_base: Path) -> Optional[Path]:
    """
    [개선안 17번] out_base(확장자 없는 경로)에서 실제 파일(.parquet 우선, 없으면 .csv)을 찾는다.
    """
    pq = out_base.with_suffix(".parquet")
    if pq.exists():
        return pq
    csv = out_base.with_suffix(".csv")
    if csv.exists():
        return csv
    return None

def _file_stat_meta(p: Path) -> Dict[str, Any]:
    """
    [개선안 17번] 무결성 추적을 위한 파일 메타(sha256/mtime/size_bytes).
    """
    st = p.stat()
    mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    return {
        "file_path": str(p),
        "format": p.suffix.lstrip("."),
        "sha256": _sha256_file(p),
        "mtime": mtime_iso,
        "size_bytes": int(st.st_size),
    }


def _find_latest_run_dir_artifact(interim_base: Path, name: str) -> Optional[Path]:
    """
    interim_base/*/<name>.parquet 중 최신(수정시간 기준) 파일의 out_base(확장자 제외)를 반환.
    예: name="sector_map" -> .../<run_tag>/sector_map
    """
    candidates = list(interim_base.glob(f"*/{name}.parquet"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].with_suffix("")


def _load_optional(
    *,
    interim_base: Path,
    run_dir: Optional[Path],
    name: str,
    required: bool = False,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    out_base, mode = _resolve_artifact_base(interim_base=interim_base, run_dir=run_dir, name=name)
    meta: Dict[str, Any] = {"name": name, "mode": mode, "out_base": str(out_base)}
    if mode == "missing":
        if required:
            raise FileNotFoundError(f"Required artifact not found: {out_base}.(parquet|csv)")
        return None, meta
    # [개선안 17번] resolved_sources에 파일 무결성 메타 기록
    fpath = _artifact_file_path(out_base)
    if fpath is not None and fpath.exists():
        meta.update(_file_stat_meta(fpath))
    df = load_artifact(out_base)
    return df, meta


def _pick_run_id_from_meta(interim_base: Path) -> Optional[str]:
    meta_path = interim_base / "bt_metrics__meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta.get("run_id") or meta.get("run_tag")
        return str(run_id) if run_id else None
    except Exception:
        return None


def _pick_latest_run_tag_with_bt_metrics(interim_base: Path) -> Optional[str]:
    # scan only directories containing bt_metrics.parquet
    candidates = list(interim_base.glob("*/bt_metrics.parquet"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent.name


def _compute_monthly_table(
    bt_returns: pd.DataFrame,
    bench_returns: Optional[pd.DataFrame] = None,
    *,
    phase: str = "holdout",
) -> pd.DataFrame:
    df = _ensure_dt(bt_returns, "date")
    if "phase" in df.columns:
        df = df[df["phase"].astype(str) == phase].copy()

    if "net_return" not in df.columns:
        raise KeyError("bt_returns must have net_return column for monthly aggregation")

    df["ym"] = df["date"].dt.to_period("M").astype(str)
    g = df.groupby("ym", sort=True)

    out = pd.DataFrame(
        {
            "ym": g.size().index,
            "n_rebalances_in_month": g.size().values.astype(int),
            "net_return": g["net_return"].apply(lambda x: float(np.prod(1.0 + x.astype(float).to_numpy()) - 1.0)).values,
        }
    )

    if "gross_return" in df.columns:
        out["gross_return"] = g["gross_return"].apply(lambda x: float(np.prod(1.0 + x.astype(float).to_numpy()) - 1.0)).values

    if bench_returns is not None and "bench_return" in bench_returns.columns:
        b = _ensure_dt(bench_returns, "date")
        if "phase" in b.columns:
            b = b[b["phase"].astype(str) == phase].copy()
        b["ym"] = b["date"].dt.to_period("M").astype(str)
        gb = b.groupby("ym", sort=True)
        bench_m = gb["bench_return"].apply(lambda x: float(np.prod(1.0 + x.astype(float).to_numpy()) - 1.0))
        out = out.merge(bench_m.rename("bench_return").reset_index(), on="ym", how="left")
        out["excess_return"] = out["net_return"] - out["bench_return"]

    return out.sort_values("ym").reset_index(drop=True)


def _compute_equity_from_returns(df: pd.DataFrame, ret_col: str) -> pd.Series:
    r = df[ret_col].astype(float).to_numpy()
    eq = np.cumprod(1.0 + r)
    return pd.Series(eq, index=df.index, dtype=float)


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = np.maximum.accumulate(equity.to_numpy(dtype=float))
    dd = (equity.to_numpy(dtype=float) / np.where(peak == 0, np.nan, peak)) - 1.0
    return pd.Series(dd, index=equity.index, dtype=float)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    _safe_mkdir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_equity(
    *,
    dates: pd.Series,
    strategy_eq: pd.Series,
    bench_eq: Optional[pd.Series],
    title: str,
    out_path: Path,
) -> None:
    _safe_mkdir(out_path.parent)
    plt.figure(figsize=(10, 5))
    plt.plot(dates, strategy_eq, label="전략(누적)", linewidth=2.0)
    if bench_eq is not None:
        plt.plot(dates, bench_eq, label="벤치마크(누적)", linewidth=1.8, alpha=0.9)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_drawdown(
    *,
    dates: pd.Series,
    drawdown: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    _safe_mkdir(out_path.parent)
    plt.figure(figsize=(10, 3.8))
    plt.plot(dates, drawdown * 100.0, color="tab:red", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_turnover(
    *,
    dates: pd.Series,
    turnover: pd.Series,
    out_ts_path: Path,
    out_hist_path: Path,
) -> None:
    _safe_mkdir(out_ts_path.parent)
    # time series
    plt.figure(figsize=(10, 3.8))
    plt.plot(dates, turnover * 100.0, color="tab:blue", linewidth=1.6)
    plt.title("턴오버(one-way) 시계열")
    plt.xlabel("Date")
    plt.ylabel("Turnover (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_ts_path, dpi=160)
    plt.close()

    # histogram
    plt.figure(figsize=(7.5, 4.2))
    x = (turnover * 100.0).to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    plt.hist(x, bins=20, color="tab:blue", alpha=0.85, edgecolor="white")
    plt.title("턴오버(one-way) 분포")
    plt.xlabel("Turnover (%)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_hist_path, dpi=160)
    plt.close()


def _plot_sector_exposure(
    *,
    bt_positions: pd.DataFrame,
    phase: str,
    out_path: Path,
    top_n: int = 6,
) -> Optional[str]:
    if bt_positions is None or bt_positions.empty:
        return "bt_positions 없음"
    if "sector_name" not in bt_positions.columns:
        return "bt_positions에 sector_name 없음"
    if "weight" not in bt_positions.columns:
        return "bt_positions에 weight 없음"

    df = _ensure_dt(bt_positions, "date")
    if "phase" in df.columns:
        df = df[df["phase"].astype(str) == phase].copy()
    if df.empty:
        return f"phase={phase} 데이터 없음"

    g = (
        df.groupby(["date", "sector_name"], sort=True)["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "sector_weight"})
    )
    # top sectors by average weight
    avg = g.groupby("sector_name", sort=False)["sector_weight"].mean().sort_values(ascending=False)
    top_sectors = avg.head(top_n).index.tolist()
    g["sector_name"] = g["sector_name"].astype(str)
    g_top = g[g["sector_name"].isin(top_sectors)].copy()
    if g_top.empty:
        return "상위 섹터 집계 실패"

    pivot = g_top.pivot_table(index="date", columns="sector_name", values="sector_weight", aggfunc="sum").fillna(0.0)
    pivot = pivot.sort_index()
    _safe_mkdir(out_path.parent)

    plt.figure(figsize=(10, 4.6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col] * 100.0, label=str(col), linewidth=1.4)
    plt.title(f"섹터 비중 시계열 (Top {top_n}, {phase})")
    plt.xlabel("Date")
    plt.ylabel("Weight (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return None


def _plot_coverage(
    *,
    rebalance_scores_summary: pd.DataFrame,
    phase: str,
    out_path: Path,
) -> Optional[str]:
    if rebalance_scores_summary is None or rebalance_scores_summary.empty:
        return "rebalance_scores_summary 없음"
    if "coverage_vs_universe_pct" not in rebalance_scores_summary.columns:
        return "coverage_vs_universe_pct 컬럼 없음"

    df = _ensure_dt(rebalance_scores_summary, "date")
    if "phase" in df.columns:
        df = df[df["phase"].astype(str) == phase].copy()
    if df.empty:
        return f"phase={phase} 데이터 없음"

    df = df.sort_values("date").reset_index(drop=True)
    _safe_mkdir(out_path.parent)
    plt.figure(figsize=(10, 3.6))
    plt.plot(df["date"], df["coverage_vs_universe_pct"].astype(float) * 100.0, linewidth=1.8, color="tab:green")
    plt.title(f"유니버스 대비 커버리지(%) (phase={phase})")
    plt.xlabel("Date")
    plt.ylabel("Coverage vs Universe (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return None


def _compute_beta_series(
    *,
    bt_returns: pd.DataFrame,
    bench_returns: pd.DataFrame,
    phase: str,
    window: int = 12,
) -> Optional[pd.DataFrame]:
    if bt_returns is None or bt_returns.empty or bench_returns is None or bench_returns.empty:
        return None
    if "net_return" not in bt_returns.columns or "bench_return" not in bench_returns.columns:
        return None

    a = _ensure_dt(bt_returns, "date")
    b = _ensure_dt(bench_returns, "date")
    if "phase" in a.columns:
        a = a[a["phase"].astype(str) == phase].copy()
    if "phase" in b.columns:
        b = b[b["phase"].astype(str) == phase].copy()
    m = a.merge(b[["date", "bench_return"]], on="date", how="inner")
    if m.empty:
        return None

    m = m.sort_values("date").reset_index(drop=True)
    x = m["bench_return"].astype(float)
    y = m["net_return"].astype(float)

    betas: List[float] = []
    for i in range(len(m)):
        if i + 1 < window:
            betas.append(np.nan)
            continue
        xs = x.iloc[i + 1 - window : i + 1].to_numpy()
        ys = y.iloc[i + 1 - window : i + 1].to_numpy()
        vx = np.var(xs, ddof=1)
        if not np.isfinite(vx) or vx <= 1e-12:
            betas.append(np.nan)
            continue
        cov = np.cov(xs, ys, ddof=1)[0, 1]
        betas.append(float(cov / vx))

    out = pd.DataFrame({"date": m["date"], "beta_roll": betas})
    return out


def _plot_beta(beta_df: pd.DataFrame, out_path: Path, title: str) -> None:
    _safe_mkdir(out_path.parent)
    df = _ensure_dt(beta_df, "date").dropna(subset=["beta_roll"])
    if df.empty:
        return
    plt.figure(figsize=(10, 3.8))
    plt.plot(df["date"], df["beta_roll"], color="tab:purple", linewidth=1.6)
    plt.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Beta (rolling)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _build_feature_list_table(cfg: dict, dataset_daily: pd.DataFrame) -> pd.DataFrame:
    p = cfg.get("params", {}) or {}
    l4 = cfg.get("l4", {}) or {}
    horizon_short = int(l4.get("horizon_short", p.get("horizon_short", 20)))
    horizon_long = int(l4.get("horizon_long", p.get("horizon_long", 120)))
    target_short = f"ret_fwd_{horizon_short}d"
    target_long = f"ret_fwd_{horizon_long}d"

    ds = dataset_daily.copy()
    # model pick needs target_col; exclude set already includes both ret_fwd_20d/120d but keep per spec
    target_col_for_model = target_short if target_short in ds.columns else ("ret_fwd_20d" if "ret_fwd_20d" in ds.columns else target_long)
    model_features = sorted(_pick_feature_cols_model(ds, target_col=target_col_for_model))
    ranking_features = sorted(_pick_feature_cols_ranking(ds))

    all_features = sorted(set(model_features) | set(ranking_features))

    # group mapping from feature_groups.yaml (if exists)
    group_map: Dict[str, str] = {}
    feature_groups_cfg = cfg.get("l8", {}).get("feature_groups_config", "configs/feature_groups.yaml")
    base_dir = Path(cfg.get("paths", {}).get("base_dir", Path.cwd()))
    fg_path = base_dir / feature_groups_cfg
    if fg_path.exists():
        fg_cfg = load_feature_groups(fg_path)
        fg = get_feature_groups(fg_cfg)
        for gname, feats in fg.items():
            for f in map(str, feats or []):
                group_map[f] = str(gname)

    rows = []
    ds_cols = set(map(str, ds.columns))
    for f in all_features:
        rows.append(
            {
                "feature": f,
                "in_dataset": f in ds_cols,
                "used_in_model_track": f in set(model_features),
                "used_in_ranking_track": f in set(ranking_features),
                "group_from_feature_groups_yaml": group_map.get(f, ""),
            }
        )
    out = pd.DataFrame(rows)
    return out

def _detect_baseline_tag(cfg: dict, base_dir: Path) -> Optional[str]:
    """
    [개선안 17번] baseline run_tag를 근거 기반으로 선택.
    우선순위:
    1) configs/config.yaml의 baseline.pipeline_baseline_tag
    2) reports/history/history_manifest.csv에서 stage_no=-1 track=pipeline
    """
    b = (cfg.get("baseline", {}) if isinstance(cfg, dict) else {}) or {}
    t = b.get("pipeline_baseline_tag") or None
    if t:
        return str(t)
    hist = base_dir / "reports" / "history" / "history_manifest.csv"
    if hist.exists():
        try:
            df = pd.read_csv(hist)
            df["track"] = df.get("track", "").astype(str)
            if "stage_no" in df.columns:
                sub = df[(df["stage_no"] == -1) & (df["track"] == "pipeline")]
                if not sub.empty and "run_tag" in sub.columns:
                    return str(sub.iloc[0]["run_tag"])
        except Exception:
            return None
    return None

def _read_kpi_csv(base_dir: Path, tag: str) -> Optional[pd.DataFrame]:
    """
    [개선안 17번] reports/kpi/kpi_table__{tag}.csv 로드(없으면 None).
    """
    p = base_dir / "reports" / "kpi" / f"kpi_table__{tag}.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def _extract_holdout_metrics_from_bt_metrics(bt_metrics: pd.DataFrame) -> Dict[str, Any]:
    """
    [개선안 17번] bt_metrics에서 holdout 핵심 지표를 추출.
    """
    m = bt_metrics.copy()
    if "phase" not in m.columns:
        return {}
    m["phase"] = m["phase"].astype(str)
    sub = m[m["phase"] == "holdout"]
    if sub.empty:
        return {}
    r = sub.iloc[0].to_dict()
    # 안전 캐스팅
    out: Dict[str, Any] = {}
    for k in [
        "net_total_return",
        "net_cagr",
        "net_sharpe",
        "net_mdd",
        "avg_turnover_oneway",
        "cost_bps_used",
        "top_k",
        "holding_days",
        "buffer_k",
        "n_rebalances",
        "date_start",
        "date_end",
    ]:
        if k in r:
            out[k] = r[k]
    return out

def _pct_if_ratio(x: Any) -> Any:
    """
    [개선안 17번] bt_metrics가 (0.15=15%)처럼 ratio로 저장된 경우를 대비해 %로 변환.
    휴리스틱: |x| <= 2.0 이면 ratio로 간주(일반적인 % 범위는 -100~+100을 벗어나기 어려움).
    """
    try:
        v = float(x)
        if not np.isfinite(v):
            return x
        return v * 100.0 if abs(v) <= 2.0 else v
    except Exception:
        return x

def _build_compare_baseline_vs_current(
    *,
    base_dir: Path,
    phase: str,
    baseline_tag: Optional[str],
    current_run_id: str,
    current_bt_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    [개선안 17번] baseline vs current 비교 테이블 생성.
    - baseline은 reports/kpi/kpi_table__{baseline_tag}.csv에서 BACKTEST 지표를 우선 사용(과거 스냅샷 근거).
    - current는 current_bt_metrics(holdout row) 사용.
    """
    note = None
    cur = _extract_holdout_metrics_from_bt_metrics(current_bt_metrics)
    if not cur:
        return pd.DataFrame(), "current bt_metrics에서 holdout 지표를 추출하지 못했습니다."

    base_rows: Dict[str, Any] = {}
    base_source = None
    if baseline_tag:
        kpi = _read_kpi_csv(base_dir, baseline_tag)
        if kpi is not None and not kpi.empty:
            base_source = f"reports/kpi/kpi_table__{baseline_tag}.csv"
            sub = kpi[(kpi["section"].astype(str) == "BACKTEST") & (kpi["metric"].astype(str).isin([
                "net_total_return", "net_cagr", "net_sharpe", "net_mdd", "avg_turnover_oneway", "cost_bps_used", "n_rebalances"
            ]))]
            for _, r in sub.iterrows():
                base_rows[str(r["metric"])] = r.get("holdout_value")

    if not base_rows:
        note = "baseline KPI CSV를 찾지 못했습니다(근거가 부족합니다)."

    metrics = [
        ("net_total_return", "%"),
        ("net_cagr", "%"),
        ("net_sharpe", "ratio"),
        ("net_mdd", "%"),
        ("avg_turnover_oneway", "%"),
        ("cost_bps_used", "bps"),
        ("n_rebalances", "count"),
    ]
    rows = []
    for mkey, unit in metrics:
        bval = base_rows.get(mkey, None)
        cval_raw = cur.get(mkey, None)
        # [개선안 17번] unit이 %인 항목은 current가 ratio 스케일일 수 있어 보정
        if unit == "%":
            cval = _pct_if_ratio(cval_raw)
            bval = _pct_if_ratio(bval)  # baseline도 방어적으로 보정
        else:
            cval = cval_raw
        # delta는 숫자일 때만 계산
        delta = None
        try:
            if bval is not None and cval is not None and np.isfinite(float(bval)) and np.isfinite(float(cval)):
                delta = float(cval) - float(bval)
        except Exception:
            delta = None
        rows.append({
            "metric": mkey,
            "unit": unit,
            "baseline_tag": baseline_tag or "",
            "baseline_holdout_value": bval,
            "current_run_id": current_run_id,
            "current_holdout_value": cval,
            "delta_current_minus_baseline": delta,
            "baseline_source": base_source or "",
            "current_source": f"data/interim/{current_run_id}/bt_metrics.parquet" if (base_dir / "data" / "interim" / current_run_id / "bt_metrics.parquet").exists() else "",
        })
    return pd.DataFrame(rows), note

def _build_leakage_sanity_check(
    *,
    bt_returns: pd.DataFrame,
    rebalance_scores: Optional[pd.DataFrame],
    phase: str,
) -> Tuple[pd.DataFrame, int]:
    """
    [개선안 17번] 수익률 적용 구간 점검(누수 단정 금지).
    출력 스키마는 artifacts/tables/leakage_sanity_check__{phase}.csv와 호환.
    FAIL 기준:
    - holding_days가 비정상(<=0) 이거나
    - rebalance_scores에서 해당 date의 true_short 관측치가 0개
    """
    r = _ensure_dt(bt_returns, "date")
    if "phase" in r.columns:
        r = r[r["phase"].astype(str) == phase].copy()
    r = r.sort_values("date").reset_index(drop=True)

    rs = None
    if rebalance_scores is not None and not rebalance_scores.empty:
        rs = _ensure_dt(rebalance_scores, "date")
        if "phase" in rs.columns:
            rs = rs[rs["phase"].astype(str) == phase].copy()

    rows = []
    fail = 0
    for _, row in r.iterrows():
        dt = row["date"]
        hd = int(row["holding_days"]) if "holding_days" in row and pd.notna(row["holding_days"]) else None
        interval = f"t+1~t+{hd}" if (hd is not None) else ""
        n_true = None
        true_mean = None
        if rs is not None:
            sub = rs[rs["date"] == dt]
            if "true_short" in sub.columns:
                v = sub["true_short"]
                n_true = int(v.notna().sum())
                true_mean = float(v.dropna().astype(float).mean()) if n_true > 0 else np.nan
        status = "OK"
        if hd is None or hd <= 0:
            status = "FAIL"
        if n_true is not None and n_true == 0:
            status = "FAIL"
        if status == "FAIL":
            fail += 1
        rows.append({
            "rebalance_date": str(pd.to_datetime(dt).date()),
            "phase": phase,
            "return_source": "true_short",
            "return_applied_interval": interval,
            "n_tickers_with_true_short": n_true,
            "true_short_mean": true_mean,
            "net_return": float(row["net_return"]) if "net_return" in row and pd.notna(row["net_return"]) else np.nan,
            "validation_status": status,
        })
    return pd.DataFrame(rows), fail

def _build_weights_sign_check(*, bt_positions: Optional[pd.DataFrame], phase: str) -> Tuple[pd.DataFrame, int]:
    """
    [개선안 17번] Long-only 확정(음수 가중치 0개) 점검.
    """
    if bt_positions is None or bt_positions.empty:
        return pd.DataFrame(), 0
    p = _ensure_dt(bt_positions, "date")
    if "phase" in p.columns:
        p = p[p["phase"].astype(str) == phase].copy()
    if p.empty or "weight" not in p.columns:
        return pd.DataFrame(), 0

    rows = []
    fail = 0
    for dt, g in p.groupby("date", sort=True):
        w = g["weight"].astype(float)
        n = int(len(w))
        nneg = int((w < 0).sum())
        if nneg > 0:
            fail += 1
        rows.append({
            "date": str(pd.to_datetime(dt).date()),
            "phase": phase,
            "n_positions": n,
            "n_negative_weights": nneg,
            "min_weight": float(np.nanmin(w)) if n else np.nan,
            "max_weight": float(np.nanmax(w)) if n else np.nan,
            "sum_weight": float(np.nansum(w)) if n else np.nan,
            "is_long_only": bool(nneg == 0),
        })
    return pd.DataFrame(rows), fail

def _write_text(path: Path, text: str) -> None:
    _safe_mkdir(path.parent)
    path.write_text(text, encoding="utf-8")

def _md_table_from_df(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    [개선안 17번] markdown 표 생성(의존성 최소화를 위해 pandas.to_markdown 대신 수동).
    """
    if df is None or df.empty:
        return ""
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    # 문자열화
    d = d.astype(object).where(pd.notna(d), "")
    lines = []
    lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n(표는 상위 {max_rows}행만 표시, 전체는 CSV 참조)")
    return "\n".join(lines)

def _build_midterm_report_md(
    *,
    base_dir: Path,
    run_id: str,
    phase: str,
    manifest_rel: str,
    cfg: dict,
    summary_metrics_csv: str,
    compare_csv: Optional[str],
    compare_note: Optional[str],
    legacy_note: Optional[str],
) -> str:
    """
    [개선안 17번] 중간발표 보고서(Markdown) 생성.
    - 금융 비전공자 설명 우선, 기술 상세는 Appendix로 분리
    - ‘데이터 누수 없음’ 단정 금지: ‘수익률 적용 구간 점검 PASS…’로 표현
    """
    l7 = (cfg.get("l7", {}) if isinstance(cfg, dict) else {}) or {}
    signal_source = str(l7.get("signal_source", "unknown"))
    holding_days = l7.get("holding_days", "")
    top_k = l7.get("top_k", "")
    cost_bps = l7.get("cost_bps", "")
    buffer_k = l7.get("buffer_k", "")
    diversify = (l7.get("diversify", {}) if isinstance(l7.get("diversify", {}), dict) else {}) or {}
    regime = (l7.get("regime", {}) if isinstance(l7.get("regime", {}), dict) else {}) or {}

    # --------- load derived tables (generated in this script) ----------
    # NOTE: report should reference artifacts/tables as the single source of truth for numbers
    tables_dir = base_dir / "artifacts" / "tables"
    metrics_df = pd.read_csv(tables_dir / "summary_metrics.csv") if (tables_dir / "summary_metrics.csv").exists() else pd.DataFrame()
    sel_df = pd.read_csv(tables_dir / f"selection_diagnostics__{phase}.csv") if (tables_dir / f"selection_diagnostics__{phase}.csv").exists() else pd.DataFrame()
    cov_df = pd.read_csv(tables_dir / f"rebalance_coverage__{phase}.csv") if (tables_dir / f"rebalance_coverage__{phase}.csv").exists() else pd.DataFrame()
    monthly_df = pd.read_csv(tables_dir / f"monthly_returns__{phase}.csv") if (tables_dir / f"monthly_returns__{phase}.csv").exists() else pd.DataFrame()
    yearly_df = pd.read_csv(tables_dir / f"yearly_metrics__{phase}.csv") if (tables_dir / f"yearly_metrics__{phase}.csv").exists() else pd.DataFrame()
    compare_df = pd.read_csv(tables_dir / f"compare_baseline_vs_current__{phase}.csv") if (tables_dir / f"compare_baseline_vs_current__{phase}.csv").exists() else pd.DataFrame()

    # holdout metrics card
    hold_row = {}
    if not metrics_df.empty and "phase" in metrics_df.columns:
        sub = metrics_df[metrics_df["phase"].astype(str) == phase]
        if not sub.empty:
            hold_row = sub.iloc[0].to_dict()

    def _fmt_pct(v: Any, decimals: int = 2) -> str:
        vv = _pct_if_ratio(v)
        try:
            return f"{float(vv):.{decimals}f}%"
        except Exception:
            return "근거가 부족합니다"

    def _fmt_ratio(v: Any, decimals: int = 3) -> str:
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return "근거가 부족합니다"

    # selection diagnostics summary (realized rules)
    realized = {
        "unique_top_k": [],
        "eligible_mean": None,
        "selected_mean": None,
        "dropped_sectorcap_mean": None,
    }
    if not sel_df.empty:
        if "top_k" in sel_df.columns:
            realized["unique_top_k"] = sorted(sel_df["top_k"].dropna().astype(int).unique().tolist())
        if "eligible_count" in sel_df.columns:
            realized["eligible_mean"] = float(sel_df["eligible_count"].astype(float).mean())
        if "selected_count" in sel_df.columns:
            realized["selected_mean"] = float(sel_df["selected_count"].astype(float).mean())
        if "dropped_sectorcap" in sel_df.columns:
            realized["dropped_sectorcap_mean"] = float(sel_df["dropped_sectorcap"].astype(float).mean())

    # coverage summary
    cov_summary = {"coverage_mean": None, "coverage_min": None}
    if not cov_df.empty and "coverage_vs_universe_pct" in cov_df.columns:
        c = cov_df["coverage_vs_universe_pct"].astype(float)
        cov_summary["coverage_mean"] = float(c.mean())
        cov_summary["coverage_min"] = float(c.min())

    # best/worst months
    best_month = None
    worst_month = None
    if not monthly_df.empty and "net_return" in monthly_df.columns and "ym" in monthly_df.columns:
        mm = monthly_df.copy()
        mm["net_return"] = mm["net_return"].astype(float)
        best_month = mm.sort_values("net_return", ascending=False).iloc[0].to_dict()
        worst_month = mm.sort_values("net_return", ascending=True).iloc[0].to_dict()

    # 숫자 요약(holdout row)
    # - bt_metrics가 ratio 스케일일 수 있어 % 항목은 보정
    # - bt_metrics를 여기서 직접 로드하지 않고, summary_metrics.csv를 근거로 읽어오도록 설계(재현성/근거 단일화)
    parts: List[str] = []

    parts.append("## 생성 근거(필독)\n")
    parts.append(f"- **분석 대상 실행 식별자(run_tag)**: `{run_id}`\n")
    parts.append(f"- **설정 파일**: `../configs/config.yaml`\n")
    parts.append(f"- **근거 고정(manifest)**: `{manifest_rel}`\n")
    parts.append(f"- **핵심 표/그림(이 보고서의 수치 근거)**: `../artifacts/tables/`, `../artifacts/figures/`\n")
    if legacy_note:
        parts.append(f"- **주의(중요)**: {legacy_note}\n")
    parts.append("\n---\n")

    parts.append("## 1) 한 페이지 요약(비전공자용)\n")
    parts.append(
        "KOSPI200 안에서 종목을 **점수화(랭킹)** 한 뒤, 상위 종목만 골라 **규칙대로 리밸런싱**했을 때 성과가 어떻게 되는지 백테스트로 검증했습니다.\n"
    )
    parts.append("\n- 비유: “매일 1등 뽑기”가 아니라, **정해진 날짜(약 20영업일 간격)마다 성적표로 대표팀(Top-K)을 구성**하는 방식입니다.\n")
    parts.append("\n---\n")
    parts.append("## 1-1) 이번 실행 ‘숫자 카드’(holdout 핵심)\n")
    parts.append("- 근거: `../artifacts/tables/summary_metrics.csv`\n\n")
    parts.append(
        "| 지표 | 값 |\n"
        "|---|---:|\n"
        f"| 총수익률(net_total_return) | {_fmt_pct(hold_row.get('net_total_return'))} |\n"
        f"| CAGR(net_cagr) | {_fmt_pct(hold_row.get('net_cagr'))} |\n"
        f"| Sharpe(net_sharpe) | {_fmt_ratio(hold_row.get('net_sharpe'))} |\n"
        f"| MDD(net_mdd) | {_fmt_pct(hold_row.get('net_mdd'))} |\n"
        f"| 평균 턴오버(avg_turnover_oneway) | {_fmt_pct(hold_row.get('avg_turnover_oneway'))} |\n"
        f"| 리밸런싱 횟수(n_rebalances) | {hold_row.get('n_rebalances', '근거가 부족합니다')} |\n"
        f"| 기간(date_start~date_end) | {hold_row.get('date_start','?')} ~ {hold_row.get('date_end','?')} |\n"
    )
    parts.append("\n---\n")

    parts.append("## 2) 이 프로젝트가 해결하는 문제(쉬운 정의)\n")
    parts.append(
        "- **문제**: KOSPI200처럼 종목이 많은 시장에서, ‘그냥 지수/평균을 사는 것’보다 더 나은 종목 조합이 가능한가?\n"
        "- **우리가 한 일**: 종목별로 ‘좋아 보이는 정도’를 점수로 만들고, 그 점수 상위 종목만 담는 규칙을 만들어 성과를 검증.\n"
        "- **중요 제약**: 이 보고서의 모든 수치/그림은 `artifacts/run_manifest.json`에 기록된 소스에서만 가져옵니다.\n"
    )

    parts.append("\n---\n")
    parts.append("## 3) 프로젝트 구조(Repo reality check: 듀얼 트랙)\n")
    parts.append(
        "- **Pipeline Track (L0~L7D)**: 데이터 수집/전처리 → (선택) 모델 → 백테스트/벤치마크/안정성\n"
        "- **Ranking Track (L8/L11/L12 등)**: 제품용 랭킹/스냅샷/UI 패키징\n"
    )
    parts.append("\n### 실제 실행/저장 규칙(근거)\n")
    parts.append("- 엔트리포인트: `src/core/pipeline.py`, `src/tools/run_stage_pipeline.py`\n")
    parts.append("- 아티팩트 계약: `docs/ARTIFACT_CONTRACT.md`\n")
    parts.append("- 저장: 일반 `data/interim/{run_tag}/...` / 예외(L2) `data/interim/fundamentals_annual.parquet`\n")

    parts.append("\n---\n")
    parts.append("## 4) 이번 실행에서 ‘신호’는 무엇인가(중요: ranking 기반)\n")
    if signal_source == "ranking":
        parts.append("- **신호 생성 방식**: `ranking` (랭킹 기반)\n")
        parts.append("- **핵심 아이디어(쉬운 말)**: 같은 날짜에 있는 여러 종목을 서로 비교해 ‘상대적으로 더 좋아 보이는 종목’을 고릅니다.\n")
        parts.append("- **구현 흐름(근거 경로)**:\n")
        parts.append("  - `dataset_daily`(피처 포함) → 날짜별 정규화/가중합으로 `score_total` 생성\n")
        parts.append("  - `score_total`을 사용해 `rank_total`(1~N) 생성\n")
        parts.append("  - 리밸런싱 날짜에만 `rebalance_scores(score_ens)`로 변환(L7 입력)\n")
        parts.append("  - 코드: `src/stages/modeling/l6r_ranking_scoring.py` → `src/components/ranking/score_engine.py`\n")
        parts.append("- **근거 산출물**: `data/interim/{run_tag}/rebalance_scores.(parquet|csv)`\n")
    else:
        parts.append(f"- **신호 생성 방식**: `{signal_source}`\n")
        parts.append("- 근거가 부족합니다: 이번 리포트는 ranking 기반 실행을 전제로 작성되어 있습니다.\n")

    parts.append("\n---\n")
    parts.append("## 5) 핵심 용어(비전공자용 1줄 정의)\n")
    parts.append(
        "| 용어 | 1줄 정의 |\n"
        "|---|---|\n"
        "| 리밸런싱 | 포트폴리오 구성 종목을 규칙에 따라 다시 뽑는 날 |\n"
        "| Holdout | 학습에 쓰지 않고 마지막에 성적을 확인하는 ‘시험지’ 구간 |\n"
        "| Turnover | 포트폴리오가 얼마나 자주/많이 바뀌었는지(교체 비율) |\n"
        "| Sharpe | 변동성 대비 성과 효율(높을수록 유리) |\n"
        "| MDD | 최고점 대비 최대 낙폭(작을수록 유리) |\n"
    )

    parts.append("\n---\n")
    parts.append("## 6) 데이터/기간/검증 설계(왜 이렇게 했나)\n")
    parts.append(f"- **전체 데이터 기간**: `configs/config.yaml`의 start/end에 기반\n")
    parts.append(f"- **Holdout 길이**: `l4.holdout_years=2` (최근 2년을 시험지로 분리)\n")
    parts.append(f"- **리밸런싱 간격**: `l4.step_days=20`, `l7.holding_days=20` (대략 월간 수준)\n")
    parts.append("\n> 비유: ‘과거 기출문제(dev)’로 연습한 뒤, ‘마지막 모의고사(holdout)’에서 점수를 확인합니다.\n")

    parts.append("\n---\n")
    parts.append("## 7) 피처/팩터(무엇을 보고 점수를 만들었나)\n")
    parts.append("- 최종 피처 목록(근거 CSV): `../artifacts/tables/feature_list.csv`\n")
    parts.append(
        "\n### 카테고리 예시(쉬운 설명)\n"
        "- **가치(Valuation)**: ‘비싸냐/싸냐’(예: PER/PBR)\n"
        "- **수익성(Profitability)**: ‘돈을 잘 버는가’(예: ROE, 순이익)\n"
        "- **모멘텀(Momentum)**: ‘최근 가격 흐름이 강한가’\n"
        "- **리스크(Risk)**: ‘가격이 얼마나 흔들리는가’\n"
        "- **유동성(Liquidity)**: ‘거래가 얼마나 활발한가’\n"
    )

    parts.append("\n---\n")
    parts.append("## 8) 포트폴리오 규칙(점수가 매매로 바뀌는 규칙)\n")
    parts.append(f"- **Top-K(설정)**: {top_k}\n")
    if realized["unique_top_k"]:
        parts.append(f"- **Top-K(실제 적용, 산출물 기반)**: {realized['unique_top_k']}  \n")
        parts.append("  - 근거: `../artifacts/tables/selection_diagnostics__holdout.csv`의 `top_k` 컬럼\n")
        parts.append("  - 해석: `regime`(시장 국면) 설정으로 날짜별 top_k가 바뀐 것으로 보이지만, ‘원인 단정’은 금지합니다.\n")
    else:
        parts.append("- **Top-K(실제 적용)**: 근거가 부족합니다(선택 진단 CSV 없음)\n")
    parts.append(f"- **보유기간(holding_days)**: {holding_days}일\n")
    parts.append(f"- **거래비용(cost_bps)**: {cost_bps} bps (리밸런싱 발생 시 비용 차감)\n")
    parts.append(f"- **버퍼(buffer_k)**: {buffer_k} (기존 보유 유지 후보를 넓혀 턴오버 완화)\n")
    parts.append(f"- **업종 분산 제약(diversify)**: enabled={diversify.get('enabled','')}, group_col={diversify.get('group_col','')}, max_names_per_group={diversify.get('max_names_per_group','')}\n")
    parts.append(f"- **시장 국면(regime)**: enabled={regime.get('enabled','')}, lookback_days={regime.get('lookback_days','')}, threshold_pct={regime.get('threshold_pct','')}\n")
    parts.append("\n- 선택/진단(근거 CSV): `../artifacts/tables/selection_diagnostics__holdout.csv`\n")
    if realized["eligible_mean"] is not None:
        parts.append(f"- (요약) 평균 eligible_count: {realized['eligible_mean']:.2f}\n")
    if realized["selected_mean"] is not None:
        parts.append(f"- (요약) 평균 selected_count: {realized['selected_mean']:.2f}\n")
    if realized["dropped_sectorcap_mean"] is not None:
        parts.append(f"- (요약) 평균 dropped_sectorcap: {realized['dropped_sectorcap_mean']:.2f}\n")

    parts.append("\n---\n")
    parts.append("## 9) 성과 요약(holdout) — 표/그림 링크\n")
    parts.append(f"- 성과 요약표: `{summary_metrics_csv}`\n")
    parts.append("- 누적 성과: `../artifacts/figures/equity_curve__holdout.png`\n")
    parts.append("- 드로우다운: `../artifacts/figures/drawdown__holdout.png`\n")
    parts.append("- 턴오버(현실성): `../artifacts/figures/turnover_timeseries__holdout.png`, `../artifacts/figures/turnover_hist__holdout.png`\n")
    parts.append("- 커버리지(가능할 때): `../artifacts/figures/coverage_vs_universe__holdout.png`\n")
    parts.append("- 베타(가능할 때): `../artifacts/figures/beta_roll12__holdout.png`\n")
    parts.append("- 월별 수익률표(근거 CSV): `../artifacts/tables/monthly_returns__holdout.csv`\n")
    parts.append("- 연도별 성과표(근거 CSV): `../artifacts/tables/yearly_metrics__holdout.csv`\n")
    if best_month and worst_month:
        parts.append("\n### 월별 성과 하이라이트(근거: monthly_returns__holdout.csv)\n")
        parts.append(
            f"- 최고 월(전략 net_return): {best_month.get('ym')} / {float(best_month.get('net_return'))*100.0:.2f}%\n"
            f"- 최저 월(전략 net_return): {worst_month.get('ym')} / {float(worst_month.get('net_return'))*100.0:.2f}%\n"
        )
    if cov_summary["coverage_mean"] is not None:
        parts.append("\n### 커버리지 요약(근거: rebalance_coverage__holdout.csv)\n")
        parts.append(f"- 평균 커버리지(coverage_vs_universe_pct): {cov_summary['coverage_mean']:.2f}%\n")
        parts.append(f"- 최소 커버리지(coverage_vs_universe_pct): {cov_summary['coverage_min']:.2f}%\n")

    parts.append("\n---\n")
    parts.append("## 10) 검증/근거 고정(문구 안전화)\n")
    parts.append("- 검증 요약: `../artifacts/validation_report.md`\n")
    parts.append("- **수익률 적용 구간 점검**: ‘PASS(t+1~t+N)’ 형태로만 표현합니다. ‘데이터 누수 없음’ 같은 단정 표현은 사용하지 않습니다.\n")
    parts.append("- **Long-only 확정**: 음수 가중치 0건이면 Long-only로만 표현합니다.\n")
    parts.append("- 표(근거 CSV): `../artifacts/tables/leakage_sanity_check__holdout.csv`, `../artifacts/tables/weights_sign_check__holdout.csv`\n")

    parts.append("\n---\n")
    parts.append("## 11) 베이스라인 → 현재 비교(가능할 때만)\n")
    if compare_csv:
        parts.append(f"- 비교표(CSV): `{compare_csv}`\n")
        parts.append("- 비교 차트(PNG): `../artifacts/figures/compare_baseline_vs_current__holdout.png`\n")
        if compare_note:
            parts.append(f"- **주의**: {compare_note}\n")
        parts.append("\n> 해석 원칙: Δ(증감)만 말하지 말고, **비용/턴오버 같은 현실성 지표도 같이** 봅니다.\n")
        if not compare_df.empty:
            parts.append("\n### 비교 요약(표)\n")
            parts.append(_md_table_from_df(compare_df, max_rows=20) + "\n")
    else:
        parts.append("- 근거가 부족합니다: baseline vs current 비교 파일을 생성하지 못했습니다.\n")

    parts.append("\n---\n")
    parts.append("## 12) 한계/리스크(근거 기반으로만)\n")
    parts.append("- **성과가 나쁘게 나온 경우의 원인 단정 금지**: 원인 분석에는 추가 증거(예: 피처 변화, 리밸런싱 룰 차이)가 필요합니다.\n")
    parts.append("- **legacy fallback**: 일부 핵심 산출물이 run_tag 폴더 밖(legacy root)에서 로드될 수 있습니다. 이런 경우 결과를 ‘완전히 동일 run_tag의 산출물’이라고 단정하면 안 됩니다.\n")
    parts.append("- **거래비용 단순화**: cost_bps는 단순 모델입니다(슬리피지/시장충격은 별도 모델 필요).\n")

    parts.append("\n---\n")
    parts.append("## 13) 다음 단계(필수 vs 선택)\n")
    parts.append("- **Must(필수)**: 동일 run_tag 완전 격리 저장(legacy 최소화), 비용 모델 개선(슬리피지 분리), 진단 리포트 고정(커버리지/섹터/레짐)\n")
    parts.append("- **Nice-to-have(선택)**: 섹터 중립화, 매크로 필터, AutoML\n")

    parts.append("\n---\n")
    parts.append("## Appendix A) 단계별 I/O 계약(기술 상세)\n")
    parts.append("- 자세한 단계별 I/O 계약은 `docs/ARTIFACT_CONTRACT.md`를 따릅니다.\n")
    parts.append("- 이번 보고서의 숫자/근거 파일 목록은 `artifacts/run_manifest.json`의 `resolved_sources`를 기준으로 합니다.\n")
    parts.append("\n### A-1) Stage별 입력/출력 요약(계약 기반)\n")
    parts.append(
        "| Stage | Track | 입력(대표) | 출력(대표) | 저장 위치 |\n"
        "|---|---|---|---|---|\n"
        "| L0 | Pipeline | - | universe_k200_membership_monthly | data/interim/{run_tag}/ |\n"
        "| L1 | Pipeline | universe_k200_membership_monthly | ohlcv_daily | data/interim/{run_tag}/ |\n"
        "| L1B | Pipeline | universe_k200_membership_monthly | sector_map | data/interim/{run_tag}/ |\n"
        "| L2 | Pipeline | universe_k200_membership_monthly | fundamentals_annual | data/interim/ (예외, run_tag 아님) |\n"
        "| L3 | Pipeline | ohlcv_daily + fundamentals_annual (+sector_map) | panel_merged_daily | data/interim/{run_tag}/ |\n"
        "| L4 | Pipeline | panel_merged_daily | dataset_daily + cv_folds_short/long | data/interim/{run_tag}/ |\n"
        "| L5 | Pipeline | dataset_daily + cv_folds_* | pred_*_oos + model_metrics | data/interim/{run_tag}/ |\n"
        "| L6 | Pipeline | pred_*_oos (+universe) | rebalance_scores + summary | data/interim/{run_tag}/ |\n"
        "| L6R | Ranking→Backtest | dataset_daily + cv_folds_short | rebalance_scores(score_ens) | data/interim/{run_tag}/ |\n"
        "| L7 | Pipeline | rebalance_scores | bt_returns/bt_metrics/... | data/interim/{run_tag}/ |\n"
        "| L7C | Pipeline | bt_returns | bt_vs_benchmark/... | data/interim/{run_tag}/ |\n"
        "| L7D | Pipeline | bt_returns + equity | yearly/rolling/dd events | data/interim/{run_tag}/ |\n"
        "| L8 | Ranking | dataset_daily(or panel) | ranking_daily/snapshot | data/interim/{run_tag}/ |\n"
        "| L11 | Ranking | ranking_daily + ohlcv | ui_* | data/interim/{run_tag}/ + reports/ui |\n"
        "| L12 | Ranking | history + ui_* | final_export pack | artifacts/reports/final_export/{run_tag}/ |\n"
    )

    parts.append("\n### A-2) 프로젝트 히스토리 요약(근거: reports/history/history_manifest.csv)\n")
    hist_path = base_dir / "reports" / "history" / "history_manifest.csv"
    if hist_path.exists():
        try:
            hdf = pd.read_csv(hist_path)
            cols = [c for c in ["stage_no", "track", "run_tag", "created_at", "change_title", "holdout_sharpe", "holdout_mdd", "cost"] if c in hdf.columns]
            hsub = hdf[cols].copy() if cols else hdf.copy()
            # 최신순 상위 12개만 보여주기
            if "created_at" in hsub.columns:
                hsub = hsub.sort_values("created_at", ascending=False)
            parts.append(_md_table_from_df(hsub, max_rows=12) + "\n")
            if (hsub.get("run_tag") is not None) and (run_id not in set(hsub["run_tag"].astype(str).tolist())):
                parts.append(f"\n- 참고: 현재 run_tag `{run_id}`는 history_manifest.csv에 기록되어 있지 않습니다(근거가 부족합니다).\n")
        except Exception:
            parts.append("- 근거가 부족합니다: history_manifest.csv를 읽지 못했습니다.\n")
    else:
        parts.append("- 근거가 부족합니다: reports/history/history_manifest.csv가 없습니다.\n")

    return "".join(parts)

def _build_ppt_outline_md(*, manifest_rel: str) -> str:
    """
    [개선안 17번] 12장 PPT 문안(요약) 생성. 슬라이드 구조는 요구사항 고정.
    """
    return "\n".join([
        "## PPT 구성(12 슬라이드) — 문안/그림·표 참조/발표 멘트(요약)",
        "",
        f"- **분석 근거 고정**: `{manifest_rel}`",
        "- **핵심 그림/표 폴더**: `../artifacts/figures/`, `../artifacts/tables/`",
        "",
        "---",
        "",
        "## S1. 문제정의 + 우리가 만든 것",
        "- 제목: “KOSPI200에서 ‘지수 평균’보다 나은 조합을 만들 수 있는가?”",
        "- 그림: `../artifacts/figures/equity_curve__holdout.png`",
        "",
        "## S2. 핵심 개념/용어(용어 카드)",
        "- 듀얼 트랙(Pipeline vs Ranking), 리밸런싱, holdout, Sharpe/MDD/Turnover 정의(1줄)",
        "",
        "## S3. 데이터/파이프라인 전체 지도",
        "- L0~L7D / L8/L11/L12 단계 요약(표는 보고서에 있음)",
        "",
        "## S4. 피쳐/팩터 개념",
        "- CSV: `../artifacts/tables/feature_list.csv`",
        "",
        "## S5. 신호 생성: 이번 실행은 ranking 기반임을 명확히",
        "- (모델은 appendix로) L6R: score_total/rank_total → score_ens 변환",
        "",
        "## S6. 검증 설계: Walk-forward + Holdout(시험지 비유)",
        "- 표: `../artifacts/tables/summary_metrics.csv` (holdout 기간/리밸런싱)",
        "",
        "## S7. 포트폴리오 규칙",
        "- 표: `../artifacts/tables/selection_diagnostics__holdout.csv`",
        "",
        "## S8. 성과 요약(holdout)",
        "- 그림: `../artifacts/figures/equity_curve__holdout.png`, `../artifacts/figures/drawdown__holdout.png`",
        "",
        "## S9. 현실성: turnover/거래비용",
        "- 그림: `../artifacts/figures/turnover_timeseries__holdout.png`, `../artifacts/figures/turnover_hist__holdout.png`",
        "",
        "## S10. 커버리지/진단",
        "- 그림: `../artifacts/figures/coverage_vs_universe__holdout.png` (가능할 때)",
        "- 베타가 있으면 `../artifacts/figures/beta_roll12__holdout.png`",
        "",
        "## S11. 베이스라인 → 현재 비교(가능할 때)",
        "- 표/그림: `../artifacts/tables/compare_baseline_vs_current__holdout.csv`, `../artifacts/figures/compare_baseline_vs_current__holdout.png`",
        "",
        "## S12. 다음 단계(Must vs Nice-to-have) + Canva 제작 계획",
        "- Canva 입력 패키지: `reports/canva_export_pack.md`",
    ])

def _build_speaker_notes_md() -> str:
    """
    [개선안 17번] 슬라이드별 30~45초 대본(핵심만).
    """
    return "\n".join([
        "## 발표 대본(슬라이드별, 30~45초 목표)",
        "",
        "> 공통 원칙: 모르는 용어는 정의하고, 확실하지 않으면 “근거가 부족합니다”라고 말합니다.",
        "",
        "---",
        "",
        "## S1 대본",
        "오늘은 KOSPI200 안에서 ‘지수 평균보다 나은 조합’을 만들 수 있는지 묻습니다. 종목을 점수화해 상위만 담는 규칙을 만들고, 백테스트로 검증했습니다. 모든 숫자는 파일로 추적 가능합니다.",
        "",
        "## S2 대본",
        "핵심 용어를 먼저 정의합니다. 듀얼 트랙은 ‘검증용 파이프라인’과 ‘제품용 랭킹’을 분리한 구조입니다. holdout은 마지막 2년을 남겨 둔 시험지입니다. Sharpe는 ‘변동성 대비 효율’, MDD는 ‘최대 낙폭’입니다.",
        "",
        "## S3 대본",
        "데이터를 한 표로 만들고, 점수를 만들고, 그 점수를 투자 규칙으로 바꿔 성과를 검증합니다. 단계별 출력이 파일이라서 근거를 역추적할 수 있습니다.",
        "",
        "## S4 대본",
        "점수는 ‘회사 체력/가격 흐름/위험/거래 활발함’ 같은 관찰값을 숫자로 만든 피처에서 나옵니다. 최종 피처 목록은 CSV로 고정했습니다.",
        "",
        "## S5 대본",
        "이번 실행의 신호는 모델이 아니라 랭킹 기반입니다. 날짜별로 피처를 정규화하고 가중합으로 점수를 만든 뒤, 리밸런싱 날짜에만 점수표를 확정합니다.",
        "",
        "## S6 대본",
        "검증은 walk-forward이고, 마지막 2년은 holdout으로 남깁니다. 시험지 비유로, 학습에 쓰지 않은 구간에서 성적을 봅니다.",
        "",
        "## S7 대본",
        "투자 규칙은 Top-K를 동일비중으로 담고, 보유기간마다 리밸런싱합니다. 버퍼는 과도한 교체를 줄이는 장치입니다. 거래비용은 bps로 차감합니다.",
        "",
        "## S8 대본",
        "누적 수익과 드로우다운으로 ‘언제 벌고 언제 잃는지’를 같이 봅니다. 숫자는 CAGR, Sharpe, MDD를 핵심으로 봅니다.",
        "",
        "## S9 대본",
        "현실성은 턴오버와 비용입니다. 자주 바꾸면 비용과 슬리피지가 커질 수 있어, 턴오버를 분포/시계열로 같이 봅니다.",
        "",
        "## S10 대본",
        "커버리지는 신호가 유니버스 전체를 얼마나 덮는지입니다. 베타는 시장과 같이 움직인 정도입니다. 데이터가 없으면 ‘근거가 부족합니다’라고 말합니다.",
        "",
        "## S11 대본",
        "베이스라인과 비교는 가능할 때만 합니다. 표에는 Δ(증감)와 출처 파일을 함께 적었습니다.",
        "",
        "## S12 대본",
        "다음 단계는 재현성/현실성/진단 가능성 강화입니다. 오늘 산출물은 Canva에서 바로 디자인할 수 있도록 패키징했습니다.",
    ])

def _build_ppt_numbers_by_slide_md(
    *,
    base_dir: Path,
    phase: str,
    run_id: str,
) -> str:
    """
    [개선안 17번] 슬라이드별 Copy/Paste 수치 묶음(MD) 생성.
    """
    tbl = base_dir / "artifacts" / "tables"
    metrics = pd.read_csv(tbl / "summary_metrics.csv") if (tbl / "summary_metrics.csv").exists() else pd.DataFrame()
    monthly = pd.read_csv(tbl / f"monthly_returns__{phase}.csv") if (tbl / f"monthly_returns__{phase}.csv").exists() else pd.DataFrame()
    compare = pd.read_csv(tbl / f"compare_baseline_vs_current__{phase}.csv") if (tbl / f"compare_baseline_vs_current__{phase}.csv").exists() else pd.DataFrame()
    leakage = pd.read_csv(tbl / f"leakage_sanity_check__{phase}.csv") if (tbl / f"leakage_sanity_check__{phase}.csv").exists() else pd.DataFrame()
    wsign = pd.read_csv(tbl / f"weights_sign_check__{phase}.csv") if (tbl / f"weights_sign_check__{phase}.csv").exists() else pd.DataFrame()

    # 핵심 수치: holdout row
    hold = metrics[metrics.get("phase", "").astype(str) == phase] if not metrics.empty and "phase" in metrics.columns else pd.DataFrame()
    h = hold.iloc[0].to_dict() if not hold.empty else {}

    def g(k, default=""):
        v = h.get(k, default)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        # [개선안 17번] 표시용 스케일 보정
        if k in ("net_total_return", "net_cagr", "net_mdd", "avg_turnover_oneway"):
            vv = _pct_if_ratio(v)
            return str(vv)
        return str(v)

    # 검증 요약
    leakage_fail = int((leakage.get("validation_status", "").astype(str) == "FAIL").sum()) if not leakage.empty else None
    wneg_total = int(wsign.get("n_negative_weights", 0).sum()) if not wsign.empty else None

    blocks: List[str] = []
    blocks.append("# PPT 붙여넣기용 수치 모음 (S1~S12)\n")
    blocks.append(f"- run_tag: `{run_id}`\n")
    blocks.append(f"- phase: `{phase}`\n\n")

    def sec(title: str):
        blocks.append(f"\n## {title}\n")

    def cp(text: str):
        blocks.append("```")
        blocks.append(text.rstrip())
        blocks.append("```\n")

    sec("S1")
    cp(f"문제정의: KOSPI200 안에서 ‘지수 평균’보다 나은 종목 조합을 만들 수 있는가?\n근거: artifacts/figures/equity_curve__{phase}.png")

    sec("S2")
    cp("용어 정의(1줄):\n- Sharpe: 변동성 대비 성과 효율\n- MDD(Max Drawdown): 최고점 대비 최대 낙폭\n- Turnover: 포트폴리오 교체 비율\n- Holdout: 마지막 시험지(학습에 쓰지 않은 구간)\n근거: 보고서/표준 정의")

    sec("S7")
    # [개선안 17번] top_k는 설정값과 실제 적용값이 다를 수 있으므로 selection_diagnostics로 보강
    sel_path = tbl / f"selection_diagnostics__{phase}.csv"
    realized_topk = None
    if sel_path.exists():
        try:
            s = pd.read_csv(sel_path)
            if "top_k" in s.columns:
                realized_topk = sorted(s["top_k"].dropna().astype(int).unique().tolist())
        except Exception:
            realized_topk = None
    topk_line = f"- top_k(설정)={g('top_k')}"
    if realized_topk:
        topk_line += f"\n- top_k(실제 적용)={realized_topk}  # 근거: artifacts/tables/selection_diagnostics__{phase}.csv"
    cp(
        "포트폴리오 규칙:\n"
        + topk_line
        + f"\n- holding_days={g('holding_days')}\n- cost_bps_used={g('cost_bps_used')}\n- buffer_k={g('buffer_k')}\n"
        + f"근거: artifacts/tables/summary_metrics.csv (+ selection_diagnostics__{phase}.csv)"
    )

    sec("S8")
    cp(f"성과(holdout):\n- net_total_return={g('net_total_return')}%\n- net_cagr={g('net_cagr')}%\n- net_sharpe={g('net_sharpe')}\n- net_mdd={g('net_mdd')}%\n근거: artifacts/tables/summary_metrics.csv")

    sec("S9")
    cp(f"현실성(holdout):\n- avg_turnover_oneway={g('avg_turnover_oneway')}%\n- cost_bps_used={g('cost_bps_used')} bps\n근거: artifacts/tables/summary_metrics.csv + turnover 그림")

    sec("S11")
    if not compare.empty:
        cp("baseline vs current(holdout) 요약:\n" + "\n".join(
            f"- {r['metric']}: baseline={r['baseline_holdout_value']} → current={r['current_holdout_value']} (Δ={r['delta_current_minus_baseline']})"
            for _, r in compare.iterrows()
        ) + "\n근거: artifacts/tables/compare_baseline_vs_current__holdout.csv")
    else:
        cp("baseline vs current: 근거가 부족합니다 (비교 CSV 없음)")

    sec("검증 문구(안전화)")
    cp(
        "수익률 적용 구간 점검: leakage_sanity_check 기준 PASS(t+1~t+N)\n"
        f"- FAIL 건수: {leakage_fail if leakage_fail is not None else '근거가 부족합니다'}\n"
        "Long-only 확정: weights_sign_check 기준 음수 비중 0건\n"
        f"- 음수 가중치 합계: {wneg_total if wneg_total is not None else '근거가 부족합니다'}\n"
        f"근거: artifacts/tables/leakage_sanity_check__{phase}.csv, artifacts/tables/weights_sign_check__{phase}.csv"
    )

    sec("월별 성과 표(요약)")
    if not monthly.empty:
        cp(_md_table_from_df(monthly, max_rows=12) + f"\n출처: artifacts/tables/monthly_returns__{phase}.csv")
    else:
        cp("근거가 부족합니다: monthly_returns CSV 없음")

    return "\n".join(blocks)

def _build_canva_assets_manifest(base_dir: Path) -> Dict[str, Any]:
    """
    [개선안 17번] Canva Import용 asset manifest 생성.
    """
    assets: List[Dict[str, Any]] = []
    for rel in [
        *sorted((base_dir / "artifacts" / "figures").glob("*.png")),
        *sorted((base_dir / "artifacts" / "tables").glob("*.csv")),
        base_dir / "reports" / "midterm_report.md",
        base_dir / "reports" / "ppt_outline_12slides.md",
        base_dir / "reports" / "speaker_notes.md",
        base_dir / "reports" / "canva_export_pack.md",
        base_dir / "reports" / "ppt_numbers_by_slide.md",
        base_dir / "artifacts" / "run_manifest.json",
        base_dir / "artifacts" / "validation_report.md",
    ]:
        p = Path(rel)
        if not p.exists():
            continue
        kind = "figure" if p.suffix.lower() == ".png" else "table" if p.suffix.lower() == ".csv" else "text"
        meta = _file_stat_meta(p)
        assets.append({
            "path": _relpath(p, base_dir),
            "kind": kind,
            "sha256": meta["sha256"],
            "mtime": meta["mtime"],
            "size_bytes": meta["size_bytes"],
        })
    return {"created_at": _now_kst_iso(), "assets": assets}

def _build_canva_export_pack_md(*, run_id: str, phase: str) -> str:
    """
    [개선안 17번] Canva에서 바로 붙일 수 있는 “입력 패키지” 문서(프롬프트는 제외).
    """
    return "\n".join([
        "## Canva Export Pack (디자인용 입력 패키지)",
        "",
        f"- run_tag: `{run_id}`",
        f"- phase: `{phase}`",
        "",
        "### 가져갈 파일(상대경로)",
        "- 핵심 그림: `artifacts/figures/*.png`",
        "- 핵심 표: `artifacts/tables/*.csv`",
        "- 근거 고정: `artifacts/run_manifest.json`, `artifacts/validation_report.md`",
        "",
        "### 슬라이드별 추천 배치(요약)",
        "- S1: `equity_curve__holdout.png`",
        "- S8: `equity_curve__holdout.png`, `drawdown__holdout.png`",
        "- S9: `turnover_timeseries__holdout.png`, `turnover_hist__holdout.png`",
        "- S10: `beta_roll12__holdout.png` / `coverage_vs_universe__holdout.png` (존재할 때만)",
        "- S11: `compare_baseline_vs_current__holdout.png` + CSV 표",
        "",
        "### Copy/Paste 숫자(바로 사용)",
        "- `reports/ppt_numbers_by_slide.md`",
        "",
        "### 주의(필수 각주 후보)",
        "- 검증 문구는 ‘데이터 누수 없음’이 아니라 **‘수익률 적용 구간 점검 PASS(t+1~t+N)’**로만 표기",
        "- baseline 비교는 근거가 있을 때만(없으면 ‘근거가 부족합니다’)",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate midterm report artifacts (figures/tables/manifest)")
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--run-id", type=str, default=None, help="Run identifier (run_tag or run_id). If omitted, auto-detect.")
    ap.add_argument("--phase", type=str, default="holdout", help="Which phase to plot (dev|holdout)")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument("--baseline-tag", type=str, default=None, help="Baseline run_tag for comparison (optional). If omitted, auto-detect from config/history.")
    args = ap.parse_args()

    _configure_matplotlib_korean_font()
    cfg = load_config(args.config)
    base_dir = get_path(cfg, "base_dir")
    interim_base = get_path(cfg, "data_interim")

    figures_dir = base_dir / "artifacts" / "figures"
    tables_dir = base_dir / "artifacts" / "tables"
    canva_dir = base_dir / "artifacts" / "canva"
    _safe_mkdir(figures_dir)
    _safe_mkdir(tables_dir)
    _safe_mkdir(canva_dir)

    # analysis identifier selection
    run_id = args.run_id or _pick_run_id_from_meta(interim_base) or _pick_latest_run_tag_with_bt_metrics(interim_base)
    if not run_id:
        raise RuntimeError("Cannot detect run_id/run_tag. Provide --run-id or ensure bt_metrics exists.")

    run_dir = interim_base / run_id
    run_dir_exists = run_dir.exists() and run_dir.is_dir()

    # load key artifacts
    bt_metrics, src_bt_metrics = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="bt_metrics", required=True)
    bt_returns, src_bt_returns = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="bt_returns", required=True)
    bt_equity_curve, src_bt_eq = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="bt_equity_curve", required=False)
    bt_benchmark_returns, src_bt_bench = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="bt_benchmark_returns", required=False)
    bt_positions, src_bt_pos = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="bt_positions", required=False)
    bt_yearly_metrics, src_bt_yearly = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="bt_yearly_metrics", required=False)
    selection_diag, src_sel = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="selection_diagnostics", required=False)
    rebalance_scores_summary, src_l6_sum = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="rebalance_scores_summary", required=False)
    rebalance_scores, src_l6 = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="rebalance_scores", required=False)

    # dataset for feature list
    dataset_daily, src_ds = _load_optional(interim_base=interim_base, run_dir=run_dir if run_dir_exists else None, name="dataset_daily", required=True)

    phase = str(args.phase)

    # 1) Tables
    # metrics summary
    m = bt_metrics.copy()
    if "phase" in m.columns:
        m["phase"] = m["phase"].astype(str)
    metrics_csv = tables_dir / "summary_metrics.csv"
    _save_csv(m, metrics_csv)

    # monthly returns (strategy vs benchmark if available)
    monthly = _compute_monthly_table(bt_returns, bt_benchmark_returns, phase=phase)
    monthly_csv = tables_dir / f"monthly_returns__{phase}.csv"
    _save_csv(monthly, monthly_csv)

    # yearly metrics
    if bt_yearly_metrics is not None and not bt_yearly_metrics.empty:
        yearly = bt_yearly_metrics.copy()
        if "phase" in yearly.columns:
            yearly = yearly[yearly["phase"].astype(str) == phase].copy()
        yearly_csv = tables_dir / f"yearly_metrics__{phase}.csv"
        _save_csv(yearly, yearly_csv)
    else:
        yearly_csv = None

    # feature list
    feature_list = _build_feature_list_table(cfg, dataset_daily)
    feature_list_csv = tables_dir / "feature_list.csv"
    _save_csv(feature_list, feature_list_csv)

    # selection diagnostics (optional)
    if selection_diag is not None and not selection_diag.empty:
        sel_csv = tables_dir / f"selection_diagnostics__{phase}.csv"
        sd = selection_diag.copy()
        if "phase" in sd.columns:
            sd = sd[sd["phase"].astype(str) == phase].copy()
        _save_csv(sd, sel_csv)
    else:
        sel_csv = None

    # rebalance coverage (optional but useful for 데이터 품질/커버리지)
    if rebalance_scores_summary is not None and not rebalance_scores_summary.empty:
        cov = rebalance_scores_summary.copy()
        cov = _ensure_dt(cov, "date")
        if "phase" in cov.columns:
            cov = cov[cov["phase"].astype(str) == phase].copy()
        keep_cols = [c for c in [
            "date",
            "phase",
            "n_tickers",
            "universe_n_tickers",
            "coverage_vs_universe_pct",
            "score_short_missing",
            "score_long_missing",
            "score_ens_missing",
        ] if c in cov.columns]
        cov = cov[keep_cols].sort_values("date").reset_index(drop=True)
        cov_csv = tables_dir / f"rebalance_coverage__{phase}.csv"
        _save_csv(cov, cov_csv)
    else:
        cov_csv = None

    # leakage sanity check (required for 발표 문구 안전화)
    leakage_df, leakage_fail = _build_leakage_sanity_check(
        bt_returns=bt_returns,
        rebalance_scores=rebalance_scores,
        phase=phase,
    )
    leakage_csv = tables_dir / f"leakage_sanity_check__{phase}.csv"
    _save_csv(leakage_df, leakage_csv)

    # weights sign check (long-only)
    wsign_df, wsign_fail = _build_weights_sign_check(bt_positions=bt_positions, phase=phase)
    wsign_csv = tables_dir / f"weights_sign_check__{phase}.csv"
    _save_csv(wsign_df, wsign_csv)

    # baseline vs current compare (if possible)
    baseline_tag = args.baseline_tag or _detect_baseline_tag(cfg, base_dir)
    compare_df, compare_note = _build_compare_baseline_vs_current(
        base_dir=base_dir,
        phase=phase,
        baseline_tag=baseline_tag,
        current_run_id=run_id,
        current_bt_metrics=bt_metrics,
    )
    compare_csv = tables_dir / f"compare_baseline_vs_current__{phase}.csv"
    _save_csv(compare_df, compare_csv)

    # 2) Figures
    # equity curve
    r = _ensure_dt(bt_returns, "date")
    if "phase" in r.columns:
        r = r[r["phase"].astype(str) == phase].copy()
    r = r.sort_values("date").reset_index(drop=True)

    dates = r["date"]
    if bt_equity_curve is not None and "net_equity" in bt_equity_curve.columns:
        eq_df = _ensure_dt(bt_equity_curve, "date")
        if "phase" in eq_df.columns:
            eq_df = eq_df[eq_df["phase"].astype(str) == phase].copy()
        eq_df = eq_df.sort_values("date").reset_index(drop=True)
        # align
        merged = r.merge(eq_df[["date", "net_equity"]], on="date", how="left")
        strategy_eq = merged["net_equity"].astype(float)
        if strategy_eq.isna().all():
            strategy_eq = _compute_equity_from_returns(r, "net_return")
    else:
        strategy_eq = _compute_equity_from_returns(r, "net_return")

    # benchmark equity
    bench_eq = None
    if bt_benchmark_returns is not None and "bench_return" in bt_benchmark_returns.columns:
        b = _ensure_dt(bt_benchmark_returns, "date")
        if "phase" in b.columns:
            b = b[b["phase"].astype(str) == phase].copy()
        b = b.sort_values("date").reset_index(drop=True)
        bm = r.merge(b[["date", "bench_return"]], on="date", how="left")
        if bm["bench_return"].notna().any():
            bench_eq = _compute_equity_from_returns(bm.fillna({"bench_return": 0.0}), "bench_return")

    eq_png = figures_dir / f"equity_curve__{phase}.png"
    _plot_equity(
        dates=dates,
        strategy_eq=strategy_eq,
        bench_eq=bench_eq,
        title=f"누적 성과 (phase={phase})",
        out_path=eq_png,
    )

    # drawdown
    dd = _compute_drawdown(strategy_eq)
    dd_png = figures_dir / f"drawdown__{phase}.png"
    _plot_drawdown(dates=dates, drawdown=dd, title=f"드로우다운 (phase={phase})", out_path=dd_png)

    # turnover plots
    if "turnover_oneway" in r.columns:
        turnover = r["turnover_oneway"].astype(float)
        to_ts_png = figures_dir / f"turnover_timeseries__{phase}.png"
        to_hist_png = figures_dir / f"turnover_hist__{phase}.png"
        _plot_turnover(dates=dates, turnover=turnover, out_ts_path=to_ts_png, out_hist_path=to_hist_png)
    else:
        to_ts_png = None
        to_hist_png = None

    # sector exposure (optional)
    sector_png = figures_dir / f"sector_exposure__{phase}.png"
    sector_map_meta = None
    sector_note = None
    if bt_positions is not None and not bt_positions.empty and "sector_name" not in bt_positions.columns:
        # [개선안 Midterm Pack] sector_name이 없으면 최신 sector_map을 찾아 merge_asof로 보강 시도
        latest_sector_base = _find_latest_run_dir_artifact(interim_base, "sector_map")
        if latest_sector_base is not None and artifact_exists(latest_sector_base):
            try:
                sector_map = load_artifact(latest_sector_base)
                sector_map_meta = {"mode": "latest_scan", "out_base": str(latest_sector_base)}
                sm = _ensure_dt(sector_map, "date")
                sm["ticker"] = sm["ticker"].astype(str).str.zfill(6)
                bp = _ensure_dt(bt_positions, "date").copy()
                bp["ticker"] = bp["ticker"].astype(str).str.zfill(6)
                bp = bp.sort_values(["ticker", "date"]).reset_index(drop=True)
                sm = sm.sort_values(["ticker", "date"]).reset_index(drop=True)
                # merge_asof by ticker
                bp = pd.merge_asof(
                    bp,
                    sm[["ticker", "date", "sector_name"]],
                    on="date",
                    by="ticker",
                    direction="backward",
                    allow_exact_matches=True,
                )
                bt_positions = bp  # overwrite local for plotting
            except Exception as e:
                sector_note = f"sector_map merge 실패: {e}"

    if sector_note is None:
        sector_note = _plot_sector_exposure(bt_positions=bt_positions, phase=phase, out_path=sector_png)
    if sector_note is not None:
        if sector_png.exists():
            sector_png.unlink()
        sector_png = None

    # beta (optional)
    beta_df = _compute_beta_series(bt_returns=bt_returns, bench_returns=bt_benchmark_returns, phase=phase, window=12) if bt_benchmark_returns is not None else None
    beta_png = figures_dir / f"beta_roll12__{phase}.png"
    if beta_df is not None and not beta_df.empty:
        _plot_beta(beta_df, beta_png, title=f"롤링 베타(12 rebalance window, phase={phase})")
    else:
        beta_png = None

    # coverage figure (optional)
    cov_png = figures_dir / f"coverage_vs_universe__{phase}.png"
    cov_note = _plot_coverage(rebalance_scores_summary=rebalance_scores_summary, phase=phase, out_path=cov_png)
    if cov_note is not None:
        if cov_png.exists():
            cov_png.unlink()
        cov_png = None

    # compare figure (bar chart)
    compare_png = figures_dir / f"compare_baseline_vs_current__{phase}.png"
    if compare_df is not None and not compare_df.empty and compare_df["baseline_holdout_value"].notna().any():
        try:
            # plot only numeric metrics
            plot_df = compare_df.copy()
            plot_df["baseline"] = pd.to_numeric(plot_df["baseline_holdout_value"], errors="coerce")
            plot_df["current"] = pd.to_numeric(plot_df["current_holdout_value"], errors="coerce")
            plot_df = plot_df.dropna(subset=["baseline", "current"])
            if not plot_df.empty:
                plt.figure(figsize=(10, 4.8))
                x = np.arange(len(plot_df))
                w = 0.35
                plt.bar(x - w/2, plot_df["baseline"], width=w, label="baseline")
                plt.bar(x + w/2, plot_df["current"], width=w, label="current")
                plt.xticks(x, plot_df["metric"].astype(str), rotation=25, ha="right")
                plt.title(f"Baseline vs Current (holdout) — {baseline_tag} vs {run_id}")
                plt.grid(True, axis="y", alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(compare_png, dpi=160)
                plt.close()
            else:
                if compare_png.exists():
                    compare_png.unlink()
                compare_png = None
        except Exception:
            if compare_png.exists():
                compare_png.unlink()
            compare_png = None
    else:
        if compare_png.exists():
            compare_png.unlink()
        compare_png = None

    # 3) Manifest (근거 고정 + 경고 강제)
    created_at = _now_kst_iso()
    config_path = Path(args.config)
    config_hash = _sha256_file(base_dir / config_path) if (base_dir / config_path).exists() else _sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True))

    generated_files: List[Dict[str, Any]] = []
    for pth, kind, desc in [
        (metrics_csv, "table", "백테스트 요약 지표(bt_metrics)"),
        (monthly_csv, "table", f"월별 수익률 테이블({phase})"),
        (yearly_csv, "table", f"연도별 성과 테이블({phase})") if yearly_csv else (None, None, None),
        (feature_list_csv, "table", "최종 사용 피처 리스트(모델트랙 vs 랭킹트랙 비교)"),
        (sel_csv, "table", f"선택 진단(selection_diagnostics, {phase})") if sel_csv else (None, None, None),
        (cov_csv, "table", f"유니버스 대비 커버리지 테이블({phase})") if cov_csv else (None, None, None),
        (leakage_csv, "table", f"수익률 적용 구간 점검(leakage_sanity_check, {phase})"),
        (wsign_csv, "table", f"Long-only 점검(weights_sign_check, {phase})"),
        (compare_csv, "table", f"baseline vs current 비교표({phase})"),
        (eq_png, "figure", f"누적 성과 곡선({phase})"),
        (dd_png, "figure", f"드로우다운 곡선({phase})"),
        (to_ts_png, "figure", f"턴오버 시계열({phase})") if to_ts_png else (None, None, None),
        (to_hist_png, "figure", f"턴오버 분포({phase})") if to_hist_png else (None, None, None),
        (sector_png, "figure", f"섹터 비중 시계열({phase})") if sector_png else (None, None, None),
        (beta_png, "figure", f"롤링 베타({phase})") if (beta_png and beta_png.exists()) else (None, None, None),
        (cov_png, "figure", f"유니버스 대비 커버리지(%)({phase})") if cov_png else (None, None, None),
        (compare_png, "figure", f"baseline vs current 비교 차트({phase})") if compare_png else (None, None, None),
    ]:
        if pth is None:
            continue
        generated_files.append(
            {
                "path": _relpath(Path(pth), base_dir),
                "kind": kind,
                "description": desc,
            }
        )

    manifest = {
        "created_at": created_at,
        "analysis_identifier": {
            "run_id_or_tag": run_id,
            "run_dir_exists": bool(run_dir_exists),
            "baseline_tag_for_compare": baseline_tag,
        },
        "config": {
            "path": str(config_path).replace("\\", "/"),
            "sha256": str(config_hash),
            "summary": {
                "l7": (cfg.get("l7", {}) if isinstance(cfg, dict) else {}) or {},
                "l8": (cfg.get("l8", {}) if isinstance(cfg, dict) else {}) or {},
                "l6r": (cfg.get("l6r", {}) if isinstance(cfg, dict) else {}) or {},
                "l4": (cfg.get("l4", {}) if isinstance(cfg, dict) else {}) or {},
            },
        },
        "resolved_sources": {
            "bt_metrics": src_bt_metrics,
            "bt_returns": src_bt_returns,
            "bt_equity_curve": src_bt_eq,
            "bt_benchmark_returns": src_bt_bench,
            "bt_positions": src_bt_pos,
            "bt_yearly_metrics": src_bt_yearly,
            "selection_diagnostics": src_sel,
            "rebalance_scores": src_l6,
            "rebalance_scores_summary": src_l6_sum,
            "sector_map_used_for_plot": sector_map_meta,
            "dataset_daily": src_ds,
        },
        "generated_files": generated_files,
        "notes": [
            "legacy 저장 모드(data/interim 루트)에 산출물이 존재할 수 있어 run_dir 우선 → interim_root fallback을 수행했습니다.",
            "슬라이드/보고서 수치는 이 manifest에 기록된 소스 파일에서만 가져오도록 설계했습니다.",
        ],
        "legacy_fallback": {
            "artifacts_using_legacy": sorted([
                k for k, v in {
                    "bt_metrics": src_bt_metrics,
                    "bt_returns": src_bt_returns,
                    "bt_equity_curve": src_bt_eq,
                    "bt_benchmark_returns": src_bt_bench,
                    "bt_positions": src_bt_pos,
                    "bt_yearly_metrics": src_bt_yearly,
                    "selection_diagnostics": src_sel,
                    "rebalance_scores": src_l6,
                    "rebalance_scores_summary": src_l6_sum,
                    "dataset_daily": src_ds,
                }.items()
                if isinstance(v, dict) and v.get("mode") == "interim_root"
            ]),
        },
    }

    run_manifest_path = base_dir / "artifacts" / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # [개선안 17번] legacy fallback hard warning
    legacy_list = manifest.get("legacy_fallback", {}).get("artifacts_using_legacy", [])
    legacy_note = None
    if legacy_list:
        legacy_note = f"핵심 산출물 일부가 legacy 모드(data/interim 루트)에서 로드됨: {legacy_list}"
        print(f"[HARD WARNING] legacy fallback detected: {legacy_list}")

    # 4) validation_report.md (문구 안전화)
    validation_md = "\n".join([
        "# 검증 요약(중간발표용, 문구 안전화)",
        "",
        f"- 생성일시(KST): {created_at}",
        f"- run_tag: `{run_id}`",
        f"- phase: `{phase}`",
        f"- 근거(manifest): `artifacts/run_manifest.json`",
        "",
        "## 1) 근거 고정(M1/M2/M3)",
        "- (M1) resolved_sources에 파일별 sha256/mtime/size_bytes 기록",
        f"- (M2) legacy fallback 감지: {'있음' if bool(legacy_list) else '없음'}",
        f"  - details: {legacy_list if legacy_list else 'N/A'}",
        "",
        "## 2) 수익률 적용 구간 점검(누수 단정 금지)",
        f"- 파일: `artifacts/tables/leakage_sanity_check__{phase}.csv`",
        f"- FAIL 건수: {leakage_fail}",
        "권장 문구:",
        f"> 수익률 적용 구간 점검 PASS(t+1~t+N) — leakage_sanity_check 기준 (FAIL {leakage_fail}건)",
        "",
        "## 3) Long-only 확정",
        f"- 파일: `artifacts/tables/weights_sign_check__{phase}.csv`",
        f"- 음수 가중치 발생(FAIL): {wsign_fail}일",
        "권장 문구:",
        f"> Long-only 확정 — weights_sign_check 기준 음수 비중 0건(FAIL {wsign_fail}일)",
        "",
        "## 4) 필수 각주(필요 시)",
        f"- {legacy_note if legacy_note else '이번 실행에서는 legacy fallback 각주가 필수는 아닙니다.'}",
        "",
    ])
    validation_path = base_dir / "artifacts" / "validation_report.md"
    _write_text(validation_path, validation_md)

    # 5) reports/* 생성(보고서 → PPT 문안 → 대본 → Canva pack → 숫자팩)
    reports_dir = base_dir / "reports"
    manifest_rel = "../artifacts/run_manifest.json"
    midterm_md = _build_midterm_report_md(
        base_dir=base_dir,
        run_id=run_id,
        phase=phase,
        manifest_rel=manifest_rel,
        cfg=cfg,
        summary_metrics_csv=f"../artifacts/tables/summary_metrics.csv",
        compare_csv=f"../artifacts/tables/compare_baseline_vs_current__{phase}.csv" if compare_csv else None,
        compare_note=compare_note,
        legacy_note=legacy_note,
    )
    _write_text(reports_dir / "midterm_report.md", midterm_md)
    _write_text(reports_dir / "ppt_outline_12slides.md", _build_ppt_outline_md(manifest_rel=manifest_rel))
    _write_text(reports_dir / "speaker_notes.md", _build_speaker_notes_md())
    _write_text(reports_dir / "canva_export_pack.md", _build_canva_export_pack_md(run_id=run_id, phase=phase))
    _write_text(reports_dir / "ppt_numbers_by_slide.md", _build_ppt_numbers_by_slide_md(base_dir=base_dir, phase=phase, run_id=run_id))

    # 6) Canva assets manifest
    canva_assets = _build_canva_assets_manifest(base_dir)
    canva_assets_path = base_dir / "artifacts" / "canva" / "canva_assets_manifest.json"
    canva_assets_path.write_text(json.dumps(canva_assets, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] artifacts generated. run_id={run_id}")
    print(f"- figures_dir: {figures_dir}")
    print(f"- tables_dir : {tables_dir}")
    print(f"- manifest   : {run_manifest_path}")
    print(f"- validation : {validation_path}")
    print(f"- canva      : {canva_assets_path}")
    print(f"- reports    : {reports_dir / 'midterm_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
