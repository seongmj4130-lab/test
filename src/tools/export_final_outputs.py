# -*- coding: utf-8 -*-
from __future__ import annotations

"""
[개선안 40번] 06_code22 최종 산출물 Export(정리) 도구

목표:
- 투트랙 실행 결과 중 "최종 산출물"만 골라서 별도 폴더(기본: ../06_code22/final_outputs/LATEST)에 저장
- 기존 결과는 LATEST 폴더를 비워서 "이번 실행 결과만" 유지
- manifest.json/summary.md를 함께 저장하여 재현 가능한 결과 스냅샷 제공

주의:
- 이 모듈은 데이터를 재계산하지 않습니다. (실행은 run_two_track_and_export.py에서 수행)
- 이 모듈은 data/interim의 캐시(ohlcv/dataset 등)는 기본적으로 Export하지 않습니다.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import get_path, load_config

FINAL_INTERIM_PREFIXES: Tuple[str, ...] = (
    # Track A (최종)
    "ranking_short_daily",
    "ranking_long_daily",
    # Track B (중간/최종)
    "rebalance_scores_from_ranking_interval_",
    "bt_positions_",
    "bt_returns_",
    "bt_equity_curve_",
    "bt_metrics_",
    "bt_regime_metrics_",
    # 진단(있으면 같이)
    "selection_diagnostics_",
    "bt_returns_diagnostics_",
    "runtime_profile_",
)

DEFAULT_STRATEGIES: Tuple[str, ...] = ("bt20_short", "bt20_ens", "bt120_long", "bt120_ens")


@dataclass(frozen=True)
class ExportResult:
    """
    [개선안 40번] Export 결과

    Args:
        out_dir: 실제 Export된 폴더
        copied_files: 복사된 파일(absolute path)
        manifest_path: manifest.json 경로
        summary_path: summary.md 경로
    """

    out_dir: Path
    copied_files: List[Path]
    manifest_path: Path
    summary_path: Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_rmtree_children(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for p in dir_path.iterdir():
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception:
            # Windows file lock 등 예상 이슈: 최대한 진행
            pass


def _should_export_interim_stem(stem: str) -> bool:
    s = str(stem)
    return any(s.startswith(prefix) for prefix in FINAL_INTERIM_PREFIXES)


def _copy_with_suffixes(src_base: Path, dst_dir: Path) -> List[Path]:
    """
    out_base(.parquet/.csv) + 선택적으로 __meta.json까지 복사한다.
    """
    copied: List[Path] = []
    for ext in (".parquet", ".csv"):
        src = src_base.with_suffix(ext)
        if src.exists():
            dst = dst_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)

    meta = src_base.with_suffix(".__meta.json")
    if meta.exists():
        dst = dst_dir / meta.name
        shutil.copy2(meta, dst)
        copied.append(dst)
    return copied


def _read_metrics(interim_dir: Path, strategy: str) -> pd.DataFrame:
    p = interim_dir / f"bt_metrics_{strategy}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _pick_phase_row(df: pd.DataFrame, phase: str) -> Dict[str, object]:
    if df is None or df.empty or "phase" not in df.columns:
        return {}
    sub = df[df["phase"].astype(str) == phase]
    if sub.empty:
        return {}
    return dict(sub.iloc[0])


def _fmt_num(x: object, nd: int = 4) -> str:
    try:
        fx = float(x)
        if pd.isna(fx):
            return ""
        return f"{fx:.{nd}f}"
    except Exception:
        return ""

def _jsonable(x: object) -> object:
    """
    [개선안 40번] json 직렬화 가능 타입으로 변환

    pandas/numpy scalar(int64/float64/Timestamp 등)로 인해 manifest 저장이 실패하는 문제를 방지한다.
    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    # pandas timestamp
    if isinstance(x, pd.Timestamp):
        try:
            return x.isoformat()
        except Exception:
            return str(x)
    # NaN/NaT 처리
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    # numpy/pandas scalar -> python scalar
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    # 마지막 fallback
    return str(x)


def _jsonify_dict(d: Dict[str, object]) -> Dict[str, object]:
    return {str(k): _jsonable(v) for k, v in (d or {}).items()}


def _build_summary_md(interim_dir: Path, strategies: Tuple[str, ...]) -> str:
    """
    [개선안 40번][최종 수치셋] Export 폴더에 함께 넣을 간단 요약 리포트 생성

    Returns:
        markdown str
    """
    headline = [
        ("net_sharpe", "Net Sharpe Ratio"),
        ("net_total_return", "Net Total Return"),
        ("net_cagr", "Net CAGR"),
        ("net_mdd", "Net MDD"),
        ("net_calmar_ratio", "Calmar Ratio"),
    ]
    alpha = [
        ("ic", "IC"),
        ("rank_ic", "Rank IC"),
        ("icir", "ICIR"),
        ("long_short_alpha", "Long/Short Alpha"),
    ]
    ops = [
        ("avg_turnover_oneway", "Avg Turnover"),
        ("net_hit_ratio", "Hit Ratio"),
        ("net_profit_factor", "Profit Factor"),
        ("avg_trade_duration", "Avg Trade Duration"),
    ]

    def _table(rows: List[Dict[str, str]], cols: List[str]) -> str:
        if not rows:
            return "_(데이터 없음)_\n"
        out = []
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            out.append("| " + " | ".join([str(r.get(c, "")) for c in cols]) + " |")
        return "\n".join(out) + "\n"

    def _section(title: str, cols: List[Tuple[str, str]]) -> str:
        rows: List[Dict[str, str]] = []
        for s in strategies:
            df = _read_metrics(interim_dir, s)
            dev = _pick_phase_row(df, "dev")
            hold = _pick_phase_row(df, "holdout")
            for ph, row in [("Dev", dev), ("Holdout", hold)]:
                r: Dict[str, str] = {"전략": s, "구간": ph}
                for key, label in cols:
                    r[label] = _fmt_num(row.get(key) if row else None, nd=4)
                rows.append(r)
        return f"## {title}\n\n" + _table(rows, ["전략", "구간"] + [lab for _, lab in cols]) + "\n"

    md: List[str] = []
    md.append("# Two-Track 최종 산출물 요약 (Export Snapshot)")
    md.append("")
    md.append("- 근거: `data/interim/bt_metrics_{strategy}.csv`, `data/interim/bt_regime_metrics_{strategy}.*`")
    md.append("")
    md.append(_section("1) 핵심 성과 (Headline Metrics)", headline))
    md.append(_section("2) 모델 예측력 (Alpha Quality)", alpha))
    md.append(_section("3) 운용 안정성 (Operational Viability)", ops))
    md.append("## 4) 국면별 성과 (Regime Robustness)\n")
    md.append("- Export에는 `bt_regime_metrics_{strategy}` 파일을 함께 포함합니다.\n")
    return "\n".join(md)


def export_final_outputs(
    *,
    config_path: str = "configs/config.yaml",
    dest_root: Optional[str] = None,
    clean_latest: bool = True,
    mode: str = "latest",  # latest | runs
    run_tag: Optional[str] = None,
    strategies: Tuple[str, ...] = DEFAULT_STRATEGIES,
) -> ExportResult:
    """
    [개선안 40번] 최종 산출물 Export 실행

    Args:
        config_path: config.yaml 경로
        dest_root: 목적지 루트(예: ../06_code22). None이면 base_dir의 상위에 있는 06_code22를 사용.
        clean_latest: mode=latest일 때 LATEST 폴더를 비워 "이번 실행 결과만" 남김
        mode: "latest"면 final_outputs/LATEST에 덮어쓰기, "runs"면 final_outputs/runs/{run_tag}에 저장
        run_tag: mode=runs일 때 사용할 태그 (None이면 자동 생성)
        strategies: manifest/summary 생성 시 참조할 전략 목록

    Returns:
        ExportResult

    Example:
        from src.tools.export_final_outputs import export_final_outputs
        export_final_outputs(config_path="configs/config.yaml")
    """
    cfg = load_config(config_path)
    base_dir = Path(get_path(cfg, "base_dir"))
    interim_dir = Path(get_path(cfg, "data_interim"))
    reports_dir = Path(get_path(cfg, "artifacts_reports"))

    dest_root_path = Path(dest_root) if dest_root else (base_dir.parent / "06_code22")
    final_root = dest_root_path / "final_outputs"

    if mode.strip().lower() == "runs":
        tag = run_tag or _now_tag()
        out_dir = final_root / "runs" / tag
    else:
        out_dir = final_root / "LATEST"
        if clean_latest:
            _safe_rmtree_children(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []

    # 1) config 스냅샷
    cfg_src = base_dir / "configs" / "config.yaml"
    if cfg_src.exists():
        cfg_dst = out_dir / "config_snapshot.yaml"
        shutil.copy2(cfg_src, cfg_dst)
        copied.append(cfg_dst)

    # 2) interim 최종 산출물만 복사
    if interim_dir.exists():
        for p in interim_dir.iterdir():
            if not p.is_file():
                continue
            # out_base 기준으로 stem/접두어 필터링 (csv/parquet/meta 모두 처리)
            # 예: bt_metrics_bt20_ens.csv → stem=bt_metrics_bt20_ens
            stem = p.stem
            if "__meta" in stem:
                stem = stem.replace("__meta", "")
            if not _should_export_interim_stem(stem):
                continue
            src_base = interim_dir / stem
            copied.extend(_copy_with_suffixes(src_base, out_dir / "data_interim"))

    # 3) reports (요약표 등)
    report_candidates = [
        reports_dir / "track_b_4strategy_final_summary.md",
    ]
    for rp in report_candidates:
        if rp.exists():
            dst = out_dir / "reports" / rp.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rp, dst)
            copied.append(dst)

    # 4) summary.md (경량 요약)
    summary_text = _build_summary_md(interim_dir=interim_dir, strategies=strategies)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    copied.append(summary_path)

    # 5) manifest.json
    # (전략별 Dev/Holdout 핵심 지표 요약 + 복사 파일 리스트)
    metrics_summary: Dict[str, Dict[str, Dict[str, object]]] = {}
    for s in strategies:
        df = _read_metrics(interim_dir, s)
        metrics_summary[s] = {
            "dev": _jsonify_dict(_pick_phase_row(df, "dev")),
            "holdout": _jsonify_dict(_pick_phase_row(df, "holdout")),
        }

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "run_tag": (run_tag if run_tag else ("LATEST" if mode != "runs" else None)),
        "source_base_dir": str(base_dir),
        "source_interim_dir": str(interim_dir),
        "source_reports_dir": str(reports_dir),
        "dest_out_dir": str(out_dir),
        "strategies": list(strategies),
        "metrics_summary": metrics_summary,
        "files": [
            {
                "path": str(p.relative_to(out_dir)) if p.exists() else str(p),
                "size_bytes": (p.stat().st_size if p.exists() else None),
                "modified_at": (datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.exists() else None),
            }
            for p in copied
        ],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    copied.append(manifest_path)

    return ExportResult(
        out_dir=out_dir,
        copied_files=copied,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
