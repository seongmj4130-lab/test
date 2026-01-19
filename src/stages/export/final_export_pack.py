# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/export/final_export_pack.py
"""
[Stage12] Final Export Pack
발표자료에 바로 넣을 수 있는 최종 export 패키지 생성

입력:
- reports/history/history_manifest.parquet
- data/interim/{baseline_tag}/ui_equity_curves.parquet
- data/interim/{baseline_tag}/ui_top_bottom_daily.parquet
- (있으면) reports/ranking/sector_concentration__*.csv
- reports/ui/ui_snapshot__{baseline_tag}.csv

출력:
- timeline_ppt.csv (변천사 타임라인)
- kpi_onepager.csv/.md (핵심 KPI 1장)
- latest_snapshot.csv (Top/Bottom/Regime/TopFeatures)
- equity_curves.csv (전략/벤치/초과)
- appendix_sources.md (사용 파일 경로/해시 목록)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def get_file_hash(file_path: Path) -> str:
    """파일 해시 계산 (SHA256)"""
    if not file_path.exists():
        return "N/A"
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def build_timeline_ppt(history_manifest: pd.DataFrame) -> pd.DataFrame:
    """
    history_manifest 기반 변천사 타임라인 생성 (PPT용)
    """
    if history_manifest.empty:
        return pd.DataFrame(columns=["created_at", "stage_no", "track", "run_tag", "change_title",
                                     "holdout_sharpe", "holdout_mdd", "holdout_cagr", "ppt_one_liner"])

    # 필요한 컬럼만 선택
    cols = ["created_at", "stage_no", "track", "run_tag", "change_title",
            "holdout_sharpe", "holdout_mdd", "holdout_cagr", "ppt_one_liner"]
    available_cols = [c for c in cols if c in history_manifest.columns]

    timeline = history_manifest[available_cols].copy()

    # 날짜 정렬 (최신순)
    if "created_at" in timeline.columns:
        timeline["created_at"] = pd.to_datetime(timeline["created_at"], errors="coerce")
        timeline = timeline.sort_values("created_at", ascending=False)

    # 결측값 처리
    timeline = timeline.fillna("")

    return timeline

def build_kpi_onepager(
    kpi_table: pd.DataFrame,
    baseline_tag: str,
) -> Tuple[pd.DataFrame, str]:
    """
    최종 run_tag 기준 핵심 지표 1장 KPI 요약 생성

    Returns:
        (kpi_df, kpi_md) 튜플
    """
    if kpi_table.empty:
        return pd.DataFrame(), ""

    # 핵심 KPI만 선택
    key_metrics = [
        "holdout_sharpe",
        "holdout_mdd",
        "holdout_cagr",
        "holdout_total_return",
        "net_sharpe",
        "net_mdd",
        "net_total_return",
        "information_ratio",
        "tracking_error_ann",
        "avg_turnover_oneway",
    ]

    available_metrics = [m for m in key_metrics if m in kpi_table.columns]

    if not available_metrics:
        # 모든 컬럼 사용
        kpi_df = kpi_table.copy()
    else:
        kpi_df = kpi_table[available_metrics].copy()

    # Markdown 생성
    md_lines = [
        f"# 핵심 KPI 요약 (Run Tag: {baseline_tag})",
        "",
        "## 성과 지표",
        "",
    ]

    if not kpi_df.empty:
        # 숫자 포맷팅
        for col in kpi_df.columns:
            if kpi_df[col].dtype in [np.float64, np.float32]:
                val = kpi_df[col].iloc[0] if len(kpi_df) > 0 else np.nan
                if not pd.isna(val):
                    if "sharpe" in col.lower() or "ratio" in col.lower():
                        md_lines.append(f"- **{col}**: {val:.3f}")
                    elif "mdd" in col.lower() or "return" in col.lower() or "cagr" in col.lower():
                        md_lines.append(f"- **{col}**: {val:.2f}%")
                    elif "turnover" in col.lower():
                        md_lines.append(f"- **{col}**: {val:.2f}%")
                    else:
                        md_lines.append(f"- **{col}**: {val:.4f}")

    md_content = "\n".join(md_lines)

    return kpi_df, md_content

def build_latest_snapshot(
    ui_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    최신일 Top/Bottom + 설명 + 시장국면 스냅샷 생성
    """
    if ui_snapshot.empty:
        return pd.DataFrame()

    # 최신일 기준으로 필터링 (이미 최신일만 있을 수도 있음)
    if "snapshot_date" in ui_snapshot.columns:
        latest_date = ui_snapshot["snapshot_date"].max()
        snapshot = ui_snapshot[ui_snapshot["snapshot_date"] == latest_date].copy()
    else:
        snapshot = ui_snapshot.copy()

    # 필요한 컬럼만 선택
    cols = [
        "snapshot_date", "snapshot_type", "snapshot_rank", "ticker",
        "rank_total", "score_total", "regime_label", "regime_score",
        "top_features", "contrib_core", "contrib_fundamental", "contrib_other"
    ]

    available_cols = [c for c in cols if c in snapshot.columns]
    result = snapshot[available_cols].copy()

    return result

def build_equity_curves_csv(
    ui_equity_curves: pd.DataFrame,
) -> pd.DataFrame:
    """
    그래프용 시계열 CSV: equity curves (전략/벤치/초과)
    """
    if ui_equity_curves.empty:
        return pd.DataFrame()

    # 필요한 컬럼만 선택
    cols = ["date", "strategy_equity", "bench_equity", "excess_equity",
            "strategy_ret", "bench_ret"]

    available_cols = [c for c in cols if c in ui_equity_curves.columns]
    result = ui_equity_curves[available_cols].copy()

    # 날짜 정렬
    if "date" in result.columns:
        result = result.sort_values("date")

    return result

def find_sector_concentration_file(
    project_root: Path,
    baseline_tag: Optional[str] = None,
) -> Optional[Path]:
    """
    섹터 농도 파일 찾기 (최신 또는 baseline_tag 기준)
    """
    sector_dir = project_root / "reports" / "ranking"

    if baseline_tag:
        # baseline_tag 기반 파일 찾기
        pattern = f"sector_concentration__{baseline_tag}.csv"
        candidate = sector_dir / pattern
        if candidate.exists():
            return candidate

    # 최신 파일 찾기
    candidates = list(sector_dir.glob("sector_concentration__*.csv"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    return None

def build_appendix_sources(
    project_root: Path,
    baseline_tag: str,
    files_used: List[Tuple[str, Path]],
) -> str:
    """
    사용 파일 경로/해시 목록 생성 (appendix_sources.md)
    """
    lines = [
        "# Appendix: 사용 파일 목록",
        "",
        f"**Run Tag**: `{baseline_tag}`",
        f"**생성 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 입력 파일",
        "",
        "| 파일명 | 경로 | 해시 (SHA256) |",
        "|--------|------|---------------|",
    ]

    for name, file_path in files_used:
        rel_path = file_path.relative_to(project_root) if file_path.is_relative_to(project_root) else str(file_path)
        file_hash = get_file_hash(file_path)
        hash_short = file_hash[:16] + "..." if len(file_hash) > 16 else file_hash
        lines.append(f"| {name} | `{rel_path}` | `{hash_short}` |")

    lines.extend([
        "",
        "## 출력 파일",
        "",
        f"| 파일명 | 경로 |",
        "|--------|------|",
        f"| timeline_ppt.csv | `artifacts/reports/final_export/{baseline_tag}/timeline_ppt.csv` |",
        f"| kpi_onepager.csv | `artifacts/reports/final_export/{baseline_tag}/kpi_onepager.csv` |",
        f"| kpi_onepager.md | `artifacts/reports/final_export/{baseline_tag}/kpi_onepager.md` |",
        f"| latest_snapshot.csv | `artifacts/reports/final_export/{baseline_tag}/latest_snapshot.csv` |",
        f"| equity_curves.csv | `artifacts/reports/final_export/{baseline_tag}/equity_curves.csv` |",
        f"| appendix_sources.md | `artifacts/reports/final_export/{baseline_tag}/appendix_sources.md` |",
    ])

    return "\n".join(lines)

def run_L12_final_export(
    cfg: dict,
    artifacts: dict,
    *,
    project_root: Path,
    baseline_tag: str,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [Stage12] Final Export Pack 실행

    Args:
        cfg: 설정 딕셔너리
        artifacts: 이전 스테이지 산출물 딕셔너리
        project_root: 프로젝트 루트 경로
        baseline_tag: Stage11 run_tag (baseline)
        force: 강제 재생성 플래그

    Returns:
        (outputs, warnings) 튜플
        - outputs: {"timeline_ppt": DataFrame, "kpi_onepager": DataFrame, ...}
        - warnings: 경고 메시지 리스트
    """
    warns: list[str] = []
    files_used: List[Tuple[str, Path]] = []

    # 1. History Manifest 로드
    history_manifest_path = project_root / "reports" / "history" / "history_manifest.parquet"
    if not history_manifest_path.exists():
        # CSV로 시도
        history_manifest_path = project_root / "reports" / "history" / "history_manifest.csv"
        if not history_manifest_path.exists():
            warns.append(f"History manifest 파일 없음: {history_manifest_path}")
            history_manifest = pd.DataFrame()
        else:
            history_manifest = pd.read_csv(history_manifest_path)
            files_used.append(("history_manifest", history_manifest_path))
    else:
        history_manifest = pd.read_parquet(history_manifest_path)
        files_used.append(("history_manifest", history_manifest_path))

    # 2. UI Equity Curves 로드
    ui_equity_path = project_root / "data" / "interim" / baseline_tag / "ui_equity_curves.parquet"
    if not ui_equity_path.exists():
        warns.append(f"UI equity curves 파일 없음: {ui_equity_path}")
        ui_equity_curves = pd.DataFrame()
    else:
        ui_equity_curves = pd.read_parquet(ui_equity_path)
        files_used.append(("ui_equity_curves", ui_equity_path))

    # 3. UI Snapshot 로드
    ui_snapshot_path = project_root / "reports" / "ui" / f"ui_snapshot__{baseline_tag}.csv"
    if not ui_snapshot_path.exists():
        warns.append(f"UI snapshot 파일 없음: {ui_snapshot_path}")
        ui_snapshot = pd.DataFrame()
    else:
        ui_snapshot = pd.read_csv(ui_snapshot_path)
        files_used.append(("ui_snapshot", ui_snapshot_path))

    # 4. KPI Table 로드
    kpi_table_path = project_root / "reports" / "kpi" / f"kpi_table__{baseline_tag}.csv"
    if not kpi_table_path.exists():
        warns.append(f"KPI table 파일 없음: {kpi_table_path}")
        kpi_table = pd.DataFrame()
    else:
        kpi_table = pd.read_csv(kpi_table_path)
        files_used.append(("kpi_table", kpi_table_path))

    # 5. Sector Concentration 파일 찾기 (선택)
    sector_concentration_path = find_sector_concentration_file(project_root, baseline_tag)
    if sector_concentration_path:
        files_used.append(("sector_concentration", sector_concentration_path))

    # 6. 각종 데이터 생성
    timeline_ppt = build_timeline_ppt(history_manifest)
    kpi_onepager_df, kpi_onepager_md = build_kpi_onepager(kpi_table, baseline_tag)
    latest_snapshot = build_latest_snapshot(ui_snapshot)
    equity_curves_csv = build_equity_curves_csv(ui_equity_curves)

    # 7. Appendix Sources 생성
    appendix_sources_md = build_appendix_sources(project_root, baseline_tag, files_used)

    outputs = {
        "timeline_ppt": timeline_ppt,
        "kpi_onepager": kpi_onepager_df,
        "kpi_onepager_md": kpi_onepager_md,  # 문자열이지만 outputs에 포함
        "latest_snapshot": latest_snapshot,
        "equity_curves": equity_curves_csv,
        "appendix_sources_md": appendix_sources_md,  # 문자열
    }

    return outputs, warns
