# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/check_stage_completion.py
"""
Stage 완료 점검 스크립트
필수 산출물 존재/스키마/기간/행수/결측률/설정값을 검사하고 PASS/FAIL 판정
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def get_path(cfg: dict, key: str) -> Path:
    """config에서 경로 추출"""
    paths = cfg.get("paths", {})
    base_dir = Path(paths.get("base_dir", Path.cwd()))
    path_template = paths.get(key, "")
    if path_template:
        return Path(path_template.format(base_dir=base_dir))
    return base_dir / key.replace("_", "/")

def get_file_hash(filepath: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_l2_protection(base_interim_dir: Path, baseline_tag: str, log_file: Path = None) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    L2 파일 보호 확인 (fundamentals_annual.parquet가 변경되지 않았는지)

    Returns:
        (is_valid, message, hash_before, hash_after)
    """
    l2_file = base_interim_dir / "fundamentals_annual.parquet"
    if not l2_file.exists():
        return False, "L2 파일이 존재하지 않습니다", None, None

    current_hash = get_file_hash(l2_file)

    # 로그 파일에서 실행 전 해시 읽기 시도
    hash_before = None
    hash_after = current_hash

    if log_file and log_file.exists():
        try:
            log_content = log_file.read_text(encoding='utf-8', errors='ignore')
            # "실행 전 해시:" 또는 "L2 검증] 실행 전 해시:" 패턴 찾기
            import re
            match = re.search(r'실행\s*전\s*해시[:\s]+([0-9a-f]{16})', log_content, re.IGNORECASE)
            if match:
                hash_before = match.group(1) + "..."  # 짧은 버전
        except Exception:
            pass

    # baseline의 L2 파일 해시와 비교 (baseline이 있으면)
    baseline_l2 = base_interim_dir / baseline_tag / "fundamentals_annual.parquet"
    if baseline_l2.exists():
        baseline_hash = get_file_hash(baseline_l2)
        if baseline_hash != current_hash:
            return False, f"L2 파일이 변경되었습니다 (baseline 해시: {baseline_hash[:16]}..., 현재 해시: {current_hash[:16]}...)", hash_before, hash_after[:16] + "..."

    # 파일 크기와 수정 시간 확인
    file_size = l2_file.stat().st_size
    mtime = datetime.fromtimestamp(l2_file.stat().st_mtime)

    hash_info = f"해시: {current_hash[:16]}..."
    if hash_before:
        hash_info = f"실행 전: {hash_before}, 실행 후: {hash_after[:16]}..."

    return True, f"L2 파일 보호 확인: 크기={file_size:,} bytes, 수정시간={mtime.strftime('%Y-%m-%d %H:%M:%S')}, {hash_info}", hash_before, hash_after[:16] + "..."

def check_stage_artifacts(
    stage: int,
    run_tag: str,
    base_interim_dir: Path,
    cfg: dict
) -> Dict:
    """
    Stage별 필수 산출물 체크
    """
    results = {
        "stage": stage,
        "run_tag": run_tag,
        "checks": [],
        "pass": True,
        "warnings": [],
        "gate_notes": [],
    }

    interim_dir = base_interim_dir / run_tag

    # Stage별 필수 산출물 정의
    stage_artifacts = {
        0: ["universe_k200_membership_monthly"],
        1: ["ohlcv_daily"],
        2: ["fundamentals_annual"],  # base_interim_dir에 있음
        3: ["panel_merged_daily"],
        4: ["dataset_daily", "cv_folds_short", "cv_folds_long"],
        5: ["pred_short_oos", "pred_long_oos", "model_metrics"],
        6: ["rebalance_scores", "rebalance_scores_summary"],
        7: ["ranking_daily"],  # Stage7: 랭킹 baseline
        8: ["ranking_daily"],  # Stage8: sector-relative 정규화
        9: ["ranking_daily"],  # Stage9: 설명가능성 확장
        10: ["market_regime_daily", "ranking_daily"],  # Stage10: 시장 국면 지표 추가
        11: ["ui_top_bottom_daily", "ui_equity_curves"],  # Stage11: UI Payload Builder
        12: ["timeline_ppt", "kpi_onepager", "latest_snapshot", "equity_curves", "appendix_sources"],  # Stage12: Final Export Pack
        13: ["bt_positions", "bt_returns", "bt_equity_curve", "bt_metrics", "selection_diagnostics"],  # Stage13: K_eff 복원
    }

    required = stage_artifacts.get(stage, [])

    for artifact_name in required:
        if stage == 2:
            # L2는 base_interim_dir에 있음
            artifact_path = base_interim_dir / f"{artifact_name}.parquet"
        elif stage == 12:
            # Stage12는 artifacts/reports/final_export/{run_tag}/ 에 있음
            script_dir = Path(__file__).resolve().parent
            base_dir = script_dir.parent.parent
            export_dir = base_dir / "artifacts" / "reports" / "final_export" / run_tag
            # 파일 확장자 결정
            if artifact_name == "appendix_sources":
                artifact_path = export_dir / f"{artifact_name}.md"
            elif artifact_name == "kpi_onepager":
                # CSV와 MD 둘 다 체크
                artifact_path = export_dir / f"{artifact_name}.csv"
            else:
                artifact_path = export_dir / f"{artifact_name}.csv"
        else:
            artifact_path = interim_dir / f"{artifact_name}.parquet"

        exists = artifact_path.exists()
        check_info = {
            "artifact": artifact_name,
            "exists": exists,
            "path": str(artifact_path),
        }

        if not exists:
            results["pass"] = False
            results["warnings"].append(f"필수 산출물 없음: {artifact_name}")
            results["checks"].append(check_info)
            continue

        # 저장 경로 확인 (Stage7)
        if stage == 7:
            expected_path = base_interim_dir / run_tag / f"{artifact_name}.parquet"
            if str(artifact_path) != str(expected_path):
                results["pass"] = False
                results["gate_notes"].append(f"저장 경로 불일치: {artifact_path} (예상: {expected_path})")

        # 스키마 체크
        try:
            if stage == 12:
                # Stage12는 CSV/MD 파일
                if artifact_path.suffix == ".md":
                    # MD 파일은 존재만 확인
                    df = pd.DataFrame()  # 빈 DataFrame
                else:
                    # CSV 파일 읽기
                    df = pd.read_csv(artifact_path, low_memory=False)
            else:
                df = pd.read_parquet(artifact_path)

            # 기본 스키마 체크
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                date_range = df["date"].dropna()
                if len(date_range) > 0:
                    date_min = date_range.min()
                    date_max = date_range.max()
                    check_info["date_range"] = {
                        "start": str(date_min),
                        "end": str(date_max),
                        "n_dates": len(date_range.unique()),
                    }

                    # Stage7: 날짜 범위 검증 강화
                    if stage == 7:
                        expected_start = pd.Timestamp("2015-01-30")
                        expected_end = pd.Timestamp("2024-12-30")
                        if date_min > expected_start:
                            results["gate_notes"].append(
                                f"날짜 범위 시작일이 늦음: {date_min.strftime('%Y-%m-%d')} "
                                f"(예상: {expected_start.strftime('%Y-%m-%d')} 이전)"
                            )
                        if date_max < expected_end:
                            results["gate_notes"].append(
                                f"날짜 범위 종료일이 빠름: {date_max.strftime('%Y-%m-%d')} "
                                f"(예상: {expected_end.strftime('%Y-%m-%d')} 이후)"
                            )

            check_info["shape"] = df.shape
            check_info["columns"] = list(df.columns)

            # Stage7/Stage8/Stage9/Stage10: ranking_daily 특화 검증
            if (stage == 7 or stage == 8 or stage == 9 or stage == 10) and artifact_name == "ranking_daily":
                # 필수 컬럼 확인
                required_cols = ["date", "ticker", "score_total", "rank_total"]
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    results["pass"] = False
                    results["warnings"].append(f"ranking_daily 필수 컬럼 누락: {missing_cols}")

                # score 결측률 체크
                if "score_total" in df.columns:
                    score_missing_pct = df["score_total"].isna().sum() / len(df) * 100
                    check_info["score_missing_pct"] = score_missing_pct
                    if score_missing_pct > 1.0:
                        results["gate_notes"].append(
                            f"score_total 결측률이 높음: {score_missing_pct:.2f}% (권장: <= 1%)"
                        )

                # rank 중복/누락 체크 (동일 date 내)
                if "rank_total" in df.columns and "date" in df.columns:
                    rank_issues = []
                    for date, group in df.groupby("date"):
                        ranks = group["rank_total"].dropna()
                        if len(ranks) == 0:
                            continue

                        # 중복 체크
                        duplicates = ranks.duplicated().sum()
                        if duplicates > 0:
                            rank_issues.append(f"{date.strftime('%Y-%m-%d')}: rank 중복 {duplicates}건")

                        # 연속성 체크 (1부터 시작하는지)
                        unique_ranks = sorted(ranks.unique())
                        if len(unique_ranks) > 0:
                            expected_min = 1
                            expected_max = len(unique_ranks)
                            if unique_ranks[0] != expected_min:
                                rank_issues.append(f"{date.strftime('%Y-%m-%d')}: rank 시작값 불일치 (실제: {unique_ranks[0]}, 예상: {expected_min})")
                            if unique_ranks[-1] != expected_max:
                                rank_issues.append(f"{date.strftime('%Y-%m-%d')}: rank 종료값 불일치 (실제: {unique_ranks[-1]}, 예상: {expected_max})")

                    if rank_issues:
                        results["gate_notes"].append(f"rank 검증 이슈: {rank_issues[:5]}")  # 최대 5개만 표시
                        check_info["rank_issues"] = len(rank_issues)
                    else:
                        check_info["rank_issues"] = 0

                # Stage9: 설명가능성 컬럼 확인
                if stage == 9:
                    contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
                    has_top_features = "top_features" in df.columns

                    if len(contrib_cols) == 0:
                        results["gate_notes"].append("Stage9: contrib_* 컬럼이 없습니다")
                    if not has_top_features:
                        results["gate_notes"].append("Stage9: top_features 컬럼이 없습니다")

                    check_info["explainability_cols"] = {
                        "contrib_cols": len(contrib_cols),
                        "has_top_features": has_top_features,
                    }

                # Stage10: regime 컬럼 확인
                if stage == 10 and artifact_name == "ranking_daily":
                    has_regime_score = "regime_score" in df.columns
                    has_regime_label = "regime_label" in df.columns

                    if not has_regime_score:
                        results["gate_notes"].append("Stage10: regime_score 컬럼이 없습니다")
                    if not has_regime_label:
                        results["gate_notes"].append("Stage10: regime_label 컬럼이 없습니다")

                    check_info["regime_cols"] = {
                        "has_regime_score": has_regime_score,
                        "has_regime_label": has_regime_label,
                    }

            # 결측률 체크
            if len(df) > 0:
                missing_pct = df.isnull().sum() / len(df) * 100
                high_missing = missing_pct[missing_pct > 50].to_dict()
                if high_missing:
                    results["warnings"].append(f"{artifact_name} 고결측률 컬럼: {high_missing}")

            results["checks"].append(check_info)

        except Exception as e:
            results["pass"] = False
            results["warnings"].append(f"{artifact_name} 읽기 실패: {e}")
            results["checks"].append(check_info)

    return results

def check_kpi_delta_reports(
    stage: int,
    run_tag: str,
    baseline_tag: str,
    base_dir: Path
) -> Dict:
    """
    KPI/Delta 리포트 존재 확인
    """
    results = {
        "kpi_csv": False,
        "kpi_md": False,
        "delta_csv": False,
        "delta_md": False,
        "pass": True,
    }

    kpi_dir = base_dir / "reports" / "kpi"
    delta_dir = base_dir / "reports" / "delta"

    kpi_csv = kpi_dir / f"kpi_table__{run_tag}.csv"
    kpi_md = kpi_dir / f"kpi_table__{run_tag}.md"
    delta_csv = delta_dir / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
    delta_md = delta_dir / f"delta_report__{baseline_tag}__vs__{run_tag}.md"

    results["kpi_csv"] = kpi_csv.exists()
    results["kpi_md"] = kpi_md.exists()
    results["delta_csv"] = delta_csv.exists()
    results["delta_md"] = delta_md.exists()

    # Stage7은 delta 리포트가 선택적, Stage8/Stage9는 필수
    if stage == 7:
        # KPI만 필수
        if not all([results["kpi_csv"], results["kpi_md"]]):
            results["pass"] = False
    elif stage == 8 or stage == 9:
        # Stage8/Stage9: KPI + Delta 모두 필수
        if not all([results["kpi_csv"], results["kpi_md"], results["delta_csv"], results["delta_md"]]):
            results["pass"] = False
    else:
        # 다른 Stage는 모두 필수
        if not all([results["kpi_csv"], results["kpi_md"], results["delta_csv"], results["delta_md"]]):
            results["pass"] = False

    return results

def check_baselines_yaml(base_dir: Path, run_tag: str, stage: int) -> Tuple[bool, str]:
    """
    baselines.yaml 확인 (Stage7)

    Returns:
        (is_valid, message)
    """
    if stage != 7:
        return True, "N/A (Stage7 아님)"

    baselines_path = base_dir / "reports" / "history" / "baselines.yaml"
    if not baselines_path.exists():
        return False, "baselines.yaml이 존재하지 않습니다"

    try:
        import yaml
        with open(baselines_path, 'r', encoding='utf-8') as f:
            baselines = yaml.safe_load(f) or {}

        ranking_baseline = baselines.get("ranking_baseline_tag")
        if ranking_baseline != run_tag:
            return False, f"ranking_baseline_tag 불일치: {ranking_baseline} (예상: {run_tag})"

        return True, f"ranking_baseline_tag 확인: {ranking_baseline}"
    except Exception as e:
        return False, f"baselines.yaml 읽기 실패: {e}"

def check_history_manifest(base_dir: Path, run_tag: str, stage: int, track: str) -> Tuple[bool, str]:
    """
    history_manifest 확인 (Stage7)

    Returns:
        (is_valid, message)
    """
    if stage != 7:
        return True, "N/A (Stage7 아님)"

    manifest_path = base_dir / "reports" / "history" / "history_manifest.parquet"
    if not manifest_path.exists():
        manifest_path = base_dir / "reports" / "history" / "history_manifest.csv"

    if not manifest_path.exists():
        return False, "history_manifest 파일이 존재하지 않습니다"

    try:
        if manifest_path.suffix == ".parquet":
            df = pd.read_parquet(manifest_path)
        else:
            df = pd.read_csv(manifest_path)

        # stage_no=7, track=ranking, run_tag row 확인
        mask = (
            (df["stage_no"] == stage) &
            (df["track"] == track) &
            (df["run_tag"] == run_tag)
        )

        if mask.sum() == 0:
            return False, f"history_manifest에 stage_no={stage}, track={track}, run_tag={run_tag} row가 없습니다"

        return True, f"history_manifest 확인: {mask.sum()}개 row 발견"
    except Exception as e:
        return False, f"history_manifest 읽기 실패: {e}"

def generate_check_report(
    stage: int,
    run_tag: str,
    baseline_tag: str,
    artifact_results: Dict,
    report_results: Dict,
    l2_protection: Tuple[bool, str, Optional[str], Optional[str]],
    baselines_check: Tuple[bool, str],
    history_check: Tuple[bool, str],
    base_dir: Path
) -> str:
    """
    체크 리포트 Markdown 생성
    """
    lines = []
    lines.append(f"# Stage {stage} 완료 점검 리포트")
    lines.append("")
    lines.append(f"**Run Tag**: `{run_tag}`")
    lines.append(f"**Baseline Tag**: `{baseline_tag}`")
    lines.append(f"**점검 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. L2 보호 확인
    lines.append("## 1. L2 파일 보호 확인")
    lines.append("")
    l2_pass, l2_msg, l2_hash_before, l2_hash_after = l2_protection
    status = "[PASS]" if l2_pass else "[FAIL]"
    lines.append(f"**결과**: {status}")
    lines.append(f"**상세**: {l2_msg}")
    if l2_hash_before and l2_hash_after:
        lines.append(f"**해시**: 실행 전: {l2_hash_before}, 실행 후: {l2_hash_after}")
    lines.append("")

    # 2. 필수 산출물 체크
    lines.append("## 2. 필수 산출물 체크")
    lines.append("")
    artifact_status = "[PASS]" if artifact_results["pass"] else "[FAIL]"
    lines.append(f"**결과**: {artifact_status}")
    lines.append("")
    # Stage7/Stage8/Stage9는 상세 테이블, 다른 Stage는 간단 테이블
    if stage == 7 or stage == 8 or stage == 9:
        if stage == 9:
            lines.append("| 산출물 | 존재 | 행수 | 컬럼 수 | 날짜 범위 | Score 결측률 | Rank 이슈 | 설명가능성 |")
            lines.append("|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| 산출물 | 존재 | 행수 | 컬럼 수 | 날짜 범위 | Score 결측률 | Rank 이슈 |")
            lines.append("|---|---|---|---|---|---|---|")

        for check in artifact_results["checks"]:
            artifact = check["artifact"]
            exists = "[OK]" if check["exists"] else "[MISSING]"
            shape = check.get("shape", "N/A")
            n_rows = shape[0] if isinstance(shape, tuple) else "N/A"
            n_cols = shape[1] if isinstance(shape, tuple) else "N/A"

            date_range = check.get("date_range", {})
            if date_range:
                date_str = f"{date_range.get('start', 'N/A')[:10]} ~ {date_range.get('end', 'N/A')[:10]}"
            else:
                date_str = "N/A"

            # Stage7/Stage8/Stage9 특화 정보
            score_missing = check.get("score_missing_pct", None)
            score_str = f"{score_missing:.2f}%" if score_missing is not None else "N/A"

            rank_issues = check.get("rank_issues", None)
            rank_str = f"{rank_issues}건" if rank_issues is not None and rank_issues > 0 else "0건" if rank_issues == 0 else "N/A"

            # Stage9: 설명가능성 컬럼 정보 추가
            if stage == 9:
                explainability = check.get("explainability_cols", {})
                contrib_count = explainability.get("contrib_cols", 0)
                has_top_features = explainability.get("has_top_features", False)
                explainability_str = f"contrib:{contrib_count}, top_features:{'Y' if has_top_features else 'N'}"
                lines.append(f"| {artifact} | {exists} | {n_rows} | {n_cols} | {date_str} | {score_str} | {rank_str} | {explainability_str} |")
            else:
                lines.append(f"| {artifact} | {exists} | {n_rows} | {n_cols} | {date_str} | {score_str} | {rank_str} |")
    else:
        lines.append("| 산출물 | 존재 | 행수 | 컬럼 수 | 날짜 범위 |")
        lines.append("|---|---|---|---|---|")

        for check in artifact_results["checks"]:
            artifact = check["artifact"]
            exists = "[OK]" if check["exists"] else "[MISSING]"
            shape = check.get("shape", "N/A")
            n_rows = shape[0] if isinstance(shape, tuple) else "N/A"
            n_cols = shape[1] if isinstance(shape, tuple) else "N/A"

            date_range = check.get("date_range", {})
            if date_range:
                date_str = f"{date_range.get('start', 'N/A')[:10]} ~ {date_range.get('end', 'N/A')[:10]}"
            else:
                date_str = "N/A"

            lines.append(f"| {artifact} | {exists} | {n_rows} | {n_cols} | {date_str} |")

    lines.append("")

    # 경고사항
    if artifact_results["warnings"]:
        lines.append("**경고사항:**")
        for warning in artifact_results["warnings"]:
            lines.append(f"- WARNING: {warning}")
        lines.append("")

    # Gate Notes (Stage7)
    if artifact_results.get("gate_notes"):
        lines.append("**Gate Notes:**")
        for note in artifact_results["gate_notes"]:
            lines.append(f"- NOTE: {note}")
        lines.append("")

    # 3. 리포트 체크
    lines.append("## 3. 리포트 생성 체크")
    lines.append("")
    report_status = "✅ PASS" if report_results["pass"] else "❌ FAIL"
    lines.append(f"**결과**: {report_status}")
    lines.append("")
    lines.append("| 리포트 | 존재 |")
    lines.append("|---|---|")
    lines.append(f"| KPI CSV | {'[OK]' if report_results['kpi_csv'] else '[MISSING]'} |")
    lines.append(f"| KPI MD | {'[OK]' if report_results['kpi_md'] else '[MISSING]'} |")
    if stage == 7:
        # Stage7은 delta 리포트가 선택적
        lines.append(f"| Delta CSV | {'[OK]' if report_results.get('delta_csv') else '[OPTIONAL]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results.get('delta_md') else '[OPTIONAL]'} |")
        lines.append(f"| Ranking Snapshot | {'[OK]' if report_results.get('ranking_snapshot') else '[MISSING]'} |")
    elif stage == 8:
        # Stage8: Delta 리포트 필수 + 섹터 농도 리포트 필수
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
        lines.append(f"| Sector Concentration | {'[OK]' if report_results.get('sector_concentration') else '[MISSING]'} |")
    elif stage == 9:
        # Stage9: Delta 리포트 필수 + Snapshot 필수
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
        lines.append(f"| Ranking Snapshot | {'[OK]' if report_results.get('ranking_snapshot') else '[MISSING]'} |")
    elif stage == 10:
        # Stage10: Delta 리포트 필수 + Regime Summary 필수
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
        lines.append(f"| Regime Summary | {'[OK]' if report_results.get('regime_summary') else '[MISSING]'} |")
    elif stage == 11:
        # Stage11: Delta 리포트 필수 + UI Snapshot/Metrics 필수
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
        lines.append(f"| UI Snapshot | {'[OK]' if report_results.get('ui_snapshot') else '[MISSING]'} |")
        lines.append(f"| UI Metrics | {'[OK]' if report_results.get('ui_metrics') else '[MISSING]'} |")
    elif stage == 12:
        # Stage12: Delta 리포트 필수 + Final Export Pack 파일들 필수
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
        lines.append(f"| Timeline PPT | {'[OK]' if report_results.get('timeline_ppt_csv') else '[MISSING]'} |")
        lines.append(f"| KPI Onepager CSV | {'[OK]' if report_results.get('kpi_onepager_csv') else '[MISSING]'} |")
        lines.append(f"| KPI Onepager MD | {'[OK]' if report_results.get('kpi_onepager_md') else '[MISSING]'} |")
        lines.append(f"| Latest Snapshot | {'[OK]' if report_results.get('latest_snapshot_csv') else '[MISSING]'} |")
        lines.append(f"| Equity Curves | {'[OK]' if report_results.get('equity_curves_csv') else '[MISSING]'} |")
        lines.append(f"| Appendix Sources | {'[OK]' if report_results.get('appendix_sources_md') else '[MISSING]'} |")
    elif stage == 13:
        # [Stage13] Delta 리포트 필수 + Selection Diagnostics 필수
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
        lines.append(f"| Selection Diagnostics | {'[OK]' if report_results.get('selection_diagnostics') else '[MISSING]'} |")

        # [Stage13] K_eff 분석 섹션 추가
        if report_results.get("k_eff_analysis"):
            lines.append("")
            lines.append("### K_eff 분석")
            lines.append("")
            k_eff = report_results["k_eff_analysis"]
            lines.append("| Phase | Top K | K_eff (평균) | Fill Rate (%) |")
            lines.append("|---|---|---|---|")

            if k_eff.get("dev_top_k") and k_eff.get("dev_k_eff_mean") is not None:
                dev_fill = k_eff.get("dev_fill_rate_pct")
                dev_fill_str = f"{dev_fill:.1f}%" if dev_fill is not None else "N/A"
                lines.append(f"| Dev | {k_eff['dev_top_k']} | {k_eff['dev_k_eff_mean']:.1f} | {dev_fill_str} |")

            if k_eff.get("holdout_top_k") and k_eff.get("holdout_k_eff_mean") is not None:
                holdout_fill = k_eff.get("holdout_fill_rate_pct")
                holdout_fill_str = f"{holdout_fill:.1f}%" if holdout_fill is not None else "N/A"
                lines.append(f"| Holdout | {k_eff['holdout_top_k']} | {k_eff['holdout_k_eff_mean']:.1f} | {holdout_fill_str} |")

            lines.append("")

            # [Stage13] Drop reason top3
            drop_reasons_top3 = k_eff.get("drop_reasons_top3", [])
            if drop_reasons_top3:
                lines.append("### Drop Reason Top 3")
                lines.append("")
                lines.append("| Reason | Count |")
                lines.append("|---|---|")
                for reason, count in drop_reasons_top3:
                    lines.append(f"| {reason} | {count} |")
                lines.append("")
                lines.append("|---|---|")
                for reason, count in report_results["drop_reasons_top3"]:
                    lines.append(f"| {reason} | {count} |")
                lines.append("")
    else:
        lines.append(f"| Delta CSV | {'[OK]' if report_results['delta_csv'] else '[MISSING]'} |")
        lines.append(f"| Delta MD | {'[OK]' if report_results['delta_md'] else '[MISSING]'} |")
    lines.append("")

    # Stage7 추가 체크
    if stage == 7:
        lines.append("## 4. baselines.yaml 확인")
        lines.append("")
        baselines_pass, baselines_msg = baselines_check
        baselines_status = "[PASS]" if baselines_pass else "[FAIL]"
        lines.append(f"**결과**: {baselines_status}")
        lines.append(f"**상세**: {baselines_msg}")
        lines.append("")

        lines.append("## 5. history_manifest 확인")
        lines.append("")
        history_pass, history_msg = history_check
        history_status = "[PASS]" if history_pass else "[FAIL]"
        lines.append(f"**결과**: {history_status}")
        lines.append(f"**상세**: {history_msg}")
        lines.append("")

    # 최종 판정
    lines.append("## 최종 판정")
    lines.append("")
    overall_pass = artifact_results["pass"] and report_results["pass"] and l2_pass
    if stage == 7:
        overall_pass = overall_pass and baselines_check[0] and history_check[0]

    final_status = "[PASS]" if overall_pass else "[FAIL]"
    lines.append(f"**결과**: {final_status}")
    lines.append("")

    if not overall_pass:
        lines.append("**실패 사유:**")
        if not artifact_results["pass"]:
            lines.append("- 필수 산출물 누락 또는 오류")
        if not report_results["pass"]:
            lines.append("- 리포트 파일 누락")
        if not l2_pass:
            lines.append("- L2 파일 보호 규칙 위반")
        if stage == 7:
            if not baselines_check[0]:
                lines.append("- baselines.yaml 검증 실패")
            if not history_check[0]:
                lines.append("- history_manifest 검증 실패")
        lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Check Stage Completion"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Config file path")
    parser.add_argument("--run-tag", type=str, required=True,
                       help="Run tag to check")
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                       help="Stage number (0-13)")
    parser.add_argument("--baseline-tag", type=str, default="baseline_prerefresh_20251219_143636",
                       help="Baseline tag")
    parser.add_argument("--out-dir", type=str, default="reports/stages",
                       help="Output directory for check reports")
    args = parser.parse_args()

    # 경로 설정
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    config_path = base_dir / args.config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    base_interim_dir = get_path(cfg, "data_interim")
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage {args.stage} Check] Run Tag: {args.run_tag}")
    print(f"[Stage {args.stage} Check] Baseline Tag: {args.baseline_tag}")

    # 로그 파일 경로 (L2 해시 확인용)
    log_file = base_dir / "reports" / "logs" / f"run__stage{args.stage}__{args.run_tag}.txt"

    # L2 보호 확인
    l2_protection = check_l2_protection(base_interim_dir, args.baseline_tag, log_file)

    # 필수 산출물 체크
    artifact_results = check_stage_artifacts(
        args.stage, args.run_tag, base_interim_dir, cfg
    )

    # 리포트 체크
    report_results = check_kpi_delta_reports(
        args.stage, args.run_tag, args.baseline_tag, base_dir
    )

    # Stage7 추가 체크: ranking_snapshot
    if args.stage == 7:
        ranking_snapshot = base_dir / "reports" / "ranking" / f"ranking_snapshot__{args.run_tag}.csv"
        if ranking_snapshot.exists():
            report_results["ranking_snapshot"] = True
        else:
            report_results["ranking_snapshot"] = False
            report_results["pass"] = False

        # baselines.yaml 확인
        baselines_check = check_baselines_yaml(base_dir, args.run_tag, args.stage)

        # history_manifest 확인
        history_check = check_history_manifest(base_dir, args.run_tag, args.stage, "ranking")
    elif args.stage == 8:
        # Stage8 추가 체크: 섹터 농도 리포트
        sector_concentration = base_dir / "reports" / "ranking" / f"sector_concentration__{args.run_tag}.csv"
        if sector_concentration.exists():
            report_results["sector_concentration"] = True
        else:
            report_results["sector_concentration"] = False
            report_results["pass"] = False

        baselines_check = (True, "N/A")
        history_check = (True, "N/A")
    elif args.stage == 9:
        # Stage9 추가 체크: ranking_snapshot
        ranking_snapshot = base_dir / "reports" / "ranking" / f"ranking_snapshot__{args.run_tag}.csv"
        if ranking_snapshot.exists():
            report_results["ranking_snapshot"] = True
        else:
            report_results["ranking_snapshot"] = False
            report_results["pass"] = False

        baselines_check = (True, "N/A")
        history_check = (True, "N/A")
    elif args.stage == 10:
        # Stage10 추가 체크: regime_summary
        regime_summary = base_dir / "reports" / "ranking" / f"regime_summary__{args.run_tag}.csv"
        if regime_summary.exists():
            report_results["regime_summary"] = True
        else:
            report_results["regime_summary"] = False
            report_results["pass"] = False

        baselines_check = (True, "N/A")
        history_check = (True, "N/A")
    elif args.stage == 11:
        # Stage11 추가 체크: UI Snapshot/Metrics
        ui_snapshot = base_dir / "reports" / "ui" / f"ui_snapshot__{args.run_tag}.csv"
        ui_metrics = base_dir / "reports" / "ui" / f"ui_metrics__{args.run_tag}.csv"

        if ui_snapshot.exists():
            report_results["ui_snapshot"] = True
        else:
            report_results["ui_snapshot"] = False
            report_results["pass"] = False

        if ui_metrics.exists():
            report_results["ui_metrics"] = True
        else:
            report_results["ui_metrics"] = False
            report_results["pass"] = False

        baselines_check = (True, "N/A")
        history_check = (True, "N/A")
    elif args.stage == 12:
        # Stage12 추가 체크: Final Export Pack 파일들
        export_dir = base_dir / "artifacts" / "reports" / "final_export" / args.run_tag

        # 필수 파일들 체크
        required_files = [
            "timeline_ppt.csv",
            "kpi_onepager.csv",
            "kpi_onepager.md",
            "latest_snapshot.csv",
            "equity_curves.csv",
            "appendix_sources.md",
        ]

        for filename in required_files:
            file_path = export_dir / filename
            key = filename.replace(".", "_")
            if file_path.exists():
                report_results[key] = True
            else:
                report_results[key] = False
                report_results["pass"] = False

        baselines_check = (True, "N/A")
        history_check = (True, "N/A")
    elif args.stage == 13:
        # [Stage13] selection_diagnostics 체크
        selection_diagnostics = base_interim_dir / args.run_tag / "selection_diagnostics.parquet"
        if selection_diagnostics.exists():
            report_results["selection_diagnostics"] = True
            # K_eff 분석 추가
            try:
                df_diag = pd.read_parquet(selection_diagnostics)
                if "phase" in df_diag.columns and "selected_count" in df_diag.columns and "top_k" in df_diag.columns:
                    # K_eff 분석
                    df_dev = df_diag[df_diag["phase"] == "dev"]
                    df_holdout = df_diag[df_diag["phase"] == "holdout"]

                    dev_k_eff_mean = df_dev["selected_count"].mean() if not df_dev.empty else None
                    holdout_k_eff_mean = df_holdout["selected_count"].mean() if not df_holdout.empty else None
                    dev_top_k = df_dev["top_k"].iloc[0] if not df_dev.empty else None
                    holdout_top_k = df_holdout["top_k"].iloc[0] if not df_holdout.empty else None

                    dev_fill_rate = (dev_k_eff_mean / dev_top_k * 100) if (dev_k_eff_mean and dev_top_k) else None
                    holdout_fill_rate = (holdout_k_eff_mean / holdout_top_k * 100) if (holdout_k_eff_mean and holdout_top_k) else None

                    # Drop reason top3
                    if "drop_reasons" in df_diag.columns:
                        # drop_reasons는 문자열이므로 파싱 필요
                        all_reasons = {}
                        for reasons_str in df_diag["drop_reasons"].dropna():
                            if isinstance(reasons_str, str):
                                # 간단한 파싱 (예: "{'missing_price': 5, 'sectorcap_IT': 3}")
                                import ast
                                try:
                                    reasons_dict = ast.literal_eval(reasons_str)
                                    for reason, count in reasons_dict.items():
                                        all_reasons[reason] = all_reasons.get(reason, 0) + count
                                except:
                                    pass

                        top3_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                        report_results["drop_reasons_top3"] = top3_reasons
                    else:
                        report_results["drop_reasons_top3"] = []

                    # [Stage13] Drop reasons top3 추출
                    drop_reasons_top3 = []
                    if "drop_reasons" in df_diag.columns:
                        # drop_reasons 컬럼이 문자열로 저장된 경우 파싱
                        all_reasons = {}
                        for _, row in df_diag.iterrows():
                            reasons_str = row.get("drop_reasons")
                            if pd.notna(reasons_str) and isinstance(reasons_str, str):
                                try:
                                    # 문자열을 dict로 파싱 시도 (예: "{'missing_required_cols': 5, 'sectorcap_IT': 2}")
                                    import ast
                                    reasons_dict = ast.literal_eval(reasons_str)
                                    for reason, count in reasons_dict.items():
                                        all_reasons[reason] = all_reasons.get(reason, 0) + count
                                except Exception:
                                    pass

                        # 상위 3개 추출
                        if all_reasons:
                            sorted_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
                            drop_reasons_top3 = sorted_reasons[:3]

                    report_results["k_eff_analysis"] = {
                        "dev_k_eff_mean": dev_k_eff_mean,
                        "holdout_k_eff_mean": holdout_k_eff_mean,
                        "dev_top_k": dev_top_k,
                        "holdout_top_k": holdout_top_k,
                        "dev_fill_rate_pct": dev_fill_rate,
                        "holdout_fill_rate_pct": holdout_fill_rate,
                        "drop_reasons_top3": drop_reasons_top3,
                    }
            except Exception as e:
                report_results["k_eff_analysis"] = None
                report_results["warnings"] = report_results.get("warnings", [])
                report_results["warnings"].append(f"K_eff 분석 실패: {e}")
        else:
            report_results["selection_diagnostics"] = False
            report_results["pass"] = False

        baselines_check = (True, "N/A")
        history_check = (True, "N/A")
    else:
        baselines_check = (True, "N/A")
        history_check = (True, "N/A")

    # 리포트 생성
    report_content = generate_check_report(
        args.stage, args.run_tag, args.baseline_tag,
        artifact_results, report_results, l2_protection,
        baselines_check, history_check, base_dir
    )

    # 저장
    report_path = out_dir / f"check__stage{args.stage}__{args.run_tag}.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"[Stage {args.stage} Check] Report saved: {report_path}")

    # 최종 판정 출력
    overall_pass = artifact_results["pass"] and report_results["pass"] and l2_protection[0]
    if args.stage == 7:
        overall_pass = overall_pass and baselines_check[0] and history_check[0]
    elif args.stage == 8:
        # Stage8: 섹터 농도 리포트 필수
        overall_pass = overall_pass and report_results.get("sector_concentration", False)
    elif args.stage == 9:
        # Stage9: ranking_snapshot 필수
        overall_pass = overall_pass and report_results.get("ranking_snapshot", False)
    elif args.stage == 10:
        # Stage10: regime_summary 필수
        overall_pass = overall_pass and report_results.get("regime_summary", False)
    elif args.stage == 11:
        # Stage11: UI Snapshot/Metrics 필수
        overall_pass = overall_pass and report_results.get("ui_snapshot", False) and report_results.get("ui_metrics", False)
    elif args.stage == 12:
        # Stage12: Final Export Pack 파일들 필수
        required_files = [
            "timeline_ppt_csv",
            "kpi_onepager_csv",
            "kpi_onepager_md",
            "latest_snapshot_csv",
            "equity_curves_csv",
            "appendix_sources_md",
        ]
        for key in required_files:
            overall_pass = overall_pass and report_results.get(key, False)
    elif args.stage == 13:
        # [Stage13] Selection Diagnostics 필수
        overall_pass = overall_pass and report_results.get("selection_diagnostics", False)

    if overall_pass:
        print(f"[Stage {args.stage} Check] [PASS]")
        sys.exit(0)
    else:
        print(f"[Stage {args.stage} Check] [FAIL]")
        if args.stage == 7:
            if not baselines_check[0]:
                print(f"  - baselines.yaml: {baselines_check[1]}")
            if not history_check[0]:
                print(f"  - history_manifest: {history_check[1]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
