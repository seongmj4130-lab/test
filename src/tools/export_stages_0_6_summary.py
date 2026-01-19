# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/export_stages_0_6_summary.py
"""
Stage0~6 종합 요약 리포트 생성 스크립트
- stage0~6 각각의 run_tag 자동 탐지
- KPI/Delta 리포트 생성/갱신
- Stage 완료 체크 실행
- 종합 요약 리포트 생성
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_path(cfg: dict, key: str) -> Path:
    """config에서 경로 추출"""
    paths = cfg.get("paths", {})
    base_dir = Path(paths.get("base_dir", Path.cwd()))
    path_template = paths.get(key, "")
    if path_template:
        return Path(path_template.format(base_dir=base_dir))
    return base_dir / key.replace("_", "/")


def detect_stage_tags(
    stage: int,
    base_dir: Path,
    base_interim_dir: Path,
    reports_kpi_dir: Path,
    reports_delta_dir: Path,
) -> Optional[str]:
    """
    Stage별 run_tag 자동 탐지

    우선순위:
    A) reports/kpi/kpi_table__stage{N}_*.csv 파일명에서 태그 파싱
    B) data/interim/ 하위 폴더에서 stage{N}_* 탐색
    C) reports/delta 파일명에서 stage{N}_* 추출
    """
    stage_tags = []
    pattern = re.compile(rf"^stage{stage}_.*")

    # A) KPI 리포트 파일명에서 탐지
    if reports_kpi_dir.exists():
        for kpi_file in reports_kpi_dir.glob(f"kpi_table__stage{stage}_*.csv"):
            tag_match = re.search(rf"kpi_table__(stage{stage}_[^\.]+)", kpi_file.name)
            if tag_match:
                tag = tag_match.group(1)
                mtime = kpi_file.stat().st_mtime
                stage_tags.append((tag, mtime, "kpi"))

    # B) interim 디렉토리에서 탐지
    if base_interim_dir.exists():
        for item in base_interim_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                mtime = item.stat().st_mtime
                stage_tags.append((item.name, mtime, "interim"))

    # C) Delta 리포트 파일명에서 탐지
    if reports_delta_dir.exists():
        for delta_file in reports_delta_dir.glob(f"delta_*__stage{stage}_*.csv"):
            tag_match = re.search(
                rf"__stage{stage}_([^_]+(?:_[^_]+)*)\.csv", delta_file.name
            )
            if tag_match:
                tag = f"stage{stage}_{tag_match.group(1)}"
                mtime = delta_file.stat().st_mtime
                stage_tags.append((tag, mtime, "delta"))

    if not stage_tags:
        return None

    # 중복 제거 및 최신 순 정렬
    unique_tags = {}
    for tag, mtime, source in stage_tags:
        if tag not in unique_tags or mtime > unique_tags[tag][1]:
            unique_tags[tag] = (tag, mtime, source)

    # 최신 순으로 정렬
    sorted_tags = sorted(unique_tags.values(), key=lambda x: -x[1])
    return sorted_tags[0][0] if sorted_tags else None


def ensure_kpi_report(base_dir: Path, run_tag: str, config_path: Path) -> bool:
    """KPI 리포트 생성/확인"""
    kpi_dir = base_dir / "reports" / "kpi"
    kpi_csv = kpi_dir / f"kpi_table__{run_tag}.csv"

    if kpi_csv.exists():
        print(f"  [KPI] 이미 존재: {kpi_csv}")
        return True

    print(f"  [KPI] 생성 중: {run_tag}")
    script = base_dir / "src" / "tools" / "export_kpi_table.py"
    if not script.exists():
        print(f"  [KPI] ERROR: 스크립트 없음: {script}", file=sys.stderr)
        return False

    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--tag", run_tag],
        cwd=str(base_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("  [KPI] 생성 완료")
        return True
    else:
        print(f"  [KPI] ERROR: {result.stderr}", file=sys.stderr)
        return False


def ensure_delta_report(base_dir: Path, baseline_tag: str, run_tag: str) -> bool:
    """Delta 리포트 생성/확인"""
    delta_dir = base_dir / "reports" / "delta"
    delta_csv = delta_dir / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"

    if delta_csv.exists():
        print(f"  [Delta] 이미 존재: {delta_csv}")
        return True

    print(f"  [Delta] 생성 중: {baseline_tag} vs {run_tag}")
    script = base_dir / "src" / "tools" / "export_delta_report.py"
    if not script.exists():
        print(f"  [Delta] ERROR: 스크립트 없음: {script}", file=sys.stderr)
        return False

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-tag",
            baseline_tag,
            "--run-tag",
            run_tag,
        ],
        cwd=str(base_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("  [Delta] 생성 완료")
        return True
    else:
        print(f"  [Delta] ERROR: {result.stderr}", file=sys.stderr)
        return False


def run_stage_check(
    base_dir: Path, stage: int, run_tag: str, baseline_tag: str, config_path: Path
) -> bool:
    """Stage 완료 체크 실행"""
    print(f"  [Check] 실행 중: Stage {stage}")
    script = base_dir / "src" / "tools" / "check_stage_completion.py"
    if not script.exists():
        print(f"  [Check] ERROR: 스크립트 없음: {script}", file=sys.stderr)
        return False

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--run-tag",
            run_tag,
            "--stage",
            str(stage),
            "--baseline-tag",
            baseline_tag,
        ],
        cwd=str(base_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("  [Check] [PASS]")
        return True
    else:
        print(f"  [Check] [FAIL]: {result.stderr}", file=sys.stderr)
        return False


def load_delta_kpi(delta_csv_path: Path) -> Optional[pd.DataFrame]:
    """Delta KPI CSV 로드"""
    if not delta_csv_path.exists():
        return None
    try:
        return pd.read_csv(delta_csv_path)
    except Exception as e:
        print(f"  [WARN] Delta KPI 로드 실패: {e}", file=sys.stderr)
        return None


def extract_top_changes(
    delta_df: pd.DataFrame, stage: int, top_n: int = 5
) -> list[dict]:
    """Delta에서 가장 크게 바뀐 KPI Top N 추출"""
    if delta_df is None or delta_df.empty:
        return []

    top_changes = []

    # Dev와 Holdout 각각에서 Top N 추출
    for phase in ["dev", "holdout"]:
        abs_diff_col = f"{phase}_abs_diff"
        pct_diff_col = f"{phase}_pct_diff"
        baseline_col = f"baseline_{phase}_value"
        current_col = f"current_{phase}_value"

        # 필요한 컬럼이 있는지 확인
        if abs_diff_col not in delta_df.columns:
            continue

        # 해당 phase에 값이 있는 행만 필터링
        phase_df = delta_df[
            delta_df[baseline_col].notna() | delta_df[current_col].notna()
        ].copy()
        if phase_df.empty:
            continue

        # abs_diff의 절대값 기준으로 정렬 (NaN 제외)
        phase_df = phase_df[phase_df[abs_diff_col].notna()].copy()
        if phase_df.empty:
            continue

        phase_df["abs_diff_abs"] = phase_df[abs_diff_col].abs()
        phase_df = phase_df.sort_values("abs_diff_abs", ascending=False).head(top_n)

        for _, row in phase_df.iterrows():
            metric = row.get("metric", "N/A")
            baseline_val = row.get(baseline_col, "N/A")
            current_val = row.get(current_col, "N/A")
            delta_abs = row.get(abs_diff_col, 0)
            delta_pct = row.get(pct_diff_col, 0)

            # 값 포맷팅
            if isinstance(baseline_val, (int, float)) and not pd.isna(baseline_val):
                baseline_val = f"{baseline_val:.4f}"
            if isinstance(current_val, (int, float)) and not pd.isna(current_val):
                current_val = f"{current_val:.4f}"
            if isinstance(delta_abs, (int, float)) and not pd.isna(delta_abs):
                delta_abs = abs(delta_abs)  # 절대값 사용
            if isinstance(delta_pct, (int, float)) and not pd.isna(delta_pct):
                delta_pct = abs(delta_pct)  # 절대값 사용

            top_changes.append(
                {
                    "stage": stage,
                    "phase": phase.capitalize(),  # Dev, Holdout
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "delta_abs": delta_abs,
                    "delta_pct": delta_pct,
                }
            )

    return top_changes


def generate_summary_report(
    stage_tags: dict[int, str],
    baseline_tag: str,
    base_dir: Path,
    top_changes_by_stage: dict[int, list[dict]],
    check_results: dict[int, bool],
) -> str:
    """종합 요약 리포트 생성"""
    lines = []
    lines.append("# Stage0~6 종합 요약 리포트")
    lines.append("")
    lines.append(f"**Baseline Tag**: `{baseline_tag}`")
    lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Stage별 Run Tag 요약
    lines.append("## 1. Stage별 Run Tag")
    lines.append("")
    lines.append("| Stage | Run Tag | 상태 |")
    lines.append("|---|---|---|")
    for stage in range(7):
        tag = stage_tags.get(stage, "N/A")
        check_status = (
            "[PASS]"
            if check_results.get(stage, False)
            else "[FAIL]"
            if tag != "N/A"
            else "N/A"
        )
        lines.append(f"| Stage {stage} | `{tag}` | {check_status} |")
    lines.append("")

    # 2. 각 Stage별 링크
    lines.append("## 2. Stage별 리포트 링크")
    lines.append("")
    for stage in range(7):
        tag = stage_tags.get(stage)
        if not tag:
            continue

        lines.append(f"### Stage {stage}")
        lines.append("")
        lines.append(f"- **Run Tag**: `{tag}`")
        lines.append(f"- [KPI 리포트](reports/kpi/kpi_table__{tag}.md)")
        lines.append(
            f"- [Delta 리포트](reports/delta/delta_report__{baseline_tag}__vs__{tag}.md)"
        )
        lines.append(f"- [완료 체크](reports/stages/check__stage{stage}__{tag}.md)")
        lines.append("")

    # 3. Stage별 주요 변화 Top 5
    lines.append("## 3. Stage별 주요 변화 Top 5")
    lines.append("")
    for stage in range(7):
        tag = stage_tags.get(stage)
        if not tag:
            continue

        top_changes = top_changes_by_stage.get(stage, [])
        if not top_changes:
            continue

        lines.append(f"### Stage {stage} ({tag})")
        lines.append("")
        lines.append("| Phase | Metric | Baseline | Current | Δ (절대값) | Δ (%) |")
        lines.append("|---|---|---|---|---|---|")

        for change in top_changes[:5]:
            phase = change.get("phase", "N/A")
            metric = change.get("metric", "N/A")
            baseline = change.get("baseline", "N/A")
            current = change.get("current", "N/A")
            delta_abs = change.get("delta_abs", 0)
            delta_pct = change.get("delta_pct", 0)

            # 숫자 포맷팅
            try:
                delta_abs_str = (
                    f"{float(delta_abs):.4f}" if delta_abs != "N/A" else "N/A"
                )
            except (ValueError, TypeError):
                delta_abs_str = str(delta_abs)

            try:
                delta_pct_str = (
                    f"{float(delta_pct):.2f}%" if delta_pct != "N/A" else "N/A"
                )
            except (ValueError, TypeError):
                delta_pct_str = str(delta_pct) if delta_pct != "N/A" else "N/A"

            lines.append(
                f"| {phase} | {metric} | {baseline} | {current} | {delta_abs_str} | {delta_pct_str} |"
            )

        lines.append("")

    # 4. 전체 주요 변화 Top 10
    lines.append("## 4. 전체 주요 변화 Top 10")
    lines.append("")
    all_changes = []
    for stage, changes in top_changes_by_stage.items():
        for change in changes:
            change["stage"] = stage
            all_changes.append(change)

    if all_changes:
        all_changes.sort(key=lambda x: x.get("delta_abs", 0), reverse=True)
        lines.append(
            "| Stage | Phase | Metric | Baseline | Current | Δ (절대값) | Δ (%) |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        for change in all_changes[:10]:
            stage = change.get("stage", "N/A")
            phase = change.get("phase", "N/A")
            metric = change.get("metric", "N/A")
            baseline = change.get("baseline", "N/A")
            current = change.get("current", "N/A")
            delta_abs = change.get("delta_abs", 0)
            delta_pct = change.get("delta_pct", 0)

            # 숫자 포맷팅
            try:
                delta_abs_str = (
                    f"{float(delta_abs):.4f}" if delta_abs != "N/A" else "N/A"
                )
            except (ValueError, TypeError):
                delta_abs_str = str(delta_abs)

            try:
                delta_pct_str = (
                    f"{float(delta_pct):.2f}%" if delta_pct != "N/A" else "N/A"
                )
            except (ValueError, TypeError):
                delta_pct_str = str(delta_pct) if delta_pct != "N/A" else "N/A"

            lines.append(
                f"| Stage {stage} | {phase} | {metric} | {baseline} | {current} | {delta_abs_str} | {delta_pct_str} |"
            )
        lines.append("")

    # 5. 경고사항 모음
    lines.append("## 5. 경고사항 모음")
    lines.append("")
    warnings_found = False

    for stage in range(7):
        tag = stage_tags.get(stage)
        if not tag:
            continue

        delta_csv = (
            base_dir / "reports" / "delta" / f"delta_kpi__{baseline_tag}__vs__{tag}.csv"
        )
        if delta_csv.exists():
            try:
                delta_df = pd.read_csv(delta_csv)
                # cost_bps mismatch 같은 경고 찾기
                if "metric" in delta_df.columns:
                    # cost_bps 관련 메트릭 찾기
                    cost_bps_rows = delta_df[
                        delta_df["metric"].str.contains(
                            "cost_bps", case=False, na=False
                        )
                    ]
                    if not cost_bps_rows.empty:
                        warnings_found = True
                        lines.append(f"### Stage {stage}")
                        lines.append("")
                        for _, row in cost_bps_rows.iterrows():
                            metric = row.get("metric", "N/A")
                            # 값이 변경되었는지 확인
                            dev_abs = row.get("dev_abs_diff", 0)
                            holdout_abs = row.get("holdout_abs_diff", 0)

                            if pd.notna(dev_abs) and abs(float(dev_abs)) > 0.01:
                                baseline_val = row.get("baseline_dev_value", "N/A")
                                current_val = row.get("current_dev_value", "N/A")
                                lines.append(
                                    f"- ⚠️ {metric} (Dev): {baseline_val} → {current_val}"
                                )

                            if pd.notna(holdout_abs) and abs(float(holdout_abs)) > 0.01:
                                baseline_val = row.get("baseline_holdout_value", "N/A")
                                current_val = row.get("current_holdout_value", "N/A")
                                lines.append(
                                    f"- ⚠️ {metric} (Holdout): {baseline_val} → {current_val}"
                                )

                        if not any(
                            [
                                pd.notna(row.get("dev_abs_diff", 0))
                                and abs(float(row.get("dev_abs_diff", 0))) > 0.01
                                for _, row in cost_bps_rows.iterrows()
                            ]
                        ) and not any(
                            [
                                pd.notna(row.get("holdout_abs_diff", 0))
                                and abs(float(row.get("holdout_abs_diff", 0))) > 0.01
                                for _, row in cost_bps_rows.iterrows()
                            ]
                        ):
                            lines.append("- ℹ️ cost_bps 관련 메트릭: 변경 없음")

                        lines.append("")
            except Exception as e:
                lines.append(f"### Stage {stage}")
                lines.append(f"- ⚠️ Delta CSV 읽기 실패: {e}")
                lines.append("")

    if not warnings_found:
        lines.append("경고사항 없음")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export Stages 0-6 Summary Report")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default="baseline_prerefresh_20251219_143636",
        help="Baseline tag",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run: 탐지만 하고 실행하지 않음"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply: KPI/Delta/Check 실행 및 리포트 생성",
    )
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
    reports_kpi_dir = base_dir / "reports" / "kpi"
    reports_delta_dir = base_dir / "reports" / "delta"

    print("=" * 80)
    print("Stage0~6 종합 요약 리포트 생성")
    print("=" * 80)
    print(f"Base Dir: {base_dir}")
    print(f"Baseline Tag: {args.baseline_tag}")
    print(
        f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY' if args.apply else 'DETECT ONLY'}"
    )
    print()

    # Stage별 run_tag 탐지
    print("[1] Stage별 Run Tag 탐지")
    print("-" * 80)
    stage_tags = {}
    for stage in range(7):
        tag = detect_stage_tags(
            stage, base_dir, base_interim_dir, reports_kpi_dir, reports_delta_dir
        )
        stage_tags[stage] = tag
        status = f"[OK] {tag}" if tag else "[MISSING] 없음"
        print(f"  Stage {stage}: {status}")
    print()

    if args.dry_run:
        print("[DRY-RUN] 탐지 완료. 실행하지 않습니다.")
        return

    if not args.apply:
        print(
            "[INFO] --apply 플래그가 없어 실행하지 않습니다. --apply를 추가하면 실행됩니다."
        )
        return

    # Stage별 처리
    print("[2] Stage별 처리 (KPI/Delta/Check)")
    print("-" * 80)
    check_results = {}
    top_changes_by_stage = {}

    for stage in range(7):
        tag = stage_tags.get(stage)
        if not tag:
            print(f"\n[Stage {stage}] Run Tag 없음, 스킵")
            continue

        print(f"\n[Stage {stage}] Run Tag: {tag}")

        # KPI 리포트 생성/확인
        kpi_ok = ensure_kpi_report(base_dir, tag, config_path)

        # Delta 리포트 생성/확인
        delta_ok = ensure_delta_report(base_dir, args.baseline_tag, tag)

        # Stage 체크 실행
        check_ok = run_stage_check(base_dir, stage, tag, args.baseline_tag, config_path)
        check_results[stage] = check_ok

        # Delta에서 Top Changes 추출
        delta_csv = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_kpi__{args.baseline_tag}__vs__{tag}.csv"
        )
        delta_df = load_delta_kpi(delta_csv)
        top_changes = extract_top_changes(delta_df, stage, top_n=5)
        top_changes_by_stage[stage] = top_changes

    print()

    # 종합 요약 리포트 생성
    print("[3] 종합 요약 리포트 생성")
    print("-" * 80)
    summary_content = generate_summary_report(
        stage_tags, args.baseline_tag, base_dir, top_changes_by_stage, check_results
    )

    reports_stages_dir = base_dir / "reports" / "stages"
    reports_stages_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_stages_dir / f"stages_0_6_summary__{args.baseline_tag}.md"
    summary_path.write_text(summary_content, encoding="utf-8")
    print(f"[Summary] 리포트 저장: {summary_path}")
    print()

    # 최종 검증
    print("[4] 최종 검증")
    print("-" * 80)
    all_pass = True

    for stage in range(7):
        tag = stage_tags.get(stage)
        if not tag:
            continue

        kpi_csv = reports_kpi_dir / f"kpi_table__{tag}.csv"
        delta_csv = reports_delta_dir / f"delta_kpi__{args.baseline_tag}__vs__{tag}.csv"
        check_md = reports_stages_dir / f"check__stage{stage}__{tag}.md"

        if not all([kpi_csv.exists(), delta_csv.exists(), check_md.exists()]):
            print(f"  Stage {stage}: [FAIL] (일부 파일 누락)")
            all_pass = False
        else:
            print(f"  Stage {stage}: [PASS]")

    if all_pass:
        print("\n[PASS] 모든 Stage 검증 통과")
    else:
        print("\n[FAIL] 일부 Stage 검증 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
