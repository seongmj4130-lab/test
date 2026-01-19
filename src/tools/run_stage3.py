# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage3.py
"""
[코드 매니저] Stage 3 실행 엔트리포인트
Stage 실행 → 산출물 생성 확인 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_NUM = 3
STAGE_NAME = "L3"
STAGE_TRACK = "pipeline"

def generate_run_tag(stage_name: str) -> str:
    """run_tag 생성: stage명 + 타임스탬프"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage_name.lower()}_{timestamp}"

def run_command(cmd: list, cwd: Path, description: str) -> int:
    """명령어 실행"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(cwd), encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"\n[FAIL] [{description}] Failed with exit code {result.returncode}")
        return result.returncode
    print(f"\n[OK] [{description}] Completed")
    return 0

def get_baseline_tag(config_path: Path, stage: int) -> str:
    """Baseline 태그 결정 (Stage0~6: pipeline_baseline_tag, Stage7+: ranking_baseline_tag)"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    if stage <= 6:
        return cfg.get("baseline", {}).get("pipeline_baseline_tag", "baseline_prerefresh_20251219_143636")
    else:
        ranking_baseline = cfg.get("baseline", {}).get("ranking_baseline_tag")
        if not ranking_baseline:
            print("ERROR: Stage7이 완료되지 않았습니다. ranking_baseline_tag가 설정되지 않았습니다.", file=sys.stderr)
            sys.exit(1)
        return ranking_baseline

def main():
    parser = argparse.ArgumentParser(description=f"[코드 매니저] Stage {STAGE_NUM} 실행")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config 파일 경로")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)")
    parser.add_argument("--baseline-tag", type=str, default=None, help="Baseline tag (없으면 config에서 읽음)")
    parser.add_argument("--change-title", type=str, default=None, help="Change title (History Manifest용)")
    parser.add_argument("--change-summary", type=str, nargs="*", default=[], help="Change summary (최대 3개)")
    parser.add_argument("--modified-files", type=str, default=None, help="Modified files")
    parser.add_argument("--modified-functions", type=str, default=None, help="Modified functions")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = generate_run_tag(f"stage{STAGE_NUM}")

    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        baseline_tag_used = get_baseline_tag(config_path, STAGE_NUM)

    print("\n" + "="*60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행")
    print("="*60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used: {baseline_tag_used}")
    print("="*60 + "\n")

    l2_file = PROJECT_ROOT / "data" / "interim" / "fundamentals_annual.parquet"
    if not l2_file.exists():
        print("ERROR: L2 파일이 없습니다. fundamentals_annual.parquet를 먼저 준비하세요.", file=sys.stderr)
        sys.exit(1)

    stage_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "run_all.py"),
        "--config", args.config,
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
        "--stage", STAGE_NAME,
        "--force-rebuild",
        "--skip-l2",
    ]

    if run_command(stage_cmd, PROJECT_ROOT, f"Stage {STAGE_NUM} 실행") != 0:
        sys.exit(1)

    kpi_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
        "--config", args.config,
        "--tag", run_tag,
    ]

    if run_command(kpi_cmd, PROJECT_ROOT, "KPI 생성") != 0:
        sys.exit(1)

    if STAGE_NUM != 7 or baseline_tag_used:
        delta_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
            "--baseline-tag", baseline_tag_used,
            "--run-tag", run_tag,
        ]

        if run_command(delta_cmd, PROJECT_ROOT, "Δ 리포트 생성") != 0:
            print("WARNING: Delta 리포트 생성 실패 (계속 진행)", file=sys.stderr)

    check_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
    ]

    if run_command(check_cmd, PROJECT_ROOT, "체크리포트 생성") != 0:
        print("WARNING: 체크리포트 생성 실패 (계속 진행)", file=sys.stderr)

    history_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "update_history_manifest.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--track", STAGE_TRACK,
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
    ]

    if args.change_title:
        history_cmd.extend(["--change-title", args.change_title])
    if args.change_summary:
        history_cmd.extend(["--change-summary"] + args.change_summary)
    if args.modified_files:
        history_cmd.extend(["--modified-files", args.modified_files])
    if args.modified_functions:
        history_cmd.extend(["--modified-functions", args.modified_functions])

    if run_command(history_cmd, PROJECT_ROOT, "History Manifest 업데이트") != 0:
        print("WARNING: History Manifest 업데이트 실패 (계속 진행)", file=sys.stderr)

    print("\n" + "="*60)
    print(f"[PASS] Stage {STAGE_NUM} 완료")
    print("="*60)
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used: {baseline_tag_used}")
    if STAGE_NUM == 7:
        print(f"Ranking Baseline Tag: {run_tag} (Stage7 완료 시 ranking_baseline_tag로 설정됨)")
    print(f"\n생성된 주요 파일:")
    print(f"  - data/interim/{run_tag}/...")
    print(f"  - reports/kpi/kpi_table__{run_tag}.csv")
    if baseline_tag_used:
        print(f"  - reports/delta/delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv")
    print(f"  - reports/stages/check__stage{STAGE_NUM}__{run_tag}.md")
    print(f"  - reports/history/history_manifest.parquet")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
