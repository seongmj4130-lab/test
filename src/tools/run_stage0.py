# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage0.py
"""
[코드 매니저] Stage 0 실행 엔트리포인트
Stage 실행 → 산출물 생성 확인 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import sys
from pathlib import Path

from stage_runner_common import (
    verify_l2_reuse,
    get_baseline_tag,
    verify_base_dir,
    run_command,
    generate_run_tag,
    get_stage_track,
    print_success_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description="[코드 매니저] Stage 0 실행")
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
    
    # base_dir 검증
    base_dir_valid, base_dir_msg = verify_base_dir(PROJECT_ROOT)
    if not base_dir_valid:
        print(f"ERROR: {base_dir_msg}", file=sys.stderr)
        sys.exit(1)
    
    # run_tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = generate_run_tag("stage0")
    
    # baseline_tag 결정
    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        baseline_tag_used = get_baseline_tag(config_path, 0)
    
    print("\n" + "="*60)
    print("[코드 매니저] Stage 0 실행")
    print("="*60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used: {baseline_tag_used}")
    print("="*60 + "\n")
    
    # 로그 파일 경로
    log_file = PROJECT_ROOT / "reports" / "logs" / f"run__stage0__{run_tag}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 0) L2 재사용 검증 (사전)
    l2_file = PROJECT_ROOT / "data" / "interim" / "fundamentals_annual.parquet"
    if not l2_file.exists():
        print("ERROR: L2 파일이 없습니다. fundamentals_annual.parquet를 먼저 준비하세요.", file=sys.stderr)
        sys.exit(1)
    
    l2_hash_before = None
    try:
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(l2_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        l2_hash_before = sha256_hash.hexdigest()[:16] + "..."
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[L2 검증] 실행 전 해시: {l2_hash_before}\n")
    except Exception as e:
        print(f"WARNING: L2 해시 계산 실패: {e}", file=sys.stderr)
    
    # 1) Stage 실행
    stage_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "run_all.py"),
        "--config", args.config,
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
        "--stage", "L0",
        "--force-rebuild",
        "--skip-l2",  # L2 재사용 강제
    ]
    
    if run_command(stage_cmd, PROJECT_ROOT, "Stage 0 실행", log_file) != 0:
        sys.exit(1)
    
    # L2 재사용 검증 (사후)
    l2_valid, l2_msg, _, l2_hash_after = verify_l2_reuse(PROJECT_ROOT, log_file)
    if not l2_valid:
        print(f"ERROR: {l2_msg}", file=sys.stderr)
        sys.exit(1)
    
    # 2) KPI 생성
    kpi_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
        "--config", args.config,
        "--tag", run_tag,
    ]
    
    if run_command(kpi_cmd, PROJECT_ROOT, "KPI 생성", log_file) != 0:
        sys.exit(1)
    
    # 3) Δ 생성
    delta_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
        "--baseline-tag", baseline_tag_used,
        "--run-tag", run_tag,
    ]
    
    if run_command(delta_cmd, PROJECT_ROOT, "Δ 리포트 생성", log_file) != 0:
        print("WARNING: Delta 리포트 생성 실패 (계속 진행)", file=sys.stderr)
    
    # 4) 체크리포트
    check_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
        "--config", args.config,
        "--stage", "0",
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
    ]
    
    if run_command(check_cmd, PROJECT_ROOT, "체크리포트 생성", log_file) != 0:
        print("WARNING: 체크리포트 생성 실패 (계속 진행)", file=sys.stderr)
    
    # 5) History Manifest 업데이트
    track = get_stage_track(0)
    history_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "update_history_manifest.py"),
        "--config", args.config,
        "--stage", "0",
        "--track", track,
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
    
    if run_command(history_cmd, PROJECT_ROOT, "History Manifest 업데이트", log_file) != 0:
        print("WARNING: History Manifest 업데이트 실패 (계속 진행)", file=sys.stderr)
    
    # 6) 최종 출력
    output_files = [
        ("산출물", PROJECT_ROOT / "data" / "interim" / run_tag / "universe_k200_membership_monthly.parquet"),
        ("KPI CSV", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"),
        ("KPI MD", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"),
        ("Delta CSV", PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"),
        ("Delta MD", PROJECT_ROOT / "reports" / "delta" / f"delta_report__{baseline_tag_used}__vs__{run_tag}.md"),
        ("체크리포트", PROJECT_ROOT / "reports" / "stages" / f"check__stage0__{run_tag}.md"),
        ("History Manifest", PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"),
        ("로그", log_file),
    ]
    
    print_success_summary(run_tag, baseline_tag_used, None, output_files)

if __name__ == "__main__":
    main()
