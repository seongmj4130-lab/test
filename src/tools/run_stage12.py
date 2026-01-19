# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage12.py
"""
[코드 매니저] Stage 12 실행 엔트리포인트
최종 발표용 Export 패키지 생성

Stage 실행 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import yaml
from stage_runner_common import (
    generate_run_tag,
    get_file_hash,
    get_stage_track,
    print_success_summary,
    run_command,
    verify_base_dir,
    verify_l2_reuse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_NUM = 12
STAGE_NAME = "L12"  # L12 = Final Export Pack
STAGE_TRACK = "ranking"  # Stage12는 ranking 트랙

def main():
    parser = argparse.ArgumentParser(description=f"[코드 매니저] Stage {STAGE_NUM} 실행 (Final Export Pack)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config 파일 경로")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)")
    parser.add_argument("--baseline-tag", type=str, default=None, help="Baseline tag (Stage11 run_tag, 없으면 자동 감지)")
    parser.add_argument("--change-title", type=str, default="최종 발표용 export 패키지", help="Change title (History Manifest용)")
    parser.add_argument("--change-summary", type=str, nargs="*", default=[
        "변천사 타임라인 export",
        "KPI onepager 생성",
        "UI 최신 스냅샷/곡선 csv 패키징"
    ], help="Change summary (최대 3개)")
    parser.add_argument("--modified-files", type=str, default="src/stages/final_export_pack.py;src/tools/run_stage12.py", help="Modified files")
    parser.add_argument("--modified-functions", type=str, default="build_final_export_pack", help="Modified functions")
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

    # Run tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = generate_run_tag("stage12_final_export")

    # Baseline 태그 결정 (Stage11 run_tag)
    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        # 최신 Stage11 run_tag 자동 감지 시도
        stage11_dirs = glob.glob(str(PROJECT_ROOT / "data" / "interim" / "stage11_*"))
        if stage11_dirs:
            stage11_tags = [Path(d).name for d in stage11_dirs]
            stage11_tags.sort(reverse=True)
            baseline_tag_used = stage11_tags[0]
            print(f"[INFO] 최신 Stage11 run_tag 자동 감지: {baseline_tag_used}")
        else:
            print("ERROR: Stage11 run_tag를 찾을 수 없습니다. --baseline-tag로 지정하세요.", file=sys.stderr)
            sys.exit(1)

    print("\n" + "="*60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행 (Final Export Pack)")
    print("="*60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used (Stage11): {baseline_tag_used}")
    print("="*60 + "\n")

    # 로그 파일 경로
    log_file = PROJECT_ROOT / "reports" / "logs" / f"run__stage{STAGE_NUM}__{run_tag}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1) Stage 실행 (L12 Final Export Pack)
    # ============================================================
    print("\n[1/6] Stage 실행 (L12 Final Export Pack)...")

    # Stage12는 run_all.py를 거치지 않고 직접 실행
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from src.stages.export.final_export_pack import run_L12_final_export

    # Config 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    # Artifacts 딕셔너리 (Stage12는 직접 파일 로드)
    artifacts = {}

    # Stage12 실행
    try:
        outputs, warns = run_L12_final_export(
            cfg=cfg,
            artifacts=artifacts,
            project_root=PROJECT_ROOT,
            baseline_tag=baseline_tag_used,
            force=True,
        )

        if warns:
            for w in warns:
                print(f"WARNING: {w}", file=sys.stderr)

        # 출력 디렉토리 생성
        export_dir = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag
        export_dir.mkdir(parents=True, exist_ok=True)

        # 산출물 저장
        # 1. timeline_ppt.csv
        timeline_path = export_dir / "timeline_ppt.csv"
        outputs["timeline_ppt"].to_csv(timeline_path, index=False, encoding='utf-8-sig')
        print(f"[OK] timeline_ppt.csv 생성 완료: {timeline_path} ({len(outputs['timeline_ppt'])} rows)")

        # 2. kpi_onepager.csv / .md
        kpi_csv_path = export_dir / "kpi_onepager.csv"
        outputs["kpi_onepager"].to_csv(kpi_csv_path, index=False, encoding='utf-8-sig')
        print(f"[OK] kpi_onepager.csv 생성 완료: {kpi_csv_path} ({len(outputs['kpi_onepager'])} rows)")

        kpi_md_path = export_dir / "kpi_onepager.md"
        kpi_md_path.write_text(outputs["kpi_onepager_md"], encoding='utf-8')
        print(f"[OK] kpi_onepager.md 생성 완료: {kpi_md_path}")

        # 3. latest_snapshot.csv
        snapshot_path = export_dir / "latest_snapshot.csv"
        outputs["latest_snapshot"].to_csv(snapshot_path, index=False, encoding='utf-8-sig')
        print(f"[OK] latest_snapshot.csv 생성 완료: {snapshot_path} ({len(outputs['latest_snapshot'])} rows)")

        # 4. equity_curves.csv
        equity_path = export_dir / "equity_curves.csv"
        outputs["equity_curves"].to_csv(equity_path, index=False, encoding='utf-8-sig')
        print(f"[OK] equity_curves.csv 생성 완료: {equity_path} ({len(outputs['equity_curves'])} rows)")

        # 5. appendix_sources.md
        appendix_path = export_dir / "appendix_sources.md"
        appendix_path.write_text(outputs["appendix_sources_md"], encoding='utf-8')
        print(f"[OK] appendix_sources.md 생성 완료: {appendix_path}")

    except Exception as e:
        print(f"ERROR: Stage12 실행 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # 2) KPI 생성
    # ============================================================
    print("\n[2/6] KPI 테이블 생성 중...")

    kpi_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
        "--config", args.config,
        "--tag", run_tag,
    ]

    if run_command(kpi_cmd, PROJECT_ROOT, "KPI 테이블 생성", log_file) != 0:
        sys.exit(1)

    kpi_csv = PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    if not kpi_csv.exists():
        print(f"ERROR: KPI CSV가 생성되지 않았습니다: {kpi_csv}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] KPI CSV 생성 완료: {kpi_csv}")

    # ============================================================
    # 3) Δ 생성
    # ============================================================
    print("\n[3/6] Δ 리포트 생성 중...")

    delta_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
        "--config", args.config,
        "--baseline-tag", baseline_tag_used,
        "--run-tag", run_tag,
    ]

    if run_command(delta_cmd, PROJECT_ROOT, "Δ 리포트 생성", log_file) != 0:
        print("WARNING: Delta 리포트 생성 실패 (계속 진행)", file=sys.stderr)
    else:
        delta_csv = PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
        if delta_csv.exists():
            print(f"[OK] Delta CSV 생성 완료: {delta_csv}")

    # ============================================================
    # 4) 체크리포트 생성
    # ============================================================
    print("\n[4/6] Stage 체크리포트 생성 중...")

    check_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
    ]

    if run_command(check_cmd, PROJECT_ROOT, "체크리포트 생성", log_file) != 0:
        print("WARNING: 체크리포트 생성 실패 (계속 진행)", file=sys.stderr)
    else:
        check_report = PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md"
        if check_report.exists():
            print(f"[OK] 체크리포트 생성 완료: {check_report}")

    # ============================================================
    # 5) History Manifest 업데이트
    # ============================================================
    print("\n[5/6] History Manifest 업데이트 중...")

    history_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "update_history_manifest.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--track", STAGE_TRACK,
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
        "--change-title", args.change_title,
    ]

    if args.change_summary:
        history_cmd.extend(["--change-summary"] + args.change_summary)
    if args.modified_files:
        history_cmd.extend(["--modified-files", args.modified_files])
    if args.modified_functions:
        history_cmd.extend(["--modified-functions", args.modified_functions])

    if run_command(history_cmd, PROJECT_ROOT, "History Manifest 업데이트", log_file) != 0:
        print("WARNING: History Manifest 업데이트 실패 (계속 진행)", file=sys.stderr)
    else:
        history_manifest = PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
        if history_manifest.exists():
            print(f"[OK] History Manifest 업데이트 완료: {history_manifest}")

    # ============================================================
    # 최종 요약
    # ============================================================
    output_files = [
        ("타임라인", export_dir / "timeline_ppt.csv"),
        ("KPI CSV", export_dir / "kpi_onepager.csv"),
        ("KPI MD", export_dir / "kpi_onepager.md"),
        ("최신 스냅샷", export_dir / "latest_snapshot.csv"),
        ("성과 곡선", export_dir / "equity_curves.csv"),
        ("부록", export_dir / "appendix_sources.md"),
        ("KPI CSV", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"),
        ("Delta CSV", PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"),
        ("체크리포트", PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md"),
        ("History Manifest", PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"),
        ("로그", log_file),
    ]

    print_success_summary(run_tag, baseline_tag_used, None, output_files)

if __name__ == "__main__":
    main()
