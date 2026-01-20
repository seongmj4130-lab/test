# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage8.py
"""
[코드 매니저] Stage 8 실행 엔트리포인트
Stage7(ranking_baseline_tag) 대비로 "sector-relative 정규화 + 섹터 농도 모니터링" 적용

Stage 실행 → 섹터 농도 리포트 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import sys
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

STAGE_NUM = 8
STAGE_NAME = "L8"  # L8 = Ranking 엔진
STAGE_TRACK = "ranking"


def get_ranking_baseline_tag(base_dir: Path) -> str:
    """
    baselines.yaml에서 ranking_baseline_tag 읽기

    Returns:
        ranking_baseline_tag 값 (Stage7의 run_tag)
    """
    baselines_path = base_dir / "reports" / "history" / "baselines.yaml"

    if not baselines_path.exists():
        print(
            f"ERROR: baselines.yaml이 존재하지 않습니다: {baselines_path}",
            file=sys.stderr,
        )
        print(
            "Stage7을 먼저 완료하여 ranking_baseline_tag를 설정하세요.", file=sys.stderr
        )
        sys.exit(1)

    try:
        with open(baselines_path, encoding="utf-8") as f:
            baselines = yaml.safe_load(f) or {}

        ranking_baseline = baselines.get("ranking_baseline_tag")
        if not ranking_baseline:
            print("ERROR: ranking_baseline_tag가 설정되지 않았습니다.", file=sys.stderr)
            print(
                "Stage7을 먼저 완료하여 ranking_baseline_tag를 설정하세요.",
                file=sys.stderr,
            )
            sys.exit(1)

        return ranking_baseline
    except Exception as e:
        print(f"ERROR: baselines.yaml 읽기 실패: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=f"[코드 매니저] Stage {STAGE_NUM} 실행 (Sector-relative 정규화)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument(
        "--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)"
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default=None,
        help="Baseline tag (없으면 baselines.yaml에서 읽음)",
    )
    parser.add_argument(
        "--change-title",
        type=str,
        default="Sector-relative 정규화 적용",
        help="Change title (History Manifest용)",
    )
    parser.add_argument(
        "--change-summary",
        type=str,
        nargs="*",
        default=[
            "섹터 내 정규화로 업종 편향 완화",
            "Top20 섹터 농도지표(HHI/MaxShare) 산출",
            "랭킹 품질 KPI 추가",
        ],
        help="Change summary (최대 3개)",
    )
    parser.add_argument(
        "--modified-files",
        type=str,
        default="src/tools/run_stage8.py;src/stages/l8_rank_engine.py;src/ranking/score_engine.py",
        help="Modified files",
    )
    parser.add_argument(
        "--modified-functions",
        type=str,
        default="run_stage8;build_ranking_daily;apply_sector_relative_norm;compute_sector_concentration",
        help="Modified functions",
    )
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
        run_tag = generate_run_tag("stage8_sector_relative")

    # baseline_tag 결정 (baselines.yaml에서 ranking_baseline_tag 읽기)
    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        baseline_tag_used = get_ranking_baseline_tag(PROJECT_ROOT)

    print("\n" + "=" * 60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행 (Sector-relative 정규화)")
    print("=" * 60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used (Stage7): {baseline_tag_used}")
    print("=" * 60 + "\n")

    # 로그 파일 경로
    log_file = (
        PROJECT_ROOT / "reports" / "logs" / f"run__stage{STAGE_NUM}__{run_tag}.txt"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 0) L2 재사용 검증 (사전)
    l2_file = PROJECT_ROOT / "data" / "interim" / "fundamentals_annual.parquet"
    if not l2_file.exists():
        print(
            "ERROR: L2 파일이 없습니다. fundamentals_annual.parquet를 먼저 준비하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    l2_hash_before = None
    try:
        l2_hash_full = get_file_hash(l2_file)
        l2_hash_before = l2_hash_full[:16] + "..."
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"[L2 검증] 실행 전 해시: {l2_hash_before}\n")
    except Exception as e:
        print(f"WARNING: L2 해시 계산 실패: {e}", file=sys.stderr)

    # 1) Stage 실행 (L8 Ranking 엔진)
    stage_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "run_all.py"),
        "--config",
        args.config,
        "--run-tag",
        run_tag,
        "--baseline-tag",
        baseline_tag_used,
        "--stage",
        STAGE_NAME,
        "--force-rebuild",
        "--skip-l2",  # L2 재사용 강제
    ]

    if (
        run_command(stage_cmd, PROJECT_ROOT, f"Stage {STAGE_NUM} 실행 (L8)", log_file)
        != 0
    ):
        sys.exit(1)

    # L2 재사용 검증 (사후)
    l2_valid, l2_msg, _, l2_hash_after = verify_l2_reuse(PROJECT_ROOT, log_file)
    if not l2_valid:
        print(f"ERROR: {l2_msg}", file=sys.stderr)
        sys.exit(1)

    # 산출물 확인
    ranking_daily_path = (
        PROJECT_ROOT / "data" / "interim" / run_tag / "ranking_daily.parquet"
    )
    if not ranking_daily_path.exists():
        print(
            f"ERROR: ranking_daily.parquet가 생성되지 않았습니다: {ranking_daily_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    import pandas as pd

    ranking_daily = pd.read_parquet(ranking_daily_path)
    if len(ranking_daily) == 0:
        print("ERROR: ranking_daily.parquet가 비어있습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] ranking_daily.parquet 생성 완료: {len(ranking_daily):,} rows")

    # 2) 섹터 농도 리포트 생성 (필수)
    sector_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "calculate_sector_concentration.py"),
        "--config",
        args.config,
        "--run-tag",
        run_tag,
        "--baseline-tag",
        baseline_tag_used,
        "--top-k",
        "20",
    ]

    if run_command(sector_cmd, PROJECT_ROOT, "섹터 농도 리포트 생성", log_file) != 0:
        print("ERROR: 섹터 농도 리포트 생성 실패", file=sys.stderr)
        sys.exit(1)

    sector_concentration_path = (
        PROJECT_ROOT / "reports" / "ranking" / f"sector_concentration__{run_tag}.csv"
    )
    if not sector_concentration_path.exists():
        print(
            f"ERROR: 섹터 농도 리포트가 생성되지 않았습니다: {sector_concentration_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[OK] 섹터 농도 리포트 생성 완료: {sector_concentration_path}")

    # 3) KPI 생성 (필수)
    kpi_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
        "--config",
        args.config,
        "--tag",
        run_tag,
    ]

    if run_command(kpi_cmd, PROJECT_ROOT, "KPI 생성", log_file) != 0:
        sys.exit(1)

    # 4) Δ 생성 (필수) - Stage7 vs Stage8
    delta_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
        "--baseline-tag",
        baseline_tag_used,
        "--run-tag",
        run_tag,
    ]

    if (
        run_command(
            delta_cmd, PROJECT_ROOT, "Δ 리포트 생성 (Stage7 vs Stage8)", log_file
        )
        != 0
    ):
        print("ERROR: Delta 리포트 생성 실패", file=sys.stderr)
        sys.exit(1)

    delta_csv = (
        PROJECT_ROOT
        / "reports"
        / "delta"
        / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
    )
    delta_md = (
        PROJECT_ROOT
        / "reports"
        / "delta"
        / f"delta_report__{baseline_tag_used}__vs__{run_tag}.md"
    )

    if not delta_csv.exists() or not delta_md.exists():
        print("ERROR: Delta 리포트 파일이 생성되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Delta 리포트 생성 완료: {delta_csv}")

    # 5) Stage8 체크리포트 생성 (필수)
    check_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
        "--config",
        args.config,
        "--stage",
        str(STAGE_NUM),
        "--run-tag",
        run_tag,
        "--baseline-tag",
        baseline_tag_used,
    ]

    if run_command(check_cmd, PROJECT_ROOT, "체크리포트 생성", log_file) != 0:
        print("WARNING: 체크리포트 생성 실패 (계속 진행)", file=sys.stderr)

    # 6) History Manifest 업데이트 (필수)
    track = get_stage_track(STAGE_NUM)
    history_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "update_history_manifest.py"),
        "--config",
        args.config,
        "--stage",
        str(STAGE_NUM),
        "--track",
        track,
        "--run-tag",
        run_tag,
        "--baseline-tag",
        baseline_tag_used,
        "--change-title",
        args.change_title,
    ]

    if args.change_summary:
        history_cmd.extend(["--change-summary"] + args.change_summary)
    if args.modified_files:
        history_cmd.extend(["--modified-files", args.modified_files])
    if args.modified_functions:
        history_cmd.extend(["--modified-functions", args.modified_functions])

    if (
        run_command(history_cmd, PROJECT_ROOT, "History Manifest 업데이트", log_file)
        != 0
    ):
        print("WARNING: History Manifest 업데이트 실패 (계속 진행)", file=sys.stderr)

    # 7) 최종 출력
    output_files = [
        ("산출물", ranking_daily_path),
        ("섹터 농도 리포트", sector_concentration_path),
        ("KPI CSV", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"),
        ("KPI MD", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"),
        ("Delta CSV", delta_csv),
        ("Delta MD", delta_md),
        (
            "체크리포트",
            PROJECT_ROOT
            / "reports"
            / "stages"
            / f"check__stage{STAGE_NUM}__{run_tag}.md",
        ),
        (
            "History Manifest",
            PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet",
        ),
        ("로그", log_file),
    ]

    print_success_summary(run_tag, baseline_tag_used, None, output_files)

    # 완료 기준 검증
    print("\n[완료 기준 검증]")
    checks = {
        "ranking_daily.parquet 존재": ranking_daily_path.exists(),
        "sector_concentration CSV 존재": sector_concentration_path.exists(),
        "Delta CSV 존재": delta_csv.exists(),
        "Delta MD 존재": delta_md.exists(),
        "체크리포트 존재": (
            PROJECT_ROOT
            / "reports"
            / "stages"
            / f"check__stage{STAGE_NUM}__{run_tag}.md"
        ).exists(),
        "History Manifest 존재": (
            PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
        ).exists(),
    }

    all_pass = all(checks.values())
    for check_name, check_result in checks.items():
        status = "[PASS]" if check_result else "[FAIL]"
        print(f"  {status} {check_name}")

    if all_pass:
        print("\n[PASS] 모든 완료 기준 충족")
    else:
        print("\n[FAIL] 일부 완료 기준 미충족")
        sys.exit(1)


if __name__ == "__main__":
    main()
