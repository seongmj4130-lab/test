# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage9.py
"""
[코드 매니저] Stage 9 실행 엔트리포인트
ranking_daily에 설명가능성 컬럼(팩터/피처 기여도) 추가

Stage 실행 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import sys
from pathlib import Path

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

STAGE_NUM = 9
STAGE_TRACK = "ranking"


def main():
    parser = argparse.ArgumentParser(
        description=f"[코드 매니저] Stage {STAGE_NUM} 실행 (랭킹 설명가능성 확장)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument(
        "--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)"
    )
    parser.add_argument(
        "--baseline-tag", type=str, required=True, help="Baseline tag (Stage8 run_tag)"
    )
    parser.add_argument(
        "--change-title",
        type=str,
        default="랭킹 설명가능성 확장",
        help="Change title (History Manifest용)",
    )
    parser.add_argument(
        "--change-summary",
        type=str,
        nargs="*",
        default=[
            "기여도/상위 기여 피처 저장",
            "Top/Bottom 설명 UI 데이터 제공",
            "ranking_daily 컬럼 확장",
        ],
        help="Change summary (최대 3개)",
    )
    parser.add_argument(
        "--modified-files",
        type=str,
        default="src/stages/ranking_explainability.py;src/tools/run_stage9.py",
        help="Modified files",
    )
    parser.add_argument(
        "--modified-functions",
        type=str,
        default="build_ranking_with_contrib;make_snapshot",
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
        run_tag = generate_run_tag("stage9_ranking_explainability")

    # baseline_tag_used 결정 (지정되지 않으면 최신 Stage8 자동 감지)
    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        # 최신 Stage8 run_tag 자동 감지
        import glob

        stage8_dirs = glob.glob(
            str(PROJECT_ROOT / "data" / "interim" / "stage8_sector_relative_*")
        )
        if stage8_dirs:
            # 디렉토리명에서 run_tag 추출하고 최신 것 선택
            stage8_tags = [Path(d).name for d in stage8_dirs]
            stage8_tags.sort(reverse=True)  # 최신이 앞에
            baseline_tag_used = stage8_tags[0]
            print(f"[INFO] 최신 Stage8 run_tag 자동 감지: {baseline_tag_used}")
        else:
            print(
                "ERROR: Stage8 run_tag를 찾을 수 없습니다. --baseline-tag로 지정하세요.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행 (랭킹 설명가능성 확장)")
    print("=" * 60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used (Stage8): {baseline_tag_used}")
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

    # Baseline ranking_daily 확인
    baseline_ranking_path = (
        PROJECT_ROOT / "data" / "interim" / baseline_tag_used / "ranking_daily.parquet"
    )
    if not baseline_ranking_path.exists():
        print(
            f"ERROR: Baseline ranking_daily가 없습니다: {baseline_ranking_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1) Stage9 실행 (랭킹 설명가능성 확장)
    print("\n[1/6] Stage9 실행 (랭킹 설명가능성 확장)...")

    # Python 스크립트로 직접 실행
    stage9_script = (
        PROJECT_ROOT / "src" / "stages" / "run_stage9_ranking_explainability.py"
    )

    # Python 모듈 import
    import sys

    src_dir = PROJECT_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import pandas as pd

    from src.stages.ranking.ranking_explainability import (
        build_ranking_with_explainability,
        make_snapshot_with_explainability,
    )
    from src.utils.config import get_path, load_config

    # 설정 로드
    cfg = load_config(str(config_path))
    l8 = cfg.get("l8", {})
    normalization_method = l8.get("normalization_method", "percentile")
    feature_groups_config = l8.get("feature_groups_config", None)
    sector_col = l8.get("sector_col", "sector_name")
    use_sector_relative = l8.get("use_sector_relative", True)

    # Baseline ranking_daily 로드
    baseline_ranking = pd.read_parquet(baseline_ranking_path)
    print(f"[OK] Baseline ranking_daily 로드: {len(baseline_ranking):,} rows")

    # 원본 입력 데이터 로드 (dataset_daily 또는 panel_merged_daily)
    base_interim_dir = get_path(cfg, "data_interim")

    # 여러 경로 시도
    input_df = None
    candidate_paths = [
        base_interim_dir / baseline_tag_used / "dataset_daily.parquet",
        base_interim_dir / baseline_tag_used / "panel_merged_daily.parquet",
        base_interim_dir / "dataset_daily.parquet",
        base_interim_dir / "panel_merged_daily.parquet",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                input_df = pd.read_parquet(path)
                print(f"[OK] 입력 데이터 로드: {path} ({len(input_df):,} rows)")
                break
            except Exception as e:
                print(f"WARNING: {path} 로드 실패: {e}", file=sys.stderr)
                continue

    if input_df is None:
        print("ERROR: 입력 데이터를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    # feature_groups_config 경로 처리
    if feature_groups_config:
        feature_groups_path = PROJECT_ROOT / feature_groups_config
        if not feature_groups_path.exists():
            feature_groups_path = None
    else:
        feature_groups_path = None

    # 설명가능성 컬럼 추가
    try:
        ranking_with_explainability = build_ranking_with_explainability(
            baseline_ranking,
            input_df,
            feature_groups_config=feature_groups_path,
            normalization_method=normalization_method,
            sector_col=sector_col,
            use_sector_relative=use_sector_relative,
            top_k_features=5,
        )
        print(
            f"[OK] 설명가능성 컬럼 추가 완료: {len(ranking_with_explainability):,} rows"
        )
        print(
            f"  추가된 컬럼: {[c for c in ranking_with_explainability.columns if c.startswith('contrib_') or c == 'top_features']}"
        )
    except Exception as e:
        print(f"ERROR: 설명가능성 컬럼 추가 실패: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 저장
    interim_dir = base_interim_dir / run_tag
    interim_dir.mkdir(parents=True, exist_ok=True)

    ranking_output_path = interim_dir / "ranking_daily.parquet"
    ranking_with_explainability.to_parquet(ranking_output_path, index=False)
    print(f"[OK] ranking_daily 저장: {ranking_output_path}")

    # L2 재사용 검증 (사후)
    l2_valid, l2_msg, _, l2_hash_after = verify_l2_reuse(PROJECT_ROOT, log_file)
    if not l2_valid:
        print(f"ERROR: {l2_msg}", file=sys.stderr)
        sys.exit(1)

    # 2) ranking_snapshot 생성 (top_features 포함)
    print("\n[2/6] ranking_snapshot 생성 중...")

    try:
        snapshot = make_snapshot_with_explainability(
            ranking_with_explainability, top_k=10
        )
        snapshot_dir = PROJECT_ROOT / "reports" / "ranking"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"ranking_snapshot__{run_tag}.csv"
        snapshot.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
        print(f"[OK] ranking_snapshot 저장: {snapshot_path} ({len(snapshot)} rows)")
    except Exception as e:
        print(f"WARNING: ranking_snapshot 생성 실패: {e}", file=sys.stderr)

    # 3) KPI 생성 (필수)
    print("\n[3/6] KPI 생성 중...")

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

    # 4) Δ 생성 (필수) - Stage8 vs Stage9
    print("\n[4/6] Δ 리포트 생성 중...")

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
            delta_cmd, PROJECT_ROOT, "Δ 리포트 생성 (Stage8 vs Stage9)", log_file
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

    # 5) Stage9 체크리포트 생성 (필수)
    print("\n[5/6] 체크리포트 생성 중...")

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
    print("\n[6/6] History Manifest 업데이트 중...")

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
        ("산출물", ranking_output_path),
        ("Snapshot", snapshot_path if "snapshot_path" in locals() else None),
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

    # None 제거
    output_files = [(desc, path) for desc, path in output_files if path is not None]

    print_success_summary(run_tag, baseline_tag_used, None, output_files)

    # 완료 기준 검증
    print("\n[완료 기준 검증]")
    checks = {
        "ranking_daily.parquet 존재": ranking_output_path.exists()
        and len(ranking_with_explainability) > 0,
        "contrib_* 컬럼 존재": any(
            c.startswith("contrib_") for c in ranking_with_explainability.columns
        ),
        "top_features 컬럼 존재": "top_features" in ranking_with_explainability.columns,
        "KPI CSV 존재": (
            PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
        ).exists(),
        "Delta CSV 존재": delta_csv.exists(),
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
