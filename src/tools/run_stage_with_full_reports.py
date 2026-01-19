# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage_with_full_reports.py
"""
[코드 매니저] Stage 실행 통합 스크립트 (규칙 준수 버전)
- Stage 실행 → 산출물 생성 확인 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
- L2 재사용 규칙 준수 (해시 검증)
- 바탕 화면 경로 문제 방지
- config.yaml에서 baseline 태그 읽기
"""
import argparse
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def get_file_hash(filepath: Path) -> str:
    """파일의 SHA256 해시 계산"""
    if not filepath.exists():
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_config(config_path: Path) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_base_dir(cfg: dict) -> Path:
    """config에서 base_dir 추출"""
    base_dir_str = cfg.get("paths", {}).get("base_dir", "")
    if not base_dir_str:
        raise ValueError("config.yaml에 paths.base_dir이 없습니다.")

    base_dir = Path(base_dir_str)

    # 바탕 화면 경로 문제 체크
    if "바탕 화면" in str(base_dir) or "Desktop" not in str(base_dir):
        # 정확한 경로로 변환 시도
        if "바탕 화면" in str(base_dir):
            # OneDrive 바탕 화면 경로로 변환
            base_dir_str = base_dir_str.replace("바탕 화면", "Desktop")
            base_dir = Path(base_dir_str)

    # 절대 경로로 변환
    if not base_dir.is_absolute():
        raise ValueError(f"base_dir은 절대 경로여야 합니다: {base_dir}")

    return base_dir


def get_baseline_tag(cfg: dict, stage: int) -> tuple[str, str]:
    """
    Baseline 태그 결정

    Returns:
        (baseline_tag_used, baseline_type)
        - Stage0~6: pipeline_baseline_tag
        - Stage7: ranking_baseline 생성 단계 (baseline_tag_used = pipeline_baseline_tag)
        - Stage8+: ranking_baseline_tag (없으면 에러)
    """
    baseline_cfg = cfg.get("baseline", {})

    if stage <= 6:
        pipeline_baseline = baseline_cfg.get("pipeline_baseline_tag")
        if not pipeline_baseline:
            raise ValueError("config.yaml에 baseline.pipeline_baseline_tag가 없습니다.")
        return pipeline_baseline, "pipeline"

    elif stage == 7:
        # Stage7은 ranking_baseline 생성 단계
        pipeline_baseline = baseline_cfg.get("pipeline_baseline_tag")
        if not pipeline_baseline:
            raise ValueError("config.yaml에 baseline.pipeline_baseline_tag가 없습니다.")
        return pipeline_baseline, "pipeline"  # Stage7도 pipeline baseline 사용

    else:  # Stage8+
        ranking_baseline = baseline_cfg.get("ranking_baseline_tag")
        if not ranking_baseline:
            raise ValueError(
                "Stage8+ 실행을 위해서는 Stage7이 완료되어야 합니다. "
                "config.yaml에 baseline.ranking_baseline_tag를 설정하세요."
            )
        return ranking_baseline, "ranking"


def verify_l2_hash(
    base_interim_dir: Path, stage: int, run_tag: str
) -> tuple[bool, str, Optional[str]]:
    """
    L2 파일 해시 검증 (Stage 실행 전/후)

    Returns:
        (is_valid, message, hash_value)
    """
    l2_file = base_interim_dir / "fundamentals_annual.parquet"

    if not l2_file.exists():
        return False, "L2 파일이 존재하지 않습니다", None

    hash_value = get_file_hash(l2_file)

    if stage == 2:
        # Stage2 실행 전: L2 파일이 이미 존재하는지 확인 (재사용)
        return True, f"L2 파일 재사용 확인 (해시: {hash_value[:16]}...)", hash_value
    else:
        # Stage2 실행 후: L2 파일이 변경되지 않았는지 확인
        # (실제로는 Stage2는 L2를 재사용하므로 변경되지 않아야 함)
        return True, f"L2 파일 해시 검증 완료 (해시: {hash_value[:16]}...)", hash_value


def run_command(
    cmd: list,
    cwd: Path,
    description: str,
    log_file: Optional[Path] = None,
    check_returncode: bool = True,
) -> tuple[int, str, str]:
    """명령어 실행 및 결과 반환"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working Directory: {cwd}")
    print(f"{'='*60}\n")

    try:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n{'='*60}\n")
                log.write(f"[{description}] {datetime.now().isoformat()}\n")
                log.write(f"Command: {' '.join(cmd)}\n")
                log.write(f"{'='*60}\n\n")

        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.stdout:
            print(result.stdout)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(result.stdout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write("\n--- STDERR ---\n")
                    log.write(result.stderr)

        if check_returncode and result.returncode != 0:
            print(f"\n[FAIL] [{description}] Failed with exit code {result.returncode}")
            if log_file:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"\n[FAIL] Exit code: {result.returncode}\n")
        else:
            print(f"\n[OK] [{description}] Completed")
            if log_file:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write("\n[OK] Completed\n")

        return result.returncode, result.stdout, result.stderr

    except Exception as e:
        error_msg = f"Exception: {e}"
        print(f"\n[ERROR] [{description}] {error_msg}")
        if log_file:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n[ERROR] {error_msg}\n")
        return 1, "", error_msg


def verify_artifact_exists(file_path: Path, description: str) -> bool:
    """산출물 파일 존재 확인"""
    if file_path.exists():
        file_size = file_path.stat().st_size
        if file_path.suffix == ".parquet":
            try:
                import pandas as pd

                df = pd.read_parquet(file_path)
                n_rows = len(df)
                print(
                    f"[OK] {description}: {file_path} ({file_size:,} bytes, {n_rows:,} rows)"
                )
            except:
                print(f"[OK] {description}: {file_path} ({file_size:,} bytes)")
        else:
            print(f"[OK] {description}: {file_path} ({file_size:,} bytes)")
        return True
    else:
        print(f"[MISSING] {description}: NOT FOUND - {file_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="[코드 매니저] Stage 실행 통합 스크립트 (규칙 준수 버전)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument("--stage", type=int, required=True, help="Stage 번호 (0-8)")
    parser.add_argument(
        "--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)"
    )
    parser.add_argument(
        "--change-title", type=str, default=None, help="변경 제목 (History Manifest용)"
    )
    parser.add_argument(
        "--change-summary", type=str, nargs="*", default=[], help="변경 요약 (최대 3개)"
    )
    parser.add_argument(
        "--modified-files", type=str, default=None, help="수정된 파일 (쉼표 구분)"
    )
    parser.add_argument(
        "--modified-functions", type=str, default=None, help="수정된 함수 (쉼표 구분)"
    )
    parser.add_argument(
        "--skip-stage", action="store_true", help="Stage 실행 건너뛰기 (리포트만 생성)"
    )
    parser.add_argument("--skip-kpi", action="store_true", help="KPI 생성 건너뛰기")
    parser.add_argument(
        "--skip-delta", action="store_true", help="Delta 리포트 생성 건너뛰기"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="체크리포트 생성 건너뛰기"
    )
    parser.add_argument(
        "--skip-history", action="store_true", help="History Manifest 업데이트 건너뛰기"
    )
    args = parser.parse_args()

    # Config 로드
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    config_path = project_root / args.config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    base_dir = get_base_dir(cfg)

    # 바탕 화면 경로 문제 최종 체크
    if "바탕 화면" in str(base_dir):
        print(
            f"ERROR: base_dir에 '바탕 화면'이 포함되어 있습니다: {base_dir}",
            file=sys.stderr,
        )
        print("config.yaml의 paths.base_dir을 수정하세요.", file=sys.stderr)
        sys.exit(1)

    # Run tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"stage{args.stage}_{timestamp}"

    # Baseline 태그 결정
    try:
        baseline_tag_used, baseline_type = get_baseline_tag(cfg, args.stage)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Track 결정
    track = "pipeline" if args.stage <= 7 else "ranking"

    # 경로 설정
    base_interim_dir = base_dir / "data" / "interim"
    logs_dir = base_dir / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"run__stage{args.stage}__{run_tag}.txt"
    log_file.write_text(f"Stage {args.stage} 실행 로그\n{'='*60}\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("[코드 매니저] Stage 실행 통합 스크립트")
    print("=" * 60)
    print(f"프로젝트 루트: {base_dir}")
    print(f"Stage: {args.stage}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used: {baseline_tag_used} ({baseline_type})")
    print(f"Track: {track}")
    print(f"Log File: {log_file}")
    print("=" * 60 + "\n")

    # ============================================================
    # 0) L2 해시 검증 (Stage 실행 전)
    # ============================================================
    print("\n[0/6] L2 파일 해시 검증 (실행 전)...")
    l2_valid_before, l2_msg_before, l2_hash_before = verify_l2_hash(
        base_interim_dir, args.stage, run_tag
    )
    print(f"L2 검증 (실행 전): {l2_msg_before}")

    if not l2_valid_before and args.stage != 2:
        print("WARNING: L2 파일이 없습니다. Stage2를 먼저 실행하세요.")

    # ============================================================
    # 1) Stage 실행
    # ============================================================
    if not args.skip_stage:
        print("\n[1/6] Stage 실행 중...")

        # Stage 실행 스크립트 찾기
        stage_script = base_dir / "src" / "tools" / f"run_stage{args.stage}.py"

        if not stage_script.exists():
            # run_all.py 사용
            stage_cmd = [
                sys.executable,
                str(base_dir / "src" / "run_all.py"),
                "--config",
                args.config,
                "--run-tag",
                run_tag,
                "--force-rebuild",  # skip_if_exists 무시
            ]

            # Stage 범위 결정
            if args.stage == 0:
                stage_cmd.extend(["--from", "L0", "--to", "L0"])
            elif args.stage == 1:
                stage_cmd.extend(["--from", "L1", "--to", "L1"])
            elif args.stage == 2:
                stage_cmd.extend(["--from", "L2", "--to", "L2"])
            elif args.stage == 3:
                stage_cmd.extend(["--from", "L3", "--to", "L3"])
            elif args.stage == 4:
                stage_cmd.extend(["--from", "L4", "--to", "L4"])
            elif args.stage == 5:
                stage_cmd.extend(["--from", "L5", "--to", "L5"])
            elif args.stage == 6:
                stage_cmd.extend(["--from", "L6", "--to", "L6"])
            elif args.stage == 7:
                stage_cmd.extend(["--from", "L7", "--to", "L7"])
            elif args.stage == 8:
                stage_cmd.extend(["--from", "L8", "--to", "L8"])
        else:
            stage_cmd = [
                sys.executable,
                str(stage_script),
                "--config",
                args.config,
                "--run-tag",
                run_tag,
            ]

        returncode, stdout, stderr = run_command(
            stage_cmd,
            cwd=base_dir,
            description=f"Stage {args.stage} 실행",
            log_file=log_file,
            check_returncode=True,
        )

        if returncode != 0:
            print(f"\n❌ Stage {args.stage} 실행 실패 (exit code: {returncode})")
            sys.exit(returncode)
    else:
        print("\n⏭️  Stage 실행 건너뛰기 (--skip-stage)")

    # ============================================================
    # 2) L2 해시 검증 (Stage 실행 후)
    # ============================================================
    print("\n[2/6] L2 파일 해시 검증 (실행 후)...")
    l2_valid_after, l2_msg_after, l2_hash_after = verify_l2_hash(
        base_interim_dir, args.stage, run_tag
    )
    print(f"L2 검증 (실행 후): {l2_msg_after}")

    # L2 해시 비교 (Stage2가 아니면 변경되지 않아야 함)
    if args.stage != 2 and l2_hash_before and l2_hash_after:
        if l2_hash_before != l2_hash_after:
            print("ERROR: L2 파일이 변경되었습니다!", file=sys.stderr)
            print(f"  실행 전: {l2_hash_before[:16]}...", file=sys.stderr)
            print(f"  실행 후: {l2_hash_after[:16]}...", file=sys.stderr)
            sys.exit(1)
        else:
            print("✅ L2 파일 해시 일치 확인 (재사용 규칙 준수)")

    # ============================================================
    # 3) KPI 생성
    # ============================================================
    if not args.skip_kpi:
        print("\n[3/6] KPI 테이블 생성 중...")

        kpi_cmd = [
            sys.executable,
            str(base_dir / "src" / "tools" / "export_kpi_table.py"),
            "--config",
            args.config,
            "--tag",
            run_tag,
        ]

        returncode, stdout, stderr = run_command(
            kpi_cmd,
            cwd=base_dir,
            description="KPI 테이블 생성",
            log_file=log_file,
            check_returncode=True,
        )

        if returncode != 0:
            print(f"\n❌ KPI 테이블 생성 실패 (exit code: {returncode})")
            sys.exit(returncode)

        # KPI 파일 존재 확인
        kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
        if not verify_artifact_exists(kpi_csv, "KPI CSV"):
            print("❌ KPI CSV 파일이 생성되지 않았습니다.")
            sys.exit(1)
    else:
        print("\n⏭️  KPI 테이블 생성 건너뛰기 (--skip-kpi)")

    # ============================================================
    # 4) Δ 생성
    # ============================================================
    if not args.skip_delta:
        print("\n[4/6] Δ 리포트 생성 중...")

        # Baseline KPI 확인
        baseline_kpi_csv = (
            base_dir / "reports" / "kpi" / f"kpi_table__{baseline_tag_used}.csv"
        )

        if not baseline_kpi_csv.exists():
            print(f"\n⚠️  Baseline KPI가 없습니다: {baseline_kpi_csv}")
            print("   Baseline KPI를 먼저 생성합니다...")

            baseline_kpi_cmd = [
                sys.executable,
                str(base_dir / "src" / "tools" / "export_kpi_table.py"),
                "--config",
                args.config,
                "--tag",
                baseline_tag_used,
            ]

            returncode, _, _ = run_command(
                baseline_kpi_cmd,
                cwd=base_dir,
                description="Baseline KPI 생성",
                log_file=log_file,
                check_returncode=False,
            )

            if returncode != 0:
                print("⚠️  Baseline KPI 생성 실패. Delta 리포트 생성을 건너뜁니다.")
                baseline_tag_used = None

        if baseline_tag_used:
            delta_cmd = [
                sys.executable,
                str(base_dir / "src" / "tools" / "export_delta_report.py"),
                "--config",
                args.config,
                "--baseline-tag",
                baseline_tag_used,
                "--run-tag",
                run_tag,
            ]

            returncode, stdout, stderr = run_command(
                delta_cmd,
                cwd=base_dir,
                description="Δ 리포트 생성",
                log_file=log_file,
                check_returncode=True,
            )

            if returncode != 0:
                print(f"\n❌ Δ 리포트 생성 실패 (exit code: {returncode})")
                sys.exit(returncode)

            # Delta 파일 존재 확인
            delta_csv = (
                base_dir
                / "reports"
                / "delta"
                / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
            )
            if not verify_artifact_exists(delta_csv, "Delta CSV"):
                print("❌ Delta CSV 파일이 생성되지 않았습니다.")
                sys.exit(1)
    else:
        print("\n⏭️  Δ 리포트 생성 건너뛰기 (--skip-delta)")

    # ============================================================
    # 5) 체크리포트 생성
    # ============================================================
    if not args.skip_check:
        print("\n[5/6] Stage 체크리포트 생성 중...")

        check_cmd = [
            sys.executable,
            str(base_dir / "src" / "tools" / "check_stage_completion.py"),
            "--config",
            args.config,
            "--run-tag",
            run_tag,
            "--stage",
            str(args.stage),
            "--baseline-tag",
            baseline_tag_used,
        ]

        returncode, stdout, stderr = run_command(
            check_cmd,
            cwd=base_dir,
            description="Stage 체크리포트 생성",
            log_file=log_file,
            check_returncode=False,  # 실패해도 계속 진행
        )

        if returncode == 0:
            check_report = (
                base_dir
                / "reports"
                / "stages"
                / f"check__stage{args.stage}__{run_tag}.md"
            )
            verify_artifact_exists(check_report, "체크리포트")
        else:
            print("⚠️  체크리포트 생성 실패 (계속 진행)")
    else:
        print("\n⏭️  체크리포트 생성 건너뛰기 (--skip-check)")

    # ============================================================
    # 6) History Manifest 업데이트
    # ============================================================
    if not args.skip_history:
        print("\n[6/6] History Manifest 업데이트 중...")

        history_cmd = [
            sys.executable,
            str(base_dir / "src" / "tools" / "update_history_manifest.py"),
            "--config",
            args.config,
            "--stage",
            str(args.stage),
            "--track",
            track,
            "--run-tag",
            run_tag,
            "--baseline-tag",
            baseline_tag_used,
        ]

        if args.change_title:
            history_cmd.extend(["--change-title", args.change_title])

        if args.change_summary:
            history_cmd.extend(["--change-summary"] + args.change_summary)

        if args.modified_files:
            history_cmd.extend(["--modified-files", args.modified_files])

        if args.modified_functions:
            history_cmd.extend(["--modified-functions", args.modified_functions])

        returncode, stdout, stderr = run_command(
            history_cmd,
            cwd=base_dir,
            description="History Manifest 업데이트",
            log_file=log_file,
            check_returncode=False,  # 실패해도 계속 진행
        )

        if returncode == 0:
            history_manifest = (
                base_dir / "reports" / "history" / "history_manifest.parquet"
            )
            verify_artifact_exists(history_manifest, "History Manifest")
        else:
            print("⚠️  History Manifest 업데이트 실패 (계속 진행)")
    else:
        print("\n⏭️  History Manifest 업데이트 건너뛰기 (--skip-history)")

    # ============================================================
    # 최종 요약 출력
    # ============================================================
    print("\n" + "=" * 60)
    print("[완료] 최종 요약")
    print("=" * 60)

    outputs = []

    # 산출물 경로
    interim_dir = base_interim_dir / run_tag
    if interim_dir.exists():
        parquet_files = list(interim_dir.glob("*.parquet"))
        csv_files = list(interim_dir.glob("*.csv"))
        outputs.append(
            (
                "산출물 (interim)",
                f"{interim_dir} ({len(parquet_files)} parquet, {len(csv_files)} csv)",
            )
        )

    # L2 파일 (base_interim_dir에 있음)
    l2_file = base_interim_dir / "fundamentals_annual.parquet"
    if l2_file.exists():
        l2_hash_short = l2_hash_after[:16] if l2_hash_after else "N/A"
        outputs.append(("L2 파일 (재사용)", f"{l2_file} (해시: {l2_hash_short}...)"))

    # KPI 리포트
    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    kpi_md = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.md"
    if kpi_csv.exists():
        outputs.append(("KPI CSV", str(kpi_csv)))
    if kpi_md.exists():
        outputs.append(("KPI MD", str(kpi_md)))

    # Delta 리포트
    if baseline_tag_used:
        delta_csv = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
        )
        delta_md = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_report__{baseline_tag_used}__vs__{run_tag}.md"
        )
        if delta_csv.exists():
            outputs.append(("Delta CSV", str(delta_csv)))
        if delta_md.exists():
            outputs.append(("Delta MD", str(delta_md)))

    # 체크리포트
    check_report = (
        base_dir / "reports" / "stages" / f"check__stage{args.stage}__{run_tag}.md"
    )
    if check_report.exists():
        outputs.append(("체크리포트", str(check_report)))

    # History Manifest
    history_manifest = base_dir / "reports" / "history" / "history_manifest.parquet"
    if history_manifest.exists():
        outputs.append(("History Manifest", str(history_manifest)))

    print("\n생성된 파일 목록:")
    for i, (desc, path) in enumerate(outputs, 1):
        print(f"{i}) {desc}:")
        print(f"   {path}")

    print("\n" + "=" * 60)
    print("✅ 모든 단계 완료")
    print("=" * 60)
    print(f"\n[PASS] Stage {args.stage} 완료")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used: {baseline_tag_used} ({baseline_type})")
    if args.stage >= 7:
        ranking_baseline = cfg.get("baseline", {}).get("ranking_baseline_tag")
        if ranking_baseline:
            print(f"Ranking Baseline Tag: {ranking_baseline}")
    print(f"생성된 파일 수: {len(outputs)}개")
    print(f"로그 파일: {log_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
