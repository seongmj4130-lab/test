# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage_pipeline.py
"""
[Cursor 실행형 코드매니저] Stage 실행 통합 파이프라인
- 단일 entrypoint로 모든 Stage 실행 및 리포트 생성
- Global baseline 자동 탐지 (Stage12 최신)
- 직전 Stage baseline + Global baseline 대비 Delta 리포트 생성
- L2 재사용 규칙 준수
- 경로 버그 방지
"""
import argparse
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# 프로젝트 루트 고정
PROJECT_ROOT = Path(r"C:\Users\seong\OneDrive\Desktop\bootcamp\03_code")


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
    """config에서 base_dir 추출 및 경로 버그 방지"""
    base_dir_str = cfg.get("paths", {}).get("base_dir", "")
    if not base_dir_str:
        raise ValueError("config.yaml에 paths.base_dir이 없습니다.")

    base_dir = Path(base_dir_str)

    # 바탕 화면 경로 문제 체크 및 교정
    if "바탕 화면" in str(base_dir):
        base_dir_str = base_dir_str.replace("바탕 화면", "Desktop")
        base_dir = Path(base_dir_str)
        print(f"[경로 교정] '바탕 화면' → 'Desktop'으로 변경: {base_dir}")

    # 절대 경로로 변환
    if not base_dir.is_absolute():
        raise ValueError(f"base_dir은 절대 경로여야 합니다: {base_dir}")

    return base_dir


def detect_global_baseline(base_dir: Path) -> Optional[str]:
    """
    Global baseline 탐지: Stage12 최신 run_tag
    history_manifest.csv에서 stage_no=12이고 가장 최신인 것 찾기
    """
    history_csv = base_dir / "reports" / "history" / "history_manifest.csv"

    if not history_csv.exists():
        print(f"[WARNING] history_manifest.csv가 없습니다: {history_csv}")
        return None

    try:
        df = pd.read_csv(history_csv)

        # Stage12 필터링
        stage12_df = df[df["stage_no"] == 12].copy()

        if stage12_df.empty:
            print("[WARNING] Stage12 레코드가 없습니다.")
            return None

        # created_at 기준 정렬 (최신이 위로)
        if "created_at" in stage12_df.columns:
            stage12_df = stage12_df.sort_values("created_at", ascending=False)

        global_baseline = stage12_df.iloc[0]["run_tag"]
        print(f"[Global Baseline] 탐지됨: {global_baseline}")
        return global_baseline

    except Exception as e:
        print(f"[WARNING] Global baseline 탐지 실패: {e}", file=sys.stderr)
        return None


def detect_prev_stage_baseline(base_dir: Path, stage: int) -> Optional[str]:
    """
    직전 Stage baseline 탐지: 같은 Stage 번호의 최신 run_tag
    history_manifest.csv에서 stage_no={stage}이고 가장 최신인 것 찾기
    """
    history_csv = base_dir / "reports" / "history" / "history_manifest.csv"

    if not history_csv.exists():
        return None

    try:
        df = pd.read_csv(history_csv)

        # 같은 Stage 필터링
        stage_df = df[df["stage_no"] == stage].copy()

        if stage_df.empty:
            return None

        # created_at 기준 정렬 (최신이 위로)
        if "created_at" in stage_df.columns:
            stage_df = stage_df.sort_values("created_at", ascending=False)

        prev_baseline = stage_df.iloc[0]["run_tag"]
        print(f"[Prev Stage Baseline] 탐지됨: {prev_baseline}")
        return prev_baseline

    except Exception:
        return None


def verify_l2_hash(
    base_interim_dir: Path, stage: int
) -> tuple[bool, str, Optional[str]]:
    """
    L2 파일 해시 검증 (재사용 규칙 준수)

    Returns:
        (is_valid, message, hash_value)
    """
    l2_file = base_interim_dir / "fundamentals_annual.parquet"

    if not l2_file.exists():
        if stage == 2:
            return False, "L2 파일이 존재하지 않습니다 (Stage2 실행 필요)", None
        else:
            return True, "L2 파일이 없지만 Stage2가 아니므로 계속 진행", None

    hash_value = get_file_hash(l2_file)

    if stage == 2:
        # Stage2 실행 전: L2 파일이 이미 존재하는지 확인 (재사용)
        return True, f"L2 파일 재사용 확인 (해시: {hash_value[:16]}...)", hash_value
    else:
        # Stage2 실행 후: L2 파일이 변경되지 않았는지 확인
        return True, f"L2 파일 해시 검증 완료 (해시: {hash_value[:16]}...)", hash_value


def run_command(
    cmd: list[str], cwd: Path, description: str, check_returncode: bool = True
) -> tuple[int, str, str]:
    """명령어 실행 및 결과 반환"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working Directory: {cwd}")
    print(f"{'='*60}\n")

    try:
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
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if check_returncode and result.returncode != 0:
            print(f"\n[FAIL] [{description}] Failed with exit code {result.returncode}")
        else:
            print(f"\n[OK] [{description}] Completed")

        return result.returncode, result.stdout, result.stderr

    except Exception as e:
        print(f"\n[ERROR] [{description}] Exception: {e}")
        return 1, "", str(e)


def verify_artifact_exists(file_path: Path, description: str) -> bool:
    """산출물 파일 존재 확인"""
    if file_path.exists():
        file_size = file_path.stat().st_size
        print(f"[OK] {description}: {file_path} ({file_size:,} bytes)")
        return True
    else:
        print(f"[MISSING] {description}: NOT FOUND - {file_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="[Cursor 실행형 코드매니저] Stage 실행 통합 파이프라인"
    )
    parser.add_argument("--stage", type=int, required=True, help="Stage 번호 (0-13)")
    parser.add_argument(
        "--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)"
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default=None,
        help="Baseline tag (직전 Stage, 없으면 자동 탐지)",
    )
    parser.add_argument(
        "--global-baseline-tag",
        type=str,
        default=None,
        help="Global baseline tag (Stage12 최신, 없으면 자동 탐지)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument(
        "--skip-stage", action="store_true", help="Stage 실행 건너뛰기 (리포트만 생성)"
    )
    parser.add_argument("--skip-kpi", action="store_true", help="KPI 생성 건너뛰기")
    parser.add_argument(
        "--skip-delta", action="store_true", help="Delta 리포트 생성 건너뛰기"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="Stage Check 리포트 생성 건너뛰기"
    )
    parser.add_argument(
        "--skip-history", action="store_true", help="History Manifest 업데이트 건너뛰기"
    )
    args = parser.parse_args()

    # 프로젝트 루트 확인
    if not PROJECT_ROOT.exists():
        print(
            f"ERROR: 프로젝트 루트가 존재하지 않습니다: {PROJECT_ROOT}", file=sys.stderr
        )
        sys.exit(1)

    # Config 로드
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config 파일이 존재하지 않습니다: {config_path}", file=sys.stderr)
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
    # 1) Global baseline (Stage12 최신)
    if args.global_baseline_tag:
        global_baseline_tag = args.global_baseline_tag
    else:
        global_baseline_tag = detect_global_baseline(base_dir)

    # 2) 직전 Stage baseline
    if args.baseline_tag:
        baseline_tag = args.baseline_tag
    else:
        baseline_tag = detect_prev_stage_baseline(base_dir, args.stage)
        if not baseline_tag:
            # 직전 Stage가 없으면 global baseline 사용
            baseline_tag = global_baseline_tag
            print(f"[Baseline] 직전 Stage가 없어 Global baseline 사용: {baseline_tag}")

    # Track 결정
    track = "pipeline" if args.stage <= 7 else "ranking"

    # 경로 설정
    base_interim_dir = base_dir / "data" / "interim"
    logs_dir = base_dir / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Cursor 실행형 코드매니저] Stage 실행 통합 파이프라인")
    print("=" * 60)
    print(f"프로젝트 루트: {base_dir}")
    print(f"Stage: {args.stage}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag (직전): {baseline_tag or 'N/A'}")
    print(f"Global Baseline Tag: {global_baseline_tag or 'N/A'}")
    print(f"Track: {track}")
    print("=" * 60 + "\n")

    # ============================================================
    # 0) L2 해시 검증 (Stage 실행 전)
    # ============================================================
    print("\n[0/7] L2 파일 해시 검증 (실행 전)...")
    l2_valid_before, l2_msg_before, l2_hash_before = verify_l2_hash(
        base_interim_dir, args.stage
    )
    print(f"L2 검증 (실행 전): {l2_msg_before}")

    if not l2_valid_before and args.stage != 2:
        print("WARNING: L2 파일이 없습니다. Stage2를 먼저 실행하세요.")

    # ============================================================
    # 1) Stage 실행
    # ============================================================
    if not args.skip_stage:
        print("\n[1/7] Stage 실행 중...")

        # Stage 실행 스크립트 찾기
        # [Stage13] Stage13은 run_stage13.py를 호출하지 않고 run_all.py를 직접 사용
        if args.stage == 13:
            stage_script = None  # run_all.py 사용
        else:
            stage_script = base_dir / "src" / "tools" / f"run_stage{args.stage}.py"

        if stage_script and stage_script.exists():
            stage_cmd = [
                sys.executable,
                str(stage_script),
                "--config",
                args.config,
                "--run-tag",
                run_tag,
            ]
        else:
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

            # [Stage13] baseline_tag 전달 (L7 입력 산출물 preload용)
            if args.stage == 13 and baseline_tag:
                stage_cmd.extend(["--baseline-tag", baseline_tag])

            # Stage 범위 결정
            stage_map = {
                0: ("L0", "L0"),
                1: ("L1", "L1"),
                2: ("L2", "L2"),
                3: ("L3", "L3"),
                4: ("L4", "L4"),
                5: ("L5", "L5"),
                6: ("L6", "L6"),
                7: ("L7", "L7"),
                8: ("L8", "L8"),
                9: ("L9", "L9"),
                10: ("L10", "L10"),
                11: ("L11", "L11"),
                12: ("L12", "L12"),
                13: ("L7", "L7"),  # [Stage13] L7 재실행 (K_eff 복원)
            }

            if args.stage in stage_map:
                from_stage, to_stage = stage_map[args.stage]
                stage_cmd.extend(["--from", from_stage, "--to", to_stage])

        returncode, stdout, stderr = run_command(
            stage_cmd,
            cwd=base_dir,
            description=f"Stage {args.stage} 실행",
            check_returncode=True,
        )

        if returncode != 0:
            print(f"\n[FAIL] Stage {args.stage} 실행 실패 (exit code: {returncode})")
            sys.exit(returncode)
    else:
        print("\n[SKIP] Stage 실행 건너뛰기 (--skip-stage)")

    # ============================================================
    # 2) L2 해시 검증 (Stage 실행 후)
    # ============================================================
    print("\n[2/7] L2 파일 해시 검증 (실행 후)...")
    l2_valid_after, l2_msg_after, l2_hash_after = verify_l2_hash(
        base_interim_dir, args.stage
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
            print("[OK] L2 파일 해시 일치 확인 (재사용 규칙 준수)")

    # ============================================================
    # 3) KPI 생성
    # ============================================================
    if not args.skip_kpi:
        print("\n[3/7] KPI 테이블 생성 중...")

        kpi_cmd = [
            sys.executable,
            str(base_dir / "src" / "tools" / "export_kpi_table.py"),
            "--config",
            args.config,
            "--tag",
            run_tag,
        ]

        returncode, stdout, stderr = run_command(
            kpi_cmd, cwd=base_dir, description="KPI 테이블 생성", check_returncode=True
        )

        if returncode != 0:
            print(f"\n[FAIL] KPI 테이블 생성 실패 (exit code: {returncode})")
            sys.exit(returncode)

        # KPI 파일 존재 확인
        kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
        if not verify_artifact_exists(kpi_csv, "KPI CSV"):
            print("[FAIL] KPI CSV 파일이 생성되지 않았습니다.")
            sys.exit(1)
    else:
        print("\n⏭️  KPI 테이블 생성 건너뛰기 (--skip-kpi)")

    # ============================================================
    # 4) Delta 리포트 생성 (2개: 직전 baseline + Global baseline)
    # ============================================================
    if not args.skip_delta:
        print("\n[4/7] Δ 리포트 생성 중...")

        # Baseline KPI 확인 및 생성
        baseline_kpi_csv = (
            base_dir / "reports" / "kpi" / f"kpi_table__{baseline_tag}.csv"
        )
        if baseline_tag and not baseline_kpi_csv.exists():
            print(f"\n[WARNING] Baseline KPI가 없습니다: {baseline_kpi_csv}")
            print("   Baseline KPI를 먼저 생성합니다...")

            baseline_kpi_cmd = [
                sys.executable,
                str(base_dir / "src" / "tools" / "export_kpi_table.py"),
                "--config",
                args.config,
                "--tag",
                baseline_tag,
            ]

            returncode, _, _ = run_command(
                baseline_kpi_cmd,
                cwd=base_dir,
                description="Baseline KPI 생성",
                check_returncode=False,
            )

        # Global baseline KPI 확인 및 생성
        global_baseline_kpi_csv = None
        if global_baseline_tag and global_baseline_tag != baseline_tag:
            global_baseline_kpi_csv = (
                base_dir / "reports" / "kpi" / f"kpi_table__{global_baseline_tag}.csv"
            )
            if not global_baseline_kpi_csv.exists():
                print(
                    f"\n[WARNING] Global Baseline KPI가 없습니다: {global_baseline_kpi_csv}"
                )
                print("   Global Baseline KPI를 먼저 생성합니다...")

                global_baseline_kpi_cmd = [
                    sys.executable,
                    str(base_dir / "src" / "tools" / "export_kpi_table.py"),
                    "--config",
                    args.config,
                    "--tag",
                    global_baseline_tag,
                ]

                returncode, _, _ = run_command(
                    global_baseline_kpi_cmd,
                    cwd=base_dir,
                    description="Global Baseline KPI 생성",
                    check_returncode=False,
                )

        # Delta 리포트 생성 (직전 baseline)
        if baseline_tag and baseline_kpi_csv.exists():
            delta_cmd = [
                sys.executable,
                str(base_dir / "src" / "tools" / "export_delta_report.py"),
                "--baseline-tag",
                baseline_tag,
                "--run-tag",
                run_tag,
            ]

            returncode, stdout, stderr = run_command(
                delta_cmd,
                cwd=base_dir,
                description=f"Δ 리포트 생성 (직전 baseline: {baseline_tag})",
                check_returncode=True,
            )

            if returncode == 0:
                delta_csv = (
                    base_dir
                    / "reports"
                    / "delta"
                    / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
                )
                verify_artifact_exists(delta_csv, "Delta CSV (직전 baseline)")

        # Delta 리포트 생성 (Global baseline)
        if global_baseline_tag and global_baseline_tag != baseline_tag:
            if global_baseline_kpi_csv and global_baseline_kpi_csv.exists():
                global_delta_cmd = [
                    sys.executable,
                    str(base_dir / "src" / "tools" / "export_delta_report.py"),
                    "--baseline-tag",
                    global_baseline_tag,
                    "--run-tag",
                    run_tag,
                ]

                returncode, stdout, stderr = run_command(
                    global_delta_cmd,
                    cwd=base_dir,
                    description=f"Δ 리포트 생성 (Global baseline: {global_baseline_tag})",
                    check_returncode=False,  # 실패해도 계속 진행
                )

                if returncode == 0:
                    global_delta_csv = (
                        base_dir
                        / "reports"
                        / "delta"
                        / f"delta_kpi__{global_baseline_tag}__vs__{run_tag}.csv"
                    )
                    verify_artifact_exists(
                        global_delta_csv, "Delta CSV (Global baseline)"
                    )
    else:
        print("\n⏭️  Δ 리포트 생성 건너뛰기 (--skip-delta)")

    # ============================================================
    # 5) Stage Check 리포트 생성
    # ============================================================
    if not args.skip_check:
        print("\n[5/7] Stage 체크리포트 생성 중...")

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
            baseline_tag or "",
        ]

        returncode, stdout, stderr = run_command(
            check_cmd,
            cwd=base_dir,
            description="Stage 체크리포트 생성",
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
            print("[WARNING] 체크리포트 생성 실패 (계속 진행)")
    else:
        print("\n⏭️  체크리포트 생성 건너뛰기 (--skip-check)")

    # ============================================================
    # 6) History Manifest 업데이트
    # ============================================================
    if not args.skip_history:
        print("\n[6/7] History Manifest 업데이트 중...")

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
            baseline_tag or "",
        ]

        returncode, stdout, stderr = run_command(
            history_cmd,
            cwd=base_dir,
            description="History Manifest 업데이트",
            check_returncode=False,  # 실패해도 계속 진행
        )

        if returncode == 0:
            history_manifest = (
                base_dir / "reports" / "history" / "history_manifest.parquet"
            )
            verify_artifact_exists(history_manifest, "History Manifest")
        else:
            print("[WARNING] History Manifest 업데이트 실패 (계속 진행)")
    else:
        print("\n⏭️  History Manifest 업데이트 건너뛰기 (--skip-history)")

    # ============================================================
    # 7) 최종 요약 출력
    # ============================================================
    print("\n" + "=" * 60)
    print("[7/7] 최종 요약")
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
    if baseline_tag:
        delta_csv = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
        )
        delta_md = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_report__{baseline_tag}__vs__{run_tag}.md"
        )
        if delta_csv.exists():
            outputs.append(("Delta CSV (직전)", str(delta_csv)))
        if delta_md.exists():
            outputs.append(("Delta MD (직전)", str(delta_md)))

    if global_baseline_tag and global_baseline_tag != baseline_tag:
        global_delta_csv = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_kpi__{global_baseline_tag}__vs__{run_tag}.csv"
        )
        global_delta_md = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_report__{global_baseline_tag}__vs__{run_tag}.md"
        )
        if global_delta_csv.exists():
            outputs.append(("Delta CSV (Global)", str(global_delta_csv)))
        if global_delta_md.exists():
            outputs.append(("Delta MD (Global)", str(global_delta_md)))

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
    print("[OK] 모든 단계 완료")
    print("=" * 60)
    print(f"\n[PASS] Stage {args.stage} 완료")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag (직전): {baseline_tag or 'N/A'}")
    print(f"Global Baseline Tag: {global_baseline_tag or 'N/A'}")
    print(f"생성된 파일 수: {len(outputs)}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
