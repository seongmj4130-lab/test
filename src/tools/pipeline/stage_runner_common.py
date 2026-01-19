# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/pipeline/stage_runner_common.py
"""
[코드 매니저] Stage 실행 공통 유틸리티
- L2 해시 검증
- Baseline 태그 결정
- 저장 경로 검증
- 공통 실행 절차
"""
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def get_file_hash(filepath: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_l2_reuse(
    base_dir: Path, log_file: Optional[Path] = None
) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    L2 파일 재사용 검증 (해시 확인)

    Returns:
        (is_valid, message, hash_before, hash_after)
    """
    l2_file = base_dir / "data" / "interim" / "fundamentals_annual.parquet"

    if not l2_file.exists():
        return False, "L2 파일이 존재하지 않습니다", None, None

    current_hash = get_file_hash(l2_file)
    hash_before = None
    hash_after = current_hash[:16] + "..."  # 짧은 버전

    # 로그 파일에서 실행 전 해시 읽기 시도
    if log_file and log_file.exists():
        try:
            log_content = log_file.read_text(encoding="utf-8", errors="ignore")
            import re

            match = re.search(
                r"L2.*해시[:\s]+([0-9a-f]{16})", log_content, re.IGNORECASE
            )
            if match:
                hash_before = match.group(1) + "..."
        except Exception:
            pass

    file_size = l2_file.stat().st_size
    mtime = datetime.fromtimestamp(l2_file.stat().st_mtime)

    hash_info = f"해시: {hash_after}"
    if hash_before:
        hash_info = f"실행 전: {hash_before}, 실행 후: {hash_after}"

    return (
        True,
        f"L2 파일 재사용 확인: 크기={file_size:,} bytes, 수정시간={mtime.strftime('%Y-%m-%d %H:%M:%S')}, {hash_info}",
        hash_before,
        hash_after,
    )


def get_baseline_tag(config_path: Path, stage: int) -> str:
    """
    Baseline 태그 결정
    - Stage0~6: pipeline_baseline_tag
    - Stage7: ranking_baseline 생성 단계 (baseline_tag_used는 pipeline_baseline_tag)
    - Stage8+: ranking_baseline_tag (Stage7이 없으면 에러)
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    baseline_cfg = cfg.get("baseline", {})

    # Stage0~6: pipeline_baseline_tag
    if stage <= 6:
        return baseline_cfg.get(
            "pipeline_baseline_tag", "baseline_prerefresh_20251219_143636"
        )

    # Stage7: ranking_baseline 생성 단계 (baseline_tag_used는 pipeline_baseline_tag)
    elif stage == 7:
        return baseline_cfg.get(
            "pipeline_baseline_tag", "baseline_prerefresh_20251219_143636"
        )

    # Stage8+: ranking_baseline_tag (Stage7이 없으면 에러)
    else:
        ranking_baseline = baseline_cfg.get("ranking_baseline_tag")
        if not ranking_baseline:
            print(
                "ERROR: Stage7이 완료되지 않았습니다. ranking_baseline_tag가 설정되지 않았습니다.",
                file=sys.stderr,
            )
            sys.exit(1)
        return ranking_baseline


def verify_base_dir(base_dir: Path) -> tuple[bool, str]:
    """
    base_dir이 올바른 경로인지 검증 (바탕 화면 경로 방지)

    Returns:
        (is_valid, message)
    """
    expected_base = Path(r"C:\Users\seong\OneDrive\Desktop\bootcamp\03_code").resolve()
    actual_base = base_dir.resolve()

    # 바탕 화면 경로 체크
    if (
        "바탕 화면" in str(actual_base)
        or "Desktop" in str(actual_base)
        and "bootcamp" not in str(actual_base)
    ):
        return False, f"저장 경로가 바탕 화면으로 설정되어 있습니다: {actual_base}"

    if actual_base != expected_base:
        return False, f"base_dir 불일치: 예상={expected_base}, 실제={actual_base}"

    return True, f"base_dir 검증 통과: {actual_base}"


def run_command(
    cmd: list[str], cwd: Path, description: str, log_file: Optional[Path] = None
) -> int:
    """명령어 실행"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{description}]\n"
            )
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'='*60}\n\n")

    result = subprocess.run(cmd, cwd=str(cwd), encoding="utf-8", errors="replace")

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Exit Code: {result.returncode}\n")
            if result.stdout:
                f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")

    if result.returncode != 0:
        print(f"\n[FAIL] [{description}] Failed with exit code {result.returncode}")
        return result.returncode

    print(f"\n[OK] [{description}] Completed")
    return 0


def generate_run_tag(stage_name: str) -> str:
    """run_tag 생성: stage명 + 타임스탬프"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage_name}_{timestamp}"


def get_stage_track(stage: int) -> str:
    """Stage 번호로 track 결정"""
    if stage <= 6:
        return "pipeline"
    elif stage == 7:
        return "ranking"  # ranking baseline 생성
    else:
        return "ranking"


def print_success_summary(
    run_tag: str,
    baseline_tag_used: str,
    ranking_baseline_tag: Optional[str],
    output_files: list[tuple[str, Path]],
):
    """성공 출력 (필수)"""
    print("\n" + "=" * 60)
    print("[PASS] Stage 완료")
    print("=" * 60)
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used: {baseline_tag_used}")
    if ranking_baseline_tag:
        print(f"Ranking Baseline Tag: {ranking_baseline_tag}")
    print(f"\n생성된 주요 파일 ({len(output_files)}개):")
    for desc, path in output_files:
        if path.exists():
            size = path.stat().st_size
            print(f"  - {desc}: {path} ({size:,} bytes)")
        else:
            print(f"  - {desc}: {path} [MISSING]")
    print("=" * 60 + "\n")
