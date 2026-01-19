#!/usr/bin/env python3
"""
설정 파일 배치 검증 CLI

configs/ 디렉토리의 모든 YAML 설정 파일을 검증합니다.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_validator import validate_config_file


def collect_config_files(configs_dir: Path) -> List[Path]:
    """설정 디렉토리에서 모든 YAML 파일을 수집합니다."""
    if not configs_dir.exists():
        print(f"설정 디렉토리가 존재하지 않습니다: {configs_dir}")
        return []

    yaml_files = list(configs_dir.glob("*.yaml"))
    return sorted(yaml_files)


def validate_configs_batch(configs_dir: Path, verbose: bool = False) -> Tuple[int, int, Dict[str, List[str]]]:
    """
    설정 파일들을 배치로 검증합니다.

    Returns:
        (총 파일 수, 성공 수, 실패 파일별 에러 딕셔너리)
    """
    config_files = collect_config_files(configs_dir)
    if not config_files:
        print("검증할 설정 파일이 없습니다.")
        return 0, 0, {}

    total_count = len(config_files)
    success_count = 0
    failures = {}

    print(f"설정 파일 검증 시작... (총 {total_count}개 파일)")
    print("=" * 50)

    for i, config_file in enumerate(config_files, 1):
        file_name = config_file.name
        print(f"[{i}/{total_count}] {file_name} 검증 중...")

        success, message = validate_config_file(str(config_file))

        if success:
            success_count += 1
            if verbose:
                print(f"  ✓ 성공")
        else:
            failures[file_name] = message.split('\n') if '\n' in message else [message]
            print(f"  ✗ 실패")
            if verbose:
                for line in failures[file_name]:
                    print(f"    {line}")

    return total_count, success_count, failures


def print_summary(total: int, success: int, failures: Dict[str, List[str]]) -> None:
    """검증 결과를 요약해서 출력합니다."""
    failed_count = len(failures)

    print("\n" + "=" * 50)
    print("검증 결과 요약")
    print("=" * 50)
    print(f"총 파일 수: {total}")
    print(f"성공: {success}")
    print(f"실패: {failed_count}")
    print(".1f")

    if failures:
        print(f"\n실패한 파일들 ({failed_count}개):")
        for file_name, errors in failures.items():
            print(f"\n📁 {file_name}:")
            for error in errors:
                print(f"  • {error}")


def get_top_failures(failures: Dict[str, List[str]], top_n: int = 5) -> List[Tuple[str, int]]:
    """가장 많이 발생한 에러 유형을 추출합니다."""
    from collections import Counter

    all_errors = []
    for error_list in failures.values():
        all_errors.extend(error_list)

    # 에러 메시지를 정규화하여 카운트
    normalized_errors = []
    for error in all_errors:
        # 구체적인 값 제거하고 패턴 추출
        if "필수 키 누락:" in error:
            normalized_errors.append("필수 키 누락")
        elif "타입 불일치:" in error:
            normalized_errors.append("타입 불일치")
        elif "범위 검증 대상이 숫자가 아님:" in error:
            normalized_errors.append("범위 검증 대상 타입 오류")
        elif "값이 최소값보다 작음:" in error or "값이 최대값보다 큼:" in error:
            normalized_errors.append("범위 초과")
        elif "잘못된 날짜 형식:" in error:
            normalized_errors.append("날짜 형식 오류")
        elif "holding_days는 양의 정수여야 함:" in error:
            normalized_errors.append("holding_days 검증 실패")
        elif "top_k는 양의 정수여야 함:" in error:
            normalized_errors.append("top_k 검증 실패")
        elif "cost_bps는" in error:
            normalized_errors.append("cost_bps 범위 오류")
        elif "rebalance_interval은" in error:
            normalized_errors.append("rebalance_interval 검증 실패")
        else:
            normalized_errors.append(error)

    error_counts = Counter(normalized_errors)
    return error_counts.most_common(top_n)


def main():
    parser = argparse.ArgumentParser(description="설정 파일 배치 검증 CLI")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="설정 파일 디렉토리 경로 (기본값: configs)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세한 검증 결과 출력"
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=5,
        help="상위 N개 에러 유형 표시 (기본값: 5)"
    )

    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)

    # 검증 실행
    total, success, failures = validate_configs_batch(configs_dir, args.verbose)

    # 결과 출력
    print_summary(total, success, failures)

    # 상위 에러 유형 출력
    if failures:
        print(f"\n상위 {args.top_errors}개 에러 유형:")
        top_errors = get_top_failures(failures, args.top_errors)
        for error_type, count in top_errors:
            print(f"  {count}회: {error_type}")

    # 종료 코드
    sys.exit(0 if len(failures) == 0 else 1)


if __name__ == "__main__":
    main()