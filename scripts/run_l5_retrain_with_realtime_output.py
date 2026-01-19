"""
L5 모델 재학습 스크립트 (단기/장기, Dev/Holdout 모두 포함)

Ridge alpha 16.0으로 모델 재학습
- 단기 랭킹 (horizon=20): Dev + Holdout
- 장기 랭킹 (horizon=120): Dev + Holdout

실행 결과를 실시간으로 터미널에 출력
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로
project_root = Path(__file__).resolve().parent.parent


def run_with_realtime_output(cmd: list, cwd: Path, description: str):
    """명령어를 실행하고 실시간으로 출력"""
    print("\n" + "=" * 80)
    print(f"[{description}]")
    print("=" * 80)
    print(f"작업 디렉토리: {cwd}")
    print(f"명령어: {' '.join(cmd)}")
    print("=" * 80 + "\n")

    # 프로세스 시작
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,  # 라인 버퍼링
        encoding="utf-8",
        errors="replace",
    )

    # 실시간 출력
    try:
        for line in process.stdout:
            print(line, end="", flush=True)

        # 프로세스 완료 대기
        return_code = process.wait()

        if return_code != 0:
            print(f"\n❌ [{description}] 실패 (종료 코드: {return_code})")
            return False
        else:
            print(f"\n✅ [{description}] 완료")
            return True

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        process.terminate()
        process.wait()
        return False


def main():
    """메인 함수"""
    print("=" * 80)
    print("L5 모델 재학습 (Ridge Alpha 16.0)")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"프로젝트 루트: {project_root}")
    print("=" * 80)

    # config.yaml 확인
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        print(f"❌ Config 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    # Ridge alpha 확인
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ridge_alpha = cfg.get("l5", {}).get("ridge_alpha", 8.0)
    print(f"\n현재 Ridge Alpha: {ridge_alpha}")
    print("목표: 16.0 (과적합 방지 강화)")

    if ridge_alpha != 16.0:
        print(f"⚠️ 경고: Ridge Alpha가 16.0이 아닙니다. 현재 값: {ridge_alpha}")
        response = input("계속 진행하시겠습니까? (y/n): ")
        if response.lower() != "y":
            print("중단됨")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("학습 범위:")
    print("  - 단기 랭킹 (horizon=20): Dev + Holdout")
    print("  - 장기 랭킹 (horizon=120): Dev + Holdout")
    print("=" * 80 + "\n")

    # L5 모델 학습 실행
    # run_stage_pipeline.py를 사용하면 자동으로 단기/장기, Dev/Holdout 모두 처리됨
    cmd = [
        sys.executable,
        str(project_root / "src" / "tools" / "run_stage_pipeline.py"),
        "--stage",
        "5",
        "--config",
        "configs/config.yaml",
    ]

    success = run_with_realtime_output(
        cmd, project_root, "L5 모델 재학습 (단기/장기, Dev/Holdout 모두 포함)"
    )

    if not success:
        print("\n❌ 모델 재학습 실패")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ L5 모델 재학습 완료")
    print("=" * 80)
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n다음 단계:")
    print("  1. Dev/Holdout 구간 성과 재평가")
    print("  2. 과적합 위험도 재확인")
    print("  3. 필요시 Ridge alpha 추가 조정")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
