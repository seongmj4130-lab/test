"""
중앙 로깅 설정 모듈

이 모듈은 프로젝트 전체의 로깅을 표준화하고 관측성을 제공합니다.
콘솔과 파일 출력이 일관되게 설정되며, 실행 메타데이터를 추적합니다.
"""

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any


class ExecutionSummary:
    """실행 요약을 관리하는 클래스"""

    def __init__(self, track_name: str, config_path: str):
        self.track_name = track_name
        self.config_path = config_path
        self.start_time = time.time()
        self.step_times: dict[str, float] = {}
        self.parameters: dict[str, Any] = {}
        self.outputs: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}

    def add_parameter(self, key: str, value: Any) -> None:
        """실행 파라미터 추가"""
        self.parameters[key] = value

    def add_output(self, key: str, path: str) -> None:
        """산출물 경로 추가"""
        self.outputs[key] = path

    def add_metadata(self, key: str, value: Any) -> None:
        """메타데이터 추가"""
        self.metadata[key] = value

    def start_step(self, step_name: str) -> None:
        """단계 시작 시간 기록"""
        self.step_times[step_name] = time.time()

    def end_step(self, step_name: str) -> float:
        """단계 종료 시간 기록 및 소요시간 반환"""
        if step_name in self.step_times:
            elapsed = time.time() - self.step_times[step_name]
            self.step_times[step_name] = elapsed
            return elapsed
        return 0.0

    def save_summary(self, output_dir: Path) -> str:
        """실행 요약을 JSON 파일로 저장"""
        total_elapsed = time.time() - self.start_time

        summary = {
            "track": self.track_name,
            "config_path": self.config_path,
            "execution_time": {
                "start": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)
                ),
                "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "total_seconds": round(total_elapsed, 2),
                "total_minutes": round(total_elapsed / 60, 2),
            },
            "step_times": {k: round(v, 2) for k, v in self.step_times.items()},
            "parameters": self.parameters,
            "outputs": self.outputs,
            "metadata": self.metadata,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = (
            output_dir / f"run_summary_{self.track_name}_{int(self.start_time)}.json"
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return str(summary_path)


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """
    중앙 로깅 설정을 초기화합니다.

    Args:
        log_level: 기본 로그 레벨 (INFO, DEBUG, WARNING, ERROR)
        log_file: 로그 파일 경로 (None이면 파일 로깅 비활성화)
        console_level: 콘솔 출력 레벨
        file_level: 파일 출력 레벨
    """
    # 기존 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 로그 레벨 설정
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    console_level_num = level_map.get(console_level.upper(), logging.INFO)
    file_level_num = level_map.get(file_level.upper(), logging.DEBUG)
    root_level = level_map.get(log_level.upper(), logging.INFO)

    root_logger.setLevel(root_level)

    # 포맷터 설정: 시간/레벨/모듈/메시지
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)20s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level_num)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 (선택적)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(file_level_num)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    표준화된 로거를 반환합니다.

    Args:
        name: 로거 이름 (일반적으로 __name__)

    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    return logging.getLogger(name)
