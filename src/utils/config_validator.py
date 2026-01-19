"""
설정 검증기 모듈

설정 파일의 유효성을 검증하는 기능을 제공합니다.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re
from datetime import datetime


class ConfigValidationError(Exception):
    """설정 검증 오류"""
    pass


class ConfigValidator:
    """설정 검증기 클래스"""

    def __init__(self):
        self.errors: List[str] = []

    def validate_required_keys(self, config: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        필수 키 존재 여부를 검증합니다.

        Args:
            config: 검증할 설정
            required_keys: 필수 키 리스트 (점 표기법 지원, 예: "l7.top_k")

        Returns:
            모든 필수 키가 존재하면 True
        """
        for key_path in required_keys:
            if not self._check_key_exists(config, key_path):
                self.errors.append(f"필수 키 누락: {key_path}")
                return False
        return True

    def validate_types(self, config: Dict[str, Any], type_specs: Dict[str, type]) -> bool:
        """
        키별 타입을 검증합니다.

        Args:
            config: 검증할 설정
            type_specs: 키 경로 -> 타입 매핑 (예: {"l7.top_k": int})

        Returns:
            모든 타입이 일치하면 True
        """
        for key_path, expected_type in type_specs.items():
            value = self._get_nested_value(config, key_path)
            if value is not None and not isinstance(value, expected_type):
                self.errors.append(
                    f"타입 불일치: {key_path} (예상: {expected_type.__name__}, 실제: {type(value).__name__})"
                )
                return False
        return True

    def validate_ranges(self, config: Dict[str, Any], range_specs: Dict[str, Tuple[Optional[float], Optional[float]]]) -> bool:
        """
        수치 범위를 검증합니다.

        Args:
            config: 검증할 설정
            range_specs: 키 경로 -> (최소값, 최대값) 매핑

        Returns:
            모든 범위가 유효하면 True
        """
        for key_path, (min_val, max_val) in range_specs.items():
            value = self._get_nested_value(config, key_path)
            if value is not None:
                if not isinstance(value, (int, float)):
                    self.errors.append(f"범위 검증 대상이 숫자가 아님: {key_path}")
                    return False

                if min_val is not None and value < min_val:
                    self.errors.append(f"값이 최소값보다 작음: {key_path} = {value} < {min_val}")
                    return False

                if max_val is not None and value > max_val:
                    self.errors.append(f"값이 최대값보다 큼: {key_path} = {value} > {max_val}")
                    return False
        return True

    def validate_dates(self, config: Dict[str, Any], date_keys: List[str]) -> bool:
        """
        날짜 형식의 유효성을 검증합니다.

        Args:
            config: 검증할 설정
            date_keys: 날짜 키 경로 리스트

        Returns:
            모든 날짜가 유효하면 True
        """
        for key_path in date_keys:
            value = self._get_nested_value(config, key_path)
            if value is not None:
                if not isinstance(value, str):
                    self.errors.append(f"날짜 형식이 문자열이 아님: {key_path}")
                    return False

                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    self.errors.append(f"잘못된 날짜 형식: {key_path} = {value} (YYYY-MM-DD 필요)")
                    return False
        return True

    def validate_backtest_params(self, config: Dict[str, Any]) -> bool:
        """
        백테스트 파라미터의 특수 검증을 수행합니다.

        Returns:
            검증 통과 시 True
        """
        # holding_days 검증
        holding_days = self._get_nested_value(config, "l7.holding_days")
        if holding_days is not None:
            if not isinstance(holding_days, int) or holding_days <= 0:
                self.errors.append(f"holding_days는 양의 정수여야 함: {holding_days}")
                return False

        # top_k 검증
        top_k = self._get_nested_value(config, "l7.top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                self.errors.append(f"top_k는 양의 정수여야 함: {top_k}")
                return False

        # cost_bps 범위 검증 (0-100 BPS)
        cost_bps = self._get_nested_value(config, "l7.cost_bps")
        if cost_bps is not None:
            if not isinstance(cost_bps, (int, float)) or cost_bps < 0 or cost_bps > 100:
                self.errors.append(f"cost_bps는 0-100 범위여야 함: {cost_bps}")
                return False

        # rebalance_interval 검증
        rebalance_interval = self._get_nested_value(config, "l7.rebalance_interval")
        if rebalance_interval is not None:
            if not isinstance(rebalance_interval, int) or rebalance_interval <= 0:
                self.errors.append(f"rebalance_interval은 양의 정수여야 함: {rebalance_interval}")
                return False

        return True

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        전체 설정에 대한 종합 검증을 수행합니다.

        Returns:
            검증 통과 시 True
        """
        self.errors = []  # 에러 초기화

        # 필수 키 검증
        required_keys = [
            "params.start_date",
            "params.end_date",
            "l7.top_k",
            "l7.holding_days",
            "l7.cost_bps"
        ]

        if not self.validate_required_keys(config, required_keys):
            return False

        # 타입 검증
        type_specs = {
            "l7.top_k": int,
            "l7.holding_days": int,
            "l7.rebalance_interval": int,
            "l7.cost_bps": (int, float),
            "l7.target_volatility": (int, float)
        }

        if not self.validate_types(config, type_specs):
            return False

        # 범위 검증
        range_specs = {
            "l7.top_k": (1, 50),
            "l7.holding_days": (1, 365),
            "l7.cost_bps": (0, 100),
            "l7.rebalance_interval": (1, 100),
            "l7.target_volatility": (0.01, 1.0)
        }

        if not self.validate_ranges(config, range_specs):
            return False

        # 날짜 검증
        date_keys = ["params.start_date", "params.end_date"]
        if not self.validate_dates(config, date_keys):
            return False

        # 백테스트 특수 검증
        if not self.validate_backtest_params(config):
            return False

        return True

    def get_errors(self) -> List[str]:
        """누적된 검증 에러들을 반환합니다."""
        return self.errors.copy()

    def get_error_summary(self) -> str:
        """에러 요약을 문자열로 반환합니다."""
        if not self.errors:
            return "검증 성공"

        summary = f"검증 실패 ({len(self.errors)}개 에러):\n"
        for i, error in enumerate(self.errors, 1):
            summary += f"  {i}. {error}\n"

        return summary

    def _check_key_exists(self, config: Dict[str, Any], key_path: str) -> bool:
        """점 표기법 키 경로가 존재하는지 확인합니다."""
        keys = key_path.split('.')
        current = config

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]

        return True

    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """점 표기법으로 중첩된 값을 가져옵니다."""
        keys = key_path.split('.')
        current = config

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        return current


def validate_config_file(config_path: str) -> Tuple[bool, str]:
    """
    설정 파일을 검증합니다.

    Args:
        config_path: 설정 파일 경로

    Returns:
        (성공 여부, 에러 메시지 또는 성공 메시지)
    """
    try:
        from .config import load_yaml_with_defaults
        config = load_yaml_with_defaults(config_path)

        validator = ConfigValidator()
        if validator.validate_config(config):
            return True, "설정 검증 성공"
        else:
            return False, validator.get_error_summary()

    except Exception as e:
        return False, f"설정 파일 로드 실패: {str(e)}"