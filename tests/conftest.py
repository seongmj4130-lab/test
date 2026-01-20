"""
공용 pytest fixtures 및 헬퍼 함수들
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_path() -> Generator[Path, None, None]:
    """임시 디렉토리 fixture - 테스트 후 자동 정리"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_data_empty() -> pd.DataFrame:
    """빈 DataFrame 생성"""
    return pd.DataFrame()


@pytest.fixture
def sample_data_basic() -> pd.DataFrame:
    """기본적인 샘플 DataFrame 생성"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "ticker": ["AAPL"] * 50 + ["GOOGL"] * 50,
            "price": np.random.uniform(100, 200, 100),
            "volume": np.random.randint(1000, 10000, 100),
            "returns": np.random.normal(0, 0.02, 100),
        }
    )


@pytest.fixture
def sample_data_with_missing(sample_data_basic) -> pd.DataFrame:
    """결측치가 포함된 DataFrame 생성"""
    df = sample_data_basic.copy()
    # 일부 행에 결측치 추가
    df.loc[10:15, "price"] = np.nan
    df.loc[20:25, "volume"] = np.nan
    df.loc[30:35, "returns"] = np.nan
    return df


@pytest.fixture
def sample_data_wrong_types() -> pd.DataFrame:
    """잘못된 타입의 데이터가 포함된 DataFrame 생성"""
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "not_a_date", "2023-01-03"],
            "price": ["not_a_number", 100.5, 101.0],
            "volume": [1000, "invalid", 2000],
        }
    )


@pytest.fixture
def sample_config_minimal(tmp_path) -> dict:
    """최소 설정 구성 생성 - 실제 config 의존 최소화"""
    base_dir = tmp_path / "project"
    base_dir.mkdir(exist_ok=True)

    return {
        "paths": {
            "base_dir": str(base_dir),
            "data_dir": str(base_dir / "data"),
            "artifacts_dir": str(base_dir / "artifacts"),
            "configs_dir": str(base_dir / "configs"),
        },
        "modeling": {"random_state": 42, "test_size": 0.2},
        "backtest": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 1000000,
        },
    }


@pytest.fixture
def sample_portfolio_weights() -> dict[str, float]:
    """샘플 포트폴리오 가중치 생성"""
    return {"AAPL": 0.3, "GOOGL": 0.25, "MSFT": 0.2, "TSLA": 0.15, "AMZN": 0.1}


@pytest.fixture
def sample_rankings() -> pd.DataFrame:
    """샘플 랭킹 데이터 생성"""
    np.random.seed(42)
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    return (
        pd.DataFrame(
            {
                "ticker": tickers,
                "score": np.random.uniform(0.1, 0.9, len(tickers)),
                "rank": range(1, len(tickers) + 1),
                "sector": ["Tech"] * len(tickers),
            }
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


@pytest.fixture(autouse=True)
def set_test_env():
    """테스트 환경 설정"""
    # 테스트용 환경 변수 설정
    original_env = os.environ.copy()

    # 테스트용 임시 경로 설정
    os.environ["BASE_DIR"] = str(Path(__file__).parent.parent)

    yield

    # 환경 변수 복원
    os.environ.clear()
    os.environ.update(original_env)
