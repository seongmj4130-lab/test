"""
데이터 수집 모듈 (Data Collection Module)

이 모듈은 L0~L4 데이터 수집 단계를 독립적으로 실행할 수 있도록 제공합니다.
기존 데이터는 그대로 유지되며, 새로운 데이터 수집만 수행합니다.

사용 예시:
    from src.data_collection import collect_all_data

    # 전체 데이터 수집
    artifacts = collect_all_data(config_path="configs/config.yaml")

    # 단계별 수집
    from src.data_collection import collect_universe, collect_ohlcv, collect_panel

    universe = collect_universe(start_date="2016-01-01", end_date="2024-12-31")
    ohlcv = collect_ohlcv(universe, start_date="2016-01-01", end_date="2024-12-31")
    panel = collect_panel(ohlcv, fundamentals_annual=None)
"""

from src.data_collection.collectors import (
    collect_all_data,
    collect_dataset,
    collect_fundamentals,
    collect_ohlcv,
    collect_panel,
    collect_universe,
)
from src.data_collection.pipeline import (
    DataCollectionPipeline,
    run_data_collection_pipeline,
)
from src.data_collection.ui_interface import (
    check_data_availability,
    collect_data_for_ui,
    get_dataset,
    get_ohlcv,
    get_panel,
    get_universe,
)

__all__ = [
    # Collectors (1단계)
    "collect_universe",
    "collect_ohlcv",
    "collect_fundamentals",
    "collect_panel",
    "collect_dataset",
    "collect_all_data",
    # Pipeline (3단계)
    "DataCollectionPipeline",
    "run_data_collection_pipeline",
    # UI Interface (2단계)
    "get_universe",
    "get_ohlcv",
    "get_panel",
    "get_dataset",
    "check_data_availability",
    "collect_data_for_ui",
]
