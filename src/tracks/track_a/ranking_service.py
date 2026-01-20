"""
Track A: 랭킹 엔진 서비스 모듈

UI에서 import 가능한 형태로 랭킹 생성 함수를 제공합니다.

[리팩토링 2단계] 함수/모듈화 - UI에서 import 가능한 형태
"""

import logging
from pathlib import Path

import pandas as pd

from src.tracks.shared.data_pipeline import prepare_common_data
from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
    run_L8_long_rank_engine,
    run_L8_short_rank_engine,
)
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def generate_rankings(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    랭킹을 생성하는 함수 (Track A 핵심 기능).

    [리팩토링 2단계] UI에서 import 가능한 형태로 모듈화

    Args:
        config_path: 설정 파일 경로
        force_rebuild: True면 캐시 무시하고 재계산

    Returns:
        dict: 랭킹 데이터
        {
            "ranking_short_daily": DataFrame,
            "ranking_long_daily": DataFrame,
        }

    Example:
        >>> from src.tracks.track_a.ranking_service import generate_rankings
        >>> rankings = generate_rankings()
        >>> short_ranking = rankings["ranking_short_daily"]
    """
    logger.info("랭킹 생성 시작 (Track A)")

    # 설정 로드
    cfg = load_config(config_path)

    # 공통 데이터 준비
    artifacts = prepare_common_data(
        config_path=config_path, force_rebuild=force_rebuild
    )

    # L8: 랭킹 엔진 실행
    logger.info("[L8] 랭킹 엔진 실행")

    # 단기 랭킹 생성
    outputs_short, warns_short = run_L8_short_rank_engine(
        cfg=cfg,
        artifacts=artifacts,
        force=force_rebuild,
    )

    # 장기 랭킹 생성
    outputs_long, warns_long = run_L8_long_rank_engine(
        cfg=cfg,
        artifacts=artifacts,
        force=force_rebuild,
    )

    logger.info("✅ 랭킹 생성 완료")

    return {
        "ranking_short_daily": outputs_short["ranking_short_daily"],
        "ranking_long_daily": outputs_long["ranking_long_daily"],
    }


def inspect_holdout_day_rankings(
    as_of: str,
    *,
    topk: int = 10,
    horizon: str = "both",
    config_path: str = "configs/config.yaml",
) -> dict:
    """
    [개선안 36번] Track A Holdout 기간 중 특정 날짜(as_of)에 대해
    TopK 랭킹 + 팩터셋(그룹) 기여도 Top3를 반환하는 서비스 함수 (UI/분석용).

    Args:
        as_of: 기준 날짜 (예: "2024-12-30")
        topk: TopK 종목 수
        horizon: "short" | "long" | "both"
        config_path: 설정 파일 경로

    Returns:
        dict:
        {
          "meta": {"date": "...", "holdout_start": "...", "holdout_end": "..."},
          "short": DataFrame | None,
          "long": DataFrame | None,
        }
    """
    from src.tracks.track_a.stages.ranking.holdout_day_inspector import (
        inspect_holdout_day,
    )
    from src.utils.io import load_artifact

    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["data_interim"])

    # 캐시 기반 로드 (Track A 결과가 있어야 함)
    dataset_daily = load_artifact(interim_dir / "dataset_daily")
    ranking_short_daily = load_artifact(interim_dir / "ranking_short_daily")
    ranking_long_daily = load_artifact(interim_dir / "ranking_long_daily")

    result = inspect_holdout_day(
        cfg=cfg,
        date=as_of,
        dataset_daily=dataset_daily,
        ranking_short_daily=ranking_short_daily,
        ranking_long_daily=ranking_long_daily,
        topk=topk,
        horizon=horizon,  # type: ignore[arg-type]
    )

    return {
        "meta": {
            "date": str(result.date.date()),
            "holdout_start": str(result.holdout_start.date()),
            "holdout_end": str(result.holdout_end.date()),
        },
        "short": result.short,
        "long": result.long,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    force = "--force" in sys.argv
    rankings = generate_rankings(force_rebuild=force)
    print(
        f"\n✅ 완료: 단기 랭킹 {len(rankings['ranking_short_daily']):,}행, 장기 랭킹 {len(rankings['ranking_long_daily']):,}행"
    )
