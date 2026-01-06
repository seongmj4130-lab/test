# -*- coding: utf-8 -*-
"""
Track A: 랭킹 엔진 서비스 모듈

UI에서 import 가능한 형태로 랭킹 생성 함수를 제공합니다.

[리팩토링 2단계] 함수/모듈화 - UI에서 import 가능한 형태
"""
from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import logging

from src.utils.config import load_config
from src.tracks.shared.data_pipeline import prepare_common_data
from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
    run_L8_short_rank_engine,
    run_L8_long_rank_engine,
)

logger = logging.getLogger(__name__)


def generate_rankings(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
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
    artifacts = prepare_common_data(config_path=config_path, force_rebuild=force_rebuild)
    
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


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    force = "--force" in sys.argv
    rankings = generate_rankings(force_rebuild=force)
    print(f"\n✅ 완료: 단기 랭킹 {len(rankings['ranking_short_daily']):,}행, 장기 랭킹 {len(rankings['ranking_long_daily']):,}행")

