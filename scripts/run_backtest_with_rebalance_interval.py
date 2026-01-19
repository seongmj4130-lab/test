# -*- coding: utf-8 -*-
"""
rebalance_interval을 적용하여 백테스트 재실행
- BT20 모델: rebalance_interval=20
- BT120 모델: rebalance_interval=120
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from src.pipeline.track_b_pipeline import run_track_b_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """4개 모델을 rebalance_interval 설정으로 재실행"""

    models = [
        ("bt20_short", 20),
        ("bt20_ens", 20),
        ("bt120_long", 120),
        ("bt120_ens", 120),
    ]

    logger.info("=" * 80)
    logger.info("rebalance_interval 적용 백테스트 재실행")
    logger.info("=" * 80)

    for strategy, interval in models:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"모델: {strategy}, rebalance_interval: {interval}")
        logger.info("=" * 80)

        try:
            result = run_track_b_pipeline(
                config_path="configs/config.yaml",
                strategy=strategy,
                force_rebuild=True,  # L6R 재실행을 위해 force_rebuild=True
            )

            logger.info(f"✓ {strategy} 완료")
            logger.info(f"  - rebalance_scores: {len(result['rebalance_scores']):,}행")
            logger.info(f"  - bt_returns: {len(result['bt_returns']):,}행")
            logger.info(f"  - 리밸런싱 날짜 수: {result['rebalance_scores']['date'].nunique():,}개")

        except Exception as e:
            logger.error(f"✗ {strategy} 실패: {e}", exc_info=True)
            continue

    logger.info("")
    logger.info("=" * 80)
    logger.info("모든 모델 실행 완료")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
