#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L8 랭킹 엔진 직접 실행
"""

import sys
from pathlib import Path

# 경로 추가
sys.path.append('src')

def run_l8_ranking():
    """L8 랭킹 엔진 실행"""

    try:
        from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
            run_L8_long_rank_engine,
            run_L8_short_rank_engine,
        )
        print('L8 모듈 임포트 성공')

        # 설정 로드
        from src.utils.config import load_config
        cfg = load_config('configs/config.yaml')
        print('설정 로드 완료')

        # artifacts 준비
        from src.utils.io import artifact_exists, load_artifact, save_artifact

        interim_dir = Path('data/interim')
        artifacts = {}

        # 패널 데이터 로드
        panel_path = interim_dir / 'panel_merged_daily.parquet'
        if artifact_exists(panel_path):
            artifacts['panel_merged_daily'] = load_artifact(panel_path)
            print(f'패널 데이터 로드: {len(artifacts["panel_merged_daily"]):,}행')
        else:
            print('패널 데이터 없음')
            return

        # dataset_daily 준비
        dataset_path = interim_dir / 'dataset_daily.parquet'
        if artifact_exists(dataset_path):
            artifacts['dataset_daily'] = load_artifact(dataset_path)
            print(f'데이터셋 로드: {len(artifacts["dataset_daily"]):,}행')
        else:
            artifacts['dataset_daily'] = artifacts['panel_merged_daily']
            print('데이터셋 대신 패널 데이터 사용')

        # 단기 랭킹 생성
        print('\n단기 랭킹 생성 시작...')
        outputs_short, warns_short = run_L8_short_rank_engine(
            cfg=cfg,
            artifacts=artifacts,
            force=True
        )

        if 'ranking_short_daily' in outputs_short:
            ranking_short = outputs_short['ranking_short_daily']
            print(f'단기 랭킹 생성 성공: {len(ranking_short):,}행')

            # 저장
            save_artifact(ranking_short, interim_dir / 'ranking_short_daily', force=True)
            print('단기 랭킹 저장 완료')

            # 샘플 출력
            print('샘플 데이터:')
            print(ranking_short.head(3).to_string())
        else:
            print('단기 랭킹 생성 실패')
            print('outputs:', outputs_short)
            print('warns:', warns_short)
            return

        # 장기 랭킹 생성
        print('\n장기 랭킹 생성 시작...')
        outputs_long, warns_long = run_L8_long_rank_engine(
            cfg=cfg,
            artifacts=artifacts,
            force=True
        )

        if 'ranking_long_daily' in outputs_long:
            ranking_long = outputs_long['ranking_long_daily']
            print(f'장기 랭킹 생성 성공: {len(ranking_long):,}행')

            # 저장
            save_artifact(ranking_long, interim_dir / 'ranking_long_daily', force=True)
            print('장기 랭킹 저장 완료')

            # 샘플 출력
            print('샘플 데이터:')
            print(ranking_long.head(3).to_string())
        else:
            print('장기 랭킹 생성 실패')
            print('outputs:', outputs_long)
            print('warns:', warns_long)
            return

        print('\n✅ 모든 랭킹 생성 완료!')

    except Exception as e:
        print(f'오류 발생: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_l8_ranking()
