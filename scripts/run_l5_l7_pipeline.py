# -*- coding: utf-8 -*-
"""
L5-L7 파이프라인 실행 스크립트 (재현성 테스트용)

ML 학습부터 백테스트까지 완전 실행
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import argparse
import random
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def set_random_seed(seed=None):
    """랜덤 시드 설정"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        print(f"🎲 랜덤 시드 설정: {seed}")
    else:
        print("🎲 랜덤 시드: 기본값 사용")

def run_l5_ml_training():
    """L5: ML 모델 학습 실행"""
    print("🔄 L5: ML 모델 학습 시작")

    try:
        # 기존 src/stages/l5_train_models.py 실행
        from src.stages.l5_train_models import main as l5_main

        # 모델 학습 실행
        l5_main()

        # 결과 검증
        interim_dir = PROJECT_ROOT / 'data' / 'interim'
        pred_short = interim_dir / 'pred_short_oos.parquet'
        pred_long = interim_dir / 'pred_long_oos.parquet'

        if pred_short.exists() and pred_long.exists():
            df_short = pd.read_parquet(pred_short)
            df_long = pd.read_parquet(pred_long)

            print("  ✅ 단기 예측: "            print("  ✅ 장기 예측: "            return True
        else:
            print("  ❌ 예측 파일 생성 실패")
            return False

    except Exception as e:
        print(f"  ❌ L5 실행 실패: {str(e)}")
        return False

def run_l6_scoring():
    """L6: 스코어 생성 실행"""
    print("🔄 L6: 스코어 생성 시작")

    try:
        # 기존 src/stages/l6_scoring.py 실행
        from src.stages.l6_scoring import main as l6_main

        # 스코어 생성 실행
        l6_main()

        # 결과 검증
        interim_dir = PROJECT_ROOT / 'data' / 'interim'
        scores_file = interim_dir / 'rebalance_scores.parquet'

        if scores_file.exists():
            df = pd.read_parquet(scores_file)
            missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            print(".1f"            print("  ✅ 스코어 생성 완료"            return True
        else:
            print("  ❌ 스코어 파일 생성 실패")
            return False

    except Exception as e:
        print(f"  ❌ L6 실행 실패: {str(e)}")
        return False

def run_l7_backtest():
    """L7: 백테스트 실행"""
    print("🔄 L7: 백테스트 시작")

    try:
        # 기존 백테스트 실행
        from scripts.run_backtest_4models import run_backtest_4models

        # 백테스트 실행
        run_backtest_4models()

        # 결과 검증
        interim_dir = PROJECT_ROOT / 'data' / 'interim'
        bt_files = [
            'bt_metrics_bt20_ens.parquet',
            'bt_metrics_bt20_short.parquet',
            'bt_metrics_bt120_ens.parquet',
            'bt_metrics_bt120_long.parquet'
        ]

        success_count = 0
        for bt_file in bt_files:
            if (interim_dir / bt_file).exists():
                success_count += 1
                df = pd.read_parquet(interim_dir / bt_file)
                holdout = df[df['phase'] == 'holdout']
                if len(holdout) > 0:
                    sharpe = holdout['net_sharpe'].iloc[0]
                    print(".4f"            else:
                print(f"  ❌ {bt_file}: Holdout 데이터 없음")

        if success_count == 4:
            print("  ✅ 모든 백테스트 완료")
            return True
        else:
            print(f"  ⚠️ 백테스트 부분 성공: {success_count}/4")
            return False

    except Exception as e:
        print(f"  ❌ L7 실행 실패: {str(e)}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='L5-L7 파이프라인 실행')
    parser.add_argument('--seed', type=int, help='랜덤 시드')
    args = parser.parse_args()

    print("🚀 L5-L7 파이프라인 실행 시작")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 랜덤 시드 설정
    set_random_seed(args.seed)

    success = True

    # L5 실행
    if not run_l5_ml_training():
        success = False

    # L6 실행
    if success and not run_l6_scoring():
        success = False

    # L7 실행
    if success and not run_l7_backtest():
        success = False

    # 결과 요약
    print("
📊 실행 결과 요약"    print("="*60)
    if success:
        print("✅ L5-L7 파이프라인 완전 성공")
        print("✅ 재현성 테스트 데이터 생성 완료")
    else:
        print("❌ L5-L7 파이프라인 실행 실패")
        sys.exit(1)

    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()