# -*- coding: utf-8 -*-
"""
Track A/B 파이프라인 상태 간단 점검
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def check_stage(stage_num, name, input_files, output_files):
    """단계별 상태 확인"""
    print(f"\n🔍 L{stage_num}: {name}")
    print("-" * 60)

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 입력 파일 확인
    input_ok = True
    print("📥 입력:")
    for file in input_files:
        exists = (interim_dir / file).exists()
        status = "✅" if exists else "❌"
        print(f"  {file}: {status}")
        if not exists:
            input_ok = False

    # 출력 파일 확인
    output_ok = True
    print("📤 출력:")
    for file in output_files:
        # 와일드카드 처리
        if '*' in file:
            matches = list(interim_dir.glob(file))
            exists = len(matches) > 0
            if exists:
                print(f"  {file}: ✅ ({len(matches)}개 파일)")
                # 데이터 품질 확인 (첫 번째 파일만)
                try:
                    df = pd.read_parquet(matches[0])
                    missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                    print(".1f")
                except:
                    print("    ⚠️ 품질 확인 실패")
            else:
                print(f"  {file}: ❌")
                output_ok = False
        else:
            exists = (interim_dir / file).exists()
            status = "✅" if exists else "❌"
            print(f"  {file}: {status}")
            if exists:
                try:
                    df = pd.read_parquet(interim_dir / file)
                    missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                    print(".1f")
                except:
                    print("    ⚠️ 품질 확인 실패")
            if not exists:
                output_ok = False

    # 상태 결정
    if input_ok and output_ok:
        status = "✅ 완료"
    elif input_ok and not output_ok:
        status = "🟡 실행 필요"
    else:
        status = "❌ 입력 누락"

    print(f"🎯 상태: {status}")
    return status

def main():
    print("🔬 Track A/B 파이프라인 상태 점검")
    print("="*80)

    # L0~L7 단계별 점검
    stages = [
        (0, "Universe 구성", [], ["universe_k200_membership_monthly.parquet"]),
        (1, "OHLCV 수집", ["universe_k200_membership_monthly.parquet"], ["dataset_daily.parquet"]),
        (2, "재무 데이터", ["dataset_daily.parquet"], []),  # 출력은 dataset_daily에 병합
        (3, "패널 병합", ["dataset_daily.parquet"], ["dataset_daily.parquet"]),
        (4, "CV 분할", ["dataset_daily.parquet"], ["cv_folds_short.parquet", "cv_folds_long.parquet", "targets_and_folds.parquet"]),
        (5, "ML 학습", ["dataset_daily.parquet", "cv_folds_short.parquet", "cv_folds_long.parquet"], ["pred_short_oos.parquet", "pred_long_oos.parquet"]),
        (6, "스코어 생성", ["pred_short_oos.parquet", "pred_long_oos.parquet"], ["rebalance_scores.parquet"]),
        (7, "백테스트", ["rebalance_scores.parquet"], ["bt_metrics_*.parquet"])
    ]

    results = []
    for stage_num, name, inputs, outputs in stages:
        status = check_stage(stage_num, name, inputs, outputs)
        results.append({
            '단계': f'L{stage_num}',
            '이름': name,
            '상태': status
        })

    # 앙상블 설정 확인
    print("\n🔧 앙상블 설정 확인")
    print("-" * 60)
    try:
        from src.utils.config import load_config
        cfg = load_config('configs/config.yaml')
        l5 = cfg.get('l5', {})
        model_type = l5.get('model_type', 'single')

        if model_type == 'ensemble':
            print("✅ 앙상블 모드 활성화")
            short_weights = l5.get('ensemble_weights_short', {})
            long_weights = l5.get('ensemble_weights_long', {})

            if short_weights and long_weights:
                print("✅ 가중치 설정됨")
                short_sum = sum(short_weights.values())
                long_sum = sum(long_weights.values())
                if abs(short_sum - 1.0) < 0.01 and abs(long_sum - 1.0) < 0.01:
                    print("✅ 가중치 합계 검증 통과")
                else:
                    print("⚠️ 가중치 합계 검증 실패")
            else:
                print("⚠️ 가중치 설정 누락")
        else:
            print("⚠️ 단일 모델 모드")
    except Exception as e:
        print(f"❌ 설정 확인 실패: {str(e)}")

    # 요약
    print("\n📋 파이프라인 요약")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    completed = sum(1 for r in results if '✅' in r['상태'])
    ready = sum(1 for r in results if '🟡' in r['상태'])
    blocked = sum(1 for r in results if '❌' in r['상태'])

    print("\n📊 통계:")
    print(f"  완료: {completed}단계")
    print(f"  실행가능: {ready}단계")
    print(f"  차단: {blocked}단계")

    health = (completed / len(results)) * 100
    print(".1f")
    if health >= 80:
        print("✅ 파이프라인 건강함")
    elif health >= 60:
        print("🟡 보통 상태")
    else:
        print("❌ 개선 필요")

if __name__ == "__main__":
    main()
