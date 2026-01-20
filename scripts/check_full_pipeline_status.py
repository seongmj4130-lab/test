"""
Track A/B 전체 파이프라인 상태 점검 스크립트

L0부터 L7까지의 각 단계별 상태를 종합적으로 점검합니다.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def check_file_status(file_path, description=""):
    """파일 존재 여부 및 기본 정보 확인"""
    if not file_path.exists():
        return {
            'exists': False,
            'size': 0,
            'modified': None,
            'description': description,
            'status': '❌ 누락'
        }

    stat = file_path.stat()
    size_mb = stat.st_size / 1024 / 1024

    # 최근 수정 시간 (24시간 이내)
    modified_time = datetime.fromtimestamp(stat.st_mtime)
    time_diff = datetime.now() - modified_time
    is_recent = time_diff.days < 1

    return {
        'exists': True,
        'size': size_mb,
        'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
        'is_recent': is_recent,
        'description': description,
        'status': '✅ 존재' + (' (최근)' if is_recent else '')
    }

def check_data_quality(file_path):
    """데이터 품질 기본 점검"""
    if not file_path.exists():
        return None

    try:
        # 파일 읽기
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            return {'error': '지원하지 않는 형식'}

        # 기본 통계
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

        return {
            'rows': len(df),
            'cols': len(df.columns),
            'missing_rate': missing_rate,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict()
        }

    except Exception as e:
        return {'error': str(e)}

def check_pipeline_stage(stage_num, stage_name, inputs, outputs, description=""):
    """단계별 파이프라인 상태 점검"""
    print(f"\n{'='*80}")
    print(f"🔍 L{stage_num}: {stage_name}")
    print(f"{'='*80}")
    print(f"📝 설명: {description}")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 입력 데이터 점검
    print(f"\n📥 입력 데이터:")
    input_status = []
    for input_file in inputs:
        file_path = interim_dir / input_file
        status = check_file_status(file_path, f"L{stage_num} 입력")
        input_status.append(status)
        print(f"  {input_file}: {status['status']}")

    # 출력 데이터 점검
    print(f"\n📤 출력 데이터:")
    output_status = []
    for output_file in outputs:
        file_path = interim_dir / output_file
        status = check_file_status(file_path, f"L{stage_num} 출력")
        output_status.append(status)
        print(f"  {output_file}: {status['status']}")

        # 데이터 품질 점검
        if status['exists']:
            quality = check_data_quality(file_path)
            if quality and 'error' not in quality:
                print(f"    📊 크기: {quality['rows']:,}행 x {quality['cols']}열")
                print(".1f")
                print(".1f")
            elif quality and 'error' in quality:
                print(f"    ❌ 품질 분석 실패: {quality['error']}")

    # 단계별 실행 가능성 평가
    input_ready = all(s['exists'] for s in input_status)
    output_ready = all(s['exists'] for s in output_status)

    if input_ready and not output_ready:
        exec_status = "🟡 실행 필요"
    elif input_ready and output_ready:
        exec_status = "✅ 실행 완료"
    elif not input_ready:
        exec_status = "❌ 입력 데이터 누락"
    else:
        exec_status = "❓ 상태 불명"

    print(f"\n🎯 실행 상태: {exec_status}")

    return {
        'stage': stage_num,
        'name': stage_name,
        'inputs': input_status,
        'outputs': output_status,
        'execution_status': exec_status,
        'input_ready': input_ready,
        'output_ready': output_ready
    }

def check_ensemble_config():
    """앙상블 설정 상태 점검"""
    print(f"\n{'='*80}")
    print("🔧 앙상블 설정 상태 점검")
    print(f"{'='*80}")

    try:
        from src.utils.config import load_config
        cfg = load_config('configs/config.yaml')

        l5 = cfg.get('l5', {})
        model_type = l5.get('model_type', 'single')
        print(f"모델 타입: {model_type}")

        if model_type == 'ensemble':
            print("✅ 앙상블 모드 활성화")

            short_weights = l5.get('ensemble_weights_short', {})
            long_weights = l5.get('ensemble_weights_long', {})

            print("\n단기 호리즌 가중치:")
            if short_weights:
                for model, weight in short_weights.items():
                    print(".3f")
                total_short = sum(short_weights.values())
                print(".3f"            else:
                print("  ❌ 설정되지 않음")

            print("\n장기 호리즌 가중치:")
            if long_weights:
                for model, weight in long_weights.items():
                    print(".3f")
                total_long = sum(long_weights.values())
                print(".3f"            else:
                print("  ❌ 설정되지 않음")

            # 가중치 검증
            short_valid = abs(total_short - 1.0) < 0.01 if short_weights else False
            long_valid = abs(total_long - 1.0) < 0.01 if long_weights else False

            if short_valid and long_valid:
                print("✅ 가중치 합계 검증 통과")
            else:
                print("⚠️ 가중치 합계 검증 실패 (합계가 1.0이 아님)")
        else:
            print("⚠️ 단일 모델 모드 (앙상블 비활성화)")

    except Exception as e:
        print(f"❌ 설정 로드 실패: {str(e)}")

def generate_pipeline_report(results):
    """종합 파이프라인 보고서 생성"""
    print(f"\n{'='*100}")
    print("📋 전체 파이프라인 종합 보고서")
    print(f"{'='*100}")

    # 단계별 요약
    summary_data = []
    for result in results:
        summary_data.append({
            '단계': f"L{result['stage']}",
            '이름': result['name'],
            '입력준비': '✅' if result['input_ready'] else '❌',
            '출력준비': '✅' if result['output_ready'] else '❌',
            '실행상태': result['execution_status']
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # 파이프라인 상태 분석
    total_stages = len(results)
    completed_stages = sum(1 for r in results if r['execution_status'] == '✅ 실행 완료')
    ready_stages = sum(1 for r in results if r['execution_status'] == '🟡 실행 필요')
    blocked_stages = sum(1 for r in results if r['execution_status'] == '❌ 입력 데이터 누락')

    print("
📊 파이프라인 상태 분석:"    print(f"  총 단계: {total_stages}")
    print(f"  완료: {completed_stages}")
    print(f"  실행 가능: {ready_stages}")
    print(f"  차단됨: {blocked_stages}")

    # 파이프라인 건강도
    health_score = (completed_stages / total_stages) * 100
    print(".1f"

    if health_score >= 80:
        health_status = "✅ 건강함"
    elif health_score >= 60:
        health_status = "🟡 보통"
    elif health_score >= 40:
        health_status = "⚠️ 주의 필요"
    else:
        health_status = "❌ 심각한 문제"

    print(f"  건강도: {health_status}")

    # 실행 가능 단계 식별
    executable_stages = [r for r in results if r['execution_status'] == '🟡 실행 필요']
    if executable_stages:
        print("
🟡 실행 가능한 단계:"        for stage in executable_stages:
            print(f"  - L{stage['stage']}: {stage['name']}")

    # 차단된 단계 식별
    blocked_stages_list = [r for r in results if r['execution_status'] == '❌ 입력 데이터 누락']
    if blocked_stages_list:
        print("
❌ 차단된 단계:"        for stage in blocked_stages_list:
            print(f"  - L{stage['stage']}: {stage['name']}")

def main():
    """메인 함수"""
    print("🔬 Track A/B 전체 파이프라인 상태 점검")
    print("="*100)
    print(f"점검 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 각 단계별 점검
    pipeline_results = []

    # L0: Universe 구성
    result_l0 = check_pipeline_stage(
        0, "Universe 구성",
        [],  # 외부 데이터 사용
        ["universe_k200_membership_monthly.parquet"],
        "KOSPI200 종목 선정 및 멤버십 데이터 생성"
    )
    pipeline_results.append(result_l0)

    # L1: OHLCV 및 기술지표
    result_l1 = check_pipeline_stage(
        1, "OHLCV 및 기술지표",
        ["universe_k200_membership_monthly.parquet"],
        ["dataset_daily.parquet"],
        "주가 데이터 수집 및 20개+ 기술지표 계산"
    )
    pipeline_results.append(result_l1)

    # L2: 재무 데이터
    result_l2 = check_pipeline_stage(
        2, "재무 데이터 수집",
        ["dataset_daily.parquet"],
        [],  # 재무 데이터는 dataset_daily에 병합됨
        "DART API를 통한 재무제표 데이터 수집"
    )
    pipeline_results.append(result_l2)

    # L3: 패널 데이터 병합
    result_l3 = check_pipeline_stage(
        3, "패널 데이터 병합",
        ["dataset_daily.parquet"],  # 재무 데이터가 포함된 상태
        ["dataset_daily.parquet"],  # 동일 파일 업데이트
        "OHLCV, 기술지표, 재무 데이터를 통합"
    )
    pipeline_results.append(result_l3)

    # L4: CV 폴드 분할
    result_l4 = check_pipeline_stage(
        4, "Walk-Forward CV 분할",
        ["dataset_daily.parquet"],
        ["cv_folds_short.parquet", "cv_folds_long.parquet", "targets_and_folds.parquet"],
        "시계열 CV 폴드 생성 (단기 20일, 장기 120일)"
    )
    pipeline_results.append(result_l4)

    # L5: ML 모델 학습
    result_l5 = check_pipeline_stage(
        5, "ML 모델 학습",
        ["dataset_daily.parquet", "cv_folds_short.parquet", "cv_folds_long.parquet", "targets_and_folds.parquet"],
        ["pred_short_oos.parquet", "pred_long_oos.parquet"],
        "Grid, Ridge, XGBoost, RF 앙상블 모델 학습"
    )
    pipeline_results.append(result_l5)

    # L6: 스코어 생성
    result_l6 = check_pipeline_stage(
        6, "스코어 생성 및 앙상블",
        ["pred_short_oos.parquet", "pred_long_oos.parquet"],
        ["rebalance_scores.parquet"],
        "개별 모델 예측을 가중치 기반 앙상블 스코어로 변환"
    )
    pipeline_results.append(result_l6)

    # L7: 백테스트
    result_l7 = check_pipeline_stage(
        7, "백테스트 실행",
        ["rebalance_scores.parquet", "targets_and_folds.parquet", "dataset_daily.parquet"],
        ["bt_metrics_*.parquet", "bt_positions_*.parquet", "bt_equity_curve.parquet"],
        "4개 전략 백테스트 및 성과 분석"
    )
    pipeline_results.append(result_l7)

    # 앙상블 설정 점검
    check_ensemble_config()

    # 종합 보고서
    generate_pipeline_report(pipeline_results)

    print(f"\n🏆 점검 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
