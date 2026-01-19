import os
import shutil
from datetime import datetime

import pandas as pd


def create_baseline_backup():
    """
    현재 프로젝트 상태를 baseline으로 백업합니다.
    모든 설정, 코드, 데이터를 baseline 폴더에 저장합니다.
    """
    base_dir = "c:/Users/seong/OneDrive/Desktop/bootcamp/000_code"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = f"{base_dir}/baseline_{timestamp}"

    print(f"🔄 Baseline 백업 생성 중... (타임스탬프: {timestamp})")
    print(f"📁 대상 폴더: {baseline_dir}")

    # Baseline 디렉토리 생성
    os.makedirs(baseline_dir, exist_ok=True)

    # 백업할 폴더들
    folders_to_backup = [
        'configs',      # 모든 설정 파일
        'src',         # 모든 파이프라인 코드
        'data',        # 현재 데이터 상태
        'artifacts'    # 모델과 리포트
    ]

    # 백업할 개별 파일들
    files_to_backup = [
        'README.md',
        'ppt_report.md',
        'final_report.md',
        'final_ranking_report.md',
        'final_backtest_report.md',
        'final_easy_report.md'
    ]

    # 폴더 백업
    for folder in folders_to_backup:
        src_path = os.path.join(base_dir, folder)
        dst_path = os.path.join(baseline_dir, folder)

        if os.path.exists(src_path):
            print(f"📋 폴더 백업 중: {folder}")
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    # 개별 파일 백업
    for file in files_to_backup:
        src_path = os.path.join(base_dir, file)
        dst_path = os.path.join(baseline_dir, file)

        if os.path.exists(src_path):
            print(f"📄 파일 백업 중: {file}")
            shutil.copy2(src_path, dst_path)

    # Baseline 정보 파일 생성
    baseline_info = f"""
# KOSPI200 퀀트 투자 전략 Baseline 정보

**생성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**타임스탬프**: {timestamp}
**프로젝트 상태**: 최종 완료 (Track A Sharpe 0.914 달성)

## 📊 Baseline 주요 성과

### Track A (랭킹 엔진)
- 단기 랭킹 Holdout Hit Ratio: 50.99%
- 통합 랭킹 Holdout Hit Ratio: 51.06%
- 장기 랭킹 Holdout Hit Ratio: 51.00%

### Track B (투자 전략)
- BT20 단기 전략 Sharpe: 0.914
- BT20 단기 전략 CAGR: 13.4%
- BT20 단기 전략 MDD: -4.4%

## 📁 Baseline 포함 파일

### 설정 파일 (configs/)
- config.yaml: 메인 설정 파일
- feature_*.yaml: 피처 가중치 설정
- features_*.yaml: 피처 리스트 설정

### 코드 파일 (src/)
- tracks/: Track A/B 구현
- stages/: 데이터 처리 단계
- pipeline/: 메인 파이프라인
- utils/: 유틸리티 함수들

### 데이터 파일 (data/)
- interim/: 중간 처리 데이터
- ui_*.csv: UI용 데이터
- strategies_*.csv: 전략 성과 데이터

### 산출물 (artifacts/)
- models/: 학습된 모델
- reports/: 분석 리포트

## 🔧 Baseline 설정값 요약

### 데이터 처리 파라미터
- 유니버스: KOSPI200
- 기간: 2016-01-01 ~ 2024-12-31
- 빈도: 일별

### 모델 파라미터
- Track A: 앙상블 (Grid 30% + Ridge 60% + XGBoost 10%)
- Track B: BT20/BT120 전략들
- 비용: 거래비용 10bps, 슬리피지 0bps

### 평가 지표
- Holdout 기간: 2023-01-31 ~ 2024-11-18
- 주요 지표: Sharpe, CAGR, MDD, Hit Ratio

## 🚀 Baseline 사용법

이 baseline을 사용하여:
1. 새로운 전략 실험
2. 모델 개선 비교
3. 성능 벤치마킹
4. 재현성 검증

## ⚠️ 주의사항

- 이 baseline은 프로젝트 완료 시점의 안정적 상태임
- 변경 시 별도 백업 권장
- 실전 적용 전 추가 검증 필요
"""

    with open(os.path.join(baseline_dir, 'BASELINE_INFO.md'), 'w', encoding='utf-8') as f:
        f.write(baseline_info)

    # 현재 상태 요약 생성
    current_status = {
        'timestamp': timestamp,
        'project_status': 'completed',
        'track_a_performance': {
            'short_holdout_hit_ratio': 50.99,
            'ensemble_holdout_hit_ratio': 51.06,
            'long_holdout_hit_ratio': 51.00
        },
        'track_b_performance': {
            'bt20_short_sharpe': 0.914,
            'bt20_short_cagr': 0.134,
            'bt20_short_mdd': -0.044
        }
    }

    # JSON으로 저장
    import json
    with open(os.path.join(baseline_dir, 'baseline_status.json'), 'w', encoding='utf-8') as f:
        json.dump(current_status, f, indent=2, ensure_ascii=False)

    print("✅ Baseline 백업 완료!")
    print(f"📍 백업 위치: {baseline_dir}")
    print("📋 백업된 폴더들:")
    for folder in folders_to_backup:
        print(f"   - {folder}/")
    print("📄 백업된 파일들:")
    for file in files_to_backup:
        print(f"   - {file}")

    print("\n🎯 Baseline 생성 완료!")
    print("이제 현재 상태가 baseline으로 설정되었습니다.")
    print("향후 변경사항은 이 baseline과 비교하여 평가할 수 있습니다.")

    return baseline_dir

if __name__ == "__main__":
    create_baseline_backup()
