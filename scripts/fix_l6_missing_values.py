"""
L6 결측치 처리 스크립트

rebalance_scores.parquet의 결측치를 보간하여 백테스트 정확도 향상
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_current_missing():
    """현재 결측치 현황 분석"""
    print("🔍 L6 결측치 현황 분석")
    print("="*50)

    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    scores_file = interim_dir / 'rebalance_scores.parquet'

    df = pd.read_parquet(scores_file)
    print(f"📊 원본 데이터: {len(df):,}행 x {len(df.columns)}열")

    # 결측치 분석
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]

    if len(missing_cols) > 0:
        print("\n❌ 결측치 있는 컬럼:")        total_missing = 0
        for col, count in missing_cols.items():
            rate = count / len(df) * 100
            print(".1f")
            total_missing += count

        total_rate = total_missing / (len(df) * len(df.columns)) * 100
        print(".2f")

        # 결측치 패턴 분석
        missing_rows = df[df.isnull().any(axis=1)]
        print(f"\n⚠️ 결측치가 있는 행 수: {len(missing_rows)}/{len(df)} ({len(missing_rows)/len(df)*100:.1f}%)")

        return df, missing_cols
    else:
        print("\n✅ 결측치 없음")
        return df, None

def apply_missing_fixes(df, missing_cols):
    """결측치 보간 적용"""
    print("\n🔧 결측치 보간 적용")
    print("="*50)

    df_fixed = df.copy()
    fixes_applied = {}

    # 1. score_ens 결측치 처리 (가장 중요)
    if 'score_ens' in missing_cols:
        score_cols = [col for col in df.columns if col.startswith('score_') and col != 'score_ens']
        if len(score_cols) > 0:
            print("1️⃣ score_ens 보간: 개별 모델 스코어 평균 사용")
            # 개별 모델 스코어의 평균으로 보간
            df_fixed['score_ens'] = df_fixed['score_ens'].fillna(
                df_fixed[score_cols].mean(axis=1)
            )
            fixes_applied['score_ens'] = 'average_of_individual_scores'
            print(f"   적용된 행 수: {df_fixed['score_ens'].isnull().sum()} → {df_fixed['score_ens'].isnull().sum()}")

    # 2. 개별 모델 스코어 결측치 처리
    individual_scores = ['score_grid', 'score_ridge', 'score_xgboost', 'score_rf']
    for score_col in individual_scores:
        if score_col in missing_cols:
            print(f"2️⃣ {score_col} 보간: 전일 값 유지")
            df_fixed[score_col] = df_fixed[score_col].fillna(method='ffill')
            # 그래도 남은 결측치는 0으로 채움
            df_fixed[score_col] = df_fixed[score_col].fillna(0.0)
            fixes_applied[score_col] = 'forward_fill_then_zero'
            print(f"   적용된 행 수: {df_fixed[score_col].isnull().sum()} → {df_fixed[score_col].isnull().sum()}")

    # 3. weight 컬럼 결측치 처리
    weight_cols = [col for col in df.columns if col.startswith('weight_')]
    for weight_col in weight_cols:
        if weight_col in missing_cols:
            print(f"3️⃣ {weight_col} 보간: 0.0으로 채움")
            df_fixed[weight_col] = df_fixed[weight_col].fillna(0.0)
            fixes_applied[weight_col] = 'fill_zero'
            print(f"   적용된 행 수: {df_fixed[weight_col].isnull().sum()} → {df_fixed[weight_col].isnull().sum()}")

    # 4. 기타 결측치 확인 및 처리
    remaining_missing = df_fixed.isnull().sum()
    remaining_cols = remaining_missing[remaining_missing > 0]

    if len(remaining_cols) > 0:
        print("
4️⃣ 잔여 결측치 처리:"        for col, count in remaining_cols.items():
            if df_fixed[col].dtype in ['float64', 'float32']:
                print(f"   {col}: 중앙값 보간")
                median_val = df_fixed[col].median()
                df_fixed[col] = df_fixed[col].fillna(median_val)
                fixes_applied[col] = f'median_fill_{median_val:.4f}'
            else:
                print(f"   {col}: 최빈값 또는 0으로 채움")
                if df_fixed[col].dtype == 'object':
                    mode_val = df_fixed[col].mode()
                    if len(mode_val) > 0:
                        df_fixed[col] = df_fixed[col].fillna(mode_val[0])
                        fixes_applied[col] = f'mode_fill_{mode_val[0]}'
                    else:
                        df_fixed[col] = df_fixed[col].fillna('')
                        fixes_applied[col] = 'empty_string_fill'
                else:
                    df_fixed[col] = df_fixed[col].fillna(0)
                    fixes_applied[col] = 'zero_fill'

    return df_fixed, fixes_applied

def verify_fix_quality(df_original, df_fixed):
    """보간 품질 검증"""
    print("\n🔍 보간 품질 검증")
    print("="*50)

    # 결측치 제거율 계산
    original_missing = df_original.isnull().sum().sum()
    fixed_missing = df_fixed.isnull().sum().sum()
    fix_rate = (original_missing - fixed_missing) / original_missing * 100 if original_missing > 0 else 100

    print(".1f")

    # 데이터 분포 변화 분석
    numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
    print(f"\n📊 수치형 컬럼 분포 변화 ({len(numeric_cols)}개 컬럼):")

    for col in numeric_cols[:5]:  # 처음 5개만 표시
        if col in df_original.columns:
            orig_mean = df_original[col].mean()
            fixed_mean = df_fixed[col].mean()
            change = (fixed_mean - orig_mean) / abs(orig_mean) * 100 if orig_mean != 0 else 0
            print(".4f")

    # 데이터 무결성 검증
    if fixed_missing == 0:
        print("\n✅ 데이터 무결성: 완전 복원")
        return True
    else:
        print(f"\n⚠️ 잔여 결측치: {fixed_missing}개")
        return False

def save_fixed_data(df_fixed, fixes_applied):
    """보간된 데이터 저장"""
    print("\n💾 보간된 데이터 저장")
    print("="*50)

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 백업 원본
    original_file = interim_dir / 'rebalance_scores.parquet'
    backup_file = interim_dir / 'rebalance_scores_original.parquet'
    if not backup_file.exists():
        import shutil
        shutil.copy2(original_file, backup_file)
        print("📋 원본 백업 완료")

    # 보간된 데이터 저장
    df_fixed.to_parquet(original_file, index=False)
    print(f"✅ 보간된 데이터 저장: {original_file}")
    print(f"📏 파일 크기: {original_file.stat().st_size / 1024:.1f} KB")

    # 보간 정보 저장
    fixes_info = {
        'timestamp': datetime.now().isoformat(),
        'original_missing': len(df_fixed) * len(df_fixed.columns) - df_fixed.count().sum(),
        'fixes_applied': fixes_applied,
        'final_missing': df_fixed.isnull().sum().sum()
    }

    fixes_file = interim_dir / 'l6_missing_fixes_info.json'
    import json
    with open(fixes_file, 'w', encoding='utf-8') as f:
        json.dump(fixes_info, f, indent=2, ensure_ascii=False)
    print(f"📝 보간 정보 저장: {fixes_file}")

def main():
    """메인 함수"""
    print("🔧 L6 결측치 처리 작업 시작")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 현재 결측치 분석
    df_original, missing_cols = analyze_current_missing()

    if missing_cols is None or len(missing_cols) == 0:
        print("\n✅ 결측치가 없어 처리 불필요")
        return

    # 2. 결측치 보간 적용
    df_fixed, fixes_applied = apply_missing_fixes(df_original, missing_cols)

    # 3. 보간 품질 검증
    quality_ok = verify_fix_quality(df_original, df_fixed)

    if quality_ok:
        # 4. 보간된 데이터 저장
        save_fixed_data(df_fixed, fixes_applied)
        print("
🎉 L6 결측치 처리 성공!"        print("📈 다음 단계: 백테스트 재실행 준비 완료")
    else:
        print("
⚠️ 보간 품질 검증 실패"    print(f"\n종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
