# -*- coding: utf-8 -*-
"""
L0~L4 공통데이터 품질 분석 스크립트

기존 데이터를 사용하여 각 단계별 산출물의 결측치를 분석하고 문제점을 파악합니다.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_data_file(file_path, stage_name, expected_cols=None):
    """단일 데이터 파일 분석"""
    print(f"\n📊 {stage_name} 분석")
    print("-" * 50)

    # 디버깅: 파일 경로 출력
    print(f"파일 경로: {file_path}")
    print(f"파일 존재 여부: {file_path.exists()}")

    if not file_path.exists():
        print(f"❌ 파일 없음: {file_path}")
        return None

    try:
        # 파일 읽기 (parquet 우선, 없으면 csv)
        parquet_path = Path(str(file_path) + '.parquet')
        csv_path = Path(str(file_path) + '.csv')

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            print(f"❌ 지원되지 않는 파일 형식: {file_path}")
            return None

        print(f"✅ 파일 로드 완료: {len(df):,}행 x {len(df.columns)}열")
        print(f"   메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # 기본 정보
        print(f"   날짜 범위: {df.index.min() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'} ~ {df.index.max() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'}")

        # 컬럼 정보
        if expected_cols:
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                print(f"   ⚠️ 누락된 예상 컬럼: {missing_cols}")

        # 결측치 분석
        missing_analysis = analyze_missing_values(df, stage_name)

        # 데이터 타입 분석
        dtype_analysis = analyze_data_types(df)

        return {
            'dataframe': df,
            'row_count': len(df),
            'col_count': len(df.columns),
            'missing_analysis': missing_analysis,
            'dtype_analysis': dtype_analysis
        }

    except Exception as e:
        print(f"❌ 파일 분석 실패: {str(e)}")
        return None

def analyze_missing_values(df, stage_name):
    """결측치 분석"""
    print(f"\n   🔍 결측치 분석:")

    # 컬럼별 결측치
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]

    if len(missing_cols) == 0:
        print("   ✅ 결측치 없음")
        return {'status': 'clean', 'missing_rate': 0.0}

    # 결측치가 있는 컬럼들
    total_cells = len(df) * len(df.columns)
    total_missing = missing_by_col.sum()
    missing_rate = total_missing / total_cells

    print(".1%")
    print(f"   결측치 있는 컬럼 수: {len(missing_cols)}/{len(df.columns)}")

    # 상위 결측치 컬럼들
    top_missing = missing_cols.nlargest(10)
    print("   주요 결측치 컬럼 (Top 10):")
    for col, count in top_missing.items():
        rate = count / len(df) * 100
        print(".1f")

    # 결측치 패턴 분석
    missing_pattern = analyze_missing_patterns(df)

    return {
        'status': 'has_missing' if missing_rate > 0 else 'clean',
        'missing_rate': missing_rate,
        'missing_cols': len(missing_cols),
        'total_missing': total_missing,
        'top_missing_cols': top_missing.to_dict(),
        'pattern_analysis': missing_pattern
    }

def analyze_missing_patterns(df):
    """결측치 패턴 분석"""
    missing_matrix = df.isnull()

    # 행별 결측치
    missing_by_row = missing_matrix.sum(axis=1)
    rows_with_missing = (missing_by_row > 0).sum()
    rows_missing_rate = rows_with_missing / len(df) * 100

    # 완전 결측 행
    complete_missing_rows = (missing_by_row == len(df.columns)).sum()

    # 결측치 상관관계 (주요 컬럼들만)
    if len(df.columns) > 50:
        # 메모리 효율을 위해 샘플링
        sample_cols = df.columns[:20]  # 상위 20개 컬럼만
        corr_matrix = missing_matrix[sample_cols].corr()
    else:
        corr_matrix = missing_matrix.corr()

    # 결측치 상관관계가 높은 쌍 찾기
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # 0.8 이상 상관관계
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    return {
        'rows_with_missing': rows_with_missing,
        'rows_missing_rate': rows_missing_rate,
        'complete_missing_rows': complete_missing_rows,
        'high_corr_missing_pairs': high_corr_pairs[:10]  # 상위 10개만
    }

def analyze_data_types(df):
    """데이터 타입 분석"""
    dtype_counts = df.dtypes.value_counts()

    # 수치형 컬럼 분석
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()

        # 이상치 분석 (IQR 방법)
        outlier_analysis = {}
        for col in numeric_cols[:10]:  # 상위 10개만
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_rate = outliers / len(df) * 100

            if outlier_rate > 1:  # 1% 이상 이상치
                outlier_analysis[col] = {
                    'outlier_count': outliers,
                    'outlier_rate': outlier_rate,
                    'bounds': [lower_bound, upper_bound]
                }

    return {
        'dtype_counts': dtype_counts.to_dict(),
        'numeric_cols': len(numeric_cols),
        'outlier_analysis': outlier_analysis if 'outlier_analysis' in locals() else {}
    }

def analyze_l0_l4_pipeline():
    """L0~L4 파이프라인 데이터 품질 분석"""
    print("🔬 L0~L4 공통데이터 품질 분석")
    print("="*80)
    print(f"분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 실제 존재하는 파일들만 분석
    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 존재하는 파일들을 찾아서 분석 대상으로 설정
    existing_files = []
    potential_files = [
        ('L5_predictions_short', 'pred_short_oos', ['date', 'ticker', 'pred']),
        ('L5_predictions_long', 'pred_long_oos', ['date', 'ticker', 'pred']),
        ('L6_rebalance_scores', 'rebalance_scores', ['date', 'ticker', 'score_ens'])
    ]

    for stage_name, file_base, expected_cols in potential_files:
        parquet_file = interim_dir / f"{file_base}.parquet"
        csv_file = interim_dir / f"{file_base}.csv"

        if parquet_file.exists() or csv_file.exists():
            existing_files.append({
                'stage': stage_name,
                'file': file_base,
                'expected_cols': expected_cols
            })

    analysis_targets = existing_files

    results = {}

    for target in analysis_targets:
        file_path = interim_dir / target['file']
        result = analyze_data_file(
            file_path,
            target['stage'],
            target.get('expected_cols')
        )
        if result:
            results[target['stage']] = result

    # 종합 분석
    generate_summary_report(results)

    return results

def generate_summary_report(results):
    """종합 분석 보고서 생성"""
    print("\n📋 종합 품질 분석 보고서")
    print("="*80)

    if not results:
        print("❌ 분석할 데이터가 없습니다.")
        return

    # 단계별 요약
    summary_data = []
    for stage, result in results.items():
        missing_rate = result['missing_analysis']['missing_rate'] * 100
        missing_cols = result['missing_analysis']['missing_cols']

        status = "✅ 양호"
        if missing_rate > 10:
            status = "❌ 심각"
        elif missing_rate > 5:
            status = "⚠️ 주의"
        elif missing_rate > 1:
            status = "🔶 보통"

        summary_data.append({
            '단계': stage,
            '행수': result['row_count'],
            '열수': result['col_count'],
            '결측률(%)': ".1f",
            '결측컬럼수': missing_cols,
            '상태': status
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # 문제점 분석
    print("\n🎯 주요 문제점 분석")
    print("-"*50)

    total_stages = len(results)
    clean_stages = sum(1 for r in results.values() if r['missing_analysis']['status'] == 'clean')
    problematic_stages = total_stages - clean_stages

    print(f"총 분석 단계: {total_stages}")
    print(f"클린 단계: {clean_stages}")
    print(f"문제 단계: {problematic_stages}")

    # 결측치 심각도 분석
    severe_missing = sum(1 for r in results.values() if r['missing_analysis']['missing_rate'] > 0.1)
    moderate_missing = sum(1 for r in results.values() if 0.05 < r['missing_analysis']['missing_rate'] <= 0.1)

    print("\n결측치 심각도:")
    print(f"  심각(>10%): {severe_missing}단계")
    print(f"  보통(5-10%): {moderate_missing}단계")
    print(f"  경미(<5%): {problematic_stages - severe_missing - moderate_missing}단계")

    # 데이터 활용성 평가
    usability_score = (clean_stages / total_stages) * 100
    print(".1f")
    if usability_score >= 80:
        print("✅ 데이터 활용성: 높음")
    elif usability_score >= 60:
        print("⚠️ 데이터 활용성: 보통")
    else:
        print("❌ 데이터 활용성: 낮음")

    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROJECT_ROOT / 'artifacts' / 'reports' / f'l0_l4_data_quality_analysis_{timestamp}.csv'
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 상세 결과 저장: {output_file}")

def main():
    """메인 함수"""
    analyze_l0_l4_pipeline()

    print("\n🏆 분석 완료")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()