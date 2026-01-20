"""
기존 데이터 파일들의 결측치 분석 스크립트
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_existing_files():
    """현재 존재하는 데이터 파일들을 분석"""
    print("🔍 기존 데이터 파일 결측치 분석")
    print("=" * 80)

    interim_dir = PROJECT_ROOT / "data" / "interim"

    # 실제 존재하는 파일들 찾기
    existing_files = []
    for file_path in interim_dir.glob("*.parquet"):
        existing_files.append(file_path)
    for file_path in interim_dir.glob("*.csv"):
        existing_files.append(file_path)

    print(f"발견된 파일 수: {len(existing_files)}")

    if len(existing_files) == 0:
        print("❌ 분석할 파일이 없습니다.")
        return

    results = []

    for file_path in existing_files:
        print(f"\n📊 파일 분석: {file_path.name}")
        print("-" * 50)

        try:
            # 파일 읽기
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            print(f"✅ 로드 완료: {len(df):,}행 x {len(df.columns)}열")
            print(
                f"   메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            )

            # 결측치 분석
            missing_by_col = df.isnull().sum()
            total_missing = missing_by_col.sum()
            total_cells = len(df) * len(df.columns)
            missing_rate = total_missing / total_cells * 100

            print("\n🔍 결측치 분석:")
            print(".1f")
            print(
                f"   결측치 있는 컬럼 수: {len(missing_by_col[missing_by_col > 0])}/{len(df.columns)}"
            )

            # 상위 결측치 컬럼
            if len(missing_by_col[missing_by_col > 0]) > 0:
                top_missing = missing_by_col[missing_by_col > 0].nlargest(5)
                print("   상위 결측치 컬럼:")
                for col, count in top_missing.items():
                    rate = count / len(df) * 100
                    print(".1f")

            # 데이터 타입 분석
            dtype_counts = df.dtypes.value_counts()
            print("\n📋 데이터 타입:")
            for dtype, count in dtype_counts.items():
                print(f"   {dtype}: {count}개 컬럼")

            # 수치형 컬럼 통계
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print("\n📈 수치형 컬럼 통계:")
                numeric_stats = df[numeric_cols].describe()
                print(f"   수치형 컬럼 수: {len(numeric_cols)}")
                print(".4f")
                print(".4f")
            # 결과 저장
            result = {
                "파일명": file_path.name,
                "행수": len(df),
                "열수": len(df.columns),
                "결측률(%)": missing_rate,
                "결측셀수": total_missing,
                "결측컬럼수": len(missing_by_col[missing_by_col > 0]),
                "수치형컬럼수": len(numeric_cols),
            }
            results.append(result)

        except Exception as e:
            print(f"❌ 분석 실패: {str(e)}")
            continue

    # 종합 보고서
    if results:
        print("\n📋 종합 분석 보고서")
        print("=" * 80)

        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False, float_format="%.2f"))

        # 문제점 분석
        print("\n🎯 데이터 품질 평가")
        print("-" * 50)

        avg_missing_rate = summary_df["결측률(%)"].mean()
        files_with_missing = sum(1 for r in results if r["결측률(%)"] > 0)

        print(".1f")
        print(f"결측치 있는 파일 수: {files_with_missing}/{len(results)}")

        if avg_missing_rate > 10:
            quality = "❌ 심각한 결측치 문제"
        elif avg_missing_rate > 5:
            quality = "⚠️ 보통 수준 결측치"
        elif avg_missing_rate > 1:
            quality = "🔶 경미한 결측치"
        else:
            quality = "✅ 양호한 데이터 품질"

        print(f"전체 품질 평가: {quality}")

        # CSV로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            PROJECT_ROOT
            / "artifacts"
            / "reports"
            / f"existing_data_quality_analysis_{timestamp}.csv"
        )
        summary_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 상세 결과 저장: {output_file}")

    print(f"\n🏆 분석 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    analyze_existing_files()
