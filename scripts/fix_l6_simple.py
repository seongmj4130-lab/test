"""
L6 결측치 처리 (간단 버전)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("🔧 L6 결측치 처리 시작")

    interim_dir = PROJECT_ROOT / "data" / "interim"
    scores_file = interim_dir / "rebalance_scores.parquet"

    # 데이터 로드
    df = pd.read_parquet(scores_file)
    print(f"📊 데이터 로드: {len(df)}행 x {len(df.columns)}열")

    # 결측치 분석
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    total_missing = missing_by_col.sum()

    print(f"❌ 총 결측치: {total_missing}개")
    print(f"❌ 결측 컬럼: {len(missing_cols)}개")

    for col, count in missing_cols.items():
        rate = count / len(df) * 100
        print(".1f")

    # 결측치 보간
    df_fixed = df.copy()

    # 1. score_ens: 개별 모델 평균
    if "score_ens" in df.columns and df["score_ens"].isnull().sum() > 0:
        score_cols = [
            col for col in df.columns if col.startswith("score_") and col != "score_ens"
        ]
        if score_cols:
            df_fixed["score_ens"] = df_fixed["score_ens"].fillna(
                df_fixed[score_cols].mean(axis=1)
            )
            print("✅ score_ens: 개별 모델 평균으로 보간")

    # 2. 개별 스코어: 전일 값 유지
    for col in ["score_grid", "score_ridge", "score_xgboost", "score_rf"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            df_fixed[col] = df_fixed[col].fillna(method="ffill").fillna(0.0)
            print(f"✅ {col}: 전일 값 유지로 보간")

    # 3. weight 컬럼: 0으로 채움
    weight_cols = [col for col in df.columns if col.startswith("weight_")]
    for col in weight_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df_fixed[col] = df_fixed[col].fillna(0.0)
            print(f"✅ {col}: 0.0으로 채움")

    # 검증
    final_missing = df_fixed.isnull().sum().sum()
    print(f"\n📊 보간 결과: {total_missing} → {final_missing}")

    if final_missing == 0:
        # 백업 및 저장
        import shutil

        backup_file = interim_dir / "rebalance_scores_original.parquet"
        if not backup_file.exists():
            shutil.copy2(scores_file, backup_file)
            print("📋 원본 백업 완료")

        df_fixed.to_parquet(scores_file, index=False)
        print("✅ 결측치 처리 완료!")
    else:
        print(f"⚠️ 잔여 결측치: {final_missing}개")


if __name__ == "__main__":
    main()
