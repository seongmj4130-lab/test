# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/validate_results.py

import pandas as pd

# -----------------------------------------------------------------------------
# 1. 파일 경로 설정
# -----------------------------------------------------------------------------
# 사용자가 제공한 절대 경로 사용
file_path = r"C:\Users\seong\OneDrive\바탕 화면\bootcamp\03_code\data\snapshots\baseline_after_L7BCD\combined__baseline_after_L7BCD.parquet"

print(f"📂 파일 로딩 중: {file_path}")

try:
    # 2. 통합 파일 로드
    df = pd.read_parquet(file_path)
    print(f"✅ 로드 완료! 데이터 크기: {df.shape}")

    # 3. 포함된 아티팩트(산출물) 목록 확인
    # '__artifact' 컬럼이 각 행이 어떤 데이터인지 알려주는 '이름표' 역할을 합니다.
    artifacts = df["__artifact"].unique()
    print(f"📋 포함된 산출물 목록: {artifacts}")
    print("-" * 60)

    # -----------------------------------------------------------------------------
    # 4. 핵심 데이터 추출 및 분석 함수
    # -----------------------------------------------------------------------------
    def analyze_artifact(target_name, description):
        # 해당 아티팩트만 필터링
        subset = df[df["__artifact"] == target_name].copy()

        if subset.empty:
            return  # 해당 아티팩트가 없으면 패스

        # 해당 데이터에서 '모두 비어있는(NaN)' 컬럼은 제거 (보기 좋게)
        subset = subset.dropna(axis=1, how="all")

        print(f"\n🔎 [{target_name}] - {description}")

        # (A) 성과 지표 (metrics)인 경우: 전체 통계 출력
        if "metrics" in target_name:
            # 주요 지표 컬럼만 골라서 보여주기 (너무 많으므로)
            key_metrics = [
                "net_sharpe",
                "net_cagr",
                "net_mdd",
                "avg_turnover_oneway",
                "rmse",
                "mae",
                "hit_ratio",
                "ic_rank",
                "corr_vs_benchmark",
            ]
            # 존재하는 컬럼만 선택
            cols_to_show = [c for c in key_metrics if c in subset.columns]

            if cols_to_show:
                print("   [핵심 지표 요약]")
                # 평균값 또는 첫 번째 행 출력
                print(subset[cols_to_show].mean(numeric_only=True).to_frame().T)
            else:
                print(subset.head())

        # (B) 포지션(positions)인 경우: 최근 날짜 보유 종목 샘플
        elif "positions" in target_name and "date" in subset.columns:
            last_date = subset["date"].max()
            daily_pos = subset[subset["date"] == last_date]
            print(f"   📅 최근 거래일({last_date}) 보유 종목 수: {len(daily_pos)}개")
            print("   [상위 비중 5개 종목]")
            if "weight" in daily_pos.columns and "ticker" in daily_pos.columns:
                print(
                    daily_pos.sort_values("weight", ascending=False)[
                        ["ticker", "weight"]
                    ].head(5)
                )
            else:
                print(daily_pos.head())

        # (C) 스코어(scores)인 경우: 점수 분포 확인
        elif "score" in target_name:
            print(f"   📊 스코어 데이터 ({len(subset)} rows)")
            # 점수 컬럼이 있다면 기초 통계 출력
            score_cols = [c for c in subset.columns if "score" in c]
            if score_cols:
                print(subset[score_cols].describe().loc[["mean", "std", "min", "max"]])

        # (D) 기타: 상위 3줄만 출력
        else:
            print(subset.head(3))

        print("-" * 60)

    # -----------------------------------------------------------------------------
    # 5. 순차적 분석 실행 (프로젝트 흐름순)
    # -----------------------------------------------------------------------------

    # [L5] 모델 성능 확인: 예측이 얼마나 잘 맞았는가?
    # (로그 컬럼에 'ic_rank', 'rmse'가 있는 것으로 보아 'metrics'나 'model_metrics'에 저장됨)
    # 정확한 이름은 위 artifacts 목록 출력 결과를 보고 매칭해야 하지만,
    # 통상적인 이름인 'model_metrics' 또는 'metrics'를 찾아봅니다.
    for art in artifacts:
        if "model" in art and "metrics" in art:
            analyze_artifact(art, "L5 모델 예측 성능 (RMSE, IC)")

    # [L6] 스코어링 상태 확인: 점수가 안정적인가?
    for art in artifacts:
        if "score" in art and "summary" not in art:  # raw score
            analyze_artifact(art, "L6 리밸런싱 스코어 분포")

    # [L7] 백테스트 최종 성과: 돈을 벌었는가?
    # 보통 'bt_metrics' 또는 'bt_metrics_...'
    for art in artifacts:
        if "bt" in art and "metrics" in art:
            analyze_artifact(art, "L7 백테스트 최종 성과 (Sharpe, Turnover)")

    # [L7] 포지션 확인: 무엇을 샀는가?
    for art in artifacts:
        if "bt" in art and "positions" in art:
            analyze_artifact(art, "L7 보유 포지션 내역")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
