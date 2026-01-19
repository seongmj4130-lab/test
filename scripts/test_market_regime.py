# -*- coding: utf-8 -*-
"""
시장 국면 판단 로직 테스트
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_market_regime():
    """시장 국면 판단 로직 테스트"""

    print("=" * 80)
    print("시장 국면 판단 로직 테스트")
    print("=" * 80)

    # 1. ohlcv_daily 데이터 로드
    data_dir = project_root / "data" / "interim"
    ohlcv_path = data_dir / "ohlcv_daily"

    if ohlcv_path.with_suffix(".parquet").exists():
        ohlcv = pd.read_parquet(ohlcv_path.with_suffix(".parquet"))
    elif ohlcv_path.with_suffix(".csv").exists():
        ohlcv = pd.read_csv(ohlcv_path.with_suffix(".csv"))
        if "date" in ohlcv.columns:
            ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
    else:
        print("❌ ohlcv_daily 파일이 없습니다.")
        return

    print(f"\n✓ ohlcv_daily 로드: {len(ohlcv):,}행")
    print(f"  컬럼: {list(ohlcv.columns)}")

    # 2. 데이터 전처리
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
    ohlcv = ohlcv.dropna(subset=["date", "ticker", "close"])
    ohlcv = ohlcv.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\n전처리 후: {len(ohlcv):,}행")
    print(f"  날짜 범위: {ohlcv['date'].min()} ~ {ohlcv['date'].max()}")
    print(f"  종목 수: {ohlcv['ticker'].nunique()}개")

    # 3. 수익률 계산 테스트 (수동으로)
    print(f"\n" + "=" * 80)
    print("수익률 계산 테스트 (수동)")
    print("=" * 80)

    # 샘플 날짜 선택
    sample_dates = sorted(ohlcv["date"].unique())[:10]
    print(f"\n샘플 날짜: {len(sample_dates)}개")

    for test_date in sample_dates[:3]:  # 처음 3개만 테스트
        print(f"\n--- 테스트 날짜: {test_date.strftime('%Y-%m-%d')} ---")

        # 해당 날짜의 데이터
        day_data = ohlcv[ohlcv["date"] == test_date].copy()
        print(f"  해당 날짜 종목 수: {len(day_data)}개")

        if len(day_data) == 0:
            print("  ⚠️  데이터 없음")
            continue

        # 전일 데이터 가져오기
        prev_date = test_date - pd.Timedelta(days=1)
        prev_data = ohlcv[ohlcv["date"] == prev_date].copy()

        if len(prev_data) == 0:
            print(f"  ⚠️  전일({prev_date.strftime('%Y-%m-%d')}) 데이터 없음")
            continue

        # 종목별로 전일 종가 매칭
        merged = day_data[["ticker", "close"]].merge(
            prev_data[["ticker", "close"]],
            on="ticker",
            how="inner",
            suffixes=("", "_prev")
        )

        if len(merged) == 0:
            print("  ⚠️  매칭되는 종목 없음")
            continue

        # 수익률 계산
        merged["daily_return"] = (merged["close"] - merged["close_prev"]) / merged["close_prev"]
        valid_returns = merged["daily_return"].dropna()

        print(f"  매칭된 종목 수: {len(merged)}개")
        print(f"  유효한 수익률: {len(valid_returns)}개")
        if len(valid_returns) > 0:
            print(f"  평균 수익률: {valid_returns.mean()*100:.2f}%")
            print(f"  수익률 범위: {valid_returns.min()*100:.2f}% ~ {valid_returns.max()*100:.2f}%")
        else:
            print("  ⚠️  유효한 수익률 없음")

    # 4. build_market_regime 함수 테스트
    print(f"\n" + "=" * 80)
    print("build_market_regime 함수 테스트")
    print("=" * 80)

    # rebalance_dates 준비 (cv_folds_short에서 가져오기)
    cv_path = data_dir / "cv_folds_short"
    if cv_path.with_suffix(".parquet").exists():
        cv = pd.read_parquet(cv_path.with_suffix(".parquet"))
    elif cv_path.with_suffix(".csv").exists():
        cv = pd.read_csv(cv_path.with_suffix(".csv"))
        if "test_end" in cv.columns:
            cv["test_end"] = pd.to_datetime(cv["test_end"], errors="coerce")
    else:
        # 테스트용 날짜 생성
        test_dates = sorted(ohlcv["date"].unique())[60:70]  # lookback_days=60을 고려
        cv = pd.DataFrame({"test_end": test_dates})
        print(f"⚠️  cv_folds_short 없음, 테스트용 날짜 사용: {len(test_dates)}개")

    if "test_end" in cv.columns:
        rebalance_dates = pd.to_datetime(cv["test_end"]).dropna().unique()[:10]  # 처음 10개만
    else:
        rebalance_dates = sorted(ohlcv["date"].unique())[60:70]

    print(f"\n테스트할 rebalance_dates: {len(rebalance_dates)}개")
    print(f"  범위: {rebalance_dates[0]} ~ {rebalance_dates[-1]}")

    try:
        regime_df = build_market_regime(
            rebalance_dates=rebalance_dates,
            ohlcv_daily=ohlcv,
            lookback_days=60,
            neutral_band=0.05,
            use_volume=True,
            use_volatility=True,
        )

        print(f"\n✓ build_market_regime 실행 완료")
        print(f"\n결과:")
        print(regime_df.to_string())

        print(f"\n국면 분포:")
        regime_counts = regime_df["regime"].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count}개 ({count/len(regime_df)*100:.1f}%)")

        # 중립만 있는지 확인
        if len(regime_counts) == 1 and "neutral" in regime_counts:
            print(f"\n⚠️  모든 날짜가 neutral로 분류되었습니다!")
            print(f"  수익률 계산에 문제가 있을 수 있습니다.")

            # 디버깅: 수익률 값 확인
            print(f"\n디버깅 정보:")
            print(f"  lookback_return_pct 범위:")
            print(f"    최소: {regime_df['lookback_return_pct'].min():.2f}%")
            print(f"    최대: {regime_df['lookback_return_pct'].max():.2f}%")
            print(f"    평균: {regime_df['lookback_return_pct'].mean():.2f}%")
            print(f"    NaN 개수: {regime_df['lookback_return_pct'].isna().sum()}개")
        else:
            print(f"\n✓ 다양한 국면으로 분류되었습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_market_regime()
