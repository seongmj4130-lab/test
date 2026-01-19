# -*- coding: utf-8 -*-
"""
피쳐 개선 적용 스크립트

기존 피쳐셋을 개선하고 새로운 파생 피쳐를 추가합니다.
단기 전략에 초점을 맞춰 IC 개선을 목표로 합니다.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import load_artifact, save_artifact
from src.features.feature_engineering import FeatureEngineer


def apply_price_improvements(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    가격 기반 피쳐 개선 적용 (간단 버전)

    기존 OHLC 피쳐들을 상대적 지표로 보완
    """
    df = panel_df.copy()
    print("🔧 가격 기반 피쳐 개선 적용 중...")

    added_features = 0

    # 그룹별 계산 (ticker별)
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask].copy()

        if len(ticker_data) < 20:  # 최소 기간 필요
            continue

        # 52주 최고/최저가 대비 가격 위치
        if 'close' in ticker_data.columns:
            ticker_data['close_to_52w_high'] = (
                ticker_data['close'] /
                ticker_data['close'].rolling(252, min_periods=60).max()
            )
            ticker_data['close_to_52w_low'] = (
                ticker_data['close'] /
                ticker_data['close'].rolling(252, min_periods=60).min()
            )

        # 일중 가격 위치
        if all(col in ticker_data.columns for col in ['close', 'high', 'low']):
            ticker_data['intraday_price_position'] = (
                (ticker_data['close'] - ticker_data['low']) /
                (ticker_data['high'] - ticker_data['low']).replace(0, np.nan)
            )

        # 결과 저장
        for col in ['close_to_52w_high', 'close_to_52w_low', 'intraday_price_position']:
            if col in ticker_data.columns:
                df.loc[mask, col] = ticker_data[col]
                added_features += 1

    # NaN 처리
    df['close_to_52w_high'] = df['close_to_52w_high'].fillna(0.5)
    df['close_to_52w_low'] = df['close_to_52w_low'].fillna(0.5)
    df['intraday_price_position'] = df['intraday_price_position'].fillna(0.5)

    print(f"🎯 가격 개선 완료: {added_features//len(df['ticker'].unique())}개 피쳐 추가")
    return df


def apply_momentum_improvements(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    모멘텀 피쳐 강화 적용 (간단 버전)

    기존 모멘텀 피쳐들을 가중 지표로 보완
    """
    df = panel_df.copy()
    print("🔧 모멘텀 피쳐 강화 적용 중...")

    added_features = 0

    # 그룹별 계산 (ticker별)
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask].copy()

        if len(ticker_data) < 126:  # 최소 6개월 데이터 필요
            continue

        # 3개월 모멘텀 가중 평균
        momentum_3m = ticker_data['close'] / ticker_data['close'].shift(63) - 1
        ticker_data['momentum_3m_ewm'] = momentum_3m.ewm(span=10).mean()

        # 6개월 모멘텀 가중 평균
        momentum_6m = ticker_data['close'] / ticker_data['close'].shift(126) - 1
        ticker_data['momentum_6m_ewm'] = momentum_6m.ewm(span=15).mean()

        # 변동성 조정 3개월 모멘텀
        if 'volatility_60d' in ticker_data.columns:
            vol_20d = ticker_data.get('volatility_20d', ticker_data['volatility_60d'])
            ticker_data['momentum_3m_vol_adj'] = momentum_3m * (1 + vol_20d)

        # 결과 저장
        for col in ['momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj']:
            if col in ticker_data.columns:
                df.loc[mask, col] = ticker_data[col]
                added_features += 1

    # NaN 처리
    df['momentum_3m_ewm'] = df['momentum_3m_ewm'].fillna(0)
    df['momentum_6m_ewm'] = df['momentum_6m_ewm'].fillna(0)
    df['momentum_3m_vol_adj'] = df['momentum_3m_vol_adj'].fillna(0)

    print(f"🎯 모멘텀 강화 완료: {added_features//max(1, len(df['ticker'].unique()))}개 피쳐 추가")
    return df


def apply_volatility_improvements(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    변동성 피쳐 개선 적용 (간단 버전)
    """
    df = panel_df.copy()
    print("🔧 변동성 피쳐 개선 적용 중...")

    # 수익률 계산
    if 'returns' not in df.columns and 'close' in df.columns:
        df['returns'] = df.groupby('ticker')['close'].pct_change()

    if 'returns' not in df.columns:
        print("⚠️ 수익률 데이터 없음, 변동성 개선 건너뜀")
        return df

    added_features = 0

    # 그룹별 계산
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask].copy()

        if len(ticker_data) < 60:
            continue

        # 변동성 비대칭도
        if 'volatility_60d' in ticker_data.columns:
            upside_vol = ticker_data['returns'].where(ticker_data['returns'] > 0, 0).rolling(60).std() * np.sqrt(252)
            vol_60d = ticker_data['volatility_60d']
            ticker_data['volatility_asymmetry'] = upside_vol / (vol_60d + 1e-8)

        # 꼬리 위험
        ticker_data['tail_risk_5pct'] = ticker_data['returns'].rolling(60).quantile(0.05)

        # 결과 저장
        for col in ['volatility_asymmetry', 'tail_risk_5pct']:
            if col in ticker_data.columns:
                df.loc[mask, col] = ticker_data[col]
                added_features += 1

    # NaN 처리
    df['volatility_asymmetry'] = df['volatility_asymmetry'].fillna(1.0)
    df['tail_risk_5pct'] = df['tail_risk_5pct'].fillna(0)

    print(f"🎯 변동성 개선 완료: {added_features//max(1, len(df['ticker'].unique()))}개 피쳐 추가")
    return df


def apply_news_improvements(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    뉴스 피쳐 강화 적용 (간단 버전)
    """
    df = panel_df.copy()
    print("🔧 뉴스 피쳐 강화 적용 중...")

    # 뉴스 관련 컬럼들 찾기
    news_cols = [col for col in df.columns if 'news' in col.lower()]
    if not news_cols:
        print("⚠️ 뉴스 데이터 없음, 뉴스 개선 건너뜀")
        return df

    added_features = 0

    # 그룹별 계산
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask].copy()

        if len(ticker_data) < 10:
            continue

        # 뉴스 감성 강도
        if 'news_sentiment' in ticker_data.columns:
            ticker_data['news_intensity'] = abs(ticker_data['news_sentiment'])

        # 뉴스 트렌드 (5일 EWM)
        if 'news_sentiment' in ticker_data.columns:
            ticker_data['news_trend'] = ticker_data['news_sentiment'].ewm(span=5).mean()

        # 결과 저장
        for col in ['news_intensity', 'news_trend']:
            if col in ticker_data.columns:
                df.loc[mask, col] = ticker_data[col]
                added_features += 1

    # NaN 처리
    df['news_intensity'] = df['news_intensity'].fillna(0)
    df['news_trend'] = df['news_trend'].fillna(0)

    print(f"🎯 뉴스 강화 완료: {added_features//max(1, len(df['ticker'].unique()))}개 피쳐 추가")
    return df


def create_feature_improvement_summary(original_df: pd.DataFrame, improved_df: pd.DataFrame):
    """
    피쳐 개선 전후 비교 요약
    """
    print("\n" + "="*60)
    print("📊 피쳐 개선 적용 결과 요약")
    print("="*60)

    original_cols = len(original_df.columns)
    improved_cols = len(improved_df.columns)
    new_features = improved_cols - original_cols

    print(f"원본 피쳐 수: {original_cols}")
    print(f"개선 후 피쳐 수: {improved_cols}")
    print(f"추가된 피쳐 수: {new_features}")

    # 개선된 피쳐 카테고리별 현황
    price_features = [col for col in improved_df.columns if any(x in col for x in ['52w', 'intraday', 'price_range'])]
    momentum_features = [col for col in improved_df.columns if 'momentum' in col and col not in original_df.columns]
    volatility_features = [col for col in improved_df.columns if any(x in col for x in ['asymmetry', 'tail_risk', 'regime']) and col not in original_df.columns]
    news_features = [col for col in improved_df.columns if 'news_' in col and col not in original_df.columns]

    print("\n카테고리별 추가 피쳐:")
    print(f"  가격 개선: {len(price_features)}개")
    print(f"  모멘텀 강화: {len(momentum_features)}개")
    print(f"  변동성 개선: {len(volatility_features)}개")
    print(f"  뉴스 강화: {len(news_features)}개")

    return {
        'original_features': original_cols,
        'improved_features': improved_cols,
        'new_features': new_features,
        'price_improvements': len(price_features),
        'momentum_improvements': len(momentum_features),
        'volatility_improvements': len(volatility_features),
        'news_improvements': len(news_features)
    }


def main():
    """
    피쳐 개선 적용 메인 함수
    """
    print("🚀 피쳐 개선 적용 시작")
    print("="*50)

    # 설정 로드
    cfg = load_config('configs/config.yaml')
    base_dir = Path(cfg['paths']['base_dir'])
    interim_dir = base_dir / 'data' / 'interim'
    reports_dir = base_dir / 'artifacts' / 'reports'

    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 데이터 로드
    print("\n[1/5] 데이터 로드 중...")
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')

    if panel_df is None:
        print("❌ 데이터 로드 실패")
        return

    print(f"✅ 데이터 로드 완료: {panel_df.shape[0]:,}행, {panel_df.shape[1]}열")
    original_df = panel_df.copy()

    # 가격 기반 개선 적용
    print("\n[2/5] 가격 기반 피쳐 개선 적용...")
    panel_df = apply_price_improvements(panel_df)

    # 모멘텀 강화 적용
    print("\n[3/5] 모멘텀 피쳐 강화 적용...")
    panel_df = apply_momentum_improvements(panel_df)

    # 변동성 개선 적용
    print("\n[4/5] 변동성 피쳐 개선 적용...")
    panel_df = apply_volatility_improvements(panel_df)

    # 뉴스 강화 적용
    print("\n[5/5] 뉴스 피쳐 강화 적용...")
    panel_df = apply_news_improvements(panel_df)

    # 개선 결과 요약
    summary = create_feature_improvement_summary(original_df, panel_df)

    # 개선된 데이터 저장
    improved_file = interim_dir / f'panel_merged_daily_improved_{timestamp}.parquet'
    save_artifact(panel_df, improved_file)
    print(f"\n💾 개선된 데이터 저장: {improved_file}")

    # 요약 리포트 저장
    summary_file = reports_dir / f'feature_improvements_applied_{timestamp}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("피쳐 개선 적용 결과\n")
        f.write("="*30 + "\n")
        f.write(f"적용 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"원본 피쳐 수: {summary['original_features']}\n")
        f.write(f"개선 후 피쳐 수: {summary['improved_features']}\n")
        f.write(f"추가된 피쳐 수: {summary['new_features']}\n")
        f.write(f"가격 개선 피쳐: {summary['price_improvements']}\n")
        f.write(f"모멘텀 강화 피쳐: {summary['momentum_improvements']}\n")
        f.write(f"변동성 개선 피쳐: {summary['volatility_improvements']}\n")
        f.write(f"뉴스 강화 피쳐: {summary['news_improvements']}\n")
        f.write("\n다음 단계: L5 모델 재학습 및 L8 랭킹 재생성 필요\n")

    print(f"\n📄 요약 리포트 저장: {summary_file}")

    print("\n" + "="*50)
    print("✅ 피쳐 개선 적용 완료!")
    print("="*50)
    print("다음 단계 제안:")
    print("1. python scripts/run_pipeline_l0_l7.py  # L5~L7 재실행")
    print("2. python scripts/measure_ranking_hit_ratio.py  # 성능 평가")
    print("3. IC 개선도 및 과적합 영향 분석")


if __name__ == "__main__":
    main()