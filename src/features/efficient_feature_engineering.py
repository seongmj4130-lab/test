# -*- coding: utf-8 -*-
"""
메모리 효율적인 피쳐 엔지니어링 모듈

배치 처리와 점진적 적용을 통해 메모리 사용을 최적화합니다.
기존 피쳐를 개선하고 새로운 파생 피쳐를 생성합니다.
"""

import gc
import warnings
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class EfficientFeatureEngineer:
    """
    메모리 효율적인 피쳐 엔지니어링 클래스

    배치 처리와 점진적 적용으로 메모리 사용 최적화
    """

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.new_features = []

    def batch_process_price_features(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        배치 단위로 가격 기반 피쳐 개선 적용

        Args:
            panel_df: 전체 패널 데이터

        Returns:
            가격 피쳐가 개선된 데이터프레임
        """
        print("🔧 배치 단위 가격 피쳐 개선 적용 중...")

        # 필요한 컬럼만 선택
        price_cols = ['date', 'ticker', 'open', 'high', 'low', 'close']
        available_cols = [col for col in price_cols if col in panel_df.columns]

        if not available_cols:
            print("⚠️ 가격 데이터가 없어 가격 개선 건너뜀")
            return panel_df

        # 결과를 저장할 새 컬럼들
        new_columns = {}

        # 티커별 배치 처리
        tickers = panel_df['ticker'].unique()
        processed_tickers = 0

        for i in range(0, len(tickers), self.batch_size):
            batch_tickers = tickers[i:i + self.batch_size]
            batch_mask = panel_df['ticker'].isin(batch_tickers)
            batch_data = panel_df[batch_mask].copy()

            print(f"  배치 {i//self.batch_size + 1}: {len(batch_tickers)}개 티커 처리 중...")

            # 배치 내 티커별 처리
            for ticker in batch_tickers:
                ticker_mask = batch_data['ticker'] == ticker
                ticker_data = batch_data[ticker_mask].copy()

                if len(ticker_data) < 20:  # 최소 기간 필요
                    continue

                # 52주 최고/최저가 대비 가격 위치
                if 'close' in ticker_data.columns:
                    close_series = ticker_data['close']
                    ticker_data['close_to_52w_high'] = (
                        close_series / close_series.rolling(252, min_periods=60).max()
                    )
                    ticker_data['close_to_52w_low'] = (
                        close_series / close_series.rolling(252, min_periods=60).min()
                    )

                # 일중 가격 위치
                if all(col in ticker_data.columns for col in ['close', 'high', 'low']):
                    ticker_data['intraday_price_position'] = (
                        (ticker_data['close'] - ticker_data['low']) /
                        (ticker_data['high'] - ticker_data['low']).replace(0, np.nan)
                    )

                # 결과를 new_columns에 저장
                for col in ['close_to_52w_high', 'close_to_52w_low', 'intraday_price_position']:
                    if col in ticker_data.columns:
                        col_key = f"{ticker}_{col}"
                        new_columns[col_key] = ticker_data[col].values

            processed_tickers += len(batch_tickers)

            # 메모리 정리
            del batch_data
            gc.collect()

            if processed_tickers % 500 == 0:
                print(f"    진행률: {processed_tickers}/{len(tickers)} 티커 완료")

        # 새로운 컬럼들을 panel_df에 추가
        added_features = 0
        for col_name in ['close_to_52w_high', 'close_to_52w_low', 'intraday_price_position']:
            if any(col_name in key for key in new_columns.keys()):
                # 컬럼별로 데이터를 모아서 추가
                col_data = []
                for ticker in tickers:
                    col_key = f"{ticker}_{col_name}"
                    if col_key in new_columns:
                        col_data.extend(new_columns[col_key])
                    else:
                        # 데이터가 없는 경우 NaN으로 채움
                        ticker_size = len(panel_df[panel_df['ticker'] == ticker])
                        col_data.extend([np.nan] * ticker_size)

                panel_df[col_name] = col_data
                added_features += 1

        # NaN 처리
        panel_df['close_to_52w_high'] = panel_df['close_to_52w_high'].fillna(0.5)
        panel_df['close_to_52w_low'] = panel_df['close_to_52w_low'].fillna(0.5)
        panel_df['intraday_price_position'] = panel_df['intraday_price_position'].fillna(0.5)

        self.new_features.extend(['close_to_52w_high', 'close_to_52w_low', 'intraday_price_position'])
        print(f"🎯 배치 가격 개선 완료: {added_features}개 피쳐 추가")

        return panel_df

    def batch_process_momentum_features(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        배치 단위로 모멘텀 피쳐 강화 적용
        """
        print("🔧 배치 단위 모멘텀 피쳐 강화 적용 중...")

        if 'close' not in panel_df.columns:
            print("⚠️ 가격 데이터 없음, 모멘텀 개선 건너뜀")
            return panel_df

        # 티커별 배치 처리
        tickers = panel_df['ticker'].unique()
        processed_tickers = 0

        # 결과를 저장할 딕셔너리
        new_columns = {}

        for i in range(0, len(tickers), self.batch_size):
            batch_tickers = tickers[i:i + self.batch_size]
            batch_mask = panel_df['ticker'].isin(batch_tickers)
            batch_data = panel_df[batch_mask].copy()

            for ticker in batch_tickers:
                ticker_mask = batch_data['ticker'] == ticker
                ticker_data = batch_data[ticker_mask].copy()

                if len(ticker_data) < 126:  # 최소 6개월 데이터
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

                # 결과를 저장
                for col in ['momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj']:
                    if col in ticker_data.columns:
                        col_key = f"{ticker}_{col}"
                        new_columns[col_key] = ticker_data[col].values

            processed_tickers += len(batch_tickers)
            del batch_data
            gc.collect()

        # 새로운 컬럼들을 추가
        added_features = 0
        for col_name in ['momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj']:
            if any(col_name in key for key in new_columns.keys()):
                col_data = []
                for ticker in tickers:
                    col_key = f"{ticker}_{col_name}"
                    if col_key in new_columns:
                        col_data.extend(new_columns[col_key])
                    else:
                        ticker_size = len(panel_df[panel_df['ticker'] == ticker])
                        col_data.extend([np.nan] * ticker_size)

                panel_df[col_name] = col_data
                added_features += 1

        # NaN 처리
        panel_df['momentum_3m_ewm'] = panel_df['momentum_3m_ewm'].fillna(0)
        panel_df['momentum_6m_ewm'] = panel_df['momentum_6m_ewm'].fillna(0)
        panel_df['momentum_3m_vol_adj'] = panel_df['momentum_3m_vol_adj'].fillna(0)

        self.new_features.extend(['momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj'])
        print(f"🎯 배치 모멘텀 강화 완료: {added_features}개 피쳐 추가")

        return panel_df

    def validate_feature_addition(self, original_df: pd.DataFrame, improved_df: pd.DataFrame) -> Dict:
        """
        피쳐 추가 검증

        Args:
            original_df: 원본 데이터프레임
            improved_df: 개선된 데이터프레임

        Returns:
            검증 결과 딕셔너리
        """
        validation_results = {
            'original_features': len(original_df.columns),
            'improved_features': len(improved_df.columns),
            'added_features': len(improved_df.columns) - len(original_df.columns),
            'new_feature_names': [col for col in improved_df.columns if col not in original_df.columns],
            'data_integrity': True,
            'nan_check': {}
        }

        # NaN 비율 체크
        for col in validation_results['new_feature_names']:
            nan_ratio = improved_df[col].isnull().mean()
            validation_results['nan_check'][col] = nan_ratio
            if nan_ratio > 0.5:  # 50% 이상 NaN이면 문제
                validation_results['data_integrity'] = False

        return validation_results

    def get_new_features_list(self) -> List[str]:
        """생성된 새 피쳐 목록 반환"""
        return self.new_features.copy()


def test_efficient_feature_engineering():
    """메모리 효율적 피쳐 엔지니어링 테스트"""
    from pathlib import Path

    from src.utils.config import load_config
    from src.utils.io import load_artifact

    # 설정 로드
    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # 작은 샘플로 테스트
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')

    # 처음 10개 티커만 테스트
    test_tickers = panel_df['ticker'].unique()[:10]
    test_df = panel_df[panel_df['ticker'].isin(test_tickers)].copy()

    print(f"테스트 데이터: {len(test_df)}행, {len(test_df.columns)}열")
    print(f"테스트 티커: {len(test_tickers)}개")

    # 효율적 피쳐 엔지니어링 테스트
    engineer = EfficientFeatureEngineer(batch_size=5)  # 작은 배치로 테스트

    original_features = len(test_df.columns)

    # 가격 피쳐 개선 적용
    test_df = engineer.batch_process_price_features(test_df)
    price_added = len(test_df.columns) - original_features

    # 모멘텀 피쳐 강화 적용
    test_df = engineer.batch_process_momentum_features(test_df)
    momentum_added = len(test_df.columns) - original_features - price_added

    print(f"최종 피쳐 수: {len(test_df.columns)} (원본: {original_features})")
    print(f"가격 피쳐 추가: {price_added}개")
    print(f"모멘텀 피쳐 추가: {momentum_added}개")
    print(f"생성된 피쳐들: {engineer.get_new_features_list()}")

    return test_df


if __name__ == "__main__":
    test_efficient_feature_engineering()
