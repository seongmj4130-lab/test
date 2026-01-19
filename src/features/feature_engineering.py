# -*- coding: utf-8 -*-
"""
피쳐 엔지니어링 모듈

기존 피쳐를 개선하고 새로운 파생 피쳐를 생성합니다.
모든 피쳐는 벡터화 구현으로 성능 최적화.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    피쳐 엔지니어링 클래스

    기존 피쳐 개선 및 새로운 파생 피쳐 생성
    """

    def __init__(self):
        self.new_features = []

    def create_price_relative_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        가격 기반 피쳐 개선: 절대 가격 → 상대적 지표로 변환

        Args:
            ohlcv_df: OHLCV 데이터프레임 (date, ticker, open, high, low, close, volume)

        Returns:
            개선된 가격 피쳐들
        """
        df = ohlcv_df.copy()
        features_created = []

        # 그룹별 계산을 위한 정렬
        df = df.sort_values(['ticker', 'date'])

        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()

            # 52주 최고/최저가 대비 가격 위치
            ticker_data['close_to_52w_high'] = (
                ticker_data['close'] /
                ticker_data['close'].rolling(252, min_periods=60).max()
            )

            ticker_data['close_to_52w_low'] = (
                ticker_data['close'] /
                ticker_data['close'].rolling(252, min_periods=60).min()
            )

            # 일중 가격 위치 (0~1)
            ticker_data['intraday_price_position'] = (
                (ticker_data['close'] - ticker_data['low']) /
                (ticker_data['high'] - ticker_data['low']).replace(0, np.nan)
            )

            # 가격 변동폭 비율
            ticker_data['price_range_ratio'] = (
                (ticker_data['high'] - ticker_data['low']) /
                ticker_data['close'].shift(1)
            )

            # 일중 상승폭/하락폭 비율
            ticker_data['intraday_up_ratio'] = (
                (ticker_data['close'] - ticker_data['open']) /
                (ticker_data['high'] - ticker_data['low']).replace(0, np.nan)
            )

            # 결과 저장
            df.loc[mask, ticker_data.columns] = ticker_data

            features_created.extend([
                'close_to_52w_high', 'close_to_52w_low',
                'intraday_price_position', 'price_range_ratio', 'intraday_up_ratio'
            ])

        # NaN 처리 및 결과 반환
        result_df = df[['date', 'ticker']].copy()
        for feature in features_created:
            if feature in df.columns:
                result_df[feature] = df[feature].fillna(0.5)  # 중립값

        self.new_features.extend(features_created)
        print(f"✅ 가격 상대적 피쳐 생성 완료: {len(features_created)}개")

        return result_df[features_created]

    def create_momentum_enhanced_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        모멘텀 피쳐 강화: 단순 수익률 → 가중/가속도/조정 지표

        Args:
            prices_df: 가격 데이터 (date, ticker, close)

        Returns:
            강화된 모멘텀 피쳐들
        """
        df = prices_df.copy()
        features_created = []

        # 그룹별 계산
        df = df.sort_values(['ticker', 'date'])

        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()

            # 기본 수익률 계산
            ticker_data['returns'] = ticker_data['close'].pct_change()

            # 3개월 모멘텀 강화
            momentum_3m = ticker_data['close'] / ticker_data['close'].shift(63) - 1
            ticker_data['momentum_3m_ewm'] = momentum_3m.ewm(span=10).mean()
            ticker_data['momentum_3m_accel'] = momentum_3m - momentum_3m.shift(21)

            # 6개월 모멘텀 강화
            momentum_6m = ticker_data['close'] / ticker_data['close'].shift(126) - 1
            ticker_data['momentum_6m_ewm'] = momentum_6m.ewm(span=15).mean()
            ticker_data['momentum_6m_accel'] = momentum_6m - momentum_6m.shift(21)

            # 변동성 조정 모멘텀 (20일 변동성 사용)
            vol_20d = ticker_data['returns'].rolling(20, min_periods=5).std() * np.sqrt(252)
            ticker_data['momentum_3m_vol_adj'] = momentum_3m * (1 + vol_20d)
            ticker_data['momentum_6m_vol_adj'] = momentum_6m * (1 + vol_20d)

            # 모멘텀 지속성 지표
            ticker_data['momentum_persistence'] = (
                momentum_3m.rolling(10).corr(momentum_6m)
            )

            df.loc[mask, ticker_data.columns] = ticker_data

            features_created.extend([
                'momentum_3m_ewm', 'momentum_3m_accel',
                'momentum_6m_ewm', 'momentum_6m_accel',
                'momentum_3m_vol_adj', 'momentum_6m_vol_adj',
                'momentum_persistence'
            ])

        # NaN 처리 및 결과 반환
        result_df = df[['date', 'ticker']].copy()
        for feature in features_created:
            if feature in df.columns:
                result_df[feature] = df[feature].fillna(0)

        self.new_features.extend(features_created)
        print(f"✅ 모멘텀 강화 피쳐 생성 완료: {len(features_created)}개")

        return result_df[features_created]

    def create_volatility_enhanced_features(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        변동성 피쳐 개선: 단방향 → 다각적 변동성 지표

        Args:
            returns_df: 수익률 데이터 (date, ticker, returns)

        Returns:
            개선된 변동성 피쳐들
        """
        df = returns_df.copy()
        features_created = []

        # 그룹별 계산
        df = df.sort_values(['ticker', 'date'])

        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()

            # 기존 변동성
            vol_60d = ticker_data['returns'].rolling(60, min_periods=20).std() * np.sqrt(252)

            # 상방/하방 변동성 분리
            upside_returns = ticker_data['returns'].where(ticker_data['returns'] > 0, 0)
            downside_returns = ticker_data['returns'].where(ticker_data['returns'] < 0, 0)

            ticker_data['upside_volatility_60d'] = (
                upside_returns.rolling(60, min_periods=20).std() * np.sqrt(252)
            )
            ticker_data['volatility_asymmetry'] = (
                ticker_data['upside_volatility_60d'] / (vol_60d + 1e-8)
            )

            # 꼬리 위험 지표
            ticker_data['tail_risk_5pct'] = (
                ticker_data['returns'].rolling(60, min_periods=20).quantile(0.05)
            )
            ticker_data['tail_risk_95pct'] = (
                ticker_data['returns'].rolling(60, min_periods=20).quantile(0.95)
            )

            # 변동성 체제 (60일 vs 252일 평균)
            vol_252d = ticker_data['returns'].rolling(252, min_periods=60).std() * np.sqrt(252)
            ticker_data['volatility_regime'] = vol_60d / (vol_252d + 1e-8)

            # 변동성 변화율
            ticker_data['volatility_momentum'] = vol_60d / vol_60d.shift(20) - 1

            df.loc[mask, ticker_data.columns] = ticker_data

            features_created.extend([
                'upside_volatility_60d', 'volatility_asymmetry',
                'tail_risk_5pct', 'tail_risk_95pct',
                'volatility_regime', 'volatility_momentum'
            ])

        # NaN 처리 및 결과 반환
        result_df = df[['date', 'ticker']].copy()
        for feature in features_created:
            if feature in df.columns:
                result_df[feature] = df[feature].fillna(0)

        self.new_features.extend(features_created)
        print(f"✅ 변동성 강화 피쳐 생성 완료: {len(features_created)}개")

        return result_df[features_created]

    def create_news_enhanced_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        뉴스 피쳐 강화: 감성 분석 심화

        Args:
            news_df: 뉴스 데이터 (date, ticker, news_sentiment, news_volume 등)

        Returns:
            강화된 뉴스 피쳐들
        """
        df = news_df.copy()
        features_created = []

        # 그룹별 계산
        df = df.sort_values(['ticker', 'date'])

        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()

            # 감성 강도 (감성 × 거래량)
            ticker_data['news_intensity'] = (
                abs(ticker_data.get('news_sentiment', 0)) *
                ticker_data.get('news_volume', 1)
            )

            # 감성 일관성 (5일 rolling std)
            ticker_data['news_consistency'] = (
                ticker_data.get('news_sentiment', 0).rolling(5, min_periods=1).std()
            )

            # 뉴스 트렌드 (20일 EWM)
            ticker_data['news_trend'] = (
                ticker_data.get('news_sentiment', 0).ewm(span=20).mean()
            )

            # 뉴스 지속성 (60일 EWM)
            ticker_data['news_persistence'] = (
                ticker_data.get('news_sentiment', 0).ewm(span=60).mean()
            )

            df.loc[mask, ticker_data.columns] = ticker_data

            features_created.extend([
                'news_intensity', 'news_consistency',
                'news_trend', 'news_persistence'
            ])

        # NaN 처리 및 결과 반환
        result_df = df[['date', 'ticker']].copy()
        for feature in features_created:
            if feature in df.columns:
                result_df[feature] = df[feature].fillna(0)

        self.new_features.extend(features_created)
        print(f"✅ 뉴스 강화 피쳐 생성 완료: {len(features_created)}개")

        return result_df[features_created]

    def create_interaction_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        피쳐 상호작용 생성: 재무 + 기술 결합

        Args:
            feature_df: 기존 피쳐 데이터프레임

        Returns:
            상호작용 피쳐들
        """
        df = feature_df.copy()
        features_created = []

        # 재무 + 변동성 결합
        if 'roe' in df.columns and 'volatility_60d' in df.columns:
            df['roe_volatility_adj'] = df['roe'] / (df['volatility_60d'] + 1e-8)
            features_created.append('roe_volatility_adj')

        # 가치 + 모멘텀 결합
        if 'per' in df.columns and 'momentum_3m' in df.columns:
            df['value_momentum_score'] = (
                self._rank_normalize(df['per']) +
                self._rank_normalize(df['momentum_3m'])
            ) / 2
            features_created.append('value_momentum_score')

        # 뉴스 + 가격 정렬도
        if 'news_sentiment' in df.columns and 'momentum_3m' in df.columns:
            df['news_price_alignment'] = (
                df['news_sentiment'] * df['momentum_3m']
            )
            features_created.append('news_price_alignment')

        # NaN 처리 및 결과 반환
        result_df = df[['date', 'ticker']].copy()
        for feature in features_created:
            if feature in df.columns:
                result_df[feature] = df[feature].fillna(0)

        self.new_features.extend(features_created)
        print(f"✅ 피쳐 상호작용 생성 완료: {len(features_created)}개")

        return result_df[features_created]

    def _rank_normalize(self, series: pd.Series) -> pd.Series:
        """랭킹 정규화 헬퍼 함수"""
        return (series.rank() - series.rank().mean()) / series.rank().std()

    def get_new_features_list(self) -> List[str]:
        """생성된 새 피쳐 목록 반환"""
        return self.new_features.copy()

    def apply_all_enhancements(
        self,
        ohlcv_df: pd.DataFrame,
        fundamental_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        모든 피쳐 개선 적용

        Args:
            ohlcv_df: OHLCV 데이터
            fundamental_df: 재무 데이터 (선택)
            news_df: 뉴스 데이터 (선택)

        Returns:
            개선된 모든 피쳐들
        """
        print("🚀 피쳐 엔지니어링 시작...")

        all_features = []

        # 1. 가격 기반 피쳐 개선
        if not ohlcv_df.empty:
            price_features = self.create_price_relative_features(ohlcv_df)
            all_features.append(price_features)

        # 2. 모멘텀 피쳐 강화
        if not ohlcv_df.empty and 'close' in ohlcv_df.columns:
            momentum_features = self.create_momentum_enhanced_features(
                ohlcv_df[['date', 'ticker', 'close']]
            )
            all_features.append(momentum_features)

        # 3. 변동성 피쳐 개선
        if not ohlcv_df.empty:
            returns_df = ohlcv_df.copy()
            if 'close' in returns_df.columns:
                returns_df['returns'] = returns_df.groupby('ticker')['close'].pct_change()

            volatility_features = self.create_volatility_enhanced_features(
                returns_df[['date', 'ticker', 'returns']]
            )
            all_features.append(volatility_features)

        # 4. 뉴스 피쳐 강화
        if news_df is not None and not news_df.empty:
            news_features = self.create_news_enhanced_features(news_df)
            all_features.append(news_features)

        # 5. 피쳐 상호작용 (기존 피쳐들과 결합)
        if fundamental_df is not None and not fundamental_df.empty:
            interaction_features = self.create_interaction_features(fundamental_df)
            all_features.append(interaction_features)

        # 결과 합치기
        if all_features:
            result_df = pd.concat(all_features, axis=1)

            # date, ticker 추가 (첫 번째 df에서 가져옴)
            if all_features[0] is not None and not all_features[0].empty:
                result_df = result_df.join(
                    all_features[0][['date', 'ticker']],
                    how='left'
                )

            print(f"✅ 총 {len(self.new_features)}개 피쳐 생성 완료")
            return result_df

        return pd.DataFrame()


# 테스트 함수
def test_feature_engineering():
    """피쳐 엔지니어링 테스트"""
    import sys
    sys.path.append('.')
    from src.utils.config import load_config
    from src.utils.io import load_artifact

    # 데이터 로드
    cfg = load_config('configs/config.yaml')
    interim_dir = cfg['paths']['base_dir'] / 'data' / 'interim'

    ohlcv_df = load_artifact(interim_dir / 'ohlcv_daily')
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')

    # 피쳐 엔지니어링 적용
    engineer = FeatureEngineer()
    enhanced_features = engineer.apply_all_enhancements(
        ohlcv_df=ohlcv_df,
        fundamental_df=panel_df
    )

    print(f"생성된 피쳐들: {engineer.get_new_features_list()}")
    print(f"피쳐 데이터 shape: {enhanced_features.shape}")

    return enhanced_features


if __name__ == "__main__":
    test_feature_engineering()
