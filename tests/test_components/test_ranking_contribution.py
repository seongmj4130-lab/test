"""
랭킹 기여도 분석 컴포넌트 단위 테스트
"""

import pandas as pd
import pytest

from src.components.ranking.contribution_engine import (
    ContributionConfig,
    _zfill_ticker,
    infer_feature_group,
)


class TestZfillTicker:
    """_zfill_ticker 함수 테스트"""

    def test_zfill_ticker_basic(self):
        """기본적인 티커 0-fill 테스트"""
        series = pd.Series(["1", "12", "123", "1234"])
        result = _zfill_ticker(series)

        expected = pd.Series(["000001", "000012", "000123", "001234"])
        pd.testing.assert_series_equal(result, expected)

    def test_zfill_ticker_already_filled(self):
        """이미 6자리인 티커 테스트"""
        series = pd.Series(["000001", "000012"])
        result = _zfill_ticker(series)

        pd.testing.assert_series_equal(result, series)

    def test_zfill_ticker_empty(self):
        """빈 시리즈 테스트"""
        series = pd.Series([], dtype=str)
        result = _zfill_ticker(series)

        assert len(result) == 0
        assert result.dtype == object

    def test_zfill_ticker_numeric_input(self):
        """숫자형 입력 테스트"""
        series = pd.Series([1, 12, 123])
        result = _zfill_ticker(series)

        expected = pd.Series(["000001", "000012", "000123"])
        pd.testing.assert_series_equal(result, expected)


class TestInferFeatureGroup:
    """infer_feature_group 함수 테스트"""

    def test_infer_news_features(self):
        """news 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("news_sentiment") == "news"
        assert infer_feature_group("news_volume") == "news"
        assert infer_feature_group("NEWS_PRICE_IMPACT") == "news"  # 대소문자 무시

    def test_infer_profitability_features(self):
        """profitability 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("roe") == "profitability"
        assert infer_feature_group("net_income") == "profitability"
        assert infer_feature_group("ROE") == "profitability"  # 대소문자 무시

    def test_infer_value_features(self):
        """value 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("equity") == "value"
        assert infer_feature_group("total_liabilities") == "value"
        assert infer_feature_group("debt_ratio") == "value"

    def test_infer_technical_features(self):
        """technical 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("rsi") == "technical"
        assert infer_feature_group("macd") == "technical"
        assert infer_feature_group("bollinger_upper") == "technical"

    def test_infer_esg_features(self):
        """ESG 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("esg_score") == "esg"
        assert infer_feature_group("environmental_score") == "esg"
        assert infer_feature_group("social_score") == "esg"
        assert infer_feature_group("governance_score") == "esg"

    def test_infer_price_features(self):
        """price 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("price_momentum") == "price"
        assert infer_feature_group("price_trend") == "price"

    def test_infer_volatility_features(self):
        """volatility 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("volatility_20d") == "volatility"
        assert infer_feature_group("realized_vol") == "volatility"

    def test_infer_size_features(self):
        """size 관련 피처 그룹 추론 테스트"""
        assert infer_feature_group("market_cap") == "size"
        assert infer_feature_group("log_market_cap") == "size"

    def test_infer_other_features(self):
        """other 그룹으로 분류되는 피처들 테스트"""
        assert infer_feature_group("unknown_feature") == "other"
        assert infer_feature_group("custom_indicator") == "other"
        assert infer_feature_group("") == "other"  # 빈 문자열

    def test_infer_feature_group_case_insensitive(self):
        """대소문자 구분 없이 동작하는지 테스트"""
        assert infer_feature_group("RSI") == "technical"
        assert infer_feature_group("MACD") == "technical"
        assert infer_feature_group("EsG_ScOrE") == "esg"

    def test_infer_feature_group_numeric_input(self):
        """숫자 입력 테스트"""
        assert infer_feature_group(123) == "other"  # 숫자는 other로 분류

    def test_infer_feature_group_none_input(self):
        """None 입력 테스트"""
        assert infer_feature_group(None) == "other"  # None은 other로 분류


class TestContributionConfig:
    """ContributionConfig 데이터 클래스 테스트"""

    def test_contribution_config_creation(self):
        """기본 ContributionConfig 생성 테스트"""
        config = ContributionConfig(
            normalization_method="percentile",
            date_col="date",
            ticker_col="ticker",
            universe_col="in_universe",
        )

        assert config.normalization_method == "percentile"
        assert config.date_col == "date"
        assert config.ticker_col == "ticker"
        assert config.universe_col == "in_universe"
        assert config.sector_col is None
        assert config.use_sector_relative is False

    def test_contribution_config_with_optional_fields(self):
        """옵션 필드 포함 ContributionConfig 생성 테스트"""
        config = ContributionConfig(
            normalization_method="zscore",
            date_col="trade_date",
            ticker_col="symbol",
            universe_col="is_universe",
            sector_col="sector_name",
            use_sector_relative=True,
        )

        assert config.normalization_method == "zscore"
        assert config.date_col == "trade_date"
        assert config.ticker_col == "symbol"
        assert config.universe_col == "is_universe"
        assert config.sector_col == "sector_name"
        assert config.use_sector_relative is True

    def test_contribution_config_frozen(self):
        """frozen=True 동작 확인"""
        config = ContributionConfig(normalization_method="percentile")

        # 속성 변경 시도 (예외 발생해야 함)
        with pytest.raises(AttributeError):
            config.normalization_method = "zscore"

    def test_contribution_config_equality(self):
        """동등성 비교 테스트"""
        config1 = ContributionConfig(
            normalization_method="percentile", sector_col="sector"
        )

        config2 = ContributionConfig(
            normalization_method="percentile", sector_col="sector"
        )

        config3 = ContributionConfig(normalization_method="zscore", sector_col="sector")

        assert config1 == config2
        assert config1 != config3

    def test_contribution_config_hashable(self):
        """해시 가능성 테스트 (frozen=True이므로)"""
        config = ContributionConfig(normalization_method="percentile")

        # set에 추가 가능해야 함
        config_set = {config}
        assert len(config_set) == 1

        # 딕셔너리 키로 사용 가능해야 함
        config_dict = {config: "value"}
        assert config_dict[config] == "value"
