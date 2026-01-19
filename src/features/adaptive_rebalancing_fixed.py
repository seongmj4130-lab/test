# -*- coding: utf-8 -*-
"""
적응형 리밸런싱 모듈 - SyntaxError 수정 버전

bt20 프로페셔널 전략을 위한 적응형 리밸런싱 시스템
시그널 강도에 따라 리밸런싱 간격을 15-25일 사이에서 동적으로 조정
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveRebalancing:
    """
    적응형 리밸런싱 클래스

    시그널 강도(롤링 IC 기반)에 따라 리밸런싱 간격을 동적으로 조정:
    - 강한 시그널 (0.8+): 15일 리밸런싱
    - 중간 시그널 (0.6-0.8): 20일 리밸런싱
    - 약한 시그널 (0.6-): 25일 리밸런싱
    """

    def __init__(self, config: Dict[str, Any]):
        """
        초기화

        Args:
            config: 설정 딕셔너리
                - min_rebalance_days: 최소 리밸런싱 간격 (기본: 15)
                - max_rebalance_days: 최대 리밸런싱 간격 (기본: 25)
                - default_rebalance_days: 기본 리밸런싱 간격 (기본: 20)
                - signal_strength_threshold_high: 강한 시그널 임계값 (기본: 0.8)
                - signal_strength_threshold_low: 약한 시그널 임계값 (기본: 0.6)
                - ic_lookback_window: IC 계산 윈도우 (기본: 60일)
                - ic_min_periods: 최소 IC 계산 기간 (기본: 20)
        """
        self.config = config
        self.min_rebalance_days = config.get('min_rebalance_days', 15)
        self.max_rebalance_days = config.get('max_rebalance_days', 25)
        self.default_rebalance_days = config.get('default_rebalance_days', 20)
        self.signal_strength_threshold_high = config.get('signal_strength_threshold_high', 0.8)
        self.signal_strength_threshold_low = config.get('signal_strength_threshold_low', 0.6)
        self.ic_lookback_window = config.get('ic_lookback_window', 60)
        self.ic_min_periods = config.get('ic_min_periods', 20)

        logger.info(f"적응형 리밸런싱 초기화: {self.min_rebalance_days}-{self.max_rebalance_days}일 범위")

    def calculate_rolling_ic(self, scores_df: pd.DataFrame) -> pd.Series:
        """
        롤링 IC (Information Coefficient) 계산

        Args:
            scores_df: 스코어 데이터프레임 (date, ticker, score_total_short, true_short 컬럼 필요)

        Returns:
            rolling_ic: 날짜별 롤링 IC 시리즈
        """
        if scores_df.empty or 'score_total_short' not in scores_df.columns or 'true_short' not in scores_df.columns:
            logger.warning("스코어 데이터프레임이 비어있거나 필요한 컬럼이 없습니다.")
            return pd.Series(dtype=float)

        # 날짜 정렬 및 datetime 변환
        scores_df = scores_df.copy()
        scores_df['date'] = pd.to_datetime(scores_df['date'])
        scores_df = scores_df.sort_values(by=['date', 'ticker'])

        # 일별 IC 계산 (스피어만 상관계수)
        try:
            daily_ic = scores_df.groupby('date').apply(
                lambda x: x['score_total_short'].corr(x['true_short'], method='spearman')
            )
            daily_ic.name = 'daily_ic'
        except Exception as e:
            logger.warning(f"일별 IC 계산 실패: {e}")
            return pd.Series(dtype=float)

        # 롤링 평균 IC 계산
        rolling_ic = daily_ic.rolling(
            window=self.ic_lookback_window,
            min_periods=self.ic_min_periods
        ).mean()

        # IC 계산 결과 로깅
        logger.info(f"IC 계산 결과 - 일별 IC 범위: {daily_ic.min():.3f} ~ {daily_ic.max():.3f}")
        logger.info(f"IC 계산 결과 - 일별 IC 평균: {daily_ic.mean():.3f}, 표준편차: {daily_ic.std():.3f}")
        logger.info(f"IC 계산 결과 - 롤링 IC 범위: {rolling_ic.min():.3f} ~ {rolling_ic.max():.3f}")
        logger.info(f"IC 계산 결과 - 롤링 IC 평균: {rolling_ic.mean():.3f}")
        logger.info(f"IC 계산 결과 - 유효 IC 개수: {rolling_ic.notna().sum()}/{len(rolling_ic)}")

        return rolling_ic

    def get_signal_strength_score(self, rolling_ic: pd.Series) -> pd.Series:
        """
        롤링 IC를 시그널 강도 점수로 변환 (0-100 스케일)

        Args:
            rolling_ic: 롤링 IC 시리즈

        Returns:
            signal_scores: 시그널 강도 점수 (0-100)
        """
        if rolling_ic.empty:
            return pd.Series(dtype=float)

        # IC를 0-100 스케일로 정규화
        # IC 범위: -1 ~ 1 → 강도 점수: 0 ~ 100
        signal_scores = (rolling_ic + 1) / 2 * 100

        # NaN은 기본값 50점으로 설정
        signal_scores = signal_scores.fillna(50)

        # 시그널 강도 계산 결과 로깅
        logger.info(f"시그널 강도 계산 - 범위: {signal_scores.min():.1f} ~ {signal_scores.max():.1f}")
        logger.info(f"시그널 강도 계산 - 평균: {signal_scores.mean():.1f}, 표준편차: {signal_scores.std():.1f}")
        logger.info(f"시그널 강도 계산 - 분포: 강함({(signal_scores >= 80).sum()}), 중간({((signal_scores >= 60) & (signal_scores < 80)).sum()}), 약함({(signal_scores < 60).sum()})")

        return signal_scores

    def determine_rebalance_interval(self, ic_value: float) -> int:
        """
        IC 값에 따른 리밸런싱 간격 결정

        IC가 양수이고 높을수록 강한 시그널 (빈번한 리밸런싱)
        IC가 음수이거나 낮을수록 약한 시그널 (드문 리밸런싱)

        Args:
            ic_value: IC 값 (-1 ~ 1)

        Returns:
            interval: 리밸런싱 간격 (일)
        """
        if ic_value >= 0.01:  # 강한 양의 IC
            return self.min_rebalance_days  # 15일
        elif ic_value >= -0.01:  # 중간 IC (약한 양수 또는 약한 음수)
            return self.default_rebalance_days  # 20일
        else:  # 강한 음의 IC
            return self.max_rebalance_days  # 25일

    def get_adaptive_rebalance_intervals(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 기간에 대한 적응형 리밸런싱 간격 계산

        Args:
            scores_df: 스코어 데이터프레임

        Returns:
            intervals_df: 날짜별 리밸런싱 간격 데이터프레임
        """
        if scores_df.empty:
            return pd.DataFrame(columns=['date', 'rebalance_interval'])

        logger.info("적응형 리밸런싱 간격 계산 시작...")

        # 롤링 IC 계산
        rolling_ic = self.calculate_rolling_ic(scores_df)

        if rolling_ic.empty:
            logger.warning("롤링 IC 계산 실패, 기본 간격 사용")
            dates = pd.to_datetime(scores_df['date'].unique())
            return pd.DataFrame({
                'date': dates,
                'rebalance_interval': self.default_rebalance_days
            })

        # 시그널 강도 계산
        signal_scores = self.get_signal_strength_score(rolling_ic)

        # 각 날짜별 리밸런싱 간격 결정
        intervals_df = pd.DataFrame({
            'date': signal_scores.index,
            'ic_value': rolling_ic.values,
            'signal_score': signal_scores.values
        })

        # IC 값에 직접 기반한 리밸런싱 간격 결정
        intervals_df['rebalance_interval'] = intervals_df['ic_value'].apply(
            self.determine_rebalance_interval
        )

        # 정렬 및 포워드 필
        intervals_df = intervals_df.sort_values('date')
        intervals_df['rebalance_interval'] = intervals_df['rebalance_interval'].ffill().fillna(self.default_rebalance_days).astype(int)

        # 리밸런싱 간격 분포 로깅
        interval_counts = intervals_df['rebalance_interval'].value_counts().sort_index()
        logger.info(f"적응형 리밸런싱 간격 계산 완료: {len(intervals_df)}개 날짜")
        logger.info(f"평균 리밸런싱 간격: {intervals_df['rebalance_interval'].mean():.1f}일")
        logger.info(f"간격 분포 상세: {dict(interval_counts)}")

        # 시그널 강도별 간격 매핑 확인
        strong_signals = intervals_df[intervals_df['rebalance_interval'] == self.min_rebalance_days]
        medium_signals = intervals_df[intervals_df['rebalance_interval'] == self.default_rebalance_days]
        weak_signals = intervals_df[intervals_df['rebalance_interval'] == self.max_rebalance_days]

        logger.info(f"강한 시그널 (15일): {len(strong_signals)}개 (평균 강도: {strong_signals['signal_score'].mean():.1f})")
        logger.info(f"중간 시그널 (20일): {len(medium_signals)}개 (평균 강도: {medium_signals['signal_score'].mean():.1f})")
        logger.info(f"약한 시그널 (25일): {len(weak_signals)}개 (평균 강도: {weak_signals['signal_score'].mean():.1f})")

        return intervals_df[['date', 'rebalance_interval']]

    def filter_adaptive_dates(self, all_dates: List[pd.Timestamp],
                            intervals_df: pd.DataFrame) -> List[pd.Timestamp]:
        """
        적응형 간격에 따라 리밸런싱 날짜 필터링

        Args:
            all_dates: 모든 가능한 리밸런싱 날짜
            intervals_df: 적응형 간격 데이터프레임

        Returns:
            filtered_dates: 필터링된 리밸런싱 날짜
        """
        if intervals_df.empty:
            return all_dates

        filtered_dates = []
        current_date = all_dates[0] if all_dates else None

        for target_date in all_dates:
            # 해당 날짜의 리밸런싱 간격 조회
            interval_info = intervals_df[intervals_df['date'] <= target_date]
            if interval_info.empty:
                interval = self.default_rebalance_days
            else:
                interval = interval_info.iloc[-1]['rebalance_interval']

            # 간격 조건 확인
            if current_date is None or (target_date - current_date).days >= interval:
                filtered_dates.append(target_date)
                current_date = target_date

        return filtered_dates


def test_adaptive_rebalancing():
    """
    적응형 리밸런싱 시스템 테스트
    """
    logger.info("=== 적응형 리밸런싱 시스템 테스트 ===")

    # 테스트 설정
    config = {
        'min_rebalance_days': 15,
        'max_rebalance_days': 25,
        'default_rebalance_days': 20,
        'signal_strength_threshold_high': 0.8,
        'signal_strength_threshold_low': 0.6,
        'ic_lookback_window': 60,
        'ic_min_periods': 20,
    }

    adaptive_rb = AdaptiveRebalancing(config)

    # 샘플 데이터 생성
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='20D')
    tickers = ['A', 'B', 'C', 'D', 'E']

    data = []
    for date in dates:
        for ticker in tickers:
            # 랜덤 스코어와 수익률 생성 (약한 상관관계)
            score = np.random.normal(0, 0.5)
            true_return = score * 0.3 + np.random.normal(0, 0.2)
            data.append({
                'date': date,
                'ticker': ticker,
                'score_total_short': score,
                'true_short': true_return
            })

    scores_df = pd.DataFrame(data)
    logger.info(f"샘플 데이터 생성: {len(scores_df)}행, {len(dates)}개 날짜")

    # 적응형 리밸런싱 간격 계산
    intervals_df = adaptive_rb.get_adaptive_rebalance_intervals(scores_df)

    if not intervals_df.empty:
        logger.info("적응형 리밸런싱 결과:")
        logger.info(f"- 총 날짜 수: {len(intervals_df)}")
        logger.info(f"- 평균 리밸런싱 간격: {intervals_df['rebalance_interval'].mean():.1f}일")
        logger.info(f"- 간격 분포: {intervals_df['rebalance_interval'].value_counts().to_dict()}")

        # 시그널 강도 분포
        logger.info("시그널 강도 분포:")
        logger.info(f"- 평균 강도: {intervals_df['signal_score'].mean():.1f}")
        logger.info(f"- 강한 시그널 비율: {(intervals_df['signal_score'] >= 80).mean():.1%}")
        logger.info(f"- 약한 시그널 비율: {(intervals_df['signal_score'] < 60).mean():.1%}")

    logger.info("=== 적응형 리밸런싱 테스트 완료 ===")
    return intervals_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_adaptive_rebalancing()