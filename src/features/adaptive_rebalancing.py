"""
적응형 리밸런싱 모듈 (bt20 프로페셔널용)

시그널 강도에 따라 리밸런싱 주기를 동적으로 조정합니다.
단기 투자자의 민첩성은 유지하면서 비용 효율성을 높입니다.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class AdaptiveRebalancing:
    """
    적응형 리밸런싱 클래스

    시그널 강도를 기반으로 최적의 리밸런싱 주기를 결정합니다.
    """

    def __init__(
        self,
        strong_threshold: float = 0.8,
        medium_threshold: float = 0.6,
        weak_threshold: float = 0.6,
        strong_interval: int = 15,
        medium_interval: int = 20,
        weak_interval: int = 25,
        window_days: int = 60,
        min_periods: int = 20,
    ):
        """
        초기화

        Args:
            strong_threshold: 강한 시그널 임계값 (0.8 = 80점)
            medium_threshold: 중간 시그널 임계값 (0.6 = 60점)
            weak_threshold: 약한 시그널 임계값 (0.6 = 60점)
            strong_interval: 강한 시그널 리밸런싱 주기 (15일)
            medium_interval: 중간 시그널 리밸런싱 주기 (20일)
            weak_interval: 약한 시그널 리밸런싱 주기 (25일)
            window_days: 시그널 강도 계산 윈도우 (60일)
            min_periods: 최소 계산 기간 (20일)
        """
        self.strong_threshold = strong_threshold
        self.medium_threshold = medium_threshold
        self.weak_threshold = weak_threshold
        self.strong_interval = strong_interval
        self.medium_interval = medium_interval
        self.weak_interval = weak_interval
        self.window_days = window_days
        self.min_periods = min_periods

    def calculate_signal_strength(
        self,
        ranking_scores: pd.Series,
        future_returns: pd.Series,
        window_days: Optional[int] = None,
    ) -> pd.Series:
        """
        시그널 강도를 계산합니다.

        Args:
            ranking_scores: 랭킹 점수들
            future_returns: 미래 수익률들
            window_days: 계산 윈도우 (기본값 사용 시 None)

        Returns:
            시그널 강도 점수들 (0-1 스케일)
        """
        if window_days is None:
            window_days = self.window_days

        # 롤링 IC 계산
        def rolling_ic(scores, returns, window):
            """롤링 IC 계산"""
            if len(scores) < self.min_periods:
                return np.nan

            # 랭킹 점수와 수익률의 상관계수
            corr = scores.rolling(window=window, min_periods=self.min_periods).corr(
                returns
            )
            return corr.iloc[-1] if len(corr) > 0 else np.nan

        # 날짜별로 그룹화하여 계산
        signal_strengths = []

        for date in ranking_scores.index.get_level_values("date").unique():
            date_mask = ranking_scores.index.get_level_values("date") == date
            date_scores = ranking_scores[date_mask]
            date_returns = future_returns[date_mask]

            if len(date_scores) == 0 or len(date_returns) == 0:
                signal_strengths.append((date, np.nan))
                continue

            # 해당 날짜의 롤링 IC 계산
            try:
                ic = date_scores.corr(date_returns)
                # IC를 0-1 스케일로 변환 (절대값 사용)
                strength = abs(ic) if not np.isnan(ic) else np.nan
                signal_strengths.append((date, strength))
            except:
                signal_strengths.append((date, np.nan))

        # 결과를 Series로 변환
        strength_series = pd.Series(
            [x[1] for x in signal_strengths],
            index=[x[0] for x in signal_strengths],
            name="signal_strength",
        )

        return strength_series

    def determine_rebalance_interval(self, signal_strength: float) -> int:
        """
        시그널 강도에 따른 리밸런싱 주기를 결정합니다.

        Args:
            signal_strength: 시그널 강도 (0-1)

        Returns:
            리밸런싱 주기 (일)
        """
        if np.isnan(signal_strength):
            return self.medium_interval  # NaN인 경우 중간값 사용

        if signal_strength >= self.strong_threshold:
            return self.strong_interval  # 강한 시그널: 15일
        elif signal_strength >= self.medium_threshold:
            return self.medium_interval  # 중간 시그널: 20일
        else:
            return self.weak_interval  # 약한 시그널: 25일

    def get_adaptive_schedule(
        self,
        ranking_data: pd.DataFrame,
        start_date: str = "2016-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        적응형 리밸런싱 스케줄을 생성합니다.

        Args:
            ranking_data: 랭킹 데이터 (date, ticker, score_total_short)
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            리밸런싱 스케줄 DataFrame
        """
        print("🔄 적응형 리밸런싱 스케줄 생성 중...")

        # 날짜 범위 생성
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        schedule_data = []

        current_date = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        while current_date <= end_dt:
            # 현재 날짜의 시그널 강도 계산
            available_data = ranking_data[ranking_data["date"] <= current_date]

            if len(available_data) >= self.min_periods:
                # 최근 데이터로 시그널 강도 계산
                recent_data = available_data.tail(100)  # 최근 100개 데이터 사용

                if len(recent_data) >= self.min_periods:
                    try:
                        signal_strength = (
                            self.calculate_signal_strength(
                                recent_data.set_index(["date", "ticker"])[
                                    "score_total_short"
                                ],
                                recent_data.set_index(["date", "ticker"])["true_short"],
                                self.window_days,
                            ).iloc[-1]
                            if len(recent_data) > 0
                            else np.nan
                        )

                        rebalance_interval = self.determine_rebalance_interval(
                            signal_strength
                        )
                    except:
                        signal_strength = np.nan
                        rebalance_interval = self.medium_interval
                else:
                    signal_strength = np.nan
                    rebalance_interval = self.medium_interval
            else:
                signal_strength = np.nan
                rebalance_interval = self.medium_interval

            # 스케줄에 추가
            schedule_data.append(
                {
                    "date": current_date,
                    "signal_strength": signal_strength,
                    "rebalance_interval": rebalance_interval,
                    "signal_category": self._categorize_signal(signal_strength),
                }
            )

            # 다음 리밸런싱 날짜 계산
            current_date += pd.Timedelta(days=rebalance_interval)

        schedule_df = pd.DataFrame(schedule_data)
        print(f"✅ 적응형 스케줄 생성 완료: {len(schedule_df)}개 리밸런싱 포인트")

        return schedule_df

    def _categorize_signal(self, strength: float) -> str:
        """시그널 강도를 카테고리로 분류"""
        if np.isnan(strength):
            return "unknown"
        elif strength >= self.strong_threshold:
            return "strong"
        elif strength >= self.medium_threshold:
            return "medium"
        else:
            return "weak"

    def analyze_schedule_statistics(self, schedule_df: pd.DataFrame) -> dict:
        """
        리밸런싱 스케줄의 통계 분석

        Args:
            schedule_df: 리밸런싱 스케줄 DataFrame

        Returns:
            통계 분석 결과
        """
        stats = {
            "total_rebalances": len(schedule_df),
            "avg_interval": schedule_df["rebalance_interval"].mean(),
            "min_interval": schedule_df["rebalance_interval"].min(),
            "max_interval": schedule_df["rebalance_interval"].max(),
            "signal_distribution": schedule_df["signal_category"]
            .value_counts()
            .to_dict(),
            "avg_signal_strength": schedule_df["signal_strength"].mean(),
            "signal_strength_by_category": {},
        }

        # 카테고리별 평균 시그널 강도
        for category in ["strong", "medium", "weak"]:
            category_data = schedule_df[schedule_df["signal_category"] == category]
            if len(category_data) > 0:
                stats["signal_strength_by_category"][category] = category_data[
                    "signal_strength"
                ].mean()
            else:
                stats["signal_strength_by_category"][category] = np.nan

        return stats

    def visualize_schedule(
        self, schedule_df: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        리밸런싱 스케줄을 시각화합니다.

        Args:
            schedule_df: 리밸런싱 스케줄 DataFrame
            save_path: 저장 경로 (선택)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

            # 1. 리밸런싱 간격 추이
            ax1.plot(
                schedule_df["date"], schedule_df["rebalance_interval"], "b-", alpha=0.7
            )
            ax1.set_title("Adaptive Rebalancing Intervals Over Time")
            ax1.set_ylabel("Interval (days)")
            ax1.grid(True, alpha=0.3)

            # 2. 시그널 강도 추이
            ax2.plot(
                schedule_df["date"], schedule_df["signal_strength"], "r-", alpha=0.7
            )
            ax2.set_title("Signal Strength Over Time")
            ax2.set_ylabel("Signal Strength (0-1)")
            ax2.grid(True, alpha=0.3)

            # 3. 시그널 카테고리 분포
            categories = schedule_df["signal_category"].value_counts()
            colors = {
                "strong": "green",
                "medium": "orange",
                "weak": "red",
                "unknown": "gray",
            }
            ax3.bar(
                categories.index,
                categories.values,
                color=[colors.get(cat, "gray") for cat in categories.index],
            )
            ax3.set_title("Signal Category Distribution")
            ax3.set_ylabel("Count")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"📊 차트 저장: {save_path}")

            plt.show()

        except ImportError:
            print("⚠️ 시각화를 위해 matplotlib과 seaborn을 설치해주세요.")
        except Exception as e:
            print(f"⚠️ 시각화 중 오류 발생: {e}")


def test_adaptive_rebalancing():
    """적응형 리밸런싱 테스트"""
    from pathlib import Path

    from src.utils.config import load_config
    from src.utils.io import load_artifact

    print("🧪 적응형 리밸런싱 테스트")
    print("=" * 40)

    # 설정 로드
    cfg = load_config("configs/config.yaml")
    interim_dir = Path(cfg["paths"]["base_dir"]) / "data" / "interim"

    # 데이터 로드
    ranking_data = load_artifact(interim_dir / "ranking_short_daily")
    rebalance_data = load_artifact(interim_dir / "rebalance_scores_from_ranking")

    if ranking_data is None or rebalance_data is None:
        print("❌ 필요한 데이터가 없습니다.")
        return

    # 테스트용 데이터 준비 (최근 1년)
    test_ranking = ranking_data[ranking_data["date"] >= "2023-01-01"].copy()
    test_rebalance = rebalance_data[rebalance_data["date"] >= "2023-01-01"].copy()

    print(
        f"테스트 데이터: {len(test_ranking)}개 랭킹, {len(test_rebalance)}개 리밸런싱 포인트"
    )

    # 적응형 리밸런싱 객체 생성
    adaptive_rb = AdaptiveRebalancing()

    # 시그널 강도 계산 테스트
    if len(test_rebalance) > 0:
        signal_strengths = adaptive_rb.calculate_signal_strength(
            test_rebalance.set_index(["date", "ticker"])["score_total_short"],
            test_rebalance.set_index(["date", "ticker"])["true_short"],
        )

        print("\n📊 시그널 강도 샘플:")
        print(signal_strengths.head())
        print(f"평균 시그널 강도: {signal_strengths.mean():.3f}")
        print(f"NaN 비율: {signal_strengths.isnull().mean():.1%}")

        # 리밸런싱 주기 결정 테스트
        sample_strengths = [0.9, 0.7, 0.5, 0.3, np.nan]
        print("\n🔄 리밸런싱 주기 결정 테스트:")
        for strength in sample_strengths:
            interval = adaptive_rb.determine_rebalance_interval(strength)
            category = adaptive_rb._categorize_signal(strength)
            print(f"  강도 {strength:.2f}: 간격 {interval}일, 카테고리 {category}")

    # 스케줄 생성 테스트 (샘플)
    print("📅 적응형 스케줄 생성 테스트:")
    try:
        schedule = adaptive_rb.get_adaptive_schedule(
            test_rebalance.head(100),
            "2023-01-01",
            "2023-03-31",  # 샘플 데이터 사용
        )

        if len(schedule) > 0:
            print("스케줄 샘플:")
            print(schedule.head())
            print(f"\n총 리밸런싱 포인트: {len(schedule)}개")

            # 통계 분석
            stats = adaptive_rb.analyze_schedule_statistics(schedule)
            print("📈 스케줄 통계:")
            print(f"평균 리밸런싱 간격: {stats['avg_interval']:.1f}일")
            print(f"최소/최대 간격: {stats['min_interval']}/{stats['max_interval']}일")
            print(f"시그널 분포: {stats['signal_distribution']}")

    except Exception as e:
        print(f"스케줄 생성 중 오류: {e}")

    print("\n✅ 적응형 리밸런싱 테스트 완료!")


if __name__ == "__main__":
    test_adaptive_rebalancing()
