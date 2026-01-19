# -*- coding: utf-8 -*-
"""
피쳐별 단위 테스트 시스템

개별 피쳐의 IC 기여도, 품질, 안정성을 평가합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureUnitTester:
    """
    피쳐별 단위 테스트 클래스

    각 피쳐의 IC, Hit Ratio, 안정성 등을 개별적으로 평가
    """

    def __init__(self):
        self.test_results = {}

    def calculate_ic(self, scores: pd.Series, returns: pd.Series) -> float:
        """IC (Information Coefficient) 계산"""
        if len(scores) == 0 or len(returns) == 0:
            return np.nan
        valid_idx = scores.notna() & returns.notna()
        if valid_idx.sum() < 2:
            return np.nan
        s = pd.to_numeric(scores[valid_idx], errors='coerce')
        r = pd.to_numeric(returns[valid_idx], errors='coerce')
        final_valid = s.notna() & r.notna()
        if final_valid.sum() < 2:
            return np.nan
        s = s[final_valid]
        r = r[final_valid]
        if s.std() == 0 or r.std() == 0:
            return np.nan
        corr = s.corr(r)
        return float(corr) if not np.isnan(corr) else np.nan

    def calculate_icir(self, ic_series: pd.Series) -> float:
        """ICIR (Information Coefficient IR) 계산"""
        if len(ic_series) == 0:
            return np.nan
        ic_valid = ic_series.dropna()
        if len(ic_valid) == 0:
            return np.nan
        ic_mean = ic_valid.mean()
        ic_std = ic_valid.std()
        if ic_std == 0 or np.isnan(ic_std) or np.isnan(ic_mean):
            return np.nan
        icir = ic_mean / ic_std
        return float(icir) if not np.isnan(icir) else np.nan

    def calculate_hit_ratio(self, scores: pd.Series, returns: pd.Series, top_k: int = 20) -> float:
        """Hit Ratio 계산"""
        if len(scores) == 0 or len(returns) == 0:
            return np.nan
        top_k_idx = scores.nlargest(top_k).index
        top_k_returns = returns.loc[top_k_idx]
        hit_ratio = (top_k_returns > 0).mean()
        return float(hit_ratio) if not np.isnan(hit_ratio) else np.nan

    def test_single_feature(
        self,
        feature_data: pd.Series,
        target_returns: pd.Series,
        cv_folds: pd.DataFrame,
        feature_name: str,
        horizon: str = 'short'
    ) -> Dict:
        """
        단일 피쳐의 성능 테스트

        Args:
            feature_data: 피쳐 값들
            target_returns: 목표 수익률
            cv_folds: CV fold 정보
            feature_name: 피쳐 이름
            horizon: 'short' 또는 'long'

        Returns:
            테스트 결과 딕셔너리
        """
        target_col = 'ret_fwd_20d' if horizon == 'short' else 'ret_fwd_120d'

        # CV fold 구분
        if 'segment' in cv_folds.columns:
            dev_folds = cv_folds[cv_folds['segment'] == 'dev']
            holdout_folds = cv_folds[cv_folds['segment'] == 'holdout']
        else:
            dev_folds = cv_folds[~cv_folds['fold_id'].str.startswith('holdout')]
            holdout_folds = cv_folds[cv_folds['fold_id'].str.startswith('holdout')]

        dev_dates = dev_folds['test_end'].unique()
        holdout_dates = holdout_folds['test_end'].unique()

        # Dev 구간 평가
        dev_ics, dev_hits = [], []
        for date in dev_dates:
            try:
                date_returns = target_returns[target_returns['date'] == date]
                date_features = feature_data[feature_data['date'] == date]

                if len(date_features) < 20:
                    continue

                merged = date_returns.merge(date_features, on=['date', 'ticker'], how='inner')
                if len(merged) < 20:
                    continue

                ic = self.calculate_ic(merged[feature_name], merged[target_col])
                hit = self.calculate_hit_ratio(merged[feature_name], merged[target_col], top_k=20)

                if not np.isnan(ic):
                    dev_ics.append(ic)
                if not np.isnan(hit):
                    dev_hits.append(hit)
            except Exception as e:
                continue

        # Holdout 구간 평가
        holdout_ics, holdout_hits = [], []
        for date in holdout_dates:
            try:
                date_returns = target_returns[target_returns['date'] == date]
                date_features = feature_data[feature_data['date'] == date]

                if len(date_features) < 20:
                    continue

                merged = date_returns.merge(date_features, on=['date', 'ticker'], how='inner')
                if len(merged) < 20:
                    continue

                ic = self.calculate_ic(merged[feature_name], merged[target_col])
                hit = self.calculate_hit_ratio(merged[feature_name], merged[target_col], top_k=20)

                if not np.isnan(ic):
                    holdout_ics.append(ic)
                if not np.isnan(hit):
                    holdout_hits.append(hit)
            except Exception as e:
                continue

        # 결과 계산
        result = {
            'feature_name': feature_name,
            'horizon': horizon,
            'dev_ic_mean': np.mean(dev_ics) if dev_ics else np.nan,
            'dev_ic_std': np.std(dev_ics) if dev_ics else np.nan,
            'dev_icir': self.calculate_icir(pd.Series(dev_ics)) if dev_ics else np.nan,
            'dev_hit_ratio': np.mean(dev_hits) if dev_hits else np.nan,
            'dev_n_periods': len(dev_ics),
            'holdout_ic_mean': np.mean(holdout_ics) if holdout_ics else np.nan,
            'holdout_ic_std': np.std(holdout_ics) if holdout_ics else np.nan,
            'holdout_icir': self.calculate_icir(pd.Series(holdout_ics)) if holdout_ics else np.nan,
            'holdout_hit_ratio': np.mean(holdout_hits) if holdout_hits else np.nan,
            'holdout_n_periods': len(holdout_ics),
        }

        # 추가 메트릭
        result.update({
            'ic_diff': result['holdout_ic_mean'] - result['dev_ic_mean'] if not (np.isnan(result['holdout_ic_mean']) or np.isnan(result['dev_ic_mean'])) else np.nan,
            'hit_ratio_diff': result['holdout_hit_ratio'] - result['dev_hit_ratio'] if not (np.isnan(result['holdout_hit_ratio']) or np.isnan(result['dev_hit_ratio'])) else np.nan,
            'ic_stability': 'stable' if abs(result.get('ic_diff', 0)) < 0.05 else 'unstable',
            'quality_score': self._calculate_quality_score(result)
        })

        self.test_results[feature_name] = result
        return result

    def _calculate_quality_score(self, result: Dict) -> float:
        """피쳐 품질 점수 계산"""
        score = 0

        # IC 기반 점수 (0-40점)
        ic_weight = 0.4
        ic_score = 0
        if not np.isnan(result.get('holdout_ic_mean', np.nan)):
            ic_value = abs(result['holdout_ic_mean'])
            ic_score = min(ic_value * 40, 40)  # 최대 40점

        # ICIR 기반 점수 (0-30점)
        icir_weight = 0.3
        icir_score = 0
        if not np.isnan(result.get('holdout_icir', np.nan)):
            icir_value = abs(result['holdout_icir'])
            icir_score = min(icir_value * 10, 30)  # 최대 30점

        # Hit Ratio 기반 점수 (0-20점)
        hit_weight = 0.2
        hit_score = 0
        if not np.isnan(result.get('holdout_hit_ratio', np.nan)):
            hit_value = result['holdout_hit_ratio']
            hit_score = min((hit_value - 0.5) * 40, 20) if hit_value > 0.5 else 0  # 최대 20점

        # 안정성 보너스 (0-10점)
        stability_bonus = 0
        if result.get('ic_stability') == 'stable':
            stability_bonus = 10

        total_score = ic_score + icir_score + hit_score + stability_bonus
        return min(total_score, 100)  # 최대 100점

    def test_feature_set(
        self,
        feature_df: pd.DataFrame,
        target_df: pd.DataFrame,
        cv_folds: pd.DataFrame,
        feature_names: List[str],
        horizon: str = 'short'
    ) -> pd.DataFrame:
        """
        여러 피쳐들의 성능 테스트

        Args:
            feature_df: 피쳐 데이터프레임
            target_df: 목표 수익률 데이터프레임
            cv_folds: CV fold 정보
            feature_names: 테스트할 피쳐 이름들
            horizon: 'short' 또는 'long'

        Returns:
            테스트 결과 데이터프레임
        """
        results = []

        print(f"🔬 {len(feature_names)}개 피쳐 단위 테스트 시작...")

        for i, feature_name in enumerate(feature_names):
            if feature_name not in feature_df.columns:
                print(f"⚠️ 피쳐 {feature_name}이 데이터에 없음, 건너뜀")
                continue

            print(f"  테스트 {i+1}/{len(feature_names)}: {feature_name}")

            # 피쳐 데이터 준비
            feature_data = feature_df[['date', 'ticker', feature_name]].copy()
            feature_data = feature_data.dropna(subset=[feature_name])

            if len(feature_data) == 0:
                print(f"    ⚠️ {feature_name}: 유효한 데이터 없음")
                continue

            # 테스트 실행
            result = self.test_single_feature(
                feature_data, target_df, cv_folds, feature_name, horizon
            )
            results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # 품질 점수 기준 정렬
            results_df = results_df.sort_values('quality_score', ascending=False)

        print(f"✅ 단위 테스트 완료: {len(results_df)}개 피쳐 평가")

        return results_df

    def get_feature_rankings(self) -> pd.DataFrame:
        """피쳐별 순위표 생성"""
        if not self.test_results:
            return pd.DataFrame()

        results_df = pd.DataFrame.from_dict(self.test_results, orient='index')

        # 품질 점수로 정렬
        results_df = results_df.sort_values('quality_score', ascending=False)

        return results_df

    def get_top_features(self, top_n: int = 10) -> List[str]:
        """상위 품질 피쳐 목록 반환"""
        rankings = self.get_feature_rankings()
        if len(rankings) == 0:
            return []

        return rankings.head(top_n).index.tolist()

    def get_feature_quality_report(self) -> str:
        """피쳐 품질 보고서 생성"""
        if not self.test_results:
            return "테스트 결과가 없습니다."

        rankings = self.get_feature_rankings()

        report = []
        report.append("# 피쳐별 단위 테스트 품질 보고서")
        report.append("")
        report.append("## 상위 10개 피쳐")
        report.append("")
        report.append("| 순위 | 피쳐명 | 품질점수 | Holdout IC | ICIR | Hit Ratio | 안정성 |")
        report.append("|------|--------|----------|------------|------|-----------|--------|")

        for i, (feature_name, row) in enumerate(rankings.head(10).iterrows()):
            report.append(
                f"| {i+1} | {feature_name} | {row['quality_score']:.1f} | "
                f"{row.get('holdout_ic_mean', 'N/A'):>.3f} | "
                f"{row.get('holdout_icir', 'N/A'):>.3f} | "
                f"{row.get('holdout_hit_ratio', 'N/A'):>.1%} | "
                f"{row.get('ic_stability', 'N/A')} |"
            )

        report.append("")
        report.append("## 품질 점수 기준")
        report.append("- IC 기여도: 40점 만점")
        report.append("- ICIR 안정성: 30점 만점")
        report.append("- Hit Ratio: 20점 만점")
        report.append("- 안정성 보너스: 10점 (IC diff < 0.05)")

        return "\n".join(report)


def test_feature_unit_tester():
    """피쳐 단위 테스트 시스템 테스트"""
    from src.utils.config import load_config
    from src.utils.io import load_artifact
    from pathlib import Path

    # 설정 로드
    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # 데이터 로드 (L6R 결과 사용)
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')
    rebalance_df = load_artifact(interim_dir / 'rebalance_scores_from_ranking')
    cv_folds = load_artifact(interim_dir / 'cv_folds_short')

    if panel_df is None or rebalance_df is None or cv_folds is None:
        print("데이터 로드 실패")
        return

    # 테스트할 피쳐들 (새로 추가된 것들 위주)
    test_features = [
        'close_to_52w_high', 'close_to_52w_low', 'intraday_price_position',
        'momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj',
        'volatility_asymmetry', 'tail_risk_5pct',
        'news_intensity', 'news_trend'
    ]

    # 실제로 존재하는 피쳐들만 테스트
    available_features = [f for f in test_features if f in panel_df.columns]
    print(f"테스트할 피쳐들: {available_features}")

    if not available_features:
        print("테스트할 피쳐가 없음")
        return

    # 단위 테스트 실행
    tester = FeatureUnitTester()

    # 피쳐 데이터 준비 (date, ticker, feature)
    feature_data = panel_df[['date', 'ticker'] + available_features].copy()

    # 목표 수익률 데이터 (rebalance_scores에서 가져옴)
    target_data = rebalance_df[['date', 'ticker', 'true_short']].copy()
    target_data = target_data.rename(columns={'true_short': 'ret_fwd_20d'})

    # 테스트 실행
    results_df = tester.test_feature_set(
        feature_data, target_data, cv_folds, available_features, 'short'
    )

    if len(results_df) > 0:
        print("\n=== 테스트 결과 요약 ===")
        print(results_df[['feature_name', 'quality_score', 'holdout_ic_mean', 'holdout_hit_ratio']].head())

        # 보고서 생성
        report = tester.get_feature_quality_report()
        print("\n=== 품질 보고서 ===")
        print(report[:500] + "..." if len(report) > 500 else report)

    return results_df


if __name__ == "__main__":
    test_feature_unit_tester()