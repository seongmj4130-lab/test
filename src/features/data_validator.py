# -*- coding: utf-8 -*-
"""
데이터 검증 자동화 시스템

피쳐 추가/변경에 대한 자동 검증을 수행합니다.
데이터 무결성, 품질, 일관성을 검사합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """
    데이터 검증 자동화 클래스

    피쳐 엔지니어링 결과의 품질과 일관성을 자동으로 검증
    """

    def __init__(self):
        self.validation_results = {}

    def validate_data_integrity(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        데이터 무결성 검증

        Args:
            df: 검증할 데이터프레임
            required_columns: 필수 컬럼 리스트

        Returns:
            검증 결과 딕셔너리
        """
        integrity_result = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': {},
            'duplicate_rows': df.duplicated().sum(),
            'required_columns_present': True,
            'critical_issues': []
        }

        # 필수 컬럼 검증
        if required_columns:
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                integrity_result['required_columns_present'] = False
                integrity_result['critical_issues'].append(
                    f"필수 컬럼 누락: {missing_required}"
                )

        # 결측치 분석
        missing_stats = df.isnull().sum()
        integrity_result['missing_values'] = missing_stats.to_dict()

        # 결측치 비율이 높은 컬럼들
        missing_ratio = missing_stats / len(df)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        if high_missing_cols:
            integrity_result['critical_issues'].append(
                f"결측치 50% 이상 컬럼: {high_missing_cols}"
            )

        # 중복 행 검증
        if integrity_result['duplicate_rows'] > 0:
            integrity_result['critical_issues'].append(
                f"중복 행 발견: {integrity_result['duplicate_rows']}개"
            )

        return integrity_result

    def validate_feature_quality(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """
        피쳐 품질 검증

        Args:
            df: 피쳐 데이터프레임
            feature_columns: 검증할 피쳐 컬럼들

        Returns:
            품질 검증 결과
        """
        quality_result = {
            'features_analyzed': len(feature_columns),
            'feature_stats': {},
            'quality_issues': [],
            'recommendations': []
        }

        for feature in feature_columns:
            if feature not in df.columns:
                quality_result['quality_issues'].append(f"피쳐 누락: {feature}")
                continue

            feature_data = df[feature]
            stats = {
                'count': len(feature_data),
                'missing': feature_data.isnull().sum(),
                'missing_ratio': feature_data.isnull().mean(),
                'unique_values': feature_data.nunique(),
                'dtype': str(feature_data.dtype)
            }

            # 수치형 피쳐 추가 통계
            if feature_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
                stats.update({
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'skewness': feature_data.skew(),
                    'kurtosis': feature_data.kurtosis(),
                    'zero_ratio': (feature_data == 0).mean(),
                    'outliers': self._detect_outliers(feature_data)
                })

                # 품질 이슈 검출
                if stats['std'] == 0:
                    quality_result['quality_issues'].append(
                        f"{feature}: 표준편차가 0 (상수 값)"
                    )

                if stats['missing_ratio'] > 0.1:
                    quality_result['quality_issues'].append(
                        f"{feature}: 결측치 비율 높음 ({stats['missing_ratio']:.1%})"
                    )

                if abs(stats['skewness']) > 3:
                    quality_result['recommendations'].append(
                        f"{feature}: 왜도 높음 ({stats['skewness']:.2f}), 변환 고려"
                    )

                if stats['outliers'] > len(feature_data) * 0.05:
                    quality_result['recommendations'].append(
                        f"{feature}: 이상치 많음 ({stats['outliers']}개), 처리 고려"
                    )

            quality_result['feature_stats'][feature] = stats

        return quality_result

    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> int:
        """이상치 검출"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            return int(outliers)
        return 0

    def validate_feature_consistency(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        groupby_column: str = 'ticker'
    ) -> Dict[str, Any]:
        """
        피쳐 일관성 검증 (종목별/시간별)

        Args:
            df: 데이터프레임
            feature_columns: 검증할 피쳐들
            groupby_column: 그룹화 기준 컬럼

        Returns:
            일관성 검증 결과
        """
        consistency_result = {
            'groupby_column': groupby_column,
            'feature_consistency': {},
            'temporal_consistency': {},
            'issues': []
        }

        if groupby_column not in df.columns:
            consistency_result['issues'].append(f"그룹화 컬럼 없음: {groupby_column}")
            return consistency_result

        # 그룹별 피쳐 일관성
        for feature in feature_columns:
            if feature not in df.columns:
                continue

            # 그룹별 통계
            group_stats = df.groupby(groupby_column)[feature].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])

            # 일관성 점수 계산 (낮을수록 일관성 좋음)
            consistency_score = group_stats['std'].mean() / abs(group_stats['mean'].mean() + 1e-8)
            consistency_result['feature_consistency'][feature] = {
                'consistency_score': consistency_score,
                'groups_with_data': len(group_stats),
                'avg_group_size': group_stats['count'].mean()
            }

            # 이슈 검출
            if consistency_score > 1.0:
                consistency_result['issues'].append(
                    f"{feature}: 그룹간 변동성 높음 (일관성 점수: {consistency_score:.2f})"
                )

        # 시간적 일관성 (날짜별)
        if 'date' in df.columns:
            for feature in feature_columns:
                if feature not in df.columns:
                    continue

                # 날짜별 통계
                date_stats = df.groupby('date')[feature].agg(['count', 'mean', 'std'])
                date_consistency = date_stats['std'].mean() / abs(date_stats['mean'].mean() + 1e-8)

                consistency_result['temporal_consistency'][feature] = {
                    'temporal_consistency_score': date_consistency,
                    'dates_with_data': len(date_stats)
                }

                if date_consistency > 0.5:
                    consistency_result['issues'].append(
                        f"{feature}: 시간적 변동성 높음 (점수: {date_consistency:.2f})"
                    )

        return consistency_result

    def validate_feature_correlations(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        피쳐 간 상관관계 및 타겟 변수와의 관계 검증

        Args:
            df: 데이터프레임
            feature_columns: 피쳐 컬럼들
            target_column: 타겟 변수 컬럼 (선택)

        Returns:
            상관관계 분석 결과
        """
        correlation_result = {
            'feature_correlations': {},
            'target_correlations': {},
            'multicollinearity_issues': [],
            'high_correlation_pairs': []
        }

        # 수치형 피쳐만 선택
        numeric_features = [col for col in feature_columns
                          if col in df.columns and
                          df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

        if len(numeric_features) < 2:
            correlation_result['multicollinearity_issues'].append("분석할 수치형 피쳐가 부족함")
            return correlation_result

        # 피쳐 간 상관관계
        corr_matrix = df[numeric_features].corr()

        # 상관관계가 높은 쌍 찾기
        high_corr_pairs = []
        for i in range(len(numeric_features)):
            for j in range(i+1, len(numeric_features)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.8:  # 상관계수 0.8 이상
                    high_corr_pairs.append({
                        'feature1': numeric_features[i],
                        'feature2': numeric_features[j],
                        'correlation': corr
                    })

        correlation_result['high_correlation_pairs'] = high_corr_pairs

        if high_corr_pairs:
            correlation_result['multicollinearity_issues'].append(
                f"높은 상관관계 피쳐 쌍: {len(high_corr_pairs)}개 발견"
            )

        # 타겟 변수와의 상관관계
        if target_column and target_column in df.columns:
            target_corr = df[numeric_features + [target_column]].corr()[target_column]
            correlation_result['target_correlations'] = target_corr.drop(target_column).to_dict()

            # 타겟과 낮은 상관관계 피쳐들
            low_corr_features = [feat for feat, corr in correlation_result['target_correlations'].items()
                               if abs(corr) < 0.01]
            if low_corr_features:
                correlation_result['multicollinearity_issues'].append(
                    f"타겟과 낮은 상관관계 피쳐: {low_corr_features}"
                )

        return correlation_result

    def run_comprehensive_validation(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        종합 데이터 검증 실행

        Args:
            df: 검증할 데이터프레임
            feature_columns: 피쳐 컬럼들
            target_column: 타겟 변수 컬럼
            required_columns: 필수 컬럼들

        Returns:
            종합 검증 결과
        """
        print("🔍 데이터 검증 시작...")

        comprehensive_result = {
            'timestamp': pd.Timestamp.now(),
            'summary': {},
            'details': {}
        }

        # 1. 데이터 무결성 검증
        print("  📋 데이터 무결성 검증 중...")
        integrity_result = self.validate_data_integrity(df, required_columns)
        comprehensive_result['details']['integrity'] = integrity_result

        # 2. 피쳐 품질 검증
        print("  📊 피쳐 품질 검증 중...")
        quality_result = self.validate_feature_quality(df, feature_columns)
        comprehensive_result['details']['quality'] = quality_result

        # 3. 피쳐 일관성 검증
        print("  🔄 피쳐 일관성 검증 중...")
        consistency_result = self.validate_feature_consistency(df, feature_columns)
        comprehensive_result['details']['consistency'] = consistency_result

        # 4. 피쳐 상관관계 검증
        print("  📈 피쳐 상관관계 검증 중...")
        correlation_result = self.validate_feature_correlations(df, feature_columns, target_column)
        comprehensive_result['details']['correlation'] = correlation_result

        # 검증 요약 생성
        comprehensive_result['summary'] = self._generate_validation_summary(
            integrity_result, quality_result, consistency_result, correlation_result
        )

        print("✅ 데이터 검증 완료")
        return comprehensive_result

    def _generate_validation_summary(
        self,
        integrity: Dict,
        quality: Dict,
        consistency: Dict,
        correlation: Dict
    ) -> Dict[str, Any]:
        """검증 결과 요약 생성"""

        # 심각도별 이슈 분류
        critical_issues = []
        warnings = []
        recommendations = []

        # 무결성 이슈
        critical_issues.extend(integrity.get('critical_issues', []))

        # 품질 이슈
        critical_issues.extend(quality.get('quality_issues', []))
        recommendations.extend(quality.get('recommendations', []))

        # 일관성 이슈
        warnings.extend(consistency.get('issues', []))

        # 상관관계 이슈
        warnings.extend(correlation.get('multicollinearity_issues', []))

        summary = {
            'overall_status': 'PASS' if not critical_issues else 'FAIL',
            'total_features': quality.get('features_analyzed', 0),
            'critical_issues_count': len(critical_issues),
            'warning_count': len(warnings),
            'recommendation_count': len(recommendations),
            'data_quality_score': self._calculate_data_quality_score(
                integrity, quality, consistency, correlation
            ),
            'critical_issues': critical_issues[:5],  # 상위 5개만
            'warnings': warnings[:5],
            'recommendations': recommendations[:5]
        }

        return summary

    def _calculate_data_quality_score(
        self,
        integrity: Dict,
        quality: Dict,
        consistency: Dict,
        correlation: Dict
    ) -> float:
        """데이터 품질 점수 계산 (0-100)"""

        score = 100.0

        # 무결성 점수 차감
        missing_ratio = sum(integrity.get('missing_values', {}).values()) / (integrity.get('total_rows', 1) * integrity.get('total_columns', 1))
        score -= missing_ratio * 20  # 결측치 5%당 1점 차감

        if integrity.get('duplicate_rows', 0) > 0:
            score -= 5

        # 품질 점수 차감
        quality_issues = len(quality.get('quality_issues', []))
        score -= quality_issues * 10  # 품질 이슈당 10점 차감

        # 일관성 점수 차감
        consistency_issues = len(consistency.get('issues', []))
        score -= consistency_issues * 5  # 일관성 이슈당 5점 차감

        # 상관관계 점수 차감
        correlation_issues = len(correlation.get('multicollinearity_issues', []))
        score -= correlation_issues * 5  # 상관관계 이슈당 5점 차감

        return max(0, min(100, score))

    def generate_validation_report(self, validation_result: Dict) -> str:
        """검증 결과 보고서 생성"""

        report = []
        report.append("# 데이터 검증 자동화 보고서")
        report.append("")
        report.append(f"생성 시간: {validation_result['timestamp']}")
        report.append("")

        # 요약 섹션
        summary = validation_result['summary']
        report.append("## 검증 요약")
        report.append("")
        report.append(f"- **전체 상태**: {summary['overall_status']}")
        report.append(f"- **분석 피쳐 수**: {summary['total_features']}")
        report.append(f"- **데이터 품질 점수**: {summary['data_quality_score']:.1f}/100")
        report.append(f"- **심각한 이슈**: {summary['critical_issues_count']}개")
        report.append(f"- **경고**: {summary['warning_count']}개")
        report.append(f"- **권장사항**: {summary['recommendation_count']}개")
        report.append("")

        # 상세 섹션
        if summary['critical_issues']:
            report.append("## 🚨 심각한 이슈")
            report.append("")
            for issue in summary['critical_issues']:
                report.append(f"- {issue}")
            report.append("")

        if summary['warnings']:
            report.append("## ⚠️ 경고")
            report.append("")
            for warning in summary['warnings']:
                report.append(f"- {warning}")
            report.append("")

        if summary['recommendations']:
            report.append("## 💡 권장사항")
            report.append("")
            for rec in summary['recommendations']:
                report.append(f"- {rec}")
            report.append("")

        return "\n".join(report)


def test_data_validator():
    """데이터 검증 시스템 테스트"""
    from src.utils.config import load_config
    from src.utils.io import load_artifact
    from pathlib import Path

    # 설정 로드
    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # 데이터 로드
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')

    if panel_df is None:
        print("데이터 로드 실패")
        return

    # 테스트할 피쳐들
    test_features = [
        'close_to_52w_high', 'close_to_52w_low', 'intraday_price_position',
        'momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj',
        'volatility_asymmetry', 'tail_risk_5pct'
    ]

    # 실제 존재하는 피쳐들만
    available_features = [f for f in test_features if f in panel_df.columns]

    if not available_features:
        print("테스트할 피쳐가 없음")
        return

    # 검증 실행
    validator = DataValidator()
    validation_result = validator.run_comprehensive_validation(
        df=panel_df,
        feature_columns=available_features,
        required_columns=['date', 'ticker', 'close'],
        target_column=None
    )

    # 보고서 생성
    report = validator.generate_validation_report(validation_result)
    print("=== 데이터 검증 보고서 ===")
    print(report)

    return validation_result


if __name__ == "__main__":
    test_data_validator()