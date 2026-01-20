"""
통합 스모크 테스트 - 외부 데이터 없이 합성 입력으로 끝까지 실행 가능한 최소 경로
"""

import pandas as pd
import pytest

from src.components.portfolio.selector import select_topk_with_fallback
from src.components.ranking.contribution_engine import infer_feature_group
from src.utils.io import load_artifact, save_artifact
from src.utils.validate import validate_df


class TestIntegrationSmoke:
    """통합 스모크 테스트"""

    def test_full_pipeline_smoke_test(
        self, tmp_path, sample_rankings, sample_portfolio_weights
    ):
        """외부 데이터 없이 완전한 파이프라인 실행 테스트"""

        # 1. 입력 데이터 준비
        rankings_data = sample_rankings.copy()

        # 2. 포트폴리오 선택 실행
        selected_df, diagnostics = select_topk_with_fallback(
            rankings_data, top_k=5, score_col="score", required_cols=["score"]
        )

        # 선택 결과 검증
        assert len(selected_df) <= 5
        assert diagnostics["selected_count"] == len(selected_df)
        assert diagnostics["eligible_count"] == len(rankings_data)

        # 3. 결과 저장/로드 테스트
        output_file = tmp_path / "selected_portfolio"

        # 선택 결과를 파일로 저장
        save_artifact(selected_df, output_file, formats=["parquet", "csv"])

        # 저장된 파일 존재 확인
        assert (output_file.with_suffix(".parquet")).exists()
        assert (output_file.with_suffix(".csv")).exists()

        # 저장된 파일 로드 및 검증
        loaded_df = load_artifact(output_file)
        pd.testing.assert_frame_equal(selected_df, loaded_df)

        # 4. 데이터 검증 실행
        validation_result = validate_df(
            loaded_df, stage="portfolio_selection", required_cols=["ticker", "score"]
        )

        assert validation_result.ok is True
        assert len(validation_result.errors) == 0

        # 5. 피처 그룹 추론 테스트 (랭킹 관련)
        test_features = ["roe", "price_momentum", "esg_score", "unknown_feature"]
        feature_groups = [infer_feature_group(feat) for feat in test_features]

        expected_groups = [
            "profitability",
            "technical",
            "other",
            "other",
        ]  # price_momentum은 momentum으로 technical
        assert feature_groups == expected_groups

        # 6. 최종 결과 구조 검증
        assert isinstance(selected_df, pd.DataFrame)
        assert len(selected_df.columns) >= 2  # 최소 ticker와 score
        assert "ticker" in selected_df.columns
        assert "score" in selected_df.columns

        # 모든 점수가 유효한 숫자인지 확인
        assert selected_df["score"].notna().all()
        assert (selected_df["score"] >= 0).all() and (selected_df["score"] <= 1).all()

        # 출력 생성 확인 (파일들이 생성되었는지)
        output_files = list(tmp_path.glob("selected_portfolio.*"))
        assert len(output_files) >= 2  # parquet와 csv 파일

        print("통합 스모크 테스트 성공!")
        print(f"선택된 종목 수: {len(selected_df)}")
        print(f"생성된 출력 파일 수: {len(output_files)}")
        print(
            f"피처 그룹 분류 결과: {dict(zip(test_features, feature_groups, strict=True))}"
        )

    def test_minimal_data_pipeline(self, tmp_path):
        """최소 데이터로 파이프라인 실행 테스트"""

        # 최소한의 랭킹 데이터 생성
        minimal_rankings = pd.DataFrame(
            {"ticker": ["A", "B", "C"], "score": [0.8, 0.6, 0.4], "rank": [1, 2, 3]}
        )

        # 포트폴리오 선택
        selected_df, diagnostics = select_topk_with_fallback(
            minimal_rankings, top_k=2, score_col="score"
        )

        # 결과 검증
        assert len(selected_df) == 2
        assert selected_df["score"].iloc[0] >= selected_df["score"].iloc[1]  # 내림차순

        # 저장/로드
        output_file = tmp_path / "minimal_portfolio"
        save_artifact(selected_df, output_file, formats=["csv"])

        loaded_df = load_artifact(output_file)
        assert len(loaded_df) == 2

        print("최소 데이터 파이프라인 테스트 성공!")

    def test_error_handling_pipeline(self, tmp_path):
        """에러 처리 파이프라인 테스트"""

        # 빈 데이터로 포트폴리오 선택 시도 (ticker, score 컬럼이 있는 빈 DF)
        empty_rankings = pd.DataFrame(columns=["ticker", "score", "rank"])
        selected_df, diagnostics = select_topk_with_fallback(
            empty_rankings, top_k=5, score_col="score"
        )

        # 빈 결과가 제대로 처리되는지 확인
        assert len(selected_df) == 0
        assert diagnostics["selected_count"] == 0

        # 빈 결과를 저장하려고 시도 (에러 없이 처리되는지)
        output_file = tmp_path / "empty_portfolio"
        save_artifact(selected_df, output_file, formats=["csv"])

        # 파일이 생성되었는지 확인
        assert (output_file.with_suffix(".csv")).exists()

        # 빈 파일 로드
        loaded_df = load_artifact(output_file)
        assert len(loaded_df) == 0

        print("에러 처리 파이프라인 테스트 성공!")

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_parametrized_pipeline(self, tmp_path, sample_rankings, top_k):
        """파라미터화된 파이프라인 테스트"""

        # 다양한 top_k 값으로 테스트
        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings, top_k=top_k, score_col="score"
        )

        # 결과 검증
        assert len(selected_df) <= top_k
        assert diagnostics["selected_count"] == len(selected_df)

        # 각 결과 저장
        output_file = tmp_path / f"portfolio_top{top_k}"
        save_artifact(selected_df, output_file, formats=["parquet"])

        loaded_df = load_artifact(output_file)
        assert len(loaded_df) == len(selected_df)

        print(f"파라미터화 테스트 성공 (top_k={top_k})!")


pytestmark = pytest.mark.ci
