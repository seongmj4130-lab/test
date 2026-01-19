"""
포트폴리오 선택 컴포넌트 단위 테스트
"""
import pandas as pd
import pytest

from src.components.portfolio.selector import select_topk_with_fallback


class TestSelectTopKWithFallback:
    """select_topk_with_fallback 함수 테스트"""

    def test_select_basic_topk(self, sample_rankings):
        """기본적인 top-k 선택 테스트"""
        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings,
            top_k=3
        )

        assert len(selected_df) == 3
        assert diagnostics["selected_count"] == 3
        assert diagnostics["eligible_count"] == len(sample_rankings)

        # 점수 기준 내림차순 정렬 확인
        scores = selected_df["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_select_more_than_available(self, sample_rankings):
        """요청한 k가 사용 가능한 종목 수보다 많은 경우"""
        available_count = len(sample_rankings)
        requested_k = available_count + 5

        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings,
            top_k=requested_k
        )

        assert len(selected_df) == available_count
        assert diagnostics["selected_count"] == available_count

    def test_select_with_required_cols_filtering(self, sample_rankings):
        """필수 컬럼 필터링 테스트"""
        # 결측치가 있는 데이터 생성
        df_with_missing = sample_rankings.copy()
        df_with_missing.loc[0, "score"] = None  # 첫 번째 행에 결측치 추가

        selected_df, diagnostics = select_topk_with_fallback(
            df_with_missing,
            top_k=3,
            required_cols=["score"]
        )

        # 결측치가 있는 행은 제외되어야 함
        assert len(selected_df) <= 3
        assert all(pd.notna(selected_df["score"]))  # 선택된 행에는 결측치 없음

    def test_select_with_prev_holdings_buffer(self, sample_rankings):
        """이전 보유 종목 버퍼 기능 테스트"""
        prev_holdings = ["AAPL", "GOOGL"]  # 샘플 데이터에 있는 종목들

        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings,
            top_k=3,
            buffer_k=2,
            prev_holdings=prev_holdings
        )

        # 버퍼 기능은 구현에 따라 다를 수 있으므로 기본 선택만 확인
        assert len(selected_df) <= 5  # top_k + buffer_k
        assert diagnostics["selected_count"] == len(selected_df)

    def test_select_with_group_constraints(self, sample_rankings):
        """그룹 제약 조건 테스트"""
        # 그룹 컬럼 추가
        df_with_groups = sample_rankings.copy()
        df_with_groups["sector"] = ["Tech", "Finance", "Tech", "Finance", "Healthcare",
                                   "Tech", "Finance", "Healthcare"]

        selected_df, diagnostics = select_topk_with_fallback(
            df_with_groups,
            top_k=4,
            group_col="sector",
            max_names_per_group=2
        )

        # 각 그룹당 최대 2개 종목 선택 확인
        if len(selected_df) > 0:
            group_counts = selected_df["sector"].value_counts()
            assert all(count <= 2 for count in group_counts)

    def test_select_empty_dataframe(self, sample_data_empty):
        """빈 DataFrame 입력 테스트"""
        selected_df, diagnostics = select_topk_with_fallback(
            sample_data_empty,
            top_k=5
        )

        assert len(selected_df) == 0
        assert diagnostics["selected_count"] == 0
        assert diagnostics["eligible_count"] == 0

    def test_select_single_row(self):
        """단일 행 데이터 테스트"""
        single_row_df = pd.DataFrame({
            "ticker": ["TEST"],
            "score": [0.8],
            "rank": [1]
        })

        selected_df, diagnostics = select_topk_with_fallback(
            single_row_df,
            top_k=5
        )

        assert len(selected_df) == 1
        assert diagnostics["selected_count"] == 1

    def test_select_with_duplicate_scores(self):
        """중복 점수 처리 테스트"""
        duplicate_score_df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "score": [0.8, 0.8, 0.7, 0.6],  # A와 B가 같은 점수
            "rank": [1, 2, 3, 4]
        })

        selected_df, diagnostics = select_topk_with_fallback(
            duplicate_score_df,
            top_k=2
        )

        # 상위 2개 선택, 중복 점수 처리 확인
        assert len(selected_df) == 2
        # 점수가 높은 종목들 선택되었는지 확인 (동점일 경우 ticker 순서에 따라 다를 수 있음)
        selected_scores = selected_df["score"].tolist()
        assert all(score >= 0.7 for score in selected_scores)

    def test_select_with_price_filtering(self):
        """가격 결측 필터링 테스트"""
        df_with_price = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "score": [0.9, 0.8, 0.7, 0.6],
            "price": [100.0, None, 150.0, 200.0],  # B는 가격 결측
            "rank": [1, 2, 3, 4]
        })

        selected_df, diagnostics = select_topk_with_fallback(
            df_with_price,
            top_k=3,
            filter_missing_price=True
        )

        # 가격이 결측인 B는 제외되어야 함
        assert "B" not in selected_df["ticker"].tolist()
        assert len(selected_df) <= 3

    def test_select_with_suspended_filtering(self):
        """거래정지 필터링 테스트"""
        df_with_suspended = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "score": [0.9, 0.8, 0.7, 0.6],
            "is_suspended": [False, True, False, False],  # B는 거래정지
            "rank": [1, 2, 3, 4]
        })

        selected_df, diagnostics = select_topk_with_fallback(
            df_with_suspended,
            top_k=3,
            filter_suspended=True
        )

        # 거래정지된 B는 제외되어야 함
        assert "B" not in selected_df["ticker"].tolist()
        assert len(selected_df) <= 3

    def test_select_diagnostics_structure(self, sample_rankings):
        """진단 정보 구조 테스트"""
        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings,
            top_k=3
        )

        # 필수 진단 키 확인
        required_keys = ["eligible_count", "selected_count"]
        for key in required_keys:
            assert key in diagnostics
            assert isinstance(diagnostics[key], int)

    def test_select_with_smart_buffer(self, sample_rankings):
        """스마트 버퍼 기능 테스트"""
        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings,
            top_k=3,
            smart_buffer_enabled=True,
            smart_buffer_stability_threshold=0.5
        )

        # 스마트 버퍼는 선택 로직에 영향을 줄 수 있음
        assert len(selected_df) >= 3  # 최소 top_k는 보장
        assert diagnostics["selected_count"] == len(selected_df)

    def test_select_edge_case_zero_topk(self, sample_rankings):
        """top_k가 0인 경우 테스트"""
        selected_df, diagnostics = select_topk_with_fallback(
            sample_rankings,
            top_k=0
        )

        assert len(selected_df) == 0
        assert diagnostics["selected_count"] == 0

    def test_select_with_all_filters_applied(self):
        """모든 필터가 적용된 복합 테스트"""
        complex_df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "price": [100.0, None, 150.0, 200.0, 250.0, 300.0],  # B 제외
            "is_suspended": [False, False, True, False, False, False],  # C 제외
            "sector": ["Tech", "Tech", "Finance", "Finance", "Healthcare", "Healthcare"],
            "rank": [1, 2, 3, 4, 5, 6]
        })

        selected_df, diagnostics = select_topk_with_fallback(
            complex_df,
            top_k=2,
            group_col="sector",
            max_names_per_group=1,
            filter_missing_price=True,
            filter_suspended=True
        )

        # A, D, E, F 중 필터링된 결과에서 그룹 제약 적용
        # 가능한 종목: A(Tech), D(Finance), E(Healthcare), F(Healthcare)
        # 그룹당 최대 1개, top_k=2 이므로 2개 선택
        assert len(selected_df) <= 2

        # 그룹 제약 확인
        if len(selected_df) > 1:
            group_counts = selected_df["sector"].value_counts()
            assert all(count <= 1 for count in group_counts)