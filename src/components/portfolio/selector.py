# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/components/portfolio/selector.py
"""
[Stage13] Top-K 선택 로직 with Fallback 및 Drop Reason 추적
- top_k=20인데 실제 K_eff가 8~13개로 떨어지는 근원 원인을 데이터 기반으로 로깅
- 가능한 범위에서 K_eff를 20에 가깝게 복원 (fallback 적용)
- drop reason을 수치로 남김
"""
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

def select_topk_with_fallback(
    g_sorted: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    score_col: str = "score_ens",
    top_k: int = 20,
    buffer_k: int = 0,
    prev_holdings: List[str] = None,
    group_col: Optional[str] = None,
    max_names_per_group: Optional[int] = None,
    required_cols: Optional[List[str]] = None,
    filter_missing_price: bool = True,
    filter_suspended: bool = True,
    smart_buffer_enabled: bool = False,
    smart_buffer_stability_threshold: float = 0.7,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Top-K 선택 with Fallback 및 Drop Reason 추적
    
    Args:
        g_sorted: score desc 정렬된 DataFrame (필수 컬럼: ticker_col, score_col)
        ticker_col: 티커 컬럼명
        score_col: 점수 컬럼명
        top_k: 목표 선택 종목 수
        buffer_k: 버퍼 종목 수 (prev_holdings 유지용)
        prev_holdings: 이전 보유 종목 리스트
        group_col: 그룹 컬럼명 (예: "sector_name", 업종 분산 제약용)
        max_names_per_group: 그룹당 최대 종목 수
        required_cols: 필수 컬럼 리스트 (결측 시 제외)
        filter_missing_price: 가격 결측 필터링 여부
        filter_suspended: 거래정지 필터링 여부
    
    Returns:
        (selected_df, diagnostics_dict)
        - selected_df: 선택된 종목 DataFrame
        - diagnostics_dict: {
            "eligible_count": int,  # 필터 전 전체 후보 수
            "selected_count": int,  # 최종 선택된 수 (K_eff)
            "dropped_missing": int,  # 결측으로 제외된 수
            "dropped_filter": int,  # 필터로 제외된 수
            "dropped_sectorcap": int,  # 업종 제한으로 제외된 수
            "filled_from_next_rank": int,  # 다음 순위로 채운 수
            "drop_reasons": Dict[str, int],  # 상세 drop reason
        }
    """
    if prev_holdings is None:
        prev_holdings = []
    
    if required_cols is None:
        required_cols = []
    
    top_k = int(top_k)
    buffer_k = int(buffer_k)
    
    # 초기 통계
    eligible_count = len(g_sorted)
    dropped_missing = 0
    dropped_filter = 0
    dropped_sectorcap = 0
    filled_from_next_rank = 0
    drop_reasons: Dict[str, int] = {}
    
    # 1) 필수 컬럼 결측 필터링
    g_filtered = g_sorted.copy()
    if required_cols:
        before = len(g_filtered)
        g_filtered = g_filtered.dropna(subset=required_cols)
        dropped_missing += before - len(g_filtered)
        drop_reasons["missing_required_cols"] = before - len(g_filtered)
    
    # 2) 가격 결측 필터링 (ret_col 등)
    if filter_missing_price:
        # ret_col 또는 price 관련 컬럼 확인
        price_cols = [c for c in g_filtered.columns if "ret" in c.lower() or "price" in c.lower() or "close" in c.lower()]
        if price_cols:
            before = len(g_filtered)
            g_filtered = g_filtered.dropna(subset=price_cols[:1])  # 첫 번째 가격 컬럼만 체크
            dropped_missing += before - len(g_filtered)
            drop_reasons["missing_price"] = before - len(g_filtered)
    
    # 3) 거래정지 필터링 (suspended, delisted 등)
    if filter_suspended:
        suspended_cols = [c for c in g_filtered.columns if "suspended" in c.lower() or "delisted" in c.lower() or "status" in c.lower()]
        if suspended_cols:
            before = len(g_filtered)
            # suspended=True 또는 status != "active" 제외
            for col in suspended_cols:
                if g_filtered[col].dtype == bool:
                    g_filtered = g_filtered[~g_filtered[col]]
                elif g_filtered[col].dtype == object:
                    g_filtered = g_filtered[g_filtered[col].astype(str).str.lower() == "active"]
            dropped_filter += before - len(g_filtered)
            drop_reasons["suspended"] = before - len(g_filtered)
    
    # 4) 업종 분산 제약 적용 (있는 경우)
    selected = []
    selected_set = set()
    group_counts: Dict[str, int] = {}
    
    allow_n = top_k + buffer_k if buffer_k > 0 else top_k
    allow = g_filtered.head(allow_n).copy()
    
    # 이전 보유 종목 중 허용 범위에 있는 것들
    allow_set = set(allow[ticker_col].astype(str).tolist())
    total_count = len(g_filtered)
    
    # [Phase 8 Step 2 방안1] 스마트 버퍼링: 안정성 임계값 기반 필터링
    if smart_buffer_enabled and buffer_k > 0 and len(prev_holdings) > 0:
        keep = []
        for t in prev_holdings:
            if t in allow_set:
                # 해당 종목의 현재 순위 확인
                ticker_rows = g_filtered[g_filtered[ticker_col].astype(str) == t]
                if len(ticker_rows) > 0:
                    rank = ticker_rows.index[0]  # 첫 번째 매치의 인덱스
                    # 전체 리스트에서의 순위 비율 계산 (0~1, 낮을수록 상위)
                    rank_pct = float(rank) / max(total_count - 1, 1)
                    # 순위가 상위 X% 내에 있으면 유지 (낮을수록 상위)
                    if rank_pct <= smart_buffer_stability_threshold:
                        keep.append(t)
    else:
        # 기존 방식: 허용 범위 내의 모든 보유 종목 유지
        keep = [t for t in prev_holdings if t in allow_set]
    
    # cap keep to top_k
    if len(keep) > top_k:
        keep = keep[:top_k]
    
    # 5) 업종 분산 제약이 있으면 적용
    if group_col and max_names_per_group and group_col in allow.columns:
        # keep 먼저 (업종 제약 고려)
        for t in keep:
            ticker_row = allow[allow[ticker_col].astype(str) == t]
            if len(ticker_row) > 0:
                sector = str(ticker_row.iloc[0][group_col]) if pd.notna(ticker_row.iloc[0][group_col]) else "기타"
                current_count = group_counts.get(sector, 0)
                if current_count < max_names_per_group:
                    selected.append(t)
                    selected_set.add(t)
                    group_counts[sector] = current_count + 1
                else:
                    dropped_sectorcap += 1
                    drop_reasons[f"sectorcap_{sector}"] = drop_reasons.get(f"sectorcap_{sector}", 0) + 1
        
        # 부족한 만큼 상위에서 채움 (업종 제약 고려)
        for _, row in allow.iterrows():
            if len(selected) >= top_k:
                break
            
            t = str(row[ticker_col])
            if t in selected_set:
                continue
            
            sector = str(row[group_col]) if pd.notna(row[group_col]) else "기타"
            current_count = group_counts.get(sector, 0)
            
            if current_count < max_names_per_group:
                selected.append(t)
                selected_set.add(t)
                group_counts[sector] = current_count + 1
            else:
                dropped_sectorcap += 1
                drop_reasons[f"sectorcap_{sector}"] = drop_reasons.get(f"sectorcap_{sector}", 0) + 1
    else:
        # 업종 분산 제약 없음
        # keep 먼저
        for t in keep:
            selected.append(t)
            selected_set.add(t)
        
        # 부족한 만큼 상위에서 채움
        for _, row in allow.iterrows():
            if len(selected) >= top_k:
                break
            
            t = str(row[ticker_col])
            if t in selected_set:
                continue
            
            selected.append(t)
            selected_set.add(t)
    
    # 6) Fallback: 부족한 만큼 다음 순위에서 채우기
    if len(selected) < top_k:
        # allow_n 이후의 후보들
        fallback_candidates = g_filtered.iloc[allow_n:].copy()
        
        # 업종 분산 제약이 있으면 적용
        if group_col and max_names_per_group and group_col in fallback_candidates.columns:
            for _, row in fallback_candidates.iterrows():
                if len(selected) >= top_k:
                    break
                
                t = str(row[ticker_col])
                if t in selected_set:
                    continue
                
                sector = str(row[group_col]) if pd.notna(row[group_col]) else "기타"
                current_count = group_counts.get(sector, 0)
                
                if current_count < max_names_per_group:
                    selected.append(t)
                    selected_set.add(t)
                    group_counts[sector] = current_count + 1
                    filled_from_next_rank += 1
        else:
            # 업종 분산 제약 없음
            for _, row in fallback_candidates.iterrows():
                if len(selected) >= top_k:
                    break
                
                t = str(row[ticker_col])
                if t in selected_set:
                    continue
                
                selected.append(t)
                selected_set.add(t)
                filled_from_next_rank += 1
    
    # final safety: never exceed top_k
    if len(selected) > top_k:
        selected = selected[:top_k]
    
    # 선택된 DataFrame 생성
    selected_df = g_sorted[g_sorted[ticker_col].astype(str).isin(selected)].copy()
    selected_df = selected_df.sort_values([score_col, ticker_col], ascending=[False, True]).reset_index(drop=True)
    
    # Diagnostics
    diagnostics = {
        "eligible_count": eligible_count,
        "selected_count": len(selected),
        "dropped_missing": dropped_missing,
        "dropped_filter": dropped_filter,
        "dropped_sectorcap": dropped_sectorcap,
        "filled_from_next_rank": filled_from_next_rank,
        "drop_reasons": drop_reasons,
    }
    
    return selected_df, diagnostics
