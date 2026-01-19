# rebalance_interval 제대로 적용하기 위한 추천 방법

## 문제 상황

1. **랭킹 데이터**: 일별로 생성됨 (2,458개 날짜)
2. **rebalance_scores**: 월별로만 존재함 (105개 날짜, `cv_folds_short.test_end` 필터링 때문)
3. **L7의 rebalance_interval**: 입력 데이터가 이미 월별이므로 효과가 제한적

## 추천 방법 (우선순위 순)

### 방법 1: L6R에서 rebalance_interval 고려 (⭐ 추천)

**개요**: L6R 단계에서 `rebalance_interval` 설정을 읽어서, 일별 랭킹 데이터를 사용하되 `rebalance_interval`에 맞게 필터링

**장점**:
- L7 코드 변경 최소화
- `rebalance_interval`이 제대로 작동
- 기존 아키텍처 유지

**구현 방법**:
```python
# l6r_ranking_scoring.py 수정
def build_rebalance_scores_from_ranking(
    ...,
    rebalance_interval: int = 1,  # 추가 파라미터
):
    # 기존: cv_folds_short.test_end만 사용
    # 수정: 일별 랭킹 데이터에서 rebalance_interval에 맞게 필터링

    if rebalance_interval == 1:
        # 기존 로직: cv_folds_short.test_end 사용
        rebal_map = folds[["test_end", "phase"]].rename(columns={"test_end": "date"}).copy()
    else:
        # 새 로직: 일별 랭킹 데이터에서 rebalance_interval에 맞게 필터링
        all_dates = sorted(ranking_short_daily["date"].unique())
        rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), rebalance_interval)]
        # phase는 cv_folds_short에서 매핑
        rebal_map = pd.DataFrame({
            "date": rebalance_dates,
            "phase": ...  # cv_folds_short에서 매핑
        })
```

**config.yaml 수정**:
```yaml
l6r:
  rebalance_interval: 1  # 기본값 (월별)
  # 또는
  rebalance_interval: 20  # 20일마다 리밸런싱
```

---

### 방법 2: L7에서 랭킹 데이터 직접 사용

**개요**: L7이 `rebalance_scores` 대신 `ranking_short_daily`/`ranking_long_daily`를 직접 사용

**장점**:
- 일별 데이터 활용 가능
- L6R 단계 생략 가능 (간단한 경우)

**단점**:
- L7 코드 대폭 수정 필요
- 기존 아키텍처 변경

**구현 방법**:
```python
# l7_backtest.py 수정
def run_backtest(
    ...,
    ranking_short_daily: Optional[pd.DataFrame] = None,
    ranking_long_daily: Optional[pd.DataFrame] = None,
    rebalance_scores: Optional[pd.DataFrame] = None,  # 기존 방식
):
    if ranking_short_daily is not None:
        # 랭킹 데이터 직접 사용
        df = ranking_short_daily.copy()
        # rebalance_interval 필터링 적용
        ...
    else:
        # 기존 방식: rebalance_scores 사용
        df = rebalance_scores.copy()
        ...
```

---

### 방법 3: L6R에서 일별 데이터 포함 옵션 추가

**개요**: L6R에서 `use_daily_ranking` 옵션을 추가하여 일별 랭킹 데이터를 그대로 사용

**장점**:
- 유연성 증가
- 기존 코드와 호환

**단점**:
- 옵션이 복잡해짐

**구현 방법**:
```python
# l6r_ranking_scoring.py 수정
def build_rebalance_scores_from_ranking(
    ...,
    use_daily_ranking: bool = False,  # 일별 랭킹 사용 여부
    rebalance_interval: int = 1,  # 일별 사용 시 적용
):
    if use_daily_ranking:
        # 일별 랭킹 데이터 사용
        all_dates = sorted(ranking_short_daily["date"].unique())
        rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), rebalance_interval)]
        ...
    else:
        # 기존 로직: cv_folds_short.test_end 사용
        ...
```

---

## 최종 추천: 방법 1

**이유**:
1. **최소 변경**: L6R만 수정하면 됨
2. **명확한 책임**: L6R이 리밸런싱 날짜 결정 책임
3. **유연성**: `rebalance_interval`을 config에서 쉽게 조정 가능
4. **기존 호환성**: 기존 코드와 호환

**구현 단계**:
1. `l6r_ranking_scoring.py`에 `rebalance_interval` 파라미터 추가
2. `rebalance_interval > 1`일 때 일별 랭킹 데이터에서 필터링
3. `config.yaml`에 `l6r.rebalance_interval` 설정 추가
4. L7의 `rebalance_interval` 필터링은 제거하거나 유지 (이중 필터링 방지)

---

## 주의사항

1. **이중 필터링 방지**: L6R과 L7 모두에서 `rebalance_interval`을 적용하면 안 됨
2. **Phase 매핑**: 일별 데이터 사용 시 `cv_folds_short`에서 phase를 올바르게 매핑해야 함
3. **성능**: 일별 데이터 사용 시 메모리 사용량 증가 가능
