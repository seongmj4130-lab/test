# rebalance_interval 개선 구현 완료

## 구현 내용

### 1. L6R에서 rebalance_interval 지원 추가

**파일**: `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py`

**주요 변경사항**:
1. `build_rebalance_scores_from_ranking` 함수에 `rebalance_interval` 파라미터 추가
2. `_map_date_to_phase` 함수 추가: 날짜를 가장 가까운 `cv_folds_short.test_end`의 phase로 매핑
3. 리밸런싱 날짜 결정 로직 개선:
   - `rebalance_interval = 1`: 기존 로직 (월별, `cv_folds_short.test_end` 사용)
   - `rebalance_interval > 1`: 일별 랭킹 데이터에서 N일마다 필터링

### 2. config.yaml 설정 추가

**파일**: `configs/config.yaml`

```yaml
l6r:
  alpha_short: 0.5
  alpha_long: null
  # [rebalance_interval 개선] 리밸런싱 주기 설정
  # 1: 월별 리밸런싱 (cv_folds_short.test_end 사용, 기본값)
  # >1: 일별 랭킹 데이터에서 N일마다 필터링 (예: 20=20일마다, 120=120일마다)
  rebalance_interval: 1
  regime_alpha:
    ...
```

## 동작 방식

### rebalance_interval = 1 (기본값, 월별)
- 기존 로직과 동일
- `cv_folds_short.test_end`를 사용하여 월별 리밸런싱 날짜 결정
- 약 105개 리밸런싱 날짜

### rebalance_interval > 1 (일별 필터링)
- 일별 랭킹 데이터에서 `rebalance_interval`에 맞게 필터링
- Dual Horizon 모드: `ranking_short_daily` 사용
- 단일 랭킹 모드: `ranking_daily` 사용
- Phase는 `cv_folds_short`에서 가장 가까운 `test_end`로 매핑

**예시**:
- `rebalance_interval = 20`: 20일마다 리밸런싱 (약 123개 날짜, 전체 2,458개 중)
- `rebalance_interval = 120`: 120일마다 리밸런싱 (약 20개 날짜)

## 사용 방법

### 1. config.yaml 설정

```yaml
l6r:
  rebalance_interval: 20  # 20일마다 리밸런싱
```

### 2. L6R 재실행

```bash
python -m src.pipeline.track_b_pipeline bt20_short
```

### 3. 결과 확인

- `rebalance_scores`의 날짜 수가 `rebalance_interval`에 맞게 변경됨
- 로그에서 "[L6R rebalance_interval] 일별 리밸런싱 사용" 메시지 확인

## 주의사항

1. **Phase 매핑**: 일별 데이터 사용 시 `cv_folds_short`에서 가장 가까운 `test_end`로 phase를 매핑합니다
2. **Dual Horizon 모드**: `ranking_short_daily`가 필수입니다
3. **단일 랭킹 모드**: `ranking_daily`가 필수입니다 (자동 생성됨)
4. **L7의 rebalance_interval**: L7에서도 `rebalance_interval` 필터링이 있지만, L6R에서 이미 필터링된 데이터를 사용하므로 이중 필터링이 발생할 수 있습니다. L7의 `rebalance_interval`은 1로 설정하는 것을 권장합니다.

## 테스트

### 테스트 케이스 1: rebalance_interval = 1 (기본값)
```yaml
l6r:
  rebalance_interval: 1
```
- 예상: 월별 리밸런싱 (약 105개 날짜)
- 확인: 로그에서 "[L6R rebalance_interval] 월별 리밸런싱 사용" 메시지

### 테스트 케이스 2: rebalance_interval = 20
```yaml
l6r:
  rebalance_interval: 20
```
- 예상: 20일마다 리밸런싱 (약 123개 날짜)
- 확인: 로그에서 "[L6R rebalance_interval] 일별 리밸런싱 사용 (interval=20)" 메시지

### 테스트 케이스 3: rebalance_interval = 120
```yaml
l6r:
  rebalance_interval: 120
```
- 예상: 120일마다 리밸런싱 (약 20개 날짜)
- 확인: 로그에서 "[L6R rebalance_interval] 일별 리밸런싱 사용 (interval=120)" 메시지

## 기대 효과

1. **rebalance_interval이 제대로 작동**: 일별 랭킹 데이터를 활용하여 원하는 주기로 리밸런싱 가능
2. **유연성 증가**: config에서 쉽게 조정 가능
3. **기존 호환성 유지**: `rebalance_interval = 1`이 기본값이므로 기존 동작 유지
