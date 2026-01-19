# 레거시 파이프라인 vs 투트랙 구조 차이점

**작성일**: 2026-01-07

---

## 📊 핵심 차이점 요약

| 구분 | 레거시 파이프라인 (L0~L7) | 투트랙 구조 (Track A/B) |
|------|-------------------------|----------------------|
| **신호 소스** | 모델 예측값 (L5 Ridge 회귀) | 랭킹 점수 (L8 Score Engine) |
| **모델 학습** | ✅ 필수 (L5 실행) | ❌ 선택적 (L5 피처 리스트만 사용) |
| **스코어 생성** | L6: 모델 예측값 → rebalance_scores | L6R: 랭킹 → rebalance_scores |
| **실행 방식** | 전체 파이프라인 한 번에 실행 | Track A와 Track B 독립 실행 |
| **목적** | 모델 기반 투자 전략 | 랭킹 기반 투자 전략 |

---

## 🔄 레거시 파이프라인 (L0~L7)

### 실행 방법
```bash
python scripts/run_pipeline_l0_l7.py
```

### 파이프라인 흐름

```
L0: 유니버스 구성
  ↓
L1: OHLCV 데이터 다운로드
  ↓
L2: 재무 데이터 로드
  ↓
L3: 패널 병합
  ↓
L4: Walk-Forward CV 분할
  ↓
L5: 모델 학습 (Ridge 회귀)
  ├─ 단기 모델: 20일 수익률 예측
  └─ 장기 모델: 120일 수익률 예측
  ↓
L6: 스코어 생성
  ├─ 입력: pred_short_oos, pred_long_oos (L5 산출물)
  ├─ 처리: 모델 예측값을 리밸런싱 스코어로 변환
  └─ 출력: rebalance_scores (score_short, score_long, score_ens)
  ↓
L7: 백테스트 실행
  └─ 입력: rebalance_scores (L6 산출물)
```

### 특징

1. **모델 기반**: Ridge 회귀 모델이 수익률을 직접 예측
2. **L5 필수**: 모델 학습이 반드시 필요함
3. **예측값 사용**: `pred_short_oos`, `pred_long_oos`의 `y_pred` 컬럼 사용
4. **단일 실행**: 전체 파이프라인을 한 번에 실행

### 코드 위치

- **L5**: `src/stages/modeling/l5_train_models.py`
- **L6**: `src/stages/modeling/l6_scoring.py`
  - 함수: `build_rebalance_scores()`
  - 입력: `pred_short_oos`, `pred_long_oos` (L5 산출물)
  - 처리: 모델 예측값(`y_pred`)을 리밸런싱 스코어로 변환

### L6 스코어 생성 로직

```python
# l6_scoring.py:159-193
# 1. 모델 예측값 집계 (fold별 평균)
ps1 = _agg_across_models(ps, score_col="y_pred")  # 단기
pl1 = _agg_across_models(pl, score_col="y_pred")  # 장기

# 2. 리밸런싱 날짜 선택 (fold의 test_end)
ps2 = _pick_rebalance_rows_by_fold_end(ps1)
pl2 = _pick_rebalance_rows_by_fold_end(pl1)

# 3. 단기/장기 스코어 결합
out["score_ens"] = (weight_short * score_short + weight_long * score_long)
```

---

## 🎯 투트랙 구조 (Track A/B)

### 실행 방법

```bash
# Track A: 랭킹 엔진
python -m src.pipeline.track_a_pipeline

# Track B: 투자 모델
python -m src.pipeline.track_b_pipeline bt20_short
```

### 파이프라인 흐름

#### Track A (랭킹 엔진)

```
L0~L4: 공통 데이터 준비
  ↓
L8: 랭킹 엔진
  ├─ 단기 랭킹: ranking_short_daily (score_total, rank_total)
  └─ 장기 랭킹: ranking_long_daily (score_total, rank_total)
  ↓
L11: UI Payload 생성 (선택적)
```

#### Track B (투자 모델)

```
Track A 산출물 확인
  ├─ ranking_short_daily
  └─ ranking_long_daily
  ↓
L6R: 랭킹 스코어 변환
  ├─ 입력: ranking_short_daily, ranking_long_daily
  ├─ 처리: 랭킹 점수를 리밸런싱 스코어로 변환
  └─ 출력: rebalance_scores (score_total_short, score_total_long, score_ens)
  ↓
L7: 백테스트 실행
  └─ 입력: rebalance_scores (L6R 산출물)
```

### 특징

1. **랭킹 기반**: 피처 가중치로 계산한 랭킹 점수 사용
2. **L5 선택적**: 모델 학습 없이도 동작 (L5의 피처 리스트만 사용)
3. **랭킹 점수 사용**: `ranking_short_daily`, `ranking_long_daily`의 `score_total` 컬럼 사용
4. **독립 실행**: Track A와 Track B를 독립적으로 실행 가능

### 코드 위치

- **L8**: `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`
- **L6R**: `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py`
  - 함수: `build_rebalance_scores_from_ranking()`
  - 입력: `ranking_short_daily`, `ranking_long_daily` (L8 산출물)
  - 처리: 랭킹 점수(`score_total`)를 리밸런싱 스코어로 변환

### L6R 스코어 생성 로직

```python
# l6r_ranking_scoring.py:81-516
# 1. 랭킹 데이터 필터링 (rebalance_interval 적용)
if rebalance_interval == 1:
    # 월별 리밸런싱: cv_folds_short.test_end 사용
    rebal_map = folds[["test_end", "phase"]].rename(columns={"test_end": "date"})
else:
    # 일별 리밸런싱: ranking_daily에서 interval만큼 필터링
    all_dates = sorted(ranking_short_daily["date"].unique())
    rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), rebalance_interval)]

# 2. 랭킹 점수 추출
score_short = ranking_short_daily["score_total"]  # 또는 "rank_total"
score_long = ranking_long_daily["score_total"]

# 3. 단기/장기 랭킹 결합 (α 가중치)
score_ens = alpha_short * score_short + (1 - alpha_short) * score_long
```

---

## 🔍 상세 비교

### 1. 신호 소스 차이

#### 레거시: 모델 예측값
```python
# L5에서 생성
pred_short_oos["y_pred"]  # Ridge 회귀 모델의 예측값
pred_long_oos["y_pred"]   # Ridge 회귀 모델의 예측값

# L6에서 사용
score_short = pred_short_oos["y_pred"]  # 모델 예측값 직접 사용
score_long = pred_long_oos["y_pred"]
```

#### 투트랙: 랭킹 점수
```python
# L8에서 생성
ranking_short_daily["score_total"]  # 피처 가중치 합산 점수
ranking_long_daily["score_total"]   # 피처 가중치 합산 점수

# L6R에서 사용
score_short = ranking_short_daily["score_total"]  # 랭킹 점수 사용
score_long = ranking_long_daily["score_total"]
```

### 2. 모델 학습 필요성

#### 레거시
- ✅ **L5 필수**: 모델 학습 없이는 L6 실행 불가
- 모델 학습 시간: 수 분 ~ 수십 분 (데이터 크기에 따라)

#### 투트랙
- ❌ **L5 선택적**: 모델 학습 없이도 Track A/B 실행 가능
- L5의 피처 리스트만 사용 (모델 학습은 안 함)
- 랭킹 생성 시간: 수 초 ~ 수 분 (훨씬 빠름)

### 3. 스코어 생성 방식

#### 레거시 (L6)
```python
# l6_scoring.py
def build_rebalance_scores(
    pred_short_oos: pd.DataFrame,  # L5 산출물
    pred_long_oos: pd.DataFrame,  # L5 산출물
    ...
):
    # 모델 예측값 집계
    ps1 = _agg_across_models(ps, score_col="y_pred")
    pl1 = _agg_across_models(pl, score_col="y_pred")
    
    # 리밸런싱 날짜 선택 (fold의 test_end)
    ps2 = _pick_rebalance_rows_by_fold_end(ps1)
    pl2 = _pick_rebalance_rows_by_fold_end(pl1)
    
    # 단기/장기 결합
    score_ens = weight_short * score_short + weight_long * score_long
```

#### 투트랙 (L6R)
```python
# l6r_ranking_scoring.py
def build_rebalance_scores_from_ranking(
    ranking_short_daily: pd.DataFrame,  # L8 산출물
    ranking_long_daily: pd.DataFrame,  # L8 산출물
    ...
):
    # 랭킹 데이터 필터링 (rebalance_interval 적용)
    if rebalance_interval == 1:
        rebal_map = folds[["test_end", "phase"]]
    else:
        rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), rebalance_interval)]
    
    # 랭킹 점수 추출
    score_short = ranking_short_daily["score_total"]
    score_long = ranking_long_daily["score_total"]
    
    # 단기/장기 결합 (α 가중치)
    score_ens = alpha_short * score_short + (1 - alpha_short) * score_long
```

### 4. 실행 흐름 차이

#### 레거시
```
전체 파이프라인 한 번에 실행
  ↓
L0~L7 순차 실행
  ↓
최종 결과: 백테스트 성과 지표
```

#### 투트랙
```
Track A 실행 (랭킹 생성)
  ↓
Track B 실행 (백테스트)
  ↓
최종 결과: 백테스트 성과 지표
```

---

## 📝 사용 시나리오

### 레거시 파이프라인 사용 시

**적합한 경우**:
- 모델 예측 성능을 확인하고 싶을 때
- 모델 기반 투자 전략을 테스트하고 싶을 때
- 전체 파이프라인을 한 번에 실행하고 싶을 때

**단점**:
- 모델 학습 시간이 오래 걸림
- 랭킹만 필요한 경우에도 모델 학습 필요

### 투트랙 구조 사용 시

**적합한 경우**:
- 랭킹 정보만 필요한 경우 (Track A만 실행)
- 빠르게 랭킹을 생성하고 싶을 때
- 랭킹과 백테스트를 독립적으로 관리하고 싶을 때
- 모델 학습 없이 랭킹 기반 전략을 테스트하고 싶을 때

**장점**:
- 빠른 실행 속도 (모델 학습 불필요)
- Track A와 Track B 독립 실행 가능
- 랭킹만 필요한 경우 Track A만 실행

---

## 🔗 공통점

1. **공통 데이터 준비**: L0~L4는 동일하게 사용
2. **백테스트**: L7은 동일한 백테스트 로직 사용
3. **최종 산출물**: `rebalance_scores` 형태는 동일 (컬럼명만 다름)

---

## 📊 산출물 비교

### 레거시 파이프라인

```
L5 산출물:
- pred_short_oos.parquet (y_pred, y_true, fold_id, phase, ...)
- pred_long_oos.parquet (y_pred, y_true, fold_id, phase, ...)
- model_metrics.parquet (RMSE, IC, Hit Ratio, ...)

L6 산출물:
- rebalance_scores.parquet
  - score_short (모델 예측값 기반)
  - score_long (모델 예측값 기반)
  - score_ens (단기/장기 결합)
  - true_short, true_long
```

### 투트랙 구조

```
L8 산출물:
- ranking_short_daily.parquet (score_total, rank_total, ...)
- ranking_long_daily.parquet (score_total, rank_total, ...)

L6R 산출물:
- rebalance_scores_from_ranking_interval_{N}.parquet
  - score_total_short (랭킹 점수 기반)
  - score_total_long (랭킹 점수 기반)
  - score_ens (단기/장기 결합)
  - true_short, true_long
```

---

## 🎯 결론

**레거시 파이프라인**은 **모델 기반** 접근 방식으로, Ridge 회귀 모델이 수익률을 직접 예측하여 투자 신호를 생성합니다.

**투트랙 구조**는 **랭킹 기반** 접근 방식으로, 피처 가중치로 계산한 랭킹 점수를 사용하여 투자 신호를 생성합니다.

현재 프로젝트는 **투트랙 구조를 권장**하며, 레거시 파이프라인은 하위 호환성을 위해 유지됩니다.

