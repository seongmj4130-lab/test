# Grid Search의 Dev/Holdout 분리 이유

## 📚 핵심 개념

**Grid Search도 ML 모델과 동일한 이유로 Dev/Holdout을 나눕니다.**

## 🔍 ML 모델의 Dev/Holdout 분리 이유 (참고)

### ML 모델의 경우:
1. **Dev 구간 (Development Set)**
   - 하이퍼파라미터 튜닝 (예: Ridge alpha, learning rate)
   - 모델 선택 (예: Ridge vs Lasso)
   - 피처 선택
   - **목적**: 모델을 "학습"하고 "튜닝"하는 구간

2. **Holdout 구간 (Test Set)**
   - 튜닝에 **사용하지 않은** 완전히 독립적인 데이터
   - 최종 모델의 **일반화 성능** 평가
   - **목적**: 실제 운용 시 성과 예측

### 왜 나누는가?
- **과적합 방지**: Dev에서 튜닝한 모델이 Dev에만 최적화되어 실제 성능이 저하되는 것을 방지
- **편향 없는 평가**: 튜닝 과정에서 본 적이 없는 데이터로 평가해야 공정함

---

## 🎯 Grid Search의 Dev/Holdout 분리 이유

### Grid Search도 동일한 원리 적용:

1. **Dev 구간 (Grid Search 실행)**
   - **목적**: 최적 가중치 조합 **선택** (튜닝)
   - **과정**:
     - 여러 가중치 조합을 Dev 구간에서 평가
     - Objective Score가 가장 높은 조합 선택
   - **예시**:
     - 조합 1: technical=-0.5, value=0.5 → Score: 0.40
     - 조합 2: technical=-1.0, value=0.0 → Score: 0.35
     - 조합 3: technical=-0.5, value=0.5 → Score: 0.41 ✅ **최적 선택**

2. **Holdout 구간 (최종 평가)**
   - **목적**: 선택된 최적 가중치의 **일반화 성능** 평가
   - **과정**:
     - Dev에서 선택한 최적 가중치를 Holdout에 적용
     - Holdout에서 성과 평가
   - **중요**: Grid Search 과정에서 **사용하지 않은** 데이터

### 왜 나누는가?

#### 1. **과적합 방지**
```
Dev 구간에서 최적 조합 선택
  ↓
이 조합이 Dev에만 최적화되었을 가능성
  ↓
Holdout에서 평가하여 실제 일반화 성능 확인
```

**예시**:
- Dev에서 최적 조합: IC Mean = 0.0200 ✅
- Holdout에서 동일 조합: IC Mean = -0.0009 ❌
- **결론**: Dev에 과적합되었음 (HIGH 위험도)

#### 2. **편향 없는 평가**
- Grid Search로 **80개 조합을 모두 Dev에서 평가**
- 이 과정에서 Dev 구간의 특성을 "학습"했을 수 있음
- Holdout은 이 과정에 **전혀 관여하지 않음** → 공정한 평가

#### 3. **실제 운용 성과 예측**
- Dev에서 선택한 가중치가 **미래 데이터**에서도 작동할지 확인
- Holdout은 "미래"를 시뮬레이션하는 역할

---

## 📊 현재 구현 방식

### Phase 2 Grid Search 프로세스:

```
1. Grid Search 실행 (Dev 구간만 사용)
   ├─ 80개 가중치 조합 생성
   ├─ 각 조합을 Dev 구간에서 평가
   │  ├─ Hit Ratio 계산
   │  ├─ IC Mean 계산
   │  ├─ ICIR 계산
   │  └─ Objective Score = f(Hit Ratio, IC, ICIR)
   └─ 최적 조합 선택 (Objective Score 최대)

2. 최적 가중치 적용 (Holdout 구간 평가)
   ├─ 선택된 최적 가중치로 L8 랭킹 생성
   ├─ Holdout 구간에서 성과 평가
   │  ├─ Hit Ratio 계산
   │  ├─ IC Mean 계산
   │  ├─ ICIR 계산
   └─ Dev/Holdout 성과 비교

3. 과적합 분석
   ├─ Dev 성과 vs Holdout 성과 비교
   ├─ 차이가 크면 → 과적합 위험 HIGH
   └─ 권장사항 제시
```

### 코드에서의 구현:

```python
# 1. Grid Search (Dev 구간만 사용)
def evaluate_group_weights(...):
    # cv_folds에서 Dev 구간만 필터링
    dev_folds = cv_folds[cv_folds["fold_id"] != "holdout"]
    dev_dates = dev_folds["test_end"].unique()

    # Dev 구간 데이터만 사용
    ranking_daily = ranking_daily[ranking_daily["date"].isin(dev_dates)]
    forward_returns = forward_returns[forward_returns["date"].isin(dev_dates)]

    # 평가 및 점수 계산
    metrics = calculate_metrics(...)
    return objective_score

# 2. Holdout 평가 (별도 스크립트)
def evaluate_holdout_performance(...):
    # 최적 가중치로 랭킹 생성
    ranking_daily = build_ranking_daily(..., feature_groups_config=optimal_weights)

    # Holdout 구간만 필터링
    holdout_folds = cv_folds[cv_folds["fold_id"].str.startswith("holdout")]
    holdout_dates = holdout_folds["test_end"].unique()

    # Holdout 구간 평가
    holdout_metrics = evaluate_on_fold(ranking_daily, forward_returns, "holdout")

    # Dev/Holdout 비교
    comparison = compare_dev_holdout(dev_metrics, holdout_metrics)
```

---

## ⚠️ 만약 Dev/Holdout을 나누지 않는다면?

### 문제점:

1. **과적합 위험 증가**
   ```
   전체 데이터에서 Grid Search 실행
     ↓
   최적 조합이 전체 데이터에 과적합
     ↓
   실제 운용 시 성과 저하 (예측 불가능)
   ```

2. **편향된 평가**
   - Grid Search 과정에서 이미 본 데이터로 평가
   - 실제 성능을 과대평가할 위험

3. **일반화 성능 불확실**
   - 미래 데이터에서의 성과를 예측할 수 없음

---

## ✅ 올바른 접근 방식

### 현재 구현 (올바름):

```
┌─────────────────────────────────────────┐
│  전체 데이터                            │
│  ┌──────────────┬──────────────────┐    │
│  │ Dev 구간     │ Holdout 구간    │    │
│  │ (튜닝용)     │ (평가용)        │    │
│  └──────────────┴──────────────────┘    │
└─────────────────────────────────────────┘
         │                    │
         │                    │
    Grid Search          최종 평가
    (최적 조합 선택)     (일반화 성능)
         │                    │
         └────────┬───────────┘
                  │
         Dev/Holdout 비교
         (과적합 분석)
```

### 핵심 원칙:

1. **Grid Search는 Dev에서만 실행** → 최적 조합 선택
2. **선택된 조합은 Holdout에서 평가** → 일반화 성능 확인
3. **Dev/Holdout 차이 분석** → 과적합 여부 판단

---

## 📈 실제 결과 예시

### 단기 랭킹:

| 구간 | IC Mean | ICIR | Rank ICIR |
|------|---------|------|-----------|
| **Grid Dev** (최적 조합) | 0.0200 | 0.2224 | 0.3753 |
| **Holdout** (동일 조합) | -0.0009 | -0.0057 | 0.0439 |
| **차이** | -0.0209 ⚠️ | -0.2281 ⚠️ | -0.3314 ⚠️ |

**해석**:
- Dev에서 좋은 성과를 보였지만
- Holdout에서 크게 저하됨
- **과적합 위험 HIGH** 확인

---

## 🎓 결론

**Grid Search도 ML 모델과 동일한 이유로 Dev/Holdout을 나눕니다:**

1. **Dev**: 최적 가중치 조합 **선택** (튜닝)
2. **Holdout**: 선택된 조합의 **일반화 성능** 평가
3. **비교**: 과적합 여부 확인

이는 **데이터 누수(data leakage) 방지**와 **공정한 평가**를 위한 필수 절차입니다.

---

**작성일**: 2026-01-08
**참고 문서**:
- `track_a_optimization_direction_validation.md`
- `dev_holdout_final_comparison_20260108_160851.md`
