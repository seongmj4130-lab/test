# 투트랙 구조 + L5 모델 학습 통합 시 얻을 수 있는 것

**작성일**: 2026-01-07

---

## 📊 핵심 요약

**투트랙 구조에 L5 모델 학습을 추가하면 랭킹 점수와 모델 예측값을 결합하여 더 강력한 투자 신호를 생성할 수 있습니다.**

---

## 🎯 현재 상황

### 투트랙 구조 (현재)

```
Track A (L8): 피처 가중치 기반 랭킹
  ↓
ranking_short_daily (score_total, rank_total)
ranking_long_daily (score_total, rank_total)
  ↓
Track B (L6R): 랭킹을 스코어로 변환
  ↓
rebalance_scores (score_ens = α * rank_short + (1-α) * rank_long)
  ↓
Track B (L7): 백테스트 실행
```

**특징**:
- 머신러닝 모델 없음
- 피처 가중치 기반 선형 결합
- 빠른 실행 속도

### 레거시 파이프라인 (L5 포함)

```
L5: 모델 학습 (Ridge 회귀)
  ↓
pred_short_oos (y_pred)
pred_long_oos (y_pred)
  ↓
L6: 모델 예측값을 스코어로 변환
  ↓
rebalance_scores (score_ens = weight_short * pred_short + weight_long * pred_long)
  ↓
L7: 백테스트 실행
```

**특징**:
- Ridge 회귀 모델 사용
- 데이터 기반 가중치 학습
- 모델 학습 시간 필요

---

## 🚀 투트랙 + L5 통합 시 얻을 수 있는 것

### 1. 앙상블 효과 (Ensemble)

**랭킹 점수 + 모델 예측값 결합**

```python
# 방법 1: 가중 평균
score_ens = (
    w_ranking * score_ranking +
    w_model * score_model
)

# 방법 2: 스태킹 (Stacking)
# 1단계: 랭킹 점수와 모델 예측값을 피처로 사용
# 2단계: 메타 모델이 최종 스코어 예측
```

**장점**:
- 랭킹의 안정성 + 모델의 예측력 결합
- 서로 다른 신호 소스의 상호 보완
- 성과 향상 가능성

**예시**:
```
랭킹 점수: 종목 A = 0.8 (상위)
모델 예측: 종목 A = 0.6 (중상위)
결합 스코어: 0.7 * 0.8 + 0.3 * 0.6 = 0.74 (더 안정적)
```

---

### 2. 데이터 기반 가중치 학습

**현재 투트랙**:
- 피처 가중치를 수동으로 설정 (config 파일)
- 경험과 직관에 의존

**L5 추가 시**:
- 모델이 데이터로부터 피처 가중치 자동 학습
- 객관적이고 데이터 기반 가중치

**예시**:
```python
# 현재: 수동 설정
feature_weights = {
    "volatility_60d": 0.021,  # 수동 설정
    "news_sentiment": 0.10,   # 수동 설정
}

# L5 추가: 데이터 기반 학습
model_coef = Ridge.fit(X, y).coef_  # 자동 학습
# volatility_60d: -0.15 (음의 상관관계)
# news_sentiment: +0.25 (양의 상관관계)
```

---

### 3. 비선형 관계 포착

**현재 투트랙**:
- 선형 결합만 가능 (피처 가중치 합산)
- 비선형 관계 포착 불가

**L5 추가 시 (XGBoost, RandomForest 사용)**:
- 비선형 관계 학습 가능
- 피처 간 상호작용 포착

**예시**:
```python
# 선형 결합 (현재)
score = 0.5 * volatility + 0.3 * momentum

# 비선형 관계 (L5 + XGBoost)
# 예: "변동성이 낮고 모멘텀은 높을 때" 특별히 좋은 성과
score = XGBoost.predict([volatility, momentum, volatility * momentum, ...])
```

---

### 4. 시장 환경별 적응

**현재 투트랙**:
- 고정된 피처 가중치
- 시장 환경 변화에 덜 유연

**L5 추가 시**:
- Walk-Forward CV로 시장 환경 변화에 적응
- 각 시기별로 최적 가중치 학습

**예시**:
```
상승장: 모멘텀 가중치 높음 (자동 학습)
하락장: 방어적 피처 가중치 높음 (자동 학습)
```

---

### 5. 예측력 검증 및 모니터링

**L5 모델 메트릭 활용**:
- IC (Information Coefficient): 예측력 측정
- Rank IC: 순위 예측력 측정
- Hit Ratio: 부호 일치율
- RMSE: 예측 오차

**활용 방법**:
```python
# 모델 성능이 좋을 때: 모델 예측값 가중치 증가
if model_ic > 0.05:
    w_model = 0.7
    w_ranking = 0.3
else:
    w_model = 0.3
    w_ranking = 0.7
```

---

### 6. 다양한 모델 실험

**현재 투트랙**:
- 피처 가중치만 조정 가능

**L5 추가 시**:
- Ridge, RandomForest, XGBoost 등 다양한 모델 실험
- 모델별 성과 비교 가능

**예시**:
```python
# config.yaml
l5:
  model_type: ridge      # 또는 random_forest, xgboost
  ridge_alpha: 8.0
  # 또는
  rf_n_estimators: 400
  xgb_n_estimators: 600
```

---

## 🔄 통합 시나리오

### 시나리오 1: 랭킹 + 모델 예측값 가중 평균

```python
# L6R 수정 버전
score_ranking = ranking_short_daily["score_total"]  # Track A 랭킹
score_model = pred_short_oos["y_pred"]              # L5 모델 예측

# 가중 평균
score_ens = (
    0.6 * score_ranking +  # 랭킹 60%
    0.4 * score_model     # 모델 40%
)
```

**장점**:
- 구현 간단
- 랭킹의 안정성 + 모델의 예측력 결합

---

### 시나리오 2: 스태킹 (Stacking)

```python
# 1단계: 랭킹과 모델 예측값을 피처로 사용
features = pd.DataFrame({
    "score_ranking": ranking_short_daily["score_total"],
    "score_model": pred_short_oos["y_pred"],
    "rank_ranking": ranking_short_daily["rank_total"],
    "rank_model": pred_short_oos["y_pred"].rank(),
})

# 2단계: 메타 모델이 최종 스코어 예측
meta_model = Ridge(alpha=1.0)
meta_model.fit(features, y_true)
score_ens = meta_model.predict(features)
```

**장점**:
- 더 정교한 결합
- 메타 모델이 최적 가중치 학습

---

### 시나리오 3: 조건부 결합

```python
# 모델 성능에 따라 가중치 조정
if model_ic > 0.05:  # 모델 예측력이 좋을 때
    score_ens = 0.3 * score_ranking + 0.7 * score_model
else:  # 모델 예측력이 낮을 때
    score_ens = 0.7 * score_ranking + 0.3 * score_model
```

**장점**:
- 동적 가중치 조정
- 시장 환경에 적응

---

## 📊 비교표

| 구분 | 투트랙만 | 투트랙 + L5 |
|------|---------|------------|
| **신호 소스** | 랭킹 점수만 | 랭킹 + 모델 예측값 |
| **가중치 설정** | 수동 설정 | 데이터 기반 학습 |
| **비선형 관계** | 불가 | 가능 (XGBoost, RF) |
| **앙상블 효과** | 없음 | 가능 |
| **예측력 검증** | 제한적 | IC, Rank IC 등 |
| **실행 속도** | 빠름 | 느림 (모델 학습) |
| **해석 가능성** | 높음 | 중간 (모델 해석 필요) |

---

## 🎯 구체적 이점

### 1. 성과 향상 가능성

**랭킹만 사용 시**:
- 피처 가중치가 최적이 아닐 수 있음
- 시장 환경 변화에 덜 유연

**L5 추가 시**:
- 데이터 기반 최적 가중치 학습
- 시장 환경 변화에 적응
- 성과 향상 가능성 증가

### 2. 리스크 분산

**랭킹 신호 실패 시**:
- 전체 전략 실패

**L5 추가 시**:
- 랭킹과 모델이 서로 다른 패턴 포착
- 하나가 실패해도 다른 하나가 보완
- 리스크 분산 효과

### 3. 모델 성능 모니터링

**L5 메트릭 활용**:
- IC, Rank IC로 모델 예측력 측정
- 모델 성능 저하 시 랭킹 가중치 증가
- 동적 전략 조정 가능

### 4. 다양한 실험 가능

**모델 타입 실험**:
- Ridge: 선형 관계
- RandomForest: 비선형 관계, 피처 상호작용
- XGBoost: 복잡한 비선형 관계

**결합 방식 실험**:
- 가중 평균
- 스태킹
- 조건부 결합

---

## ⚠️ 주의사항

### 1. 실행 시간 증가

**현재 투트랙**:
- L8 랭킹 생성: 수 초 ~ 수 분

**L5 추가 시**:
- L5 모델 학습: 수 분 ~ 수십 분
- 전체 실행 시간 증가

### 2. 복잡도 증가

**현재 투트랙**:
- 단순한 피처 가중치 합산
- 해석 가능성 높음

**L5 추가 시**:
- 모델 학습 과정 추가
- 모델 해석 필요
- 복잡도 증가

### 3. 과적합 위험

**L5 모델 학습 시**:
- Dev 성과는 좋지만 Holdout 성과 저하 가능
- Walk-Forward CV로 완화 필요

### 4. 유지보수 부담

**현재 투트랙**:
- 설정 파일만 수정하면 됨

**L5 추가 시**:
- 모델 학습 파이프라인 관리
- 모델 성능 모니터링
- 유지보수 부담 증가

---

## 🔧 구현 방법

### 방법 1: L6R 수정 (랭킹 + 모델 결합)

```python
# src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py 수정
def build_rebalance_scores_from_ranking_and_model(
    ranking_short_daily: pd.DataFrame,
    pred_short_oos: pd.DataFrame,  # L5 산출물 추가
    ...
):
    # 랭킹 점수
    score_ranking = ranking_short_daily["score_total"]

    # 모델 예측값
    score_model = pred_short_oos["y_pred"]

    # 결합
    w_ranking = 0.6
    w_model = 0.4
    score_ens = w_ranking * score_ranking + w_model * score_model
```

### 방법 2: 새로운 L6M 단계 추가

```python
# L6M: 랭킹 + 모델 결합 단계
def run_L6M_ensemble_scoring(
    ranking_short_daily: pd.DataFrame,
    ranking_long_daily: pd.DataFrame,
    pred_short_oos: pd.DataFrame,
    pred_long_oos: pd.DataFrame,
    ...
):
    # 랭킹 스코어
    score_ranking = ...

    # 모델 스코어
    score_model = ...

    # 앙상블
    score_ens = ensemble(score_ranking, score_model)
```

---

## 📝 결론

**투트랙 구조에 L5 모델 학습을 추가하면:**

1. ✅ **앙상블 효과**: 랭킹 + 모델 예측값 결합
2. ✅ **데이터 기반 가중치**: 객관적이고 최적화된 가중치
3. ✅ **비선형 관계 포착**: XGBoost, RandomForest 사용 시
4. ✅ **시장 환경 적응**: Walk-Forward CV로 시기별 최적화
5. ✅ **예측력 검증**: IC, Rank IC 등으로 모델 성능 측정
6. ✅ **다양한 실험**: 여러 모델 타입 및 결합 방식 실험

**단점**:
- 실행 시간 증가
- 복잡도 증가
- 과적합 위험
- 유지보수 부담

**권장 사항**:
- 초기에는 투트랙만 사용 (빠르고 간단)
- 성과 개선이 필요할 때 L5 추가 검토
- L5 추가 시 Walk-Forward CV로 과적합 방지 필수
