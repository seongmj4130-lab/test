# 투트랙 구조에서 머신러닝 모델 사용 현황

**작성일**: 2026-01-07

---

## 📊 핵심 결론

**투트랙 구조에서는 머신러닝 모델을 사용하지 않습니다.**

- **Track A**: 피처 가중치 기반 선형 결합 방식으로 랭킹 점수 계산
- **Track B**: 머신러닝 모델 없음 (랭킹 데이터를 백테스트용 스코어로 변환 및 백테스트 실행만 수행)

---

## 🔍 투트랙 구조 (Track A/B)

### Track A: 랭킹 엔진 (L8)

**사용 방식**: 머신러닝 모델 없음 ❌

**점수 계산 방식**: 피처 가중치 합산 (선형 결합)

```python
# src/components/ranking/score_engine.py:363-366
score_total = pd.Series(0.0, index=out.index)
for feat, normalized_values in normalized_features.items():
    weight = feature_weights.get(feat, 0.0)
    score_total += weight * normalized_values.fillna(0.0)
```

**공식**:
```
score_total = Σ (feature_weight[i] × normalized_feature[i])
```

**특징**:
- 모델 학습 과정 없음
- 피처 가중치를 직접 설정 (config 파일에서)
- 규칙 기반 방식 (Rule-based)
- 매우 빠른 실행 속도

**예시**:
```python
# 피처 가중치 예시 (configs/feature_weights_short_hitratio_optimized.yaml)
feature_weights = {
    "volatility_60d": 0.021,
    "momentum_rank": 0.021,
    "news_sentiment": 0.10,
    "roe": 0.062,
    ...
}

# 점수 계산
score_total = (
    0.021 * normalized_volatility_60d +
    0.021 * normalized_momentum_rank +
    0.10 * normalized_news_sentiment +
    0.062 * normalized_roe +
    ...
)
```

---

## 🎯 Track B: 투자 모델 (L6R, L7)

### L6R: 랭킹 스코어 변환

**사용 방식**: 머신러닝 모델 없음 ❌

**역할**: Track A의 랭킹 데이터를 백테스트용 스코어로 변환

```python
# src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py:340-350
# 단기/장기 랭킹 결합 (α 가중치)
score_ens = alpha_short * score_total_short + (1 - alpha_short) * score_total_long
```

**처리 과정**:
1. 랭킹 데이터 필터링 (rebalance_interval 적용)
2. 단기/장기 랭킹 결합 (α 가중치)
3. 리밸런싱 날짜/phase 매핑
4. rebalance_scores 생성

**특징**:
- 모델 학습 없음
- 단순 데이터 변환 및 결합
- 매우 빠른 실행 속도

### L7: 백테스트 실행

**사용 방식**: 머신러닝 모델 없음 ❌

**역할**: rebalance_scores를 사용하여 포지션 선택 및 성과 계산

```python
# src/tracks/track_b/stages/backtest/l7_backtest.py
# 포지션 선택 (상위 top_k 종목)
selected = df.nlargest(top_k, score_col)

# 수익률 계산
returns = (selected[ret_col] - cost) * position_weight

# 성과 지표 계산
sharpe = mean(returns) / std(returns) * sqrt(252)
```

**처리 과정**:
1. 포지션 선택 (상위 top_k 종목)
2. 수익률 계산 (거래비용 반영)
3. 성과 지표 계산 (Sharpe, MDD, CAGR 등)

**특징**:
- 모델 학습 없음
- 규칙 기반 포지션 선택
- 수익률 및 성과 지표 계산만 수행

---

## 🔄 레거시 파이프라인 (L0~L7)

### L5: 모델 학습

**사용 모델**: Ridge 회귀 (L2 정규화 선형 회귀)

```python
# src/stages/modeling/l5_train_models.py:265-269
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=8.0)),  # Ridge 회귀 모델
])
```

**학습 과정**:
1. Walk-Forward CV 각 fold별로 모델 학습
2. 타깃 변환: Cross-Sectional Rank 변환
3. 피처 전처리: 결측치 처리, 표준화
4. 모델 학습: Ridge 회귀 (alpha=8.0)
5. 예측: OOS 예측값 생성

**공식**:
```
y_pred = Ridge(X_train, y_train).predict(X_test)
```

**특징**:
- 모델 학습 과정 필요 (수 분 ~ 수십 분)
- 데이터로부터 가중치 자동 학습
- 일반화 성능 향상 (L2 정규화)
- 예측값 기반 투자 신호 생성

---

## 📊 비교표

| 구분 | 투트랙 구조 | 레거시 파이프라인 (L5) |
|------|-----------|---------------------|
| **Track A (L8)** | | |
| - 머신러닝 모델 | ❌ 사용 안 함 | - |
| - 점수 계산 | 피처 가중치 합산 | - |
| - 가중치 설정 | 수동 설정 (config) | - |
| **Track B (L6R, L7)** | | |
| - 머신러닝 모델 | ❌ 사용 안 함 | - |
| - 역할 | 랭킹 변환 + 백테스트 | - |
| **레거시 (L5)** | | |
| - 머신러닝 모델 | - | ✅ Ridge 회귀 |
| - 점수 계산 | - | 모델 예측값 |
| - 가중치 설정 | - | 자동 학습 (데이터) |
| - 학습 과정 | - | 필수 (Walk-Forward CV) |
| **실행 속도** | 매우 빠름 (수 초) | 느림 (수 분 ~ 수십 분) |
| **유연성** | 가중치 직접 조정 가능 | 모델 재학습 필요 |
| **해석 가능성** | 높음 (가중치 명확) | 중간 (모델 계수 해석) |

---

## 🤔 왜 투트랙에서는 모델을 사용하지 않나요?

### 1. 목적의 차이

**레거시 파이프라인**:
- 목적: 수익률을 직접 예측하는 모델 학습
- 접근: 데이터 기반 자동 학습

**투트랙 구조**:
- 목적: 피처 기반 랭킹 제공 (이용자에게 정보 제공)
- 접근: 명시적 가중치로 투명한 랭킹 생성

### 2. 실행 속도

**투트랙**:
- 모델 학습 없이 즉시 랭킹 생성 가능
- 실시간 랭킹 업데이트 용이

**레거시**:
- 모델 학습 시간 필요
- 실시간 업데이트 어려움

### 3. 해석 가능성

**투트랙**:
- 피처 가중치가 명확하게 설정됨
- "왜 이 종목이 상위인가?" 설명 가능

**레거시**:
- 모델 계수는 학습 결과이지만 해석이 상대적으로 어려움

### 4. 유연성

**투트랙**:
- 가중치를 직접 조정하여 랭킹 특성 변경 가능
- 다양한 투자 성향에 맞춘 랭킹 제공 가능

**레거시**:
- 모델 재학습 없이는 특성 변경 어려움

---

## 🔗 L5와의 관계

### L5는 선택적으로 사용 가능

**현재 투트랙 구조**:
- L5의 **피처 리스트만** 사용
- L5의 **모델 학습은** 사용하지 않음

**코드 위치**:
```python
# src/tracks/track_a/stages/ranking/l8_dual_horizon.py:45-52
l5 = cfg.get("l5", {}) or {}
feature_list_short = l5.get("feature_list_short")  # 피처 리스트만 사용
feature_list_long = l5.get("feature_list_long")   # 피처 리스트만 사용
```

**이유**:
- L5에서 정의한 피처 리스트를 재사용하여 일관성 유지
- 모델 학습은 하지 않지만, 같은 피처를 사용하여 비교 가능

---

## 📝 결론

**투트랙 구조에서는 머신러닝 모델을 사용하지 않습니다.**

대신:
- **피처 가중치 기반 선형 결합** 방식 사용
- **규칙 기반 (Rule-based)** 접근
- **명시적 가중치 설정**으로 투명한 랭킹 생성

**레거시 파이프라인에서는**:
- **Ridge 회귀 모델** 사용 (L5)
- **데이터 기반 자동 학습** 방식

두 방식은 서로 다른 목적과 접근 방식을 가지고 있으며, 각각의 장단점이 있습니다.

