# 피쳐 개선 방안 제안 보고서
**작성 일시**: 2026-01-11
**목적**: 현재 피쳐셋(단기 20개, 장기 19개)을 모두 사용하되 개선하는 방안 연구

## 📊 현재 피쳐셋 현황 분석

### 피쳐 구성
- **단기**: 20개 (Core 11개 + Short 5개 + News 4개)
- **장기**: 19개 (Core 12개 + Long 7개)

### 그룹별 IC 성과
| 그룹 | 단기 IC | 장기 IC | 피쳐 수 | 개선 필요성 |
|------|---------|---------|---------|-------------|
| **Value** | 0.0252 | 0.0342 | 5개 | 보통 |
| **Technical** | -0.0221 | -0.0271 | 20개 | 높음 |
| **Profitability** | -0.0009 | -0.0211 | 2개 | 높음 |
| **News** | 0.0000 | 0.0000 | 0개 | 높음 |

### 낮은 IC 피쳐들 (개선 우선순위)
**단기 (IC < -0.01)**:
- `ret_daily`: -0.011
- `low/high/open/close`: -0.018 ~ -0.020
- `momentum_6m`: -0.027
- `downside_volatility_60d`: -0.030
- `momentum_3m`: -0.034
- `price_momentum_*`: -0.036

**장기 (IC < -0.01)**:
- `turnover`: -0.010
- `momentum_6m`: -0.016
- `roe_sector_z`: -0.018
- `momentum_rank`: -0.022
- `price_momentum_*`: -0.022
- `roe`: -0.024

## 🚀 피쳐 개선 방안 제안

### 1. 낮은 IC 피쳐 개선/대체 방안

#### A. 가격 기반 피쳐 개선 (`low/high/open/close`)
**문제**: IC가 낮음 (-0.018 ~ -0.020)
**원인**: 절대 가격 정보로는 랭킹에 제한적 기여

**개선 방안**:
```python
# 기존: 절대 가격 사용
close, high, low, open

# 개선안: 상대적 가격 지표들
close_to_52w_high = close / close.rolling(252).max()  # 52주 최고가 대비
close_to_52w_low = close / close.rolling(252).min()   # 52주 최저가 대비
price_position = (close - low) / (high - low)         # 일중 가격 위치
price_range_ratio = (high - low) / close             # 가격 변동폭 비율
```

**기대 효과**: 가격 위치 정보로 더 유의미한 신호 생성

#### B. 모멘텀 피쳐 강화 (`momentum_3m, momentum_6m`)
**문제**: IC 낮음 (-0.034, -0.027)
**원인**: 단순 과거 수익률만 사용

**개선 방안**:
```python
# 기존: 단순 수익률
momentum_3m = (close / close.shift(63) - 1)
momentum_6m = (close / close.shift(126) - 1)

# 개선안: 가중 모멘텀 + 속도
momentum_3m_ewm = momentum_3m.ewm(span=10).mean()  # 지수 가중 평균
momentum_acceleration = momentum_3m - momentum_3m.shift(21)  # 모멘텀 가속도
momentum_robust = momentum_3m * (1 + volatility_20d)  # 변동성 조정 모멘텀
```

**기대 효과**: 모멘텀의 지속성과 강도를 더 잘 반영

#### C. 변동성 피쳐 개선 (`downside_volatility_60d`)
**문제**: IC 낮음 (-0.030)
**원인**: 단방향 변동성만 고려

**개선 방안**:
```python
# 기존: 하방 변동성만
downside_volatility_60d = returns[returns < 0].std() * sqrt(252)

# 개선안: 비대칭 변동성 지표들
upside_volatility = returns[returns > 0].std() * sqrt(252)    # 상방 변동성
volatility_skew = upside_volatility / downside_volatility     # 변동성 비대칭도
tail_risk = returns.quantile(0.05)                           # 꼬리 위험
volatility_regime = volatility_60d / volatility_60d.rolling(252).mean()  # 변동성 체제
```

**기대 효과**: 시장 상황별 변동성 특성을 더 잘 포착

### 2. 새로운 파생 피쳐 생성 방안

#### A. 시장 구조 피쳐
```python
# 시장 집중도 (상위 종목 비중)
market_concentration = (top_10_returns.abs() / total_market_cap).rolling(60).mean()

# 섹터 모멘텀 차이
sector_momentum_diff = sector_momentum - market_momentum

# 크기 효과 강화
size_effect_enhanced = market_cap * (1 + volatility_adjustment)
```

#### B. 기술적 지표 확장
```python
# RSI 기반 피쳐
rsi_14 = ta.RSI(close, 14)
rsi_divergence = rsi_14 - rsi_14.rolling(60).mean()

# 볼린저 밴드 기반
bb_position = (close - bb_lower) / (bb_upper - bb_lower)
bb_squeeze = (bb_upper - bb_lower) / close

# 거래량 기반 모멘텀
volume_momentum = volume / volume.rolling(20).mean()
volume_trend = volume_momentum.ewm(span=10).mean()
```

#### C. 재무 + 기술 결합 피쳐
```python
# 재무 효율성 + 변동성
roe_adjusted_by_vol = roe / volatility_60d

# 가치 + 모멘텀 결합
value_momentum_score = (per_rank + momentum_rank) / 2

# 수익성 + 성장성
profit_growth_score = roe * (sales_growth + eps_growth)
```

### 3. 뉴스 피쳐 강화 방안

#### A. 감성 분석 심화
```python
# 기존: news_sentiment, news_volume

# 개선안: 뉴스 품질 지표들
news_sentiment_intensity = abs(news_sentiment) * news_volume  # 감성 강도
news_consistency = news_sentiment.rolling(5).std()           # 감성 일관성
news_trend = news_sentiment.ewm(span=20).mean()              # 뉴스 트렌드

# 뉴스 영향력 지표
news_market_impact = news_sentiment * volume_ratio          # 시장 영향력
news_persistence = news_sentiment.ewm(span=60).mean()       # 뉴스 지속성
```

#### B. 뉴스 + 가격 상호작용
```python
# 뉴스와 가격의 관계
price_news_alignment = price_momentum * news_sentiment       # 가격-뉴스 정렬도
news_lead_lag = news_sentiment.shift(-5).corr(price_change) # 뉴스 선행성
news_reaction = abs(price_change) * news_volume             # 뉴스 반응도
```

### 4. 그룹별 가중치 최적화

#### A. Technical 그룹 개선 (현재 IC 낮음)
**단기 Technical IC**: -0.0221 (20개 피쳐)
**장기 Technical IC**: -0.0271 (20개 피쳐)

**개선 방안**:
- 낮은 IC 피쳐 제거/대체: `ret_daily`, 가격 OHLC 등
- 새로운 파생 피쳐 추가: RSI, 볼린저밴드, 거래량 지표
- 그룹 내 가중치 재분배: 높은 IC 피쳐에 더 높은 가중치

#### B. Profitability 그룹 개선
**단기 Profitability IC**: -0.0009 (2개 피쳐)
**장기 Profitability IC**: -0.0211 (2개 피쳐)

**개선 방안**:
- 재무 변화율 추가: `roe_change`, `net_income_change`
- 상대적 지표 추가: `roe_sector_z`, `profit_margin_sector_z`
- 품질 지표 추가: `gross_margin`, `operating_margin`

### 5. 피쳐 선택 및 엔지니어링 우선순위

#### Phase 1: 즉시 적용 가능 (낮은 위험)
1. **가격 기반 피쳐 개선**: OHLC → 상대적 지표로 변경
2. **모멘텀 피쳐 강화**: 단순 수익률 → 가중/가속도 추가
3. **뉴스 피쳐 심화**: 감성 강도, 일관성 지표 추가

#### Phase 2: 중기 적용 (중간 위험)
1. **기술적 지표 확장**: RSI, 볼린저밴드 추가
2. **재무+기술 결합**: roe_volatility, value_momentum_score
3. **시장 구조 피쳐**: 집중도, 섹터 차이

#### Phase 3: 장기 적용 (높은 위험)
1. **머신러닝 피쳐**: 비선형 조합, 상호작용
2. **대안 데이터**: ESG 세부 지표, 뉴스 토픽 모델링
3. **시계열 피쳐**: LSTM 기반 예측값

## 📈 기대 개선 효과

### 단기 목표 (3개월)
- **IC 개선**: 0.0366 → 0.038~0.040 (약 5~10% 개선)
- **Hit Ratio 개선**: 현재 48.64% → 49.0%+ (목표 50%에 근접)
- **과적합 감소**: 낮은 IC 피쳐 제거로 일반화 성능 향상

### 중기 목표 (6개월)
- **IC 개선**: 0.040~0.042 (약 15% 개선)
- **새로운 알파 소스**: 파생 피쳐로 새로운 전략 기회
- **안정성 향상**: 그룹별 밸런스 개선

## 🔧 구현 계획

### Step 1: 피쳐 엔지니어링 파이프라인 구축
```python
# src/features/feature_engineering.py
class FeatureEngineer:
    def create_price_relative_features(self, ohlcv):
        # 상대적 가격 지표 생성
        pass

    def create_momentum_enhanced_features(self, prices):
        # 강화된 모멘텀 지표 생성
        pass

    def create_news_advanced_features(self, news_data):
        # 심화 뉴스 지표 생성
        pass
```

### Step 2: 피쳐 검증 및 선택
```python
# 각 새 피쳐의 IC, 상관관계, 과적합 위험 평가
# 상위 IC 피쳐만 선택 (현재 20/19개 유지)
# 그룹별 밸런스 유지
```

### Step 3: 모델 재학습 및 검증
```python
# L5 모델 재학습 (Ridge, XGBoost, RF)
# L8 랭킹 재생성
# 백테스트 및 IC 검증
```

## ⚠️ 리스크 관리

### 잠재적 리스크
1. **과적합 증가**: 새로운 피쳐가 Dev에서만 좋은 성능 보임
2. **계산 비용 증가**: 복잡한 파생 피쳐로 처리 시간 증가
3. **데이터 품질**: 새로운 피쳐 계산으로 인한 결측치 증가

### 완화 방안
1. **교차 검증 강화**: Holdout 성능 우선 검증
2. **피쳐 선택 엄격화**: IC > -0.01 기준 유지
3. **계산 최적화**: 벡터화 구현으로 성능 유지

## 🎯 결론 및 다음 단계

**현재 피쳐셋은 이미 잘 구성되어 있지만, 개선의 여지가 충분합니다.**

### 우선 적용 추천
1. **가격 기반 피쳐 개선** (즉시 적용 가능, 낮은 리스크)
2. **모멘텀 피쳐 강화** (즉시 적용 가능, 중간 리스크)
3. **뉴스 피쳐 심화** (단기 적용 가능, 낮은 리스크)

### 기대 ROI
- **개발 노력**: 중간 (약 2-3주)
- **성과 개선**: IC 0.002~0.005 (5~15% 향상)
- **리스크**: 낮음~중간 (기존 피쳐 유지하므로)

**다음 단계**: 구체적인 피쳐 엔지니어링 코드 구현 및 실험 시작.

---
**보고서 작성**: 2026-01-11
**다음 검토**: 피쳐 개선 실험 결과에 따라
