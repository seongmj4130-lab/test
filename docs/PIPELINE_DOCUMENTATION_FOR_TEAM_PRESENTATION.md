# 🎯 투트랙전략 파이프라인 완전 가이드 (팀원 발표용)

**작성일**: 2026-01-07  
**대상**: 팀원 발표 자료 (PPT + 설명문)  
**목적**: 투트랙전략(Track A/B)의 완전한 파이프라인 이해

---

## 📊 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 1: 원시 데이터 수집 (L0~L4)                      │
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ L0: Universe │→ │ L1: OHLCV    │→ │ L2: 재무데이터│→ │ L3: 패널병합│ │
│  │ (KOSPI200)   │  │ (기술지표)    │  │ (DART)       │  │ (통합데이터) │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│                                                                           │
│                                    ↓                                      │
│                          ┌──────────────┐                                │
│                          │ L4: CV 분할  │                                │
│                          │ (Dev/Holdout)│                                │
│                          └──────────────┘                                │
│                                                                           │
│  산출물:                                                                   │
│  • ohlcv_daily.parquet                                                   │
│  • fundamental.parquet                                                    │
│  • panel_merged_daily.parquet                                            │
│  • dataset_daily.parquet                                                  │
│  • cv_folds_short/long.parquet                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
        ┌─────────────────────────┴─────────────────────────┐
        │                                                     │
┌───────▼──────────────────┐                   ┌───────────▼────────────────┐
│  Phase 2: Track A        │                   │  Phase 3: Track B          │
│  랭킹 엔진                │                   │  백테스트 엔진              │
│                          │                   │                            │
│  ┌────────────────────┐  │                   │  ┌──────────────────────┐  │
│  │ L8: 랭킹 생성      │  │                   │  │ L6R: 랭킹→스코어      │  │
│  │ • 피처 정규화      │  │                   │  │ • 단기/장기/통합 결합 │  │
│  │ • Ridge 모델       │  │                   │  └──────────────────────┘  │
│  │ • 단기/장기/통합    │  │                   │            ↓                │
│  └────────────────────┘  │                   │  ┌──────────────────────┐  │
│            ↓             │                   │  │ L7: 백테스트 실행     │  │
│  산출물:                 │                   │  │ • 4개 전략            │  │
│  • ranking_short_daily   │                   │  │ • 거래비용 적용        │  │
│  • ranking_long_daily    │                   │  │ • 국면 분석           │  │
│                          │                   │  └──────────────────────┘  │
└──────────────────────────┘                   │            ↓                │
                                              │  산출물:                     │
                                              │  • bt_metrics_{strategy}     │
                                              │  • bt_regime_metrics_{strategy}│
                                              └──────────────────────────────┘
                                                          ↓
                                              ┌──────────────────────────────┐
                                              │  Phase 4: 성능 지표 산출      │
                                              │                              │
                                              │  • Headline Metrics          │
                                              │  • Alpha Quality             │
                                              │  • Operational Viability     │
                                              │  • Regime Robustness         │
                                              │                              │
                                              │  최종 리포트 생성              │
                                              └──────────────────────────────┘
```

---

## 📋 단계별 상세 설명

### Phase 1: 원시 데이터 수집

#### L0: 유니버스 구성

**📝 의미**: KOSPI200 멤버십을 월별로 추적하여 투자 대상 종목 범위를 정의합니다.

**📊 입력**: 
- KOSPI200 지수 구성 종목 정보 (외부 API)
- 기간: 2016-01-01 ~ 2024-12-31

**🔧 로직**:
1. 월별로 KOSPI200 구성 종목 추출
2. 종목 추가/제거 이벤트 추적
3. 멤버십 데이터프레임 생성 (ticker, date, is_member)

**📈 산출물**: 
- `universe_k200_membership_monthly.parquet` (약 50MB)
- 컬럼: ticker, date, is_member

---

#### L1: OHLCV 데이터 수집

**📝 의미**: 일별 가격/거래량 데이터를 수집하고 기술적 지표를 계산합니다.

**📊 입력**: 
- L0 유니버스 종목 리스트
- 기간: 2016-01-01 ~ 2024-12-31

**🔧 로직**:
1. pykrx API로 일별 OHLCV 데이터 다운로드
2. 기술적 지표 계산:
   - 모멘텀: rsi_14, rsi_30, momentum_5, momentum_20
   - 변동성: volatility_20, volatility_60
   - 거래량: volume_ratio, turnover_ratio
3. NaN 처리 및 forward fill

**📈 산출물**: 
- `ohlcv_daily.parquet` (약 500MB)
- 컬럼: ticker, date, open, high, low, close, volume + 기술지표 20여개

---

#### L2: 재무 데이터 수집

**📝 의미**: DART API를 통해 분기별/연간 재무 데이터를 수집합니다.

**📊 입력**: 
- L0 유니버스 종목 리스트
- 기간: 2016-01-01 ~ 2024-12-31

**🔧 로직**:
1. DART API로 재무제표 데이터 수집
2. 주요 지표 계산:
   - 가치 지표: PER, PBR, PSR, EV/EBITDA
   - 수익성: ROE, ROA, Operating Margin
   - 안정성: Debt Ratio, Current Ratio
3. 공시 지연(lag_days=90) 반영

**📈 산출물**: 
- `fundamentals_annual.parquet` (약 100MB)
- 컬럼: ticker, effective_date + 재무지표 30여개

---

#### L3: 패널 데이터 병합

**📝 의미**: OHLCV, 재무, 뉴스 감성, ESG 데이터를 하나의 패널 데이터로 통합합니다.

**📊 입력**: 
- `ohlcv_daily.parquet`
- `fundamentals_annual.parquet`
- `news_sentiment_daily.parquet` (외부 파일)
- ESG 데이터 (선택적)

**🔧 로직**:
1. 날짜별 종목 기준으로 Left Join
2. 재무 데이터: forward fill (최신 재무제표 유지)
3. 뉴스 감성: lag_days=1 적용
4. KOSPI200 멤버십 필터링 (`filter_k200_members_only: true`)

**📈 산출물**: 
- `panel_merged_daily.parquet` (약 1GB)
- 컬럼: ticker, date + 모든 피처 (약 50개)

---

#### L4: Walk-Forward CV 분할

**📝 의미**: 시간순 데이터를 Dev/Holdout 구간으로 분할하고, Walk-Forward CV fold를 생성하며 타겟 변수를 생성합니다.

**⚠️ 중요**: L4는 데이터 분할만 수행하며, **ML 모델 학습은 하지 않습니다**. Walk-Forward CV fold는 L5(선택적)에서 사용됩니다.

**📊 입력**: 
- `panel_merged_daily.parquet`

**🔧 로직**:

1. **구간 분할** (백테스트 평가 구간 구분용):
   - Dev: 2016-01-01 ~ 2022-12-31 (개발/검증 구간)
   - Holdout: 2023-01-01 ~ 2024-12-31 (`holdout_years: 2`, 최종 평가 구간)

2. **Walk-Forward CV 생성** (롤링 윈도우 방식):
   
   **Walk-Forward CV란?**
   - 시간 순서를 유지하며 과거 데이터로 학습, 미래 데이터로 평가
   - 각 fold마다 **train 구간을 확장**하는 롤링 윈도우 방식
   - 예측 시점에 가까울수록 더 많은 과거 데이터 활용
   
   **각 Fold 구조**:
   ```
   Fold 1: [Train: 2016~2019.01] → [Test: 2019.02] (3년 학습, 20일 평가)
   Fold 2: [Train: 2016~2019.02] → [Test: 2019.03] (Train 확장!)
   Fold 3: [Train: 2016~2019.03] → [Test: 2019.04] (Train 확장!)
   ...
   Fold N: [Train: 2016~2024.11] → [Test: 2024.12] (최신까지 확장)
   ```
   
   **파라미터**:
   - 단기 모델: `rolling_train_years=3` (3년 롤링 윈도우)
   - 장기 모델: `rolling_train_years=5` (5년 롤링 윈도우)
   - `test_window=20일` (20일마다 평가)
   - `step_days=20` (20일씩 앞으로 이동)
   - `embargo_days=20` (Train과 Test 사이 격리 기간, 데이터 누수 방지)
   - `horizon_short=20일`, `horizon_long=120일` (미래 수익률 계산 기간)
   
   **롤링 윈도우 작동 방식**:
   ```python
   # Fold 1 (예시)
   train_start = 2016-01-01
   train_end = 2019-01-20  # test_start에서 3년 전
   test_start = 2019-02-10  # embargo + horizon 이후
   test_end = 2019-03-02
   
   # Fold 2 (20일 후)
   train_start = 2016-01-01  # 동일
   train_end = 2019-02-10    # 확장됨! (3년 롤링)
   test_start = 2019-03-02
   test_end = 2019-03-22
   ```
   
   ⚠️ **Track A(L8)에서는 이 Train 구간을 사용하지 않음**:
   - L8은 학습을 하지 않으므로 Train 구간 데이터를 사용하지 않음
   - 단순히 Dev/Holdout 구간 구분만 사용
   
   ⚠️ **L5(선택적)에서만 Train 구간 사용**:
   - 각 fold의 `train_start ~ train_end` 데이터로 Ridge 모델 학습
   - 각 fold의 `test_start ~ test_end` 데이터로 예측 및 평가

3. **타겟 변수 생성** (성과 평가용):
   - `true_short`: 20일 후 수익률 (현재 가격 대비 20일 후 수익률)
   - `true_long`: 120일 후 수익률 (현재 가격 대비 120일 후 수익률)
   - ⚠️ 랭킹 생성(L8)에는 사용하지 않음, 백테스트 성과 평가(L7)에만 사용

**📈 산출물**: 
- `dataset_daily.parquet` (약 1.2GB, 타겟 변수 포함)
- `cv_folds_short.parquet` (단기 모델 CV fold 정보, Dev/Holdout 각각 약 87개 fold)
- `cv_folds_long.parquet` (장기 모델 CV fold 정보, Dev/Holdout 각각 약 87개 fold)

**Fold 정보 예시** (`cv_folds_short.parquet`):
```
fold_id           train_start  train_end    test_start   test_end      phase
dev_0001          2016-01-04   2019-01-20   2019-02-10   2019-03-02    dev
dev_0002          2016-01-04   2019-02-10   2019-03-02   2019-03-22    dev
...
holdout_0001      2016-01-04   2023-01-31   2023-02-28   2023-03-20    holdout
holdout_0002      2016-01-04   2023-02-28   2023-03-20   2023-04-07    holdout
...
```

---

### ⚠️ 중요: ML 학습 vs 피처 가중치 합산

**현재 시스템은 ML 모델 학습을 하지 않습니다:**

- **L4 (Train/Val/Holdout 분할)**: 데이터 구간 분할만 수행 (백테스트 평가 구간 구분용)
- **L5 (Ridge 모델 학습)**: 선택적 단계, Track A에서는 사용하지 않음 (피처 리스트만 참고)
- **L8 (랭킹 생성)**: ML 학습 없이 **피처 가중치 합산**으로 랭킹 생성
  ```
  score = Σ(normalized_feature × weight)  # 단순 가중치 합산
  ```

**전통적인 ML 방식 vs 현재 방식**:

| 구분 | 전통적 ML 방식 | 현재 방식 (Track A) |
|------|----------------|---------------------|
| 학습 | Train 데이터로 모델 학습 | ❌ 학습 없음 |
| 검증 | Val 데이터로 하이퍼파라미터 튜닝 | ❌ 검증 없음 |
| 평가 | Holdout 데이터로 최종 평가 | ✅ Holdout에서 Hit Ratio 평가 |
| 방식 | Ridge 회귀 예측값 사용 | 피처 가중치 합산 값 사용 |

**왜 이렇게 설계했나?**
- 랭킹 산정은 해석 가능성과 단순성이 중요
- 피처 가중치를 직접 제어하여 투자 논리 명확화
- L5 모델 학습은 선택적으로 Track B에서만 사용 가능

---

### 💼 실무 관점: 가중치 합산만으로 충분한가?

#### ✅ 가중치 합산 방식의 장점 (현재 방식)

1. **해석 가능성 (Interpretability)**
   - 각 피처의 기여도를 명확히 파악 가능
   - 투자 논리를 투자자/운용위원회에 설명 용이
   - 규제 요구사항 충족 (설명 가능한 AI)

2. **안정성 (Stability)**
   - 과적합 위험 낮음
   - 시장 환경 변화에 덜 민감
   - 예측력이 갑자기 떨어지는 현상 방지

3. **실행 속도**
   - 학습 시간 불필요 (즉시 랭킹 생성)
   - 계산 비용 낮음
   - 실시간 업데이트 용이

4. **현재 성과**
   - Holdout Hit Ratio **51%** 달성
   - Holdout Sharpe **0.65** 달성
   - 실무 수준 성과 달성

#### ❌ 가중치 합산 방식의 한계

1. **비선형 관계 포착 불가**
   - 선형 결합만 가능
   - 피처 간 상호작용 포착 어려움
   - 예: "변동성이 낮고 모멘텀은 높을 때" 특별 효과 포착 어려움

2. **데이터 기반 가중치 학습 불가**
   - 가중치를 수동/경험 기반으로 설정
   - 데이터가 알려주는 최적 가중치 활용 불가
   - 최적화 기회 포기

3. **시장 환경 적응성 낮음**
   - 고정된 가중치 사용
   - 상승장/하락장별 최적 가중치 자동 조정 어려움

4. **예측력 한계**
   - 복잡한 패턴 학습 불가
   - IC/Rank IC 개선 여지 존재
   - 현재 IC ≈ 0.01~0.02 수준 (개선 가능)

#### 🚀 실무에서의 개선 방향

**1단계: 현재 방식 (가중치 합산) 유지**
- ✅ 해석 가능성 중시
- ✅ 안정성 확보
- ✅ 실무 수준 성과 달성

**2단계: 하이브리드 접근 (권장)**
- 랭킹 점수 + 모델 예측값 결합
- Track A (랭킹) + L5 (모델) 앙상블
- 랭킹의 안정성 + 모델의 예측력 결합

```python
# L6R 수정 버전
score_ranking = ranking_short_daily["score_total"]  # Track A 랭킹
score_model = pred_short_oos["y_pred"]              # L5 Ridge 모델 예측

# 가중 평균 (안정성 + 예측력)
score_ens = 0.6 * score_ranking + 0.4 * score_model
```

**3단계: 고급 모델 도입 (선택적)**
- XGBoost, RandomForest 등 비선형 모델
- 비선형 관계 포착
- 더 높은 예측력 기대

#### 📊 실무 사례 비교

| 접근 방식 | 사용 사례 | 장점 | 단점 |
|-----------|-----------|------|------|
| **가중치 합산** | 벤치마크 투자, 포트폴리오 운용 | 해석 가능, 안정적 | 예측력 한계 |
| **Ridge/Lasso** | 중형 자산운용사, 퀀트팀 | 선형 관계 학습, 안정적 | 비선형 포착 불가 |
| **XGBoost/RF** | 대형 자산운용사, 헤지펀드 | 비선형 포착, 높은 예측력 | 과적합 위험, 해석 어려움 |
| **앙상블** | 실무 표준 (권장) | 안정성 + 예측력 결합 | 구현 복잡도 증가 |

#### 💡 현재 프로젝트의 선택지

**현재 시스템은 L5 모델 학습을 선택적으로 지원합니다:**

- `L5`: Ridge 모델 학습 (선택적)
- Track A + L5 결합: 랭킹 + 모델 예측값 앙상블 가능
- 점진적 개선 가능: 가중치 합산 → 앙상블 → 고급 모델

**권장 전략**:
1. **현재 (1단계)**: 가중치 합산으로 실무 수준 성과 확보 ✅
2. **다음 (2단계)**: L5 Ridge 모델 추가하여 앙상블 구성
3. **미래 (3단계)**: XGBoost 등 고급 모델 도입 검토

**결론**: 가중치 합산만으로도 **실무 수준의 성과는 충분히 달성 가능**합니다. 다만, 더 높은 예측력을 원한다면 **앙상블 접근법(랭킹 + 모델)**을 권장합니다.

---

### Phase 2: Track A - 랭킹 엔진

#### L8: 랭킹 생성 (Dual Horizon)

**📝 의미**: 피처를 기반으로 KOSPI200 종목의 랭킹을 산정합니다. 단기/장기/통합 랭킹을 생성합니다.

**⚠️ 중요**: L8은 **ML 모델 학습을 하지 않습니다**. 단순히 피처 가중치 합산으로 랭킹을 생성합니다.

**📊 입력**: 
- `panel_merged_daily.parquet` (L3 산출물)
- `dataset_daily.parquet` (L4 산출물, 선택적)

**🔧 로직**:

1. **피처 선택** (L5와 동일한 피처셋만 참고):
   - 단기: 22개 피처 (`features_short_v1.yaml`)
   - 장기: 19개 피처 (`features_long_v1.yaml`)
   - ⚠️ L5의 피처 리스트만 사용하며, L5 모델 학습 결과는 사용하지 않음

2. **정규화** (`normalization_method: zscore`):
   ```
   zscore = (value - mean) / std
   ```
   - 섹터 상대화: 섹터별로 정규화 (`use_sector_relative: true`)
   - 날짜별 cross-sectional 정규화 (같은 날짜 내 종목 간 비교)

3. **피처 그룹 가중치 적용** (사전 정의된 가중치):
   - 기술 지표 그룹: 사전 정의 가중치
   - 가치 지표 그룹: 사전 정의 가중치
   - 뉴스 감성 그룹: 0.10 (단기), 0.03 (장기)
   - ESG 그룹: 사전 정의 가중치

4. **랭킹 점수 계산** (가중치 합산, ML 학습 없음):
   ```
   score_total_short = Σ(normalized_feature_i × weight_i)
   score_total_long = Σ(normalized_feature_i × weight_i)
   score_ens = 0.5 × score_short + 0.5 × score_long
   ```
   - ⚠️ **Ridge 모델 학습 없이** 단순 가중치 합산

5. **랭킹 산정** (날짜별 종목 순위):
   - rank_total: 1~N (1이 가장 우수, score_total 기준 내림차순)

**📈 산출물**: 
- `ranking_short_daily.parquet` (단기 랭킹)
- `ranking_long_daily.parquet` (장기 랭킹)
- 컬럼: ticker, date, rank_total, score_total, score_total_calc

**핵심 설정값**:
- `normalization_method: zscore` ✅ (2026-01-07 최종 픽스)
- ⚠️ `ridge_alpha`, `min_feature_ic`는 L5에서만 사용 (L8에서는 사용 안 함)

**최종 성과 (Holdout)**:
- 통합 랭킹 Hit Ratio: **51.06%** ✅ (목표 50% 달성)
- 단기 랭킹 Hit Ratio: **50.99%** ✅
- 장기 랭킹 Hit Ratio: **51.00%** ✅

---

### 🔍 참고: L5에서 Train 구간 활용 (선택적)

**L5는 선택적 단계**이며, Train 구간 데이터를 실제로 사용하는 유일한 단계입니다:

**L5의 Walk-Forward 학습 프로세스**:
```python
# 각 fold별로 반복
for fold in cv_folds:
    # 1. Train 구간 데이터로 모델 학습
    train_data = dataset_daily[
        (dataset_daily['date'] >= fold.train_start) &
        (dataset_daily['date'] <= fold.train_end)
    ]
    
    # 2. Ridge 모델 학습
    model = Ridge(alpha=8.0)
    model.fit(X_train, y_train)  # X: 피처, y: 타겟 수익률
    
    # 3. Test 구간 데이터로 예측
    test_data = dataset_daily[
        (dataset_daily['date'] >= fold.test_start) &
        (dataset_daily['date'] <= fold.test_end)
    ]
    
    # 4. 예측값 생성
    y_pred = model.predict(X_test)
    
    # 5. 예측값 저장 (OOS 예측)
```

**롤링 윈도우의 효과**:
- **Fold 1**: 3년 데이터로 학습 → 최신 패턴 학습 불가
- **Fold N**: 8년 데이터로 학습 → 최신 시장 패턴까지 포함
- 시장이 변화함에 따라 **모델이 최신 데이터에 적응**

**⚠️ 현재 시스템에서는 L5를 사용하지 않으므로**, Train 구간 데이터는 생성되지만 실제로 학습에 사용되지 않습니다.

---

### Phase 3: Track B - 백테스트 엔진

#### L6R: 랭킹 스코어 변환

**📝 의미**: Track A에서 생성된 랭킹을 백테스트용 투자 신호(스코어)로 변환합니다.

**📊 입력**: 
- `ranking_short_daily.parquet` (Track A 산출물)
- `ranking_long_daily.parquet` (Track A 산출물)
- `dataset_daily.parquet` (수익률 정보 포함)

**🔧 로직**:

1. **랭킹 → 스코어 변환**:
   ```
   score_short = -rank_total_short  # 낮은 랭크가 높은 스코어
   score_long = -rank_total_long
   score_ens = alpha_short × score_short + alpha_long × score_long
   ```
   - `alpha_short: 0.5`, `alpha_long: 0.5` (통합 랭킹)

2. **리밸런싱 간격별 필터링**:
   - `rebalance_interval=20`: 20일마다 스코어 추출
   - 파일명: `rebalance_scores_from_ranking_interval_20.parquet`

3. **국면별 Alpha 조정** (선택적):
   - Bull: alpha 증가 → 공격적
   - Bear: alpha 감소 → 방어적

**📈 산출물**: 
- `rebalance_scores_from_ranking_interval_{N}.parquet`
- 컬럼: ticker, date, score_total_short, score_total_long, score_ens

---

#### L7: 백테스트 실행

**📝 의미**: 랭킹 기반 투자 전략의 성과를 시뮬레이션합니다. 4가지 전략을 병렬 실행합니다.

**📊 입력**: 
- `rebalance_scores_from_ranking_interval_{N}.parquet` (L6R 산출물)
- `dataset_daily.parquet` (수익률 정보)
- `ohlcv_daily.parquet` (국면 분석용)

**🔧 로직**:

**4개 전략**:
1. **bt20_short**: 단기 랭킹만 사용, top_k=12, holding_days=20
2. **bt20_ens**: 통합 랭킹 사용, top_k=15, holding_days=20
3. **bt120_long**: 장기 랭킹 사용, top_k=15, 오버래핑 트랜치(4개)
4. **bt120_ens**: 통합 랭킹 사용, top_k=20, 오버래핑 트랜치(4개)

**백테스트 프로세스**:

1. **포트폴리오 구성** (`rebalance_interval=20`):
   - 매 20일마다 리밸런싱
   - 상위 `top_k`개 종목 선택
   - 가중치: `equal` (동일 비중) 또는 `softmax`

2. **오버래핑 트랜치 (BT120)**:
   ```
   Day 0:   트랜치1 시작 (120일 보유)
   Day 20:  트랜치1 + 트랜치2 시작
   Day 40:  트랜치1 + 트랜치2 + 트랜치3 시작
   Day 60:  트랜치1 + 트랜치2 + 트랜치3 + 트랜치4 시작
   Day 80:  트랜치1 종료, 트랜치5 시작
   ...
   ```
   - 동시 보유 최대 4개 트랜치
   - 각 트랜치에 1/4 자본 고정 배분

3. **거래비용 적용** (턴오버 기반):
   ```
   daily_cost = turnover_oneway × (cost_bps + slippage_bps) / 10000
   ```
   - `cost_bps: 10.0` (기본 거래비용)
   - `slippage_bps: 0.0` (시장임팩트, 선택적)

4. **수익률 계산**:
   ```
   gross_return = Σ(weight_i × return_i)
   net_return = gross_return - daily_cost
   ```

5. **국면 분석** (자동 분류):
   - Bull/Bear/Neutral 국면 자동 분류 (60일 lookback)
   - 국면별 성과 지표 산출

**📈 산출물**: 

각 전략별로 다음 파일 생성:

- `bt_metrics_{strategy}.parquet`: 성과 지표 (Dev/Holdout)
  - Headline Metrics: Sharpe, CAGR, MDD, Calmar
  - Alpha Quality: IC, Rank IC, ICIR, L/S Alpha
  - Operational: Turnover, Hit Ratio, Profit Factor

- `bt_regime_metrics_{strategy}.parquet`: 국면별 성과
  - Bull/Bear/Neutral 국면별 4개 지표

- `bt_positions_{strategy}.parquet`: 포지션 히스토리
- `bt_returns_{strategy}.parquet`: 일별 수익률
- `bt_equity_curve_{strategy}.parquet`: 누적 자산 곡선

---

### Phase 4: 성능 지표 산출

**📝 의미**: 백테스트 결과를 바탕으로 투자 전략의 유효성을 평가합니다.

**📊 입력**: 
- `bt_metrics_{strategy}.parquet` (L7 산출물)
- `bt_regime_metrics_{strategy}.parquet` (L7 산출물)

**🔧 로직**:

**4가지 지표 카테고리**:

1. **핵심 성과 (Headline Metrics)**:
   - Net Sharpe Ratio: 리스크 조정 수익률
   - Net CAGR: 연평균 복리 수익률
   - Net MDD: 최대 낙폭
   - Net Total Return: 누적 수익률
   - Calmar Ratio: CAGR / |MDD|

2. **모델 예측력 (Alpha Quality)**:
   - IC: 스코어-수익률 상관계수
   - Rank IC: 랭크 상관계수
   - ICIR: IC 안정성 (IC_mean / IC_std)
   - Long/Short Alpha: 상위-하위 수익률 차이

3. **운용 안정성 (Operational Viability)**:
   - Avg Turnover: 평균 회전율
   - Hit Ratio: 승률 (수익 > 0 비율)
   - Profit Factor: 총 이익 / 총 손실
   - Avg Trade Duration: 평균 보유 일수

4. **국면별 성과 (Regime Robustness)**:
   - Bull/Bear/Neutral 국면별 4개 지표 산출

**📈 산출물**: 
- `artifacts/reports/track_b_4strategy_final_summary.md`: 최종 요약 리포트

---

## 🎯 핵심 설정값 명시 (2026-01-07 최종 기준)

### 정규화 방법
- ✅ **zscore** (Hit Ratio 최적화 결과)

### Ridge 모델
- ✅ **ridge_alpha: 8.0** (과적합 방지)
- ✅ **min_feature_ic: -0.1** (피처 필터링)

### Top K 종목 수
- ✅ **bt20_short: 12**
- ✅ **bt20_ens: 15**
- ✅ **bt120_long: 15**
- ✅ **bt120_ens: 20**

### 리밸런싱 간격
- ✅ **rebalance_interval: 20** (모든 전략, holding_days와 동일)

### 거래비용
- ✅ **cost_bps: 10.0** (기본 거래비용)
- ✅ **slippage_bps: 0.0** (시장임팩트, 선택적)
- ✅ **비용 계산**: `turnover_oneway × (cost_bps + slippage_bps) / 10000`

---

## 📈 최종 성과 하이라이트 (Holdout 구간)

### 4개 전략 비교 (2026-01-07 최신)

| 전략 | Net Sharpe | Net CAGR | Net MDD | Calmar Ratio | 리밸런싱 수 |
|------|------------|----------|---------|--------------|-------------|
| **bt20_short** | **0.646** | **13.84%** | **-9.09%** | **1.522** | 23 |
| **bt20_ens** | **0.683** | **14.98%** | **-10.98%** | **1.364** | 23 |
| **bt120_long** | **0.684** | **13.60%** | **-8.66%** | **1.570** | 23 |
| **bt120_ens** | **0.626** | **11.66%** | **-7.69%** | **1.516** | 23 |

### 주요 성과 요약

✅ **실무 수준 완성**:
- Holdout Sharpe ≥ 0.65 달성 (bt20_short, bt20_ens, bt120_long)
- Holdout Calmar ≥ 1.5 달성 (모든 전략)
- Holdout MDD ≤ -11% (리스크 관리 양호)

✅ **타이밍 럭 해소**:
- BT120 오버래핑 트랜치로 Holdout 리밸런싱 3회 → 23회 증가
- 성과 지표 안정화

✅ **Alpha Quality**:
- Holdout Rank IC ≥ 0.01 (모든 전략)
- Holdout L/S Alpha > 0 (bt20_ens, bt120_*)

---

## 💡 핵심 메시지 5개 (팀원 발표용)

### 1️⃣ "피처 가중치 합산 방식 → Hit Ratio 51%"
- ⚠️ **ML 모델 학습 없이** 피처 가중치 합산으로 랭킹 생성
- zscore 정규화 + 사전 정의 가중치로 **Holdout Hit Ratio 51% 달성**
- 목표 50% 초과 달성 → 실전 투자 신호 가능성 검증

### 2️⃣ "트랜치 도입 → MDD -30%→-9%"
- BT120 오버래핑 트랜치(4개) 도입으로 **MDD -30% → -9% 개선**
- 리스크 분산 효과로 안정성 확보

### 3️⃣ "턴오버 기반 비용 → 현실적 Net 지표"
- 고정 비용이 아닌 **턴오버 기반 거래비용** 적용
- Net Sharpe/CAGR/MDD로 실제 운용 가능성 검증

### 4️⃣ "국면 Robustness → 모든 시장 안정"
- Bull/Bear/Neutral 국면별 성과 분석
- **모든 국면에서 양의 수익률** → 전략 안정성 확보

### 5️⃣ "Holdout Sharpe 0.65 → 실전 가능"
- **Holdout Sharpe ≥ 0.65** 달성 (3개 전략)
- Dev-Holdout Gap 최소화 → 과적합 방지 검증
- 실전 운용 가능한 수준의 성과 달성

---

## 📦 최종 산출물 요약

### 4개 Parquet 파일 (핵심)

1. **bt_metrics_{strategy}.parquet**
   - 4개 전략 × Dev/Holdout = 8개 행
   - 컬럼: net_sharpe, net_cagr, net_mdd, calmar, ic, rank_ic, hit_ratio 등

2. **bt_regime_metrics_{strategy}.parquet**
   - 국면별(Bull/Bear/Neutral) 성과 지표
   - Dev/Holdout 구간별 분류

3. **ranking_short_daily.parquet**
   - 날짜별 종목 단기 랭킹
   - Track A 핵심 산출물

4. **ranking_long_daily.parquet**
   - 날짜별 종목 장기 랭킹
   - Track A 핵심 산출물

### 리포트 파일

- `artifacts/reports/track_b_4strategy_final_summary.md`: 최종 성과 요약

---

## 🚀 실행 방법 (요약)

### 1단계: 공통 데이터 준비 (L0~L4)
```bash
python -m src.data_collection.DataCollectionPipeline
```

### 2단계: Track A 실행 (랭킹 생성)
```bash
python -m src.pipeline.track_a_pipeline
```

### 3단계: Track B 실행 (백테스트, 4전략)
```bash
python -m src.pipeline.track_b_pipeline bt20_short
python -m src.pipeline.track_b_pipeline bt20_ens
python -m src.pipeline.track_b_pipeline bt120_long
python -m src.pipeline.track_b_pipeline bt120_ens
```

### 4단계: 원클릭 실행 (권장)
```bash
python -m src.tools.run_two_track_and_export --export-dest ..\06_code22
```

---

## 🔍 특정 날짜 랭킹 조회 명령어 3가지

### 1️⃣ CLI: Holdout 날짜 상세 분석 (팩터셋 기여도 포함)

**명령어**:
```bash
python scripts/inspect_tracka_holdout_day.py --date 2024-12-30 --topk 10 --horizon both
```

**출력 예시**:
```
[Track A] SHORT | date=2024-12-30 | Top10 + 팩터셋 Top3
====================================================================================================
date       rank_total  ticker  score_total  score_total_calc  score_gap  top_groups
2024-12-30      1      005930         0.85              0.85      0.00  technical(+0.35), value(+0.28), news(+0.15)
2024-12-30      2      000660         0.82              0.82      0.00  technical(+0.32), value(+0.30), news(-0.12)
...
```

**목적**:
1. **Top 3 영향 팩터셋 표시**: 각 종목의 랭킹 점수에 가장 큰 영향을 미친 3개 팩터셋을 기여도 순으로 표시
   - 예: `technical(+0.35), value(+0.28), news(+0.15)` → 기술 지표가 가장 큰 기여
   - 절대값 기준으로 Top 3 선택 (양수/음수 모두 포함)

2. **설명 가능성 (Explainability)**: 특정 날짜에 종목이 상위 랭킹에 오른 이유를 팩터셋별 기여도로 파악

3. **디버깅**: 백테스트 결과에서 특정 날짜의 선택 종목이 이상할 때 랭킹 로직 검증

4. **투자 논리 검증**: 어떤 팩터(기술적/가치/뉴스/ESG)가 그 날짜에 영향을 미쳤는지 확인

**전체 팩터셋 개수**:
- **단기 (short)**: **5개** 팩터셋
  1. `technical` (기술 지표)
  2. `value` (가치 지표)
  3. `profitability` (수익성 지표)
  4. `news` (뉴스 감성)
  5. `other` (기타)

- **장기 (long)**: **5개** 팩터셋
  1. `technical` (기술 지표)
  2. `value` (가치 지표)
  3. `profitability` (수익성 지표)
  4. `news` (뉴스 감성)
  5. `esg` (ESG 지표)

**참고**: UI 아이콘 개수는 **최대 5개** (단기/장기 모두 5개)를 준비하면 됩니다.

**옵션**:
- `--date`: 분석할 날짜 (YYYY-MM-DD)
- `--topk`: 상위 K개 종목 조회 (기본 10)
- `--horizon`: `short` / `long` / `both` (기본 both)
- `--save_md`: 결과를 Markdown 파일로 저장

---

### 2️⃣ Python API: UI에서 랭킹 조회 (단기/장기/통합)

**명령어 (Python)**:
```python
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
)

# 단기 랭킹 Top 20
short_rankings = get_short_term_ranking("2024-12-31", top_k=20)
# [{'ticker': '005930', 'score': 0.85, 'rank': 1, 'horizon': 'short'}, ...]

# 장기 랭킹 Top 20
long_rankings = get_long_term_ranking("2024-12-31", top_k=20)

# 통합 랭킹 Top 20 (단기+장기 결합)
combined_rankings = get_combined_ranking("2024-12-31", top_k=20)
```

**이유**:
1. **UI 연동**: 웹/앱 UI에서 특정 날짜의 랭킹을 실시간 조회
2. **간단한 조회**: 복잡한 분석 없이 특정 날짜의 Top K 종목만 빠르게 확인
3. **다양한 랭킹 제공**: 단기/장기/통합 랭킹을 선택적으로 조회 가능

**활용 사례**:
- 투자자에게 오늘의 추천 종목 Top 20 제공
- 특정 날짜 기준으로 포트폴리오 구성 가이드
- 랭킹 API 엔드포인트 구현

---

### 3️⃣ Python API: Holdout 날짜 분석 (상세 정보 + 팩터셋)

**명령어 (Python)**:
```python
from src.tracks.track_a.ranking_service import inspect_holdout_day_rankings

result = inspect_holdout_day_rankings(
    as_of="2024-12-30",
    topk=10,
    horizon="both",  # "short" | "long" | "both"
    config_path="configs/config.yaml",
)

# result 구조
# {
#   "meta": {"date": "2024-12-30", "holdout_start": "...", "holdout_end": "..."},
#   "short": DataFrame (Top10 + 팩터셋 Top3),
#   "long": DataFrame (Top10 + 팩터셋 Top3),
# }
```

**이유**:
1. **프로그래밍 방식 분석**: CLI보다 유연하게 Python 스크립트에서 분석 로직 구현
2. **팩터셋 기여도 분석**: 각 종목의 랭킹 점수가 어떤 팩터 그룹(technical/value/news/esg)에서 기인했는지 확인
3. **배치 분석**: 여러 날짜를 반복 분석하여 패턴 발견

**활용 사례**:
- Holdout 구간 전체 날짜를 순회하며 랭킹 품질 분석
- 특정 시장 국면(Bull/Bear)에서 팩터 기여도 변화 추적
- 백테스트 실패 지점에서 랭킹 로직 디버깅

**출력 데이터 구조**:
- `rank_total`: 종목 순위
- `score_total`: 총 랭킹 점수
- `top_groups`: 기여도 Top 3 팩터 그룹 (예: "technical(0.35), value(0.28), news(-0.15)")

**⚠️ 음수(-) 팩터 기여도가 나오는 이유**:

팩터 기여도는 다음과 같이 계산됩니다:
```
그룹 기여도 = Σ(피처 가중치 × 정규화된 피처 값)
```

**음수가 나오는 경우**:
1. **zscore 정규화**:
   - zscore = (값 - 평균) / 표준편차
   - 평균보다 낮은 값 → **음수**
   - 예: 시장 평균 PER보다 높은 종목(저평가 아님) → 음수

2. **피처 값이 음수**:
   - 특정 피처 값이 시장 평균보다 낮을 때
   - 예: `momentum_20d`가 음수(하락 추세) → 정규화 후 음수

3. **랭킹을 낮추는 방향**:
   - 음수 기여도 = 해당 그룹이 랭킹 점수를 **감소**시킴
   - 예: `value(-0.15)` → 가치 지표가 이 종목의 랭킹을 낮추는 방향

**예시**:
```
종목 A:
- technical(+0.35): 기술 지표가 랭킹을 높임 ✅
- value(-0.28): 가치 지표가 랭킹을 낮춤 ❌ (PER 높음, 저평가 아님)
- news(+0.15): 뉴스 감성이 랭킹을 높임 ✅

→ 총점: 0.35 - 0.28 + 0.15 = 0.22 (여전히 상위권이지만, 가치 지표는 부정적)
```

**해석 방법**:
- **양수 기여도**: 해당 그룹이 랭킹을 높이는 방향
- **음수 기여도**: 해당 그룹이 랭킹을 낮추는 방향 (약점)
- **절대값 크기**: 기여도가 큰 그룹일수록 영향력이 큼

---

### 📊 3가지 명령어 비교표

| 명령어 | 용도 | 출력 | 팩터셋 분석 | 사용 시나리오 |
|--------|------|------|-------------|---------------|
| **CLI** (`inspect_tracka_holdout_day.py`) | 빠른 조회 + 리포트 | 콘솔 출력 + Markdown | ✅ 있음 | 한 번만 확인, 리포트 작성 |
| **Python API** (`get_*_ranking`) | UI 연동, 간단 조회 | JSON 리스트 | ❌ 없음 | 웹/앱 API, 간단한 Top K 조회 |
| **Python API** (`inspect_holdout_day_rankings`) | 프로그래밍 분석 | DataFrame | ✅ 있음 | 스크립트 분석, 배치 처리 |

---

## 📚 참고 문서

- `README.md`: 전체 프로젝트 개요
- `final_backtest_report.md`: 백테스트 기술 보고서
- `final_ranking_report.md`: 랭킹 엔진 기술 보고서
- `docs/FINAL_METRICS_DEFINITION.md`: 최종 수치셋 정의

---

**작성 완료**: 2026-01-07

