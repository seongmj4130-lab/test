# L8-L5 피처 통일 작업 완료 보고서

## 작업 개요

**목적**: L8 랭킹 엔진이 L5 모델 학습과 동일한 피처셋을 사용하도록 통일

**작업 일시**: 2026-01-XX

## 완료된 작업

### 1단계: L8 피처를 L5와 통일 ✅

#### 코드 수정
1. **`src/components/ranking/score_engine.py`**
   - `build_score_total()` 함수에 `feature_list_path`, `cfg` 파라미터 추가
   - L5 피처 리스트 파일을 읽어서 동일한 피처 사용하도록 수정
   - 로그 출력: `"[L8-L5 통일] Loaded {n} features from {path}"`

2. **`src/tracks/track_a/stages/ranking/l8_dual_horizon.py`**
   - `run_L8_short_rank_engine()`: L5의 `feature_list_short` 사용
   - `run_L8_long_rank_engine()`: L5의 `feature_list_long` 사용
   - `build_ranking_daily()` 호출 시 `feature_list_path`, `cfg` 전달

#### 설정 파일 생성
- **`configs/features_short_v1.yaml`**: 단기 22개 피처 리스트
- **`configs/features_long_v1.yaml`**: 장기 19개 피처 리스트 (이미 존재)

#### 피처 구성

**단기 (22개)**:
- Core 공통: 12개
- Short 전용: 6개 (`price_momentum_20d`, `momentum_3m`, `momentum_reversal`, `ret_daily`, `volume_ratio`, `equity`)
- News 감성: 4개 (`news_sentiment`, `news_sentiment_ewm5`, `news_sentiment_surprise`, `news_volume`)

**장기 (19개)**:
- Core 공통: 12개
- Long 전용: 7개 (`total_liabilities`, `debt_ratio`, `esg_score`, `environmental_score`, `social_score`, `governance_score`, `news_sentiment_ewm20`)

### 2단계: feature_weights 정리 ✅

#### 수정된 파일

1. **`configs/feature_weights_short_hitratio_optimized.yaml`**
   - **변경 전**: 28개 피처 (OHLCV 포함)
   - **변경 후**: 22개 피처 (L5 단기 피처와 동일)
   - **제거된 피처**: `close`, `high`, `low`, `open` (OHLCV)
   - **News 피처 포함**: 4개 (`news_sentiment`, `news_sentiment_ewm5`, `news_sentiment_surprise`, `news_volume`)

2. **`configs/feature_weights_long_ic_optimized.yaml`**
   - **변경 전**: 28개 피처 (OHLCV 포함)
   - **변경 후**: 19개 피처 (L5 장기 피처와 동일)
   - **제거된 피처**: `close`, `high`, `low`, `open` (OHLCV)
   - **ESG 피처 포함**: 4개 (`esg_score`, `environmental_score`, `social_score`, `governance_score`)

#### 가중치 구조

**단기 (22개)**:
- Core 공통: 12개 (각 0.025, net_income 0.04, roe 0.075)
- Short 전용: 6개 (각 0.025, equity 0.04)
- News 감성: 4개 (각 0.05)

**장기 (19개)**:
- Core 공통: 12개 (각 0.02, net_income 0.05, roe 0.1)
- Long 전용: 7개 (각 0.05)

### 3단계: Config 설정 변경 ✅

#### `configs/config.yaml` 수정

1. **`l5.ridge_alpha`**: `1.5` → `8.0`
   ```yaml
   ridge_alpha: 8.0  # [Hit Ratio 개선안] 과적합 방지 강화
   ```

2. **`l5.min_feature_ic`**: `-0.1` (이미 설정됨)
   ```yaml
   min_feature_ic: -0.1  # 모든 피처 사용 (음수 IC 포함)
   ```

## 검증 방법

### L8 실행 시 확인 사항

1. **로그 확인**: 
   ```
   [L8-L5 통일] Loaded 22 features from configs/features_short_v1.yaml
   [L8-L5 통일] Loaded 19 features from configs/features_long_v1.yaml
   ```

2. **피처 수 확인**:
   - 단기 랭킹: 22개 피처 사용
   - 장기 랭킹: 19개 피처 사용

3. **OHLCV 제거 확인**:
   - `close`, `high`, `low`, `open` 피처가 사용되지 않음

## 예상 효과

1. **피처셋 통일**: L5와 L8이 동일한 피처 사용 → 일관성 향상
2. **OHLCV 제거**: 불필요한 피처 제거 → 노이즈 감소
3. **News/ESG 포함**: 단기/장기 각각 News/ESG 피처 포함 → 예측력 향상 가능
4. **ridge_alpha 증가**: 8.0으로 증가 → 과적합 방지 강화
5. **모든 피처 사용**: min_feature_ic=-0.1 → 더 많은 피처 활용

## 다음 단계

1. **L8 랭킹 엔진 재실행**:
   ```bash
   python -m src.pipeline.track_a_pipeline
   ```

2. **Hit Ratio 재측정**:
   ```bash
   python scripts/measure_ranking_hit_ratio.py --generate-l8
   ```

3. **결과 확인**:
   - 로그에서 "Loaded 22 features" 확인
   - Hit Ratio 개선 여부 확인

## 변경 파일 목록

### 코드 파일
- `src/components/ranking/score_engine.py`
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`

### 설정 파일
- `configs/config.yaml`
- `configs/features_short_v1.yaml` (신규 생성)
- `configs/feature_weights_short_hitratio_optimized.yaml` (수정)
- `configs/feature_weights_long_ic_optimized.yaml` (수정)

