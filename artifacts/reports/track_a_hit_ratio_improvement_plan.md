# Track A 랭킹산정 Hit Ratio 개선 방안

**생성일시**: 2025-01-XX
**분석 대상**: Track A 랭킹산정 파이프라인 실제 코드

---

## 🔍 현재 랭킹산정 파이프라인 분석

### 핵심 코드 경로

1. **Track A 파이프라인**: `src/pipeline/track_a_pipeline.py`
2. **L8 단기/장기 랭킹**: `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`
3. **랭킹 엔진**: `src/components/ranking/score_engine.py`

### 현재 Hit Ratio

- 단기: 41.58% (Dev: 41.16%, Holdout: 43.08%)
- 장기: 38.72% (Dev: 38.13%, Holdout: 41.45%)
- 통합: 41.58% (Dev: 41.16%, Holdout: 43.08%)

**목표**: Hit Ratio ≥ 50%

---

## 🎯 Hit Ratio 개선 방안 (우선순위별)

### 1️⃣ 정규화 방법 변경 (즉시 적용 가능)

**현재**: `normalization_method: percentile`

**개선안**: `normalization_method: zscore`

**코드 위치**:
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py` (line 45, 191)
- `src/components/ranking/score_engine.py` (line 47-124)

**이유**:
- Percentile은 순위만 반영 (0~1 범위)
- Z-score는 실제 값의 분포를 반영 (평균 중심, 표준편차 스케일)
- 극단값에 덜 민감하여 노이즈 감소 기대

**예상 효과**: +2~3%p

**수정 방법**:
```yaml
l8_short:
  normalization_method: zscore  # percentile → zscore

l8_long:
  normalization_method: zscore  # percentile → zscore
```

---

### 2️⃣ 국면별 가중치 활성화 (중요도 높음)

**현재**: `market_regime_df=None` (국면별 가중치 미사용)

**개선안**: 시장 국면별 피처 가중치 적용

**코드 위치**:
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py` (line 136, 282)
- `src/components/ranking/score_engine.py` (line 244-302)

**이유**:
- Bull/Bear/Neutral 시장에서 효과적인 피처가 다름
- 국면별 가중치로 적응형 랭킹 가능
- 코드에 이미 구현되어 있으나 비활성화 상태

**예상 효과**: +3~5%p

**수정 방법**:
```python
# l8_dual_horizon.py 수정
# line 136, 282: market_regime_df=None → market_regime_df 생성
from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime

# 시장 국면 데이터 생성
ohlcv_daily = artifacts.get("ohlcv_daily")
if ohlcv_daily is not None:
    dates = input_df["date"].unique()
    market_regime_df = build_market_regime(
        rebalance_dates=dates,
        ohlcv_daily=ohlcv_daily,
        lookback_days=60,
    )

    # 국면별 가중치 로드
    regime_weights_config = load_regime_weights(...)
else:
    market_regime_df = None
    regime_weights_config = None
```

---

### 3️⃣ 피처 가중치 극단화 (즉시 적용 가능)

**현재**:
- 단기: Value 0.04, Profitability 0.075, Technical 0.025
- 장기: Value 0.05, Profitability 0.1, Technical 0.02

**개선안**: 예측력 높은 피처에 가중치 집중

**코드 위치**:
- `configs/feature_weights_short_hitratio_optimized.yaml`
- `configs/feature_weights_long_ic_optimized.yaml`

**예상 효과**: +1~2%p

**수정 예시**:
```yaml
# 단기: 모멘텀/기술적 지표 강조
feature_weights:
  roe: 0.15  # 0.075 → 0.15 (2배)
  roe_sector_z: 0.15  # 0.075 → 0.15
  price_momentum_20d: 0.05  # 0.025 → 0.05 (2배)
  momentum_3m: 0.05  # 0.025 → 0.05
  # 나머지 Technical 피처는 0.01로 축소
```

---

### 4️⃣ Sector-Relative 정규화 조정 (즉시 적용 가능)

**현재**: `use_sector_relative: true`

**개선안**:
- Option A: `use_sector_relative: false` (전체 시장 기준)
- Option B: 섹터별 정규화 유지하되 가중치 조정

**코드 위치**:
- `src/components/ranking/score_engine.py` (line 47-124)
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py` (line 48, 194)

**이유**:
- 섹터별 정규화가 일부 피처의 예측력을 약화시킬 수 있음
- 전체 시장 기준 정규화가 더 나은 성과를 낼 수 있음

**예상 효과**: +1~2%p

---

### 5️⃣ 피처 선택 최적화 (중기)

**현재**: `_pick_feature_cols()`에서 자동 선택

**개선안**: IC 기반 피처 필터링 추가

**코드 위치**:
- `src/components/ranking/score_engine.py` (line 25-45)

**수정 방법**:
```python
def _pick_feature_cols(df: pd.DataFrame, min_ic: float = 0.0) -> List[str]:
    """IC 기반 피처 필터링 추가"""
    cols = _pick_feature_cols_original(df)

    # IC 파일에서 필터링
    ic_df = pd.read_csv("artifacts/reports/feature_ic_dev.csv")
    good_features = set(ic_df[ic_df["rank_ic"] > min_ic]["feature"].tolist())
    cols = [c for c in cols if c in good_features]

    return cols
```

**예상 효과**: +1~2%p

---

## 📊 우선순위별 실행 계획

### 즉시 실행 (1일)

1. **정규화 방법 변경**: percentile → zscore
2. **피처 가중치 극단화**: 예측력 높은 피처 가중치 2배 증가

### 단기 개선 (2~3일)

3. **국면별 가중치 활성화**: 코드 수정 및 테스트
4. **Sector-Relative 정규화 조정**: false로 변경 테스트

### 중기 개선 (1주)

5. **피처 선택 최적화**: IC 기반 필터링 추가
6. **통합 가중치 최적화**: 단기/장기 결합 가중치 튜닝

---

## 🔧 구체적 코드 수정 사항

### 수정 1: 정규화 방법 변경

**파일**: `configs/config.yaml`

```yaml
l8_short:
  normalization_method: zscore  # percentile → zscore 변경

l8_long:
  normalization_method: zscore  # percentile → zscore 변경
```

### 수정 2: 국면별 가중치 활성화

**파일**: `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`

```python
# line 136, 282 수정
# market_regime_df=None → 시장 국면 데이터 생성

# 시장 국면 데이터 생성
ohlcv_path = interim_dir / "ohlcv_daily"
if artifact_exists(ohlcv_path):
    ohlcv_daily = load_artifact(ohlcv_path)
    from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime

    dates = input_df["date"].unique()
    market_regime_df = build_market_regime(
        rebalance_dates=dates,
        ohlcv_daily=ohlcv_daily,
        lookback_days=60,
        neutral_band=0.05,
        use_volume=True,
        use_volatility=True,
    )

    # 국면별 가중치 로드
    regime_weights_config = load_regime_weights(
        config_path=l8_short.get("regime_aware_weights_config"),
        base_dir=base_dir,
    ) if l8_short.get("regime_aware_weights_config") else None
else:
    market_regime_df = None
    regime_weights_config = None
```

### 수정 3: 피처 가중치 극단화

**파일**: `configs/feature_weights_short_hitratio_optimized.yaml`

```yaml
feature_weights:
  # 예측력 높은 피처 가중치 증가
  roe: 0.15  # 0.075 → 0.15
  roe_sector_z: 0.15  # 0.075 → 0.15
  price_momentum_20d: 0.05  # 0.025 → 0.05
  momentum_3m: 0.05  # 0.025 → 0.05
  momentum_6m: 0.05  # 0.025 → 0.05
  # 나머지 피처는 가중치 축소하여 합=1.0 유지
```

---

## 📈 예상 개선 효과

| 개선안 | 예상 효과 | 난이도 | 우선순위 |
|--------|-----------|--------|----------|
| 정규화 방법 변경 | +2~3%p | 낮음 | 1 |
| 국면별 가중치 활성화 | +3~5%p | 중간 | 2 |
| 피처 가중치 극단화 | +1~2%p | 낮음 | 3 |
| Sector-Relative 조정 | +1~2%p | 낮음 | 4 |
| 피처 선택 최적화 | +1~2%p | 높음 | 5 |

**누적 예상 효과**: +8~14%p (현재 41.58% → 49.58~55.58%)

---

## 🎯 즉시 실행 가능한 최적 조합

### 조합 1: 빠른 개선 (1일)

1. 정규화 방법: percentile → zscore
2. 피처 가중치 극단화
3. Sector-Relative: true → false

**예상 효과**: +4~7%p (41.58% → 45.58~48.58%)

### 조합 2: 최대 개선 (3일)

1. 정규화 방법: percentile → zscore
2. 국면별 가중치 활성화
3. 피처 가중치 극단화
4. Sector-Relative 조정

**예상 효과**: +7~12%p (41.58% → 48.58~53.58%)

---

## 📝 다음 단계

1. **즉시**: 정규화 방법 변경 테스트
2. **단기**: 국면별 가중치 활성화 구현
3. **중기**: 피처 가중치 그리드 서치 최적화

---

**분석 기준**: Track A 랭킹산정 파이프라인 실제 코드
**코드 경로**: `src/pipeline/track_a_pipeline.py`, `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`, `src/components/ranking/score_engine.py`
