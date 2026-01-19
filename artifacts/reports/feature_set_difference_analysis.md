# L5 vs L8 피처셋 차이 분석

## 핵심 차이점

**L5 (모델 학습)**와 **L8 (랭킹 엔진)**은 **서로 다른 피처셋**을 사용합니다!

### L5: 모델 학습 단계 (Ridge 회귀)

**피처 선택 방식**: `feature_list_short.yaml`, `feature_list_long.yaml` (고정 리스트)

**단기 모델**: **22개 피처**
- Core 공통: 12개
- Short 전용: 6개
- News 감성: 4개

**장기 모델**: **19개 피처**
- Core 공통: 12개
- Long 전용: 7개

**코드 위치**: `src/stages/modeling/l5_train_models.py::_pick_feature_cols()`

**설정**: `config.yaml::l5.feature_list_short`, `feature_list_long`

### L8: 랭킹 엔진 단계 (Score Engine)

**피처 선택 방식**: `_pick_feature_cols()` (자동 선택) + `feature_weights_config` (가중치)

**단기 랭킹**: **28개 피처** (feature_weights_short_hitratio_optimized.yaml)
**장기 랭킹**: **28개 피처** (feature_weights_long_ic_optimized.yaml)

**코드 위치**: `src/components/ranking/score_engine.py::_pick_feature_cols()`

**설정**: `config.yaml::l8_short.feature_weights_config`, `l8_long.feature_weights_config`

## 자동 피처 선택 로직 (L8)

```python
# score_engine.py:25-45
def _pick_feature_cols(df: pd.DataFrame) -> List[str]:
    """피처 컬럼 선택 (식별자/타겟 제외)"""
    exclude = {
        "date", "ticker",
        "ret_fwd_20d", "ret_fwd_120d",  # 타겟 제외
        "split", "phase", "segment", "fold_id",
        "in_universe", "ym", "corp_code",
        "open", "high", "low", "close", "volume",  # OHLCV는 피처로 사용하지 않음
    }

    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            cols.append(c)  # 숫자형 컬럼만 선택

    return sorted(cols)
```

**의미**: L8은 **데이터에 있는 모든 숫자형 피처를 자동으로 선택**합니다 (OHLCV 제외).

## 피처셋 비교

### L5 단기 모델 (22개) vs L8 단기 랭킹 (28개)

| 구분 | L5 (모델 학습) | L8 (랭킹 엔진) |
|------|---------------|----------------|
| **피처 수** | 22개 | 28개 |
| **선택 방식** | 고정 리스트 (feature_list_short.yaml) | 자동 선택 + 가중치 파일 |
| **News 피처** | 4개 포함 | 포함 여부 불명확 |
| **OHLCV** | 제외 | 제외 (명시적) |
| **ESG** | 제외 | 포함 가능 |

### L5 장기 모델 (19개) vs L8 장기 랭킹 (28개)

| 구분 | L5 (모델 학습) | L8 (랭킹 엔진) |
|------|---------------|----------------|
| **피처 수** | 19개 | 28개 |
| **선택 방식** | 고정 리스트 (feature_list_long.yaml) | 자동 선택 + 가중치 파일 |
| **ESG 피처** | 4개 포함 | 포함 가능 |
| **News 피처** | 1개 (ewm20) | 포함 가능 |

## 현재 L8에서 사용 중인 피처 (28개)

### feature_weights_short_hitratio_optimized.yaml 기준

1. **Value 그룹 (5개)**
   - `equity`
   - `total_liabilities`
   - `net_income`
   - `debt_ratio`
   - `debt_ratio_sector_z`

2. **Profitability 그룹 (2개)**
   - `roe`
   - `roe_sector_z`

3. **Technical 그룹 (20개)**
   - `volatility_60d`
   - `volatility_20d`
   - `volatility`
   - `downside_volatility_60d`
   - `price_momentum_60d`
   - `price_momentum_20d`
   - `price_momentum`
   - `momentum_rank`
   - `momentum_3m`
   - `momentum_6m`
   - `momentum_reversal`
   - `max_drawdown_60d`
   - `volume`
   - `volume_ratio`
   - `turnover`
   - `close` ⚠️ (OHLCV인데 포함됨?)
   - `high` ⚠️
   - `low` ⚠️
   - `open` ⚠️
   - `ret_daily`

4. **Other 그룹 (1개)**
   - `in_universe`

**총 28개**

## 문제점 발견

### 1. OHLCV 피처 포함
- `close`, `high`, `low`, `open`이 feature_weights에 포함되어 있음
- 하지만 `_pick_feature_cols()`는 OHLCV를 제외함
- **결과**: 이 피처들은 실제로 사용되지 않을 수 있음

### 2. L5와 L8 피처셋 불일치
- L5는 22개/19개 피처 사용
- L8은 28개 피처 가중치 정의
- **실제 사용되는 피처는 데이터에 존재하는 피처에 따라 달라짐**

### 3. News 피처 포함 여부 불명확
- L5 단기 모델: News 4개 포함
- L8 feature_weights: News 그룹 가중치 있음 (0.05)이지만 피처 목록에 없음

## 해결 방안

### 1. L8 피처 선택 명시화
- `feature_list_short.yaml`, `feature_list_long.yaml`을 L8에서도 사용
- 또는 L8 전용 피처 리스트 파일 생성

### 2. feature_weights 정리
- 실제 사용되는 피처만 가중치 정의
- OHLCV 피처 제거
- News 피처 포함 여부 명확화

### 3. L5와 L8 피처셋 통일 검토
- L5에서 사용하는 피처를 L8에서도 사용
- 또는 L8 전용 피처 추가 (랭킹 엔진 특화)

## 현재 Hit Ratio에 관여하는 피처

**실제로 사용되는 피처**는:
1. `dataset_daily.parquet`에 존재하는 피처
2. `_pick_feature_cols()`가 선택한 피처 (OHLCV 제외)
3. `feature_weights_config`에 가중치가 정의된 피처

**현재 상황**:
- feature_weights에 28개 피처 가중치 정의
- 실제 사용되는 피처는 데이터에 따라 달라짐
- OHLCV (`close`, `high`, `low`, `open`)는 제외됨
- **실제 사용 피처 수: 약 24개 정도로 추정**

## 결론

1. **L5와 L8은 서로 다른 피처셋 사용**
2. **L8은 자동 피처 선택** (데이터 기반)
3. **feature_weights는 28개 정의**하지만 실제 사용은 데이터에 따라 달라짐
4. **OHLCV 피처는 제외**되지만 가중치 파일에 포함되어 있음 (불일치)

**Hit Ratio 개선을 위해서는**:
- 실제 사용되는 피처 확인 필요
- L5 피처셋과 L8 피처셋 통일 검토
- feature_weights 정리 (사용되지 않는 피처 제거)
