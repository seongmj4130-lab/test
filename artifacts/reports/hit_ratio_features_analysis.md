# Hit Ratio 관여 피처 분석 (단기/장기 각각 50% 목표)

## 현재 Hit Ratio 현황

| 랭킹 타입 | 현재 Hit Ratio | 목표 | 차이 |
|----------|---------------|------|------|
| **단기 랭킹** | **41.58%** | 50% | **-8.42%p** |
| **장기 랭킹** | **38.72%** | 50% | **-11.28%p** |

## 현재 사용 중인 피처 (총 28개)

### 1. Value 그룹 (5개 피처) - 가치 팩터
- `equity` (자본금)
- `total_liabilities` (총 부채)
- `net_income` (순이익)
- `debt_ratio` (부채비율)
- `debt_ratio_sector_z` (부채비율 섹터 상대)

**현재 가중치**:
- 단기: 20% (그룹별)
- 장기: 25% (그룹별)

**IC 성과**:
- 단기: Rank IC = 0.025 (양수, 좋음)
- 장기: Rank IC = 0.034 (양수, 더 좋음)

### 2. Profitability 그룹 (2개 피처) - 수익성 팩터
- `roe` (자기자본이익률)
- `roe_sector_z` (ROE 섹터 상대)

**현재 가중치**:
- 단기: 15% (그룹별)
- 장기: 20% (그룹별)

**IC 성과**:
- 단기: Rank IC = -0.001 (거의 0, 약한 음수)
- 장기: Rank IC = -0.021 (음수, 약함)

### 3. Technical 그룹 (20개 피처) - 기술적 지표
- `volatility_60d` (60일 변동성)
- `volatility_20d` (20일 변동성)
- `volatility` (변동성)
- `downside_volatility_60d` (하방 변동성)
- `price_momentum_60d` (60일 가격 모멘텀)
- `price_momentum_20d` (20일 가격 모멘텀)
- `price_momentum` (가격 모멘텀)
- `momentum_rank` (모멘텀 랭킹)
- `momentum_3m` (3개월 모멘텀)
- `momentum_6m` (6개월 모멘텀)
- `momentum_reversal` (모멘텀 리버설)
- `max_drawdown_60d` (60일 최대 낙폭)
- `volume` (거래량)
- `volume_ratio` (거래량 비율)
- `turnover` (회전율)
- `close` (종가)
- `high` (고가)
- `low` (저가)
- `open` (시가)
- `ret_daily` (일일 수익률)

**현재 가중치**:
- 단기: 50% (그룹별) - 가장 높음
- 장기: 40% (그룹별)

**IC 성과**:
- 단기: Rank IC = -0.022 (음수, 약함)
- 장기: Rank IC = -0.027 (음수, 더 약함)

### 4. Other 그룹 (1개 피처)
- `in_universe` (유니버스 포함 여부)

**현재 가중치**:
- 단기: 10% (그룹별)
- 장기: 10% (그룹별)

**IC 성과**: 거의 0 (의미 없음)

### 5. News 그룹 (0개 피처)
- 현재 사용되지 않음

## 피처별 IC 상세 분석

### 단기 랭킹 (20일 수익률 예측) - Top 10 피처

| 순위 | 피처 | Rank IC | IC 방향 | 평가 |
|------|------|---------|---------|------|
| 1 | `equity` | 0.120 | 양수 ✅ | 강한 예측력 |
| 2 | `total_liabilities` | 0.094 | 양수 ✅ | 좋은 예측력 |
| 3 | `net_income` | 0.055 | 양수 ✅ | 양호 |
| 4 | `volume` | 0.016 | 양수 ✅ | 약함 |
| 5 | `max_drawdown_60d` | 0.012 | 양수 ✅ | 약함 |
| 6 | `debt_ratio` | 0.011 | 양수 ✅ | 약함 |
| 7 | `debt_ratio_sector_z` | 0.002 | 양수 ✅ | 매우 약함 |
| 8 | `turnover` | -0.002 | 음수 ❌ | 약한 음수 |
| 9 | `roe_sector_z` | -0.001 | 음수 ❌ | 거의 0 |
| 10 | `roe` | -0.001 | 음수 ❌ | 거의 0 |

**단기 랭킹 문제점**:
- Technical 그룹 피처들이 대부분 음수 IC (모멘텀, 변동성 등)
- Value 그룹만 양수 IC를 보임
- Technical 그룹에 50% 가중치를 주고 있으나 IC가 음수

### 장기 랭킹 (120일 수익률 예측) - Top 10 피처

| 순위 | 피처 | Rank IC | IC 방향 | 평가 |
|------|------|---------|---------|------|
| 1 | `equity` | 0.224 | 양수 ✅ | 매우 강한 예측력 |
| 2 | `total_liabilities` | 0.150 | 양수 ✅ | 강한 예측력 |
| 3 | `net_income` | 0.107 | 양수 ✅ | 좋은 예측력 |
| 4 | `volume` | 0.098 | 양수 ✅ | 양호 |
| 5 | `max_drawdown_60d` | 0.093 | 양수 ✅ | 양호 |
| 6 | `debt_ratio` | 0.018 | 양수 ✅ | 약함 |
| 7 | `momentum_reversal` | 0.020 | 양수 ✅ | 약함 |
| 8 | `volume_ratio` | 0.014 | 양수 ✅ | 약함 |
| 9 | `debt_ratio_sector_z` | 0.004 | 양수 ✅ | 매우 약함 |
| 10 | `turnover` | -0.010 | 음수 ❌ | 약한 음수 |

**장기 랭킹 문제점**:
- Value 그룹이 더 강한 예측력을 보임 (단기보다 좋음)
- Technical 그룹 피처들이 여전히 음수 IC
- Profitability 그룹(ROE)이 음수 IC

## Hit Ratio 개선을 위한 피처 전략

### 단기 랭킹 (41.58% → 50% 목표, +8.42%p 필요)

#### 1. Value 그룹 가중치 증가
- **현재**: 20%
- **제안**: 30~35%
- **이유**: Value 그룹이 유일하게 양수 IC를 보임

#### 2. Technical 그룹 가중치 감소
- **현재**: 50%
- **제안**: 35~40%
- **이유**: Technical 그룹이 음수 IC를 보임

#### 3. 음수 IC 피처 제거 또는 가중치 감소
- `volatility_60d`: Rank IC = -0.283 (매우 강한 음수)
- `volatility_20d`: Rank IC = -0.345 (매우 강한 음수)
- `price_momentum_60d`: Rank IC = -0.443 (매우 강한 음수)
- `price_momentum_20d`: Rank IC = -0.513 (매우 강한 음수)
- `price_momentum`: Rank IC = -0.513 (매우 강한 음수)
- `momentum_rank`: Rank IC = -0.513 (매우 강한 음수)

**제안**: 이 피처들의 가중치를 0으로 설정하거나 매우 낮게 설정

#### 4. 양수 IC 피처 가중치 증가
- `equity`: 가중치 증가
- `total_liabilities`: 가중치 증가
- `net_income`: 가중치 증가
- `volume`: 가중치 증가

### 장기 랭킹 (38.72% → 50% 목표, +11.28%p 필요)

#### 1. Value 그룹 가중치 대폭 증가
- **현재**: 25%
- **제안**: 40~45%
- **이유**: Value 그룹이 매우 강한 양수 IC를 보임 (단기보다 더 좋음)

#### 2. Technical 그룹 가중치 감소
- **현재**: 40%
- **제안**: 25~30%
- **이유**: Technical 그룹이 음수 IC를 보임

#### 3. Profitability 그룹 가중치 감소
- **현재**: 20%
- **제안**: 10~15%
- **이유**: ROE 피처들이 음수 IC를 보임

#### 4. 음수 IC 피처 제거
- `volatility_60d`: Rank IC = -0.369 (매우 강한 음수)
- `volatility_20d`: Rank IC = -0.389 (매우 강한 음수)
- `downside_volatility_60d`: Rank IC = -0.186 (강한 음수)
- `price_momentum_60d`: Rank IC = -0.330 (강한 음수)
- `roe`: Rank IC = -0.080 (음수)
- `roe_sector_z`: Rank IC = -0.066 (음수)

## 구체적인 개선 제안

### 단기 랭킹 피처 가중치 재조정

```yaml
# configs/feature_weights_short_hitratio_optimized.yaml (수정안)
group_weights:
  value: 0.35        # 0.2 → 0.35 (75% 증가)
  profitability: 0.10 # 0.15 → 0.10 (33% 감소)
  technical: 0.40    # 0.5 → 0.4 (20% 감소)
  other: 0.10        # 유지
  news: 0.05         # 유지

# 개별 피처 가중치 (음수 IC 피처 제거/감소)
feature_weights:
  # Value 그룹 (증가)
  equity: 0.10              # 0.04 → 0.10
  total_liabilities: 0.10   # 0.04 → 0.10
  net_income: 0.10          # 0.04 → 0.10
  debt_ratio: 0.03          # 0.04 → 0.03
  debt_ratio_sector_z: 0.02 # 0.04 → 0.02
  
  # Profitability 그룹 (감소)
  roe: 0.05                 # 0.075 → 0.05
  roe_sector_z: 0.05        # 0.075 → 0.05
  
  # Technical 그룹 (음수 IC 피처 제거/감소)
  volatility_60d: 0.0       # 0.025 → 0.0 (제거)
  volatility_20d: 0.0       # 0.025 → 0.0 (제거)
  volatility: 0.0           # 0.025 → 0.0 (제거)
  price_momentum_60d: 0.0   # 0.025 → 0.0 (제거)
  price_momentum_20d: 0.0   # 0.025 → 0.0 (제거)
  price_momentum: 0.0       # 0.025 → 0.0 (제거)
  momentum_rank: 0.0        # 0.025 → 0.0 (제거)
  
  # 양수 IC 피처 유지/증가
  volume: 0.05              # 0.025 → 0.05 (증가)
  max_drawdown_60d: 0.03    # 0.025 → 0.03 (증가)
  
  # 기타
  downside_volatility_60d: 0.01
  momentum_3m: 0.01
  momentum_6m: 0.01
  momentum_reversal: 0.01
  volume_ratio: 0.01
  turnover: 0.01
  close: 0.01
  high: 0.01
  low: 0.01
  open: 0.01
  ret_daily: 0.01
  in_universe: 0.1
```

### 장기 랭킹 피처 가중치 재조정

```yaml
# configs/feature_weights_long_ic_optimized.yaml (수정안)
group_weights:
  value: 0.45        # 0.25 → 0.45 (80% 증가)
  profitability: 0.10 # 0.2 → 0.10 (50% 감소)
  technical: 0.30    # 0.4 → 0.3 (25% 감소)
  other: 0.10        # 유지
  news: 0.05         # 유지

# 개별 피처 가중치
feature_weights:
  # Value 그룹 (대폭 증가)
  equity: 0.15              # 0.05 → 0.15
  total_liabilities: 0.12   # 0.05 → 0.12
  net_income: 0.12          # 0.05 → 0.12
  debt_ratio: 0.03          # 0.05 → 0.03
  debt_ratio_sector_z: 0.03 # 0.05 → 0.03
  
  # Profitability 그룹 (감소)
  roe: 0.05                 # 0.1 → 0.05
  roe_sector_z: 0.05        # 0.1 → 0.05
  
  # Technical 그룹 (음수 IC 피처 제거)
  volatility_60d: 0.0       # 제거
  volatility_20d: 0.0       # 제거
  volatility: 0.0           # 제거
  downside_volatility_60d: 0.0 # 제거
  price_momentum_60d: 0.0   # 제거
  price_momentum_20d: 0.0   # 제거
  price_momentum: 0.0       # 제거
  momentum_rank: 0.0        # 제거
  
  # 양수 IC 피처 유지/증가
  volume: 0.08              # 0.02 → 0.08
  max_drawdown_60d: 0.08    # 0.02 → 0.08
  momentum_reversal: 0.05   # 0.02 → 0.05
  volume_ratio: 0.03        # 0.02 → 0.03
  
  # 기타
  momentum_3m: 0.01
  momentum_6m: 0.01
  close: 0.01
  high: 0.01
  low: 0.01
  open: 0.01
  ret_daily: 0.01
  turnover: 0.01
  in_universe: 0.1
```

## 예상 효과

### 단기 랭킹
- **현재**: 41.58%
- **예상**: 45~48% (+3.5~6.5%p)
- **목표 달성**: 50%까지 추가 2~5%p 필요

### 장기 랭킹
- **현재**: 38.72%
- **예상**: 45~48% (+6.3~9.3%p)
- **목표 달성**: 50%까지 추가 2~5%p 필요

## 추가 개선 방안

1. **새로운 피처 추가**
   - PER, PBR 등 추가 가치 지표
   - 섹터 상대 모멘텀
   - 거래량 모멘텀

2. **피처 변환**
   - 음수 IC 피처를 역변환 (예: -momentum → reversal signal)
   - 섹터 상대 변환 강화

3. **국면별 피처 가중치**
   - Bull 시장: 모멘텀 피처 강조
   - Bear 시장: 가치 피처 강조

## 다음 단계

1. ✅ 피처 분석 완료
2. ⏳ 피처 가중치 재조정 파일 생성
3. ⏳ Hit Ratio 재측정
4. ⏳ 결과 비교 및 추가 최적화

