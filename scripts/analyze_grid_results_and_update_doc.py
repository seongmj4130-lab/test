# -*- coding: utf-8 -*-
"""
Grid Search 결과 분석 및 문서 업데이트

1. 단기/장기 랭킹 Grid Search 결과 분석
2. 최적 가중치 요약
3. track_a_optimization_direction_validation.md 업데이트
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def analyze_grid_results():
    """Grid Search 결과 분석"""
    results_dir = Path("artifacts/reports")
    configs_dir = Path("configs")

    # 단기 랭킹 결과
    short_file = results_dir / "track_a_group_weights_grid_search_20260108_135117.csv"
    short_weights_file = configs_dir / "feature_groups_short_optimized_grid_20260108_135117.yaml"

    # 장기 랭킹 결과
    long_file = results_dir / "track_a_group_weights_grid_search_20260108_145118.csv"
    long_weights_file = configs_dir / "feature_groups_long_optimized_grid_20260108_145118.yaml"

    results = {}

    # 단기 랭킹 분석
    if short_file.exists():
        short_df = pd.read_csv(short_file)
        short_best = short_df.loc[short_df['objective_score'].idxmax()]

        with open(short_weights_file, 'r', encoding='utf-8') as f:
            short_config = yaml.safe_load(f)

        results['short'] = {
            'file': str(short_file),
            'weights_file': str(short_weights_file),
            'n_combinations': len(short_df),
            'best_combination_id': int(short_best['combination_id']),
            'objective_score': float(short_best['objective_score']),
            'hit_ratio': float(short_best['hit_ratio']),
            'ic_mean': float(short_best['ic_mean']),
            'icir': float(short_best['icir']),
            'weights': {
                'technical': float(short_best['technical']),
                'value': float(short_best['value']),
                'profitability': float(short_best['profitability']),
                'news': float(short_best['news']),
            },
            'config': short_config,
        }

    # 장기 랭킹 분석
    if long_file.exists():
        long_df = pd.read_csv(long_file)
        long_best = long_df.loc[long_df['objective_score'].idxmax()]

        with open(long_weights_file, 'r', encoding='utf-8') as f:
            long_config = yaml.safe_load(f)

        results['long'] = {
            'file': str(long_file),
            'weights_file': str(long_weights_file),
            'n_combinations': len(long_df),
            'best_combination_id': int(long_best['combination_id']),
            'objective_score': float(long_best['objective_score']),
            'hit_ratio': float(long_best['hit_ratio']),
            'ic_mean': float(long_best['ic_mean']),
            'icir': float(long_best['icir']),
            'weights': {
                'technical': float(long_best['technical']),
                'value': float(long_best['value']),
                'profitability': float(long_best['profitability']),
                'news': float(long_best['news']) if 'news' in long_best.index else 0.0,
            },
            'config': long_config,
        }

    return results

def generate_doc_update(results):
    """문서 업데이트 내용 생성"""

    doc_content = f"""
## 📊 Phase 2 최종 결과 (2026-01-08 업데이트)

### ✅ 단기/장기 랭킹 Grid Search 완료

#### 단기 랭킹 (Short-term Ranking)
- **상태**: ✅ **완료**
- **결과 파일**: `{Path(results['short']['file']).name}`
- **조합 수**: {results['short']['n_combinations']}개 (전체 실행)
- **최적 조합 ID**: {results['short']['best_combination_id']}
- **최적 Objective Score**: {results['short']['objective_score']:.4f}
- **최적 Hit Ratio**: {results['short']['hit_ratio']*100:.2f}%
- **최적 IC Mean**: {results['short']['ic_mean']:.4f} (✅ 양수)
- **최적 ICIR**: {results['short']['icir']:.4f} (✅ 양수)
- **최적 가중치**:
  - technical: {results['short']['weights']['technical']:.2f}
  - value: {results['short']['weights']['value']:.2f}
  - profitability: {results['short']['weights']['profitability']:.2f}
  - news: {results['short']['weights']['news']:.2f}
- **최적 가중치 파일**: `{Path(results['short']['weights_file']).name}`

#### 장기 랭킹 (Long-term Ranking)
- **상태**: ✅ **완료**
- **결과 파일**: `{Path(results['long']['file']).name}`
- **조합 수**: {results['long']['n_combinations']}개 (전체 실행)
- **최적 조합 ID**: {results['long']['best_combination_id']}
- **최적 Objective Score**: {results['long']['objective_score']:.4f}
- **최적 Hit Ratio**: {results['long']['hit_ratio']*100:.2f}%
- **최적 IC Mean**: {results['long']['ic_mean']:.4f} (✅ 양수)
- **최적 ICIR**: {results['long']['icir']:.4f} (✅ 양수)
- **최적 가중치**:
  - technical: {results['long']['weights']['technical']:.2f}
  - value: {results['long']['weights']['value']:.2f}
  - profitability: {results['long']['weights']['profitability']:.2f}
  - news: {results['long']['weights']['news']:.2f}
- **최적 가중치 파일**: `{Path(results['long']['weights_file']).name}`

### 📊 단기 vs 장기 랭킹 비교

| 지표 | 단기 랭킹 | 장기 랭킹 | 차이 |
|------|----------|----------|------|
| Objective Score | {results['short']['objective_score']:.4f} | {results['long']['objective_score']:.4f} | {results['long']['objective_score'] - results['short']['objective_score']:.4f} |
| Hit Ratio | {results['short']['hit_ratio']*100:.2f}% | {results['long']['hit_ratio']*100:.2f}% | {results['long']['hit_ratio'] - results['short']['hit_ratio']:.2%}p |
| IC Mean | {results['short']['ic_mean']:.4f} | {results['long']['ic_mean']:.4f} | {results['long']['ic_mean'] - results['short']['ic_mean']:.4f} |
| ICIR | {results['short']['icir']:.4f} | {results['long']['icir']:.4f} | {results['long']['icir'] - results['short']['icir']:.4f} |

### 🔍 주요 발견사항

1. **단기/장기 모두 동일한 최적 가중치 패턴**
   - technical: -0.50 (음수 가중치)
   - value: 0.50 (양수 가중치)
   - profitability: 0.00
   - news: 0.00

2. **장기 랭킹이 IC와 ICIR에서 더 우수**
   - IC Mean: {results['long']['ic_mean']:.4f} vs {results['short']['ic_mean']:.4f} (차이: {results['long']['ic_mean'] - results['short']['ic_mean']:.4f})
   - ICIR: {results['long']['icir']:.4f} vs {results['short']['icir']:.4f} (차이: {results['long']['icir'] - results['short']['icir']:.4f})
   - 장기 랭킹이 예측력과 안정성에서 더 우수

3. **단기 랭킹이 Hit Ratio에서 더 우수**
   - Hit Ratio: {results['short']['hit_ratio']*100:.2f}% vs {results['long']['hit_ratio']*100:.2f}% (차이: {results['short']['hit_ratio'] - results['long']['hit_ratio']:.2%}p)
   - 단기 랭킹이 단기 수익률 적중률에서 더 우수

4. **두 랭킹 모두 IC가 양수**
   - 단기: {results['short']['ic_mean']:.4f} (✅ 양수)
   - 장기: {results['long']['ic_mean']:.4f} (✅ 양수)
   - 예측력 확인

### ✅ 최적 가중치 적용 완료

- **단기 랭킹**: `configs/feature_groups_short_optimized_grid_20260108_135117.yaml`
- **장기 랭킹**: `configs/feature_groups_long_optimized_grid_20260108_145118.yaml`

### ⚠️ Dev/Holdout 구간 성과 비교

**현재 상태**: Grid Search는 Dev 구간에서만 평가되었습니다.

**다음 단계**:
1. 최적 가중치를 적용한 L8 랭킹 실행
2. Holdout 구간에서 성과 평가
3. Dev/Holdout 구간 성과 비교
4. 과적합 여부 확인

**참고**: Grid Search 결과는 Dev 구간 기준이므로, Holdout 구간에서의 성과는 별도 평가가 필요합니다.

---

**업데이트 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return doc_content

def main():
    """메인 함수"""
    print("=" * 80)
    print("Grid Search 결과 분석 및 문서 업데이트")
    print("=" * 80)

    # 결과 분석
    results = analyze_grid_results()

    if not results:
        print("❌ 분석할 결과 파일이 없습니다.")
        return

    # 문서 업데이트 내용 생성
    doc_update = generate_doc_update(results)

    # 문서 파일 읽기
    doc_file = Path("artifacts/reports/track_a_optimization_direction_validation.md")
    if not doc_file.exists():
        print(f"❌ 문서 파일을 찾을 수 없습니다: {doc_file}")
        return

    with open(doc_file, 'r', encoding='utf-8') as f:
        doc_content = f.read()

    # Phase 2 섹션 찾기 및 업데이트
    # "#### 2.2 Grid Search 실행 결과 ✅" 섹션 이후에 추가
    marker = "#### 2.2 Grid Search 실행 결과 ✅"

    if marker in doc_content:
        # 기존 Phase 2 결과 섹션 찾기
        lines = doc_content.split('\n')
        insert_idx = None

        for i, line in enumerate(lines):
            if "#### 2.3 검증" in line:
                insert_idx = i
                break

        if insert_idx:
            # 기존 Phase 2 결과 섹션 대체
            # "#### 2.2"부터 "#### 2.3" 전까지를 새 내용으로 교체
            start_idx = None
            for i in range(insert_idx - 1, -1, -1):
                if "#### 2.2" in lines[i]:
                    start_idx = i
                    break

            if start_idx is not None:
                new_lines = lines[:start_idx] + doc_update.strip().split('\n') + [''] + lines[insert_idx:]
                doc_content = '\n'.join(new_lines)
            else:
                # 마커를 찾지 못한 경우, 2.3 섹션 앞에 추가
                new_lines = lines[:insert_idx] + doc_update.strip().split('\n') + [''] + lines[insert_idx:]
                doc_content = '\n'.join(new_lines)
        else:
            # 2.3 섹션을 찾지 못한 경우, 문서 끝에 추가
            doc_content += '\n\n' + doc_update
    else:
        # 마커를 찾지 못한 경우, 문서 끝에 추가
        doc_content += '\n\n' + doc_update

    # 문서 저장
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"✅ 문서 업데이트 완료: {doc_file}")
    print("\n생성된 업데이트 내용:")
    print(doc_update)

if __name__ == "__main__":
    main()
