# 데이터 사용 가이드

## 📊 포함된 샘플 데이터

이 폴더에는 프로젝트 재현을 위한 **최소 샘플 데이터**가 포함되어 있습니다.

### interim/ 폴더
- `bt_metrics_bt120_long.csv` - 주요 백테스트 전략 성과 데이터
- 기타 백테스트 메트릭 CSV 파일들

### external/ 폴더
- `sector_map.csv` - 섹터 분류 매핑 데이터

## 🔄 전체 데이터 복원 (선택사항)

프로젝트의 완전한 실행을 위해서는 추가 데이터가 필요할 수 있습니다:

```bash
# 전체 외부 데이터 복원 (ESG, 뉴스 데이터)
cp -r LOCAL_TRASH/artifacts_data/data/external/* data/external/

# 추가 중간 데이터 복원
cp -r LOCAL_TRASH/artifacts_data/data/interim/* data/interim/
```

## 💡 데이터 용량 최적화

- **샘플 데이터**: 재현성 확보를 위한 최소 데이터셋
- **전체 데이터**: LOCAL_TRASH에 안전하게 보관
- **선택적 복원**: 필요시 언제든지 복원 가능

## 📈 데이터 구조

```
data/
├── interim/          # 백테스트 중간 결과
├── external/         # 외부 데이터 (ESG, 뉴스, 섹터)
└── sample_data_readme.md  # 이 파일
```
