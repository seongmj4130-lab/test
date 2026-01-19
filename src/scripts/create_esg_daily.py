# -*- coding: utf-8 -*-
# src/scripts/create_esg_daily.py
# ESG 데이터를 통합하여 esg_daily.parquet 생성
# - 동적 코스피 유니버스 기준 필터링
# - 2016/01/01 이후 데이터만 포함

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import load_artifact


def main():
    """ESG 데이터를 통합하여 esg_daily.parquet 생성"""

    # 1. 경로 설정 (root는 이미 위에서 정의됨)
    cfg_path = root / "configs" / "config.yaml"
    cfg = load_config(str(cfg_path))

    esg_source_dir = root / "data" / "external" / "esg_by_company"
    output_path = root / "data" / "external" / "esg_daily.parquet"
    interim_dir = get_path(cfg, "data_interim")
    universe_path = interim_dir / "universe_k200_membership_monthly"

    print("=" * 60)
    print("ESG Daily 데이터 생성")
    print("=" * 60)
    print(f"원본 폴더: {esg_source_dir}")
    print(f"출력 파일: {output_path}")
    print(f"유니버스 파일: {universe_path}")

    # 2. 유니버스 로드
    print("\n[1/4] 유니버스 데이터 로드 중...")
    if not universe_path.with_suffix(".parquet").exists():
        raise FileNotFoundError(
            f"유니버스 파일을 찾을 수 없습니다: {universe_path}.parquet\n"
            "먼저 L0 단계를 실행하여 universe_k200_membership_monthly를 생성하세요."
        )

    universe = load_artifact(universe_path)
    print(f"  - 유니버스 로드 완료: {len(universe):,}행")
    print(f"  - 기간: {universe['date'].min()} ~ {universe['date'].max()}")
    print(f"  - 종목 수: {universe['ticker'].nunique():,}개")

    # 유니버스 데이터 정리
    universe['date'] = pd.to_datetime(universe['date'])
    universe['ticker'] = universe['ticker'].astype(str).str.zfill(6)

    # 3. ESG CSV 파일 읽기
    print("\n[2/4] ESG CSV 파일 읽기 중...")
    csv_files = sorted(esg_source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ESG CSV 파일을 찾을 수 없습니다: {esg_source_dir}")

    print(f"  - 발견된 CSV 파일: {len(csv_files):,}개")

    dfs = []
    for i, csv_file in enumerate(csv_files, 1):
        if i % 50 == 0:
            print(f"  - 진행 중: {i}/{len(csv_files)} 파일 처리 완료")

        try:
            df = pd.read_csv(csv_file, encoding='utf-8')

            # 컬럼명 확인
            if '일자' not in df.columns or 'ticker' not in df.columns:
                print(f"    경고: {csv_file.name} - 필수 컬럼 누락, 스킵")
                continue

            # 컬럼명 표준화
            df = df.rename(columns={'일자': 'date'})

            # 날짜 변환
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['date'])

            # ticker 정리
            df['ticker'] = df['ticker'].astype(str).str.zfill(6)

            dfs.append(df)

        except Exception as e:
            print(f"    경고: {csv_file.name} 읽기 실패 - {e}")
            continue

    if not dfs:
        raise RuntimeError("읽을 수 있는 ESG 파일이 없습니다.")

    # 4. 데이터 통합
    print("\n[3/4] 데이터 통합 중...")
    esg_all = pd.concat(dfs, ignore_index=True)
    print(f"  - 통합 전 행 수: {len(esg_all):,}")

    # 5. 기간 필터링 (2016/01/01 이후)
    print("\n[4/4] 필터링 중...")
    cutoff_date = pd.to_datetime('2016-01-01')
    esg_all = esg_all[esg_all['date'] >= cutoff_date].copy()
    print(f"  - 기간 필터링 후: {len(esg_all):,}행 (2016-01-01 이후)")

    # 6. 동적 유니버스 필터링
    # 유니버스는 월말 기준이므로, 각 ESG 데이터의 날짜가 속한 월의 유니버스와 매칭
    esg_all['ym'] = esg_all['date'].dt.to_period('M').astype(str)
    universe['ym'] = universe['date'].dt.to_period('M').astype(str)

    # 각 월별로 유니버스에 포함된 ticker만 필터링
    universe_by_ym = universe.groupby('ym')['ticker'].apply(set).to_dict()

    def is_in_universe(row):
        ym = row['ym']
        ticker = row['ticker']
        if ym not in universe_by_ym:
            return False
        return ticker in universe_by_ym[ym]

    before_universe_filter = len(esg_all)
    esg_all['in_universe'] = esg_all.apply(is_in_universe, axis=1)
    esg_all = esg_all[esg_all['in_universe']].copy()
    esg_all = esg_all.drop(columns=['in_universe', 'ym'])

    print(f"  - 유니버스 필터링 후: {len(esg_all):,}행")
    print(f"  - 제거된 행: {before_universe_filter - len(esg_all):,}행")

    # 7. 최종 정리
    # 컬럼 순서 정리
    cols = ['date', 'ticker']
    if 'pred_label' in esg_all.columns:
        cols.append('pred_label')
    if 'ESG_Label' in esg_all.columns:
        cols.append('ESG_Label')
    cols.extend([c for c in esg_all.columns if c not in cols])

    esg_all = esg_all[cols].copy()
    esg_all = esg_all.sort_values(['date', 'ticker']).reset_index(drop=True)

    # 8. 저장
    print(f"\n[저장] {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    esg_all.to_parquet(output_path, index=False, engine='pyarrow')

    # 9. 요약 정보 출력
    print("\n" + "=" * 60)
    print("생성 완료!")
    print("=" * 60)
    print(f"출력 파일: {output_path}")
    print(f"총 행 수: {len(esg_all):,}")
    print(f"기간: {esg_all['date'].min().date()} ~ {esg_all['date'].max().date()}")
    print(f"종목 수: {esg_all['ticker'].nunique():,}개")
    print(f"컬럼: {list(esg_all.columns)}")

    if 'pred_label' in esg_all.columns:
        print(f"\npred_label 분포:")
        print(esg_all['pred_label'].value_counts().to_string())

    if 'ESG_Label' in esg_all.columns:
        print(f"\nESG_Label 분포:")
        print(esg_all['ESG_Label'].value_counts().to_string())


if __name__ == "__main__":
    main()
