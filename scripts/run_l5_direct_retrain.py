"""
L5 모델 직접 재학습 스크립트 (단기/장기, Dev/Holdout 모두 포함)

Ridge alpha 16.0으로 모델 재학습
- 단기 랭킹 (horizon=20): Dev + Holdout
- 장기 랭킹 (horizon=120): Dev + Holdout

실행 결과를 실시간으로 터미널에 출력
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stages.modeling.l5_train_models import train_oos_predictions


def main():
    """메인 함수"""
    print("=" * 80)
    print("L5 모델 재학습 (Ridge Alpha 16.0)")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"프로젝트 루트: {project_root}")
    print("=" * 80)

    # Config 로드
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        print(f"❌ Config 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ridge alpha 확인
    ridge_alpha = cfg.get("l5", {}).get("ridge_alpha", 8.0)
    print(f"\n현재 Ridge Alpha: {ridge_alpha}")
    if ridge_alpha != 16.0:
        print(f"⚠️ 경고: Ridge Alpha가 16.0이 아닙니다. 현재 값: {ridge_alpha}")

    # 데이터 경로
    data_dir = project_root / "data" / "interim"
    interim_dir = data_dir / f"l5_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 필수 데이터 파일 확인
    dataset_daily = data_dir / "dataset_daily.parquet"
    cv_folds_short = data_dir / "cv_folds_short.parquet"
    cv_folds_long = data_dir / "cv_folds_long.parquet"

    required_files = [dataset_daily, cv_folds_short, cv_folds_long]
    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print("❌ 필수 데이터 파일이 없습니다:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("학습 범위:")
    print("  - 단기 랭킹 (horizon=20): Dev + Holdout")
    print("  - 장기 랭킹 (horizon=120): Dev + Holdout")
    print("=" * 80 + "\n")

    # 데이터 로드
    print("데이터 로드 중...")
    dataset_df = pd.read_parquet(dataset_daily)
    cv_folds_short_df = pd.read_parquet(cv_folds_short)
    cv_folds_long_df = pd.read_parquet(cv_folds_long)

    print(f"  - dataset_daily: {len(dataset_df):,}행")
    print(f"  - cv_folds_short: {len(cv_folds_short_df)}개 fold")
    print(f"  - cv_folds_long: {len(cv_folds_long_df)}개 fold")

    all_warns = []
    all_results = {}

    # 단기 랭킹 학습
    print("\n" + "=" * 80)
    print("[단기 랭킹 (horizon=20)] 학습 시작")
    print("=" * 80)

    try:
        pred_short, metrics_short, report_short, warns_short = train_oos_predictions(
            dataset_daily=dataset_df,
            cv_folds=cv_folds_short_df,
            target_col="ret_fwd_20d",
            horizon=20,
            cfg=cfg,
            interim_dir=interim_dir,
        )

        all_warns.extend(warns_short)
        all_results["short"] = {
            "predictions": pred_short,
            "metrics": metrics_short,
            "report": report_short,
        }

        print("\n✅ 단기 랭킹 학습 완료")
        print(f"  - 예측 행 수: {len(pred_short):,}")
        print(f"  - Dev folds: {report_short.get('dev_folds', 0)}")
        print(f"  - Holdout folds: {report_short.get('holdout_folds', 0)}")
        if "dev_ic_rank_mean" in report_short:
            print(f"  - Dev IC Rank Mean: {report_short['dev_ic_rank_mean']:.4f}")
        if "holdout_ic_rank_mean" in report_short:
            print(
                f"  - Holdout IC Rank Mean: {report_short['holdout_ic_rank_mean']:.4f}"
            )

    except Exception as e:
        print(f"\n❌ 단기 랭킹 학습 실패: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 장기 랭킹 학습
    print("\n" + "=" * 80)
    print("[장기 랭킹 (horizon=120)] 학습 시작")
    print("=" * 80)

    try:
        pred_long, metrics_long, report_long, warns_long = train_oos_predictions(
            dataset_daily=dataset_df,
            cv_folds=cv_folds_long_df,
            target_col="ret_fwd_120d",
            horizon=120,
            cfg=cfg,
            interim_dir=interim_dir,
        )

        all_warns.extend(warns_long)
        all_results["long"] = {
            "predictions": pred_long,
            "metrics": metrics_long,
            "report": report_long,
        }

        print("\n✅ 장기 랭킹 학습 완료")
        print(f"  - 예측 행 수: {len(pred_long):,}")
        print(f"  - Dev folds: {report_long.get('dev_folds', 0)}")
        print(f"  - Holdout folds: {report_long.get('holdout_folds', 0)}")
        if "dev_ic_rank_mean" in report_long:
            print(f"  - Dev IC Rank Mean: {report_long['dev_ic_rank_mean']:.4f}")
        if "holdout_ic_rank_mean" in report_long:
            print(
                f"  - Holdout IC Rank Mean: {report_long['holdout_ic_rank_mean']:.4f}"
            )

    except Exception as e:
        print(f"\n❌ 장기 랭킹 학습 실패: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 결과 저장
    print("\n" + "=" * 80)
    print("결과 저장 중...")
    print("=" * 80)

    # 예측 결과 저장
    pred_short_path = interim_dir / "predictions_short.parquet"
    pred_long_path = interim_dir / "predictions_long.parquet"
    all_results["short"]["predictions"].to_parquet(pred_short_path, index=False)
    all_results["long"]["predictions"].to_parquet(pred_long_path, index=False)
    print(f"  - 단기 예측: {pred_short_path}")
    print(f"  - 장기 예측: {pred_long_path}")

    # 메트릭 저장
    metrics_short_path = interim_dir / "metrics_short.parquet"
    metrics_long_path = interim_dir / "metrics_long.parquet"
    all_results["short"]["metrics"].to_parquet(metrics_short_path, index=False)
    all_results["long"]["metrics"].to_parquet(metrics_long_path, index=False)
    print(f"  - 단기 메트릭: {metrics_short_path}")
    print(f"  - 장기 메트릭: {metrics_long_path}")

    # 리포트 출력
    print("\n" + "=" * 80)
    print("✅ L5 모델 재학습 완료")
    print("=" * 80)
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과 저장 위치: {interim_dir}")
    print("\n[단기 랭킹 요약]")
    for key, value in all_results["short"]["report"].items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")
    print("\n[장기 랭킹 요약]")
    for key, value in all_results["long"]["report"].items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")

    if all_warns:
        print(f"\n⚠️ 경고 메시지 ({len(all_warns)}개):")
        for warn in all_warns[:10]:  # 최대 10개만 출력
            print(f"  - {warn}")
        if len(all_warns) > 10:
            print(f"  ... 외 {len(all_warns) - 10}개")

    print("\n다음 단계:")
    print("  1. Dev/Holdout 구간 성과 재평가")
    print("  2. 과적합 위험도 재확인")
    print("  3. 필요시 Ridge alpha 추가 조정")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
