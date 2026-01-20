from __future__ import annotations

"""
[개선안 41번] 투트랙 실행 + 06_code22 최종 산출물 Export 원클릭 엔트리포인트

실행 순서:
1) 공통 데이터(L0~L4) 준비 (캐시 우선)
2) Track A (랭킹 생성)
3) Track B (4전략 백테스트)
4) Track B 4전략 요약표 생성(artifacts/reports)
5) 06_code22/final_outputs/LATEST로 "최종 산출물만" Export + manifest/summary 저장

Example (PowerShell):
  cd C:/Users/seong/OneDrive/Desktop/bootcamp/03_code
  python -m src.tools.run_two_track_and_export --export-dest ..\06_code22
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_collection.pipeline import DataCollectionPipeline
from src.pipeline.track_a_pipeline import run_track_a_pipeline
from src.pipeline.track_b_pipeline import run_track_b_pipeline
from src.tools.export_final_outputs import export_final_outputs
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact

logger = logging.getLogger(__name__)


STRATEGIES = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]


def _validate_track_b_outputs(
    strategy: str, bt_returns: pd.DataFrame, cfg_l7: dict
) -> list[str]:
    """
    [개선안 41번] 5개 핵심 작업이 실제 산출물에 반영됐는지(최소) 검증

    검증 항목:
    - 거래비용: turnover_cost/turnover_oneway 컬럼 존재
    - BT120 오버래핑 트랜치: overlapping_tranches_enabled=True면 tranche_* 컬럼 존재
    - 리밸런싱 주기 일원화: (간접) n_rebalances/리밸런스 날짜 기반으로 bt_returns가 생성됐는지
    - Regime 기반: regime가 켜져 있으면 mode/regime 관련 컬럼 또는 bt_regime_metrics 생성(별도 파일)
    - Alpha Quality: ic/rank_ic/long_short_alpha 컬럼 존재
    """
    # NOTE: bt_returns는 [Stage13] core/diagnostics로 분리될 수 있다.
    # - 비용/턴오버는 core(bt_returns_{strategy})에 남는 것이 정상
    # - tranche/regime/alpha quality 일부는 diagnostics/metrics 파일에 존재할 수 있다.
    warns: list[str] = []

    cols = set(bt_returns.columns) if isinstance(bt_returns, pd.DataFrame) else set()
    need_cost_cols = {"turnover_oneway", "turnover_cost", "total_cost"}
    missing_cost = [c for c in need_cost_cols if c not in cols]
    if missing_cost:
        warns.append(
            f"[{strategy}] 거래비용(턴오버 기반) 핵심 컬럼 누락(bt_returns): {missing_cost}"
        )

    return warns


def _validate_track_b_artifacts(
    *,
    strategy: str,
    interim_dir: Path,
    cfg_l7: dict,
    bt_metrics: pd.DataFrame,
) -> list[str]:
    """
    [개선안 41번] 파일/지표 관점의 기능 검증(5개 작업)
    """
    warns: list[str] = []

    # 1) Alpha Quality는 bt_metrics에 집계되어야 한다.
    met_cols = (
        set(bt_metrics.columns) if isinstance(bt_metrics, pd.DataFrame) else set()
    )
    need_alpha = {"ic", "rank_ic", "icir", "long_short_alpha"}
    miss_alpha = [c for c in need_alpha if c not in met_cols]
    if miss_alpha:
        warns.append(
            f"[{strategy}] Alpha Quality 집계 컬럼 누락(bt_metrics): {miss_alpha}"
        )

    # 2) Regime Robustness는 bt_regime_metrics_{strategy} 아티팩트로 저장되는 것이 정상
    regime_enabled = isinstance(cfg_l7.get("regime", None), dict) and bool(
        cfg_l7.get("regime", {}).get("enabled", False)
    )
    if regime_enabled:
        reg_base = interim_dir / f"bt_regime_metrics_{strategy}"
        if not artifact_exists(reg_base):
            warns.append(
                f"[{strategy}] regime 활성인데 bt_regime_metrics_{strategy} 아티팩트가 없습니다."
            )

    # 3) BT120 오버래핑 트랜치: tranche 컬럼은 diagnostics에 있을 수 있음
    if bool(cfg_l7.get("overlapping_tranches_enabled", False)):
        diag_base = interim_dir / f"bt_returns_diagnostics_{strategy}"
        if not artifact_exists(diag_base):
            # 하위호환: core bt_returns에 남는 버전도 있음. (둘 다 없으면 경고)
            warns.append(
                f"[{strategy}] 오버래핑 트랜치 활성인데 bt_returns_diagnostics_{strategy} 아티팩트가 없습니다."
            )
        else:
            try:
                diag = load_artifact(diag_base)
                diag_cols = set(diag.columns)
                need_tranche = {
                    "tranche_active",
                    "tranche_holding_days",
                    "tranche_max_active",
                }
                miss_tr = [c for c in need_tranche if c not in diag_cols]
                if miss_tr:
                    warns.append(
                        f"[{strategy}] tranche 컬럼 누락(bt_returns_diagnostics): {miss_tr}"
                    )
            except Exception as e:
                warns.append(f"[{strategy}] bt_returns_diagnostics 로드/검증 실패: {e}")

    return warns


def run(
    *,
    config_path: str = "configs/config.yaml",
    export_dest: str | None = None,
    force_shared: bool = False,
    force_track_a: bool = True,
    force_track_b: bool = True,
    export_mode: str = "latest",
) -> dict:
    """
    [개선안 41번] 투트랙 실행 + Export 실행

    Args:
        config_path: 설정 파일 경로
        export_dest: Export 목적지 루트 (기본: ../06_code22)
        force_shared: 공통 데이터(L0~L4) 캐시 무시 여부
        force_track_a: Track A 재생성 여부(권장: True)
        force_track_b: Track B 재생성 여부(권장: True)
        export_mode: latest | runs

    Returns:
        dict: 실행 결과(경로/경고 포함)
    """
    results: dict[str, object] = {"warnings": []}

    # 공통 경로 계산(검증용)
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))

    # 1) Shared
    logger.info("[Two-Track] 1) 공통 데이터(L0~L4)")
    shared = DataCollectionPipeline(config_path=config_path, force_rebuild=force_shared)
    shared.run_all()

    # 2) Track A
    logger.info("[Two-Track] 2) Track A (랭킹)")
    ra = run_track_a_pipeline(
        config_path=config_path,
        force_rebuild=force_track_a,
        run_ui_payload=False,  # [개선안 41번] 외부 API 의존(L11) 기본 OFF
    )
    results["track_a_artifacts_path"] = ra.get("artifacts_path")

    # 3) Track B (4 strategies)
    logger.info("[Two-Track] 3) Track B (4전략)")
    tb_paths = {}
    for s in STRATEGIES:
        logger.info(f"  - strategy={s}")
        out = run_track_b_pipeline(
            config_path=config_path, strategy=s, force_rebuild=force_track_b
        )
        tb_paths[s] = out.get("artifacts_path")

        cfg_l7 = out.get("config", {}) or {}

        # [개선안 41번] 산출물 기반 검증(1) core(bt_returns)
        warns = _validate_track_b_outputs(
            strategy=s, bt_returns=out.get("bt_returns"), cfg_l7=cfg_l7
        )
        results["warnings"].extend(warns)

        # [개선안 41번] 산출물 기반 검증(2) 아티팩트/지표(bt_metrics/diagnostics/regime)
        results["warnings"].extend(
            _validate_track_b_artifacts(
                strategy=s,
                interim_dir=interim_dir,
                cfg_l7=cfg_l7,
                bt_metrics=out.get("bt_metrics"),
            )
        )

    results["track_b_artifacts_path"] = tb_paths

    # 4) Track B 요약표 생성(이미 존재하는 스크립트 재사용)
    logger.info("[Two-Track] 4) Track B 4전략 요약표 생성")
    try:
        from scripts.generate_trackb_4strategy_final_summary import (
            main as generate_summary,
        )

        generate_summary()
    except Exception as e:
        results["warnings"].append(
            f"[summary] track_b_4strategy_final_summary 생성 실패: {e}"
        )

    # 5) Export
    logger.info("[Two-Track] 5) 06_code22 최종 산출물 Export")
    export_res = export_final_outputs(
        config_path=config_path,
        dest_root=export_dest,
        mode=export_mode,
        clean_latest=True,
    )

    results["export_out_dir"] = str(export_res.out_dir)
    results["export_manifest"] = str(export_res.manifest_path)
    results["export_summary"] = str(export_res.summary_path)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-Track 실행 + 06_code22 최종 산출물 Export"
    )
    p.add_argument("--config", dest="config_path", default="configs/config.yaml")
    p.add_argument(
        "--export-dest", dest="export_dest", default=None, help="기본: ../06_code22"
    )
    p.add_argument(
        "--force-shared", action="store_true", help="L0~L4 캐시 무시(시간 오래)"
    )
    p.add_argument(
        "--no-force-track-a",
        action="store_true",
        help="Track A 캐시 사용(재생성 안 함)",
    )
    p.add_argument(
        "--no-force-track-b",
        action="store_true",
        help="Track B 캐시 사용(재생성 안 함)",
    )
    p.add_argument("--export-mode", default="latest", choices=["latest", "runs"])
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_arg_parser().parse_args()

    res = run(
        config_path=args.config_path,
        export_dest=args.export_dest,
        force_shared=bool(args.force_shared),
        force_track_a=not bool(args.no_force_track_a),
        force_track_b=not bool(args.no_force_track_b),
        export_mode=str(args.export_mode),
    )

    print("\n=== DONE ===")
    print(f"- export_out_dir : {res.get('export_out_dir')}")
    print(f"- manifest       : {res.get('export_manifest')}")
    print(f"- summary        : {res.get('export_summary')}")
    if res.get("warnings"):
        print("\n=== WARNINGS ===")
        for w in res["warnings"]:
            print("-", w)
