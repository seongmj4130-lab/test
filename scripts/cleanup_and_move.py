# -*- coding: utf-8 -*-
"""
파일 정리 스크립트: README.md 기준으로 필요한 파일만 유지하고 나머지는 05_backup으로 이동

유지할 것:
- src/ (전체)
- configs/config.yaml (메인 설정 파일만)
- data/raw/ (기초데이터)
- data/external/ (기초데이터)
- data/interim/ (기초데이터 + 산출물만)
  - 기초데이터: universe_k200_membership_monthly.*, ohlcv_daily.*, fundamentals_annual.*, panel_merged_daily.*, dataset_daily.*, cv_folds_*
  - 산출물: ranking_short_daily.*, ranking_long_daily.*, rebalance_scores_from_ranking.*, bt_metrics_*, bt_returns_*, bt_equity_curve_*, bt_positions_*
- scripts/run_pipeline_l0_l7.py (기초데이터 수집용)
- final_*.md (4개)
- README.md

이동할 것:
- backup/
- backups/
- docs/ (일부 제외)
- artifacts/ (일부 제외)
- scripts/ (run_pipeline_l0_l7.py 제외)
- configs/ (config.yaml 제외)
- ui/
- 기타 루트 파일들
"""
import shutil
import logging
from pathlib import Path
from typing import List, Set

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
BACKUP_ROOT = Path(PROJECT_ROOT.parent) / "05_backup"

# 유지할 파일/폴더 (상대 경로)
KEEP_ITEMS: Set[str] = {
    "src",
    "README.md",
    "final_report.md",
    "final_easy_report.md",
    "final_backtest_report.md",
    "final_ranking_report.md",
    "configs/config.yaml",
    "scripts/run_pipeline_l0_l7.py",
    "data/raw",
    "data/external",
}

# 유지할 data/interim 파일 패턴 (기초데이터 + 산출물)
KEEP_INTERIM_PATTERNS = [
    # 기초데이터
    "universe_k200_membership_monthly",
    "ohlcv_daily",
    "fundamentals_annual",
    "panel_merged_daily",
    "dataset_daily",
    "cv_folds_short",
    "cv_folds_long",
    "cv_inner_folds_short",
    "cv_inner_folds_long",
    "market_regime_daily",
    "market_regime",
    "l1_technical_features",
    # 산출물 (Track A)
    "ranking_short_daily",
    "ranking_long_daily",
    # 산출물 (Track B)
    "rebalance_scores_from_ranking",
    "bt_metrics_bt20_short",
    "bt_metrics_bt20_ens",
    "bt_metrics_bt120_long",
    "bt_metrics_bt120_ens",
    "bt_returns_bt20_short",
    "bt_returns_bt20_ens",
    "bt_returns_bt120_long",
    "bt_returns_bt120_ens",
    "bt_equity_curve_bt20_short",
    "bt_equity_curve_bt20_ens",
    "bt_equity_curve_bt120_long",
    "bt_equity_curve_bt120_ens",
    "bt_positions_bt20_short",
    "bt_positions_bt20_ens",
    "bt_positions_bt120_long",
    "bt_positions_bt120_ens",
    "bt_regime_metrics_bt20_short",
    "bt_regime_metrics_bt20_ens",
    "bt_regime_metrics_bt120_long",
    "bt_regime_metrics_bt120_ens",
    "selection_diagnostics_bt20_short",
    "selection_diagnostics_bt20_ens",
    "selection_diagnostics_bt120_long",
    "selection_diagnostics_bt120_ens",
    "bt_returns_diagnostics_bt20_short",
    "bt_returns_diagnostics_bt20_ens",
    "bt_returns_diagnostics_bt120_long",
    "bt_returns_diagnostics_bt120_ens",
    "runtime_profile_bt20_short",
    "runtime_profile_bt20_ens",
    "runtime_profile_bt120_long",
    "runtime_profile_bt120_ens",
]

# 이동할 폴더/파일 (상대 경로)
MOVE_ITEMS: Set[str] = {
    "backup",
    "backups",
    "ui",
    "tests",
}

# 이동할 configs 파일들 (config.yaml 제외)
MOVE_CONFIGS = [
    "config_defence_model_1_backup.yaml",
    "config_midterm_baseline.yaml",
    "config.yaml.backup",
    "feature_groups_long.yaml",
    "feature_groups_short.yaml",
    "feature_groups.yaml",
    "feature_weights_long.yaml",
    "feature_weights_optimized.yaml",
    "feature_weights_regime_detailed.yaml",
    "feature_weights_short.yaml",
    "features_long_v1.yaml",
    "features_short_v1.yaml",
    "final_metrics_runs.yaml",
    "phase4_4_roe_ic_long.yaml",
    "phase4_4_roe_ic_short_latest.yaml",
    "phase4_4_roe_ic_short.yaml",
    "ui_icons.yaml",
]

# 이동할 scripts 파일들 (run_pipeline_l0_l7.py 제외)
MOVE_SCRIPTS_PATTERNS = [
    "analyze_",
    "backup_",
    "check_",
    "compare_",
    "generate_",
    "get_optimal_",
    "grid_search_",
    "monitor_",
    "optimize_",
    "run_parallel_",
    "verify_",
    "README_BACKUP_AND_COMPARE.md",
]


def should_keep_interim_file(file_path: Path) -> bool:
    """data/interim 파일이 유지해야 할 파일인지 확인"""
    file_stem = file_path.stem
    
    # 패턴 매칭
    for pattern in KEEP_INTERIM_PATTERNS:
        if pattern in file_stem:
            return True
    
    # 메타데이터 파일도 유지 (__meta.json)
    if file_path.suffix == ".json" and "__meta" in file_stem:
        # 해당하는 parquet 파일이 유지 대상이면 메타데이터도 유지
        parquet_stem = file_stem.replace("__meta", "")
        for pattern in KEEP_INTERIM_PATTERNS:
            if pattern in parquet_stem:
                return True
    
    return False


def should_move_script(script_path: Path) -> bool:
    """scripts 파일이 이동해야 할 파일인지 확인"""
    if script_path.name == "run_pipeline_l0_l7.py":
        return False
    
    if script_path.name == "__init__.py":
        return False
    
    for pattern in MOVE_SCRIPTS_PATTERNS:
        if script_path.name.startswith(pattern):
            return True
    
    return False


def move_item(src: Path, dst: Path, dry_run: bool = False):
    """파일/폴더 이동"""
    if not src.exists():
        logger.warning(f"  ⚠ 존재하지 않음: {src}")
        return False
    
    if dry_run:
        logger.info(f"  [DRY RUN] 이동: {src} → {dst}")
        return True
    
    try:
        # 부모 디렉토리 생성
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # 이동
        if src.is_dir():
            shutil.move(str(src), str(dst))
        else:
            shutil.move(str(src), str(dst))
        
        logger.info(f"  ✓ 이동 완료: {src.name}")
        return True
    except Exception as e:
        logger.error(f"  ✗ 이동 실패: {src} → {e}")
        return False


def cleanup_interim_dir(dry_run: bool = False):
    """data/interim 디렉토리 정리"""
    interim_dir = PROJECT_ROOT / "data" / "interim"
    if not interim_dir.exists():
        logger.warning(f"data/interim 디렉토리가 없습니다.")
        return
    
    logger.info(f"[data/interim 정리] {interim_dir}")
    
    moved_count = 0
    kept_count = 0
    
    # 모든 파일과 폴더 확인
    for item in interim_dir.iterdir():
        if item.is_file():
            if should_keep_interim_file(item):
                kept_count += 1
                logger.debug(f"  유지: {item.name}")
            else:
                # 이동
                dst = BACKUP_ROOT / "data" / "interim" / item.name
                if move_item(item, dst, dry_run):
                    moved_count += 1
        elif item.is_dir():
            # 폴더는 모두 이동 (기초데이터는 파일로만 존재)
            dst = BACKUP_ROOT / "data" / "interim" / item.name
            if move_item(item, dst, dry_run):
                moved_count += 1
    
    logger.info(f"  유지: {kept_count}개, 이동: {moved_count}개")


def cleanup_configs_dir(dry_run: bool = False):
    """configs 디렉토리 정리"""
    configs_dir = PROJECT_ROOT / "configs"
    if not configs_dir.exists():
        return
    
    logger.info(f"[configs 정리] {configs_dir}")
    
    moved_count = 0
    
    for config_file in configs_dir.iterdir():
        if config_file.is_file() and config_file.name in MOVE_CONFIGS:
            dst = BACKUP_ROOT / "configs" / config_file.name
            if move_item(config_file, dst, dry_run):
                moved_count += 1
    
    logger.info(f"  이동: {moved_count}개")


def cleanup_scripts_dir(dry_run: bool = False):
    """scripts 디렉토리 정리"""
    scripts_dir = PROJECT_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    
    logger.info(f"[scripts 정리] {scripts_dir}")
    
    moved_count = 0
    
    for script_file in scripts_dir.iterdir():
        if script_file.is_file() and should_move_script(script_file):
            dst = BACKUP_ROOT / "scripts" / script_file.name
            if move_item(script_file, dst, dry_run):
                moved_count += 1
    
    logger.info(f"  이동: {moved_count}개")


def cleanup_root_files(dry_run: bool = False):
    """루트 디렉토리 파일 정리"""
    logger.info(f"[루트 파일 정리] {PROJECT_ROOT}")
    
    # final_*.md와 README.md 제외한 모든 .md 파일 이동
    moved_count = 0
    
    for root_file in PROJECT_ROOT.iterdir():
        if root_file.is_file():
            # 유지할 파일 체크
            if root_file.name in ["README.md", "final_report.md", "final_easy_report.md", 
                                 "final_backtest_report.md", "final_ranking_report.md"]:
                continue
            
            # .md 파일이면 이동
            if root_file.suffix == ".md":
                dst = BACKUP_ROOT / root_file.name
                if move_item(root_file, dst, dry_run):
                    moved_count += 1
    
    logger.info(f"  이동: {moved_count}개")


def main(dry_run: bool = True):
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("파일 정리 스크립트 실행")
    logger.info(f"프로젝트 루트: {PROJECT_ROOT}")
    logger.info(f"백업 루트: {BACKUP_ROOT}")
    logger.info(f"모드: {'DRY RUN' if dry_run else '실제 실행'}")
    logger.info("=" * 80)
    
    # 백업 루트 생성
    if not dry_run:
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 1. 루트 폴더 이동
    logger.info("\n[1단계] 루트 폴더 이동")
    for item_name in MOVE_ITEMS:
        src = PROJECT_ROOT / item_name
        dst = BACKUP_ROOT / item_name
        move_item(src, dst, dry_run)
    
    # 2. data/interim 정리
    logger.info("\n[2단계] data/interim 정리")
    cleanup_interim_dir(dry_run)
    
    # 3. configs 정리
    logger.info("\n[3단계] configs 정리")
    cleanup_configs_dir(dry_run)
    
    # 4. scripts 정리
    logger.info("\n[4단계] scripts 정리")
    cleanup_scripts_dir(dry_run)
    
    # 5. 루트 파일 정리
    logger.info("\n[5단계] 루트 파일 정리")
    cleanup_root_files(dry_run)
    
    # 6. docs, artifacts 이동 (전체)
    logger.info("\n[6단계] docs, artifacts 이동")
    for item_name in ["docs", "artifacts"]:
        src = PROJECT_ROOT / item_name
        if src.exists():
            dst = BACKUP_ROOT / item_name
            move_item(src, dst, dry_run)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 파일 정리 완료")
    logger.info("=" * 80)
    
    if dry_run:
        logger.info("\n⚠️  DRY RUN 모드입니다. 실제로 실행하려면 dry_run=False로 설정하세요.")


if __name__ == "__main__":
    import sys
    
    # 명령줄 인자 확인
    dry_run = "--execute" not in sys.argv
    
    if dry_run:
        logger.warning("⚠️  DRY RUN 모드입니다. 실제로 실행하려면 --execute 플래그를 추가하세요.")
    
    main(dry_run=dry_run)

