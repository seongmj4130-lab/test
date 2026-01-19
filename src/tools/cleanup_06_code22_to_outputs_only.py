# -*- coding: utf-8 -*-
from __future__ import annotations

"""
[개선안 44번] 06_code22를 "최종 산출물 저장소"로 정리하는 도구

정책:
- 06_code22 루트에 기존에 있던 실행 워크스페이스( src/, data/, configs/ 등 )를 삭제하지 않고,
  `_archive_pre_outputs_<timestamp>/`로 안전하게 이동한다.
- 결과적으로 06_code22 루트에는 다음만 남기는 것을 목표로 한다:
  - README.md
  - final_outputs/
  - _archive_pre_outputs_*/

Example:
  cd C:/Users/seong/OneDrive/Desktop/bootcamp/03_code
  python -m src.tools.cleanup_06_code22_to_outputs_only --target ..\06_code22
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

KEEP_NAMES = {"README.md", "final_outputs"}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cleanup_06_code22_to_outputs_only(target_root: Path) -> List[Path]:
    """
    [개선안 44번] 06_code22 루트를 정리한다(archive 이동).

    Args:
        target_root: 06_code22 폴더 경로

    Returns:
        moved: 이동된 항목 경로 리스트(archive 내 경로)
    """
    target_root = Path(target_root)
    if not target_root.exists():
        raise FileNotFoundError(f"target_root not found: {target_root}")

    archive_dir = target_root / f"_archive_pre_outputs_{_now_tag()}"
    moved: List[Path] = []

    for p in target_root.iterdir():
        name = p.name
        if name in KEEP_NAMES:
            continue
        if name.startswith("_archive_pre_outputs_"):
            continue
        # final_outputs 내부는 건드리지 않음
        dst = archive_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        moved.append(dst)

    # archive가 비어있으면 만들지 않음
    if archive_dir.exists():
        # README가 없으면 최소 README 생성(가이드 역할)
        readme = target_root / "README.md"
        if not readme.exists():
            readme.write_text(
                "## 06_code22 (최종 산출물 저장소)\n\n"
                "- 최종 산출물: `final_outputs/LATEST/`\n"
                "- 이전 워크스페이스는 `_archive_pre_outputs_*/`로 이동됨\n",
                encoding="utf-8",
            )

    return moved


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="06_code22를 최종 산출물 저장소로 정리(archive 이동)")
    p.add_argument("--target", dest="target", default=r"..\06_code22")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    moved = cleanup_06_code22_to_outputs_only(Path(args.target))
    print(f"moved={len(moved)}")
    for m in moved[:20]:
        print("-", m)
    if len(moved) > 20:
        print(f"... (+{len(moved)-20} more)")


