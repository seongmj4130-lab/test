# -*- coding: utf-8 -*-
# [개선안 36번] Track A: Holdout 하루 찍어서 Top10 랭킹 + 팩터셋(그룹) 기여도 Top3 출력
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.tracks.track_a.ranking_service import inspect_holdout_day_rankings
from src.utils.config import load_config


def _maybe_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="예: 2024-06-03 (Holdout 구간 날짜)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--horizon", choices=["short", "long", "both"], default="both")
    p.add_argument("--save_md", action="store_true", help="artifacts/reports/ 에 Markdown 저장")
    args = p.parse_args()

    cfg = load_config("configs/config.yaml")
    date = pd.Timestamp(args.date)

    result = inspect_holdout_day_rankings(
        as_of=str(date.date()),
        topk=args.topk,
        horizon=args.horizon,
        config_path="configs/config.yaml",
    )
    results = {k: v for k, v in [("short", result.get("short")), ("long", result.get("long"))] if v is not None}

    # 콘솔 출력
    for h, df in results.items():
        print("\n" + "=" * 100)
        print(f"[Track A] {h.upper()} | date={date.date()} | Top{args.topk} + 팩터셋 Top3")
        print("=" * 100)
        print(df.to_string(index=False))

    # Markdown 저장
    if args.save_md:
        out_dir = Path(cfg["paths"]["artifacts_reports"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"track_a_holdout_day_{date.date()}_{args.horizon}_top{args.topk}.md"

        md = []
        md.append(f"# [개선안 36번] Track A Holdout Day Inspect ({date.date()})")
        md.append("")
        md.append(f"- date: {date.date()}")
        md.append(f"- horizon: {args.horizon}")
        md.append(f"- topk: {args.topk}")
        md.append("")
        for h, df in results.items():
            md.append(f"## {h.upper()} Top{args.topk}")
            md.append("")
            md.append(_maybe_to_markdown(df))
            md.append("")

        out_path.write_text("\n".join(md), encoding="utf-8")
        print(f"\n[저장 완료] {out_path}")


if __name__ == "__main__":
    main()
