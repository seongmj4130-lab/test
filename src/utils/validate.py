# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/utils/validate.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


class ValidationError(RuntimeError):
    pass


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]


def _missing_pct(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    miss = (df.isna().mean() * 100.0).sort_values(ascending=False)
    return {k: float(v) for k, v in miss.items()}


def validate_df(
    df: pd.DataFrame,
    *,
    stage: str,
    required_cols: Optional[Sequence[str]] = None,
    key_cols: Sequence[str] = ("date", "ticker"),
    enforce_unique_key: bool = True,
    enforce_time_sorted: bool = True,
    max_missing_pct: Optional[float] = None,  # e.g. 80.0 (컬럼 결측 80% 초과는 에러)
    optional_cols: Optional[
        Sequence[str]
    ] = None,  # [Stage13] 옵션 컬럼 (missing pct 체크에서 제외)
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    if df is None:
        errors.append("df is None")
        return ValidationResult(False, errors, warnings, stats)

    if not isinstance(df, pd.DataFrame):
        errors.append(f"df is not a DataFrame: {type(df)}")
        return ValidationResult(False, errors, warnings, stats)

    # 1) required columns
    if required_cols:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

    # 2) key uniqueness
    if enforce_unique_key and all(c in df.columns for c in key_cols):
        dup = int(df.duplicated(subset=list(key_cols)).sum())
        stats["key_dup_count"] = dup
        if dup > 0:
            errors.append(f"Duplicate keys found for {key_cols}: {dup}")

    # 3) time sorted per ticker
    if enforce_time_sorted and ("date" in df.columns) and ("ticker" in df.columns):
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.isna().all():
            errors.append("date column cannot be parsed to datetime (all NaT).")
        else:
            tmp = df[["ticker"]].copy()
            tmp["_date"] = d
            tmp = tmp.sort_values(["ticker", "_date"], kind="mergesort")

            # ticker별 단조 증가 여부
            mono = tmp.groupby("ticker")["_date"].apply(
                lambda x: x.is_monotonic_increasing
            )
            bad = mono[~mono].index.astype(str).tolist()
            if bad:
                warnings.append(
                    f"Non-monotonic dates for some tickers (sample): {bad[:10]}"
                )

    # 4) missing
    miss = _missing_pct(df)
    if miss:
        top5 = sorted(miss.items(), key=lambda x: x[1], reverse=True)[:5]
        stats["missing_top5_pct"] = {k: round(v, 4) for k, v in top5}

    if max_missing_pct is not None and miss:
        # [Stage13] optional_cols는 missing pct 체크에서 제외
        optional_set = set(optional_cols) if optional_cols else set()
        too_missing = [
            c
            for c, p in miss.items()
            if p > float(max_missing_pct) and c not in optional_set
        ]
        if too_missing:
            errors.append(
                f"Columns exceed max_missing_pct({max_missing_pct}%): {too_missing[:20]}"
            )

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, stats=stats)


def raise_if_invalid(result: ValidationResult, *, stage: str) -> None:
    if result.ok:
        return
    msg = f"[{stage}] Validation failed:\n- " + "\n- ".join(result.errors)
    raise ValidationError(msg)
