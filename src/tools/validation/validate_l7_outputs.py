# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/validate_l7_outputs.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact


def _load_meta(interim: Path, name: str) -> dict:
    mp = interim / f"{name}__meta.json"
    if not mp.exists():
        raise FileNotFoundError(f"[FAIL] meta not found: {mp}")
    return json.loads(mp.read_text(encoding="utf-8"))


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[FAIL] {name} missing columns: {missing}. got={list(df.columns)}"
        )


def main():
    print("=== L7 Validation Runner ===")
    ROOT = Path(__file__).resolve().parents[2]
    CFG = ROOT / "configs" / "config.yaml"

    print("ROOT  :", ROOT)
    print("CFG   :", CFG)

    cfg = load_config(str(CFG))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    names = ["bt_positions", "bt_returns", "bt_equity_curve", "bt_metrics"]
    for n in names:
        if not artifact_exists(interim / n):
            raise SystemExit(f"[FAIL] missing artifact: {n}")

    pos = load_artifact(interim / "bt_positions")
    ret = load_artifact(interim / "bt_returns")
    eq = load_artifact(interim / "bt_equity_curve")
    met = load_artifact(interim / "bt_metrics")

    # meta check
    mpos = _load_meta(interim, "bt_positions")
    mret = _load_meta(interim, "bt_returns")
    meq = _load_meta(interim, "bt_equity_curve")
    mmet = _load_meta(interim, "bt_metrics")

    print("\n=== Meta check ===")
    print("- bt_positions stage:", mpos.get("stage"))
    print("- bt_returns stage  :", mret.get("stage"))
    print("- bt_equity_curve stage:", meq.get("stage"))
    print("- bt_metrics stage  :", mmet.get("stage"))

    # schema
    print("\n=== Schema check ===")
    _require_cols(
        pos, ["date", "phase", "ticker", "weight", "score_used"], "bt_positions"
    )
    _require_cols(
        ret,
        [
            "date",
            "phase",
            "port_ret_gross",
            "port_ret_net",
            "turnover_oneway",
            "cost",
            "n_tickers",
        ],
        "bt_returns",
    )
    _require_cols(
        eq,
        ["date", "phase", "equity_gross", "equity_net", "dd_gross", "dd_net"],
        "bt_equity_curve",
    )
    _require_cols(
        met,
        [
            "phase",
            "top_k",
            "holding_days",
            "cost_bps",
            "gross_sharpe",
            "net_sharpe",
            "gross_mdd",
            "net_mdd",
        ],
        "bt_metrics",
    )

    # duplicates
    print("\n=== Duplicate checks ===")
    d1 = pos.duplicated(["date", "phase", "ticker"]).sum()
    d2 = ret.duplicated(["date", "phase"]).sum()
    d3 = eq.duplicated(["date", "phase"]).sum()
    if d1 or d2 or d3:
        raise SystemExit(f"[FAIL] duplicates found: pos={d1}, ret={d2}, eq={d3}")

    # weight sanity (per date/phase sum ~= 1)
    print("\n=== Weight sanity ===")
    ws = pos.groupby(["phase", "date"])["weight"].sum().reset_index(name="w_sum")
    bad = ws[(ws["w_sum"] < 0.999) | (ws["w_sum"] > 1.001)]
    if len(bad):
        print(bad.head(10))
        raise SystemExit("[FAIL] weight sum not close to 1 for some rebalance dates.")

    # equity consistency
    print("\n=== Equity consistency ===")
    # eq가 ret로부터 만들어졌는지 재검증(phase별 마지막 equity 비교)
    for phase, g in ret.groupby("phase", sort=True):
        g = g.sort_values("date")
        eq_g = float((1.0 + g["port_ret_gross"].astype(float)).cumprod().iloc[-1])
        eq_n = float((1.0 + g["port_ret_net"].astype(float)).cumprod().iloc[-1])

        e = eq[eq["phase"] == phase].sort_values("date")
        if len(e) == 0:
            raise SystemExit(f"[FAIL] equity missing for phase={phase}")

        last_g = float(e["equity_gross"].iloc[-1])
        last_n = float(e["equity_net"].iloc[-1])

        if abs(eq_g - last_g) > 1e-9 or abs(eq_n - last_n) > 1e-9:
            raise SystemExit(
                f"[FAIL] equity mismatch for phase={phase}: recomputed({eq_g},{eq_n}) vs saved({last_g},{last_n})"
            )

    print("\n✅ L7 VALIDATION COMPLETE: All critical checks passed.")
    print(
        "➡️ Next: reporting / plots / final summary tables (and optional L7 meta-quality extension)."
    )


if __name__ == "__main__":
    main()
