# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/maintenance/rebuild_bt_rolling_sharpe.py
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.meta import build_meta, save_meta


def _root() -> Path:
    # .../03_code/src/stages/rebuild_bt_rolling_sharpe.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]


def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"


def compute_bt_rolling_sharpe(
    bt_returns: pd.DataFrame,
    *,
    holding_days: int,
    window_rebalances: int,
    return_col: str,
) -> pd.DataFrame:
    need = {"date", "phase", return_col}
    missing = sorted(list(need - set(bt_returns.columns)))
    if missing:
        raise SystemExit(f"[FAIL] bt_returns missing columns: {missing}")

    df = bt_returns[["date", "phase", return_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise SystemExit("[FAIL] bt_returns has invalid 'date' (NaT)")
    df["phase"] = df["phase"].astype(str)
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    periods_per_year = 252.0 / float(holding_days)
    ann_factor = np.sqrt(periods_per_year)

    out_rows = []
    for phase, g in df.groupby("phase", sort=False):
        s = g.sort_values("date").reset_index(drop=True)

        r = s[return_col].astype(float)

        roll_n = r.rolling(window_rebalances, min_periods=1).count()
        roll_mean = r.rolling(window_rebalances, min_periods=1).mean()
        roll_std = r.rolling(window_rebalances, min_periods=2).std(ddof=1)

        roll_mean = roll_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        roll_std = roll_std.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        roll_vol_ann = roll_std * ann_factor
        mean_np = roll_mean.to_numpy(dtype=float)
        std_np = roll_std.to_numpy(dtype=float)

        ratio = np.zeros_like(mean_np, dtype=float)
        np.divide(mean_np, std_np, out=ratio, where=(std_np > 0.0))

        roll_sharpe = ratio * ann_factor

        out = pd.DataFrame(
            {
                "phase": phase,
                "date": s["date"],
                "net_rolling_n": roll_n.astype(int),
                "net_rolling_mean": roll_mean.astype(float),
                "net_rolling_vol_ann": roll_vol_ann.astype(float),
                "net_rolling_sharpe": pd.Series(roll_sharpe, index=s.index).astype(
                    float
                ),
                "net_return_col_used": return_col,
            }
        )
        out_rows.append(out)

    res = pd.concat(out_rows, ignore_index=True)

    keys = ["date", "phase", "net_return_col_used"]
    if res.duplicated(keys).any():
        dup = res.loc[res.duplicated(keys, keep=False), keys].head(20)
        raise SystemExit(
            f"[FAIL] rolling sharpe rebuild has duplicates on {keys}. sample:\n{dup}"
        )

    # finite 강제
    for c in ["net_rolling_mean", "net_rolling_vol_ann", "net_rolling_sharpe"]:
        res[c] = (
            pd.to_numeric(res[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return res


def main():
    print("=== REBUILD bt_rolling_sharpe from bt_returns ===")
    root = _root()
    cfg_path = _cfg_path(root)

    print("ROOT :", root)
    print("CFG  :", cfg_path)

    if not cfg_path.exists():
        raise SystemExit(f"[FAIL] config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    base_returns = interim / "bt_returns"
    if not artifact_exists(base_returns):
        raise SystemExit(f"[FAIL] artifact missing: {base_returns}")

    bt_returns = load_artifact(base_returns)
    if not isinstance(bt_returns, pd.DataFrame) or bt_returns.shape[0] == 0:
        raise SystemExit(
            f"[FAIL] bt_returns invalid: shape={getattr(bt_returns,'shape',None)}"
        )

    # 설정값: config 우선, 없으면 프로젝트 기본값(월 리밸=20, 12개월=12스텝)
    l7 = cfg.get("l7", {}) or {}
    l7d = cfg.get("l7d", {}) or {}
    holding_days = int(l7.get("holding_days", 20))
    window_rebalances = int(l7d.get("rolling_window_rebalances", 12))

    # bt_returns 컬럼은 L7C에서 이미 사용되므로 여기서는 고정
    return_col = "net_return"
    if return_col not in bt_returns.columns:
        raise SystemExit(
            f"[FAIL] bt_returns missing '{return_col}'. cols={bt_returns.columns.tolist()}"
        )

    rebuilt = compute_bt_rolling_sharpe(
        bt_returns,
        holding_days=holding_days,
        window_rebalances=window_rebalances,
        return_col=return_col,
    )

    out_base = interim / "bt_rolling_sharpe"
    save_formats = cfg.get("run", {}).get("save_formats", ["parquet", "csv"])
    save_artifact(rebuilt, out_base, force=True, formats=save_formats)

    meta = build_meta(
        stage="L7D:bt_rolling_sharpe",
        run_id="rebuild_bt_rolling_sharpe",
        df=rebuilt,
        out_base_path=out_base,
        warnings=[
            f"rebuilt from bt_returns with holding_days={holding_days}, window_rebalances={window_rebalances}"
        ],
        inputs={"source": "bt_returns"},
        repo_dir=get_path(cfg, "base_dir"),
        quality={
            "stability": {
                "holding_days": holding_days,
                "window_rebalances": window_rebalances,
                "net_return_col_used": return_col,
                "rows": int(rebuilt.shape[0]),
            }
        },
    )
    save_meta(out_base, meta, force=True)

    print("✅ REBUILD COMPLETE: bt_rolling_sharpe overwritten (unique keys).")


if __name__ == "__main__":
    main()
