from __future__ import annotations

import json
import sys
from pathlib import Path

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact

################################################################################
# START OF FILE: __init__.py
################################################################################

# package marker


# END OF FILE: __init__.py


################################################################################
# START OF FILE: audit_l0_l7.py
################################################################################


ARTIFACTS = [
    "universe_k200_membership_monthly",
    "ohlcv_daily",
    "fundamentals_annual",
    "panel_merged_daily",
    "dataset_daily",
    "cv_folds_short",
    "cv_folds_long",
    "pred_short_oos",
    "pred_long_oos",
    "model_metrics",
    "rebalance_scores",
    "rebalance_scores_summary",
    "bt_positions",
    "bt_returns",
    "bt_equity_curve",
    "bt_metrics",
]


def _root() -> Path:
    # .../03_code/src/stages/audit_l0_l7.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]


def _must(cond: bool, msg: str):
    if not cond:
        raise SystemExit(f"[FAIL] {msg}")


def _meta_path(interim: Path, name: str) -> Path:
    return interim / f"{name}__meta.json"


def _meta_exists(interim: Path, name: str) -> bool:
    return _meta_path(interim, name).exists()


def _load_meta(interim: Path, name: str) -> dict:
    mp = _meta_path(interim, name)
    _must(mp.exists(), f"missing meta json: {mp}")
    with mp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load(interim: Path, name: str):
    base = interim / name
    _must(artifact_exists(base), f"artifact missing: {base}")
    return load_artifact(base)


def main():
    print("=== L0~L7 AUDIT RUNNER ===")
    root = _root()
    cfg_path = root / "configs" / "config.yaml"
    _must(cfg_path.exists(), f"config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")

    print("ROOT  :", root)
    print("CFG   :", cfg_path)
    print("INTERIM:", interim)

    # 1) existence
    for name in ARTIFACTS:
        base = interim / name
        _must(artifact_exists(base), f"missing artifact: {name}")
        _must(_meta_exists(interim, name), f"missing meta: {name}__meta.json")

    print("[PASS] all artifacts/meta exist")

    # 2) minimal schema checks
    uni = _load(interim, "universe_k200_membership_monthly")
    _must(set(["date", "ticker"]).issubset(uni.columns), "L0 schema: need date,ticker")

    ohlcv = _load(interim, "ohlcv_daily")
    _must(
        set(["date", "ticker", "close"]).issubset(ohlcv.columns),
        "L1 schema: need date,ticker,close",
    )
    _must(ohlcv.duplicated(["date", "ticker"]).sum() == 0, "L1 duplicate (date,ticker)")

    fa = _load(interim, "fundamentals_annual")
    _must(set(["date", "ticker"]).issubset(fa.columns), "L2 schema: need date,ticker")

    panel = _load(interim, "panel_merged_daily")
    _must(
        set(["date", "ticker"]).issubset(panel.columns), "L3 schema: need date,ticker"
    )
    _must(panel.duplicated(["date", "ticker"]).sum() == 0, "L3 duplicate (date,ticker)")

    ds = _load(interim, "dataset_daily")
    _must(set(["date", "ticker"]).issubset(ds.columns), "L4 schema: need date,ticker")
    _must(
        ("ret_fwd_20d" in ds.columns) and ("ret_fwd_120d" in ds.columns),
        "L4 targets missing",
    )

    cv_s = _load(interim, "cv_folds_short")
    cv_l = _load(interim, "cv_folds_long")
    _must("fold_id" in cv_s.columns, "cv_folds_short missing fold_id")
    _must("fold_id" in cv_l.columns, "cv_folds_long missing fold_id")
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        _must(c in cv_s.columns, f"cv_folds_short missing {c}")
        _must(c in cv_l.columns, f"cv_folds_long missing {c}")

    ps = _load(interim, "pred_short_oos")
    pl = _load(interim, "pred_long_oos")
    need_pred_cols = {
        "date",
        "ticker",
        "y_true",
        "y_pred",
        "fold_id",
        "phase",
        "horizon",
    }
    _must(
        need_pred_cols.issubset(ps.columns),
        f"pred_short_oos schema missing: {sorted(need_pred_cols - set(ps.columns))}",
    )
    _must(
        need_pred_cols.issubset(pl.columns),
        f"pred_long_oos schema missing: {sorted(need_pred_cols - set(pl.columns))}",
    )
    _must(
        ps[["y_true", "y_pred"]].isna().any(axis=1).mean() == 0.0,
        "pred_short_oos has NA in y_true/y_pred",
    )
    _must(
        pl[["y_true", "y_pred"]].isna().any(axis=1).mean() == 0.0,
        "pred_long_oos has NA in y_true/y_pred",
    )

    mm = _load(interim, "model_metrics")
    _must(
        set(["phase", "horizon", "fold_id", "rmse"]).issubset(mm.columns),
        "model_metrics schema insufficient",
    )

    rs = _load(interim, "rebalance_scores")
    rss = _load(interim, "rebalance_scores_summary")
    _must(
        set(["date", "ticker", "phase"]).issubset(rs.columns),
        "rebalance_scores schema: need date,ticker,phase",
    )
    _must(
        rs.duplicated(["date", "ticker", "phase"]).sum() == 0,
        "rebalance_scores duplicate (date,ticker,phase)",
    )
    _must(
        set(["date", "phase", "coverage_ticker_pct"]).issubset(rss.columns),
        "rebalance_scores_summary schema insufficient",
    )

    bt_m = _load(interim, "bt_metrics")
    _must(
        set(["phase", "net_sharpe", "net_mdd", "net_cagr"]).issubset(bt_m.columns),
        "bt_metrics missing key cols",
    )

    print("[PASS] core schema/duplicate/NA checks")

    # 3) meta quality keys existence
    m3 = _load_meta(interim, "panel_merged_daily")
    _must(
        "fundamental" in (m3.get("quality") or {}),
        "L3 meta missing quality.fundamental",
    )

    m4 = _load_meta(interim, "dataset_daily")
    _must(
        "walkforward" in (m4.get("quality") or {}),
        "L4 meta missing quality.walkforward",
    )

    m5s = _load_meta(interim, "pred_short_oos")
    m5l = _load_meta(interim, "pred_long_oos")
    _must(
        "model_oos" in (m5s.get("quality") or {}),
        "L5 meta missing quality.model_oos (short)",
    )
    _must(
        "model_oos" in (m5l.get("quality") or {}),
        "L5 meta missing quality.model_oos (long)",
    )

    m6 = _load_meta(interim, "rebalance_scores")
    _must("scoring" in (m6.get("quality") or {}), "L6 meta missing quality.scoring")

    print("[PASS] meta quality keys check")
    print("✅ AUDIT COMPLETE: proceed to B/C/D extensions and final reporting.")


if __name__ == "__main__":
    main()


# END OF FILE: audit_l0_l7.py


################################################################################
# START OF FILE: combine_snapshot_outputs_to_one.py
################################################################################

# src/stages/combine_snapshot_outputs_to_one.py
import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import save_artifact


def _project_root() -> Path:
    # .../03_code/src/stages/xxx.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]


def _load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_artifact(snapshot_dir: Path, name: str) -> pd.DataFrame:
    p_parq = snapshot_dir / f"{name}.parquet"
    p_csv = snapshot_dir / f"{name}.csv"

    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv, low_memory=False)
    raise FileNotFoundError(
        f"Missing data for artifact='{name}' in snapshot_dir={snapshot_dir}"
    )


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return out


def _get_snapshots_dir(cfg: dict, root: Path, snapshots_dir_arg: str = "") -> Path:
    """
    우선순위:
    1) --snapshots-dir 인자
    2) config의 paths.data_snapshots
    3) config의 paths.base_dir / data / snapshots
    4) project root / data / snapshots
    """
    if snapshots_dir_arg and snapshots_dir_arg.strip():
        return Path(snapshots_dir_arg).expanduser().resolve()

    # 2) config에 정의된 경우
    try:
        return get_path(cfg, "data_snapshots")
    except KeyError:
        pass

    # 3) base_dir 기반 폴백
    try:
        base_dir = get_path(cfg, "base_dir")
    except KeyError:
        base_dir = root

    cand = base_dir / "data" / "snapshots"
    return cand


def main():
    parser = argparse.ArgumentParser(
        description="Combine snapshot outputs into ONE table (parquet + csv)."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="snapshot tag folder name (e.g., baseline_after_L7BCD)",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="",
        help="base output name (no extension). default=combined__<tag>",
    )
    parser.add_argument(
        "--out-dir", type=str, default="", help="optional override output directory"
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="",
        help="optional override snapshots base dir",
    )
    parser.add_argument(
        "--include-meta-cols",
        action="store_true",
        help="attach meta.stage/meta.run_id as columns",
    )
    args = parser.parse_args()

    root = _project_root()
    cfg_path = (root / args.config).resolve()
    cfg = load_config(str(cfg_path))

    snapshots_dir = _get_snapshots_dir(cfg, root, args.snapshots_dir)
    snapshot_dir = snapshots_dir / args.tag
    if not snapshot_dir.exists():
        # 마지막 폴백: ROOT/data/snapshots/<tag>
        alt = root / "data" / "snapshots" / args.tag
        if alt.exists():
            snapshot_dir = alt
        else:
            raise FileNotFoundError(
                f"Snapshot folder not found: {snapshot_dir} (also tried: {alt})"
            )

    out_name = args.out_name.strip() or f"combined__{args.tag}"
    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = snapshot_dir  # snapshot 폴더 안에 저장(가장 안전)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== COMBINE SNAPSHOT OUTPUTS ===")
    print(f"ROOT       : {root}")
    print(f"CFG        : {cfg_path}")
    print(f"SNAPSHOT   : {snapshot_dir}")
    print(f"OUT_DIR    : {out_dir}")
    print(f"OUT_NAME   : {out_name}")
    print(f"include_meta_cols: {bool(args.include_meta_cols)}")

    meta_files = sorted(snapshot_dir.glob("*__meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta files found in snapshot: {snapshot_dir}")

    dfs: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []

    for mp in meta_files:
        name = mp.name.replace("__meta.json", "")
        meta = _load_meta(mp)

        df = _read_artifact(snapshot_dir, name)
        df = _normalize_df(df)

        df.insert(0, "__artifact", name)
        df.insert(1, "__snapshot_tag", args.tag)

        if args.include_meta_cols:
            df.insert(2, "__meta_stage", meta.get("stage", None))
            df.insert(3, "__meta_run_id", meta.get("run_id", None))

        dfs.append(df)

        manifest_rows.append(
            {
                "artifact": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "has_date": bool("date" in df.columns),
                "meta_stage": meta.get("stage", None),
                "meta_run_id": meta.get("run_id", None),
            }
        )

        print(f"- loaded: {name:30s} shape={df.shape}")

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    manifest = (
        pd.DataFrame(manifest_rows).sort_values(["artifact"]).reset_index(drop=True)
    )

    out_base = out_dir / out_name
    save_artifact(combined, out_base, force=True, formats=["parquet", "csv"])

    man_base = out_dir / f"{out_name}__manifest"
    save_artifact(manifest, man_base, force=True, formats=["parquet", "csv"])

    print("\n✅ DONE")
    print(f"- combined saved: {out_base}.parquet / {out_base}.csv")
    print(f"- manifest saved: {man_base}.parquet / {man_base}.csv")
    print(f"- combined shape: {combined.shape}")
    print(f"- manifest shape: {manifest.shape}")


if __name__ == "__main__":
    main()


# END OF FILE: combine_snapshot_outputs_to_one.py


################################################################################
# START OF FILE: l0_universe.py
################################################################################

import pandas as pd


def _require_pykrx():
    try:
        from pykrx import stock

        return stock
    except Exception as e:
        raise ImportError(
            "pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요."
        ) from e


def _to_yyyymmdd(s: str) -> str:
    return pd.to_datetime(s).strftime("%Y%m%d")


def build_k200_membership_month_end(
    *,
    start_date: str,
    end_date: str,
    index_code: str = "1028",  # KOSPI200
    anchor_ticker: str = "005930",
) -> pd.DataFrame:
    """
    출력: date(리밸 기준일=월말 거래일), ticker
    - 멘토 피드백(구성종목 변동) 대응용: '시점별 구성 종목' 스냅샷을 월말 단위로 생성
    """
    stock = _require_pykrx()

    s = _to_yyyymmdd(start_date)
    e = _to_yyyymmdd(end_date)

    # trading calendar (삼성전자 기준)
    cal = stock.get_market_ohlcv_by_date(s, e, anchor_ticker)
    if cal is None or len(cal) == 0:
        raise RuntimeError(
            "거래일 캘린더를 생성하지 못했습니다. start/end/anchor_ticker 확인 필요"
        )

    dates = pd.to_datetime(cal.index)
    month_end = (
        pd.Series(dates)
        .groupby(pd.Series(dates).dt.to_period("M"))
        .max()
        .sort_values()
        .tolist()
    )

    records = []
    for d in month_end:
        ymd = d.strftime("%Y%m%d")

        # pykrx 버전차 대비: date 파라미터 유무 모두 시도
        tickers = None
        try:
            tickers = stock.get_index_portfolio_deposit_file(index_code, date=ymd)
        except TypeError:
            tickers = stock.get_index_portfolio_deposit_file(index_code, ymd)

        if tickers is None or len(tickers) == 0:
            raise RuntimeError(
                f"KOSPI200 구성종목 조회 실패: index={index_code}, date={ymd}"
            )

        for t in tickers:
            records.append({"date": d.strftime("%Y-%m-%d"), "ticker": str(t).zfill(6)})

    df = pd.DataFrame(records).sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


# END OF FILE: l0_universe.py


################################################################################
# START OF FILE: l1_ohlcv.py
################################################################################

import pandas as pd


def _require_pykrx():
    try:
        from pykrx import stock

        return stock
    except Exception as e:
        raise ImportError(
            "pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요."
        ) from e


def _to_yyyymmdd(s: str) -> str:
    return pd.to_datetime(s).strftime("%Y%m%d")


def download_ohlcv_panel(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    출력: date,ticker,open,high,low,close,volume,value
    """
    stock = _require_pykrx()

    s = _to_yyyymmdd(start_date)
    e = _to_yyyymmdd(end_date)

    frames = []
    for t in sorted(set(tickers)):
        o = stock.get_market_ohlcv_by_date(s, e, t)
        if o is None or len(o) == 0:
            # 상폐/휴면 등으로 비어있을 수 있음 -> 이후 validate에서 걸러도 됨
            continue

        df = o.copy()
        df = df.reset_index()
        # 컬럼명이 한글일 수 있음(시가/고가/저가/종가/거래량/거래대금)
        rename = {
            "날짜": "date",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "value",
        }
        df = df.rename(columns=rename)

        # pykrx가 인덱스를 날짜로 주는 경우가 많아서 'date'가 없을 수도 있음
        if "date" not in df.columns:
            # reset_index 후 첫 컬럼이 날짜일 가능성
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["ticker"] = str(t).zfill(6)

        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        if "value" in df.columns:
            keep.append("value")
        df = df[keep]

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "OHLCV 다운로드 결과가 비었습니다. tickers/start/end 확인 필요"
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


# END OF FILE: l1_ohlcv.py


################################################################################
# START OF FILE: l2_fundamentals_dart.py
################################################################################

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import redirect_stdout
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _require_opendart():
    try:
        import OpenDartReader

        return OpenDartReader
    except Exception as e:
        raise ImportError(
            "OpenDartReader가 필요합니다. `pip install OpenDartReader` 후 재실행하세요."
        ) from e


def _to_float_safe(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    if s in {"", "-", "nan", "NaN", "None"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _pick_amount(df: pd.DataFrame, names: list[str]) -> float | None:
    """
    OpenDartReader 재무 DF에서 account_nm 기반으로 금액을 뽑는다.
    - 공시 스키마 차이를 고려해 amount 컬럼을 방어적으로 선택한다.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "account_nm" not in df.columns:
        return None

    # amount 컬럼 후보
    cand_cols = [c for c in ["thstrm_amount", "thstrm_amount "] if c in df.columns]
    if not cand_cols:
        # 확실하지 않지만 amount 유사 컬럼 탐색
        num_like = [c for c in df.columns if "amount" in str(c).lower()]
        cand_cols = num_like[:1]

    if not cand_cols:
        return None

    col = cand_cols[0]
    s = df[df["account_nm"].isin(names)][col]
    if s.empty:
        return None

    return _to_float_safe(s.iloc[0])


def _load_corp_map(dart) -> dict[str, str]:
    """
    stock_code(6) -> corp_code(8) 매핑 생성
    """
    corp = getattr(dart, "corp_codes", None)
    corp_df = corp() if callable(corp) else corp
    if corp_df is None or not isinstance(corp_df, pd.DataFrame):
        raise RuntimeError("OpenDartReader corp_codes 로드 실패(버전/환경 확인 필요)")

    if "stock_code" not in corp_df.columns or "corp_code" not in corp_df.columns:
        raise RuntimeError(f"corp_codes 스키마 불일치: {list(corp_df.columns)}")

    c = corp_df.copy()
    c = c[c["stock_code"].notna() & (c["stock_code"].astype(str).str.strip() != "")]
    c["stock_code"] = c["stock_code"].astype(str).str.zfill(6)
    c["corp_code"] = c["corp_code"].astype(str).str.zfill(8)

    return dict(zip(c["stock_code"], c["corp_code"]))


def _call_finstate_safely(
    dart, corp_code: str, year: int, *, reprt_code: str, fs_div: str | None
):
    """
    - OpenDartReader가 '조회 없음'을 dict(status=013)로 반환/출력하는 케이스 방어
    - stdout(불필요 출력) 억제
    - return: (DataFrame|None, status_code|None, message|None)
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            # fs_div 지원 여부가 버전마다 다를 수 있으므로 TypeError 방어
            if fs_div is not None:
                try:
                    res = dart.finstate(
                        corp_code, year, reprt_code=reprt_code, fs_div=fs_div
                    )
                except TypeError:
                    res = dart.finstate(corp_code, year, reprt_code=reprt_code)
            else:
                res = dart.finstate(corp_code, year, reprt_code=reprt_code)
    except Exception as e:
        return None, "EXC", str(e)

    # 케이스 1) dict 반환 (예: {'status':'013','message':'조회된 데이타가 없습니다.'})
    if isinstance(res, dict):
        status = str(res.get("status", "")).strip()
        msg = str(res.get("message", "")).strip()
        return None, status or "DICT", msg or None

    # 케이스 2) DataFrame 반환
    if isinstance(res, pd.DataFrame):
        if res.empty:
            return None, "EMPTY", "empty dataframe"
        return res, None, None

    # 그 외 타입
    return None, "UNKNOWN", f"unexpected type: {type(res)}"


def download_annual_fundamentals(
    *,
    tickers: list[str],
    start_year: int,
    end_year: int,
    api_key: str | None = None,
    sleep_sec: float = 0.2,
    # --- 추가 옵션 (기본값 유지: 기존 호출 깨지지 않음) ---
    reprt_code: str = "11011",  # 사업보고서
    fs_div_order: tuple[str, ...] = ("CFS", "OFS"),  # 연결 우선 -> 개별 fallback
    log_every: int = 100,  # 진행 로그 빈도
    min_success_ratio: float = 0.03,  # 성공률이 너무 낮으면 버그로 판단
) -> pd.DataFrame:
    """
    출력:
      - date(YYYY-12-31), ticker(6), corp_code(8),
        net_income, total_liabilities, equity, debt_ratio, roe

    정책:
      - api_key 없으면 즉시 실패(생략 금지)
      - ticker->corp_code 매핑 실패는 warnings로 누적(하지만 전체가 실패면 예외)
      - DART 조회 결과가 dict(status=013 등)인 케이스를 안전하게 처리
      - 성공률이 극단적으로 낮으면(대부분 013) "파라미터/매핑 오류"로 보고 실패
    """
    if api_key is None:
        api_key = os.getenv("DART_API_KEY")
    api_key = (api_key or "").strip()

    if not api_key:
        raise RuntimeError(
            "DART_API_KEY가 없습니다. 환경변수 DART_API_KEY를 설정해야 L2를 진행할 수 있습니다."
        )

    OpenDartReader = _require_opendart()
    dart = OpenDartReader(api_key)

    corp_map = _load_corp_map(dart)

    # ticker 정규화
    norm_tickers = sorted(
        {str(t).strip().zfill(6) for t in tickers if str(t).strip() != ""}
    )

    # 매핑 성공률 체크(초기 버그 탐지)
    mapped = [t for t in norm_tickers if t in corp_map]
    map_ratio = len(mapped) / max(len(norm_tickers), 1)
    logger.info(
        f"[L2] corp_code mapping: {len(mapped)}/{len(norm_tickers)} ({map_ratio:.1%})"
    )

    if len(mapped) == 0:
        raise RuntimeError(
            "[L2] ticker->corp_code 매핑이 0건입니다. "
            "ticker 포맷(6자리), corp_codes 로드, 유니버스 생성 로직을 확인하세요."
        )

    records: list[dict[str, Any]] = []

    # 통계
    req_cnt = 0
    ok_cnt = 0
    no_data_cnt = 0
    map_miss_cnt = 0
    exc_cnt = 0

    for t in norm_tickers:
        corp_code = corp_map.get(t)
        if not corp_code:
            map_miss_cnt += 1
            continue

        for y in range(start_year, end_year + 1):
            req_cnt += 1

            fs = None
            status = None
            msg = None

            # CFS -> OFS 순서로 시도
            for fs_div in fs_div_order:
                fs, status, msg = _call_finstate_safely(
                    dart, corp_code, y, reprt_code=reprt_code, fs_div=fs_div
                )
                if fs is not None:
                    break

            if fs is None:
                # status가 013/EMPTY 등: 데이터 없음으로 처리
                if status in {"013", "EMPTY"}:
                    no_data_cnt += 1
                elif status == "EXC":
                    exc_cnt += 1
                else:
                    # UNKNOWN/DICT 등도 no-data로 보되, 카운팅은 별도로
                    no_data_cnt += 1

                # 여기서 row를 굳이 만들면 "df는 비지 않지만 값은 전부 None"이 되어 검증이 약해짐.
                # → fundamentals는 '있는 것만' 적재하고, 없는 것은 merge에서 NaN으로 남기는 게 정상.
                if req_cnt % log_every == 0:
                    logger.info(
                        f"[L2] progress req={req_cnt} ok={ok_cnt} no_data={no_data_cnt} map_miss={map_miss_cnt} exc={exc_cnt}"
                    )
                time.sleep(sleep_sec)
                continue

            # 정상 DF를 받았을 때만 파싱
            ok_cnt += 1

            net_income = _pick_amount(
                fs,
                [
                    "당기순이익",
                    "당기순이익(손실)",
                    "지배기업소유주지분에 대한 당기순이익",
                    "지배기업소유주지분당기순이익",
                ],
            )
            total_liab = _pick_amount(fs, ["부채총계", "부채총액"])
            equity = _pick_amount(
                fs,
                [
                    "자본총계",
                    "자본총액",
                    "자본총계(지배기업 소유주지분)",
                    "지배기업소유주지분",
                ],
            )

            debt_ratio = None
            roe = None
            if (total_liab is not None) and (equity is not None) and (equity != 0):
                debt_ratio = (total_liab / equity) * 100.0
            if (net_income is not None) and (equity is not None) and (equity != 0):
                roe = (net_income / equity) * 100.0

            records.append(
                {
                    "date": f"{y}-12-31",
                    "ticker": t,
                    "corp_code": str(corp_code).zfill(8),
                    "net_income": net_income,
                    "total_liabilities": total_liab,
                    "equity": equity,
                    "debt_ratio": debt_ratio,
                    "roe": roe,
                }
            )

            if req_cnt % log_every == 0:
                logger.info(
                    f"[L2] progress req={req_cnt} ok={ok_cnt} no_data={no_data_cnt} map_miss={map_miss_cnt} exc={exc_cnt}"
                )

            time.sleep(sleep_sec)

    df = pd.DataFrame(records)

    # 최종 품질 체크(너가 원한 '검증을 강제'하는 부분)
    if req_cnt == 0:
        raise RuntimeError(
            "[L2] 요청 건수가 0입니다. tickers/start_year/end_year 입력을 확인하세요."
        )

    success_ratio = ok_cnt / max(req_cnt, 1)
    logger.info(
        f"[L2] done req={req_cnt} ok={ok_cnt} no_data={no_data_cnt} map_miss={map_miss_cnt} exc={exc_cnt} success_ratio={success_ratio:.2%}"
    )

    if df.empty:
        raise RuntimeError(
            "[L2] DART 재무 수집 결과가 비었습니다. "
            f"(req={req_cnt}, ok={ok_cnt}, no_data={no_data_cnt}, map_miss={map_miss_cnt}, exc={exc_cnt}) "
            "corp_code 매핑/파라미터(reprt_code/fs_div)/API키/호출 제한을 확인하세요."
        )

    # “대부분 013”이면 버그 가능성이 높으니 강제 실패(조용히 다음 단계로 못 넘어가게)
    if success_ratio < min_success_ratio:
        raise RuntimeError(
            "[L2] DART 조회 성공률이 비정상적으로 낮습니다. "
            f"success_ratio={success_ratio:.2%} (min={min_success_ratio:.2%}). "
            "corp_code 사용 여부, finstate 파라미터(reprt_code/fs_div), ticker 포맷(zfill) 문제를 점검하세요."
        )

    # 타입 정리
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)

    return df


# END OF FILE: l2_fundamentals_dart.py


################################################################################
# START OF FILE: l3_panel_merge.py
################################################################################

# src/stages/l3_panel_merge.py
from __future__ import annotations

import pandas as pd


def build_panel_merged_daily(
    *,
    ohlcv_daily: pd.DataFrame,
    fundamentals_annual: pd.DataFrame,
    universe_membership_monthly: pd.DataFrame | None = None,
    fundamental_lag_days: int = 90,
    filter_k200_members_only: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    ohlcv_daily(date,ticker,OHLCV...) + fundamentals_annual(date,ticker,...)를
    fundamental_lag_days 만큼 지연시킨 effective_date 기준으로 asof merge하여
    panel_merged_daily를 생성한다.

    핵심:
    - merge_asof는 left_on 키(date)가 "전역적으로" 정렬돼 있어야 해서
      merge 직전 정렬을 반드시 ['date','ticker']로 맞춘다.
    """
    warns: list[str] = []

    # -------------------------
    # 0) 기본 방어
    # -------------------------
    if ohlcv_daily is None or ohlcv_daily.empty:
        raise ValueError("ohlcv_daily가 비었습니다.")
    if fundamentals_annual is None or fundamentals_annual.empty:
        warns.append(
            "fundamentals_annual이 비었습니다. 머지는 되지만 재무컬럼은 대부분 NaN이 됩니다."
        )

    o = ohlcv_daily.copy()
    f = (
        fundamentals_annual.copy()
        if fundamentals_annual is not None
        else pd.DataFrame()
    )

    # -------------------------
    # 1) 키 표준화
    # -------------------------
    if "date" not in o.columns or "ticker" not in o.columns:
        raise ValueError(f"ohlcv_daily에 date/ticker가 없습니다: {list(o.columns)}")

    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    o["ticker"] = o["ticker"].astype(str).str.zfill(6)
    o = o.dropna(subset=["date", "ticker"])

    if not f.empty:
        if "date" not in f.columns or "ticker" not in f.columns:
            raise ValueError(
                f"fundamentals_annual에 date/ticker가 없습니다: {list(f.columns)}"
            )

        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f["ticker"] = f["ticker"].astype(str).str.zfill(6)
        f = f.dropna(subset=["date", "ticker"])

        # (ticker,date) 중복 방지
        dup = f.duplicated(["ticker", "date"])
        if dup.any():
            ndup = int(dup.sum())
            warns.append(
                f"fundamentals_annual duplicate (ticker,date)={ndup} -> keep='last'로 제거"
            )
            f = f.sort_values(["ticker", "date"], kind="mergesort").drop_duplicates(
                ["ticker", "date"], keep="last"
            )

        # 지연 반영
        f["effective_date"] = f["date"] + pd.to_timedelta(
            int(fundamental_lag_days), unit="D"
        )

    # -------------------------
    # 2) (옵션) K200 멤버만 필터
    # -------------------------
    # 지금은 기본 False로 두고, 필요 시 확장.
    # (월말 멤버십을 일별로 정교하게 매핑하려면 별도 정책 정의가 필요)
    if (
        filter_k200_members_only
        and universe_membership_monthly is not None
        and not universe_membership_monthly.empty
    ):
        if (
            "date" in universe_membership_monthly.columns
            and "ticker" in universe_membership_monthly.columns
        ):
            u = universe_membership_monthly.copy()
            u["date"] = pd.to_datetime(u["date"], errors="coerce")
            u["ticker"] = u["ticker"].astype(str).str.zfill(6)
            u = u.dropna(subset=["date", "ticker"])

            # 가장 단순한 정책: 월말 멤버십 테이블에 존재하는 ticker만 남김(기간 전체)
            valid_tickers = set(u["ticker"].unique().tolist())
            before = len(o)
            o = o[o["ticker"].isin(valid_tickers)].copy()
            after = len(o)
            warns.append(f"filter_k200_members_only 적용: rows {before} -> {after}")
        else:
            warns.append(
                "filter_k200_members_only 요청했으나 universe_membership_monthly 스키마가 달라 스킵"
            )

    # -------------------------
    # 3) merge_asof (핵심 수정 포인트)
    #    - left: ['date','ticker'] 정렬
    #    - right: ['effective_date','ticker'] 정렬
    # -------------------------
    if f.empty:
        merged = o
    else:
        o_sorted = o.sort_values(["date", "ticker"], kind="mergesort").reset_index(
            drop=True
        )
        f_sorted = f.sort_values(
            ["effective_date", "ticker"], kind="mergesort"
        ).reset_index(drop=True)

        # right의 원래 date는 남기면 혼동이 생기니 제거(필요 시 fiscal_year 같은 컬럼으로 따로 남길 것)
        f_join = f_sorted.drop(columns=["date"], errors="ignore")

        merged = pd.merge_asof(
            o_sorted,
            f_join,
            left_on="date",
            right_on="effective_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )

        # 정리
        merged = merged.drop(columns=["effective_date"], errors="ignore")

    # -------------------------
    # 4) 후처리: downstream 편의 위해 ticker-date 정렬로 되돌림
    # -------------------------
    merged = merged.sort_values(["ticker", "date"], kind="mergesort").reset_index(
        drop=True
    )

    return merged, warns


# END OF FILE: l3_panel_merge.py


################################################################################
# START OF FILE: l4_walkforward_split.py
################################################################################

# src/stages/l4_walkforward_split.py
from __future__ import annotations

import pandas as pd


def _sanitize_panel(panel: pd.DataFrame, price_col: str | None = None):
    df = panel.copy()

    # 중복 컬럼 제거
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if "date" not in df.columns or "ticker" not in df.columns:
        raise KeyError("panel_merged_daily must have columns: date, ticker")

    # date/ticker 타입 강제
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .fillna(df["ticker"].astype(str))
    )
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    # 가격 컬럼 결정
    if price_col and price_col in df.columns:
        px = price_col
    elif "adj_close" in df.columns:
        px = "adj_close"
    elif "close" in df.columns:
        px = "close"
    else:
        raise KeyError("Need price column: adj_close or close (or set l4.price_col)")

    df[px] = pd.to_numeric(df[px], errors="coerce")

    # 정렬(shift 안정성)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 필수 결측 체크(너무 심하면 중단)
    if df["date"].isna().mean() > 0.01:
        raise RuntimeError(
            "Too many NaT in date after parsing. Check panel_merged_daily['date']."
        )

    return df, px


def build_targets_and_folds(
    panel_merged_daily: pd.DataFrame,
    *,
    holdout_years: int = 2,
    step_days: int = 20,
    test_window_days: int = 20,
    embargo_days: int = 20,
    horizon_short: int = 20,
    horizon_long: int = 120,
    rolling_train_years_short: int = 3,
    rolling_train_years_long: int = 5,
    price_col: str | None = None,
):
    warnings: list[str] = []

    df, px = _sanitize_panel(panel_merged_daily, price_col)

    # ✅ forward return 계산 (수정 핵심: 분모는 df[px]!)
    cur = df[px]
    g = df.groupby("ticker", sort=False)[px]

    fwd_s = g.shift(-horizon_short)
    fwd_l = g.shift(-horizon_long)

    # 0/NaN 분모 방어
    cur_safe = cur.where(cur != 0)

    df[f"ret_fwd_{horizon_short}d"] = fwd_s / cur_safe - 1.0
    df[f"ret_fwd_{horizon_long}d"] = fwd_l / cur_safe - 1.0

    # 타깃 결측(마지막 horizon 구간)은 정상이나, 비율 로그용으로 경고만 남김
    miss_s = df[f"ret_fwd_{horizon_short}d"].isna().mean()
    miss_l = df[f"ret_fwd_{horizon_long}d"].isna().mean()
    warnings.append(
        f"[L4] target_missing ret_fwd_{horizon_short}d={miss_s:.2%}, ret_fwd_{horizon_long}d={miss_l:.2%}"
    )

    # trading dates
    dates = pd.DatetimeIndex(pd.unique(df["date"].dropna())).sort_values()
    if len(dates) < 300:
        raise RuntimeError("Too few trading dates after sanitize. Check date parsing.")

    overall_end = dates[-1]
    holdout_threshold = overall_end - pd.DateOffset(years=holdout_years)
    holdout_start = dates[dates.searchsorted(holdout_threshold, side="left")]

    def _build_folds(
        train_years: int,
        horizon_days: int,
        segment: str,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp,
    ):
        seg_start = dates[dates.searchsorted(seg_start, side="left")]
        seg_end = dates[dates.searchsorted(seg_end, side="right") - 1]

        start_pos = int(dates.searchsorted(seg_start, side="left"))
        end_pos = int(dates.searchsorted(seg_end, side="right")) - 1

        folds = []
        fold_i = 0
        max_test_start = end_pos - horizon_days - (test_window_days - 1)

        pos = start_pos
        while pos <= max_test_start:
            test_start_pos = pos
            test_end_pos = pos + (test_window_days - 1)

            train_end_pos = test_start_pos - embargo_days - horizon_days - 1
            if train_end_pos < 0:
                pos += step_days
                continue

            train_end = dates[train_end_pos]
            train_start_threshold = train_end - pd.DateOffset(years=train_years)
            train_start_pos = int(
                dates.searchsorted(train_start_threshold, side="left")
            )
            train_start = dates[train_start_pos]

            fold_i += 1
            folds.append(
                {
                    "fold_id": f"{segment}_{fold_i:04d}",
                    "segment": segment,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": dates[test_start_pos],
                    "test_end": dates[test_end_pos],
                    "train_years": train_years,
                    "horizon_days": horizon_days,
                    "embargo_days": embargo_days,
                    "step_days": step_days,
                    "test_window_days": test_window_days,
                }
            )
            pos += step_days

        return pd.DataFrame(folds)

    dev_start = dates[0]
    dev_end = holdout_start - pd.Timedelta(days=1)

    cv_short = pd.concat(
        [
            _build_folds(
                rolling_train_years_short, horizon_short, "dev", dev_start, dev_end
            ),
            _build_folds(
                rolling_train_years_short,
                horizon_short,
                "holdout",
                holdout_start,
                overall_end,
            ),
        ],
        ignore_index=True,
    )

    cv_long = pd.concat(
        [
            _build_folds(
                rolling_train_years_long, horizon_long, "dev", dev_start, dev_end
            ),
            _build_folds(
                rolling_train_years_long,
                horizon_long,
                "holdout",
                holdout_start,
                overall_end,
            ),
        ],
        ignore_index=True,
    )

    # fold 개수 로그용 경고(메타에 남기기 좋음)
    warnings.append(
        f"[L4] folds_short={len(cv_short):,}, folds_long={len(cv_long):,}, holdout_start={holdout_start.date()}"
    )

    return df, cv_short, cv_long, warnings


# END OF FILE: l4_walkforward_split.py


################################################################################
# START OF FILE: l5_train_models.py
################################################################################

# src/stages/l5_train_models.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    phase: str  # "dev" | "holdout"
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="raise")


def _infer_phase(phase_raw: str, fold_id: str) -> str:
    ph = (phase_raw or "").strip().lower()
    if ph in ("dev", "holdout"):
        return ph
    fid = (fold_id or "").strip().lower()
    if fid.startswith("dev"):
        return "dev"
    if fid.startswith("holdout") or fid.startswith("test"):
        return "holdout"
    # phase가 없거나 예상 외면 dev로 통일 (폴드 생성 규칙상 dev/holdout만 존재해야 함)
    return "dev"


def _standardize_folds(folds: pd.DataFrame) -> list[FoldSpec]:
    if folds is None or not isinstance(folds, pd.DataFrame) or folds.empty:
        raise ValueError("cv_folds is empty or not a DataFrame.")

    phase_col = (
        "phase"
        if "phase" in folds.columns
        else ("segment" if "segment" in folds.columns else None)
    )
    required = ["fold_id", "train_start", "train_end", "test_start", "test_end"]
    missing = [c for c in required if c not in folds.columns]
    if phase_col is None:
        missing.append("phase(or segment)")
    if missing:
        raise ValueError(
            f"cv_folds schema missing columns: {missing}. got={list(folds.columns)}"
        )

    f = folds.copy()
    f["train_start"] = _to_datetime(f["train_start"])
    f["train_end"] = _to_datetime(f["train_end"])
    f["test_start"] = _to_datetime(f["test_start"])
    f["test_end"] = _to_datetime(f["test_end"])

    # 날짜 정합성 체크
    bad = f[(f["train_start"] > f["train_end"]) | (f["test_start"] > f["test_end"])]
    if not bad.empty:
        ex = bad.head(5)
        raise ValueError(f"cv_folds has invalid date ranges. examples:\n{ex}")

    specs: list[FoldSpec] = []
    for _, r in f.iterrows():
        fid = str(r["fold_id"]).strip()
        ph = _infer_phase(str(r[phase_col]), fid)
        specs.append(
            FoldSpec(
                fold_id=fid,
                phase=ph,
                train_start=pd.Timestamp(r["train_start"]),
                train_end=pd.Timestamp(r["train_end"]),
                test_start=pd.Timestamp(r["test_start"]),
                test_end=pd.Timestamp(r["test_end"]),
            )
        )
    return specs


def _pick_feature_cols(df: pd.DataFrame, *, target_col: str) -> list[str]:
    exclude = {
        "date",
        "ticker",
        target_col,
        "ret_fwd_20d",
        "ret_fwd_120d",
        "split",
        "phase",
        "segment",
        "fold_id",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if not cols:
        raise ValueError(
            "No numeric feature columns found after excluding identifiers/targets."
        )
    return cols


def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(y_true).rank(pct=True)
    s2 = pd.Series(y_pred).rank(pct=True)
    v = float(s1.corr(s2))
    return 0.0 if np.isnan(v) else v


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ic = _rank_ic(y_true, y_pred)
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {"rmse": rmse, "mae": mae, "ic_rank": ic, "hit_ratio": hit}


def _build_model(cfg: dict) -> tuple[Pipeline, str]:
    l5 = (cfg.get("l5", {}) if isinstance(cfg, dict) else {}) or {}
    ridge_alpha = float(l5.get("ridge_alpha", 1.0))

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("model", Ridge(alpha=ridge_alpha)),
        ]
    )
    return pipe, f"ridge(alpha={ridge_alpha})"


def _slice_by_date_sorted(
    df: pd.DataFrame, date_arr: np.ndarray, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    # df는 date 기준 오름차순 정렬되어 있어야 함
    left = np.searchsorted(date_arr, np.datetime64(start), side="left")
    right = np.searchsorted(date_arr, np.datetime64(end), side="right")
    if right <= left:
        return df.iloc[0:0]
    return df.iloc[left:right]


def train_oos_predictions(
    *,
    dataset_daily: pd.DataFrame,
    cv_folds: pd.DataFrame,
    cfg: dict,
    target_col: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    warns: list[str] = []

    df = dataset_daily.copy()

    # 필수 컬럼 체크
    for c in ["date", "ticker", target_col]:
        if c not in df.columns:
            raise ValueError(f"dataset_daily missing required column: {c}")

    # date dtype 확정 + 정렬 확정
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    if not df["date"].is_monotonic_increasing:
        # sort 후에도 단조가 아니면 이상 (동일 date는 ticker로만 정렬)
        raise ValueError("dataset_daily date is not monotonic increasing after sort.")

    date_arr = df["date"].to_numpy(dtype="datetime64[ns]", copy=False)

    feature_cols = _pick_feature_cols(df, target_col=target_col)
    fold_specs = _standardize_folds(cv_folds)

    model, model_name = _build_model(cfg)

    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    possible_test_rows = 0
    predicted_rows = 0
    dropped_all_nan_union: set[str] = set()

    for fs in fold_specs:
        dtrain_all = _slice_by_date_sorted(df, date_arr, fs.train_start, fs.train_end)
        dtest_all = _slice_by_date_sorted(df, date_arr, fs.test_start, fs.test_end)

        # target NaN 제거
        dtrain = dtrain_all.dropna(subset=[target_col])
        dtest = dtest_all.dropna(subset=[target_col])

        possible_test_rows += int(dtest.shape[0])

        if dtrain.shape[0] < 2000:
            warns.append(
                f"[L5] fold={fs.fold_id} horizon={horizon}: too few train rows={dtrain.shape[0]}"
            )
            continue
        if dtest.shape[0] < 200:
            warns.append(
                f"[L5] fold={fs.fold_id} horizon={horizon}: too few test rows={dtest.shape[0]}"
            )
            continue

        # ✅ ALL-NaN 피처 제거(폴드별)
        use_cols = [c for c in feature_cols if dtrain[c].notna().any()]
        dropped = [c for c in feature_cols if c not in use_cols]
        if dropped:
            dropped_all_nan_union.update(dropped)

        if len(use_cols) < 5:
            warns.append(
                f"[L5] fold={fs.fold_id} horizon={horizon}: too few usable features={len(use_cols)}"
            )
            continue

        X_train = dtrain[use_cols].to_numpy(dtype=np.float32, copy=False)
        y_train = dtrain[target_col].to_numpy(dtype=np.float32, copy=False)
        X_test = dtest[use_cols].to_numpy(dtype=np.float32, copy=False)
        y_test = dtest[target_col].to_numpy(dtype=np.float32, copy=False)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).astype(np.float32)

        m = _metrics(y_test, y_pred)
        metric_rows.append(
            {
                "horizon": int(horizon),
                "target_col": target_col,
                "model": model_name,
                "fold_id": fs.fold_id,
                "phase": fs.phase,
                "n_train": int(dtrain.shape[0]),
                "n_test": int(dtest.shape[0]),
                "n_features": int(len(use_cols)),
                **m,
            }
        )

        out = dtest[["date", "ticker"]].copy()
        out["y_true"] = y_test
        out["y_pred"] = y_pred
        out["fold_id"] = fs.fold_id
        out["phase"] = fs.phase
        out["horizon"] = int(horizon)
        out["model"] = model_name

        pred_rows.append(out)
        predicted_rows += int(out.shape[0])

    if not pred_rows:
        raise RuntimeError(f"No OOS predictions generated for horizon={horizon}.")

    pred_oos = pd.concat(pred_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    # (date,ticker) 유일성 체크
    dup = int(pred_oos.duplicated(subset=["date", "ticker"]).sum())
    if dup > 0:
        raise RuntimeError(f"OOS predictions have duplicate (date,ticker) rows: {dup}")

    coverage = (
        (predicted_rows / possible_test_rows * 100.0) if possible_test_rows > 0 else 0.0
    )

    report: dict = {
        "horizon": int(horizon),
        "target_col": target_col,
        "model": model_name,
        "possible_test_rows": int(possible_test_rows),
        "predicted_rows": int(predicted_rows),
        "oos_coverage_pct": round(float(coverage), 4),
        "folds_total": int(len(fold_specs)),
        "folds_used": (
            int(metrics_df["fold_id"].nunique()) if not metrics_df.empty else 0
        ),
        "dev_folds": (
            int((metrics_df["phase"] == "dev").sum()) if not metrics_df.empty else 0
        ),
        "holdout_folds": (
            int((metrics_df["phase"] == "holdout").sum()) if not metrics_df.empty else 0
        ),
        "dropped_all_nan_features": sorted(list(dropped_all_nan_union)),
    }

    if not metrics_df.empty:
        for ph in ["dev", "holdout"]:
            sub = metrics_df[metrics_df["phase"] == ph]
            if len(sub) > 0:
                report[f"{ph}_rmse_mean"] = round(float(sub["rmse"].mean()), 8)
                report[f"{ph}_ic_rank_mean"] = round(float(sub["ic_rank"].mean()), 8)
                report[f"{ph}_hit_ratio_mean"] = round(
                    float(sub["hit_ratio"].mean()), 8
                )

    return pred_oos, metrics_df, report, warns


# END OF FILE: l5_train_models.py


################################################################################
# START OF FILE: l6_scoring.py
################################################################################

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"missing column: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="raise")


def _ensure_ticker(df: pd.DataFrame) -> None:
    if "ticker" not in df.columns:
        raise KeyError("missing column: ticker")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name} missing columns: {miss}. got={list(df.columns)}")


def _agg_across_models(df: pd.DataFrame, score_col: str = "y_pred") -> pd.DataFrame:
    """
    (date,ticker,fold_id,phase,horizon) 단위로 모델별 예측을 평균내서 단일 score로 만든다.
    """
    gcols = ["date", "ticker", "fold_id", "phase", "horizon"]
    keep_true = "y_true" in df.columns

    agg = {score_col: "mean"}
    if keep_true:
        agg["y_true"] = "mean"

    out = (
        df.groupby(gcols, sort=False, as_index=False)
        .agg(agg)
        .rename(columns={score_col: "score", "y_true": "true"})
    )
    return out


def _pick_rebalance_rows_by_fold_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    fold별 test window 마지막 날짜(=fold 내 max(date))만 남긴다.
    결과: (fold_id, phase, horizon, ticker)당 1행
    """
    # fold_end 계산
    end = (
        df.groupby(["fold_id", "phase", "horizon"], sort=False)["date"]
        .max()
        .rename("rebalance_date")
        .reset_index()
    )
    out = df.merge(
        end, on=["fold_id", "phase", "horizon"], how="inner", validate="many_to_one"
    )
    out = out.loc[out["date"] == out["rebalance_date"]].copy()
    out.drop(columns=["date"], inplace=True)
    out.rename(columns={"rebalance_date": "date"}, inplace=True)
    return out


def _rank_within_date(df: pd.DataFrame, col: str, suffix: str) -> pd.DataFrame:
    # 높은 점수 = 상위
    df[f"rank_{suffix}"] = df.groupby(["date", "phase"], sort=False)[col].rank(
        ascending=False, method="first"
    )
    df[f"pct_{suffix}"] = df.groupby(["date", "phase"], sort=False)[col].rank(
        pct=True, ascending=False, method="first"
    )
    return df


def build_rebalance_scores(
    *,
    pred_short_oos: pd.DataFrame,
    pred_long_oos: pd.DataFrame,
    weight_short: float = 0.5,
    weight_long: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """
    L6:
    - L5 OOS 예측을 리밸런싱(date) 단위 스코어로 축약
    - fold별 test window 마지막 날짜를 리밸런싱 기준일로 사용
    """
    warns: list[str] = []

    req = ["date", "ticker", "y_pred", "fold_id", "phase", "horizon"]
    _require_cols(pred_short_oos, req, "pred_short_oos")
    _require_cols(pred_long_oos, req, "pred_long_oos")

    ps = pred_short_oos.copy()
    pl = pred_long_oos.copy()

    _ensure_datetime(ps, "date")
    _ensure_datetime(pl, "date")
    _ensure_ticker(ps)
    _ensure_ticker(pl)

    # 모델별 평균으로 단일 score 생성
    ps1 = _agg_across_models(ps, score_col="y_pred")
    pl1 = _agg_across_models(pl, score_col="y_pred")

    # fold test window 마지막 날짜만 남김 (20영업일 리밸과 정합)
    ps2 = _pick_rebalance_rows_by_fold_end(ps1)
    pl2 = _pick_rebalance_rows_by_fold_end(pl1)

    # 컬럼명 분리
    ps2 = ps2.rename(columns={"score": "score_short", "true": "true_short"})
    pl2 = pl2.rename(columns={"score": "score_long", "true": "true_long"})

    # short/long 결합
    key = ["date", "ticker", "phase"]
    out = ps2.merge(pl2, on=key, how="outer", validate="one_to_one")

    # 앙상블 점수
    out["score_ens"] = (
        weight_short * out["score_short"] + weight_long * out["score_long"]
    )

    # 랭킹
    out = _rank_within_date(out, "score_ens", "ens")

    # 중복키 검증
    dup = int(out.duplicated(subset=key).sum())
    if dup != 0:
        raise ValueError(f"rebalance_scores duplicate keys(date,ticker,phase)={dup}")

    # summary 생성
    uni_tickers = int(out["ticker"].nunique())
    summary = (
        out.groupby(["date", "phase"], sort=False)
        .agg(
            n_tickers=("ticker", "nunique"),
            score_ens_mean=("score_ens", "mean"),
            score_ens_std=("score_ens", "std"),
            score_short_missing=("score_short", lambda s: float(s.isna().mean())),
            score_long_missing=("score_long", lambda s: float(s.isna().mean())),
            score_ens_missing=("score_ens", lambda s: float(s.isna().mean())),
        )
        .reset_index()
    )
    summary["coverage_ticker_pct"] = summary["n_tickers"] / uni_tickers * 100.0

    # 품질 리포트(dict)
    quality = {
        "scoring": {
            "rows": int(len(out)),
            "unique_tickers": int(uni_tickers),
            "unique_dates": int(out["date"].nunique()),
            "phases": sorted(out["phase"].dropna().unique().tolist()),
            "avg_coverage_ticker_pct": float(
                round(summary["coverage_ticker_pct"].mean(), 4)
            ),
            "avg_score_short_missing_pct": float(
                round(summary["score_short_missing"].mean() * 100.0, 6)
            ),
            "avg_score_long_missing_pct": float(
                round(summary["score_long_missing"].mean() * 100.0, 6)
            ),
            "avg_score_ens_missing_pct": float(
                round(summary["score_ens_missing"].mean() * 100.0, 6)
            ),
            "weights": {"short": float(weight_short), "long": float(weight_long)},
        }
    }

    return out, summary, quality, warns


# END OF FILE: l6_scoring.py


################################################################################
# START OF FILE: l7_backtest.py
################################################################################

# src/stages/l7_backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 20
    top_k: int = 20
    cost_bps: float = 10.0
    score_col: str = "score_ens"
    ret_col: str = "true_short"
    weighting: str = "equal"  # equal | softmax
    softmax_temp: float = 1.0


def _ensure_datetime(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s)
    return s


def _pick_score_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in ["score_ens", "score", "score_total", "score_ensemble"]:
        if c in df.columns:
            return c
    raise KeyError(
        f"score column not found. tried: {preferred}, score_ens, score, score_total, score_ensemble"
    )


def _pick_ret_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in ["true_short", "y_true", "ret_fwd_20d", "ret"]:
        if c in df.columns:
            return c
    raise KeyError(
        f"return/true column not found. tried: {preferred}, true_short, y_true, ret_fwd_20d, ret"
    )


def _compute_turnover_oneway(
    prev_w: dict[str, float], new_w: dict[str, float]
) -> float:
    keys = set(prev_w) | set(new_w)
    s = 0.0
    for k in keys:
        s += abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0))
    return 0.5 * s


def _weights_from_scores(scores: pd.Series, method: str, temp: float) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.array([], dtype=float)
    if method == "equal":
        return np.full(n, 1.0 / n, dtype=float)

    if method == "softmax":
        x = scores.astype(float).to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        t = float(temp) if float(temp) > 0 else 1.0
        x = x / t
        x = x - np.max(x)  # 안정화
        w = np.exp(x)
        sw = w.sum()
        if sw <= 0:
            return np.full(n, 1.0 / n, dtype=float)
        return w / sw

    raise ValueError(f"unknown weighting: {method}. expected equal|softmax")


def run_backtest(
    rebalance_scores: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    phase_col: str = "phase",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    warns: list[str] = []

    need_cols = [date_col, ticker_col, phase_col]
    for c in need_cols:
        if c not in rebalance_scores.columns:
            raise KeyError(f"rebalance_scores missing required column: {c}")

    score_col = _pick_score_col(rebalance_scores, cfg.score_col)
    ret_col = _pick_ret_col(rebalance_scores, cfg.ret_col)

    df = rebalance_scores[[date_col, ticker_col, phase_col, score_col, ret_col]].copy()
    df[date_col] = _ensure_datetime(df[date_col])
    df[ticker_col] = df[ticker_col].astype(str)
    df[phase_col] = df[phase_col].astype(str)

    # ret_col은 실현 수익률(holding_days)로 간주. 결측 제거.
    before = len(df)
    df = df.dropna(subset=[ret_col])
    dropped = before - len(df)
    if dropped > 0:
        warns.append(f"dropped {dropped} rows with NA {ret_col}")

    # date,phase별로 top_k 선택 (점수 내림차순, 동률은 ticker로 안정화)
    df_sorted = df.sort_values(
        [phase_col, date_col, score_col, ticker_col],
        ascending=[True, True, False, True],
    )

    # groupby.head는 pandas warning 없이 동작
    picked = (
        df_sorted.groupby(
            [phase_col, date_col], sort=False, as_index=False, group_keys=False
        )
        .head(int(cfg.top_k))
        .copy()
    )

    # weights
    positions_rows: list[dict] = []
    returns_rows: list[dict] = []

    for phase, dphase in picked.groupby(phase_col, sort=False):
        prev_w: dict[str, float] = {}
        for dt, g in dphase.groupby(date_col, sort=True):
            g = g.sort_values(
                [score_col, ticker_col], ascending=[False, True]
            ).reset_index(drop=True)
            scores = g[score_col]
            w = _weights_from_scores(scores, cfg.weighting, cfg.softmax_temp)

            new_w = {t: float(wi) for t, wi in zip(g[ticker_col].tolist(), w.tolist())}
            turnover_oneway = _compute_turnover_oneway(prev_w, new_w)

            gross_ret = float(np.dot(w, g[ret_col].astype(float).to_numpy()))
            cost = float(turnover_oneway) * float(cfg.cost_bps) / 10000.0
            net_ret = gross_ret - cost

            for t, wi, sc, tr in zip(g[ticker_col], w, g[score_col], g[ret_col]):
                positions_rows.append(
                    {
                        "date": dt,
                        "phase": phase,
                        "ticker": str(t),
                        "weight": float(wi),
                        "score": float(sc) if pd.notna(sc) else np.nan,
                        "ret_realized": float(tr),
                        "top_k": int(cfg.top_k),
                        "holding_days": int(cfg.holding_days),
                        "cost_bps": float(cfg.cost_bps),
                        "weighting": cfg.weighting,
                    }
                )

            returns_rows.append(
                {
                    "date": dt,
                    "phase": phase,
                    "top_k": int(cfg.top_k),
                    "holding_days": int(cfg.holding_days),
                    "cost_bps": float(cfg.cost_bps),
                    "weighting": cfg.weighting,
                    "n_tickers": int(len(g)),
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                    "turnover_oneway": float(turnover_oneway),
                }
            )

            prev_w = new_w

    bt_positions = (
        pd.DataFrame(positions_rows)
        .sort_values(["phase", "date", "ticker"])
        .reset_index(drop=True)
    )
    bt_returns = (
        pd.DataFrame(returns_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    )

    # equity curve
    eq_rows: list[dict] = []
    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        eq = 1.0
        peak = 1.0
        for dt, r in zip(g["date"], g["net_return"]):
            eq *= 1.0 + float(r)
            peak = max(peak, eq)
            dd = (eq / peak) - 1.0
            eq_rows.append(
                {"date": dt, "phase": phase, "equity": float(eq), "drawdown": float(dd)}
            )
    bt_equity_curve = (
        pd.DataFrame(eq_rows).sort_values(["phase", "date"]).reset_index(drop=True)
    )

    # metrics
    met_rows: list[dict] = []
    periods_per_year = 252.0 / float(cfg.holding_days) if cfg.holding_days > 0 else 12.6

    for phase, g in bt_returns.groupby("phase", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        r_gross = g["gross_return"].astype(float).to_numpy()
        r_net = g["net_return"].astype(float).to_numpy()

        eq_g = (1.0 + pd.Series(r_gross)).cumprod().iloc[-1]
        eq_n = (1.0 + pd.Series(r_net)).cumprod().iloc[-1]

        d0 = pd.to_datetime(g["date"].iloc[0])
        d1 = pd.to_datetime(g["date"].iloc[-1])
        years = max((d1 - d0).days / 365.25, 1e-9)

        gross_cagr = float(eq_g ** (1.0 / years) - 1.0)
        net_cagr = float(eq_n ** (1.0 / years) - 1.0)

        gross_vol = (
            float(np.std(r_gross, ddof=1) * np.sqrt(periods_per_year))
            if len(r_gross) > 1
            else 0.0
        )
        net_vol = (
            float(np.std(r_net, ddof=1) * np.sqrt(periods_per_year))
            if len(r_net) > 1
            else 0.0
        )

        gross_sharpe = (
            float(
                (np.mean(r_gross) / (np.std(r_gross, ddof=1) + 1e-12))
                * np.sqrt(periods_per_year)
            )
            if len(r_gross) > 1
            else 0.0
        )
        net_sharpe = (
            float(
                (np.mean(r_net) / (np.std(r_net, ddof=1) + 1e-12))
                * np.sqrt(periods_per_year)
            )
            if len(r_net) > 1
            else 0.0
        )

        # MDD from equity curve
        eq = 1.0
        peak = 1.0
        mdd_g = 0.0
        for r in r_gross:
            eq *= 1.0 + float(r)
            peak = max(peak, eq)
            mdd_g = min(mdd_g, (eq / peak) - 1.0)

        eq = 1.0
        peak = 1.0
        mdd_n = 0.0
        for r in r_net:
            eq *= 1.0 + float(r)
            peak = max(peak, eq)
            mdd_n = min(mdd_n, (eq / peak) - 1.0)

        met_rows.append(
            {
                "phase": phase,
                "top_k": int(cfg.top_k),
                "holding_days": int(cfg.holding_days),
                "cost_bps": float(cfg.cost_bps),
                "n_rebalances": int(len(g)),
                "gross_total_return": float(eq_g - 1.0),
                "net_total_return": float(eq_n - 1.0),
                "gross_cagr": gross_cagr,
                "net_cagr": net_cagr,
                "gross_vol_ann": gross_vol,
                "net_vol_ann": net_vol,
                "gross_sharpe": gross_sharpe,
                "net_sharpe": net_sharpe,
                "gross_mdd": float(mdd_g),
                "net_mdd": float(mdd_n),
                "gross_hit_ratio": (
                    float((r_gross > 0).mean()) if len(r_gross) else np.nan
                ),
                "net_hit_ratio": float((r_net > 0).mean()) if len(r_net) else np.nan,
                "avg_turnover_oneway": float(g["turnover_oneway"].mean()),
                "avg_n_tickers": float(g["n_tickers"].mean()),
                "date_start": d0,
                "date_end": d1,
                "weighting": cfg.weighting,
            }
        )

    bt_metrics = pd.DataFrame(met_rows)

    quality = {
        "backtest": {
            "holding_days": int(cfg.holding_days),
            "top_k": int(cfg.top_k),
            "cost_bps": float(cfg.cost_bps),
            "score_col_used": score_col,
            "ret_col_used": ret_col,
            "weighting": cfg.weighting,
            "softmax_temp": float(cfg.softmax_temp),
            "rows_positions": int(len(bt_positions)),
            "rows_returns": int(len(bt_returns)),
        }
    }

    return bt_positions, bt_returns, bt_equity_curve, bt_metrics, quality, warns


# END OF FILE: l7_backtest.py


################################################################################
# START OF FILE: l7b_sensitivity.py
################################################################################

# src/stages/l7b_sensitivity.py
from __future__ import annotations

import pandas as pd

from src.stages.backtest.l7_backtest import BacktestConfig, run_backtest


def run_sensitivity(
    rebalance_scores: pd.DataFrame,
    *,
    holding_days: int,
    top_k_grid: list[int],
    cost_bps_grid: list[float],
    weighting_grid: list[str],
    score_col: str,
    ret_col: str,
) -> tuple[pd.DataFrame, dict, list[str]]:
    warns: list[str] = []
    rows = []

    for w in weighting_grid:
        for k in top_k_grid:
            for c in cost_bps_grid:
                cfg = BacktestConfig(
                    holding_days=int(holding_days),
                    top_k=int(k),
                    cost_bps=float(c),
                    score_col=str(score_col),
                    ret_col=str(ret_col),
                    weighting=str(w),
                )
                _, _, _, met, q, wns = run_backtest(rebalance_scores, cfg)
                warns.extend(wns or [])
                met = met.copy()
                met["grid_top_k"] = int(k)
                met["grid_cost_bps"] = float(c)
                met["grid_weighting"] = str(w)
                rows.append(met)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    quality = {
        "sensitivity": {
            "holding_days": int(holding_days),
            "top_k_grid": list(map(int, top_k_grid)),
            "cost_bps_grid": list(map(float, cost_bps_grid)),
            "weighting_grid": list(map(str, weighting_grid)),
            "n_runs": int(len(top_k_grid) * len(cost_bps_grid) * len(weighting_grid)),
        }
    }
    return out, quality, warns


# END OF FILE: l7b_sensitivity.py


################################################################################
# START OF FILE: l7c_benchmark.py
################################################################################

# src/stages/l7c_benchmark.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _pick_strategy_return_series(bt_returns: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    bt_returns에서 전략 수익률(리밸런스 단위)을 의미하는 컬럼을 '확정적으로' 확보한다.
    우선순위:
      1) net_return 계열
      2) gross_return + (cost/turnover*cost_bps)로 net 재구성
      3) gross_return만 있으면 gross를 사용(비용 반영 못한 상태)
    """
    cols = set(bt_returns.columns)

    # 1) net_return 후보
    net_candidates = [
        "net_return",
        "ret_net",
        "net_ret",
        "return_net",
        "net",
        "net_r",
        "portfolio_net_return",
    ]
    for c in net_candidates:
        if c in cols:
            return bt_returns[c].astype(float), c

    # 2) gross_return + cost 로 net 재구성
    gross_candidates = ["gross_return", "ret_gross", "gross_ret", "return_gross"]
    gross_col: Optional[str] = None
    for c in gross_candidates:
        if c in cols:
            gross_col = c
            break

    if gross_col is not None:
        g = bt_returns[gross_col].astype(float)

        # (2-a) cost 컬럼이 이미 있으면 바로 차감
        cost_candidates = ["cost", "trade_cost", "tcost", "cost_rate"]
        for cc in cost_candidates:
            if cc in cols:
                return (g - bt_returns[cc].astype(float)), f"{gross_col}-({cc})"

        # (2-b) turnover * cost_bps/10000 으로 비용 재구성
        turnover_candidates = ["turnover_oneway", "turnover", "oneway_turnover"]
        cost_bps_candidates = ["cost_bps", "tcost_bps"]
        tcol = next((c for c in turnover_candidates if c in cols), None)
        cbps = next((c for c in cost_bps_candidates if c in cols), None)
        if tcol is not None and cbps is not None:
            cost = (
                bt_returns[tcol].astype(float)
                * bt_returns[cbps].astype(float)
                / 10000.0
            )
            return (g - cost), f"{gross_col}-({tcol}*{cbps}/10000)"

        # 3) gross_return만 사용
        return g, gross_col

    # 4) 최후 후보(혹시 컬럼명이 단순한 경우)
    fallback = ["return", "ret", "r", "pnl", "strategy_return", "portfolio_return"]
    for c in fallback:
        if c in cols:
            return bt_returns[c].astype(float), c

    raise KeyError(
        "bt_returns에 전략 수익률 컬럼이 없습니다. " f"현재 컬럼={sorted(list(cols))}"
    )


def build_universe_benchmark_returns(
    rebalance_scores: pd.DataFrame,
    *,
    ret_col_candidates=None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    phase_col: str = "phase",
) -> pd.DataFrame:
    if ret_col_candidates is None:
        ret_col_candidates = ["true_short", "y_true", "ret_fwd_20d", "ret", "return"]

    for c in [date_col, phase_col, ticker_col]:
        if c not in rebalance_scores.columns:
            raise KeyError(f"rebalance_scores missing required column: {c}")

    # benchmark에 쓸 '실현수익률' 컬럼 선택
    ret_col = None
    for c in ret_col_candidates:
        if c in rebalance_scores.columns:
            ret_col = c
            break
    if ret_col is None:
        raise KeyError(
            f"no benchmark return col in rebalance_scores. tried={ret_col_candidates}"
        )

    df = rebalance_scores[[date_col, phase_col, ticker_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[phase_col] = df[phase_col].astype(str)
    df[ticker_col] = df[ticker_col].astype(str)

    bench = (
        df.dropna(subset=[ret_col])
        .groupby([phase_col, date_col], sort=False)[ret_col]
        .mean()
        .reset_index()
        .rename(columns={ret_col: "bench_return"})
        .sort_values([phase_col, date_col])
        .reset_index(drop=True)
    )

    bench["bench_equity"] = 1.0
    for phase, g in bench.groupby(phase_col, sort=False):
        idx = g.index
        bench.loc[idx, "bench_equity"] = (
            (1.0 + g["bench_return"].astype(float)).cumprod().values
        )

    return bench


def compare_strategy_vs_benchmark(
    bt_returns: pd.DataFrame,
    bench_returns: pd.DataFrame,
    *,
    holding_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    warns: list[str] = []

    br = bt_returns.copy()
    br["date"] = pd.to_datetime(br["date"])
    br["phase"] = br["phase"].astype(str)

    bench = bench_returns.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench["phase"] = bench["phase"].astype(str)

    strat_ret, used_col = _pick_strategy_return_series(br)
    br["_strategy_return_"] = strat_ret

    m = br.merge(
        bench[["phase", "date", "bench_return"]], on=["phase", "date"], how="inner"
    )
    if len(m) == 0:
        raise ValueError("no overlapping dates between bt_returns and benchmark")

    m["excess_return"] = m["_strategy_return_"].astype(float) - m[
        "bench_return"
    ].astype(float)

    periods_per_year = 252.0 / float(holding_days) if holding_days > 0 else 12.6

    rows = []
    for phase, g in m.groupby("phase", sort=False):
        ex = g["excess_return"].astype(float).to_numpy()
        te = (
            float(np.std(ex, ddof=1) * np.sqrt(periods_per_year))
            if len(ex) > 1
            else 0.0
        )
        ir = (
            float(
                (np.mean(ex) / (np.std(ex, ddof=1) + 1e-12)) * np.sqrt(periods_per_year)
            )
            if len(ex) > 1
            else 0.0
        )

        strat = g["_strategy_return_"].astype(float).to_numpy()
        b = g["bench_return"].astype(float).to_numpy()

        corr = float(np.corrcoef(strat, b)[0, 1]) if len(ex) > 1 else np.nan
        beta = (
            float(np.cov(strat, b, ddof=1)[0, 1] / (np.var(b, ddof=1) + 1e-12))
            if len(ex) > 1
            else np.nan
        )

        rows.append(
            {
                "phase": phase,
                "n_rebalances": int(len(g)),
                "tracking_error_ann": te,
                "information_ratio": ir,
                "corr_vs_benchmark": corr,
                "beta_vs_benchmark": beta,
                "date_start": g["date"].min(),
                "date_end": g["date"].max(),
                "strategy_return_col_used": used_col,
            }
        )

    compare_metrics = pd.DataFrame(rows)

    quality = {
        "benchmark": {
            "holding_days": int(holding_days),
            "rows_overlap": int(len(m)),
            "strategy_return_col_used": used_col,
        }
    }
    return (
        m.sort_values(["phase", "date"]).reset_index(drop=True),
        compare_metrics,
        quality,
        warns,
    )


# END OF FILE: l7c_benchmark.py


################################################################################
# START OF FILE: l7d_stability.py
################################################################################

# src/stages/l7d_stability.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class L7DConfig:
    holding_days: int = 20
    date_col: str = "date"
    phase_col: str = "phase"
    # bt_returns에서 사용할 net return 컬럼 후보(앞에서부터 우선)
    net_return_candidates: tuple[str, ...] = (
        "net_return",
        "net_ret",
        "net_period_return",
    )


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _pick_net_return_col(bt_returns: pd.DataFrame, cfg: L7DConfig) -> str:
    for c in cfg.net_return_candidates:
        if c in bt_returns.columns:
            return c
    raise KeyError(
        f"[L7D] bt_returns must contain one of {list(cfg.net_return_candidates)}. "
        f"got={bt_returns.columns.tolist()}"
    )


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    """
    r: period returns (net_return) 1D
    return: MDD (negative number, e.g., -0.2)
    """
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(np.min(dd)) if dd.size else 0.0


def build_bt_yearly_metrics(
    bt_returns: pd.DataFrame,
    *,
    holding_days: int = 20,
    date_col: str = "date",
    phase_col: str = "phase",
    net_return_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    bt_returns -> 연도별 성과 지표(DataFrame) 생성

    Output columns (fixed):
      ['phase','year','n_rebalances','net_total_return','net_vol_ann','net_sharpe',
       'net_mdd','net_hit_ratio','date_start','date_end','net_return_col_used']
    """
    if not isinstance(bt_returns, pd.DataFrame) or bt_returns.empty:
        raise ValueError("[L7D] bt_returns must be a non-empty DataFrame.")

    cfg = L7DConfig(holding_days=holding_days, date_col=date_col, phase_col=phase_col)

    df = bt_returns.copy()

    # 필수 컬럼 체크
    for c in (cfg.date_col, cfg.phase_col):
        if c not in df.columns:
            raise KeyError(f"[L7D] bt_returns missing required column: {c}")

    df = _ensure_datetime(df, cfg.date_col)
    if df[cfg.date_col].isna().any():
        bad = df[df[cfg.date_col].isna()].head(5)
        raise ValueError(f"[L7D] bt_returns has non-parsable dates. sample:\n{bad}")

    used_col = net_return_col if net_return_col else _pick_net_return_col(df, cfg)
    if used_col not in df.columns:
        raise KeyError(
            f"[L7D] net_return_col='{used_col}' not found in bt_returns columns."
        )

    # 숫자형 강제
    df[used_col] = pd.to_numeric(df[used_col], errors="coerce")

    # year 생성
    df["year"] = df[cfg.date_col].dt.year.astype(int)

    ann_factor = math.sqrt(252.0 / float(cfg.holding_days))

    rows: list[dict] = []
    g = df.groupby([cfg.phase_col, "year"], sort=True)

    for (phase, year), d in g:
        d = d.sort_values(cfg.date_col)

        r = d[used_col].to_numpy(dtype=float)
        r = r[~np.isnan(r)]  # 결측 제거

        n = int(r.size)

        # 연도 내 리밸런스가 0개면 스킵하지 않고 0으로 기록
        if n == 0:
            net_total_return = 0.0
            net_vol_ann = 0.0
            net_sharpe = 0.0
            net_mdd = 0.0
            net_hit_ratio = 0.0
        else:
            net_total_return = float(np.prod(1.0 + r) - 1.0)

            # 표준편차 0 방지
            std = float(np.std(r, ddof=1)) if n >= 2 else 0.0
            mean = float(np.mean(r))

            net_vol_ann = std * ann_factor if std > 0 else 0.0
            net_sharpe = (mean / std) * ann_factor if std > 0 else 0.0
            net_mdd = _max_drawdown_from_returns(r)
            net_hit_ratio = float(np.mean(r > 0.0))

        date_start = pd.Timestamp(d[cfg.date_col].min())
        date_end = pd.Timestamp(d[cfg.date_col].max())

        rows.append(
            {
                "phase": phase,
                "year": int(year),
                "n_rebalances": int(d.shape[0]),
                "net_total_return": net_total_return,
                "net_vol_ann": net_vol_ann,
                "net_sharpe": net_sharpe,
                "net_mdd": net_mdd,
                "net_hit_ratio": net_hit_ratio,
                "date_start": date_start,
                "date_end": date_end,
                "net_return_col_used": used_col,
            }
        )

    out = pd.DataFrame(rows)

    # 컬럼 순서 고정(요청 스키마 그대로)
    out = out[
        [
            "phase",
            "year",
            "n_rebalances",
            "net_total_return",
            "net_vol_ann",
            "net_sharpe",
            "net_mdd",
            "net_hit_ratio",
            "date_start",
            "date_end",
            "net_return_col_used",
        ]
    ].sort_values(["phase", "year"], ignore_index=True)

    return out


def run_l7d_stability_from_artifacts(
    *,
    bt_returns: pd.DataFrame,
    holding_days: int = 20,
) -> tuple[pd.DataFrame, list]:
    """
    run_all에서 바로 호출하기 위한 래퍼
    """
    yearly = build_bt_yearly_metrics(bt_returns, holding_days=holding_days)
    warns: list = []
    return yearly, warns


def build_bt_rolling_sharpe(bt_returns, cfg):
    import numpy as np
    import pandas as pd

    l7 = cfg.get("l7", {}) or {}
    l7d = cfg.get("l7d", {}) or {}

    holding_days = int(l7.get("holding_days", 20))
    window_rebalances = int(l7d.get("rolling_window_rebalances", 12))

    return_col = "net_return"
    if return_col not in bt_returns.columns:
        raise KeyError(f"bt_returns missing '{return_col}'")

    df = bt_returns[["date", "phase", return_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["phase"] = df["phase"].astype(str)
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")

    # (안전) inf -> nan
    df[return_col] = df[return_col].replace([np.inf, -np.inf], np.nan)

    periods_per_year = 252.0 / float(holding_days)
    ann_factor = np.sqrt(periods_per_year)

    out = []
    for phase, g in df.groupby("phase", sort=False):
        s = g.sort_values("date").reset_index(drop=True)
        r = s[return_col].astype(float)

        # 관측치 수(rolling window 내 유효값 개수)
        roll_n = r.rolling(window_rebalances, min_periods=1).count()

        # 평균: min_periods=1로 해서 초반 NaN 제거
        roll_mean = r.rolling(window_rebalances, min_periods=1).mean()

        # 표준편차: 1개 표본이면 NaN이 정상 -> 0으로 처리
        roll_std = r.rolling(window_rebalances, min_periods=2).std(ddof=1)

        # NaN 정리 (검증 스크립트가 finite 요구)
        roll_mean = roll_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        roll_std = roll_std.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        roll_vol_ann = roll_std * ann_factor

        # 교체(경고 제거): division을 std>0인 곳에서만 수행
        mean_np = roll_mean.to_numpy(dtype=float)
        std_np = roll_std.to_numpy(dtype=float)

        ratio = np.zeros_like(mean_np, dtype=float)
        np.divide(mean_np, std_np, out=ratio, where=(std_np > 0.0))

        roll_sharpe = ratio * ann_factor

        out.append(
            pd.DataFrame(
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
        )

    res = pd.concat(out, ignore_index=True)

    # 최종 finite 강제 (혹시라도 남아있으면 0)
    for c in ["net_rolling_mean", "net_rolling_vol_ann", "net_rolling_sharpe"]:
        res[c] = pd.to_numeric(res[c], errors="coerce")
        res[c] = res[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keys = ["date", "phase", "net_return_col_used"]
    if res.duplicated(keys).any():
        raise ValueError(
            "bt_rolling_sharpe must be unique on (date,phase,net_return_col_used)"
        )

    return res


# END OF FILE: l7d_stability.py


################################################################################
# START OF FILE: rebuild_bt_rolling_sharpe.py
################################################################################

# src/stages/rebuild_bt_rolling_sharpe.py
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


# END OF FILE: rebuild_bt_rolling_sharpe.py


################################################################################
# START OF FILE: repair_bt_rolling_sharpe.py
################################################################################

# src/stages/repair_bt_rolling_sharpe.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.meta import build_meta, save_meta


def _root() -> Path:
    # .../03_code/src/stages/repair_bt_rolling_sharpe.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]


def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"


def _infer_keys(df: pd.DataFrame) -> list[str]:
    # validator가 현재 date/phase/net_return_col_used 까지 잡고 있으므로 동일하게 사용
    keys = ["date", "phase"]
    if "net_return_col_used" in df.columns:
        keys.append("net_return_col_used")

    # 만약 실제로 window/series 컬럼이 존재한다면 키에 포함(있으면 더 안전)
    extra_candidates = [
        "window_days",
        "window",
        "lookback_days",
        "lookback",
        "rolling_window",
        "series",
        "kind",
        "metric",
        "return_col",
    ]
    for c in extra_candidates:
        if c in df.columns and c not in keys:
            keys.append(c)

    return keys


def _check_conflicting_duplicates(
    df: pd.DataFrame, keys: list[str]
) -> tuple[bool, pd.DataFrame]:
    """
    keys가 동일한데 다른 값(충돌)이 있는지 검사.
    - 충돌이 없으면 (True, empty)
    - 충돌이 있으면 (False, sample_df)
    """
    dup_mask = df.duplicated(keys, keep=False)
    if not dup_mask.any():
        return True, pd.DataFrame()

    ddup = df.loc[dup_mask].copy()

    non_keys = [c for c in df.columns if c not in keys]

    # 각 key 그룹에서 non-key 컬럼의 nunique가 1을 초과하면 "충돌"
    nunique = ddup.groupby(keys, dropna=False)[non_keys].nunique(dropna=False)
    conflict_groups = (nunique > 1).any(axis=1)
    if conflict_groups.any():
        bad_keys = conflict_groups[conflict_groups].index.to_frame(index=False)
        sample_keys = bad_keys.head(5)
        sample = ddup.merge(sample_keys, on=keys, how="inner").head(30)
        return False, sample

    return True, pd.DataFrame()


def main():
    print("=== REPAIR bt_rolling_sharpe (dedup) ===")
    root = _root()
    cfg_path = _cfg_path(root)

    print("ROOT :", root)
    print("CFG  :", cfg_path)

    if not cfg_path.exists():
        raise SystemExit(f"[FAIL] config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    base = interim / "bt_rolling_sharpe"
    if not artifact_exists(base):
        raise SystemExit(f"[FAIL] artifact missing: {base}")

    df = load_artifact(base)
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise SystemExit(
            f"[FAIL] invalid DataFrame loaded: shape={getattr(df,'shape',None)}"
        )

    # 타입 정리
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise SystemExit("[FAIL] bt_rolling_sharpe has invalid 'date' (NaT)")
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str)

    keys = _infer_keys(df)

    # 중복 현황
    before_rows = int(df.shape[0])
    before_unique_keys = int(df[keys].drop_duplicates().shape[0])
    dup_cnt = int(df.duplicated(keys, keep=False).sum())

    print(f"keys used: {keys}")
    print(
        f"rows(before)={before_rows}, unique_keys(before)={before_unique_keys}, dup_rows={dup_cnt}"
    )

    # 충돌 여부 확인 (같은 key인데 값이 다르면 여기서 FAIL)
    ok, sample = _check_conflicting_duplicates(df, keys)
    if not ok:
        print(
            "\n[FAIL] Found conflicting duplicates (same keys, different values). Sample:"
        )
        print(sample)
        raise SystemExit(
            "[FAIL] cannot auto-dedup safely. Fix L7D generation logic first."
        )

    # 안전한 dedup: "완전히 동일한 행" 제거 → 그 다음에도 key 중복 있으면 key 기준 집계
    df2 = df.drop_duplicates().copy()

    if df2.duplicated(keys).any():
        # key가 같고 값 충돌은 없다고 확인됐으므로, key 기준으로 안전 집계 가능
        non_keys = [c for c in df2.columns if c not in keys]
        num_cols = [c for c in non_keys if pd.api.types.is_numeric_dtype(df2[c])]
        other_cols = [c for c in non_keys if c not in num_cols]

        agg = {c: "mean" for c in num_cols}
        agg.update({c: "first" for c in other_cols})

        df2 = df2.groupby(keys, as_index=False, dropna=False).agg(agg)

    # 최종 검증
    after_rows = int(df2.shape[0])
    after_unique_keys = int(df2[keys].drop_duplicates().shape[0])
    if after_rows != after_unique_keys:
        raise SystemExit(
            f"[FAIL] dedup failed: rows(after)={after_rows} != unique_keys(after)={after_unique_keys}"
        )

    print(f"rows(after)={after_rows}, unique_keys(after)={after_unique_keys}")

    # 저장 (기존 bt_rolling_sharpe 덮어쓰기)
    save_formats = cfg.get("run", {}).get("save_formats", ["parquet", "csv"])
    save_artifact(df2, base, force=True, formats=save_formats)

    meta = build_meta(
        stage="L7D:bt_rolling_sharpe",
        run_id="repair_bt_rolling_sharpe_dedup",
        df=df2,
        out_base_path=base,
        warnings=[f"dedup applied: rows {before_rows} -> {after_rows} on keys={keys}"],
        inputs={"source": "bt_rolling_sharpe (existing)"},
        repo_dir=get_path(cfg, "base_dir"),
        quality={
            "repair": {
                "keys": keys,
                "rows_before": before_rows,
                "rows_after": after_rows,
                "dup_rows_before": dup_cnt,
            }
        },
    )
    save_meta(base, meta, force=True)

    print("✅ REPAIR COMPLETE: bt_rolling_sharpe dedup saved with updated meta.")


if __name__ == "__main__":
    main()


# END OF FILE: repair_bt_rolling_sharpe.py


################################################################################
# START OF FILE: run_all.py
################################################################################

# src/run_all.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.stages.data.l0_universe import build_k200_membership_month_end
from src.stages.data.l1_ohlcv import download_ohlcv_panel
from src.stages.data.l2_fundamentals_dart import download_annual_fundamentals
from src.stages.data.l3_panel_merge import build_panel_merged_daily
from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact
from src.utils.meta import build_meta, save_meta
from src.utils.quality import fundamental_coverage_report, walkforward_quality_report
from src.utils.validate import raise_if_invalid, validate_df

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Stage 함수들
# -----------------------------
def run_L0_universe(cfg, artifacts=None, *, force=False):
    p = cfg.get("params", {})
    df = build_k200_membership_month_end(
        start_date=p.get("start_date", "2015-01-02"),
        end_date=p.get("end_date", "2024-12-31"),
        index_code=p.get("index_code", "1028"),
        anchor_ticker=p.get("anchor_ticker", "005930"),
    )
    return {"universe_k200_membership_monthly": df}, []


def run_L1_base(cfg, artifacts, *, force=False):
    p = cfg.get("params", {})
    uni = artifacts["universe_k200_membership_monthly"]
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())

    df = download_ohlcv_panel(
        tickers=tickers,
        start_date=p.get("start_date", "2016-01-01"),
        end_date=p.get("end_date", "2024-12-31"),
    )
    return {"ohlcv_daily": df}, []


def run_L1B_pykrx_fundamentals(cfg, artifacts, *, force=False):
    """L1B: pykrx 재무데이터 다운로드"""
    from src.stages.data.l1b_pykrx_fundamentals import download_pykrx_fundamentals_daily

    l1b = cfg.get("l1b", {}) or {}
    if not l1b.get("enabled", True):
        return {}, ["[L1B] pykrx fundamentals disabled"]

    p = cfg.get("params", {})
    ohlcv = artifacts.get("ohlcv_daily")

    if ohlcv is None or ohlcv.empty:
        return {}, ["[L1B] ohlcv_daily가 없어 pykrx fundamentals를 건너뜁니다."]

    tickers = sorted(ohlcv["ticker"].unique().tolist())

    df = download_pykrx_fundamentals_daily(
        tickers=tickers,
        start_date=p.get("start_date", "2016-01-01"),
        end_date=p.get("end_date", "2024-12-31"),
        sleep_sec=float(l1b.get("sleep_sec", 0.1)),
        log_every=int(l1b.get("log_every", 50)),
    )

    return {"pykrx_fundamentals_daily": df}, []


def run_L2_merge(cfg, artifacts, *, force=False):
    p = cfg.get("params", {})
    uni = artifacts["universe_k200_membership_monthly"]
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())

    start_year = int(pd.to_datetime(p.get("start_date", "2015-01-02")).year)
    end_year = int(pd.to_datetime(p.get("end_date", "2024-12-31")).year)

    df = download_annual_fundamentals(
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
        api_key=cfg.get("secrets", {}).get("dart_api_key"),
    )
    return {"fundamentals_annual": df}, []


def run_L3_features(cfg, artifacts, *, force=False):
    p = cfg.get("params", {})
    lag_days = int(p.get("fundamental_lag_days", 90))

    df, warns = build_panel_merged_daily(
        ohlcv_daily=artifacts["ohlcv_daily"],
        fundamentals_annual=artifacts["fundamentals_annual"],
        universe_membership_monthly=artifacts.get("universe_k200_membership_monthly"),
        fundamental_lag_days=lag_days,
        filter_k200_members_only=bool(p.get("filter_k200_members_only", False)),
        pykrx_fundamentals_daily=artifacts.get("pykrx_fundamentals_daily"),
    )
    return {"panel_merged_daily": df}, warns


def run_L4_split(cfg, artifacts, *, force=False):
    from src.stages.data.l4_walkforward_split import build_targets_and_folds

    l4 = cfg.get("l4", {}) or {}
    panel = artifacts["panel_merged_daily"]

    df, cv_s, cv_l, warns = build_targets_and_folds(
        panel,
        holdout_years=int(l4.get("holdout_years", 2)),
        step_days=int(l4.get("step_days", 20)),
        test_window_days=int(l4.get("test_window_days", 20)),
        embargo_days=int(l4.get("embargo_days", 20)),
        horizon_short=int(l4.get("horizon_short", 20)),
        horizon_long=int(l4.get("horizon_long", 120)),
        rolling_train_years_short=int(l4.get("rolling_train_years_short", 3)),
        rolling_train_years_long=int(l4.get("rolling_train_years_long", 5)),
        price_col=l4.get("price_col", None),
    )

    return {
        "dataset_daily": df,
        "cv_folds_short": cv_s,
        "cv_folds_long": cv_l,
    }, warns


def run_L5_modeling(cfg, artifacts, *, force=False):
    df = artifacts["dataset_daily"]
    cv_s = artifacts["cv_folds_short"]
    cv_l = artifacts["cv_folds_long"]

    l4 = cfg.get("l4", {}) or {}
    hs = int(l4.get("horizon_short", 20))
    hl = int(l4.get("horizon_long", 120))

    target_s = f"ret_fwd_{hs}d"
    target_l = f"ret_fwd_{hl}d"

    pred_s, met_s, rep_s, w_s = train_oos_predictions(
        dataset_daily=df,
        cv_folds=cv_s,
        cfg=cfg,
        target_col=target_s,
        horizon=hs,
    )
    pred_l, met_l, rep_l, w_l = train_oos_predictions(
        dataset_daily=df,
        cv_folds=cv_l,
        cfg=cfg,
        target_col=target_l,
        horizon=hl,
    )

    metrics = pd.concat([met_s, met_l], ignore_index=True)
    warns = (w_s or []) + (w_l or [])

    # meta.quality에 실어줄 L5 report를 artifacts에 임시 보관
    artifacts["_l5_report_short"] = rep_s
    artifacts["_l5_report_long"] = rep_l

    return {
        "pred_short_oos": pred_s,
        "pred_long_oos": pred_l,
        "model_metrics": metrics,
    }, warns


def run_L6_scoring(cfg, artifacts, *, force=False):
    # config: params.l6 또는 l6 둘 다 지원
    p = cfg.get("params", {}) or {}
    l6 = p.get("l6", {}) if isinstance(p.get("l6", {}), dict) else {}
    if not l6:
        l6 = cfg.get("l6", {}) or {}

    w_s = float(l6.get("weight_short", 0.5))
    w_l = float(l6.get("weight_long", 0.5))

    scores, summary, quality, warns = build_rebalance_scores(
        pred_short_oos=artifacts["pred_short_oos"],
        pred_long_oos=artifacts["pred_long_oos"],
        weight_short=w_s,
        weight_long=w_l,
    )

    # meta.quality에 실어줄 scoring quality 임시 저장
    artifacts["_l6_quality"] = (
        {"scoring": quality} if isinstance(quality, dict) else {"scoring": quality}
    )

    return {
        "rebalance_scores": scores,
        "rebalance_scores_summary": summary,
    }, warns


def run_L7_backtest(cfg, artifacts, *, force=False):
    # 기존 프로젝트의 stages/l7_backtest.py를 사용
    import stages.l7_backtest as l7m

    # L7 config: params.l7 또는 l7 둘 다 지원
    p = cfg.get("params", {}) or {}
    l7 = p.get("l7", {}) if isinstance(p.get("l7", {}), dict) else {}
    if not l7:
        l7 = cfg.get("l7", {}) or {}

    # (가능한 범용 호출)
    rebalance_scores = artifacts["rebalance_scores"]
    rebalance_scores_summary = artifacts.get("rebalance_scores_summary")

    # 후보 함수명들(프로젝트 코드 변화에 대비)
    candidates = [
        "build_backtest_outputs",
        "run_backtest",
        "backtest_from_scores",
        "build_backtest",
    ]
    fn = None
    for name in candidates:
        if hasattr(l7m, name):
            fn = getattr(l7m, name)
            break
    if fn is None:
        raise AttributeError(f"[L7] stages.l7_backtest missing any of {candidates}")

    # 호출 시도(키워드/포지셔널 둘 다 대응)
    try:
        out = fn(
            rebalance_scores=rebalance_scores,
            rebalance_scores_summary=rebalance_scores_summary,
            cfg=cfg,
            l7=l7,
        )
    except TypeError:
        out = fn(rebalance_scores, rebalance_scores_summary, cfg)

    # out 형태 표준화
    # 허용:
    # 1) (positions, returns, equity, metrics, warns, quality)
    # 2) (positions, returns, equity, metrics, warns)
    # 3) {"bt_positions":..., ...}, warns, quality (또는 warns만)
    stage_quality = {}
    warns = []

    if isinstance(out, tuple):
        if len(out) == 6:
            bt_pos, bt_ret, bt_eq, bt_met, warns, stage_quality = out
        elif len(out) == 5:
            bt_pos, bt_ret, bt_eq, bt_met, warns = out
        else:
            raise ValueError(f"[L7] unexpected tuple return length: {len(out)}")
        outputs = {
            "bt_positions": bt_pos,
            "bt_returns": bt_ret,
            "bt_equity_curve": bt_eq,
            "bt_metrics": bt_met,
        }
    elif isinstance(out, dict):
        outputs = out
    else:
        raise ValueError(f"[L7] unexpected return type: {type(out)}")

    if stage_quality:
        artifacts["_l7_quality"] = stage_quality

    return outputs, (warns or [])


def run_L7B_sensitivity(cfg, artifacts, *, force=False):
    import stages.l7b_sensitivity as m

    candidates = ["run_l7b_sensitivity", "build_sensitivity", "run", "main"]
    fn = None
    for name in candidates:
        if hasattr(m, name):
            fn = getattr(m, name)
            break
    if fn is None:
        raise AttributeError(
            f"[L7B] stages.l7b_sensitivity missing any of {candidates}"
        )

    try:
        out = fn(rebalance_scores=artifacts["rebalance_scores"], cfg=cfg)
    except TypeError:
        out = fn(artifacts["rebalance_scores"], cfg)

    if isinstance(out, tuple) and len(out) == 2:
        outputs, warns = out
    elif isinstance(out, dict):
        outputs, warns = out, []
    else:
        raise ValueError(f"[L7B] unexpected return: {type(out)}")

    return outputs, warns


def run_L7C_benchmark(cfg, artifacts, *, force=False):
    import stages.l7c_benchmark as m

    candidates = ["run_l7c_benchmark", "build_benchmark", "run", "main"]
    fn = None
    for name in candidates:
        if hasattr(m, name):
            fn = getattr(m, name)
            break
    if fn is None:
        raise AttributeError(f"[L7C] stages.l7c_benchmark missing any of {candidates}")

    try:
        out = fn(
            rebalance_scores=artifacts["rebalance_scores"],
            bt_returns=artifacts["bt_returns"],
            cfg=cfg,
        )
    except TypeError:
        out = fn(artifacts["rebalance_scores"], artifacts["bt_returns"], cfg)

    if isinstance(out, tuple) and len(out) == 2:
        outputs, warns = out
    elif isinstance(out, dict):
        outputs, warns = out, []
    else:
        raise ValueError(f"[L7C] unexpected return: {type(out)}")

    return outputs, warns


def run_L7D_stability(cfg, artifacts, *, force=False):
    # ✅ 이번에 확정한 산출물 스키마(bt_yearly_metrics 11컬럼) 생성 모듈 사용
    from src.stages.backtest.l7d_stability import run_l7d_stability_from_artifacts

    # holding_days는 L7 설정을 우선 사용(없으면 20)
    p = cfg.get("params", {}) or {}
    l7 = p.get("l7", {}) if isinstance(p.get("l7", {}), dict) else {}
    if not l7:
        l7 = cfg.get("l7", {}) or {}
    holding_days = int(l7.get("holding_days", 20))

    yearly, warns = run_l7d_stability_from_artifacts(
        bt_returns=artifacts["bt_returns"],
        holding_days=holding_days,
    )
    return {"bt_yearly_metrics": yearly}, (warns or [])


# -----------------------------
# Stage Registry
# -----------------------------
STAGES = {
    "L0": run_L0_universe,
    "L1": run_L1_base,
    "L1B": run_L1B_pykrx_fundamentals,
    "L2": run_L2_merge,
    "L3": run_L3_features,
    "L4": run_L4_split,
    "L5": run_L5_modeling,
    "L6": run_L6_scoring,
    "L7": run_L7_backtest,
    "L7B": run_L7B_sensitivity,
    "L7C": run_L7C_benchmark,
    "L7D": run_L7D_stability,
}

# stage별 필수 입력 preload
REQUIRED_INPUTS = {
    "L0": [],
    "L1": ["universe_k200_membership_monthly"],
    "L1B": ["ohlcv_daily"],
    "L2": ["universe_k200_membership_monthly"],
    "L3": ["ohlcv_daily", "fundamentals_annual"],
    "L4": ["panel_merged_daily"],
    "L5": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L6": ["pred_short_oos", "pred_long_oos"],
    "L7": ["rebalance_scores", "rebalance_scores_summary"],
    "L7B": ["rebalance_scores"],
    "L7C": ["rebalance_scores", "bt_returns"],
    "L7D": [
        "bt_returns",
        "bt_equity_curve",
    ],  # bt_equity_curve는 안정성 확장에 필요할 수 있어 preload 유지
}

# stage별 대표 output(스킵 판정용)
STAGE_OUTPUTS = {
    "L0": ["universe_k200_membership_monthly"],
    "L1": ["ohlcv_daily"],
    "L1B": ["pykrx_fundamentals_daily"],
    "L2": ["fundamentals_annual"],
    "L3": ["panel_merged_daily"],
    "L4": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L5": ["pred_short_oos", "pred_long_oos", "model_metrics"],
    "L6": ["rebalance_scores", "rebalance_scores_summary"],
    "L7": ["bt_positions", "bt_returns", "bt_equity_curve", "bt_metrics"],
    "L7D": ["bt_yearly_metrics"],
}

# output별 required cols (검증 스키마)
REQUIRED_COLS_BY_OUTPUT = {
    "universe_k200_membership_monthly": ["date", "ticker"],
    "ohlcv_daily": ["date", "ticker"],
    "fundamentals_annual": ["date", "ticker"],
    "panel_merged_daily": ["date", "ticker"],
    "dataset_daily": ["date", "ticker"],
    "cv_folds_short": [
        "fold_id",
        "segment",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ],
    "cv_folds_long": [
        "fold_id",
        "segment",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ],
    "pred_short_oos": [
        "date",
        "ticker",
        "y_true",
        "y_pred",
        "fold_id",
        "phase",
        "horizon",
    ],
    "pred_long_oos": [
        "date",
        "ticker",
        "y_true",
        "y_pred",
        "fold_id",
        "phase",
        "horizon",
    ],
    "model_metrics": ["horizon", "fold_id", "phase", "rmse"],
    "rebalance_scores": ["date", "ticker", "phase"],
    "rebalance_scores_summary": ["date", "phase", "n_tickers", "coverage_ticker_pct"],
    "bt_positions": ["date", "phase", "ticker"],
    "bt_returns": [
        "date",
        "phase",
    ],  # net_return 등은 구현에 따라 달라질 수 있어 최소만 강제
    "bt_equity_curve": ["date", "phase"],
    "bt_metrics": ["phase", "net_total_return", "net_sharpe", "net_mdd"],
    # ✅ L7D는 실제 산출물 결과(untitled35) 기준으로 11개 컬럼을 고정
    "bt_yearly_metrics": [
        "phase",
        "year",
        "n_rebalances",
        "net_total_return",
        "net_vol_ann",
        "net_sharpe",
        "net_mdd",
        "net_hit_ratio",
        "date_start",
        "date_end",
        "net_return_col_used",
    ],
}


def _preload_required_inputs(stage_name: str, interim_dir: Path, artifacts: dict):
    for name in REQUIRED_INPUTS.get(stage_name, []):
        if name in artifacts:
            continue
        base = interim_dir / name
        if artifact_exists(base):
            artifacts[name] = load_artifact(base)
            logger.info(
                f"[PRELOAD] {stage_name} <- loaded required input from interim: {name}"
            )
        else:
            raise KeyError(f"{stage_name} requires '{name}' but not found: {base}")


def _maybe_skip_stage(
    stage_name: str,
    interim_dir: Path,
    artifacts: dict,
    *,
    force: bool,
    skip_if_exists: bool,
) -> bool:
    if force or (not skip_if_exists):
        return False
    outs = STAGE_OUTPUTS.get(stage_name, [])
    if not outs:
        return False
    bases = [(o, interim_dir / o) for o in outs]
    if all(artifact_exists(b) for _, b in bases):
        for o, b in bases:
            artifacts[o] = load_artifact(b)
        logger.info(f"[SKIP] {stage_name} -> loaded from interim ({', '.join(outs)})")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Integrated Pipeline Runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--from", dest="from_stage", type=str, default="L0")
    parser.add_argument("--to", dest="to_stage", type=str, default="L7")
    parser.add_argument("--stage", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default="default_run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    stage_names = list(STAGES.keys())
    if args.stage:
        if args.stage not in STAGES:
            logger.error(f"Invalid stage: {args.stage}")
            sys.exit(1)
        target_stages = [args.stage]
    else:
        start_idx = stage_names.index(args.from_stage)
        end_idx = stage_names.index(args.to_stage)
        target_stages = stage_names[start_idx : end_idx + 1]

    logger.info(f"🚀 Start Pipeline | Run ID: {args.run_id}")
    logger.info(f"Target Stages: {target_stages}")
    logger.info(f"Config: {args.config}")

    if args.dry_run:
        logger.info("[Dry-Run] Skipping actual execution.")
        return

    interim_dir = get_path(cfg, "data_interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    save_formats = cfg.get("run", {}).get("save_formats", ["parquet", "csv"])
    fail_on_validation_error = bool(
        cfg.get("run", {}).get("fail_on_validation_error", True)
    )
    write_meta = bool(cfg.get("run", {}).get("write_meta", True))
    skip_if_exists = bool(cfg.get("run", {}).get("skip_if_exists", True))

    artifacts: dict[str, pd.DataFrame] = {}

    for stage_name in target_stages:
        try:
            _preload_required_inputs(stage_name, interim_dir, artifacts)

            if _maybe_skip_stage(
                stage_name,
                interim_dir,
                artifacts,
                force=args.force,
                skip_if_exists=skip_if_exists,
            ):
                continue

            func = STAGES[stage_name]
            outputs, stage_warnings = func(cfg, artifacts, force=args.force)

            if not isinstance(outputs, dict) or not outputs:
                raise ValueError(
                    f"{stage_name} must return dict[str, DataFrame] with at least one output."
                )

            for out_name, df in outputs.items():
                out_base = interim_dir / out_name

                required_cols = REQUIRED_COLS_BY_OUTPUT.get(out_name, None)
                result = validate_df(
                    df,
                    stage=stage_name,
                    required_cols=required_cols,
                    max_missing_pct=95.0,
                )
                all_warnings = (stage_warnings or []) + (result.warnings or [])

                if fail_on_validation_error:
                    raise_if_invalid(result, stage=f"{stage_name}:{out_name}")

                save_artifact(df, out_base, force=args.force, formats=save_formats)

                # --- quality ---
                quality = {}

                if stage_name == "L3" and out_name == "panel_merged_daily":
                    quality["fundamental"] = fundamental_coverage_report(df)

                if stage_name == "L4" and out_name == "dataset_daily":
                    quality["walkforward"] = walkforward_quality_report(
                        dataset_daily=df,
                        cv_folds_short=outputs.get(
                            "cv_folds_short", artifacts.get("cv_folds_short")
                        ),
                        cv_folds_long=outputs.get(
                            "cv_folds_long", artifacts.get("cv_folds_long")
                        ),
                        cfg=cfg,
                    )

                if stage_name == "L5":
                    if out_name == "pred_short_oos":
                        quality["model_oos"] = artifacts.get("_l5_report_short", {})
                    elif out_name == "pred_long_oos":
                        quality["model_oos"] = artifacts.get("_l5_report_long", {})

                if stage_name == "L6":
                    q = artifacts.get("_l6_quality", {})
                    if isinstance(q, dict) and q:
                        quality.update(q)

                if stage_name == "L7":
                    q = artifacts.get("_l7_quality", {})
                    if isinstance(q, dict) and q:
                        quality.update(q)

                # L7D는 산출물 자체가 안정성 보고서이므로 meta.quality는 비워도 무방
                # (원하면 여기서 quality["stability"] = {...} 형태로 확장 가능)

                if write_meta:
                    meta = build_meta(
                        stage=f"{stage_name}:{out_name}",
                        run_id=args.run_id,
                        df=df,
                        out_base_path=out_base,
                        warnings=all_warnings,
                        inputs={"prev_outputs": list(artifacts.keys())},
                        repo_dir=get_path(cfg, "base_dir"),
                        quality=quality,
                    )
                    save_meta(out_base, meta, force=True)

                artifacts[out_name] = df

        except Exception as e:
            logger.error(f"❌ Failed at {stage_name}: {e}")
            sys.exit(1)

    logger.info("✅ Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()


# END OF FILE: run_all.py


################################################################################
# START OF FILE: snapshot_l0_l7_outputs.py
################################################################################

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.config import get_path, load_config

# -----------------------------
# Config
# -----------------------------
DEFAULT_ARTIFACTS = [
    # L0~L4
    "universe_k200_membership_monthly",
    "ohlcv_daily",
    "fundamentals_annual",
    "panel_merged_daily",
    "dataset_daily",
    "cv_folds_short",
    "cv_folds_long",
    # L5
    "pred_short_oos",
    "pred_long_oos",
    "model_metrics",
    # L6
    "rebalance_scores",
    "rebalance_scores_summary",
    # L7
    "bt_positions",
    "bt_returns",
    "bt_equity_curve",
    "bt_metrics",
    # L7B/L7C/L7D extensions
    "bt_sensitivity_metrics",
    "bt_vs_benchmark",
    "bt_benchmark_returns",
    "bt_benchmark_compare",
    "bt_yearly_metrics",
    "bt_drawdown_events",
    "bt_rolling_sharpe",
]

EXPORT_EXTS = [".parquet", ".csv"]


@dataclass
class ArtifactRecord:
    name: str
    src_base: str
    dst_base: str
    has_parquet: bool
    has_csv: bool
    has_meta: bool
    parquet_bytes: int
    csv_bytes: int
    meta_bytes: int
    meta_stage: str
    meta_run_id: str
    meta_n_rows: int
    meta_n_cols: int


def _root() -> Path:
    # .../03_code/src/stages/snapshot_l0_l7_outputs.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]


def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _discover_artifacts_from_meta(interim: Path) -> list[str]:
    names = []
    for mp in sorted(interim.glob("*__meta.json")):
        # e.g., pred_short_oos__meta.json -> pred_short_oos
        stem = mp.name.replace("__meta.json", "")
        if stem:
            names.append(stem)
    return sorted(set(names))


def _file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _parse_meta(meta_path: Path) -> tuple[str, str, int, int]:
    """
    meta JSON 구조는 utils.meta.build_meta() 결과를 따른다고 가정.
    최소한 stage/run_id/df_shape(혹은 n_rows/n_cols)를 안전하게 읽는다.
    """
    m = _read_json(meta_path)

    stage = str(m.get("stage", ""))
    run_id = str(m.get("run_id", ""))

    # build_meta에서 df info가 어떤 키로 저장되든, 아래 순서로 우선 탐색
    n_rows = -1
    n_cols = -1

    # 1) df_shape
    if isinstance(m.get("df_shape", None), (list, tuple)) and len(m["df_shape"]) == 2:
        n_rows = int(m["df_shape"][0])
        n_cols = int(m["df_shape"][1])

    # 2) df / summary 내부
    if (n_rows < 0 or n_cols < 0) and isinstance(m.get("df", None), dict):
        d = m["df"]
        if "n_rows" in d:
            n_rows = int(d["n_rows"])
        if "n_cols" in d:
            n_cols = int(d["n_cols"])

    # 3) fallback
    if n_rows < 0:
        n_rows = int(m.get("n_rows", -1))
    if n_cols < 0:
        n_cols = int(m.get("n_cols", -1))

    return stage, run_id, n_rows, n_cols


def snapshot(
    *,
    root: Path,
    interim: Path,
    out_dir: Path,
    include_discovered: bool,
) -> list[ArtifactRecord]:
    _safe_mkdir(out_dir)

    # export 대상 artifact 목록 확정
    names = list(DEFAULT_ARTIFACTS)
    if include_discovered:
        names += _discover_artifacts_from_meta(interim)
    names = sorted(set(names))

    records: list[ArtifactRecord] = []

    for name in names:
        src_base = interim / name
        dst_base = out_dir / name

        src_parquet = src_base.with_suffix(".parquet")
        src_csv = src_base.with_suffix(".csv")
        src_meta = interim / f"{name}__meta.json"

        dst_parquet = dst_base.with_suffix(".parquet")
        dst_csv = dst_base.with_suffix(".csv")
        dst_meta = out_dir / f"{name}__meta.json"

        has_parquet = _copy_if_exists(src_parquet, dst_parquet)
        has_csv = _copy_if_exists(src_csv, dst_csv)
        has_meta = _copy_if_exists(src_meta, dst_meta)

        parquet_bytes = _file_size(dst_parquet) if has_parquet else 0
        csv_bytes = _file_size(dst_csv) if has_csv else 0
        meta_bytes = _file_size(dst_meta) if has_meta else 0

        meta_stage = ""
        meta_run_id = ""
        meta_n_rows = -1
        meta_n_cols = -1
        if has_meta:
            meta_stage, meta_run_id, meta_n_rows, meta_n_cols = _parse_meta(dst_meta)

        # parquet/csv/meta 중 하나도 없으면 기록은 남기되, 존재 여부로 확인 가능하게 한다.
        rec = ArtifactRecord(
            name=name,
            src_base=str(src_base),
            dst_base=str(dst_base),
            has_parquet=has_parquet,
            has_csv=has_csv,
            has_meta=has_meta,
            parquet_bytes=parquet_bytes,
            csv_bytes=csv_bytes,
            meta_bytes=meta_bytes,
            meta_stage=meta_stage,
            meta_run_id=meta_run_id,
            meta_n_rows=meta_n_rows,
            meta_n_cols=meta_n_cols,
        )
        records.append(rec)

    return records


def write_manifest(out_dir: Path, records: list[ArtifactRecord]) -> None:
    # CSV
    manifest_csv = out_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "has_parquet",
                "has_csv",
                "has_meta",
                "parquet_bytes",
                "csv_bytes",
                "meta_bytes",
                "meta_stage",
                "meta_run_id",
                "meta_n_rows",
                "meta_n_cols",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.name,
                    int(r.has_parquet),
                    int(r.has_csv),
                    int(r.has_meta),
                    r.parquet_bytes,
                    r.csv_bytes,
                    r.meta_bytes,
                    r.meta_stage,
                    r.meta_run_id,
                    r.meta_n_rows,
                    r.meta_n_cols,
                ]
            )

    # JSON
    manifest_json = out_dir / "manifest.json"
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": [r.__dict__ for r in records],
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # README
    readme = out_dir / "README.txt"
    n_total = len(records)
    n_ok = sum(1 for r in records if r.has_parquet and r.has_csv and r.has_meta)
    with readme.open("w", encoding="utf-8") as f:
        f.write("Snapshot created.\n")
        f.write(f"- total records: {n_total}\n")
        f.write(f"- fully packaged (parquet+csv+meta): {n_ok}\n")
        f.write("- files:\n")
        f.write("  - *.parquet / *.csv per artifact (if existed in interim)\n")
        f.write("  - *__meta.json per artifact (if existed in interim)\n")
        f.write("  - manifest.csv / manifest.json\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--include-discovered", action="store_true")
    args = parser.parse_args()

    root = _root()
    cfg_path = _cfg_path(root)
    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")

    tag = args.tag.strip()
    if not tag:
        tag = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")

    out_dir = root / "data" / "snapshots" / tag
    _safe_mkdir(out_dir)

    print("=== SNAPSHOT RUNNER ===")
    print("ROOT  :", root)
    print("CFG   :", cfg_path)
    print("INTERIM:", interim)
    print("OUT   :", out_dir)
    print("include_discovered:", bool(args.include_discovered))

    # config도 같이 복사(재현성)
    _copy_if_exists(cfg_path, out_dir / "config.yaml")

    records = snapshot(
        root=root,
        interim=Path(interim),
        out_dir=out_dir,
        include_discovered=bool(args.include_discovered),
    )
    write_manifest(out_dir, records)

    # 요약 출력
    n_total = len(records)
    n_meta = sum(1 for r in records if r.has_meta)
    n_parq = sum(1 for r in records if r.has_parquet)
    n_csv = sum(1 for r in records if r.has_csv)
    n_full = sum(1 for r in records if r.has_parquet and r.has_csv and r.has_meta)

    print("\n=== SUMMARY ===")
    print(f"records: {n_total}")
    print(f"has_meta: {n_meta} / has_parquet: {n_parq} / has_csv: {n_csv}")
    print(f"fully packaged (parquet+csv+meta): {n_full}")
    print("✅ Snapshot completed.")


if __name__ == "__main__":
    main()


# END OF FILE: snapshot_l0_l7_outputs.py


################################################################################
# START OF FILE: validate_l5_outputs.py
################################################################################

# src/stages/validate_l5_outputs.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------
# Path bootstrap (Spyder/Windows 안정화)
# ----------------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # .../03_code
SRC = THIS.parents[1]  # .../03_code/src
CFG_PATH = ROOT / "configs" / "config.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact


# ----------------------------
# Helpers
# ----------------------------
def _must_exist(base: Path, name: str) -> None:
    if not artifact_exists(base):
        raise FileNotFoundError(
            f"[FAIL] artifact missing: {name} -> {base}(.parquet/.csv)"
        )


def _load_df(interim: Path, name: str) -> pd.DataFrame:
    base = interim / name
    _must_exist(base, name)
    df = load_artifact(base)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"[FAIL] artifact empty or not DataFrame: {name}")
    return df


def _load_meta(interim: Path, name: str) -> dict:
    meta_path = interim / f"{name}__meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[FAIL] meta missing: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"[FAIL] missing column: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="raise")


def _ensure_ticker_str(df: pd.DataFrame) -> None:
    if "ticker" not in df.columns:
        raise KeyError("[FAIL] missing column: ticker")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)


def _basic_checks(df: pd.DataFrame, name: str) -> None:
    for c in ["date", "ticker"]:
        if c not in df.columns:
            raise KeyError(f"[FAIL] {name} missing column: {c}")
    _ensure_datetime(df, "date")
    _ensure_ticker_str(df)

    dup = int(df.duplicated(subset=["date", "ticker"]).sum())
    if dup != 0:
        raise ValueError(f"[FAIL] {name} has duplicate (date,ticker) keys: {dup}")

    if df["date"].isna().any():
        raise ValueError(f"[FAIL] {name} has NaT in date")
    if df["ticker"].isna().any():
        raise ValueError(f"[FAIL] {name} has NA in ticker")


def _pick_pred_col(df: pd.DataFrame) -> str:
    candidates = ["y_pred", "pred", "prediction", "yhat"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"[FAIL] prediction column not found. existing={list(df.columns)}")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "hit_ratio": np.nan,
        }

    yy = y[mask]
    pp = p[mask]
    rmse = float(np.sqrt(np.mean((pp - yy) ** 2)))
    mae = float(np.mean(np.abs(pp - yy)))
    corr = float(np.corrcoef(pp, yy)[0, 1]) if len(yy) > 1 else np.nan
    hit = float(np.mean((pp > 0) == (yy > 0)))
    return {
        "n": int(mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "hit_ratio": hit,
    }


def _merged_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = sorted(intervals, key=lambda x: x[0])
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in intervals:
        if not out:
            out.append((s, e))
            continue
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _mask_by_intervals(
    dates: pd.Series, merged: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for s, e in merged:
        m |= (dates >= s) & (dates <= e)
    return m


def _attach_target_from_dataset(
    pred: pd.DataFrame, dataset: pd.DataFrame, target_col: str, *, name: str
) -> pd.DataFrame:
    """
    pred에 target_col이 없으면 dataset_daily에서 (date,ticker)로 조인해 붙인다.
    (검증 용도이며, 누락률/성능 계산을 위해 필요)
    """
    if target_col in pred.columns:
        return pred

    if target_col not in dataset.columns:
        raise KeyError(f"[FAIL] dataset_daily missing target col: {target_col}")

    # 키 중복 방지(앞에서 pred는 이미 체크됨)
    dkey = dataset[["date", "ticker", target_col]].copy()
    _ensure_datetime(dkey, "date")
    _ensure_ticker_str(dkey)
    dup = int(dkey.duplicated(subset=["date", "ticker"]).sum())
    if dup != 0:
        raise ValueError(
            f"[FAIL] dataset_daily has duplicate (date,ticker) keys: {dup} (cannot attach targets safely)"
        )

    merged = pred.merge(dkey, on=["date", "ticker"], how="left", validate="one_to_one")
    print(
        f"[INFO] {name}: target '{target_col}' attached from dataset_daily (missing before attach = {merged[target_col].isna().mean()*100:.4f}%)"
    )
    return merged


def _coverage_against_folds(
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    cv_folds: pd.DataFrame,
    target_col: str,
) -> dict:
    _ensure_datetime(dataset, "date")
    _ensure_ticker_str(dataset)

    if target_col not in dataset.columns:
        raise KeyError(f"[FAIL] dataset_daily missing target col: {target_col}")

    need_cols = ["test_start", "test_end"]
    for c in need_cols:
        if c not in cv_folds.columns:
            raise KeyError(f"[FAIL] cv_folds missing column: {c}")

    starts = pd.to_datetime(cv_folds["test_start"], errors="raise")
    ends = pd.to_datetime(cv_folds["test_end"], errors="raise")
    intervals = [(s, e) for s, e in zip(starts.tolist(), ends.tolist())]
    merged_intv = _merged_intervals(intervals)

    eligible_mask = (
        _mask_by_intervals(dataset["date"], merged_intv) & dataset[target_col].notna()
    )
    eligible = dataset.loc[eligible_mask, ["date", "ticker"]].drop_duplicates()

    pred_keys = pred.loc[pred[target_col].notna(), ["date", "ticker"]].drop_duplicates()

    pred_outside = ~_mask_by_intervals(pred["date"], merged_intv)
    outside_cnt = int(pred_outside.sum())
    if outside_cnt != 0:
        raise ValueError(
            f"[FAIL] pred_oos contains {outside_cnt} rows outside test windows (leakage or bad slicing)."
        )

    cov = (len(pred_keys) / len(eligible)) * 100.0 if len(eligible) > 0 else np.nan
    return {
        "eligible_rows": int(len(eligible)),
        "pred_rows": int(len(pred_keys)),
        "coverage_pct": float(round(cov, 4)) if np.isfinite(cov) else np.nan,
        "folds": int(len(cv_folds)),
        "test_date_min": str(starts.min().date()),
        "test_date_max": str(ends.max().date()),
    }


def _fold_level_report(
    pred: pd.DataFrame, pred_col: str, true_col: str
) -> pd.DataFrame:
    if "fold_id" not in pred.columns:
        y = pd.to_numeric(pred[true_col], errors="coerce").to_numpy()
        p = pd.to_numeric(pred[pred_col], errors="coerce").to_numpy()
        m = _metrics(y, p)
        return pd.DataFrame([{"fold_id": "ALL", **m}])

    rows = []
    for fid, g in pred.groupby("fold_id", sort=False):
        y = pd.to_numeric(g[true_col], errors="coerce").to_numpy()
        p = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        m = _metrics(y, p)
        rows.append({"fold_id": str(fid), **m})
    return pd.DataFrame(rows)


def main():
    print("=== L5 Validation Runner ===")
    print("ROOT:", ROOT)
    print("CFG :", CFG_PATH)

    if not CFG_PATH.exists():
        raise FileNotFoundError(f"[FAIL] config.yaml not found: {CFG_PATH}")

    cfg = load_config(str(CFG_PATH))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    # --- Load artifacts
    ds = _load_df(interim, "dataset_daily")
    cv_s = _load_df(interim, "cv_folds_short")
    cv_l = _load_df(interim, "cv_folds_long")
    ps = _load_df(interim, "pred_short_oos")
    pl = _load_df(interim, "pred_long_oos")
    mm = _load_df(interim, "model_metrics")

    # --- Basic checks
    _basic_checks(ps, "pred_short_oos")
    _basic_checks(pl, "pred_long_oos")

    # --- Print columns (빠른 실체 확인)
    print("\n=== Columns snapshot ===")
    print("pred_short_oos cols:", list(ps.columns))
    print("pred_long_oos  cols:", list(pl.columns))
    print("dataset_daily  cols(head 30):", list(ds.columns)[:30])
    print("cv_folds_short cols:", list(cv_s.columns))
    print("cv_folds_long  cols:", list(cv_l.columns))

    # --- Determine horizon from cv_folds (single-valued)
    hs = int(pd.to_numeric(cv_s["horizon_days"], errors="raise").iloc[0])
    hl = int(pd.to_numeric(cv_l["horizon_days"], errors="raise").iloc[0])

    t_s = f"ret_fwd_{hs}d"
    t_l = f"ret_fwd_{hl}d"

    pred_col_s = _pick_pred_col(ps)
    pred_col_l = _pick_pred_col(pl)

    # --- Attach targets if missing
    ps = _attach_target_from_dataset(ps, ds, t_s, name="pred_short_oos")
    pl = _attach_target_from_dataset(pl, ds, t_l, name="pred_long_oos")

    # --- Missingness
    miss_s = float(round(ps[[t_s, pred_col_s]].isna().any(axis=1).mean() * 100, 6))
    miss_l = float(round(pl[[t_l, pred_col_l]].isna().any(axis=1).mean() * 100, 6))

    # --- Overall metrics
    ms = _metrics(
        pd.to_numeric(ps[t_s], errors="coerce").to_numpy(),
        pd.to_numeric(ps[pred_col_s], errors="coerce").to_numpy(),
    )
    ml = _metrics(
        pd.to_numeric(pl[t_l], errors="coerce").to_numpy(),
        pd.to_numeric(pl[pred_col_l], errors="coerce").to_numpy(),
    )

    # --- Coverage vs folds (no leakage)
    cov_s = _coverage_against_folds(ds, ps, cv_s, t_s)
    cov_l = _coverage_against_folds(ds, pl, cv_l, t_l)

    # --- Fold-level sample report
    rep_s = _fold_level_report(ps, pred_col_s, t_s).sort_values("fold_id").head(5)
    rep_l = _fold_level_report(pl, pred_col_l, t_l).sort_values("fold_id").head(5)

    # --- Meta check
    meta_ps = _load_meta(interim, "pred_short_oos")
    meta_pl = _load_meta(interim, "pred_long_oos")
    meta_mm = _load_meta(interim, "model_metrics")

    q_ps = meta_ps.get("quality", {})
    q_pl = meta_pl.get("quality", {})
    q_mm = meta_mm.get("quality", {})

    print("\n=== [PASS] L5 artifacts loaded and basic checks ok ===")
    print(
        f"- pred_short_oos rows={len(ps):,} cols={ps.shape[1]} pred_col={pred_col_s} target={t_s}"
    )
    print(
        f"- pred_long_oos  rows={len(pl):,} cols={pl.shape[1]} pred_col={pred_col_l} target={t_l}"
    )
    print(f"- model_metrics  rows={len(mm):,} cols={mm.shape[1]}")

    print("\n=== Missingness (row has any NA in [target,pred]) ===")
    print(f"- short: {miss_s}%")
    print(f"- long : {miss_l}%")

    print("\n=== Overall OOS Metrics ===")
    print(
        f"- short: n={ms['n']:,} rmse={ms['rmse']:.6f} mae={ms['mae']:.6f} corr={ms['corr']:.6f} hit={ms['hit_ratio']:.6f}"
    )
    print(
        f"- long : n={ml['n']:,} rmse={ml['rmse']:.6f} mae={ml['mae']:.6f} corr={ml['corr']:.6f} hit={ml['hit_ratio']:.6f}"
    )

    print("\n=== Coverage vs Eligible rows (test windows ∩ target notna) ===")
    print(
        f"- short: coverage={cov_s['coverage_pct']}% pred={cov_s['pred_rows']:,} eligible={cov_s['eligible_rows']:,} folds={cov_s['folds']} "
        f"test_range=[{cov_s['test_date_min']} ~ {cov_s['test_date_max']}]"
    )
    print(
        f"- long : coverage={cov_l['coverage_pct']}% pred={cov_l['pred_rows']:,} eligible={cov_l['eligible_rows']:,} folds={cov_l['folds']} "
        f"test_range=[{cov_l['test_date_min']} ~ {cov_l['test_date_max']}]"
    )

    print("\n=== Fold-level sample (first 5 rows) ===")
    print("[short]\n", rep_s.to_string(index=False))
    print("[long]\n", rep_l.to_string(index=False))

    print("\n=== Meta quality keys (existence only) ===")
    print("- pred_short_oos meta quality keys:", list(q_ps.keys()))
    print("- pred_long_oos  meta quality keys:", list(q_pl.keys()))
    print("- model_metrics  meta quality keys:", list(q_mm.keys()))

    print("\n✅ L5 VALIDATION COMPLETE: All critical checks passed.")
    print("➡️ Next: proceed to L6 (scoring / ranking / rebalance inputs).")


if __name__ == "__main__":
    main()


# END OF FILE: validate_l5_outputs.py


################################################################################
# START OF FILE: validate_l6_outputs.py
################################################################################

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _fail(msg: str) -> None:
    raise SystemExit(msg)


def _ensure_datetime(s: pd.Series, name: str) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="raise")
    return s


def _read_meta(interim: Path, artifact_name: str) -> dict[str, Any]:
    meta_path = interim / f"{artifact_name}__meta.json"
    if not meta_path.exists():
        _fail(f"[FAIL] meta file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_l6_weights(cfg: dict) -> tuple[float, float]:
    p = cfg.get("params", {}) or {}
    l6 = p.get("l6", {})
    if not isinstance(l6, dict):
        l6 = {}
    w_s = float(l6.get("weight_short", 0.5))
    w_l = float(l6.get("weight_long", 0.5))
    if w_s < 0 or w_l < 0 or (w_s + w_l) <= 0:
        _fail(f"[FAIL] invalid L6 weights: weight_short={w_s}, weight_long={w_l}")
    s = w_s + w_l
    return (w_s / s, w_l / s)


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@dataclass(frozen=True)
class L6CheckResult:
    ok: bool
    messages: list[str]


def _validate_fold_windows_nonoverlap(cv: pd.DataFrame, *, name: str) -> None:
    # segment(phase)별로 test window가 서로 겹치지 않는지 확인(겹치면 fold_id->date 검증이 애매해짐)
    for seg, g in cv.groupby("segment", sort=False):
        gg = g.sort_values(["test_start", "test_end"]).reset_index(drop=True)
        # 바로 이전 test_end 다음날 이후에 시작해야(동일 구간 중복 방지)
        prev_end = None
        for i, row in gg.iterrows():
            ts = row["test_start"]
            te = row["test_end"]
            if prev_end is not None and ts <= prev_end:
                _fail(
                    f"[FAIL] {name} has overlapping test windows in segment='{seg}'. "
                    f"prev_end={prev_end.date()} cur_start={ts.date()} fold_id={row['fold_id']}"
                )
            prev_end = te


def _fold_membership_check(
    scores: pd.DataFrame,
    cv: pd.DataFrame,
    *,
    fold_col: str,
    score_present_mask: pd.Series,
    name: str,
) -> None:
    """
    scores의 (date, phase, fold_id_*)가 cv_folds의 (segment, fold_id, test_start~test_end)에
    논리적으로 부합하는지 확인한다.
    - score_present_mask=True인 행만 검증 대상으로 삼는다(예: long 없는 구간 제외)
    """
    # 필요한 컬럼만
    s = scores.loc[score_present_mask, ["date", "phase", fold_col]].copy()
    s = s.dropna(subset=[fold_col])
    s[fold_col] = s[fold_col].astype(str)

    cv2 = cv.copy()
    cv2["test_start"] = _ensure_datetime(cv2["test_start"], f"{name}.test_start")
    cv2["test_end"] = _ensure_datetime(cv2["test_end"], f"{name}.test_end")
    cv2["segment"] = cv2["segment"].astype(str)
    cv2["fold_id"] = cv2["fold_id"].astype(str)

    # fold_id + phase(segment)로 붙이고, date가 test window 안인지 확인
    m = s.merge(
        cv2[["fold_id", "segment", "test_start", "test_end"]],
        left_on=[fold_col, "phase"],
        right_on=["fold_id", "segment"],
        how="left",
        validate="many_to_one",
    )

    if m["test_start"].isna().any():
        bad = m.loc[m["test_start"].isna(), ["date", "phase", fold_col]].head(20)
        _fail(
            f"[FAIL] {name}: fold_id not found in cv_folds for given phase(segment). sample:\n{bad}"
        )

    in_window = (m["date"] >= m["test_start"]) & (m["date"] <= m["test_end"])
    bad_cnt = int((~in_window).sum())
    if bad_cnt > 0:
        bad = m.loc[
            ~in_window, ["date", "phase", fold_col, "test_start", "test_end"]
        ].head(20)
        _fail(
            f"[FAIL] {name}: date not in the test window of its fold_id. sample:\n{bad}"
        )


def validate_l6_outputs(cfg_path: Path) -> L6CheckResult:
    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")

    msgs: list[str] = []
    msgs.append("=== L6 Validation Runner ===")
    msgs.append(f"ROOT  : {_root_dir()}")
    msgs.append(f"CFG   : {cfg_path}")
    msgs.append(f"INTERIM: {interim}")

    required = [
        "rebalance_scores",
        "rebalance_scores_summary",
        "cv_folds_short",
        "cv_folds_long",
    ]
    for n in required:
        if not artifact_exists(interim / n):
            _fail(f"[FAIL] missing artifact: {n} at {interim / n}")

    scores = load_artifact(interim / "rebalance_scores")
    summary = load_artifact(interim / "rebalance_scores_summary")
    cv_s = load_artifact(interim / "cv_folds_short")
    cv_l = load_artifact(interim / "cv_folds_long")

    meta_scores = _read_meta(interim, "rebalance_scores")
    meta_summary = _read_meta(interim, "rebalance_scores_summary")

    msgs.append("\n=== Meta check ===")
    msgs.append(f"- rebalance_scores meta.stage: {meta_scores.get('stage')}")
    msgs.append(
        f"- rebalance_scores meta.quality keys: {list((meta_scores.get('quality') or {}).keys())}"
    )
    msgs.append(f"- rebalance_scores_summary meta.stage: {meta_summary.get('stage')}")
    msgs.append(
        f"- rebalance_scores_summary meta.quality keys: {list((meta_summary.get('quality') or {}).keys())}"
    )

    if "scoring" not in (meta_scores.get("quality") or {}):
        _fail("[FAIL] rebalance_scores meta.quality missing key 'scoring'")
    if "scoring" not in (meta_summary.get("quality") or {}):
        _fail("[FAIL] rebalance_scores_summary meta.quality missing key 'scoring'")

    msgs.append("\n=== Schema check ===")
    for c in ["date", "ticker", "phase"]:
        if c not in scores.columns:
            _fail(f"[FAIL] rebalance_scores missing required col: {c}")
    for c in ["date", "phase", "n_tickers", "coverage_ticker_pct"]:
        if c not in summary.columns:
            _fail(f"[FAIL] rebalance_scores_summary missing required col: {c}")
    for c in ["fold_id", "segment", "test_start", "test_end"]:
        if c not in cv_s.columns:
            _fail(f"[FAIL] cv_folds_short missing required col: {c}")
        if c not in cv_l.columns:
            _fail(f"[FAIL] cv_folds_long missing required col: {c}")

    scores = scores.copy()
    summary = summary.copy()
    scores["date"] = _ensure_datetime(scores["date"], "scores.date")
    summary["date"] = _ensure_datetime(summary["date"], "summary.date")
    scores["ticker"] = scores["ticker"].astype(str)
    scores["phase"] = scores["phase"].astype(str)
    summary["phase"] = summary["phase"].astype(str)

    msgs.append("\n=== Duplicate check (date,ticker,phase) ===")
    dup = scores.duplicated(["date", "ticker", "phase"], keep=False)
    if int(dup.sum()) > 0:
        sample = scores.loc[dup, ["date", "ticker", "phase"]].head(20)
        _fail(f"[FAIL] duplicates detected in rebalance_scores. sample:\n{sample}")

    score_short_col = _detect_col(scores, ["score_short"])
    score_long_col = _detect_col(scores, ["score_long"])
    score_ens_col = _detect_col(scores, ["score_ens", "score_ensemble", "score"])
    if score_short_col is None:
        _fail("[FAIL] rebalance_scores missing 'score_short'")
    if score_long_col is None:
        _fail("[FAIL] rebalance_scores missing 'score_long'")
    if score_ens_col is None:
        _fail(
            "[FAIL] rebalance_scores missing ensemble score col (score_ens/score_ensemble/score)"
        )

    # fold cols (merge suffix 대응)
    fold_short = _detect_col(scores, ["fold_id_short", "fold_id_x", "fold_id"])
    fold_long = _detect_col(scores, ["fold_id_long", "fold_id_y"])
    if fold_short is None:
        _fail(
            "[FAIL] rebalance_scores missing short fold id col (fold_id_short/fold_id_x/fold_id)"
        )
    # long은 없는 기간이 있으므로 컬럼 자체는 있어야 한다
    if fold_long is None:
        _fail(
            "[FAIL] rebalance_scores missing long fold id col (fold_id_long/fold_id_y)"
        )

    s_short = pd.to_numeric(scores[score_short_col], errors="coerce")
    s_long = pd.to_numeric(scores[score_long_col], errors="coerce")
    s_ens = pd.to_numeric(scores[score_ens_col], errors="coerce")

    long_present = s_long.notna()
    long_missing = ~long_present

    # long이 없는 행에서는 fold_long도 결측이어야(정합성)
    fl = scores[fold_long].astype("string")
    bad = long_missing & fl.notna()
    if int(bad.sum()) > 0:
        sample = scores.loc[
            bad, ["date", "ticker", "phase", fold_long, score_long_col]
        ].head(20)
        _fail(f"[FAIL] long missing rows have non-null fold_id_long. sample:\n{sample}")

    msgs.append("\n=== CV windows sanity ===")
    cv_s2 = cv_s.copy()
    cv_l2 = cv_l.copy()
    cv_s2["test_start"] = _ensure_datetime(cv_s2["test_start"], "cv_s.test_start")
    cv_s2["test_end"] = _ensure_datetime(cv_s2["test_end"], "cv_s.test_end")
    cv_l2["test_start"] = _ensure_datetime(cv_l2["test_start"], "cv_l.test_start")
    cv_l2["test_end"] = _ensure_datetime(cv_l2["test_end"], "cv_l.test_end")
    _validate_fold_windows_nonoverlap(cv_s2, name="cv_folds_short")
    _validate_fold_windows_nonoverlap(cv_l2, name="cv_folds_long")

    # 핵심: fold_id equality가 아니라, 각 fold_id가 자기 test window에 date를 포함하는지 확인
    msgs.append("\n=== Fold membership check ===")
    _fold_membership_check(
        scores=scores,
        cv=cv_s2,
        fold_col=fold_short,
        score_present_mask=s_short.notna(),
        name="SHORT",
    )
    _fold_membership_check(
        scores=scores,
        cv=cv_l2,
        fold_col=fold_long,
        score_present_mask=long_present,
        name="LONG",
    )

    # Ensemble 검증(가중 평균)
    w_s, w_l = _get_l6_weights(cfg)
    msgs.append("\n=== Ensemble score check ===")
    msgs.append(f"- weights: short={w_s:.6f}, long={w_l:.6f}")

    wsum = (~s_short.isna()).astype(float) * w_s + (~s_long.isna()).astype(float) * w_l
    exp = s_short.fillna(0) * w_s + s_long.fillna(0) * w_l
    exp = exp.where(wsum > 0, np.nan) / wsum.where(wsum > 0, np.nan)

    diff = (s_ens - exp).abs()
    med = float(diff.median(skipna=True))
    p99 = float(diff.quantile(0.99))
    msgs.append(f"- |ens - expected| median={med:.10f}, p99={p99:.10f}")
    if not np.isfinite(med) or p99 > 1e-6:
        _fail(
            "[FAIL] ensemble score does not match expected weighted merge (check build_rebalance_scores)"
        )

    # Summary 정합성
    msgs.append("\n=== Coverage vs summary ===")
    calc = scores.groupby(["date", "phase"], as_index=False).agg(
        n_tickers_calc=("ticker", "nunique"),
        ens_missing=(
            score_ens_col,
            lambda x: float(pd.to_numeric(x, errors="coerce").isna().mean()),
        ),
    )
    merged = summary.merge(calc, on=["date", "phase"], how="left")
    if merged["n_tickers_calc"].isna().any():
        miss = merged.loc[merged["n_tickers_calc"].isna(), ["date", "phase"]].head(20)
        _fail(f"[FAIL] summary has (date,phase) not found in scores. sample:\n{miss}")

    diff_nt = (merged["n_tickers"] - merged["n_tickers_calc"]).abs()
    if int(diff_nt.max()) != 0:
        bad = merged.loc[
            diff_nt != 0, ["date", "phase", "n_tickers", "n_tickers_calc"]
        ].head(20)
        _fail(f"[FAIL] n_tickers mismatch between summary and scores. sample:\n{bad}")

    if (summary["coverage_ticker_pct"] < 0).any() or (
        summary["coverage_ticker_pct"] > 100
    ).any():
        bad = summary.loc[
            (summary["coverage_ticker_pct"] < 0)
            | (summary["coverage_ticker_pct"] > 100)
        ].head(20)
        _fail(f"[FAIL] coverage_ticker_pct out of range [0,100]. sample:\n{bad}")

    msgs.append("\n✅ L6 VALIDATION COMPLETE: All critical checks passed.")
    msgs.append("➡️ Next: proceed to L7 (backtest / portfolio construction).")
    return L6CheckResult(ok=True, messages=msgs)


def main():
    root = _root_dir()
    cfg_path = root / "configs" / "config.yaml"
    res = validate_l6_outputs(cfg_path)
    for m in res.messages:
        print(m)


if __name__ == "__main__":
    main()


# END OF FILE: validate_l6_outputs.py


################################################################################
# START OF FILE: validate_l7_outputs.py
################################################################################

# src/stages/validate_l7_outputs.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


# END OF FILE: validate_l7_outputs.py


################################################################################
# START OF FILE: validate_l7b_l7c_l7d_outputs.py
################################################################################

# src/stages/validate_l7b_l7c_l7d_outputs.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cfg_path(root: Path) -> Path:
    return root / "configs" / "config.yaml"


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _artifact_base(interim: Path, name: str) -> Path:
    return interim / name


def _fail(msg: str) -> None:
    raise SystemExit(msg)


def _check_numeric_finite(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if (~np.isfinite(s.to_numpy())).any():
            _fail(f"[FAIL] {label}: non-finite values found in numeric col '{c}'")


def _check_dupes(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if not all(k in df.columns for k in keys):
        _fail(f"[FAIL] {label}: duplicate key columns not found: {keys}")
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(10)
        _fail(f"[FAIL] {label}: duplicates on keys={keys}. sample:\n{sample}")


def _infer_rolling_keys(df: pd.DataFrame) -> list[str]:
    """
    bt_rolling_sharpe 같은 다차원 결과는 (date, phase)만으로 유니크가 아닐 수 있으므로,
    window/series 계열 컬럼을 찾아 유니크 키에 포함한다.
    """
    base = ["date", "phase"]

    candidates = [
        "window_days",
        "window",
        "lookback_days",
        "lookback",
        "rolling_window",
        "window_months",
        "series",
        "kind",
        "metric",
        "return_col",
        "return_col_used",
        "net_return_col_used",
    ]

    extras = [c for c in candidates if c in df.columns]

    # 최소 1개 이상은 있어야 (date,phase) 중복이 정당화됨.
    if len(extras) == 0:
        return base

    # extras 전부 포함(보수적으로 유니크 보장)
    return base + extras


def main():
    print("=== L7B/L7C/L7D Validation Runner ===")
    root = _root()
    cfg_path = _cfg_path(root)

    print("ROOT  :", root)
    print("CFG   :", cfg_path)

    if not cfg_path.exists():
        _fail(f"[FAIL] config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    meta_files = sorted(interim.glob("*__meta.json"))
    if not meta_files:
        _fail(f"[FAIL] no meta files found in: {interim}")

    discovered = []
    for mp in meta_files:
        m = _load_meta(mp)
        stage = str(m.get("stage", ""))
        if (
            stage.startswith("L7B:")
            or stage.startswith("L7C:")
            or stage.startswith("L7D:")
        ):
            out_name = mp.name.replace("__meta.json", "")
            discovered.append((out_name, stage, mp))

    if not discovered:
        _fail("[FAIL] no L7B/L7C/L7D meta found. did you run those stages?")

    print("\n=== Discovered outputs (from meta.stage) ===")
    for out_name, stage, mp in discovered:
        print(f"- {stage} -> {out_name} (meta={mp.name})")

    # L7D yearly metrics는 스키마를 엄격히 고정
    strict_required = {
        "bt_yearly_metrics": [
            "phase",
            "year",
            "n_rebalances",
            "net_total_return",
            "net_vol_ann",
            "net_sharpe",
            "net_mdd",
            "net_hit_ratio",
            "date_start",
            "date_end",
            "net_return_col_used",
        ]
    }

    print("\n=== Artifact existence & schema checks ===")
    for out_name, stage, mp in discovered:
        base = _artifact_base(interim, out_name)
        if not artifact_exists(base):
            _fail(f"[FAIL] artifact missing for {stage}: {base}")

        df = load_artifact(base)
        if not isinstance(df, pd.DataFrame):
            _fail(f"[FAIL] {stage}:{out_name} is not a DataFrame")
        if df.shape[0] == 0:
            _fail(f"[FAIL] {stage}:{out_name} has 0 rows")

        # strict schema check
        if out_name in strict_required:
            need = strict_required[out_name]
            miss = [c for c in need if c not in df.columns]
            if miss:
                _fail(
                    f"[FAIL] {stage}:{out_name} missing required cols: {miss}\n"
                    f"cols={list(df.columns)}"
                )

        # date parsing sanity
        for dc in ["date", "date_start", "date_end"]:
            if dc in df.columns:
                d = pd.to_datetime(df[dc], errors="coerce")
                if d.isna().any():
                    _fail(
                        f"[FAIL] {stage}:{out_name} has invalid '{dc}' values (NaT present)"
                    )

        # duplicate checks (output별로 키를 다르게 적용)
        if out_name == "bt_rolling_sharpe":
            keys = _infer_rolling_keys(df)
            if keys == ["date", "phase"]:
                # window/series 계열 컬럼이 없는데 (date,phase) 중복이면 진짜 문제로 취급
                _check_dupes(df, ["date", "phase"], f"{stage}:{out_name}")
            else:
                _check_dupes(df, keys, f"{stage}:{out_name}")

        else:
            # 일반 규칙: 가능하면 가장 세밀한 키로 검증
            if all(c in df.columns for c in ["date", "ticker", "phase"]):
                _check_dupes(df, ["date", "ticker", "phase"], f"{stage}:{out_name}")
            elif all(c in df.columns for c in ["date", "phase"]):
                _check_dupes(df, ["date", "phase"], f"{stage}:{out_name}")
            elif all(c in df.columns for c in ["phase", "year"]):
                _check_dupes(df, ["phase", "year"], f"{stage}:{out_name}")

        # numeric finite check
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        _check_numeric_finite(df, numeric_cols, f"{stage}:{out_name}")

        print(f"[PASS] {stage}:{out_name} shape={df.shape}")

    if artifact_exists(interim / "bt_yearly_metrics"):
        y = load_artifact(interim / "bt_yearly_metrics")
        years = sorted(pd.Series(y["year"]).dropna().astype(int).unique().tolist())
        print("\n=== Yearly metrics summary ===")
        print("years:", years)
        print("phases:", sorted(pd.Series(y["phase"]).astype(str).unique().tolist()))

    print("\n✅ L7B/L7C/L7D VALIDATION COMPLETE: All critical checks passed.")
    print(
        "➡️ Next: run full audit (L0~L7 + extensions) and then final reporting tables."
    )


if __name__ == "__main__":
    main()


# END OF FILE: validate_l7b_l7c_l7d_outputs.py


################################################################################
# START OF FILE: 결과값확인코드.py
################################################################################

import os

import pandas as pd

# -----------------------------------------------------------------------------
# 1. 파일 경로 설정
# -----------------------------------------------------------------------------
# 사용자가 제공한 절대 경로 사용
file_path = r"C:\Users\seong\OneDrive\바탕 화면\bootcamp\03_code\data\snapshots\baseline_after_L7BCD\combined__baseline_after_L7BCD.parquet"

print(f"📂 파일 로딩 중: {file_path}")

try:
    # 2. 통합 파일 로드
    df = pd.read_parquet(file_path)
    print(f"✅ 로드 완료! 데이터 크기: {df.shape}")

    # 3. 포함된 아티팩트(산출물) 목록 확인
    # '__artifact' 컬럼이 각 행이 어떤 데이터인지 알려주는 '이름표' 역할을 합니다.
    artifacts = df["__artifact"].unique()
    print(f"📋 포함된 산출물 목록: {artifacts}")
    print("-" * 60)

    # -----------------------------------------------------------------------------
    # 4. 핵심 데이터 추출 및 분석 함수
    # -----------------------------------------------------------------------------
    def analyze_artifact(target_name, description):
        # 해당 아티팩트만 필터링
        subset = df[df["__artifact"] == target_name].copy()

        if subset.empty:
            return  # 해당 아티팩트가 없으면 패스

        # 해당 데이터에서 '모두 비어있는(NaN)' 컬럼은 제거 (보기 좋게)
        subset = subset.dropna(axis=1, how="all")

        print(f"\n🔎 [{target_name}] - {description}")

        # (A) 성과 지표 (metrics)인 경우: 전체 통계 출력
        if "metrics" in target_name:
            # 주요 지표 컬럼만 골라서 보여주기 (너무 많으므로)
            key_metrics = [
                "net_sharpe",
                "net_cagr",
                "net_mdd",
                "avg_turnover_oneway",
                "rmse",
                "mae",
                "hit_ratio",
                "ic_rank",
                "corr_vs_benchmark",
            ]
            # 존재하는 컬럼만 선택
            cols_to_show = [c for c in key_metrics if c in subset.columns]

            if cols_to_show:
                print("   [핵심 지표 요약]")
                # 평균값 또는 첫 번째 행 출력
                print(subset[cols_to_show].mean(numeric_only=True).to_frame().T)
            else:
                print(subset.head())

        # (B) 포지션(positions)인 경우: 최근 날짜 보유 종목 샘플
        elif "positions" in target_name and "date" in subset.columns:
            last_date = subset["date"].max()
            daily_pos = subset[subset["date"] == last_date]
            print(f"   📅 최근 거래일({last_date}) 보유 종목 수: {len(daily_pos)}개")
            print("   [상위 비중 5개 종목]")
            if "weight" in daily_pos.columns and "ticker" in daily_pos.columns:
                print(
                    daily_pos.sort_values("weight", ascending=False)[
                        ["ticker", "weight"]
                    ].head(5)
                )
            else:
                print(daily_pos.head())

        # (C) 스코어(scores)인 경우: 점수 분포 확인
        elif "score" in target_name:
            print(f"   📊 스코어 데이터 ({len(subset)} rows)")
            # 점수 컬럼이 있다면 기초 통계 출력
            score_cols = [c for c in subset.columns if "score" in c]
            if score_cols:
                print(subset[score_cols].describe().loc[["mean", "std", "min", "max"]])

        # (D) 기타: 상위 3줄만 출력
        else:
            print(subset.head(3))

        print("-" * 60)

    # -----------------------------------------------------------------------------
    # 5. 순차적 분석 실행 (프로젝트 흐름순)
    # -----------------------------------------------------------------------------

    # [L5] 모델 성능 확인: 예측이 얼마나 잘 맞았는가?
    # (로그 컬럼에 'ic_rank', 'rmse'가 있는 것으로 보아 'metrics'나 'model_metrics'에 저장됨)
    # 정확한 이름은 위 artifacts 목록 출력 결과를 보고 매칭해야 하지만,
    # 통상적인 이름인 'model_metrics' 또는 'metrics'를 찾아봅니다.
    for art in artifacts:
        if "model" in art and "metrics" in art:
            analyze_artifact(art, "L5 모델 예측 성능 (RMSE, IC)")

    # [L6] 스코어링 상태 확인: 점수가 안정적인가?
    for art in artifacts:
        if "score" in art and "summary" not in art:  # raw score
            analyze_artifact(art, "L6 리밸런싱 스코어 분포")

    # [L7] 백테스트 최종 성과: 돈을 벌었는가?
    # 보통 'bt_metrics' 또는 'bt_metrics_...'
    for art in artifacts:
        if "bt" in art and "metrics" in art:
            analyze_artifact(art, "L7 백테스트 최종 성과 (Sharpe, Turnover)")

    # [L7] 포지션 확인: 무엇을 샀는가?
    for art in artifacts:
        if "bt" in art and "positions" in art:
            analyze_artifact(art, "L7 보유 포지션 내역")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")


# END OF FILE: 결과값확인코드.py
