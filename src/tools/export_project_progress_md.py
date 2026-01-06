# -*- coding: utf-8 -*-
"""
[개선안 17번] 프로젝트 진행상황 보고서(MD) 자동 생성기

목적:
  - configs/config.yaml + data/interim 산출물(.parquet) + reports/*.md를 기반으로
    현재 프로젝트 진행상황을 사람이 읽기 좋은 보고서 형태로 요약한다.

출력:
  - reports/project_progress.md

주의:
  - data/interim에는 실험 태그별 폴더가 다수 존재할 수 있다.
  - 본 스크립트는 '대표 산출물'로 (1) data/interim 루트, (2) option_a_only_20d 를 우선 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import json

import pandas as pd


def _project_root() -> Path:
    # .../src/tools/export_project_progress_md.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise ImportError("PyYAML이 필요합니다. `pip install pyyaml` 후 재실행하세요.") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_cfg(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise ValueError(f"config.yaml을 찾을 수 없습니다. 경로 확인 필요: {cfg_path}")
    return _read_yaml(cfg_path)


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass(frozen=True)
class Snapshot:
    name: str
    base_dir: Path

    def parquet(self, artifact_name: str) -> Path:
        # data/interim/<tag>/bt_metrics.parquet 형태를 우선 지원
        return self.base_dir / f"{artifact_name}.parquet"

    def meta(self, artifact_name: str) -> Path:
        # data/interim/<tag>/bt_metrics__meta.json 형태
        return self.base_dir / f"{artifact_name}__meta.json"


def _fmt_pct(x: Any, ndigits: int = 2) -> str:
    try:
        v = float(x) * 100.0
        return f"{v:.{ndigits}f}%"
    except Exception:
        return ""


def _fmt_num(x: Any, ndigits: int = 2) -> str:
    try:
        v = float(x)
        return f"{v:.{ndigits}f}"
    except Exception:
        return ""


def _pick_phase_rows(bt_metrics: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if bt_metrics is None or bt_metrics.empty:
        return out
    for _, r in bt_metrics.iterrows():
        ph = str(r.get("phase", "")).strip().lower()
        if ph:
            out[ph] = dict(r)
    return out


def _metrics_table_from_bt(bt: pd.DataFrame) -> str:
    rows = _pick_phase_rows(bt)
    # bt_metrics schema differs by experiment; handle best-effort
    def get(r: dict, key: str) -> Any:
        return r.get(key, "")

    # Prefer net_* columns if present
    has_net = "net_sharpe" in bt.columns
    if has_net:
        cols = [
            ("Net Hit Ratio", "net_hit_ratio"),
            ("Net Total Return", "net_total_return"),
            ("Net CAGR", "net_cagr"),
            ("Net Sharpe", "net_sharpe"),
            ("Net MDD", "net_mdd"),
            ("리밸런싱 횟수", "n_rebalances"),
        ]
    else:
        cols = [
            ("Net Hit Ratio", "net_hit_ratio"),
            ("Net Total Return", "net_total_return"),
            ("Net CAGR", "net_cagr"),
            ("Net Sharpe", "net_sharpe"),
            ("Net MDD", "net_mdd"),
            ("리밸런싱 횟수", "n_rebalances"),
        ]

    def format_cell(k: str, v: Any) -> str:
        if k in ("net_total_return", "gross_total_return"):
            return _fmt_pct(v, 2)
        if k in ("net_cagr", "gross_cagr"):
            return _fmt_pct(v, 2)
        if k in ("net_mdd", "gross_mdd"):
            return _fmt_pct(v, 2)
        if k.endswith("hit_ratio"):
            return _fmt_pct(v, 2)
        if k.endswith("sharpe"):
            return _fmt_num(v, 2)
        if k == "n_rebalances":
            try:
                return str(int(v))
            except Exception:
                return ""
        return _fmt_num(v, 4)

    # Markdown table
    header = "| 지표 | Dev | Holdout |\n|---|---:|---:|\n"
    lines = [header]
    for label, key in cols:
        dev = format_cell(key, get(rows.get("dev", {}), key))
        ho = format_cell(key, get(rows.get("holdout", {}), key))
        lines.append(f"| {label} | {dev} | {ho} |\n")
    return "".join(lines)


def _load_feature_lists(root: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    feature_groups.yaml / feature_groups_short.yaml / feature_groups_long.yaml에서
    등장하는 피처명을 리스트로 수집한다.
    """
    cfg_dir = root / "configs"
    paths = [
        cfg_dir / "feature_groups.yaml",
        cfg_dir / "feature_groups_short.yaml",
        cfg_dir / "feature_groups_long.yaml",
    ]
    out: List[List[str]] = []
    for p in paths:
        d = _read_yaml(p) if p.exists() else {}
        feats: List[str] = []
        fg = d.get("feature_groups", {}) or {}
        if isinstance(fg, dict):
            for _, grp in fg.items():
                if isinstance(grp, dict):
                    xs = grp.get("features", []) or []
                    if isinstance(xs, list):
                        for x in xs:
                            if isinstance(x, str) and x.strip():
                                feats.append(x.strip())
        # unique preserve order
        seen = set()
        uniq = []
        for f in feats:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        out.append(uniq)
    # pad to 3
    while len(out) < 3:
        out.append([])
    return out[0], out[1], out[2]


def build_report_md(
    *,
    cfg: dict,
    root: Path,
    bt_metrics_root: Optional[pd.DataFrame],
    bt_metrics_option_a_20d: Optional[pd.DataFrame],
    model_metrics_root: Optional[pd.DataFrame],
    feature_group_balance_root: Optional[pd.DataFrame],
    feature_importance_summary_root: Optional[pd.DataFrame],
) -> str:
    p = cfg.get("params", {}) or {}
    l4 = cfg.get("l4", {}) or {}
    l5 = cfg.get("l5", {}) or {}
    l6 = cfg.get("l6", {}) or {}
    l7 = cfg.get("l7", {}) or {}

    feature_main, feature_short, feature_long = _load_feature_lists(root)

    # Model metrics summary (mean by phase/horizon)
    model_summary_md = ""
    if model_metrics_root is not None and not model_metrics_root.empty:
        mm = model_metrics_root.copy()
        for c in ["phase", "horizon", "ic_rank", "hit_ratio", "rmse", "r2_oos"]:
            if c not in mm.columns:
                model_summary_md = "_model_metrics.parquet 스키마가 예상과 달라 요약을 생략했습니다._\n"
                break
        else:
            g = (
                mm.groupby(["phase", "horizon"], sort=False)
                .agg(
                    folds=("fold_id", "nunique"),
                    ic_rank_mean=("ic_rank", "mean"),
                    hit_ratio_mean=("hit_ratio", "mean"),
                    rmse_mean=("rmse", "mean"),
                    r2_oos_mean=("r2_oos", "mean"),
                    n_features_mean=("n_features", "mean"),
                )
                .reset_index()
            )
            model_summary_md += "| phase | horizon | folds | ic_rank_mean | hit_ratio_mean | rmse_mean | r2_oos_mean | n_features_mean |\n"
            model_summary_md += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
            for _, r in g.iterrows():
                model_summary_md += (
                    f"| {r['phase']} | {int(r['horizon'])} | {int(r['folds'])} | "
                    f"{_fmt_num(r['ic_rank_mean'], 4)} | {_fmt_num(r['hit_ratio_mean'], 4)} | "
                    f"{_fmt_num(r['rmse_mean'], 4)} | {_fmt_num(r['r2_oos_mean'], 4)} | {_fmt_num(r['n_features_mean'], 1)} |\n"
                )
    else:
        model_summary_md = "_model_metrics.parquet을 찾지 못해 요약을 생략했습니다._\n"

    # Feature group balance
    group_balance_md = ""
    if feature_group_balance_root is not None and not feature_group_balance_root.empty:
        cols_need = ["group_name", "n_features", "target_weight", "actual_weight", "balance_ratio"]
        if all(c in feature_group_balance_root.columns for c in cols_need):
            gb = feature_group_balance_root[cols_need].copy()
            group_balance_md += "| group | n_features | target_weight | actual_weight | balance_ratio |\n"
            group_balance_md += "|---|---:|---:|---:|---:|\n"
            for _, r in gb.iterrows():
                group_balance_md += (
                    f"| {r['group_name']} | {int(r['n_features'])} | {_fmt_num(r['target_weight'], 2)} | "
                    f"{_fmt_num(r['actual_weight'], 2)} | {_fmt_num(r['balance_ratio'], 2)} |\n"
                )
        else:
            group_balance_md = "_feature_group_balance.parquet 스키마가 예상과 달라 요약을 생략했습니다._\n"
    else:
        group_balance_md = "_feature_group_balance.parquet을 찾지 못해 요약을 생략했습니다._\n"

    # Feature importance (top by abs_coef_mean)
    feat_imp_md = ""
    if feature_importance_summary_root is not None and not feature_importance_summary_root.empty:
        cols_need = ["horizon", "phase", "feature", "abs_coef_mean", "coef_sign_stability", "n_folds"]
        if all(c in feature_importance_summary_root.columns for c in cols_need):
            fi = feature_importance_summary_root[cols_need].copy()
            fi = fi.sort_values(["phase", "horizon", "abs_coef_mean"], ascending=[True, True, False])
            feat_imp_md += "상위 10개(phase=dev, horizon=20 기준, abs_coef_mean 내림차순):\n\n"
            sub = fi[(fi["phase"] == "dev") & (fi["horizon"] == 20)].head(10)
            feat_imp_md += "| feature | abs_coef_mean | coef_sign_stability | n_folds |\n"
            feat_imp_md += "|---|---:|---:|---:|\n"
            for _, r in sub.iterrows():
                feat_imp_md += (
                    f"| {r['feature']} | {_fmt_num(r['abs_coef_mean'], 4)} | {_fmt_num(r['coef_sign_stability'], 2)} | {int(r['n_folds'])} |\n"
                )
        else:
            feat_imp_md = "_feature_importance_summary.parquet 스키마가 예상과 달라 요약을 생략했습니다._\n"
    else:
        feat_imp_md = "_feature_importance_summary.parquet을 찾지 못해 요약을 생략했습니다._\n"

    # Backtest metrics
    bt_root_md = "_data/interim 루트 bt_metrics.parquet_:\n\n"
    bt_root_md += _metrics_table_from_bt(bt_metrics_root) if bt_metrics_root is not None else "_없음_\n"
    bt_a_md = "_Option A only (20d) bt_metrics.parquet_:\n\n"
    bt_a_md += _metrics_table_from_bt(bt_metrics_option_a_20d) if bt_metrics_option_a_20d is not None else "_없음_\n"

    # Config summary
    cfg_md = f"""- 기간: **{p.get('start_date')} ~ {p.get('end_date')}**\n- 유니버스: **KOSPI200** (index_code={p.get('index_code')})\n- L4: horizon_short={l4.get('horizon_short')}, horizon_long={l4.get('horizon_long')}, step_days={l4.get('step_days')}, embargo_days={l4.get('embargo_days')}\n- L5(모델): model_type={l5.get('model_type')}, target_transform={l5.get('target_transform')}, ridge_alpha={l5.get('ridge_alpha')}, tune_alpha={l5.get('tune_alpha')}\n- L6(앙상블): weight_short={l6.get('weight_short')}, weight_long={l6.get('weight_long')}, invert_score_sign={l6.get('invert_score_sign')}\n- L7(백테스트): holding_days={l7.get('holding_days')}, top_k={l7.get('top_k')}, cost_bps={l7.get('cost_bps')}, buffer_k={l7.get('buffer_k')}, weighting={l7.get('weighting')}\n- L7 Regime: enabled={((l7.get('regime') or {}).get('enabled'))}, lookback_days={((l7.get('regime') or {}).get('lookback_days'))}\n"""

    # Note about mismatches
    mismatch_note = ""
    if bt_metrics_root is not None and not bt_metrics_root.empty and "buffer_k" in bt_metrics_root.columns:
        # config vs root interim may mismatch due to skip
        try:
            buf_cfg = int(l7.get("buffer_k", 0))
            buf_art = int(bt_metrics_root[bt_metrics_root["phase"] == "dev"]["buffer_k"].iloc[0])
            if buf_cfg != buf_art:
                mismatch_note = (
                    "\n> ⚠️ **주의**: 현재 `configs/config.yaml`의 값과 `data/interim` 루트 산출물이 불일치할 수 있습니다.\n"
                    "> (예: config의 `l7.buffer_k` vs bt_metrics의 buffer_k)\n"
                    "> 이는 `run.skip_if_exists=true`로 인해 **기존 산출물을 재사용**했기 때문일 가능성이 큽니다.\n"
                )
        except Exception:
            pass

    md = f"""# 프로젝트 진행상황 요약 (자동 생성)\n\n- 생성일: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S %Z')}\n- 프로젝트: KOSPI200 퀀트 백테스트 (모델+랭킹 하이브리드)\n\n## 1) 현재 파이프라인 개요\n\n- 데이터: L0(유니버스) → L1(OHLCV) → L2(재무) → L3(패널 머지+피처) → L4(WF split)\n- 모델 트랙: L5(Ridge OOS 예측) → L6(단/장기 예측 앙상블) → L7(백테스트)\n- 랭킹 트랙: L8/L8_short/L8_long(랭킹) → L6R(리밸런싱 스코어 변환/듀얼호라이즌 결합) → L7(백테스트)\n- UI 산출물: L11(UI payload: Top/Bottom + 성과곡선)\n\n## 2) 현재 설정 요약(config.yaml)\n\n{cfg_md}\n{mismatch_note}\n\n## 3) 사용 피처 요약\n\n- feature_groups.yaml(모델/공통): {', '.join(feature_main) if feature_main else '(없음)'}\n- feature_groups_short.yaml(단기): {', '.join(feature_short) if feature_short else '(없음)'}\n- feature_groups_long.yaml(장기): {', '.join(feature_long) if feature_long else '(없음)'}\n\n## 4) 성과 요약 (백테스트)\n\n### 4.1 대표 산출물(현재 data/interim 루트)\n\n{bt_root_md}\n\n### 4.2 Option A only (20d)\n\n{bt_a_md}\n\n## 5) 모델 품질 요약 (L5 model_metrics)\n\n{model_summary_md}\n\n## 6) 피처 그룹 밸런스 (Stage6)\n\n{group_balance_md}\n\n## 7) 피처 중요도(계수 기반) 스냅샷\n\n{feat_imp_md}\n\n## 8) 발견된 이슈/리스크\n\n- `reports/option_a_comprehensive_analysis.md` 기준: **Dev < Holdout** 역전 현상(시장환경/표본크기/구조적 이슈 가능성)\n- `news.enabled`는 현재 비활성화되어 있으나, 관련 모듈은 이미 존재(파일 투입 시 자동 머지 가능)\n\n## 9) 다음 액션(요약)\n\n- 벤치마크 표준화: 전략 vs (KOSPI200) vs (적금)\n- UI MVP: 랭킹 중심 Top/Bottom + 비교곡선\n- 뉴스/ESG: 감성 피처 ingestion + lookahead 방지(lag)\n- 듀얼호라이즌: 단기/장기 결합 α 및 국면별 α 검증\n\n---\n\n_이 파일은 `src/tools/export_project_progress_md.py`가 생성합니다._\n"""
    return md


def main():
    root = _project_root()
    cfg_path = root / "configs" / "config.yaml"
    cfg = _load_cfg(cfg_path)

    interim = root / "data" / "interim"
    snap_root = Snapshot(name="interim_root", base_dir=interim)
    snap_a = Snapshot(name="option_a_only_20d", base_dir=interim / "option_a_only_20d")

    bt_root = _safe_read_parquet(snap_root.parquet("bt_metrics"))
    bt_a = _safe_read_parquet(snap_a.parquet("bt_metrics"))
    mm_root = _safe_read_parquet(snap_root.parquet("model_metrics"))
    gb_root = _safe_read_parquet(snap_root.parquet("feature_group_balance"))
    fi_root = _safe_read_parquet(snap_root.parquet("feature_importance_summary"))

    md = build_report_md(
        cfg=cfg,
        root=root,
        bt_metrics_root=bt_root,
        bt_metrics_option_a_20d=bt_a,
        model_metrics_root=mm_root,
        feature_group_balance_root=gb_root,
        feature_importance_summary_root=fi_root,
    )

    out_path = root / "reports" / "project_progress.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()


