"""
[개선안 19번] 모델 정교화 패키지(Model Refinement Pack) 보고서 생성기

산출물:
  - reports/model_refinement_pack.md

포함 내용:
  - 모델 성능 요약(model_metrics)
  - 피처 그룹 밸런스(feature_group_balance)
  - 피처 계수 안정성(feature_importance_summary)
  - Dev↔Holdout 역전 현상 점검 체크리스트
  - (옵션) alpha 튜닝 설정 가이드
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise ImportError(
            "PyYAML이 필요합니다. `pip install pyyaml` 후 재실행하세요."
        ) from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _fmt(x, nd=4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def build_md(
    *,
    cfg: dict,
    model_metrics: Optional[pd.DataFrame],
    feature_group_balance: Optional[pd.DataFrame],
    feature_importance_summary: Optional[pd.DataFrame],
    bt_metrics_option_a_20d: Optional[pd.DataFrame],
) -> str:
    l5 = cfg.get("l5", {}) or {}

    # model_metrics summary
    mm_md = "_model_metrics.parquet 없음_\n"
    if model_metrics is not None and not model_metrics.empty:
        mm = model_metrics.copy()

        def _nan_mean(s: pd.Series) -> float:
            try:
                return float(pd.to_numeric(s, errors="coerce").mean())
            except Exception:
                return float("nan")

        def _nan_median(s: pd.Series) -> float:
            try:
                return float(pd.to_numeric(s, errors="coerce").median())
            except Exception:
                return float("nan")

        def _nan_pct_true(s: pd.Series) -> float:
            try:
                x = pd.to_numeric(s, errors="coerce")
                return float(x.mean() * 100.0)
            except Exception:
                return float("nan")

        # [개선안 19번] alpha 튜닝 컬럼이 없을 수도 있으므로, 없으면 NaN으로 표기
        g = (
            mm.groupby(["phase", "horizon"], sort=False)
            .agg(
                folds=(
                    ("fold_id", "nunique")
                    if "fold_id" in mm.columns
                    else ("phase", "size")
                ),
                ic_rank_mean=(
                    ("ic_rank", _nan_mean)
                    if "ic_rank" in mm.columns
                    else ("phase", lambda _: float("nan"))
                ),
                hit_ratio_mean=(
                    ("hit_ratio", _nan_mean)
                    if "hit_ratio" in mm.columns
                    else ("phase", lambda _: float("nan"))
                ),
                rmse_mean=(
                    ("rmse", _nan_mean)
                    if "rmse" in mm.columns
                    else ("phase", lambda _: float("nan"))
                ),
                r2_oos_mean=(
                    ("r2_oos", _nan_mean)
                    if "r2_oos" in mm.columns
                    else ("phase", lambda _: float("nan"))
                ),
                alpha_tuned_pct=(
                    ("alpha_tuned", _nan_pct_true)
                    if "alpha_tuned" in mm.columns
                    else ("phase", lambda _: float("nan"))
                ),
                ridge_alpha_median=(
                    ("ridge_alpha_used", _nan_median)
                    if "ridge_alpha_used" in mm.columns
                    else ("phase", lambda _: float("nan"))
                ),
            )
            .reset_index()
        )
        mm_md = "| phase | horizon | folds | ic_rank_mean | hit_ratio_mean | rmse_mean | r2_oos_mean | alpha_tuned_% | ridge_alpha_median |\n"
        mm_md += "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        for _, r in g.iterrows():
            mm_md += (
                f"| {r['phase']} | {int(r['horizon'])} | {int(r['folds'])} | "
                f"{_fmt(r.get('ic_rank_mean', None), 4)} | {_fmt(r.get('hit_ratio_mean', None), 4)} | "
                f"{_fmt(r.get('rmse_mean', None), 4)} | {_fmt(r.get('r2_oos_mean', None), 4)} | "
                f"{_fmt(r.get('alpha_tuned_pct', None), 2)} | {_fmt(r.get('ridge_alpha_median', None), 4)} |\n"
            )

    # feature group balance
    gb_md = "_feature_group_balance.parquet 없음_\n"
    if feature_group_balance is not None and not feature_group_balance.empty:
        cols_need = [
            "group_name",
            "n_features",
            "target_weight",
            "actual_weight",
            "balance_ratio",
        ]
        if all(c in feature_group_balance.columns for c in cols_need):
            gb_md = "| group | n_features | target_weight | actual_weight | balance_ratio |\n"
            gb_md += "|---|---:|---:|---:|---:|\n"
            for _, r in feature_group_balance[cols_need].iterrows():
                gb_md += f"| {r['group_name']} | {int(r['n_features'])} | {_fmt(r['target_weight'],2)} | {_fmt(r['actual_weight'],2)} | {_fmt(r['balance_ratio'],2)} |\n"

    # coefficient stability (top unstable)
    fi_md = "_feature_importance_summary.parquet 없음_\n"
    if feature_importance_summary is not None and not feature_importance_summary.empty:
        cols_need = [
            "phase",
            "horizon",
            "feature",
            "abs_coef_mean",
            "coef_sign_stability",
            "n_folds",
        ]
        if all(c in feature_importance_summary.columns for c in cols_need):
            fi = feature_importance_summary[cols_need].copy()
            # low sign_stability but large abs coef => 위험
            fi["risk_score"] = fi["abs_coef_mean"] * (1.0 - fi["coef_sign_stability"])
            top = fi.sort_values("risk_score", ascending=False).head(15)
            fi_md = "부호 불안정(계수 sign stability 낮음) + 영향력 큼(abs coef 큼) 상위 15개:\n\n"
            fi_md += "| feature | phase | horizon | abs_coef_mean | sign_stability | risk_score | n_folds |\n"
            fi_md += "|---|---|---:|---:|---:|---:|---:|\n"
            for _, r in top.iterrows():
                fi_md += (
                    f"| {r['feature']} | {r['phase']} | {int(r['horizon'])} | "
                    f"{_fmt(r['abs_coef_mean'],4)} | {_fmt(r['coef_sign_stability'],2)} | {_fmt(r['risk_score'],4)} | {int(r['n_folds'])} |\n"
                )

    # Dev/Holdout inversion note (Option A 20d)
    inv_md = ""
    if (
        bt_metrics_option_a_20d is not None
        and not bt_metrics_option_a_20d.empty
        and "phase" in bt_metrics_option_a_20d.columns
    ):
        bt = bt_metrics_option_a_20d.set_index("phase")
        if "net_sharpe" in bt.columns:
            dev = float(bt.loc["dev", "net_sharpe"]) if "dev" in bt.index else None
            ho = (
                float(bt.loc["holdout", "net_sharpe"])
                if "holdout" in bt.index
                else None
            )
            if (dev is not None) and (ho is not None):
                inv_md = f"- Option A(20d) 기준: **Dev net_sharpe={dev:.4f} vs Holdout net_sharpe={ho:.4f}** (역전 여부 확인)\n"

    md = f"""# 모델 정교화 패키지 (Model Refinement Pack)\n\n- 생성일: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n## 1) 현재 모델 설정 요약\n\n- model_type: `{l5.get('model_type')}`\n- target_transform: `{l5.get('target_transform')}`\n- ridge_alpha: `{l5.get('ridge_alpha')}`\n- tune_alpha: `{l5.get('tune_alpha')}`\n- alpha_grid: `{l5.get('alpha_grid')}`\n\n## 2) 모델 품질 요약(model_metrics)\n\n{mm_md}\n\n## 3) 피처 그룹 밸런스(feature_group_balance)\n\n{gb_md}\n\n## 4) 피처 계수 안정성(feature_importance_summary)\n\n{fi_md}\n\n## 5) Dev↔Holdout 역전 현상 점검 체크리스트\n\n{inv_md}\n- 데이터/기간 체크\n  - Dev/Holdout 기간이 서로 다른 시장환경인지(변동성/추세/금리)\n  - 리밸런싱 횟수(표본)가 충분한지\n- 파이프라인/설정 체크\n  - `run.skip_if_exists=true`로 인해 config와 산출물 불일치가 없는지\n  - 거래비용(cost_bps), buffer_k, regime 설정이 동일하게 적용되었는지\n- 모델/피처 체크\n  - 특정 피처가 holdout에서만 강하게 작동하는지(계수 안정성/부호)\n  - Universe coverage / 결측률이 기간별로 다른지\n\n## 6) (옵션) alpha 튜닝 사용 가이드\n\n- `configs/config.yaml`에서:\n  - `l5.tune_alpha: true`\n  - `l5.alpha_grid: [0.1, 0.3, 1.0, 3.0, 10.0]` 같이 후보 지정\n- 튜닝 정책(코드): **fold train 내부에서만 time-split validation**으로 ic_rank 최대화\n  - 구현: `src/stages/modeling/l5_train_models.py`의 `_select_ridge_alpha_by_time_split`\n\n---\n\n_이 파일은 `src/tools/export_model_refinement_pack_md.py`가 생성합니다._\n"""
    return md


def main():
    root = _project_root()
    cfg = _read_yaml(root / "configs" / "config.yaml")

    interim = root / "data" / "interim"
    model_metrics = _safe_read_parquet(interim / "model_metrics.parquet")
    feature_group_balance = _safe_read_parquet(
        interim / "feature_group_balance.parquet"
    )
    feature_importance_summary = _safe_read_parquet(
        interim / "feature_importance_summary.parquet"
    )
    bt_metrics_option_a_20d = _safe_read_parquet(
        interim / "option_a_only_20d" / "bt_metrics.parquet"
    )

    md = build_md(
        cfg=cfg,
        model_metrics=model_metrics,
        feature_group_balance=feature_group_balance,
        feature_importance_summary=feature_importance_summary,
        bt_metrics_option_a_20d=bt_metrics_option_a_20d,
    )

    out_path = root / "reports" / "model_refinement_pack.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
