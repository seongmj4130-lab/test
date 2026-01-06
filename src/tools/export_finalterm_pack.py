# -*- coding: utf-8 -*-
"""
[개선안 21번] finalterm 최종 보고서 패키지 생성기

요구사항:
  - 지금부터 생성하는 보고서는 finalterm/ 폴더에 몰아서 작성
  - 어떤 모델을 사용했고 왜 사용했는지(근거/제약/리스크) 포함
  - UI는 로컬 FastAPI viewer, holdout 예시 기반(실시간 X)

출력:
  - finalterm/00_overview.md
  - finalterm/10_system_design.md
  - finalterm/20_models_and_rationale.md
  - finalterm/30_experiments_summary.md
  - finalterm/40_ui_spec.md

입력(가능한 경우 활용):
  - reports/project_progress.md
  - reports/model_refinement_pack.md
  - reports/dual_horizon_comparison.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise ImportError("PyYAML이 필요합니다. `pip install pyyaml` 후 재실행하세요.") from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _slurp(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _cfg_summary(cfg: dict) -> str:
    p = cfg.get("params", {}) or {}
    l4 = cfg.get("l4", {}) or {}
    l5 = cfg.get("l5", {}) or {}
    l6 = cfg.get("l6", {}) or {}
    l7 = cfg.get("l7", {}) or {}
    news = cfg.get("news", {}) or {}
    l11 = cfg.get("l11", {}) or {}

    return (
        f"- 기간: **{p.get('start_date')} ~ {p.get('end_date')}**\n"
        f"- 유니버스: **KOSPI200** (index_code={p.get('index_code')})\n"
        f"- 예측/보유: horizon_short={l4.get('horizon_short')}, horizon_long={l4.get('horizon_long')}, holding_days={l7.get('holding_days')}\n"
        f"- 거래비용: cost_bps={l7.get('cost_bps')}\n"
        f"- 모델(L5): model_type={l5.get('model_type')}, target_transform={l5.get('target_transform')}, ridge_alpha={l5.get('ridge_alpha')}\n"
        f"- 앙상블(L6): weight_short={l6.get('weight_short')}, weight_long={l6.get('weight_long')}\n"
        f"- 국면(L7.regime): enabled={((l7.get('regime') or {}).get('enabled'))}, lookback_days={((l7.get('regime') or {}).get('lookback_days'))}\n"
        f"- 뉴스(감성): enabled={news.get('enabled')}, source_path={news.get('source_path')}, lag_days={news.get('lag_days')}\n"
        f"- UI(L11): benchmark_types={l11.get('benchmark_types')}, savings_apr={l11.get('savings_apr')}\n"
    )


def export_finalterm_pack() -> None:
    root = _root()
    cfg = _read_yaml(root / "configs" / "config.yaml")

    final_dir = root / "finalterm"
    _ensure_dir(final_dir)

    now = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S %Z")

    # existing evidence (optional)
    progress_md = _slurp(root / "reports" / "project_progress.md")
    refine_md = _slurp(root / "reports" / "model_refinement_pack.md")
    dual_md = _slurp(root / "reports" / "dual_horizon_comparison.md")

    # 00_overview
    overview = f"""# 00. 프로젝트 개요 (Finalterm)\n\n- 생성일: {now}\n\n## 1) 한 문장 요약\n\nKOSPI200 유니버스에서 **Cross-sectional 랭킹 신호**와 **머신러닝 예측 신호**를 결합(스태킹)하여, 거래비용을 고려한 백테스트로 성능을 검증하는 시스템.\n\n## 2) 문제 정의\n\n- 목표: “KOSPI200 내부에서 기계적으로 종목을 선별했을 때, 벤치마크 대비 의미 있는 성과/리스크 특성을 보이는가?”\n- 제약: 투자 조언이 아닌 과거 데이터 기반 시뮬레이션\n\n## 3) 평가 지표(고정)\n\n- 1차: **Net Sharpe / Net MDD / Net CAGR / Net Hit Ratio**\n- 2차: Turnover, IR(벤치 대비), 국면별 성과\n\n## 4) 현재 설정 요약\n\n{_cfg_summary(cfg)}\n\n## 5) 참고(자동 생성 근거)\n\n- `reports/project_progress.md` / `reports/model_refinement_pack.md` / `reports/dual_horizon_comparison.md` 내용을 기반으로 finalterm 문서를 구성합니다.\n"""
    _write(final_dir / "00_overview.md", overview)

    # 10_system_design
    design = f"""# 10. 시스템 설계 (System Design)\n\n- 생성일: {now}\n\n## 1) 파이프라인 요약\n\n- 데이터: L0(유니버스) → L1(OHLCV) → L2(재무) → L3(피처/머지) → L4(Walk-forward split)\n- 랭킹 트랙: L8(/short/long) → L6R(리밸런싱 스코어) → L7(백테스트)\n- 모델 트랙: L5(예측) → L6(앙상블) → L7(백테스트)\n- 제품/시각화: L11(UI payload) + FastAPI viewer\n\n## 2) 신호 모드(최종)\n\n- **RankingSingle**: 단일 랭킹 점수 기반\n- **RankingDual**: 단기/장기 랭킹 결합(α + 국면별 α)\n- **ModelLinear**: Ridge 기반 예측\n- **ModelML**: XGBoost/RandomForest 기반 예측(추가 예정)\n- **Stacked**: (RankingDual + ModelLinear + ModelML) → 메타모델로 결합\n\n## 3) 누수 방지 규칙(스태킹 핵심)\n\n- 베이스 모델/랭킹 신호는 **OOS 기준으로만** 사용\n- 메타모델 학습은 **dev OOS만 사용**, holdout은 블라인드\n\n## 4) 시장 국면 3단계 설계\n\n- 목표: 최종 발표/운영 안정성을 위해 국면을 단순화\n- 정의(초안): lookback_return_pct 기반\n  - bull: > +x%\n  - bear: < -x%\n  - neutral: 그 외\n\n## 5) 데이터 계약(요약)\n\n- L7 입력: `rebalance_scores(date,ticker,phase,score_ens,true_short,...)`\n- UI 입력: `ranking_daily`, `ui_top_bottom_daily`, `ui_equity_curves`, `ui_metrics`\n"""
    _write(final_dir / "10_system_design.md", design)

    # 20_models_and_rationale
    models = f"""# 20. 모델/신호 선택 근거 (Models & Rationale)\n\n- 생성일: {now}\n\n## 1) 랭킹(크로스섹셔널) 기반을 유지하는 이유\n\n- **설명가능성**: 같은 날짜의 종목들을 상대평가(정규화) 후 점수 합산 → “왜 이 종목이 상위인가” 설명이 쉽다.\n- **안정성**: 팩터/그룹 기반 설계가 가능(모멘텀/가치/리스크 등)하며, 모델이 복잡해져도 기준점 역할.\n\n## 2) ML(XGBoost/RandomForest)을 추가하려는 이유\n\n- **비선형/상호작용**: 단순 선형(Ridge)로 잡기 어려운 관계를 학습 가능\n- **스태킹에서의 역할**: 랭킹 신호가 강한 구간/모델 예측이 강한 구간이 다를 수 있어 결합 이점\n\n## 3) 리스크/주의사항(왜 ‘스태킹 절차 강제’가 필요한가)\n\n- 누수 위험(시계열/패널): split/lag/validation을 잘못 설계하면 성능이 과대평가됨\n- 과적합 위험: XGB는 특히 dev 성능을 “쉽게” 올릴 수 있어, holdout 일반화가 필수\n\n## 4) 최종 제안(채택 기준)\n\n- 베이스 3개를 먼저 고정 비교: **RankingDual + Ridge + XGB(or RF)**\n- 그 다음 스태킹: 메타모델은 dev OOS로만 학습, holdout은 평가 전용\n\n## 5) 현재 사용 중인 모델(코드 기준)\n\n- 모델 트랙 기본: Ridge + cs_rank 타깃\n- 랭킹 트랙: percentile 정규화 기반 score_total/rank_total, 듀얼호라이즌 결합(L6R)\n"""
    _write(final_dir / "20_models_and_rationale.md", models)

    # 30_experiments_summary (reuse existing dual report as evidence)
    experiments = f"""# 30. 실험 요약 (Experiments Summary)\n\n- 생성일: {now}\n\n## 1) 동일 기준 비교(핵심)\n\n- 동일 L7 규칙/비용을 적용하여 모드별 성과를 비교한다.\n- 벤치마크는 universe_mean/kospi200/savings 3종을 기본으로 한다.\n\n## 2) 현재 비교 결과(근거)\n\n아래는 자동 생성 리포트에서 발췌:\n\n### 2.1 듀얼호라이즌 비교 리포트\n\n{dual_md if dual_md else '_(dual_horizon_comparison.md 없음)_'}\n\n### 2.2 진행상황/모델 진단\n\n{refine_md if refine_md else '_(model_refinement_pack.md 없음)_'}\n\n## 3) 다음 실험(최종까지)\n\n- 3단계 국면(bull/neutral/bear)로 단순화 후, 듀얼호라이즌 α/전략 파라미터 재검증\n- ML 베이스 모델(XGB/RF) 추가 후, 스태킹 성능 검증\n"""
    _write(final_dir / "30_experiments_summary.md", experiments)

    # 40_ui_spec
    ui_spec = f"""# 40. 로컬 UI 명세 (FastAPI Viewer)\n\n- 생성일: {now}\n\n## 1) UI 목표\n\n- 실시간이 아닌 **holdout 예시 결과를 로드해서 보여주는 viewer**\n- 비교 대상: (전략) vs (KOSPI200) vs (적금)\n- 추가: 모드 비교(model vs ranking_single vs ranking_dual vs stacked)\n\n## 2) 데이터 소스(예시)\n\n- `data/interim/*` 산출물을 읽어 표시\n- 우선 순위:\n  - `ui_snapshot`, `ui_top_bottom_daily`, `ui_equity_curves`, `ui_metrics`\n  - `bt_metrics`, `bt_vs_benchmark_multi`, `dual_horizon_comparison` 결과\n\n## 3) 엔드포인트(초안)\n\n- `GET /` : 간단한 홈/가이드\n- `GET /api/metrics` : 모드별 성과 요약\n- `GET /api/equity?mode=...&bench=...` : 누적곡선\n- `GET /api/top-bottom?date=YYYY-MM-DD` : 해당일 Top/Bottom 리스트\n\n## 4) 화면(초안)\n\n- Home: 프로젝트 요약 + holdout 기간 표시\n- Ranking: 날짜 선택 → Top/Bottom + 기여도(contrib_*)\n- Compare: 모드별 성과표 + 벤치 3종 대비\n\n## 5) 구현 제약\n\n- 로컬 실행(Windows), 실시간 업데이트 없음\n- 데이터 파일이 없으면 안내 메시지로 graceful degradation\n"""
    _write(final_dir / "40_ui_spec.md", ui_spec)


def main():
    export_finalterm_pack()
    print("[OK] finalterm pack written: finalterm/*.md")


if __name__ == "__main__":
    main()


