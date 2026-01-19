# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/audit_pipeline_features.py
"""
파이프라인 기능 적용 여부 자동 감사 스크립트
7개 항목을 코드/설정에서 탐지하여 True/False로 출력 + 근거 기록
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def search_in_file(file_path: Path, keywords: List[str], case_sensitive: bool = False) -> List[Tuple[int, str]]:
    """파일에서 키워드 검색"""
    matches = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_lower = line if case_sensitive else line.lower()
                for keyword in keywords:
                    kw = keyword if case_sensitive else keyword.lower()
                    if kw in line_lower:
                        matches.append((line_num, line.strip()))
                        break
    except Exception:
        pass
    return matches

def search_in_directory(root: Path, patterns: List[str], keywords: List[str], exclude_dirs: List[str] = None) -> Dict[str, List[Tuple[int, str]]]:
    """디렉토리에서 패턴과 키워드로 검색"""
    exclude_dirs = exclude_dirs or ["__pycache__", ".git", "node_modules", ".venv", "venv"]
    results = {}

    for pattern in patterns:
        for file_path in root.rglob(pattern):
            # 제외 디렉토리 스킵
            if any(exclude in str(file_path) for exclude in exclude_dirs):
                continue

            matches = search_in_file(file_path, keywords)
            if matches:
                rel_path = str(file_path.relative_to(root))
                results[rel_path] = matches

    return results

def audit_dynamic_k200_universe(root: Path) -> Tuple[bool, Dict]:
    """동적 K200 유니버스 사용 여부"""
    keywords = [
        "universe_k200_membership_monthly",
        "filter_k200_members_only",
        "k200_membership",
        "dynamic_universe",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    # config.yaml에서도 확인
    config_path = root / "configs" / "config.yaml"
    config_evidence = []
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                if cfg.get("params", {}).get("filter_k200_members_only") is not None:
                    config_evidence.append(f"config.yaml: filter_k200_members_only = {cfg['params']['filter_k200_members_only']}")
        except Exception:
            pass

    found = len(results) > 0 or len(config_evidence) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
        "config_evidence": config_evidence,
    }

    return found, evidence

def audit_regime_switching(root: Path) -> Tuple[bool, Dict]:
    """레짐 스위칭 사용 여부"""
    keywords = [
        "regime",
        "regime_switching",
        "market_regime",
        "bull_market",
        "bear_market",
        "vix_regime",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    found = len(results) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
    }

    return found, evidence

def audit_sector_relative_debt_ratio(root: Path) -> Tuple[bool, Dict]:
    """섹터 상대 부채비율 사용 여부"""
    keywords = [
        "sector_relative",
        "sector_relative_debt",
        "debt_ratio_sector",
        "sector_debt_ratio",
        "relative_debt",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    found = len(results) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
    }

    return found, evidence

def audit_feature_explainability_report(root: Path) -> Tuple[bool, Dict]:
    """피처 설명가능성 리포트 생성 여부"""
    keywords = [
        "explainability",
        "feature_importance",
        "shap",
        "permutation_importance",
        "explainability_report",
        "feature_explainability",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    # 리포트 파일 존재 여부 확인
    report_paths = [
        root / "reports" / "feature_importance",
        root / "reports" / "explainability",
        root / "artifacts" / "reports" / "feature_importance",
    ]
    report_files = []
    for rp in report_paths:
        if rp.exists():
            for f in rp.glob("*.csv"):
                report_files.append(str(f.relative_to(root)))
            for f in rp.glob("*.md"):
                report_files.append(str(f.relative_to(root)))

    found = len(results) > 0 or len(report_files) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
        "report_files": report_files[:10],  # 최대 10개만
    }

    return found, evidence

def audit_kfold_cv(root: Path) -> Tuple[bool, Dict]:
    """Time-series aware K-Fold CV 사용 여부"""
    keywords = [
        "walkforward",
        "walk_forward",
        "time_series_split",
        "purged_kfold",
        "embargo",
        "TimeSeriesSplit",
        "walkforward_split",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    # config.yaml에서 embargo_days 확인
    config_path = root / "configs" / "config.yaml"
    config_evidence = []
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                l4 = cfg.get("l4", {})
                if l4.get("embargo_days") is not None:
                    config_evidence.append(f"config.yaml: l4.embargo_days = {l4['embargo_days']}")
                if l4.get("step_days") is not None:
                    config_evidence.append(f"config.yaml: l4.step_days = {l4['step_days']}")
        except Exception:
            pass

    found = len(results) > 0 or len(config_evidence) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
        "config_evidence": config_evidence,
    }

    return found, evidence

def audit_diversification_constraints(root: Path) -> Tuple[bool, Dict]:
    """다각화 제약조건 사용 여부 (섹터 cap, size cap 등)"""
    keywords = [
        "sector_cap",
        "sector_limit",
        "size_cap",
        "diversification",
        "max_weight_per_sector",
        "max_weight_per_stock",
        "constraint",
        "optimization",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    found = len(results) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
    }

    return found, evidence

def audit_factor_count_balancing(root: Path) -> Tuple[bool, Dict]:
    """팩터 개수 균형 사용 여부"""
    keywords = [
        "factor_balancing",
        "factor_count",
        "balanced_factors",
        "factor_weight",
        "equal_weight_factors",
    ]

    results = search_in_directory(root, ["*.py"], keywords)

    # config.yaml에서 weight_short, weight_long 확인
    config_path = root / "configs" / "config.yaml"
    config_evidence = []
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                l6 = cfg.get("l6", {})
                if l6.get("weight_short") is not None and l6.get("weight_long") is not None:
                    if abs(l6.get("weight_short", 0) - l6.get("weight_long", 0)) < 0.1:
                        config_evidence.append(f"config.yaml: l6.weight_short = {l6['weight_short']}, weight_long = {l6['weight_long']} (balanced)")
        except Exception:
            pass

    found = len(results) > 0 or len(config_evidence) > 0

    evidence = {
        "code_matches": {k: v[:5] for k, v in results.items()} if results else {},
        "config_evidence": config_evidence,
    }

    return found, evidence

def main():
    parser = argparse.ArgumentParser(description="Audit pipeline features usage")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument("--run-tag", type=str, required=True, help="Run tag")
    parser.add_argument("--out-dir", type=str, default="reports/audit", help="Output directory")
    args = parser.parse_args()

    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Audit] Starting pipeline feature audit...")

    # 각 항목 감사
    audits = {
        "dynamic_k200_universe_used": audit_dynamic_k200_universe(root),
        "regime_switching_used": audit_regime_switching(root),
        "sector_relative_debt_ratio_used": audit_sector_relative_debt_ratio(root),
        "feature_explainability_report_generated": audit_feature_explainability_report(root),
        "kfold_cv_used": audit_kfold_cv(root),
        "diversification_constraints_used": audit_diversification_constraints(root),
        "factor_count_balancing_used": audit_factor_count_balancing(root),
    }

    # 결과 정리
    results = {}
    for feature_name, (found, evidence) in audits.items():
        results[feature_name] = {
            "used": found,
            "evidence": evidence,
        }

    # JSON 저장
    json_path = out_dir / f"audit__{args.run_tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown 저장
    md_path = out_dir / f"audit__{args.run_tag}.md"
    md_lines = [
        f"# Pipeline Feature Audit: {args.run_tag}",
        "",
        "## Summary",
        "",
        "| Feature | Used |",
        "|---|---|",
    ]

    for feature_name, result in results.items():
        status = "✅ Yes" if result["used"] else "❌ No"
        md_lines.append(f"| {feature_name} | {status} |")

    md_lines.extend([
        "",
        "## Details",
        "",
    ])

    for feature_name, result in results.items():
        md_lines.append(f"### {feature_name}")
        md_lines.append("")
        md_lines.append(f"**Used**: {result['used']}")
        md_lines.append("")

        evidence = result["evidence"]
        if evidence.get("code_matches"):
            md_lines.append("**Code Matches:**")
            for file_path, matches in list(evidence["code_matches"].items())[:5]:
                md_lines.append(f"- `{file_path}`:")
                for line_num, line in matches[:3]:
                    md_lines.append(f"  - Line {line_num}: `{line[:80]}`")
            md_lines.append("")

        if evidence.get("config_evidence"):
            md_lines.append("**Config Evidence:**")
            for ev in evidence["config_evidence"]:
                md_lines.append(f"- {ev}")
            md_lines.append("")

        if evidence.get("report_files"):
            md_lines.append("**Report Files:**")
            for rf in evidence["report_files"][:5]:
                md_lines.append(f"- `{rf}`")
            md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[Audit] JSON saved: {json_path}")
    print(f"[Audit] Markdown saved: {md_path}")
    print("\n=== Audit Results ===")
    for feature_name, result in results.items():
        status = "[OK]" if result["used"] else "[NO]"
        print(f"{status} {feature_name}: {result['used']}")

    return json_path, md_path

if __name__ == "__main__":
    main()
