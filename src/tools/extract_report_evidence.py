"""
[개선안 11번] 보고서 증거자료 자동 추출기

요구사항 요약:
- 추정/가정 금지: 파일/코드/산출물에서 읽히는 값만 출력
- 결과는 프로젝트 루트 하위 reports/extract/ 에 저장(없으면 생성)
- 복붙 가능한 형태(MD/CSV/JSON)

실행:
  python src/tools/extract_report_evidence.py
  python src/tools/extract_report_evidence.py --run-tag stage14_checklist_final_20251222_141500 --baseline-tag stage12_final_export_20251221_013411
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# =========================
# Constants (User-provided)
# =========================
DEFAULT_RUN_TAG = "stage14_checklist_final_20251222_141500"
DEFAULT_BASELINE_TAG = "stage12_final_export_20251221_013411"

CORE_PIP_PACKAGES = ["numpy", "pandas", "scikit-learn", "pyarrow", "pykrx"]

CHART_ARTIFACTS = [
    "bt_equity_curve",
    "bt_returns",
    "bt_benchmark_returns",
    "bt_vs_benchmark",
    "bt_drawdown_events",
    "bt_yearly_metrics",
    "bt_sensitivity",
]

KPI_CORE_METRICS = ["net_total_return", "net_cagr", "net_sharpe", "net_mdd"]


@dataclass(frozen=True)
class ResolvedPaths:
    """
    [개선안 11번] 프로젝트 경로 해석 결과

    - project_root: 산출물을 저장할 프로젝트 루트
    - config_path: 발견된 configs/config.yaml 경로(없으면 None)
    - base_dir_from_config: config에서 읽힌 base_dir (없거나 invalid면 None)
    """

    project_root: Path
    config_path: Optional[Path]
    base_dir_from_config: Optional[Path]


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_text(v: Any) -> str:
    if v is None:
        return "알 수 없습니다"
    s = str(v)
    return s if s.strip() else "알 수 없습니다"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", errors="replace")


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def resolve_project_root(cwd: Path) -> ResolvedPaths:
    """
    [개선안 11번] PROJECT_ROOT 자동 탐지

    규칙:
    - 현재 작업 폴더(cwd)에서 configs/config.yaml을 찾고 paths.base_dir을 확인
    - base_dir이 존재하는 디렉터리면 project_root=base_dir
    - config.yaml이 없거나 base_dir이 없거나 유효하지 않으면 project_root=cwd
    """
    # configs/config.yaml: cwd 또는 상위 디렉터리에서 탐색(최대 6레벨)
    cfg_path: Optional[Path] = None
    for p in [cwd, *cwd.parents[:6]]:
        cand = p / "configs" / "config.yaml"
        if cand.exists():
            cfg_path = cand
            break

    # reports/extract: cwd 또는 상위 디렉터리에서 탐색(최대 6레벨)
    extract_root: Optional[Path] = None
    for p in [cwd, *cwd.parents[:6]]:
        cand = p / "reports" / "extract"
        if cand.exists() and cand.is_dir():
            extract_root = p
            break

    if cfg_path is None:
        # config가 없으면, reports/extract가 존재하는 상위 폴더를 root로 사용 (근거: 폴더 존재)
        if extract_root is not None:
            return ResolvedPaths(project_root=extract_root, config_path=None, base_dir_from_config=None)
        return ResolvedPaths(project_root=cwd, config_path=None, base_dir_from_config=None)

    base_dir: Optional[Path] = None
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8", errors="replace"))
        paths = (cfg or {}).get("paths", {}) if isinstance(cfg, dict) else {}
        raw = paths.get("base_dir") if isinstance(paths, dict) else None
        if isinstance(raw, str) and raw.strip():
            cand = Path(raw)
            # Windows 경로 문자열을 그대로 Path로 받음
            if cand.exists() and cand.is_dir():
                base_dir = cand
    except Exception:
        base_dir = None

    return ResolvedPaths(
        project_root=(base_dir if base_dir is not None else cwd),
        config_path=cfg_path,
        base_dir_from_config=base_dir,
    )


# =========================
# Task 1: Reproducibility
# =========================
def collect_git_repro(project_root: Path) -> Dict[str, Any]:
    """
    [개선안 11번] git branch/commit/status(dirty 파일 목록) 수집.
    git이 없거나 저장소가 아니면 해당 항목은 '알 수 없습니다'로 기록.
    """

    def run_git(args: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
        try:
            cp = subprocess.run(
                ["git", *args],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            out = (cp.stdout or "").strip()
            err = (cp.stderr or "").strip()
            if cp.returncode != 0:
                return None, (err if err else out if out else f"git rc={cp.returncode}")
            return out, None
        except FileNotFoundError:
            return None, "git executable not found"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    branch, branch_err = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit, commit_err = run_git(["rev-parse", "HEAD"])
    status, status_err = run_git(["status", "--porcelain"])

    dirty_files: List[str] = []
    if status:
        for line in status.splitlines():
            # porcelain: XY <path>
            m = re.match(r"^[ MARCUD\?]{1}[ MARCUD\?]{1}\s+(.*)$", line)
            if m:
                dirty_files.append(m.group(1).strip())
            else:
                dirty_files.append(line.strip())

    return {
        "collected_at": _now_iso(),
        "project_root": str(project_root),
        "branch": branch if branch is not None else "알 수 없습니다",
        "commit": commit if commit is not None else "알 수 없습니다",
        "status_porcelain": status if status is not None else "알 수 없습니다",
        "dirty_files": dirty_files,
        "errors": {
            "branch": branch_err,
            "commit": commit_err,
            "status": status_err,
        },
    }


def collect_env_repro() -> Dict[str, Any]:
    """
    [개선안 11번] python 버전 + pip freeze(핵심 패키지만) 수집.
    pip이 없거나 실패하면 '알 수 없습니다'로 기록.
    """
    py = {
        "executable": sys.executable,
        "version": sys.version.replace("\n", " "),
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
    }

    freeze_raw: Optional[str] = None
    freeze_err: Optional[str] = None
    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if cp.returncode != 0:
            freeze_err = (cp.stderr or "").strip() or f"pip freeze rc={cp.returncode}"
        else:
            freeze_raw = cp.stdout or ""
    except Exception as e:
        freeze_err = f"{type(e).__name__}: {e}"

    pkgs: Dict[str, str] = {}
    if freeze_raw:
        wanted = {p.lower() for p in CORE_PIP_PACKAGES}
        for line in freeze_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 일반적으로 pkg==ver
            name = line.split("==", 1)[0].strip().lower()
            if name in wanted:
                pkgs[name] = line

    # 누락 패키지 표기(추정 금지: 단순히 freeze에 없으면 모름)
    missing = [p for p in CORE_PIP_PACKAGES if p.lower() not in pkgs]

    return {
        "collected_at": _now_iso(),
        "python": py,
        "pip_freeze_core": pkgs,
        "pip_freeze_core_missing": missing,
        "errors": {"pip_freeze": freeze_err},
    }


def collect_command_logs(project_root: Path) -> Tuple[str, List[Path]]:
    """
    [개선안 11번] 실행 커맨드 로그 수집.
    - 우선: reports/history, reports/stages/history
    - 추가 탐색: reports/logs, reports/analysis (실제 존재하는 runlog가 여기서 발견됨)

    반환:
      - md_text
      - sources(list of files used)
    """
    candidates = [
        project_root / "reports" / "history",
        project_root / "reports" / "stages" / "history",
        project_root / "reports" / "stages",
        project_root / "reports" / "logs",
        project_root / "reports" / "analysis",
    ]

    rx = re.compile(
        r"(Command:\s+.*run_all\.py.*|"
        r"\bpython\b.*(?:src[/\\]run_all\.py|src[/\\]run_backtest\.py).*)",
        flags=re.IGNORECASE,
    )

    found_files: List[Path] = []
    md_lines: List[str] = []
    md_lines.append("# Repro Commands")
    md_lines.append("")
    md_lines.append(f"- collected_at: `{_now_iso()}`")
    md_lines.append(f"- project_root: `{project_root}`")
    md_lines.append("")

    any_hit = False
    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        # 너무 넓게 퍼지지 않게 확장자 제한
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".md", ".txt", ".log", ".csv"}:
                continue
            try:
                text = _read_text(p)
            except Exception:
                continue
            lines = text.splitlines()
            hit_idx = [i for i, line in enumerate(lines) if rx.search(line)]
            if not hit_idx:
                continue
            any_hit = True
            found_files.append(p)
            md_lines.append(f"## Source: `{p.relative_to(project_root)}`")
            md_lines.append("")
            for i in hit_idx[:200]:
                # 주변 컨텍스트 2줄
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                md_lines.append("```")
                for j in range(start, end):
                    md_lines.append(lines[j])
                md_lines.append("```")
                md_lines.append("")
    if not any_hit:
        md_lines.append("## 결과")
        md_lines.append("")
        md_lines.append("reports/history 또는 reports/stages/history 등에 실행 커맨드 로그가 **발견되지 않았습니다**. (알 수 없습니다)")
        md_lines.append("")

    return "\n".join(md_lines).rstrip() + "\n", found_files


# =========================
# Task 2: CLI / Config evidence
# =========================
def parse_run_all_cli_options(run_all_path: Path) -> List[Dict[str, str]]:
    """
    [개선안 11번] src/run_all.py의 argparse 옵션 파싱(AST 기반; 평가/실행 없음).
    추정 금지: default/help가 코드에 없으면 '알 수 없습니다'로 남김.
    """
    src = _read_text(run_all_path)
    tree = ast.parse(src)

    rows: List[Dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # parser.add_argument(...)
        is_add_argument = isinstance(func, ast.Attribute) and func.attr == "add_argument"
        if not is_add_argument:
            continue

        opt_strs: List[str] = []
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                opt_strs.append(a.value)
            else:
                seg = ast.get_source_segment(src, a)
                if seg:
                    opt_strs.append(seg)
        opt = ", ".join(opt_strs) if opt_strs else "알 수 없습니다"

        kw: Dict[str, str] = {}
        for k in node.keywords:
            if not k.arg:
                continue
            seg = ast.get_source_segment(src, k.value)
            if isinstance(k.value, ast.Constant):
                seg = repr(k.value.value)
            kw[k.arg] = seg if seg is not None else "알 수 없습니다"

        rows.append(
            {
                "option": opt,
                "help": kw.get("help", "알 수 없습니다"),
                "default": kw.get("default", "알 수 없습니다"),
                "action": kw.get("action", "알 수 없습니다"),
                "type": kw.get("type", "알 수 없습니다"),
                "required": kw.get("required", "알 수 없습니다"),
                "choices": kw.get("choices", "알 수 없습니다"),
                "dest": kw.get("dest", "알 수 없습니다"),
            }
        )
    # 중복 제거(동일 option 문자열)
    seen = set()
    uniq: List[Dict[str, str]] = []
    for r in rows:
        k = r.get("option", "")
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq


def build_cli_md(rows: List[Dict[str, str]], *, title: str) -> str:
    """
    [개선안 11번] CLI 옵션을 Markdown 표로 출력.
    """
    out: List[str] = []
    out.append(f"# {title}")
    out.append("")
    out.append(f"- collected_at: `{_now_iso()}`")
    out.append("")
    out.append("| option | help | default | action | type | required | choices | dest |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            "| {option} | {help} | {default} | {action} | {type} | {required} | {choices} | {dest} |".format(
                option=str(r.get("option", "")).replace("\n", " "),
                help=str(r.get("help", "")).replace("\n", " "),
                default=str(r.get("default", "")).replace("\n", " "),
                action=str(r.get("action", "")).replace("\n", " "),
                type=str(r.get("type", "")).replace("\n", " "),
                required=str(r.get("required", "")).replace("\n", " "),
                choices=str(r.get("choices", "")).replace("\n", " "),
                dest=str(r.get("dest", "")).replace("\n", " "),
            )
        )
    out.append("")
    return "\n".join(out)


def extract_config_snapshot(config_path: Path) -> Dict[str, Any]:
    """
    [개선안 11번] configs/config.yaml에서 l4/l5/l6/l7 관련 키/값 추출.
    전체를 복붙하지 않고, 해당 섹션의 key/value만 출력.
    """
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(_read_text(config_path))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "l4": None, "l5": None, "l6": None, "l7": None}

    def sec(name: str) -> Any:
        if not isinstance(cfg, dict):
            return None
        # run_all.py 규칙(_resolve_section): params 아래 또는 최상위
        params = cfg.get("params", {}) if isinstance(cfg.get("params", {}), dict) else {}
        if isinstance(params, dict) and isinstance(params.get(name), dict):
            return params.get(name)
        if isinstance(cfg.get(name), dict):
            return cfg.get(name)
        return None

    return {
        "collected_at": _now_iso(),
        "config_path": str(config_path),
        "l4": sec("l4"),
        "l5": sec("l5"),
        "l6": sec("l6"),
        "l7": sec("l7"),
    }


def build_config_snapshot_md(snapshot: Dict[str, Any]) -> str:
    """
    [개선안 11번] config snapshot을 Markdown으로 출력.
    """
    out: List[str] = []
    out.append("# Config Snapshot (l4~l7)")
    out.append("")
    out.append(f"- collected_at: `{_safe_text(snapshot.get('collected_at'))}`")
    out.append(f"- config_path: `{_safe_text(snapshot.get('config_path'))}`")
    out.append("")

    if snapshot.get("error"):
        out.append("## 오류")
        out.append("")
        out.append(f"- error: `{snapshot.get('error')}`")
        out.append("")
        return "\n".join(out)

    for name in ["l4", "l5", "l6", "l7"]:
        out.append(f"## {name}")
        out.append("")
        sec = snapshot.get(name)
        if not isinstance(sec, dict) or not sec:
            out.append("알 수 없습니다")
            out.append("")
            continue
        for k in sorted(sec.keys()):
            v = sec.get(k)
            out.append(f"- **{k}**: `{v}`")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# =========================
# Task 3: Features (L5)
# =========================
def _find_dataset_daily_for_features(project_root: Path, run_tag: str) -> Tuple[Optional[Path], List[str]]:
    """
    [개선안 11번] feature 산출용 dataset_daily.parquet 위치 결정(추정 금지).
    검색 우선순위:
      1) data/interim/{run_tag}/dataset_daily.parquet
      2) data/interim/dataset_daily.parquet
    둘 다 없으면 FAIL.
    """
    notes: List[str] = []
    cand1 = project_root / "data" / "interim" / run_tag / "dataset_daily.parquet"
    if cand1.exists():
        return cand1, [f"selected: {cand1}"]
    notes.append(f"not found: {cand1}")

    cand2 = project_root / "data" / "interim" / "dataset_daily.parquet"
    if cand2.exists():
        return cand2, [*notes, f"selected: {cand2}"]
    notes.append(f"not found: {cand2}")

    return None, notes


def extract_l5_feature_cols_from_code(dataset_df: Any, *, target_col: str) -> Tuple[Optional[List[str]], str]:
    """
    [개선안 11번] L5의 실제 feature selection 로직을 호출해 feature list를 산출.
    - src/stages/l5_train_models.py 의 _pick_feature_cols(df, target_col=...) 사용
    - import 실패/호출 실패 시 reason을 반환
    """
    try:
        # 프로젝트 구조상 src가 import path에 없을 수 있어, <project_root>/src를 추가
        # (추정이 아니라: 실행 위치에 따라 필요할 수 있으므로 방어)
        src_dir = Path(__file__).resolve().parents[1]
        src_dir_str = str(src_dir)
        if src_dir_str not in sys.path:
            sys.path.insert(0, src_dir_str)

        from stages.l5_train_models import _pick_feature_cols  # type: ignore

        cols = _pick_feature_cols(dataset_df, target_col=target_col)
        return list(cols), "ok"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def build_features_final_json(
    *,
    project_root: Path,
    cfg_snapshot: Dict[str, Any],
    run_tag: str,
) -> Dict[str, Any]:
    """
    [개선안 11번] L5 최종 Ridge 모델 feature column 리스트 산출.
    - dataset_daily.parquet를 읽고
    - L5의 _pick_feature_cols 로직으로 feature 후보를 산출한 뒤
    - 실제 존재 컬럼과 교집합만 남김
    - 최종 리스트가 비어있으면 FAIL 사유 기록
    """
    target_col = "ret_fwd_20d"
    l7 = cfg_snapshot.get("l7") if isinstance(cfg_snapshot, dict) else None
    if isinstance(l7, dict):
        # L7 return_col은 백테스트용. L5 target은 horizon별로 다르지만,
        # 여기서는 코드상 exclude에 ret_fwd_20d/120d가 포함되어 있고,
        # 학습은 target_col 인자로 들어가므로 명시 필요.
        pass

    ds_path, notes = _find_dataset_daily_for_features(project_root, run_tag)
    if ds_path is None:
        return {
            "collected_at": _now_iso(),
            "run_tag": run_tag,
            "dataset_daily_path": None,
            "status": "FAIL",
            "reason": "dataset_daily.parquet를 찾을 수 없습니다 (근거가 부족합니다).",
            "notes": notes,
            "features_final": [],
        }

    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(ds_path)
    except Exception as e:
        return {
            "collected_at": _now_iso(),
            "run_tag": run_tag,
            "dataset_daily_path": str(ds_path),
            "status": "FAIL",
            "reason": f"dataset_daily.parquet 로드 실패: {type(e).__name__}: {e}",
            "notes": notes,
            "features_final": [],
        }

    if not hasattr(df, "columns"):
        return {
            "collected_at": _now_iso(),
            "run_tag": run_tag,
            "dataset_daily_path": str(ds_path),
            "status": "FAIL",
            "reason": "dataset_daily가 DataFrame이 아닙니다 (근거가 부족합니다).",
            "notes": notes,
            "features_final": [],
        }

    # target_col 근거: dataset_daily에 존재하는 forward return 컬럼 사용
    # (추정 금지: 실제 존재하는 컬럼 중 ret_fwd_20d 우선, 없으면 ret_fwd_120d, 둘 다 없으면 FAIL)
    if "ret_fwd_20d" in df.columns:
        target_col = "ret_fwd_20d"
    elif "ret_fwd_120d" in df.columns:
        target_col = "ret_fwd_120d"
    else:
        return {
            "collected_at": _now_iso(),
            "run_tag": run_tag,
            "dataset_daily_path": str(ds_path),
            "status": "FAIL",
            "reason": "dataset_daily에 target 후보(ret_fwd_20d/ret_fwd_120d)가 없습니다.",
            "notes": notes,
            "features_final": [],
        }

    feature_cols, reason = extract_l5_feature_cols_from_code(df, target_col=target_col)
    if feature_cols is None:
        return {
            "collected_at": _now_iso(),
            "run_tag": run_tag,
            "dataset_daily_path": str(ds_path),
            "status": "FAIL",
            "reason": f"L5 feature selection 호출 실패: {reason}",
            "notes": notes,
            "features_final": [],
        }

    # 교집합(명시적으로 요구됨)
    existing = set(map(str, df.columns))
    final = [c for c in feature_cols if c in existing]

    status = "OK" if len(final) > 0 else "FAIL"
    fail_reason = None
    if status != "OK":
        fail_reason = "교집합 적용 후 feature list가 비었습니다. (근거가 부족합니다)"

    return {
        "collected_at": _now_iso(),
        "run_tag": run_tag,
        "dataset_daily_path": str(ds_path),
        "target_col_used_for_feature_selection": target_col,
        "status": status,
        "reason": fail_reason,
        "notes": notes,
        "n_rows": int(getattr(df, "shape", (0, 0))[0]),
        "n_cols": int(getattr(df, "shape", (0, 0))[1]),
        "features_final": final,
    }


# =========================
# Task 4: Backtest / Benchmark contract (L7/L7C)
# =========================
def _snippet_by_regex(path: Path, pattern: str, *, context: int = 4, max_hits: int = 3) -> List[Dict[str, Any]]:
    """
    [개선안 11번] 코드에서 regex 매치 주변을 snippet으로 수집.
    """
    text = _read_text(path)
    lines = text.splitlines()
    rx = re.compile(pattern)
    hits: List[int] = [i for i, ln in enumerate(lines) if rx.search(ln)]
    out: List[Dict[str, Any]] = []
    for i in hits[:max_hits]:
        s = max(0, i - context)
        e = min(len(lines), i + context + 1)
        out.append(
            {
                "path": str(path),
                "start_line": s + 1,
                "end_line": e,
                "match_line": i + 1,
                "lines": lines[s:e],
            }
        )
    return out


def build_backtest_contract_md(project_root: Path) -> str:
    """
    [개선안 11번] L7/L7C 코드에서 백테스트/벤치 계약(정의/수식/데이터 소스)을 추출해 Markdown으로 저장.
    """
    l7 = project_root / "src" / "stages" / "l7_backtest.py"
    l7c = project_root / "src" / "stages" / "l7c_benchmark.py"
    out: List[str] = []
    out.append("# Backtest Contract (L7 / L7C)")
    out.append("")
    out.append(f"- collected_at: `{_now_iso()}`")
    out.append(f"- source_l7: `{l7}`")
    out.append(f"- source_l7c: `{l7c}`")
    out.append("")

    if not l7.exists():
        out.append("## L7")
        out.append("")
        out.append("알 수 없습니다 (src/stages/l7_backtest.py 없음)")
        out.append("")
    else:
        out.append("## L7: 리밸런싱 날짜 정의")
        out.append("")
        snips = _snippet_by_regex(l7, r"rebalance_dates\s*=", context=6, max_hits=2)
        if snips:
            for s in snips:
                out.append(f"- code: `{Path(s['path']).relative_to(project_root)}` L{s['start_line']}~L{s['end_line']}")
                out.append("```")
                out.extend(s["lines"])
                out.append("```")
                out.append("")
        else:
            out.append("알 수 없습니다 (코드에서 rebalance_dates 정의를 찾지 못함)")
            out.append("")

        out.append("## L7: top_k / buffer_k / holding_days 적용 방식")
        out.append("")
        for pat, label in [
            (r"def _select_with_buffer", "buffer_k 로직"),
            (r"allow_n\s*=\s*top_k\s*\+\s*buffer_k", "허용범위(allow_n=top_k+buffer_k)"),
            (r"holding_days", "holding_days 사용 위치"),
            (r"top_k", "top_k 사용 위치"),
        ]:
            sn = _snippet_by_regex(l7, pat, context=6, max_hits=2)
            if not sn:
                continue
            out.append(f"### {label}")
            out.append("")
            for s in sn:
                out.append(f"- code: `{Path(s['path']).relative_to(project_root)}` L{s['start_line']}~L{s['end_line']}")
                out.append("```")
                out.extend(s["lines"])
                out.append("```")
                out.append("")

        out.append("## L7: cost_bps 적용 방식(수식 포함)")
        out.append("")
        cost_snips = _snippet_by_regex(l7, r"(daily_trading_cost\s*=|turnover_cost\s*=|net_ret\s*=)", context=6, max_hits=6)
        if cost_snips:
            for s in cost_snips:
                out.append(f"- code: `{Path(s['path']).relative_to(project_root)}` L{s['start_line']}~L{s['end_line']}")
                out.append("```")
                out.extend(s["lines"])
                out.append("```")
                out.append("")
        else:
            out.append("알 수 없습니다 (cost_bps 수식 관련 코드를 찾지 못함)")
            out.append("")

    # L7C
    if not l7c.exists():
        out.append("## L7C: 벤치마크 계산 방식(수식 + 데이터 소스)")
        out.append("")
        out.append("알 수 없습니다 (src/stages/l7c_benchmark.py 없음)")
        out.append("")
    else:
        out.append("## L7C: 벤치마크 계산 방식(수식 + 데이터 소스)")
        out.append("")
        # 데이터 소스: rebalance_scores에서 ret_col 후보 중 발견되는 컬럼
        # 수식: bench_return = mean(ret_col) by (phase, date), bench_equity = cumprod(1+bench_return)
        for pat in [
            r"def build_universe_benchmark_returns",
            r"\.groupby\(\[phase_col,\s*date_col\].*\)\[ret_col\]\s*\.mean",
            r"bench\[\s*\"bench_equity\"\s*\]\s*=\s*\(1\.0\s*\+\s*g\[\s*\"bench_return\"\s*\]\).*cumprod",
            r"def compare_strategy_vs_benchmark",
            r"excess_return",
        ]:
            sn = _snippet_by_regex(l7c, pat, context=6, max_hits=2)
            for s in sn:
                out.append(f"- code: `{Path(s['path']).relative_to(project_root)}` L{s['start_line']}~L{s['end_line']}")
                out.append("```")
                out.extend(s["lines"])
                out.append("```")
                out.append("")

    return "\n".join(out).rstrip() + "\n"


# =========================
# Task 5: KPI snapshot
# =========================
def parse_kpi_md_table(md_text: str) -> Dict[str, Dict[str, str]]:
    """
    [개선안 11번] KPI Table(.md)의 표( | Metric | Dev | Holdout | Unit | )를 파싱.
    반환: metric -> {dev, holdout, unit}
    """
    rows: Dict[str, Dict[str, str]] = {}
    for line in md_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue
        metric, dev, holdout, unit = parts
        if metric in ("Metric", "---"):
            continue
        if metric.startswith("---"):
            continue
        rows[metric] = {"dev": dev, "holdout": holdout, "unit": unit}
    return rows


def build_kpi_snapshot_rows(project_root: Path, tags: List[str]) -> List[Dict[str, Any]]:
    """
    [개선안 11번] reports/kpi/kpi_table__{tag}.md 에서 핵심 KPI를 뽑아 CSV row 생성.
    """
    rows: List[Dict[str, Any]] = []
    for tag in tags:
        p = project_root / "reports" / "kpi" / f"kpi_table__{tag}.md"
        if not p.exists():
            # 요구사항: 없으면 스킵
            continue
        d = parse_kpi_md_table(_read_text(p))

        def get(metric: str, side: str = "holdout") -> Optional[str]:
            if metric not in d:
                return None
            return d[metric].get(side)

        rows.append(
            {
                "tag": tag,
                "net_total_return": get("net_total_return"),
                "net_cagr": get("net_cagr"),
                "net_sharpe": get("net_sharpe"),
                "net_mdd": get("net_mdd"),
                "cost_bps_used": get("cost_bps_used"),
                "n_rebalances": get("n_rebalances"),
            }
        )
    return rows


# =========================
# Task 6: Charts export + manifest
# =========================
def _find_artifact_file(base_dir: Path, name: str) -> Optional[Path]:
    """
    [개선안 11번] 아티팩트 파일 찾기.
    우선순위: parquet -> csv (둘 다 존재할 수 있으므로, 변환을 일관되게 parquet 우선)
    """
    pq = base_dir / f"{name}.parquet"
    if pq.exists():
        return pq
    csvp = base_dir / f"{name}.csv"
    if csvp.exists():
        return csvp
    # 일부는 *_metrics 같은 이름일 수 있음(요구사항: bt_sensitivity)
    # 추정 금지: 여기서는 name 그대로만 찾고, 못 찾으면 None
    return None


def _find_tag_or_root_artifact(project_root: Path, tag: str, name: str) -> Tuple[Optional[Path], str]:
    """
    [개선안 11번] data/interim/{tag}/ 우선, 없으면 data/interim/ 루트 fallback으로 아티팩트 탐색.
    반환: (path or None, selected='tag'|'root'|'missing')
    """
    tag_dir = project_root / "data" / "interim" / str(tag)
    root_dir = project_root / "data" / "interim"

    for ext in (".parquet", ".csv"):
        p = tag_dir / f"{name}{ext}"
        if p.exists():
            return p, "tag"
    for ext in (".parquet", ".csv"):
        p = root_dir / f"{name}{ext}"
        if p.exists():
            return p, "root"
    return None, "missing"


def _read_any_df(path: Path) -> Any:
    """
    [개선안 11번] parquet/csv를 DataFrame으로 로드.
    """
    import pandas as pd  # type: ignore

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported extension: {path}")


def _export_any_to_csv(src: Path, dst: Path) -> Tuple[Optional[Tuple[int, int]], Optional[str], Optional[str], str]:
    """
    [개선안 11번] (parquet/csv) -> csv 저장 + (rows, cols, date_min, date_max) 추출.
    date range는 'date' 컬럼 또는 datetime index가 있을 때만 계산.
    """
    import pandas as pd  # type: ignore

    df = None
    if src.suffix.lower() == ".parquet":
        df = pd.read_parquet(src)
    elif src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    else:
        return None, None, None, "unsupported_extension"

    _ensure_dir(dst.parent)
    df.to_csv(dst, index=False)

    rows, cols = (int(df.shape[0]), int(df.shape[1]))

    date_min = None
    date_max = None
    try:
        # 추정 금지:
        # - date range는 명시적 'date' 컬럼이 있고 파싱 가능한 경우에만 계산
        # - 또는 index dtype이 이미 datetime64 인 경우에만 계산(숫자 index를 datetime으로 해석하지 않음)
        if "date" in df.columns:
            dd = pd.to_datetime(df["date"], errors="coerce")
            if dd.notna().any():
                date_min = str(dd.min().date())
                date_max = str(dd.max().date())
        else:
            try:
                if hasattr(df.index, "dtype") and str(df.index.dtype).startswith("datetime64"):
                    di = pd.to_datetime(df.index, errors="coerce")
                    if di.notna().any():
                        date_min = str(di.min().date())
                        date_max = str(di.max().date())
            except Exception:
                date_min, date_max = None, None
    except Exception:
        date_min, date_max = None, None

    return (rows, cols), date_min, date_max, "ok"


def export_charts_and_manifest(project_root: Path, tags: List[str], out_dir: Path) -> List[Dict[str, Any]]:
    """
    [개선안 11번] 차트 아티팩트들을 reports/extract/charts/로 CSV export하고 manifest 생성용 row 반환.
    검색 순서(추정 금지: 순서만 고정, 실제 존재하면 그 경로 사용):
      1) data/interim/{tag}/
      2) data/interim/
    """
    manifest_rows: List[Dict[str, Any]] = []
    charts_dir = out_dir / "charts"
    _ensure_dir(charts_dir)

    for tag in tags:
        # tag dir 우선
        bases = [
            project_root / "data" / "interim" / tag,
            project_root / "data" / "interim",
        ]

        for artifact in CHART_ARTIFACTS:
            src: Optional[Path] = None
            chosen_base: Optional[Path] = None
            for b in bases:
                if b.exists() and b.is_dir():
                    p = _find_artifact_file(b, artifact)
                    if p is not None:
                        src = p
                        chosen_base = b
                        break

            exported = charts_dir / f"{tag}__{artifact}.csv"
            if src is None:
                manifest_rows.append(
                    {
                        "tag": tag,
                        "artifact": artifact,
                        "status": "missing",
                        "source_path": "",
                        "exported_csv": str(exported.relative_to(project_root)),
                        "rows": "",
                        "cols": "",
                        "date_min": "",
                        "date_max": "",
                        "notes": "알 수 없습니다 (source artifact not found)",
                    }
                )
                continue

            try:
                shape, dmin, dmax, st = _export_any_to_csv(src, exported)
                rows, cols = (shape if shape is not None else (None, None))
                manifest_rows.append(
                    {
                        "tag": tag,
                        "artifact": artifact,
                        "status": st,
                        "source_path": str(src.relative_to(project_root)),
                        "exported_csv": str(exported.relative_to(project_root)),
                        "rows": rows,
                        "cols": cols,
                        "date_min": dmin or "",
                        "date_max": dmax or "",
                        "notes": f"source_base={str(chosen_base.relative_to(project_root)) if chosen_base else ''}",
                    }
                )
            except Exception as e:
                manifest_rows.append(
                    {
                        "tag": tag,
                        "artifact": artifact,
                        "status": "error",
                        "source_path": str(src.relative_to(project_root)),
                        "exported_csv": str(exported.relative_to(project_root)),
                        "rows": "",
                        "cols": "",
                        "date_min": "",
                        "date_max": "",
                        "notes": f"{type(e).__name__}: {e}",
                    }
                )

    return manifest_rows


# =========================
# Task 7: Integrity + to_markdown hits + README
# =========================
def build_integrity_report_md(
    *,
    cfg_snapshot: Dict[str, Any],
    kpi_rows: List[Dict[str, Any]],
) -> str:
    """
    [개선안 11번] 주요 값 불일치 검사 요약.
    비교 항목:
      - cost_bps_used vs config(l7.cost_bps)
      - top_k_used vs config(l7.top_k)  (KPI 테이블에 없으면 알 수 없습니다)
    """
    out: List[str] = []
    out.append("# Integrity Report")
    out.append("")
    out.append(f"- collected_at: `{_now_iso()}`")
    out.append("")

    l7 = cfg_snapshot.get("l7") if isinstance(cfg_snapshot, dict) else None
    cfg_cost = None
    cfg_topk = None
    if isinstance(l7, dict):
        cfg_cost = l7.get("cost_bps")
        cfg_topk = l7.get("top_k")

    out.append("## Config 기준값")
    out.append("")
    out.append(f"- l7.cost_bps: `{cfg_cost}`" if cfg_cost is not None else "- l7.cost_bps: `알 수 없습니다`")
    out.append(f"- l7.top_k: `{cfg_topk}`" if cfg_topk is not None else "- l7.top_k: `알 수 없습니다`")
    out.append("")

    out.append("## KPI 기반 사용값(실측)")
    out.append("")
    out.append("| tag | cost_bps_used(holdout) | top_k_used | 판정 | 근거 |")
    out.append("|---|---:|---:|---|---|")
    for r in kpi_rows:
        tag = r.get("tag")
        used_cost = r.get("cost_bps_used")
        # KPI 테이블 스펙상 top_k_used는 별도 필드가 없어 '알 수 없습니다'
        used_topk = r.get("top_k_used") if "top_k_used" in r else None

        verdicts: List[str] = []
        evid: List[str] = []

        # cost 비교
        if cfg_cost is not None and used_cost is not None and used_cost != "N/A":
            try:
                v_cfg = float(cfg_cost)
                v_used = float(str(used_cost))
                if abs(v_cfg - v_used) > 1e-9:
                    verdicts.append("불일치")
                else:
                    verdicts.append("일치")
                evid.append(f"cfg={v_cfg}, used={v_used}")
            except Exception:
                verdicts.append("알 수 없습니다")
                evid.append("cost 파싱 실패")
        else:
            verdicts.append("알 수 없습니다")
            evid.append("cfg 또는 used 누락")

        # top_k 비교(근거 없으면 unknown)
        if cfg_topk is not None and used_topk is not None and used_topk != "N/A":
            try:
                v_cfg = int(cfg_topk)
                v_used = int(str(used_topk))
                if v_cfg != v_used:
                    verdicts.append("top_k 불일치")
                else:
                    verdicts.append("top_k 일치")
                evid.append(f"topk cfg={v_cfg}, used={v_used}")
            except Exception:
                evid.append("top_k 파싱 실패")
        else:
            evid.append("top_k_used 근거 부족(KPI에 없음)")

        out.append(
            f"| {tag} | {used_cost if used_cost is not None else ''} | {used_topk if used_topk is not None else ''} | "
            f"{', '.join(dict.fromkeys(verdicts))} | {', '.join(evid)} |"
        )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def collect_to_markdown_hits(project_root: Path, out_path: Path) -> Dict[str, Any]:
    """
    [개선안 11번] ripgrep로 to_markdown() 호출 검색.
    - rg가 없으면 '알 수 없습니다'로 기록하고, 결과 파일에 원인만 남김.
    """
    try:
        cp = subprocess.run(
            ["rg", "-n", r"to_markdown\("],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if cp.returncode not in (0, 1):  # 1은 no matches
            _write_text(out_path, cp.stderr or f"rg failed rc={cp.returncode}")
            return {"status": "error", "rc": cp.returncode, "stderr": (cp.stderr or "").strip()}
        _write_text(out_path, cp.stdout or "")
        return {"status": "ok", "rc": cp.returncode, "matches_found": bool((cp.stdout or "").strip())}
    except FileNotFoundError:
        # rg 미설치 환경 폴백: 자체 스캔(추정 아님, 실제 파일 내용을 직접 검색)
        hits: List[str] = []
        for base in [project_root / "src", project_root / "reports"]:
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".py", ".md", ".txt"}:
                    continue
                try:
                    lines = _read_text(p).splitlines()
                except Exception:
                    continue
                for i, line in enumerate(lines, start=1):
                    if "to_markdown(" in line:
                        rel = str(p.relative_to(project_root))
                        hits.append(f"{rel}:{i}:{line}")

        header = [
            "주의: ripgrep(rg) 실행 파일을 찾지 못해 Python 폴백 스캔 결과를 기록합니다.",
            "",
        ]
        _write_text(out_path, "\n".join(header + hits) + ("\n" if hits else ""))
        return {"status": "missing_rg_fallback_used", "matches_found": bool(hits), "n_hits": len(hits)}
    except Exception as e:
        _write_text(out_path, f"알 수 없습니다: {type(e).__name__}: {e}\n")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def build_readme_md() -> str:
    """
    [개선안 11번] reports/extract/README.md 매핑 표 생성.
    """
    out: List[str] = []
    out.append("# Reports Extract Output Map")
    out.append("")
    out.append("| 파일 | 용도(보고서 어디에 사용) |")
    out.append("|---|---|")
    out.append("| `repro_git.json` | 재현성: git 브랜치/커밋/dirty 상태 근거 |")
    out.append("| `repro_env.json` | 재현성: Python 버전 및 핵심 패키지 버전(pip freeze) 근거 |")
    out.append("| `repro_commands.md` | 재현성: 실제 실행 커맨드/로그 증거 (reports/*에서 수집) |")
    out.append("| `cli_run_all.md` | 실행기 근거: `src/run_all.py` CLI 옵션(설명/기본값) 표 |")
    out.append("| `config_snapshot.md` | 설정 근거: `configs/config.yaml`의 l4~l7 핵심 파라미터 스냅샷 |")
    out.append("| `features_final.json` | 모델 입력 근거: L5 Ridge가 사용하는 feature 리스트(데이터셋 교집합) |")
    out.append("| `backtest_contract.md` | 백테스트/벤치 정의 근거: L7/L7C 리밸런싱/비용/벤치 수식 및 소스 코드 스니펫 |")
    out.append("| `kpi_snapshot.csv` | 실측 KPI 스냅샷: RUN_TAG/BASELINE_TAG 핵심 KPI 추출(복붙용) |")
    out.append("| `charts/` | 발표용 그래프 입력 CSV: 지정된 bt_* 아티팩트 CSV export |")
    out.append("| `charts_manifest.csv` | 그래프 입력 메타: 각 CSV의 rows/cols/date range 및 source 경로 |")
    out.append("| `integrity_report.md` | 무결성: config vs 실측(예: cost_bps_used) 불일치 요약 |")
    out.append("| `to_markdown_hits.txt` | Stage 체크리포트에서 지목된 to_markdown() 호출 위치(rg 결과) |")
    out.append("| `kpi_units_check.md` | KPI 단위(% vs decimal) 정합성 점검 및 혼용 탐지 |")
    out.append("| `phase_date_ranges.csv` | bt_returns 기반 phase별 기간(min/max date) 및 리밸런싱 횟수(n_rows) 근거 |")
    out.append("| `data_quality__dataset_daily.csv` | dataset_daily 품질 통계(결측률/분포) CSV (피처별) |")
    out.append("| `data_quality__dataset_daily.md` | dataset_daily 품질 통계(결측률/분포) 요약 Markdown |")
    out.append("| `config_vs_code_matrix.csv` | 설정 존재(configured) vs 코드 적용(implemented) 매핑 표 |")
    out.append("| `cost_model_contract.md` | 거래비용/turnover 수식 근거 + 산출물 컬럼 불일치(mismatch) 점검 |")
    out.append("| `artifact_isolation_audit.csv` | 아티팩트 격리 감사: tag 경로 vs fallback 경로 집계 |")
    out.append("| `artifact_isolation_audit.md` | 아티팩트 격리 감사 요약 및 missing_inputs 목록 |")
    out.append("")
    return "\n".join(out).rstrip() + "\n"


# =========================
# 추가 추출 1: KPI 단위 정규화 점검
# =========================
def _parse_float_maybe(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s in ("", "N/A", "nan", "NaN", "None"):
            return None
        return float(s)
    except Exception:
        return None


def _ratio_classify(a: Optional[float], b: Optional[float]) -> str:
    """
    [개선안 11번] 두 값의 스케일 관계를 분류(추정 금지: 근접 여부만 사용).
    """
    if a is None or b is None:
        return "unknown"
    if abs(a) < 1e-12 or abs(b) < 1e-12:
        return "unknown"
    r = a / b
    if abs(r - 1.0) < 0.05:
        return "same_scale(~1x)"
    if abs(abs(r) - 100.0) < 5.0:
        return "scale_100x(~100x)"
    if abs(abs(r) - 0.01) < 0.001:
        return "scale_0.01x(~0.01x)"
    return "unknown"


def build_kpi_units_check_md(*, project_root: Path, out_dir: Path, run_tag: str, baseline_tag: str) -> str:
    """
    [개선안 11번] KPI 단위(% vs decimal) 정합성 점검 및 혼용 탐지.
    - kpi_snapshot.csv의 값만으로 단위 판정 불가 시 unit_unknown 플래그를 남김.
    - bt_returns / bt_metrics / bt_yearly_metrics로 교차확인하여 단위 혼용 여부 탐지.
    """
    import pandas as pd  # type: ignore

    kpi_path = out_dir / "kpi_snapshot.csv"
    out: List[str] = []
    out.append("# KPI Units Check (% vs decimal)")
    out.append("")
    out.append(f"- collected_at: `{_now_iso()}`")
    out.append(f"- run_tag: `{run_tag}`")
    out.append(f"- baseline_tag: `{baseline_tag}`")
    out.append("")

    if not kpi_path.exists():
        out.append("kpi_snapshot.csv: 알 수 없습니다(근거 파일 없음)")
        out.append("")
        return "\n".join(out)

    kpi = pd.read_csv(kpi_path)
    if kpi.empty or "tag" not in kpi.columns:
        out.append("kpi_snapshot.csv가 비어있거나 tag 컬럼이 없습니다. 알 수 없습니다(근거 파일 없음)")
        out.append("")
        return "\n".join(out)

    tags = [t for t in [run_tag, baseline_tag] if t]
    kpi = kpi[kpi["tag"].astype(str).isin(tags)].copy()
    if kpi.empty:
        out.append("kpi_snapshot.csv에 지정 태그 행이 없습니다. 알 수 없습니다(근거 파일 없음)")
        out.append("")
        return "\n".join(out)

    def load_bt_returns(tag: str) -> Tuple[Optional[Any], str]:
        p = out_dir / "charts" / f"{tag}__bt_returns.csv"
        if p.exists():
            try:
                return pd.read_csv(p), str(p.relative_to(project_root))
            except Exception:
                pass
        ap, _sel = _find_tag_or_root_artifact(project_root, tag, "bt_returns")
        if ap is None:
            return None, "알 수 없습니다(근거 파일 없음)"
        try:
            return _read_any_df(ap), str(ap.relative_to(project_root))
        except Exception:
            return None, str(ap.relative_to(project_root))

    def load_bt_metrics(tag: str) -> Tuple[Optional[Any], str]:
        ap, _sel = _find_tag_or_root_artifact(project_root, tag, "bt_metrics")
        if ap is None:
            return None, "알 수 없습니다(근거 파일 없음)"
        try:
            return _read_any_df(ap), str(ap.relative_to(project_root))
        except Exception:
            return None, str(ap.relative_to(project_root))

    def load_bt_yearly(tag: str) -> Tuple[Optional[Any], str]:
        p = out_dir / "charts" / f"{tag}__bt_yearly_metrics.csv"
        if p.exists():
            try:
                return pd.read_csv(p), str(p.relative_to(project_root))
            except Exception:
                pass
        ap, _sel = _find_tag_or_root_artifact(project_root, tag, "bt_yearly_metrics")
        if ap is None:
            return None, "알 수 없습니다(근거 파일 없음)"
        try:
            return _read_any_df(ap), str(ap.relative_to(project_root))
        except Exception:
            return None, str(ap.relative_to(project_root))

    rows_out: List[Dict[str, Any]] = []

    for tag in tags:
        row = kpi[kpi["tag"].astype(str) == str(tag)].head(1)
        if row.empty:
            continue
        r0 = row.iloc[0].to_dict()
        kpi_vals = {m: _parse_float_maybe(r0.get(m)) for m in KPI_CORE_METRICS}

        bt_ret, bt_ret_src = load_bt_returns(tag)
        bt_met, bt_met_src = load_bt_metrics(tag)
        bt_year, bt_year_src = load_bt_yearly(tag)

        # bt_returns 기반 holdout 누적 수익률/최대낙폭 계산(가능한 경우)
        derived_total_holdout: Optional[float] = None
        derived_mdd_holdout: Optional[float] = None
        if bt_ret is not None and "phase" in bt_ret.columns and "net_return" in bt_ret.columns:
            try:
                g = bt_ret[bt_ret["phase"].astype(str) == "holdout"].copy()
                x = g["net_return"].astype(float).to_numpy()
                eq = (1.0 + pd.Series(x)).cumprod()
                derived_total_holdout = float(eq.iloc[-1] - 1.0) if len(eq) else None
                peak = eq.cummax()
                dd = (eq / peak) - 1.0
                derived_mdd_holdout = float(dd.min()) if len(dd) else None
            except Exception:
                derived_total_holdout = None
                derived_mdd_holdout = None

        # bt_metrics holdout
        bt_met_hold = {}
        if bt_met is not None and "phase" in bt_met.columns:
            try:
                h = bt_met[bt_met["phase"].astype(str) == "holdout"]
                if len(h) > 0:
                    h0 = h.iloc[0]
                    for m in KPI_CORE_METRICS:
                        if m in h.columns:
                            bt_met_hold[m] = _parse_float_maybe(h0[m])
            except Exception:
                pass

        # yearly sample (holdout 마지막 행)
        yearly_sample = {}
        if bt_year is not None and "phase" in bt_year.columns:
            try:
                h = bt_year[bt_year["phase"].astype(str) == "holdout"]
                if len(h) > 0:
                    y0 = h.tail(1).iloc[0]
                    for m in ["net_total_return", "net_mdd", "net_sharpe"]:
                        if m in h.columns:
                            yearly_sample[m] = _parse_float_maybe(y0[m])
            except Exception:
                pass

        # unit 판정
        unit_flag = "unit_unknown"
        ev: List[str] = []
        mix: List[str] = []
        if kpi_vals.get("net_total_return") is not None and derived_total_holdout is not None:
            cls = _ratio_classify(kpi_vals["net_total_return"], derived_total_holdout)
            ev.append(f"net_total_return kpi_vs_bt_returns_total={cls}")
            if cls == "same_scale(~1x)":
                unit_flag = "unit_same_as_bt_returns"
            elif cls == "scale_100x(~100x)":
                unit_flag = "unit_mixed_percent_vs_decimal"
        else:
            ev.append("net_total_return: cannot compare (missing kpi or derived)")

        if kpi_vals.get("net_mdd") is not None and derived_mdd_holdout is not None:
            mix.append(f"net_mdd kpi_vs_bt_returns_mdd={_ratio_classify(kpi_vals['net_mdd'], derived_mdd_holdout)}")
        if kpi_vals.get("net_mdd") is not None and bt_met_hold.get("net_mdd") is not None:
            mix.append(f"net_mdd kpi_vs_bt_metrics={_ratio_classify(kpi_vals['net_mdd'], bt_met_hold.get('net_mdd'))}")
        if bt_met_hold.get("net_mdd") is not None and derived_mdd_holdout is not None:
            mix.append(f"net_mdd bt_metrics_vs_bt_returns={_ratio_classify(bt_met_hold.get('net_mdd'), derived_mdd_holdout)}")
        if kpi_vals.get("net_mdd") is not None and yearly_sample.get("net_mdd") is not None:
            mix.append(f"net_mdd kpi_vs_yearly_sample={_ratio_classify(kpi_vals['net_mdd'], yearly_sample.get('net_mdd'))}")
        if kpi_vals.get("net_total_return") is not None and bt_met_hold.get("net_total_return") is not None:
            mix.append(f"net_total_return kpi_vs_bt_metrics={_ratio_classify(kpi_vals['net_total_return'], bt_met_hold.get('net_total_return'))}")
        if bt_met_hold.get("net_total_return") is not None and derived_total_holdout is not None:
            mix.append(f"net_total_return bt_metrics_vs_bt_returns={_ratio_classify(bt_met_hold.get('net_total_return'), derived_total_holdout)}")

        rows_out.append(
            {
                "tag": tag,
                "unit_flag": unit_flag,
                "kpi_net_total_return": kpi_vals.get("net_total_return"),
                "bt_returns_holdout_total_return": derived_total_holdout,
                "kpi_net_mdd": kpi_vals.get("net_mdd"),
                "bt_returns_holdout_mdd": derived_mdd_holdout,
                "bt_metrics_holdout_net_total_return": bt_met_hold.get("net_total_return"),
                "bt_metrics_holdout_net_mdd": bt_met_hold.get("net_mdd"),
                "yearly_holdout_sample_net_total_return": yearly_sample.get("net_total_return"),
                "sources": {
                    "kpi_snapshot": str(kpi_path.relative_to(project_root)),
                    "bt_returns": bt_ret_src,
                    "bt_metrics": bt_met_src,
                    "bt_yearly_metrics": bt_year_src,
                },
                "evidence": ev,
                "mix_checks": mix,
            }
        )

    out.append("## 판정 요약")
    out.append("")
    out.append("| tag | unit_flag | kpi_net_total_return | bt_returns_total_return(holdout) |")
    out.append("|---|---|---:|---:|")
    for r in rows_out:
        out.append(
            f"| {r['tag']} | {r['unit_flag']} | "
            f"{'' if r['kpi_net_total_return'] is None else r['kpi_net_total_return']} | "
            f"{'' if r['bt_returns_holdout_total_return'] is None else r['bt_returns_holdout_total_return']} |"
        )
    out.append("")

    out.append("## 혼용 탐지 상세")
    out.append("")
    for r in rows_out:
        out.append(f"### {r['tag']}")
        out.append("")
        out.append(f"- sources: `{json.dumps(r['sources'], ensure_ascii=False)}`")
        for e in r.get("evidence", []):
            out.append(f"- evidence: `{e}`")
        for m in r.get("mix_checks", []):
            out.append(f"- mix_check: `{m}`")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# =========================
# 추가 추출 2: Holdout 기간 확정
# =========================
def build_phase_date_ranges_rows(*, project_root: Path, out_dir: Path, run_tag: str, baseline_tag: str) -> List[Dict[str, Any]]:
    """
    [개선안 11번] bt_returns.csv (RUN_TAG/Baseline)에서 phase별 min/max(date), n_rows를 집계.
    """
    import pandas as pd  # type: ignore

    rows: List[Dict[str, Any]] = []
    for tag in [run_tag, baseline_tag]:
        src = out_dir / "charts" / f"{tag}__bt_returns.csv"
        df = None
        src_label = ""
        if src.exists():
            try:
                df = pd.read_csv(src)
                src_label = str(src.relative_to(project_root))
            except Exception:
                df = None
        if df is None:
            ap, _sel = _find_tag_or_root_artifact(project_root, tag, "bt_returns")
            if ap is not None:
                try:
                    df = _read_any_df(ap)
                    src_label = str(ap.relative_to(project_root))
                except Exception:
                    df = None
                    src_label = str(ap.relative_to(project_root))

        if df is None or "phase" not in df.columns or "date" not in df.columns:
            rows.append(
                {
                    "tag": tag,
                    "phase": "알 수 없습니다(근거 파일 없음)",
                    "date_min": "",
                    "date_max": "",
                    "n_rows": "",
                    "source": src_label if src_label else "알 수 없습니다(근거 파일 없음)",
                }
            )
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for phase, g in df.groupby("phase", sort=False):
            dmin = ""
            dmax = ""
            if g["date"].notna().any():
                dmin = str(g["date"].min().date())
                dmax = str(g["date"].max().date())
            rows.append(
                {
                    "tag": tag,
                    "phase": str(phase),
                    "date_min": dmin,
                    "date_max": dmax,
                    "n_rows": int(len(g)),
                    "source": src_label,
                }
            )
    return rows


# =========================
# 추가 추출 3: 데이터 품질(결측률/커버리지)
# =========================
def build_dataset_daily_quality_outputs(*, project_root: Path, out_dir: Path, run_tag: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    [개선안 11번] dataset_daily.parquet(가능하면 RUN_TAG 경로 우선, 없으면 루트 fallback)에서
    - rows/cols, ticker 수, date 수, (ticker,date) 중복 여부
    - features_final.json의 features별 NaN%, min/median/max, p01/p99
    산출.
    """
    import pandas as pd  # type: ignore

    features_json = out_dir / "features_final.json"
    features: List[str] = []
    if features_json.exists():
        try:
            j = json.loads(_read_text(features_json))
            features = list(j.get("features_final", []) or [])
        except Exception:
            features = []

    ds_path, notes = _find_dataset_daily_for_features(project_root, run_tag)
    if ds_path is None:
        md = [
            "# Data Quality: dataset_daily",
            "",
            f"- collected_at: `{_now_iso()}`",
            "",
            "dataset_daily.parquet: 알 수 없습니다(근거 파일 없음)",
            "",
        ]
        return [], "\n".join(md)

    try:
        df = pd.read_parquet(ds_path)
    except Exception as e:
        md = [
            "# Data Quality: dataset_daily",
            "",
            f"- collected_at: `{_now_iso()}`",
            f"- dataset_path: `{ds_path}`",
            "",
            f"로드 실패: `{type(e).__name__}: {e}`",
            "",
        ]
        return [], "\n".join(md)

    rows_n, cols_n = int(df.shape[0]), int(df.shape[1])
    ticker_n = int(df["ticker"].astype(str).nunique()) if "ticker" in df.columns else None
    date_n = None
    if "date" in df.columns:
        try:
            date_n = int(pd.to_datetime(df["date"], errors="coerce").nunique())
        except Exception:
            date_n = None
    dup_count = None
    if "ticker" in df.columns and "date" in df.columns:
        try:
            tmp = df[["ticker", "date"]].copy()
            tmp["ticker"] = tmp["ticker"].astype(str)
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            dup_count = int(tmp.duplicated(subset=["ticker", "date"]).sum())
        except Exception:
            dup_count = None

    out_rows: List[Dict[str, Any]] = []
    # summary row
    out_rows.append(
        {
            "feature": "_SUMMARY_",
            "exists": True,
            "nan_pct": "",
            "min": "",
            "p01": "",
            "median": "",
            "p99": "",
            "max": "",
            "rows": rows_n,
            "cols": cols_n,
            "ticker_nunique": ticker_n if ticker_n is not None else "알 수 없습니다(근거 파일 없음)",
            "date_nunique": date_n if date_n is not None else "알 수 없습니다(근거 파일 없음)",
            "ticker_date_dup_count": dup_count if dup_count is not None else "알 수 없습니다(근거 파일 없음)",
            "dataset_path": str(ds_path),
        }
    )

    for f in features:
        if f not in df.columns:
            out_rows.append(
                {
                    "feature": f,
                    "exists": False,
                    "nan_pct": "알 수 없습니다(근거 파일 없음)",
                    "min": "",
                    "p01": "",
                    "median": "",
                    "p99": "",
                    "max": "",
                    "rows": "",
                    "cols": "",
                    "ticker_nunique": "",
                    "date_nunique": "",
                    "ticker_date_dup_count": "",
                    "dataset_path": str(ds_path),
                }
            )
            continue
        s = df[f]
        # quantile 계산에서 bool dtype이 깨질 수 있어 float로 강제(통계 계산 목적)
        s_num = pd.to_numeric(s, errors="coerce").astype("float64")
        nan_pct = float(s_num.isna().mean() * 100.0)
        v = s_num.dropna()
        if len(v) == 0:
            out_rows.append(
                {
                    "feature": f,
                    "exists": True,
                    "nan_pct": nan_pct,
                    "min": "",
                    "p01": "",
                    "median": "",
                    "p99": "",
                    "max": "",
                    "rows": "",
                    "cols": "",
                    "ticker_nunique": "",
                    "date_nunique": "",
                    "ticker_date_dup_count": "",
                    "dataset_path": str(ds_path),
                }
            )
            continue
        out_rows.append(
            {
                "feature": f,
                "exists": True,
                "nan_pct": nan_pct,
                "min": float(v.min()),
                "p01": float(v.quantile(0.01)),
                "median": float(v.quantile(0.50)),
                "p99": float(v.quantile(0.99)),
                "max": float(v.max()),
                "rows": "",
                "cols": "",
                "ticker_nunique": "",
                "date_nunique": "",
                "ticker_date_dup_count": "",
                "dataset_path": str(ds_path),
            }
        )

    md_lines: List[str] = []
    md_lines.append("# Data Quality: dataset_daily")
    md_lines.append("")
    md_lines.append(f"- collected_at: `{_now_iso()}`")
    md_lines.append(f"- dataset_path: `{ds_path}`")
    md_lines.append(f"- notes: `{notes}`")
    md_lines.append("")
    md_lines.append("## 기본 통계")
    md_lines.append("")
    md_lines.append(f"- rows: `{rows_n}`")
    md_lines.append(f"- cols: `{cols_n}`")
    md_lines.append(f"- ticker_nunique: `{ticker_n if ticker_n is not None else '알 수 없습니다(근거 파일 없음)'}`")
    md_lines.append(f"- date_nunique: `{date_n if date_n is not None else '알 수 없습니다(근거 파일 없음)'}`")
    md_lines.append(f"- (ticker,date) duplicates: `{dup_count if dup_count is not None else '알 수 없습니다(근거 파일 없음)'}`")
    md_lines.append("")
    md_lines.append("## 피처 후보 통계")
    md_lines.append("")
    md_lines.append(f"- features_source: `{features_json.relative_to(project_root) if features_json.exists() else '알 수 없습니다(근거 파일 없음)'}`")
    md_lines.append(f"- output_csv: `reports/extract/data_quality__dataset_daily.csv`")
    md_lines.append("")

    return out_rows, "\n".join(md_lines).rstrip() + "\n"


# =========================
# 추가 추출 4: 설정 존재 vs 코드 적용 매핑
# =========================
def build_config_vs_code_matrix_rows(*, project_root: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """
    [개선안 11번] l7.diversify.*, l7.regime.* 키 존재(configured) vs 코드 참조(implemented) 매핑.
    """
    cfg_md = out_dir / "config_snapshot.md"
    l7_div = None
    l7_reg = None
    if cfg_md.exists():
        text = _read_text(cfg_md)
        for line in text.splitlines():
            if line.startswith("- **diversify**:"):
                m = re.search(r"`(.+)`", line)
                if m:
                    try:
                        l7_div = ast.literal_eval(m.group(1))
                    except Exception:
                        l7_div = None
            if line.startswith("- **regime**:"):
                m = re.search(r"`(.+)`", line)
                if m:
                    try:
                        l7_reg = ast.literal_eval(m.group(1))
                    except Exception:
                        l7_reg = None

    code_path = project_root / "src" / "stages" / "l7_backtest.py"
    code_lines: List[str] = _read_text(code_path).splitlines() if code_path.exists() else []

    def hits(substr: str, limit: int = 3) -> str:
        out = []
        for i, ln in enumerate(code_lines, start=1):
            if substr in ln:
                out.append(f"L{i}:{ln.strip()}")
            if len(out) >= limit:
                break
        return "; ".join(out)

    def has(substr: str) -> bool:
        return any(substr in ln for ln in code_lines)

    rows: List[Dict[str, Any]] = []

    div_keys = ["enabled", "group_col", "max_names_per_group"]
    for k in div_keys:
        cfg_val = ""
        configured = False
        if isinstance(l7_div, dict) and k in l7_div:
            configured = True
            cfg_val = str(l7_div.get(k))
        # code mapping
        if k == "enabled":
            impl = has("diversify_enabled")
            ev = hits("diversify_enabled")
        elif k == "group_col":
            impl = has("group_col")
            ev = hits("group_col")
        else:
            impl = has("max_names_per_group")
            ev = hits("max_names_per_group")
        rows.append(
            {
                "section": "l7.diversify",
                "key": k,
                "configured": configured,
                "config_value": cfg_val,
                "implemented": impl,
                "code_hits": ev,
                "code_file": str(code_path.relative_to(project_root)) if code_path.exists() else "알 수 없습니다(근거 파일 없음)",
            }
        )

    reg_keys = ["enabled", "lookback_days", "threshold_pct", "top_k_bull", "top_k_bear", "exposure_bull", "exposure_bear"]
    for k in reg_keys:
        cfg_val = ""
        configured = False
        if isinstance(l7_reg, dict) and k in l7_reg:
            configured = True
            cfg_val = str(l7_reg.get(k))

        if k == "enabled":
            impl = has("regime_enabled")
            ev = hits("regime_enabled")
        elif k == "top_k_bull":
            impl = has("regime_top_k_bull")
            ev = hits("regime_top_k_bull")
        elif k == "top_k_bear":
            impl = has("regime_top_k_bear")
            ev = hits("regime_top_k_bear")
        elif k == "exposure_bull":
            impl = has("regime_exposure_bull")
            ev = hits("regime_exposure_bull")
        elif k == "exposure_bear":
            impl = has("regime_exposure_bear")
            ev = hits("regime_exposure_bear")
        else:
            # lookback_days/threshold_pct는 l7_backtest.py에서 직접 참조되지 않을 수 있음
            impl = has("lookback") or has("threshold")
            ev = "; ".join([x for x in [hits("lookback"), hits("threshold")] if x])

        rows.append(
            {
                "section": "l7.regime",
                "key": k,
                "configured": configured,
                "config_value": cfg_val,
                "implemented": impl,
                "code_hits": ev,
                "code_file": str(code_path.relative_to(project_root)) if code_path.exists() else "알 수 없습니다(근거 파일 없음)",
            }
        )

    return rows


# =========================
# 추가 추출 5: 거래비용 수식 확정
# =========================
def build_cost_model_contract_md(*, project_root: Path, out_dir: Path, run_tag: str, baseline_tag: str) -> str:
    """
    [개선안 11번] L7 거래비용/turnover 수식 근거 추출 + bt_returns 산출물 컬럼 불일치 리포트.
    """
    import pandas as pd  # type: ignore

    l7 = project_root / "src" / "stages" / "l7_backtest.py"
    out: List[str] = []
    out.append("# Cost Model Contract (L7)")
    out.append("")
    out.append(f"- collected_at: `{_now_iso()}`")
    out.append("")

    if not l7.exists():
        out.append("src/stages/l7_backtest.py: 알 수 없습니다(근거 파일 없음)")
        out.append("")
        return "\n".join(out)

    out.append("## turnover 정의 함수")
    out.append("")
    sn = _snippet_by_regex(l7, r"def _compute_turnover_oneway", context=8, max_hits=1)
    if sn:
        s = sn[0]
        out.append(f"- code: `{Path(s['path']).relative_to(project_root)}` L{s['start_line']}~L{s['end_line']}")
        out.append("```")
        out.extend(s["lines"])
        out.append("```")
        out.append("")
    else:
        out.append("알 수 없습니다(근거 파일 없음)")
        out.append("")

    out.append("## cost 계산 라인(코드 근거)")
    out.append("")
    for pat in [r"daily_trading_cost\s*=", r"turnover_cost\s*=", r"if turnover_oneway\s*>\s*0", r"total_cost\s*=", r"net_ret\s*="]:
        for s in _snippet_by_regex(l7, pat, context=3, max_hits=2):
            out.append(f"- code: `{Path(s['path']).relative_to(project_root)}` L{s['start_line']}~L{s['end_line']}")
            out.append("```")
            out.extend(s["lines"])
            out.append("```")
            out.append("")

    out.append("## bt_returns 컬럼 정합성 체크")
    out.append("")
    expected = ["turnover_oneway", "daily_trading_cost", "turnover_cost", "total_cost", "gross_return", "net_return", "cost_bps", "cost_bps_used"]
    out.append(f"- expected_cols_from_code: `{expected}`")
    out.append("")
    out.append("| tag | bt_returns_source | missing_cols | mismatch |")
    out.append("|---|---|---|---|")

    for tag in [run_tag, baseline_tag]:
        src = out_dir / "charts" / f"{tag}__bt_returns.csv"
        df = None
        src_label = ""
        if src.exists():
            try:
                df = pd.read_csv(src)
                src_label = str(src.relative_to(project_root))
            except Exception:
                df = None
        if df is None:
            ap, _sel = _find_tag_or_root_artifact(project_root, tag, "bt_returns")
            if ap is not None:
                try:
                    df = _read_any_df(ap)
                    src_label = str(ap.relative_to(project_root))
                except Exception:
                    df = None
                    src_label = str(ap.relative_to(project_root))

        if df is None:
            out.append(f"| {tag} | 알 수 없습니다(근거 파일 없음) | {','.join(expected)} | mismatch |")
            continue
        cols = set(map(str, df.columns))
        missing = [c for c in expected if c not in cols]
        out.append(f"| {tag} | {src_label} | {','.join(missing)} | {'mismatch' if missing else 'ok'} |")

    out.append("")
    return "\n".join(out).rstrip() + "\n"


# =========================
# 추가 추출 6: 아티팩트 혼용 원인 규명
# =========================
def build_artifact_isolation_audit_outputs(*, project_root: Path, out_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    """
    [개선안 11번] charts_manifest.csv 기반:
    - tag 경로인지 fallback 경로인지 집계
    - fallback 비율이 높으면 tag 경로 파일 미존재를 missing_inputs로 열거
    """
    import pandas as pd  # type: ignore

    manifest = out_dir / "charts_manifest.csv"
    md: List[str] = []
    md.append("# Artifact Isolation Audit (tag vs fallback)")
    md.append("")
    md.append(f"- collected_at: `{_now_iso()}`")
    md.append("")

    if not manifest.exists():
        md.append("charts_manifest.csv: 알 수 없습니다(근거 파일 없음)")
        md.append("")
        return [], "\n".join(md)

    df = pd.read_csv(manifest)
    if df.empty:
        md.append("charts_manifest.csv가 비어있습니다. 알 수 없습니다(근거 파일 없음)")
        md.append("")
        return [], "\n".join(md)

    rows: List[Dict[str, Any]] = []
    missing_inputs: Dict[str, List[str]] = {}

    for _, r in df.iterrows():
        tag = str(r.get("tag") or "")
        artifact = str(r.get("artifact") or "")
        source_path = str(r.get("source_path") or "")
        norm = source_path.replace("\\", "/")
        is_tag = norm.startswith(f"data/interim/{tag}/")
        kind = "tag" if is_tag else ("missing" if not source_path else "fallback")

        miss_reason = ""
        if kind == "fallback":
            tag_dir = project_root / "data" / "interim" / tag
            cand_pq = tag_dir / f"{artifact}.parquet"
            cand_csv = tag_dir / f"{artifact}.csv"
            if not cand_pq.exists() and not cand_csv.exists():
                miss_reason = "missing_inputs"
                missing_inputs.setdefault(tag, []).append(f"{tag}/{artifact} (missing {cand_pq.name} and {cand_csv.name})")

        rows.append(
            {
                "tag": tag,
                "artifact": artifact,
                "status": str(r.get("status") or ""),
                "source_path": source_path,
                "source_kind": kind,
                "missing_reason": miss_reason,
            }
        )

    md.append("## 요약")
    md.append("")
    md.append("| tag | total | tag_path | fallback | missing | fallback_ratio |")
    md.append("|---|---:|---:|---:|---:|---:|")
    gdf = pd.DataFrame(rows)
    for tag, g in gdf.groupby("tag", sort=False):
        total = int(len(g))
        n_tag = int((g["source_kind"] == "tag").sum())
        n_fb = int((g["source_kind"] == "fallback").sum())
        n_miss = int((g["source_kind"] == "missing").sum())
        ratio = (n_fb / total) if total > 0 else 0.0
        md.append(f"| {tag} | {total} | {n_tag} | {n_fb} | {n_miss} | {ratio:.2f} |")
    md.append("")

    md.append("## missing_inputs (fallback 원인 후보)")
    md.append("")
    if not missing_inputs:
        md.append("알 수 없습니다(근거 파일 없음) 또는 fallback이 없습니다.")
        md.append("")
    else:
        for tag, items in missing_inputs.items():
            md.append(f"### {tag}")
            md.append("")
            for it in items:
                md.append(f"- {it}")
            md.append("")

    return rows, "\n".join(md).rstrip() + "\n"


# =========================
# Main
# =========================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    [개선안 11번] 실행 엔트리포인트.
    """
    import argparse

    ap = argparse.ArgumentParser(description="보고서 증거자료 자동 추출기 (reports/extract 생성)")
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    ap.add_argument("--baseline-tag", default=DEFAULT_BASELINE_TAG)
    args = ap.parse_args(argv)

    cwd = Path.cwd()
    resolved = resolve_project_root(cwd)
    project_root = resolved.project_root

    out_dir = project_root / "reports" / "extract"
    _ensure_dir(out_dir)

    # 1) git repro
    repro_git = collect_git_repro(project_root)
    _write_json(out_dir / "repro_git.json", repro_git)

    # 2) env repro
    repro_env = collect_env_repro()
    _write_json(out_dir / "repro_env.json", repro_env)

    # 3) commands log
    repro_cmd_md, _sources = collect_command_logs(project_root)
    _write_text(out_dir / "repro_commands.md", repro_cmd_md)

    # 4) CLI options
    run_all_path = project_root / "src" / "run_all.py"
    if run_all_path.exists():
        cli_rows = parse_run_all_cli_options(run_all_path)
        _write_text(out_dir / "cli_run_all.md", build_cli_md(cli_rows, title="CLI Options: src/run_all.py"))
    else:
        _write_text(out_dir / "cli_run_all.md", "# CLI Options: src/run_all.py\n\n알 수 없습니다 (src/run_all.py 없음)\n")

    # 5) config snapshot
    if resolved.config_path and resolved.config_path.exists():
        cfg_snapshot = extract_config_snapshot(resolved.config_path)
        _write_text(out_dir / "config_snapshot.md", build_config_snapshot_md(cfg_snapshot))
    else:
        cfg_snapshot = {"error": "configs/config.yaml not found", "l4": None, "l5": None, "l6": None, "l7": None}
        _write_text(out_dir / "config_snapshot.md", "# Config Snapshot (l4~l7)\n\n알 수 없습니다 (configs/config.yaml 없음)\n")

    # 6) features_final.json
    features_json = build_features_final_json(project_root=project_root, cfg_snapshot=cfg_snapshot, run_tag=args.run_tag)
    _write_json(out_dir / "features_final.json", features_json)

    # 7) backtest contract
    _write_text(out_dir / "backtest_contract.md", build_backtest_contract_md(project_root))

    # 8) KPI snapshot
    tags = [args.run_tag, args.baseline_tag]
    kpi_rows = build_kpi_snapshot_rows(project_root, tags)
    _write_csv(
        out_dir / "kpi_snapshot.csv",
        kpi_rows,
        fieldnames=["tag", "net_total_return", "net_cagr", "net_sharpe", "net_mdd", "cost_bps_used", "n_rebalances"],
    )

    # 9) charts export + manifest
    chart_manifest_rows = export_charts_and_manifest(project_root, tags, out_dir)
    _write_csv(
        out_dir / "charts_manifest.csv",
        chart_manifest_rows,
        fieldnames=["tag", "artifact", "status", "source_path", "exported_csv", "rows", "cols", "date_min", "date_max", "notes"],
    )

    # 10) integrity report
    _write_text(out_dir / "integrity_report.md", build_integrity_report_md(cfg_snapshot=cfg_snapshot, kpi_rows=kpi_rows))

    # 11) to_markdown hits via rg
    collect_to_markdown_hits(project_root, out_dir / "to_markdown_hits.txt")

    # =========================
    # 추가 추출물 (요구사항 보강)
    # =========================
    # [추가 1] KPI 단위 정규화/혼용 점검
    _write_text(
        out_dir / "kpi_units_check.md",
        build_kpi_units_check_md(project_root=project_root, out_dir=out_dir, run_tag=args.run_tag, baseline_tag=args.baseline_tag),
    )

    # [추가 2] phase별 기간 확정
    phase_rows = build_phase_date_ranges_rows(project_root=project_root, out_dir=out_dir, run_tag=args.run_tag, baseline_tag=args.baseline_tag)
    _write_csv(
        out_dir / "phase_date_ranges.csv",
        phase_rows,
        fieldnames=["tag", "phase", "date_min", "date_max", "n_rows", "source"],
    )

    # [추가 3] dataset_daily 품질(결측률/분포)
    dq_rows, dq_md = build_dataset_daily_quality_outputs(project_root=project_root, out_dir=out_dir, run_tag=args.run_tag)
    _write_csv(
        out_dir / "data_quality__dataset_daily.csv",
        dq_rows,
        fieldnames=[
            "feature",
            "exists",
            "nan_pct",
            "min",
            "p01",
            "median",
            "p99",
            "max",
            "rows",
            "cols",
            "ticker_nunique",
            "date_nunique",
            "ticker_date_dup_count",
            "dataset_path",
        ],
    )
    _write_text(out_dir / "data_quality__dataset_daily.md", dq_md)

    # [추가 4] 설정 존재 vs 코드 적용 매핑
    matrix_rows = build_config_vs_code_matrix_rows(project_root=project_root, out_dir=out_dir)
    _write_csv(
        out_dir / "config_vs_code_matrix.csv",
        matrix_rows,
        fieldnames=["section", "key", "configured", "config_value", "implemented", "code_hits", "code_file"],
    )

    # [추가 5] 거래비용 수식 확정 + 산출물 컬럼 불일치
    _write_text(
        out_dir / "cost_model_contract.md",
        build_cost_model_contract_md(project_root=project_root, out_dir=out_dir, run_tag=args.run_tag, baseline_tag=args.baseline_tag),
    )

    # [추가 6] 아티팩트 혼용(격리) 감사
    audit_rows, audit_md = build_artifact_isolation_audit_outputs(project_root=project_root, out_dir=out_dir)
    _write_csv(
        out_dir / "artifact_isolation_audit.csv",
        audit_rows,
        fieldnames=["tag", "artifact", "status", "source_path", "source_kind", "missing_reason"],
    )
    _write_text(out_dir / "artifact_isolation_audit.md", audit_md)

    # README mapping
    _write_text(out_dir / "README.md", build_readme_md())

    # 완료 표식(콘솔)
    print(f"[extract_report_evidence] done. out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
