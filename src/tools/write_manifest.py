# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/write_manifest.py
"""
Manifest 생성 스크립트
run_tag, git_commit, config 해시, 생성 파일 리스트, 파일 크기, mtime 기록
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def get_git_commit(repo_dir: Path) -> Optional[str]:
    """Git 커밋 해시 반환"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def compute_config_hash(config_path: Path) -> str:
    """Config 파일의 해시 계산"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    except Exception:
        return "unknown"

def scan_artifacts(interim_dir: Path, run_tag: str) -> List[Dict]:
    """Interim 디렉토리에서 생성된 파일 스캔"""
    artifacts = []

    # 태그 기반 디렉토리
    tag_dir = interim_dir / run_tag
    if tag_dir.exists():
        scan_dir = tag_dir
    else:
        # 레거시 모드: 루트 디렉토리
        scan_dir = interim_dir

    # parquet, csv, meta 파일 스캔
    for ext in [".parquet", ".csv", "__meta.json"]:
        for file_path in scan_dir.glob(f"*{ext}"):
            if file_path.is_file():
                stat = file_path.stat()
                artifacts.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(interim_dir.parent.parent)),
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": ext.replace(".", "").replace("__meta.json", "meta"),
                })

    # 정렬: 이름 순
    artifacts.sort(key=lambda x: x["name"])
    return artifacts

def main():
    parser = argparse.ArgumentParser(description="Generate manifest for pipeline run")
    parser.add_argument("--run-tag", type=str, required=True, help="Run tag")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--interim-dir", type=str, default=None, help="Interim directory (default: from config)")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument("--out-dir", type=str, default="reports/manifests", help="Output directory")
    args = parser.parse_args()

    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]

    config_path = root / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Interim 디렉토리 결정
    if args.interim_dir:
        interim_dir = Path(args.interim_dir)
    else:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            sys.path.insert(0, str(root / "src"))
            from src.utils.config import get_path
            interim_dir = Path(get_path(cfg, "data_interim"))
        except Exception as e:
            print(f"ERROR: Failed to get interim_dir from config: {e}", file=sys.stderr)
            sys.exit(1)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Git 커밋
    git_commit = get_git_commit(root)

    # Config 해시
    config_hash = compute_config_hash(config_path)

    # 생성 파일 스캔
    artifacts = scan_artifacts(interim_dir, args.run_tag)

    # Manifest 생성
    # interim_dir 경로 처리 (절대 경로일 수 있음)
    try:
        interim_dir_rel = str(interim_dir.relative_to(root))
    except ValueError:
        # 절대 경로인 경우 그대로 사용
        interim_dir_rel = str(interim_dir)

    manifest = {
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit,
        "config_path": str(config_path.relative_to(root)),
        "config_hash": config_hash,
        "interim_dir": interim_dir_rel,
        "artifacts_count": len(artifacts),
        "artifacts": artifacts,
    }

    # JSON 저장
    json_path = out_dir / f"manifest__{args.run_tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Manifest] Saved: {json_path}")
    print(f"[Manifest] Run Tag: {args.run_tag}")
    print(f"[Manifest] Git Commit: {git_commit or 'N/A'}")
    print(f"[Manifest] Config Hash: {config_hash}")
    print(f"[Manifest] Artifacts: {len(artifacts)} files")

    return json_path

if __name__ == "__main__":
    main()
