#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f /home/bjh/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/bjh/anaconda3/etc/profile.d/conda.sh
  conda activate NLP
fi

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export HF_HUB_DISABLE_TELEMETRY=1

MANIFEST_DIR="${PROJECT_ROOT}/data/raw/_download_manifests"
mkdir -p \
  "${MANIFEST_DIR}" \
  "${HF_HOME}" \
  data/raw/fineweb_edu \
  data/raw/finemath_or_openwebmath \
  data/raw/pes2o \
  data/raw/pg19 \
  data/raw/wikipedia_en \
  logs

echo "Project root: ${PROJECT_ROOT}"
echo "Python: $(command -v python)"
echo "HF_HOME: ${HF_HOME}"

if [[ "${USE_EXISTING_MANIFESTS:-0}" == "1" && -s "${MANIFEST_DIR}/all_datasets.json" ]]; then
  echo "USE_EXISTING_MANIFESTS=1, using manifests already present in ${MANIFEST_DIR}."
else
python - <<'PY'
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

project_root = Path.cwd()
manifest_dir = project_root / "data/raw/_download_manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)

datasets = [
    {
        "name": "fineweb_edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "api_path": "sample/10BT",
        "pattern": r"^sample/10BT/.+\.parquet$",
        "local_dir": "data/raw/fineweb_edu",
    },
    {
        "name": "finemath_or_openwebmath",
        "repo": "HuggingFaceTB/finemath",
        "api_path": "finemath-4plus",
        "pattern": r"^finemath-4plus/train-.+\.parquet$",
        "local_dir": "data/raw/finemath_or_openwebmath",
    },
    {
        "name": "pes2o",
        "repo": "allenai/peS2o",
        "api_path": "data/v3",
        "pattern": r"^data/v3/train-.+\.zst$",
        "local_dir": "data/raw/pes2o",
    },
    {
        "name": "pg19",
        "repo": "emozilla/pg19",
        "api_path": "data",
        "pattern": r"^data/train-.+\.parquet$",
        "local_dir": "data/raw/pg19",
    },
    {
        "name": "wikipedia_en",
        "repo": "wikimedia/wikipedia",
        "api_path": "20231101.en",
        "pattern": r"^20231101\.en/train-.+\.parquet$",
        "local_dir": "data/raw/wikipedia_en",
    },
]


def fetch_tree(repo: str, api_path: str) -> list[dict]:
    quoted = "/".join(urllib.parse.quote(part) for part in api_path.split("/"))
    url = (
        "https://huggingface.co/api/datasets/"
        + urllib.parse.quote(repo, safe="/")
        + "/tree/main/"
        + quoted
        + "?recursive=false&expand=false"
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "45",
                    "--user-agent",
                    "qwen3-01b-corpus-downloader",
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout)
        except Exception as exc:  # pragma: no cover - operational retry path
            last_error = exc
            print(f"tree fetch failed for {repo}/{api_path}, attempt {attempt}: {exc}", file=sys.stderr)
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"failed to fetch tree for {repo}/{api_path}: {last_error}")


all_records: list[dict] = []
for spec in datasets:
    regex = re.compile(spec["pattern"])
    files = []
    for item in fetch_tree(spec["repo"], spec["api_path"]):
        path = item.get("path", "")
        if item.get("type") != "file" or not regex.match(path):
            continue
        size = int(item.get("size") or item.get("lfs", {}).get("size") or 0)
        filename = Path(path).name
        local_dir = project_root / spec["local_dir"]
        files.append(
            {
                "dataset": spec["name"],
                "repo": spec["repo"],
                "path": path,
                "filename": filename,
                "local_dir": str(local_dir),
                "local_path": str(local_dir / filename),
                "size": size,
                "url": f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{path}",
            }
        )
    files.sort(key=lambda row: row["path"])
    if not files:
        raise RuntimeError(f"no files matched for {spec['name']} in {spec['repo']}/{spec['api_path']}")

    total = sum(row["size"] for row in files)
    print(f"{spec['name']}: {len(files)} files, {total / 1024**3:.2f} GiB")

    (manifest_dir / f"{spec['name']}.json").write_text(
        json.dumps(files, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (manifest_dir / f"{spec['name']}.aria2.txt").open("w", encoding="utf-8") as f:
        for row in files:
            f.write(row["url"] + "\n")
            f.write(f"  dir={row['local_dir']}\n")
            f.write(f"  out={row['filename']}\n")
    all_records.extend(files)

(manifest_dir / "all_datasets.json").write_text(
    json.dumps(all_records, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"TOTAL: {len(all_records)} files, {sum(row['size'] for row in all_records) / 1024**3:.2f} GiB")
PY
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not starting downloads."
  exit 0
fi

if command -v aria2c >/dev/null 2>&1; then
  downloader=(aria2c
    --continue=true
    --auto-file-renaming=false
    --allow-overwrite=true
    --conditional-get=true
    --check-certificate=true
    --max-tries=0
    --retry-wait=30
    --timeout=120
    --connect-timeout=60
    --max-connection-per-server="${ARIA2_CONN_PER_SERVER:-4}"
    --split="${ARIA2_SPLIT:-4}"
    --min-split-size=64M
    --max-concurrent-downloads="${ARIA2_JOBS:-2}"
    --summary-interval=60
    --console-log-level=notice
  )

  selected="${ONLY_DATASETS:-fineweb_edu finemath_or_openwebmath pes2o pg19 wikipedia_en}"
  for name in ${selected}; do
    input="${MANIFEST_DIR}/${name}.aria2.txt"
    if [[ ! -s "${input}" ]]; then
      echo "Missing aria2 input: ${input}" >&2
      exit 1
    fi
    echo "Starting download: ${name}"
    "${downloader[@]}" --input-file="${input}"
  done
else
  echo "aria2c is required for this downloader but was not found." >&2
  exit 1
fi

python - <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest = Path("data/raw/_download_manifests/all_datasets.json")
records = json.loads(manifest.read_text(encoding="utf-8"))
bad = []
by_dataset = {}
for row in records:
    path = Path(row["local_path"])
    size = int(row["size"])
    ok = path.is_file() and (size <= 0 or path.stat().st_size == size)
    by_dataset.setdefault(row["dataset"], [0, 0, 0])
    by_dataset[row["dataset"]][0] += 1
    by_dataset[row["dataset"]][1] += 1 if ok else 0
    by_dataset[row["dataset"]][2] += path.stat().st_size if path.exists() else 0
    if not ok:
        actual = path.stat().st_size if path.exists() else -1
        bad.append((row["dataset"], row["filename"], size, actual))

for name, (total, ok, bytes_done) in sorted(by_dataset.items()):
    print(f"{name}: {ok}/{total} verified, {bytes_done / 1024**3:.2f} GiB on disk")

if bad:
    print("Incomplete or size-mismatched files:", file=sys.stderr)
    for dataset, filename, expected, actual in bad[:50]:
        print(f"  {dataset}/{filename}: expected={expected} actual={actual}", file=sys.stderr)
    raise SystemExit(2)

print("All selected pretraining corpus files verified.")
PY
