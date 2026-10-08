#!/usr/bin/env bash
# =============================================================================
# install_superfocus.sh -- one-shot installer for the SuperFocus conda env
#
# Related manuscript: https://doi.org/10.64898/2025.12.26.696575
# Code to reproduce the results in the manuscript is in the tutorials folder.
#
# Usage:
#   bash install_superfocus.sh [-f ENV_YML] [-n ENV_NAME | -p ENV_PREFIX] [--force]
#
#   -f FILE     environment file  (default: ./SuperFocus_final_environment.yml)
#   -n NAME     environment name  (default: the "name:" field of the yml)
#   -p PREFIX   install to an explicit prefix path instead of a named env
#   --force     remove an existing env with that name/prefix and rebuild
#   -h          show this help
#
# Steps (same order as the manual recipe):
#   1. conda env create -f <yml>
#   2. pip install git+https://github.com/JEFworks-Lab/STalign.git
#   3. conda env update -f <yml>
#   4. pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 (cu118 index)
#   5. pip install numpy==1.26.4
#   6. pip install --force-reinstall setuptools==69.1.1
#   +  version check of the pinned packages
#
# Everything is tee'd to install_superfocus_<timestamp>.log in the cwd.
# =============================================================================
set -euo pipefail

YML="SuperFocus_final_environment.yml"
ENV_NAME=""
ENV_PREFIX=""
FORCE=0

TORCH_VER="2.0.1+cu118"
TV_VER="0.15.2+cu118"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
NUMPY_VER="1.26.4"
SETUPTOOLS_VER="69.1.1"
STALIGN_URL="git+https://github.com/JEFworks-Lab/STalign.git"

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }
log()   { printf '\n==> [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()   { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------- arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f) YML="${2:?-f needs a file}"; shift 2 ;;
    -n) ENV_NAME="${2:?-n needs a name}"; shift 2 ;;
    -p) ENV_PREFIX="${2:?-p needs a path}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1" ;;
  esac
done

[[ -f "$YML" ]] || die "environment file not found: $YML"
[[ -n "$ENV_NAME" && -n "$ENV_PREFIX" ]] && die "use either -n or -p, not both"

# Default env name = "name:" field of the yml (tolerates quotes and CRLF).
if [[ -z "$ENV_NAME" && -z "$ENV_PREFIX" ]]; then
  ENV_NAME="$(awk -F':' '/^name:/{print $2; exit}' "$YML" | tr -d ' \t\r"'"'")"
  [[ -n "$ENV_NAME" ]] || die "no 'name:' field in $YML -- pass -n NAME or -p PREFIX"
fi

if [[ -n "$ENV_PREFIX" ]]; then
  TARGET=(-p "$ENV_PREFIX")
  ENV_LABEL="$ENV_PREFIX"
else
  TARGET=(-n "$ENV_NAME")
  ENV_LABEL="$ENV_NAME"
fi

# ------------------------------------------------------------------- logging
LOG="install_superfocus_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; [[ $rc -ne 0 ]] && printf "\nInstallation FAILED (exit %s). See %s\n" "$rc" "$LOG"' EXIT

# --------------------------------------------------------------------- conda
if ! command -v conda >/dev/null 2>&1 && type module >/dev/null 2>&1; then
  set +u; module load miniconda || true; set -u     # e.g. YCRC clusters
fi
command -v conda >/dev/null 2>&1 \
  || die "conda not found. Load it first (e.g. 'module load miniconda') and rerun."

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Keep ~/.local site-packages from leaking into the env during pip installs.
export PYTHONNOUSERSITE=1

env_exists() {
  if [[ -n "$ENV_PREFIX" ]]; then
    [[ -d "$ENV_PREFIX/conda-meta" ]]
  else
    # Capture first: 'conda ... | grep -q' can SIGPIPE conda under pipefail.
    local envs
    envs="$(conda env list --json)"
    grep -qF "/${ENV_NAME}\"" <<<"$envs"
  fi
}

log "Environment: $ENV_LABEL   |   file: $YML   |   log: $LOG"

# -------------------------------------------------- 1. conda env create
SKIP_CREATE=0
if env_exists; then
  if (( FORCE )); then
    log "Removing existing environment (--force)"
    conda env remove -y "${TARGET[@]}"
  else
    log "Step 1/6: environment already exists -- skipping create (use --force to rebuild)"
    SKIP_CREATE=1
  fi
fi

if (( ! SKIP_CREATE )); then
  log "Step 1/6: conda env create"
  # The pip stage of the yml can fail on this first pass (that is what the
  # STalign install + 'conda env update' in steps 2-3 repair). So a non-zero
  # exit is tolerated here *iff* the conda part succeeded and the env exists.
  if ! conda env create -f "$YML" "${TARGET[@]}"; then
    env_exists || die "conda env create failed before the environment was created (solver/download error)."
    echo "WARNING: 'conda env create' exited non-zero, but the env exists; continuing (step 3 must succeed)."
  fi
fi

# ------------------------------------------------------------ activate env
set +u
conda activate "${ENV_PREFIX:-$ENV_NAME}"
set -u
[[ "$(command -v python)" == "$CONDA_PREFIX/bin/python" ]] \
  || die "python on PATH ($(command -v python)) is not the env's python ($CONDA_PREFIX/bin/python)"
log "Active env prefix: $CONDA_PREFIX"

# -------------------------------------------------- 2. STalign
log "Step 2/6: pip install STalign"
python -m pip install "$STALIGN_URL"

# -------------------------------------------------- 3. conda env update
log "Step 3/6: conda env update"
conda env update -f "$YML" "${TARGET[@]}"

# -------------------------------------------------- 4. torch / torchvision
log "Step 4/6: torch==$TORCH_VER torchvision==$TV_VER"
python -m pip install "torch==$TORCH_VER" "torchvision==$TV_VER" \
  --index-url "$TORCH_INDEX" --force-reinstall

# -------------------------------------------------- 5. numpy
log "Step 5/6: numpy==$NUMPY_VER"
python -m pip install "numpy==$NUMPY_VER"

# -------------------------------------------------- 6. setuptools
log "Step 6/6: setuptools==$SETUPTOOLS_VER"
python -m pip install --force-reinstall "setuptools==$SETUPTOOLS_VER"

# -------------------------------------------------- verification
log "Verifying pinned versions"
TORCH_VER="$TORCH_VER" TV_VER="$TV_VER" NUMPY_VER="$NUMPY_VER" SETUPTOOLS_VER="$SETUPTOOLS_VER" \
python - <<'PY'
import os, sys
from importlib.metadata import version, PackageNotFoundError

expected = {
    "torch": os.environ["TORCH_VER"],
    "torchvision": os.environ["TV_VER"],
    "numpy": os.environ["NUMPY_VER"],
    "setuptools": os.environ["SETUPTOOLS_VER"],
    "STalign": None,          # any version, must be installed
}
ok = True
for pkg, want in expected.items():
    try:
        got = version(pkg)
    except PackageNotFoundError:
        got = None
    good = got is not None and (want is None or got == want)
    ok &= good
    print(f"  {'OK  ' if good else 'FAIL'} {pkg:<12} installed={got}  expected={want or 'any'}")

try:
    import torch
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}"
          "   (False is expected on a node without a GPU)")
except Exception as e:                       # noqa: BLE001
    ok = False
    print(f"  FAIL import torch: {e!r}")

try:
    import STalign                           # noqa: F401
    print("  OK   import STalign")
except Exception as e:                       # noqa: BLE001
    print(f"  WARN import STalign raised: {e!r}")

sys.exit(0 if ok else 1)
PY

log "pip check (informational only)"
python -m pip check || true

log "Done. Activate with:  conda activate ${ENV_PREFIX:-$ENV_NAME}"
