#!/usr/bin/env bash

set -euo pipefail

KUIPER_HOME=${KUIPER_HOME:-"$(pwd)/.kuiper"}
mkdir -p .kuipy_cache/checked

exec "$KUIPER_HOME/inst/bin/fstar.exe" \
  --silent \
  --include "$KUIPER_HOME/src" \
  --include "$KUIPER_HOME/obj" \
  --include kuiops \
  --already_cached '*,-Kuiops' \
  --cache_dir .kuipy_cache/checked \
  --odir .kuipy_cache/checked \
  --warn_error -291 \
  --warn_error -249-321 \
  --warn_error @242@250 \
  --z3version 4.13.3 \
  --ext kuiper \
  --ext __unrefine \
  --ext no_krml_private \
  --warn_error -288 \
  --ext context_pruning_no_ambients \
  --ext freshen \
  "$@"
