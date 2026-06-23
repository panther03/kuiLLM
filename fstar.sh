#!/bin/bash

if [ -z "$KUIPER_INST" ]; then
  KUIPER_INST=$(pwd)/inst
fi

mkdir -p .kuipy_cache/checked
gcmd () {
	echo "$KUIPER_INST/bin/fstar.exe \
    --silent \
    --include kuiops \
    --cache_dir .kuipy_cache/checked \
    --odir  .kuipy_cache/checked \
    --warn_error  -291 \
    --warn_error  -249-321 \
    --warn_error  @242@250 \
    --z3version   4.13.3 \
    --ext  kuiper \
    --ext  __unrefine \
    --ext  no_krml_private \
    --warn_error  -288 \
    --ext  context_pruning_no_ambients \
    --ext  freshen"
}

exec $(gcmd) "$@"