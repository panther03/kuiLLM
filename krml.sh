#!/bin/bash

if [ -z "$KUIPER_INST" ]; then
  KUIPER_INST=$(pwd)/inst
fi

mkdir -p .kuipy_cache/pre
gcmd () {
	echo "$KUIPER_INST/bin/krml \
    -add-early-include <kuiper.h> \
    -add-early-include <kuiops_compat.h> \
    -fc++-compat \
    -fcast-allocations \
    -skip-compilation \
    -skip-makefiles \
    -faggressive-inlining \
    -fauto-for-loops \
    -fnoshort-enums \
    -cuda \
    -dbacktrace \
    -silent \
    -drop Prims \
    -minimal \
    -header /dev/null \
    -warn-error @6 \
    -warn-error -2@4-10@18"
}

exec $(gcmd) "$@"
