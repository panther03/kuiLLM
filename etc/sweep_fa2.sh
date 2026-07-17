#!/usr/bin/env bash
# Sweep FA2 prefill tile params: compile with each (WARPS_M, BN) and report
# correctness + latency, so we tune empirically instead of guessing.
set -u
export PATH=/usr/local/cuda-13/bin:$PATH
cd "$(dirname "$0")"
MODE="${1:-prefill}"
echo "=== FA2 sweep ($MODE) ==="
for WM in 2 4 8; do
  for KS in 1 2 4 8; do
    out=$(nvcc -arch=sm_86 -O3 -DWARPS_M=$WM -DKSTEP=$KS dev_fa2.cu -o /tmp/fa2_${WM}_${KS} 2>/tmp/nvcc_err_${WM}_${KS})
    if [ $? -ne 0 ]; then
      echo "WM=$WM KSTEP=$KS  COMPILE FAIL"; continue
    fi
    /tmp/fa2_${WM}_${KS} "$MODE"
  done
done
