#!/usr/bin/env bash

set -euo pipefail

per_job_mib=${1:?usage: ci-jobs.sh MIB_PER_JOB}
[[ $per_job_mib =~ ^[1-9][0-9]*$ ]] || exit 2

cpu_jobs=$(nproc)
available_kib=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)

if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
  limit=$(< /sys/fs/cgroup/memory.max)
  current=$(< /sys/fs/cgroup/memory.current)
  if [[ $limit =~ ^[0-9]+$ && $current =~ ^[0-9]+$ && $limit -gt $current ]]; then
    cgroup_available_kib=$(( (limit - current) / 1024 ))
    (( cgroup_available_kib >= available_kib )) || available_kib=$cgroup_available_kib
  fi
fi

memory_jobs=$(( available_kib / 1024 / per_job_mib ))
(( memory_jobs >= 1 )) || memory_jobs=1
(( cpu_jobs <= memory_jobs )) && echo "$cpu_jobs" || echo "$memory_jobs"
