module Kuiops.SHMem.Aligned

#lang-pulse

open Kuiper
open Kuiper.SHMem { is_block_array }

#push-options "--admit_smt_queries true"
let shmem_aligned16 (#et : Type0) (a : array et { is_block_array a }) = ()
#pop-options
