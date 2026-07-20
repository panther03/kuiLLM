.PHONY: infer infer-kuiper infer-batched infer-no-kuiper verify dump-kernels profile-kuiper-nsys profile-no-kuiper-nsys golden golden-compiled profile-golden profile-golden-triton profile-golden-no-triton test verify-kuiops install-kuiper

# Default install location (cwd/inst), matching install_kuiper.sh.
KUIPER_INST ?= $(CURDIR)/inst

NSYS := nsys profile --force-overwrite=true -t cuda --cuda-graph-trace=node
NSYS_RANGE := nsys profile --force-overwrite=true -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop

# Compiled Qwen2.5 with Kuiper kernels hooked into torch.compile (default).
infer-kuiper:
	python3 infer.py

# Build every matched kernel in one combined compilation (batch compile).
infer-batched:
	python3 infer.py --batch-compile

infer: infer-batched

# Stock torch.compile baseline (identical to etc/infer_golden_compiled.py).
infer-no-kuiper:
	python3 infer.py --no-kuiper

# Check every Kuiper op against stock PyTorch (relative-Frobenius tolerance).
verify:
	python3 infer.py --verify

# Trace the ops the compiled graph runs and (re)write KERNELS.md.
dump-kernels:
	python3 infer.py --dump-kernels KERNELS.md

# Profile only the measured decode (nsys --capture-range=cudaProfilerApi via --nsys).
profile-kuiper-nsys:
	$(NSYS_RANGE) -o data/kuiper.nsys-rep python3 infer.py --nsys

profile-no-kuiper-nsys:
	$(NSYS_RANGE) -o data/no-kuiper.nsys-rep python3 infer.py --no-kuiper --nsys

# --- Steelman pure-PyTorch reference (etc/infer_golden.py) ---
# Hand-written Qwen2 forward, single hyper-optimized path (batch 256, bf16),
# manual fully-folded decode CUDA graph. No flags / no A-B modes.
golden:
	python3 etc/infer_golden.py --batch 256

# torch.compile(reduce-overhead) steelman (the baseline infer.py --no-kuiper matches).
golden-compiled:
	python3 etc/infer_golden_compiled.py --batch 256

profile-golden:
	$(NSYS_RANGE) -o data/golden.nsys-rep python3 etc/infer_golden.py --nsys --batch 256

# The old flag-laden A/B experiment (transformers + monkeypatch) lives in
# infer_golden_exp.py: Inductor/Triton reduce-overhead vs forced no-triton path.
profile-golden-triton:
	$(NSYS_RANGE) -o data/golden-triton.nsys-rep python3 etc/infer_golden_exp.py --nsys --batch 256

profile-golden-no-triton:
	$(NSYS_RANGE) -o data/golden-no-triton.nsys-rep python3 etc/infer_golden_exp.py --nsys --no-triton --manual-cudagraph --batch 256

# control # of fstar worker processes for JIT generation in tests
NPROCS ?= 4
test:
	KUIPY_JIT_NVCC_FAST=1 python3 -m pytest tests/ -n $(NPROCS)

_reset-kuiper-touch:
	@rm -f .kuiper.touch

install-kuiper: _reset-kuiper-touch
	@$(MAKE) .kuiper.touch # EVIL !!!! 

# TODO: "update-kuiper" for faster iteration (just copy over the .checked files without removing the whole thing)

.kuiper.touch:
	@if [ -z "$(KUIPER_HOME)" ]; then \
		echo "Error: KUIPER_HOME is not defined." >&2; \
		exit 1; \
	fi
	@+set -e; \
	KUIPER_HOME=$$(realpath "$(KUIPER_HOME)"); \
	KUIPER_INST=$$(realpath -m "$(KUIPER_INST)"); \
	$(MAKE) -C "$$KUIPER_HOME" -f verify.mk prepare verify-all ADMIT=1; \
	rm -rf "$$KUIPER_INST"; \
	cp -r "$$KUIPER_HOME/inst" "$$KUIPER_INST"; \
	mkdir -p "$$KUIPER_INST/lib/fstar/kuiper.checked/"; \
	find "$$KUIPER_HOME/obj" -name "*.checked" -type f -exec cp {} "$$KUIPER_INST/lib/fstar/kuiper.checked/" \; ; \
	mkdir -p "$$KUIPER_INST/lib/fstar/kuiper"; \
	find "$$KUIPER_HOME/src" -type f \( -name "*.fst" -o -name "*.fsti" \) -exec cp {} "$$KUIPER_INST/lib/fstar/kuiper/" \; ; \
	echo "kuiper" >> "$$KUIPER_INST/lib/fstar/fstar.include"; \
	echo "kuiper.checked" >> "$$KUIPER_INST/lib/fstar/fstar.include"; \
	mkdir -p "$$KUIPER_INST/kuiper_extr/"; \
	cp -r "$$KUIPER_HOME"/extraction/dune/_build/default/kuiper_extr* "$$KUIPER_INST/kuiper_extr/"; \
	cp "$$KUIPER_HOME"/scripts/fixup.sed "$$KUIPER_INST/"; \
	mkdir -p "$$KUIPER_INST"/include/kuiper; \
	cp -r "$$KUIPER_HOME"/include/* "$$KUIPER_INST"/include/kuiper; \
	touch .kuiper.touch

verify-kuiops: .kuiper.touch
	@+$(MAKE) -f verify.mk verify-kuiops