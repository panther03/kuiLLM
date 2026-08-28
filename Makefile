include kuiper.mk

.DEFAULT_GOAL := verify-kuiops
.PHONY: infer infer-kuiper infer-batched infer-no-kuiper verify dump-kernels \
	profile-kuiper-nsys profile-no-kuiper-nsys golden golden-compiled \
	profile-golden profile-golden-triton profile-golden-no-triton test \
	prepare verify-kuiops lint lint-fstar lint-generated list-admits

KUIPER_INSTALLER_URL ?= https://raw.githubusercontent.com/FStarLang/kuiper/main/scripts/install-kuiper.sh

NSYS := nsys profile --force-overwrite=true -t cuda --cuda-graph-trace=node
NSYS_RANGE := nsys profile --force-overwrite=true -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop

# Compiled Qwen2.5 with Kuiper kernels hooked into torch.compile (default).
infer-kuiper: prepare
	python3 infer.py

# Build every matched kernel in one combined compilation (batch compile).
# WARNING: do not use this target to measure performance. The CUDA graph is
# recorded during the warm-up that runs inside the batch capture, at which point
# every matched op is still deferred to stock PyTorch, so the timed decode
# replays a graph containing no Kuiper kernels and reports the stock number
# (155.5 vs 65.2 tok/s/seq on an A6000). Use `make infer-kuiper`. See the
# comment at the batch_capture() call in infer.py.
infer-batched: prepare
	python3 infer.py --batch-compile

infer: infer-batched

# Stock torch.compile baseline (identical to etc/infer_golden_compiled.py).
infer-no-kuiper:
	python3 infer.py --no-kuiper

# Check every Kuiper op against stock PyTorch (relative-Frobenius tolerance).
verify: prepare
	python3 infer.py --verify

# Trace the compiled graph and write its visualization and inventory to KERNELS.md.
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

# Number of parallel pytest workers (requires pytest-xdist). Each worker drives
# its own F*/karamel extraction; since a build now only reads the shared
# .checked cache (see kuipy/toolchain.py) these run fully in parallel, and a
# cold-cache run is dominated by F* extraction, so this scales well.
NPROCS ?= 12
test: prepare
	KUIPY_JIT_NVCC_FAST=1 python3 -m pytest tests/ -n $(NPROCS)

prepare: $(KUIPER_MARKER) $(CLANG_FORMAT) $(KUIPER_PLUGIN).cmxs

ifeq ($(KUIPER_HOME_MANAGED),1)
$(KUIPER_MARKER):
	@set -e; \
	workdir=$$(mktemp -d); \
	trap 'rm -rf "$$workdir"' EXIT; \
	curl -fsSL "$(KUIPER_INSTALLER_URL)" -o "$$workdir/install-kuiper.sh"; \
	bash "$$workdir/install-kuiper.sh" --nightly --version "$(KUIPER_NIGHTLY)" \
		--dest "$$workdir/kuiper" --no-link; \
	test -f "$$workdir/kuiper/.packaged"; \
	rm -rf "$(KUIPER_HOME)"; \
	mv "$$workdir/kuiper" "$(KUIPER_HOME)"; \
	rm -rf .kuipy_cache; \
	rm -f .depend; \
	touch "$@"
else
$(KUIPER_MARKER):
	$(error KUIPER_HOME does not contain a binary Kuiper package: $(KUIPER_HOME))
endif

$(CLANG_FORMAT): scripts/install-clang-format.sh
	@CLANG_FORMAT_VERSION=$(CLANG_FORMAT_VERSION) \
		./scripts/install-clang-format.sh "$(TOOLS_DIR)/clang-format-$(CLANG_FORMAT_VERSION)"

$(KUIPER_PLUGIN).cmxs: $(KUIPER_PLUGIN_SOURCE) | $(KUIPER_MARKER)
	@mkdir -p "$(dir $(KUIPER_PLUGIN))"
	@ln -sf "$(KUIPER_PLUGIN_SOURCE)" "$@"

verify-kuiops: prepare
	@+$(MAKE) -f verify.mk verify-kuiops

lint: lint-fstar lint-generated

lint-fstar: $(KUIPER_MARKER)
	( cd kuiops && "$(KUIPER_HOME)/scripts/git-sed" 's/[[:space:]]*$$//' )
	( cd kuiops && "$(KUIPER_HOME)/scripts/find-pulse-noix.sh" )
	( cd kuiops && "$(KUIPER_HOME)/scripts/check-attrs.sh" )

lint-generated:
	@for sh in $$(find kuiops -type f -name '*.fst.sh'); do \
		fst=$${sh%.sh}; \
		if git ls-files --error-unmatch "$$fst" >/dev/null 2>&1; then \
			echo "ERROR: $$fst is generated from $$sh and should not be tracked" >&2; \
			exit 1; \
		fi; \
	done

list-admits: $(KUIPER_MARKER)
	@git ls-files -z -- \
		':(glob)kuiops/*.fst' ':(glob)kuiops/*.fsti' \
		':(glob)kuiops/**/*.fst' ':(glob)kuiops/**/*.fsti' | \
		xargs -0 python3 "$(KUIPER_HOME)/scripts/list-admits.py"

# Delegate
.PHONY: .force
.PRECIOUS: $(filter .kuipy_cache/%,$(MAKECMDGOALS))
$(filter .kuipy_cache/%,$(MAKECMDGOALS)) &: .force
	$(MAKE) -f verify.mk $(filter .kuipy_cache/%,$(MAKECMDGOALS))
