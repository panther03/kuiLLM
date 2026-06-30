.PHONY: infer infer-kuiper infer-no-kuiper profile-kuiper-calls profile-no-kuiper-calls profile-kuiper-nsys profile-no-kuiper-nsys test verify-kuiops install-kuiper

# Default install location (cwd/inst), matching install_kuiper.sh.
KUIPER_INST ?= $(CURDIR)/inst

infer: infer-kuiper
infer-kuiper:
	python3 infer.py

infer-no-kuiper:
	python3 infer.py --no-kuiper

profile-kuiper-calls:
	KUIPY_PRINT_PROFILING=1 python3 infer.py

profile-no-kuiper-calls:
	KUIPY_PRINT_PROFILING=1 python3 infer.py --no-kuiper

profile-kuiper-nsys:
	nsys profile -o data/kuiper.nsys-rep --force-overwrite=true -t cuda python3 infer.py

profile-no-kuiper-nsys:
	nsys profile -o data/no-kuiper.nsys-rep --force-overwrite=true -t cuda python3 infer.py --no-kuiper

test:
	python3 -m pytest tests/

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