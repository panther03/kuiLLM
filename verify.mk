KUIPER_REPO := $(CURDIR)/kuiper

# $(KUIPER_REPO)/Makefile:
#  	$(error $@ not found; run `git submodule update --init --recursive` if you haven't)

verify-kuiper:
	+$(MAKE) -C $(KUIPER_REPO) minimal

OBJ := $(KUIPER_REPO)/obj

ROOTS := $(shell find kuiops/ -name '*.fst' -o -name '*.fsti')
CHECKED_FILES := $(foreach f, $(ROOTS), $(OBJ)/$(notdir $(f)).checked)

KRML_EXE  := $(KUIPER_REPO)/inst/bin/krml
FSTAR_EXE := $(KUIPER_REPO)/inst/bin/fstar.exe

FSTAR_FLAGS += --cache_dir $(OBJ)
FSTAR_FLAGS += --odir $(OBJ)
FSTAR_FLAGS += --warn_error -291 # inspect_ln warnings, benign
FSTAR_FLAGS += --warn_error -249-321
FSTAR_FLAGS += --warn_error @242@250 # 242, 250: abort if could not extract something
FSTAR_FLAGS += --z3version 4.13.3
FSTAR_FLAGS += --ext kuiper
FSTAR_FLAGS += --ext __unrefine
FSTAR_FLAGS += --ext no_krml_private
# FSTAR_FLAGS += --ext core_phase2
FSTAR_FLAGS += --warn_error -288 # using has_type (we only use it in SMT patterns)
# FSTAR_FLAGS += --ext krml_inline_all
# FSTAR_FLAGS += --error_contexts true
FSTAR_FLAGS += --ext context_pruning_no_ambients
FSTAR_FLAGS += --ext freshen
FSTAR = $(FSTAR_EXE) --include $(KUIPER_REPO)/src $(FSTAR_FLAGS)

# I HATE MAKE!
.SUFFIXES:
# .SECONDARY:
# ^ Don't ask me why, but SECONDARY makes this makefile very slow on
# no-ops. NOTINTERMEDIATE has a similar effect.
#.NOTINTERMEDIATE:
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

define msg =
@printf "  %-8s  %s\n" $(1) $(if $(2),$(2),$(shell realpath --relative-to=. $<))
endef

.depend: $(ROOTS) | verify-kuiper
	$(call msg,"DEPEND",$@)
	$(FSTAR) --codegen krml --already_cached 'FStar,LowStar,Prims,Pulse,PulseCore,Kuiper,Klas' --dep full $(ROOTS) -o $@

include .depend

$(OBJ)/%.checked:
	@$(call msg,"CHECK")
	@$(FSTAR) --already_cached '*' -c $< -o $@
	@touch -c $@

verify-kuiops: .depend $(CHECKED_FILES)
