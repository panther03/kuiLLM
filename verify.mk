OBJ := .kuipy_cache/checked
PRE := .kuipy_cache/pre
CU  := .kuipy_cache/cu
KUIPER_INST ?= $(CURDIR)/inst
PLUGIN := $(KUIPER_INST)/kuiper_extr/kuiper_extr
FIXUP := $(KUIPER_INST)/fixup.sed

# Keep intermediates (pre/*.cu, *.krml) around, and stay clear of make's
# builtin rules, which would happily try to build .cu files from nothing.
.SECONDARY:
.DELETE_ON_ERROR:
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

ROOTS := $(shell find kuiops/ -name '*.fst' -o -name '*.fsti')
CHECKED_FILES := $(foreach f, $(ROOTS), $(OBJ)/$(notdir $(f)).checked)

# Recover a module name from the underscored form F*/karamel use for filenames.
# A blind s/_/./ is wrong: JIT instantiation modules have underscores inside
# their last segment (Kuiops.Mm.<...>.P_bm128_bn32_...), so instead match the
# stem against the (dot-separated) source file names.
mod_of = $(basename $(notdir $(firstword $(foreach f,$(ROOTS),\
  $(if $(filter $(1),$(subst .,_,$(basename $(notdir $(f))))),$(f))))))

define msg =
@printf "  %-8s  %s\n" $(1) $(if $(2),$(2),$(shell realpath --relative-to=. $<))
endef

mkobj:
	@mkdir -p $(OBJ) $(PRE) $(CU)

.depend: $(ROOTS) | mkobj
	$(call msg,"DEPEND",$@)
	@$(CURDIR)/fstar.sh --codegen krml --already_cached '*,-Kuiops' --dep full $(ROOTS) -o $@

include .depend

ifneq ($(ADMIT),)
OTHERFLAGS += --admit_smt_queries true
endif

$(OBJ)/%.checked:
ifneq ($(ADMIT),)
	@$(call msg,"CHECK(A)")
else
	@$(call msg,"CHECK")
endif
	@$(CURDIR)/fstar.sh --already_cached '*' $(OTHERFLAGS) -c $< -o $@
	@touch -c $@

verify-kuiops: .depend $(CHECKED_FILES)

# --- Extraction: .fst -> .krml -> pre/*.{cu,h} -> cu/*.{cu,h} ---
# The .krml rules take their prerequisites from .depend, which forces them to
# live in F*'s --odir ($(OBJ)).

# Turning something like .kuipy_cache/checked/Kuiops_Mm.krml into Kuiops.Mm
$(OBJ)/%.krml: | mkobj
	@$(call msg,"EXTRACT")
	@$(CURDIR)/fstar.sh --already_cached '*,-Kuiops' --codegen krml \
	  --load_cmxs $(PLUGIN) --extract "-*,+Kuiops,+Kuiper" -o $@ $<

$(PRE)/%.cu $(PRE)/%.h &: MOD=$(call mod_of,$*)
$(PRE)/%.cu $(PRE)/%.h &: $(OBJ)/%.krml | mkobj
	@$(call msg,"KRML")
	@$(CURDIR)/krml.sh -bundle "$(MOD)=*" -tmpdir $(PRE)/ $<

# Postprocess via sed and generate the actual target.
# Do NOT use a wildcard without an extension or this can match object files.
$(CU)/%.cu: $(PRE)/%.cu $(FIXUP) | mkobj
	@$(call msg,"FIXUP")
	@sed -f $(FIXUP) $< | indent -linux -i4 -nut > $@
$(CU)/%.h: $(PRE)/%.h $(FIXUP) | mkobj
	@$(call msg,"FIXUP")
	@sed -f $(FIXUP) $< | indent -linux -i4 -nut > $@
