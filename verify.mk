OBJ := .kuipy_cache/checked
ROOTS := $(shell find kuiops/ -name '*.fst' -o -name '*.fsti')
CHECKED_FILES := $(foreach f, $(ROOTS), $(OBJ)/$(notdir $(f)).checked)

define msg =
@printf "  %-8s  %s\n" $(1) $(if $(2),$(2),$(shell realpath --relative-to=. $<))
endef

mkobj:
	mkdir -p $(OBJ)

.depend: $(ROOTS) | mkobj
	$(call msg,"DEPEND",$@)
	@$(CURDIR)/fstar.sh --already_cached '*,-Kuiops' --dep full $(ROOTS) -o $@

include .depend

$(OBJ)/%.checked:
	@$(call msg,"CHECK")
	@$(CURDIR)/fstar.sh --already_cached '*' -c $< -o $@
	@touch -c $@

verify-kuiops: .depend $(CHECKED_FILES)
