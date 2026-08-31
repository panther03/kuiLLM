ifeq ($(origin KUIPER_HOME),undefined)
KUIPER_HOME := $(CURDIR)/.kuiper
KUIPER_HOME_MANAGED := 1
else
KUIPER_HOME := $(abspath $(KUIPER_HOME))
KUIPER_HOME_MANAGED := 0
endif

KUIPER_NIGHTLY ?= 2026-08-30
ifeq ($(KUIPER_HOME_MANAGED),1)
KUIPER_MARKER := $(KUIPER_HOME)/.kuillm-nightly-$(KUIPER_NIGHTLY)
else
KUIPER_MARKER := $(KUIPER_HOME)/.packaged
endif

FSTAR_EXE := $(KUIPER_HOME)/inst/bin/fstar.exe
KRML_EXE := $(KUIPER_HOME)/inst/bin/krml
KUIPER_PLUGIN_SOURCE := $(KUIPER_HOME)/extraction/dune/_build/default/kuiper_extr.cmxs
KUIPER_PLUGIN := $(CURDIR)/build/kuiper_extr
FIXUP_SED := $(KUIPER_HOME)/scripts/fixup.sed
KUIPER_INCLUDE := $(KUIPER_HOME)/include

TOOLS_DIR := $(CURDIR)/.tools
CLANG_FORMAT_VERSION := 19.1.7
CLANG_FORMAT := $(TOOLS_DIR)/clang-format-$(CLANG_FORMAT_VERSION)/bin/clang-format
CLANG_FORMAT_FLAGS := --Werror --fail-on-incomplete-format \
	--style=file:$(CURDIR)/.clang-format

export KUIPER_HOME
export KUIPER_NIGHTLY
