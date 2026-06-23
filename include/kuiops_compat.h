// Compatibility shim for JIT-compiled Kuiper kernels.
// karamel (C++-compat mode) emits compound-literal struct returns as
// KRML_CLITERAL(T){ ... } but this karamel build ships no definition for it.
#pragma once
#ifndef KRML_CLITERAL
#define KRML_CLITERAL(...) __VA_ARGS__
#endif
