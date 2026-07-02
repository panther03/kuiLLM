// Compatibility shim for JIT-compiled Kuiper kernels.
// karamel (C++-compat mode) emits compound-literal struct returns as
// KRML_CLITERAL(T){ ... } but this karamel build ships no definition for it.
#pragma once
#ifndef KRML_CLITERAL
#define KRML_CLITERAL(...) __VA_ARGS__
#endif

// `-drop Prims` omits the krml integer typedefs, but karamel may still emit a
// (dead) monomorphised tuple type over `krml_checked_int_t` when an intermediate
// `int` tuple in a spec isn't fully evaluated away (same class of leak as the
// KRML_CLITERAL shim above). Provide the typedef so such dead declarations still
// compile; the value never materialises in device code.
#ifndef KRML_CHECKED_INT_T_DEFINED
#define KRML_CHECKED_INT_T_DEFINED
#include <inttypes.h>
typedef int32_t krml_checked_int_t;
#endif
