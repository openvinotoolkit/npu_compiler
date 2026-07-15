//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Empty translation unit that exists solely to give the npu_compiler_pch_base
// anchor target something to compile.  The PCH is built once for this target
// and then reused by all consumer libraries via REUSE_FROM.
//
// No manual #include of precomp.hpp is needed here.  When ENABLE_FASTER_BUILD
// is ON, CMake's target_precompile_headers() force-injects cmake_pch.hxx into
// every TU of npu_compiler_pch_base (including this one) via the compiler
// -include flag.  That force-injection is what triggers compilation of the
// .gch/.pchi artifact that consumers then reuse.
namespace vpux {
void npu_compiler_pch_base_anchor() {
}
}  // namespace vpux
