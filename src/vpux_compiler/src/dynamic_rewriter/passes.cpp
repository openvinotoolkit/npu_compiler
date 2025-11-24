//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dynamic_rewriter/passes.hpp"

namespace vpux {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dynamic_rewriter/passes.hpp.inc"
}  // namespace

void registerDynamicRewriterExecutorPass() {
    registerDynamicRewriterPasses();
}

}  // namespace vpux
