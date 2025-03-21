//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/transforms/passes.hpp"

namespace vpux::Core {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerCorePasses();
}

}  // namespace vpux::Core
