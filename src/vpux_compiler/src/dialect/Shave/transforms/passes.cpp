//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"

namespace vpux::Shave {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/Shave/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerShavePasses();
}

}  // namespace vpux::Shave
