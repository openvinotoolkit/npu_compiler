//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

namespace vpux::VPU {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUPasses();
}

}  // namespace vpux::VPU
