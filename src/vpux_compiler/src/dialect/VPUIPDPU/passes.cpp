//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"

namespace vpux::VPUIPDPU {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUIPDPUPasses();
}

}  // namespace vpux::VPUIPDPU
