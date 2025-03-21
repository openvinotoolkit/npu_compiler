//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux::VPUIP::arch37xx {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU37XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUIPPasses();
}

}  // namespace vpux::VPUIP::arch37xx
