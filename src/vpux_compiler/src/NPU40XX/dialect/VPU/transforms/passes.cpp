//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"

namespace vpux::VPU::arch40xx {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU40XX/dialect/VPU/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUPasses();
}

}  // namespace vpux::VPU::arch40xx
