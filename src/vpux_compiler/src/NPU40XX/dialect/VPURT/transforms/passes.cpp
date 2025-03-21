//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"

namespace vpux::VPURT::arch40xx {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPURTPasses();
}

}  // namespace vpux::VPURT::arch40xx
