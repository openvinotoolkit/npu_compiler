//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

namespace vpux::VPURegMapped {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPURegMappedPasses();
}

}  // namespace vpux::VPURegMapped
