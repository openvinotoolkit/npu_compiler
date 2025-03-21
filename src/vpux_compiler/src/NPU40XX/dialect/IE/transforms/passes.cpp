//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"

namespace vpux::IE::arch40xx {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU40XX/dialect/IE/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerIEPasses();
}

}  // namespace vpux::IE::arch40xx
