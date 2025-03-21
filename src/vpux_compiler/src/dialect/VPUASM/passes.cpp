//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/passes.hpp"

namespace vpux::VPUASM {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPUASM/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUASMPasses();
}

}  // namespace vpux::VPUASM
