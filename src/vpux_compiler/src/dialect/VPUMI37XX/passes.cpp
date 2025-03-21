//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"

namespace vpux::VPUMI37XX {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUMI37XXPasses();
}

}  // namespace vpux::VPUMI37XX
