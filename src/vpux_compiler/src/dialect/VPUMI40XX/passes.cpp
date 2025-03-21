//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPUMI40XX {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUMI40XXPasses();
}

}  // namespace vpux::VPUMI40XX
