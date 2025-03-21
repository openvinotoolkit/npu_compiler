//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"

namespace vpux::ELFNPU37XX {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerELFNPU37XXPasses();
}

}  // namespace vpux::ELFNPU37XX
