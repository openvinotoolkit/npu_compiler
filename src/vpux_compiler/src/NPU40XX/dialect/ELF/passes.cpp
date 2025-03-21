//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"

namespace vpux::ELF {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerELFPasses();
}

}  // namespace vpux::ELF
