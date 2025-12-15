//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp"

namespace vpux::NPUReg50XX {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerNPUReg50XXPasses();
}

}  // namespace vpux::NPUReg50XX
