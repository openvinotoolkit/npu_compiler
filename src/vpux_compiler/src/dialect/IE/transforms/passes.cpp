//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::IE {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerIEPasses();
}

}  // namespace vpux::IE
