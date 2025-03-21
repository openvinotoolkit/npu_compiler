//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/passes.hpp"

namespace vpux::Const {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/const/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerConstPasses();
}

}  // namespace vpux::Const
