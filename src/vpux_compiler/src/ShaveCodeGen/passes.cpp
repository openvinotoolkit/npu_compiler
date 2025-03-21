//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

namespace vpux::ShaveCodeGen {

namespace details {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace details

void registerPasses() {
    details::registerShaveCodeGenPasses();
}

}  // namespace vpux::ShaveCodeGen
