//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

namespace vpux {

namespace details {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace details

void registerConversionPasses() {
    details::registerConversionPasses();
}

}  // namespace vpux
