//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"

namespace vpux::arch40xx {

namespace details {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU37XX/conversion/passes.hpp.inc"
}  // namespace details

void registerConversionPasses() {
    details::registerConversionPasses();
}

}  // namespace vpux::arch40xx
