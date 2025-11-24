//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/rewriters.hpp"
#include "llvm/Support/Debug.h"

namespace vpux::VPUIP {

void registerVPUIPRewriters(RewriterRegistry& registry) {
    registerOptimizeCopiesSection(registry);
}

}  // namespace vpux::VPUIP
