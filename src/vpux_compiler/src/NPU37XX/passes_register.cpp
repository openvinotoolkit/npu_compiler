//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/passes_register.hpp"
#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

using namespace vpux;

//
// PassesRegistry37XX::registerPasses
//

void PassesRegistry37XX::registerPasses() {
    vpux::VPUIP::arch37xx::registerPasses();
}
