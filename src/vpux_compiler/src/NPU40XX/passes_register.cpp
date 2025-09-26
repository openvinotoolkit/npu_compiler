//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/passes_register.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

namespace vpux::VPUIP::arch37xx {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU37XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch37xx
namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

//
// PassesRegistry40XX::registerPasses
//

void PassesRegistry40XX::registerPasses() {
    vpux::VPUIP::arch37xx::registerAddSwKernelCacheHandlingOpsPass();
    vpux::VPUIP::arch40xx::registerPasses();

    vpux::VPURegMapped::registerPasses();
}
