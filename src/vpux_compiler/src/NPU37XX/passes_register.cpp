//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/passes_register.hpp"
#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"

using namespace vpux;

//
// PassesRegistry37XX::registerPasses
//

void PassesRegistry37XX::registerPasses() {
    vpux::arch37xx::registerConversionPasses();
    vpux::IE::arch37xx::registerPasses();
    vpux::VPU::arch37xx::registerPasses();
    vpux::VPUIP::arch37xx::registerPasses();
    vpux::VPURT::arch37xx::registerPasses();
}
