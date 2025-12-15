//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/passes_register.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

using namespace vpux;

//
// PassesRegistry50XX::registerPasses
//

void PassesRegistry50XX::registerPasses() {
    vpux::IE::arch50xx::registerPasses();

    vpux::VPU::arch50xx::registerPasses();

    vpux::VPUIP::arch50xx::registerPasses();

    vpux::VPURegMapped::registerPasses();
}
