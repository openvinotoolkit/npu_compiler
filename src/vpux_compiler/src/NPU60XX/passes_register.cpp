//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/passes_register.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/conversion.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

namespace vpux::IE::arch50xx {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU50XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch50xx

using namespace vpux;

//
// PassesRegistry60XX::registerPasses
//

void PassesRegistry60XX::registerPasses() {
    vpux::IE::arch60xx::registerPasses();
    vpux::IE::arch50xx::registerPasses();

    vpux::VPU::arch50xx::registerPasses();
    vpux::VPU::arch60xx::registerPasses();

    vpux::VPUIP::arch60xx::registerPasses();

    vpux::VPURegMapped::registerPasses();
}
