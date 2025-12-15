//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/wlm_register_config.hpp"

namespace vpux::VPU::arch40xx {

struct RegisterConfig final {
    static llvm::SmallVector<uint32_t> getSHVRegisterAddrs();
    static llvm::SmallVector<uint32_t> getDPURegisterAddrs();
    static uint32_t getNCEBarrierFifoAddr();
    static uint32_t getNCEBarrierFifoDepth();
};

}  // namespace vpux::VPU::arch40xx
