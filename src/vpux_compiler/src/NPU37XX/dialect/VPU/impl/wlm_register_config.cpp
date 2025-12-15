//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/wlm_register_config.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux::VPU::arch37xx;

llvm::SmallVector<uint32_t> RegisterConfig::getSHVRegisterAddrs() {
    return {};
}

llvm::SmallVector<uint32_t> RegisterConfig::getDPURegisterAddrs() {
    return {};
}

uint32_t RegisterConfig::getNCEBarrierFifoAddr() {
    return 0;
}

uint32_t RegisterConfig::getNCEBarrierFifoDepth() {
    return 0;
}
