//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/wlm_register_config.hpp"

using namespace vpux::VPU;

llvm::SmallVector<uint32_t> RegisterConfig::getSHVRegisterAddrs() const {
    return self->getSHVRegisterAddrs();
}

llvm::SmallVector<uint32_t> RegisterConfig::getDPURegisterAddrs() const {
    return self->getDPURegisterAddrs();
}

uint32_t RegisterConfig::getNCEBarrierFifoAddr() const {
    return self->getNCEBarrierFifoAddr();
}
