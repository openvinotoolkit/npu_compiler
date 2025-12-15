//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/wlm_register_config.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux::VPU::arch40xx;

constexpr uint32_t FIFO_BARRIERS_NCE_FILL_BARRIER_FIFO_ADR = 0x2F010000U;
// Although the HW barrier FIFO depth is 32, we use only 4 entries.
// This allows configuration with a single register write and helps reduce preemption time.
constexpr uint32_t BARRIER_FIFO_DEPTH = 4;

llvm::SmallVector<uint32_t> RegisterConfig::getSHVRegisterAddrs() {
    // SHV FIFO true addresses (assuming 2 shaves per tile -> 12 entries total)
    return {0x2F00C000, 0x2F00C020, 0x2F00C040, 0x2F00C060, 0x2F00C080, 0x2F00C0A0,
            0x2F00C0C0, 0x2F00C0E0, 0x2F00C100, 0x2F00C120, 0x2F00C140, 0x2F00C160};
}

llvm::SmallVector<uint32_t> RegisterConfig::getDPURegisterAddrs() {
    // DPU FIFO true addresses (1 per tile)
    return {0x2F000000, 0x2F000020, 0x2F000040, 0x2F000060, 0x2F000080, 0x2F0000A0};
}

uint32_t RegisterConfig::getNCEBarrierFifoAddr() {
    return FIFO_BARRIERS_NCE_FILL_BARRIER_FIFO_ADR;
}

uint32_t RegisterConfig::getNCEBarrierFifoDepth() {
    return BARRIER_FIFO_DEPTH;
}
