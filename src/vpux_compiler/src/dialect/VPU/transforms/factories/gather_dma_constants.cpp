//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

size_t VPU::getGatherDMAMaxIndicesListLength(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU40XX:
        return VPU::arch40xx::DMA_MAX_INDICES_LIST_LENGTH;
    case config::ArchKind::NPU50XX:
        return VPU::arch50xx::DMA_MAX_INDICES_LIST_LENGTH;
    case config::ArchKind::NPU60XX:
        return VPU::arch60xx::DMA_MAX_INDICES_LIST_LENGTH;
    default:
        VPUX_THROW("Unable to get GatherDMAMaxIndicesListLength for arch {0}", arch);
    }
};

size_t VPU::getGatherDMAMaxElementSize(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU40XX:
        return VPU::arch40xx::GATHER_DMA_MAX_ELEMENT_SIZE;
    case config::ArchKind::NPU50XX:
        return VPU::arch50xx::GATHER_DMA_MAX_ELEMENT_SIZE;
    case config::ArchKind::NPU60XX:
        return VPU::arch60xx::GATHER_DMA_MAX_ELEMENT_SIZE;
    default:
        VPUX_THROW("Unable to get GatherDMAMaxElementSize for arch {0}", arch);
    }
};
