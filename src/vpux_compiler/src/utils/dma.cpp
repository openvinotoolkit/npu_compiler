//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

int64_t vpux::getDMAPortValue(mlir::Operation* wrappedTaskOp) {
    if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(wrappedTaskOp)) {
        auto portAttr = dmaOp.getPortAttribute();
        if (portAttr == nullptr) {
            return 0;
        }
        return portAttr.getInt();
    }

    VPUX_THROW("Could not cast to DMA task '{0}'", *wrappedTaskOp);
}

SmallVector<VPUIP::DmaChannelType> vpux::getDMAChannelsWithIndependentLinkAgents(VPU::ArchKind arch) {
    if (arch <= VPU::ArchKind::NPU37XX) {
        return {VPUIP::DmaChannelType::NOT_SPECIFIED};
    }

    return {VPUIP::DmaChannelType::DDR, VPUIP::DmaChannelType::CMX};
}

// Encode DMA port and channel setting into a single integer for convenient usage during barrier scheduling
int64_t vpux::getDMAQueueIdEncoding(int64_t port, int64_t channelIdx) {
    return port * (VPUIP::getMaxEnumValForDmaChannelType() + 1) + channelIdx;
}
int64_t vpux::getDMAQueueIdEncoding(int64_t port, std::optional<vpux::VPUIP::DmaChannelType> channel) {
    return getDMAQueueIdEncoding(port, static_cast<int64_t>(channel.value_or(VPUIP::DmaChannelType::NOT_SPECIFIED)));
}
int64_t vpux::getDMAQueueIdEncoding(std::optional<vpux::VPUIP::DmaChannelType> channel) {
    return getDMAQueueIdEncoding(0, static_cast<int64_t>(channel.value_or(VPUIP::DmaChannelType::NOT_SPECIFIED)));
}

int64_t vpux::getDMAQueueIdEncoding(VPU::MemoryKind srcMemKind, VPU::ArchKind arch) {
    if (arch <= VPU::ArchKind::NPU37XX) {
        return getDMAQueueIdEncoding(std::nullopt);
    }

    if (srcMemKind == VPU::MemoryKind::DDR) {
        return getDMAQueueIdEncoding(VPUIP::DmaChannelType::DDR);
    }
    return getDMAQueueIdEncoding(VPUIP::DmaChannelType::CMX);
}

VPUIP::DmaChannelType vpux::getDMAQueueTypeFromEncodedId(int64_t dmaQueueIdEncoding, VPU::ArchKind arch) {
    if (arch <= VPU::ArchKind::NPU37XX) {
        return VPUIP::DmaChannelType::NOT_SPECIFIED;
    }

    return static_cast<VPUIP::DmaChannelType>(dmaQueueIdEncoding % (VPUIP::getMaxEnumValForDmaChannelType() + 1));
}

std::string vpux::getDMAChannelTypeAsString(VPUIP::DmaChannelType channelType, VPU::ArchKind arch) {
    if (arch <= VPU::ArchKind::NPU37XX) {
        return "";
    }

    return stringifyEnum(channelType).str();
}

std::string vpux::getDMAChannelTypeAsString(int64_t dmaQueueIdEncoding, VPU::ArchKind arch) {
    if (arch <= VPU::ArchKind::NPU37XX) {
        return "";
    }

    return stringifyEnum(getDMAQueueTypeFromEncodedId(dmaQueueIdEncoding, arch)).str();
}
