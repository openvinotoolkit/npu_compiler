//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/memref_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {

mlir::IntegerAttr getAllocSizeAttr(mlir::Type type) {
    mlir::IntegerAttr allocSizeAttr;

    if (type == nullptr) {
        return allocSizeAttr;
    }

    mlir::MemRefLayoutAttrInterface layout;

    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
        layout = memref.getLayout();
    } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
        layout = distributedBuffer.getLayout();
    } else if (auto itiBuffer = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type)) {
        layout = itiBuffer.getLayout();
    }

    if (layout) {
        if (const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
            allocSizeAttr = memRefAttr.allocSize();
        }
    }

    return allocSizeAttr;
}

// Updates the swizzling scheme, adjusts the sizeAlignment added for distributedBuffer
vpux::NDTypeInterface setAllocSizeAttr(vpux::NDTypeInterface type, int64_t allocSize) {
    auto* ctx = type.getContext();
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto strides = type.getStrides();
    const auto memSpace = type.getMemSpace();

    mlir::IntegerAttr allocSizeAttr = getIntAttr(ctx, allocSize);

    if (mlir::isa<mlir::MemRefType>(type)) {
        return vpux::getMemRefType(shape, elemType, order, memSpace, strides, getBounds(type),
                                   VPUIP::getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                                   allocSizeAttr);
    } else if (mlir::isa<vpux::VPUIP::DistributedBufferType>(type) || mlir::isa<vpux::VPUIP::ITIBufferType>(type)) {
        vpux::MemRefAttr memRefAttr;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        mlir::ArrayAttr stridesAttr = nullptr;
        vpux::MemRefAttr::HwFields hwSpecificFields{};

        auto itiBufferType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type);
        auto distBufferType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type);

        if (itiBufferType) {
            memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(itiBufferType.getLayout());
        } else if (distBufferType) {
            memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(distBufferType.getLayout());
        }
        if (memRefAttr) {
            stridesAttr = memRefAttr.strides();
            hwSpecificFields = memRefAttr.hwSpecificFields();
        }

        const auto layoutAttr = vpux::MemRefAttr::get(orderAttr, stridesAttr, allocSizeAttr, /*optionalBounds=*/{},
                                                      hwSpecificFields, ctx);

        if (itiBufferType) {
            return VPUIP::ITIBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                             itiBufferType.getIduSegmentation(), itiBufferType.getInwardHaloRegions(),
                                             itiBufferType.getOutwardHaloRegions());
        }

        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                                 distBufferType.getDistribution(),
                                                 distBufferType.getSparsityCompression());
    }

    VPUX_THROW("Unsupported type for storing allocSize setting - {0}", type);
}

}  // namespace vpux::VPUIP
