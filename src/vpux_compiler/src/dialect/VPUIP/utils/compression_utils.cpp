//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/compression_utils.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/memref_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace VPUIP {

bool isSupportedBufferSizeForCompression(vpux::NDTypeInterface ndType) {
    // Compression HW supports buffers > 256 bytes
    return ndType.getTotalAllocSize().count() > ACT_COMPRESSION_MIN_BUF_SIZE;
}

VPUIP::CompressionStateAttr getCompressionStateAttr(mlir::Type type) {
    VPUIP::CompressionStateAttr compressionAttr;

    if (type == nullptr) {
        return compressionAttr;
    }

    mlir::MemRefLayoutAttrInterface layout;

    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
        layout = memref.getLayout();
    } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
        layout = distributedBuffer.getLayout();
    } else if (auto itiBuffer = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type)) {
        layout = itiBuffer.getLayout();
    } else {
        return compressionAttr;
    }

    if (layout) {
        if (const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
            compressionAttr = memRefAttr.hwSpecificField<vpux::VPUIP::CompressionStateAttr>();
        }
    }

    return compressionAttr;
}

VPUIP::CompressionState getCompressionState(mlir::Type type) {
    auto compressionAttr = getCompressionStateAttr(type);

    if (compressionAttr == nullptr) {
        return VPUIP::CompressionState::NoCompression;
    }

    return compressionAttr.getValue();
}

mlir::Type setCompressionStateAttribute(mlir::Type type, VPUIP::CompressionStateAttr compressionAttr) {
    VPUX_THROW_WHEN(type == nullptr, "NULL type provided");

    if (!compressionAttr) {
        return type;
    }

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    auto* ctx = type.getContext();

    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();
    const auto order = ndType.getDimsOrder();
    const auto strides = ndType.getStrides();
    const auto memSpace = ndType.getMemSpace();

    if (mlir::isa<mlir::MemRefType>(type)) {
        return vpux::getMemRefType(shape, elemType, order, memSpace, strides, getBounds(type),
                                   getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                                   VPUIP::getAllocSizeAttr(type), compressionAttr);
    } else if (mlir::isa<vpux::VPUIP::DistributedBufferType>(type) || mlir::isa<vpux::VPUIP::ITIBufferType>(type)) {
        mlir::ArrayAttr stridesAttr;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const Bit elemSize = ndType.getElemTypeSize();
        const auto memShape = order.toMemoryOrder(shape);
        const auto memStrides = order.toMemoryOrder(strides);
        const auto compactReqs = StrideReqs::compact(shape.size());
        if (!compactReqs.checkStrides(memStrides, elemSize, memShape)) {
            // Have strides only if they are not compact
            const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                         return stride.count() / elemSize.count();
                                                     }));

            stridesAttr = getIntArrayAttr(ctx, elemStrides);
        }

        const auto layoutAttr = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                                      /*allocSize=*/nullptr, /*optionalBounds=*/{},
                                                      {getSwizzlingSchemeAttr(type), compressionAttr}, ctx);

        if (auto itiBufferType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type)) {
            return VPUIP::ITIBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                             itiBufferType.getIduSegmentation(), itiBufferType.getInwardHaloRegions(),
                                             itiBufferType.getOutwardHaloRegions());
        }

        auto distBufferType = mlir::cast<vpux::VPUIP::DistributedBufferType>(type);
        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                                 distBufferType.getDistribution(),
                                                 distBufferType.getSparsityCompression());
    }

    VPUX_THROW("Unsupported type for storing compression setting");
}

mlir::Type setCompressionState(mlir::Type type, VPUIP::CompressionState compression) {
    VPUX_THROW_WHEN(type == nullptr, "NULL type provided");

    auto compressionAttr = VPUIP::CompressionStateAttr::get(type.getContext(), compression);

    return setCompressionStateAttribute(type, compressionAttr);
}

}  // namespace VPUIP
}  // namespace vpux
