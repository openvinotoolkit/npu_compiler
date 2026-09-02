//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {

VPUIP::SwizzlingSchemeAttr createSwizzlingSchemeAttr(mlir::MLIRContext* ctx, config::ArchKind archKind,
                                                     int64_t swizzlingKey) {
    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr;
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return swizzlingSchemeAttr;
    }

    int64_t swizzlingSizeAlignment = getSizeAlignmentForSwizzling(archKind);
    auto swizzlingKeyAttr = getIntAttr(ctx, swizzlingKey);
    auto swizzlingSizeAlignmentAttr = getIntAttr(ctx, swizzlingSizeAlignment);

    swizzlingSchemeAttr = VPUIP::SwizzlingSchemeAttr::get(ctx, swizzlingKeyAttr, swizzlingSizeAlignmentAttr);
    return swizzlingSchemeAttr;
}

// Retrieve swizzling key setting embedded in layout with buffer types
VPUIP::SwizzlingSchemeAttr getSwizzlingSchemeAttr(mlir::Type type) {
    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr;

    if (type == nullptr) {
        return swizzlingSchemeAttr;
    }

    mlir::MemRefLayoutAttrInterface layout;

    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
        layout = memref.getLayout();
    } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
        layout = distributedBuffer.getLayout();
    } else if (auto itiBuffer = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type)) {
        layout = itiBuffer.getLayout();
    } else {
        return swizzlingSchemeAttr;
    }

    if (layout) {
        if (const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
            swizzlingSchemeAttr = memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        }
    }

    return swizzlingSchemeAttr;
}

int64_t getSwizzlingKey(mlir::Type type) {
    if (const auto swizzlingSchemeAttr = VPUIP::getSwizzlingSchemeAttr(type)) {
        return swizzlingSchemeAttr.getKey().getInt();
    }
    return 0;
}

mlir::Type setSwizzlingKey(mlir::Type type, mlir::IntegerAttr swizzlingKeyAttr, config::ArchKind archKind) {
    VPUX_THROW_WHEN(type == nullptr, "NULL type provided");

    if (!swizzlingKeyAttr) {
        return type;
    }

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    auto* ctx = type.getContext();

    auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(ctx, archKind, swizzlingKeyAttr.getInt());

    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();
    const auto order = ndType.getDimsOrder();
    const auto strides = ndType.getStrides();
    const auto memSpace = ndType.getMemSpace();

    if (mlir::isa<mlir::MemRefType>(type)) {
        return vpux::getMemRefType(shape, elemType, order, memSpace, strides, getBounds(ndType), swizzlingSchemeAttr,
                                   VPUIP::getSparsityCompressionAttr(type));
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

        const auto layoutAttr =
                vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                      /*allocSize=*/nullptr, /*optionalBounds=*/{}, {swizzlingSchemeAttr}, ctx);

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

    VPUX_THROW("Unsupported type for storing swizzling setting");
}

mlir::Type setSwizzlingKey(mlir::Type type, int64_t swizzlingKey, config::ArchKind archKind) {
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return type;
    }
    auto* ctx = type.getContext();
    auto swizzlingKeyAttr = getIntAttr(ctx, swizzlingKey);
    return setSwizzlingKey(type, swizzlingKeyAttr, archKind);
}

// Updates the swizzling scheme, adjusts the sizeAlignment added for distributedBuffer
vpux::NDTypeInterface updateSwizzlingSchemeBasedOnDistributedType(VPUIP::DistributedBufferType inputType,
                                                                  vpux::NDTypeInterface newType) {
    auto parentSwizzlingSchemeAttr = VPUIP::getSwizzlingSchemeAttr(inputType);
    auto swizzlingSchemeAttr = VPUIP::getSwizzlingSchemeAttr(newType);
    if (swizzlingSchemeAttr == nullptr || parentSwizzlingSchemeAttr == nullptr) {
        return newType;
    }

    const auto strides = newType.getStrides();
    return vpux::getMemRefType(newType.getShape(), newType.getElementType(), newType.getDimsOrder(),
                               newType.getMemSpace(), strides, getBounds(newType), parentSwizzlingSchemeAttr,
                               VPUIP::getSparsityCompressionAttr(newType));
}

}  // namespace vpux::VPUIP
