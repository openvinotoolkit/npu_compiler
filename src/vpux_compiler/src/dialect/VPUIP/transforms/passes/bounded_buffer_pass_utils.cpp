//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/bounded_buffer_pass_utils.hpp"

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <intel_npu/prefix.hpp>

#include <mlir/IR/Builders.h>

#include <iterator>

using namespace vpux;

vpux::VPUIP::BoundedBufferComponents vpux::VPUIP::unpackBoundedBufferType(VPUIP::BoundedBufferType type) {
    // TODO: support for other buffer types will be added separately
    // Track E#118173
    return {mlir::cast<mlir::MemRefType>(type.getData()), mlir::cast<mlir::MemRefType>(type.getDynamicShape())};
}

vpux::VPUIP::BoundedBufferComponents vpux::VPUIP::unpackDynamicMemRefType(mlir::MemRefType type, BoundsRef bounds) {
    if (bounds.empty()) {
        bounds = getBounds(type);
    }
    const auto ndType = mlir::cast<NDTypeInterface>(type);
    const auto strides = ndType.getStrides();
    // Note: shape change removes strides, but it makes sense to preserve them
    const auto dataMemref = ndType.changeShape(ShapeRef(bounds.raw())).changeStrides(strides);
    const auto shapeMemref = getMemRefType({static_cast<int32_t>(bounds.size())}, getSInt32Type(type.getContext()),
                                           DimsOrder::C, ndType.getMemSpace());

    return {mlir::cast<mlir::MemRefType>(dataMemref), shapeMemref};
}

void vpux::VPUIP::addShapeTensorDataInfo(mlir::func::FuncOp funcOp, mlir::MemRefType dynamicShapeMemRef,
                                         mlir::Block& infoBlock, mlir::StringRef dataInfoName, size_t dataBufferCount) {
    const auto type = mlir::RankedTensorType::get(dynamicShapeMemRef.getShape(), dynamicShapeMemRef.getElementType());
    const auto name = std::string(intel_npu::SHAPE_TENSOR_PREFIX) + dataInfoName.str();

    auto insertionPointAfter = std::next(infoBlock.begin(), static_cast<int64_t>(dataBufferCount));
    auto infoBuilder = mlir::OpBuilder(&infoBlock, insertionPointAfter);
    net::DataInfoOp::create(infoBuilder, takeOpLoc(funcOp, "{0}", funcOp.getName()), name, type);
}
