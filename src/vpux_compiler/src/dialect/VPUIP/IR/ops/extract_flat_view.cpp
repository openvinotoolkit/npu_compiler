//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::ExtractFlatSliceOp::getViewSource() {
    return getSource();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::ExtractFlatSliceOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                                std::optional<mlir::Location> optLoc,
                                                                mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                                mlir::OpaqueProperties props,
                                                                mlir::RegionRange /*regions*/,
                                                                mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::ExtractFlatSliceOpAdaptor extractOp(operands, attrs, props);
    if (mlir::failed(extractOp.verify(loc))) {
        return mlir::failure();
    }

    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(extractOp.getSource().getType());
    if (distributedType == nullptr) {
        return errorAt(loc, "ExtractFlatSliceOp must operate on Distributed buffers");
    }
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(distributedType.getElementType())) {
        return errorAt(loc, "ExtractFlatSliceOp does not support UniformQuantizedPerAxisType");
    }

    auto inputShape = distributedType.getShape();
    auto stridesReq = StrideReqs::compact(inputShape.size());
    if (!stridesReq.checkStrides(distributedType)) {
        return errorAt(loc, "ExtractFlatSliceOp must operate on compact buffers");
    }

    const auto distribution = distributedType.getDistribution();
    const auto distMode = distribution.getMode().getValue();
    if (distMode != VPU::DistributionMode::SEGMENTED) {
        return errorAt(loc, "ExtractFlatSliceOp must operate on SEGMENTED Distributed buffers");
    }

    auto maybeTileIndex = VPUIP::getTilingDimIndex(distributedType);
    if (!maybeTileIndex.has_value()) {
        return errorAt(loc, "Cannot infer tiling dim for ExtractFlatSliceOp");
    }
    auto tileIndex = maybeTileIndex.value();

    for (int64_t i = 0; i < tileIndex; ++i) {
        // Compiler expects to feed this operation to GenericReshape, which can operate on compact buffer. If
        // tiling dim if not leading we'll get strided memref, which is not supported yet
        if (inputShape[Dim(i)] != 1) {
            return errorAt(loc, "Tiling dim must be first dimension greater than 1");
        }
    }
    auto tileDim = Dim(tileIndex);

    auto tilingDimOffset = extractOp.getOffset();
    auto shape = distributedType.getShape();
    if (tilingDimOffset < 0 || tilingDimOffset >= shape[tileDim]) {
        return errorAt(loc, "Offset is exceeding original shape");
    }
    auto perClusterOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    size_t clusterId = perClusterOffsets.size() - 1;
    for (size_t idx = 0; idx < perClusterOffsets.size() - 1; ++idx) {
        auto nextClusterOffset = perClusterOffsets[idx + 1][tileDim];
        if (tilingDimOffset < nextClusterOffset) {
            clusterId = idx;
            break;
        }
    }
    auto tilingDimLength = extractOp.getLength();
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    if (tilingDimLength + tilingDimOffset >
        perClusterOffsets[clusterId][tileDim] + perClusterShapes[clusterId][tileDim]) {
        return errorAt(loc, "Destination is not inside single cluster");
    }
    auto newShape = shape.toValues();
    newShape[tileDim] = tilingDimLength;

    const auto memSpaceCMX =
            vpux::IndexedSymbolAttr::get(loc.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), clusterId);

    auto newType = vpux::getMemRefType(newShape, distributedType.getElementType(), distributedType.getDimsOrder(),
                                       memSpaceCMX);

    inferredTypes.push_back(newType);

    return mlir::success();
}

mlir::LogicalResult VPUIP::ExtractFlatSliceOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    mlir::SmallVector<mlir::Type> inferredTypes;
    if (inferReturnTypes(getContext(), getLoc(), op->getOperands(), op->getAttrDictionary(), op->getPropertiesStorage(),
                         op->getRegions(), inferredTypes)
                .failed()) {
        logCb(formatv("Can't infer return types"));
        return mlir::failure();
    }

    return mlir::success();
}
