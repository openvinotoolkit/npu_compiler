//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/dynamic_shape_propagation.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DynamicTileOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicTileOpAdaptor tile(operands, attrs, prop);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(tile.getInput().getType());
    mlir::RankedTensorType outType;

    const auto outShape = parseIntArrayAttr<int64_t>(tile.getOutputShapeAttr());
    const auto outBounds = parseIntArrayAttr<int64_t>(tile.getOutputBoundsAttr());

    auto typeComponents = TypeComponents().setDimsOrder(DimsOrder::fromNumDims(outShape.size()));

    const auto isDynamicDim = [](int64_t dim) {
        return dim == mlir::ShapedType::kDynamic;
    };

    // DynamicTile might have static outShape
    if (none_of(outShape, isDynamicDim)) {
        const auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), inType.getMemSpace());
        outType = mlir::RankedTensorType::get(outShape, inType.getElementType(), outDesc);
    } else {
        assignDynamicTypeComponents(typeComponents, tile.getBoundsRepresentation(), outShape, outBounds);
        outType = mlir::cast<mlir::RankedTensorType>(inType.changeTypeComponents(typeComponents));
    }
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// fit into CMX
//

bool vpux::VPU::DynamicTileOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "DynamicTileOp requires 2 inputs and 1 output, but the number of buffers is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::DynamicTileOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DynamicTileOp::supportCycleCostCalculation() {
    return false;
}
