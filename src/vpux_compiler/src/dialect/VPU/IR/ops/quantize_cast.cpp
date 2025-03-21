//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::QuantizeCastOp::verify() {
    const auto dstElemType = getDstElemType();
    const auto inputElemType = mlir::cast<NDTypeInterface>(getInput().getType()).getElementType();

    if (mlir::failed(isQuantizeCastValid(getLoc(), inputElemType, dstElemType))) {
        return errorAt(getLoc(), "Unsupported quantize cast: '{0}'->'{1}'", inputElemType, dstElemType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::QuantizeCastOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::QuantizeCastOpAdaptor quantizeCast(operands, attrs, prop);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<NDTypeInterface>(quantizeCast.getInput().getType());
    const auto inElemType = inType.getElementType();
    const auto dstElemType = quantizeCast.getDstElemType();

    if (mlir::failed(isQuantizeCastValid(loc, inElemType, dstElemType))) {
        return errorAt(loc, "Unsupported quantize cast: '{0}'->'{1}'", inElemType, dstElemType);
    }

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::QuantizeCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    } else if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput())) {
        auto elemType = getDstElemTypeAttr().getValue();
        return attr.transform().castElemType(elemType).get();
    }

    return nullptr;
}

//
// TilingViewLikeOpInterface
//

vpux::InputTiling vpux::VPU::QuantizeCastOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    SmallVector<TileInfo> inputTiles;
    const auto inputShape = getShape(getInput());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile QuantizeCast operation at '{0}', which has operands with different rank",
                      this->getLoc());
    inputTiles.push_back(outputTile);
    return TilingInfo{inputTiles};
}

void vpux::VPU::QuantizeCastOp::adjustAttrs(const TilingInfo&, const TileInfo&, ShapeRef) {
    // Do nothing
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::QuantizeCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }
    const auto typeComponents = TypeComponents().setMemSpace(inType.getMemSpace());
    auto returnType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType()).changeTypeComponents(typeComponents);
    return std::make_pair(mlir::cast<mlir::Type>(returnType), distribution);
}
