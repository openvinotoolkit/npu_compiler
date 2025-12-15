//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GroupConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupConvolutionOpAdaptor conv(operands, attrs, prop);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(conv.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(conv.getFilter().getType());
    auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    auto filterShapeInfo = ShapeInfo::fromNDType(filterType);

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.getDilations());

    const auto outShapeInfo = inferGroupConvolutionOutputShapeInfo(
            inShapeInfo, filterShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove, windowDilations,
            conv.getGroups(), conv.getOutputPadding().has_value());
    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outShapeInfo.bounds));

    const auto outputType = mlir::RankedTensorType::get(outShapeInfo.shape, inputType.getElementType(), outDesc);
    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

InputTiling vpux::VPU::GroupConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = getShape(getFilter());
    const auto origBiasShape = getBias() != nullptr ? getShape(getBias()) : ShapeRef();
    const auto origPadding = PadInfo(getPadsBegin(), getPadsEnd());
    const auto origGroups = getGroups().value_or(1);

    return backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, getStrides(), origPadding,
                                  origGroups);
}

//
// fitIntoCMX
//

bool vpux::VPU::GroupConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                               vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::GroupConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                               vpux::NDTypeInterface output, Byte reservedMem) {
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize()};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

void vpux::VPU::GroupConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    const auto& inputTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(inputTiles.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                      inputTiles.size());

    IE::adjustPaddings(this, inputTiling);

    const auto& inputTile = inputTiles[0];
    const auto& filterTile = inputTiles[1];
    const auto groups = inputTile.shape[Dims4D::Act::C] / filterTile.shape[Dims4D::Filter::IC];
    const auto groupsNewAttr = getIntAttr(getContext(), groups);

    setGroupsAttr(groupsNewAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::GroupConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
