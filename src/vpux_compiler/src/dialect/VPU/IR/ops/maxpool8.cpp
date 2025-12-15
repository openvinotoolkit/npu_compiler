//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MaxPool8Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MaxPool8OpAdaptor maxPool8(operands, attrs, prop);
    if (mlir::failed(maxPool8.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool8.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool8.getPadsBegin());
    const auto windowDilations = parseIntArrayAttr<int64_t>(maxPool8.getDilations());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool8.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool8.getStrides());
    const auto roundingType = maxPool8.getRoundingType();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(maxPool8.getInput().getType());

    const auto shapeI64 = inferMaxPool8OutputShape(ShapeInfo::fromNDType(inType), windowStrides, windowDilations,
                                                   dataPaddingBelow, dataPaddingAbove, windowShape, roundingType);

    const auto outType = inType.changeShape(ShapeRef(shapeI64.shape));
    inferredReturnTypes.push_back(outType);

    const auto outType1 = outType.changeElemType(maxPool8.getIndexElementType());
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::MaxPool8Op::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto origInputShape = getShape(getInput());
    const auto pads = PadInfo(getPadsBegin(), getPadsEnd());
    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, getKernelSize(), getStrides(), pads);
    return inputTiling;
}

vpux::OutputTiling vpux::VPU::MaxPool8Op::getOutputTiling(const vpux::TileInfo& firstOutputTile, vpux::Logger /*log*/) {
    return OutputTiling{firstOutputTile, firstOutputTile};
}

void vpux::VPU::MaxPool8Op::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    IE::adjustPaddings(this, inputTiling);
    if (!inputTiling.tiles.size()) {
        return;
    }
    mlir::Builder builder(*this);
    TileInfo inputTile = inputTiling.tiles.begin()[0];

    const auto initialInputOffset = builder.getI64ArrayAttr(to_small_vector(inputTile.offsets));
    const auto initialOutputOffset = builder.getI64ArrayAttr(to_small_vector(outputTile.offsets));
    setInitialInputOffsetAttrAttr(initialInputOffset);
    setInitialOutputOffsetAttrAttr(initialOutputOffset);
}

mlir::FailureOr<OutputTiling> vpux::VPU::MaxPool8Op::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// SWOpInterface
//

bool vpux::VPU::MaxPool8Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "MaxPool8Op requires 1 inputs and 2 output, but the number of buffers is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    llvm::transform(buffers, std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::MaxPool8Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MaxPool8Op::supportCycleCostCalculation() {
    return false;
}
