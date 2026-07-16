//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExpandOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExpandOpAdaptor expand(operands, attrs, prop);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(expand.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(expand.getPadsEnd());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(expand.getInput().getType());

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    inferredReturnTypes.push_back(newType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::ExpandOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    auto operands = adaptor.getOperands();
    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto padsBefore = Shape(parseIntArrayAttr<int64_t>(getPadsBegin()));
        const auto padsAfter = Shape(parseIntArrayAttr<int64_t>(getPadsEnd()));
        return static_cast<Const::ContentAttr>(attr).transform().padWithZero(padsBefore, padsAfter).get();
    }

    return nullptr;
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::ExpandOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                           mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto outputType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    const auto padsBegin = parseIntArrayAttr<int64_t>(getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(getPadsEnd());
    if (static_cast<size_t>(outputType.getRank()) != padsBegin.size() || padsBegin.size() != padsEnd.size()) {
        return mlir::failure();
    }

    SmallVector<mlir::OpFoldResult> shapes;
    shapes.reserve(outputType.getRank());
    const auto loc = getLoc();

    for (const auto dim : irange(outputType.getRank())) {
        if (!outputType.isDynamicDim(dim)) {
            shapes.push_back(builder.getIndexAttr(outputType.getDimSize(dim)));
            continue;
        }

        auto inputDim = builder.createOrFold<mlir::tensor::DimOp>(loc, getInput(), dim);
        const auto pad = padsBegin[dim] + padsEnd[dim];
        if (pad == 0) {
            shapes.push_back(inputDim);
            continue;
        }

        auto inputDimValue = getValueOrCreateConstantIndexOp(builder, loc, inputDim);
        auto padValue = builder.create<mlir::arith::ConstantIndexOp>(loc, pad);
        shapes.push_back(builder.create<mlir::arith::AddIOp>(loc, inputDimValue, padValue).getResult());
    }

    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}

//
// Tiling support
//
// `VPU.Expand` supports SCF Vertical Fusion only for single-dim end-padding
// (`pads_begin == 0`, exactly one `pads_end[d] > 0`) and only when tiling
// dimensions do not include the padded dimension.
// The padded dim must stay untiled, enforced by `isVFSupported` (VF candidate
// gate), `isSupportedTilingDim` (rejects axis d), and `isSupportedOutTile`
// (per-tile check). `backInferTileInfo` throws on unsupported configs; the SCF
// VF driver catches it in `calculateTilingRegions`.
//

bool vpux::VPU::ExpandOp::isSupportedTilingDim(DimArrRef tilingDims) {
    const auto paddedDim = VPU::getSinglePaddedExpandDim(getOperation());
    if (!paddedDim.has_value()) {
        return false;
    }
    // Non-padded dims map 1:1; the padded dim must not be split.
    return !llvm::is_contained(tilingDims, *paddedDim);
}

// Mirror of `SCFExpandTilingModelOp::backInferSCFTileInfo` (OpFoldResult-valued);
// keep in sync.
vpux::InputTiling vpux::VPU::ExpandOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto paddedDim = VPU::getSinglePaddedExpandDim(getOperation());
    VPUX_THROW_UNLESS(paddedDim.has_value(), "VPU.Expand at '{0}' supports tiling only for single-dim end-padding",
                      getLoc());

    const auto inputShape = getShape(getInput());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile Expand operation at '{0}', operand rank mismatch", getLoc());
    const auto pIdx = static_cast<size_t>(paddedDim->ind());
    VPUX_THROW_UNLESS(pIdx < outputTile.shape.size(),
                      "Padded dim {0} is out of range for VPU.Expand '{1}' outputTile rank {2}", paddedDim->ind(),
                      getLoc(), outputTile.shape.size());

    // Non-padded dims pass through 1:1; the padded dim is materialized in full
    // on the input regardless of the output tile.
    auto inputTile = outputTile;
    inputTile.shape[*paddedDim] = inputShape[*paddedDim];
    inputTile.offsets[*paddedDim] = 0;

    return TilingInfo{{std::move(inputTile)}};
}

void vpux::VPU::ExpandOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/,
                                      ShapeRef /*outputShape*/) {
    // `pads_begin`/`pads_end` are unchanged: the padded dim is materialized in full per tile.
}

bool vpux::VPU::ExpandOp::isSupportedOutTile(const vpux::TileInfo& outputTile) {
    const auto paddedDim = VPU::getSinglePaddedExpandDim(getOperation());
    if (!paddedDim.has_value()) {
        return false;
    }
    if (static_cast<size_t>(paddedDim->ind()) >= outputTile.shape.size()) {
        return false;
    }
    // Padded dim must remain at full extent with zero offset.
    const auto outputShape = getShape(getOutput());
    return outputTile.shape[*paddedDim] == outputShape[*paddedDim] && outputTile.offsets[*paddedDim] == 0;
}

bool vpux::VPU::ExpandOp::isVFSupported() {
    // Only single-dim end-padded Expand can be fused as an SCF VF producer.
    return VPU::getSinglePaddedExpandDim(getOperation()).has_value();
}
