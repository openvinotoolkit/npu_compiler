//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dynamic_shape_propagation.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <cstdint>
#include <optional>

using namespace vpux;

namespace {

struct StridedSliceInputData final {
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> strides;
};

StridedSliceInputData extractData(VPU::StridedSliceOpAdaptor stridedSlice) {
    auto begins = stridedSlice.getBeginsAttr().has_value()
                          ? parseIntArrayAttr<int64_t>(stridedSlice.getBeginsAttr().value())
                          : SmallVector<int64_t>{};
    auto ends = stridedSlice.getEndsAttr().has_value() ? parseIntArrayAttr<int64_t>(stridedSlice.getEndsAttr().value())
                                                       : SmallVector<int64_t>{};
    auto strides = stridedSlice.getStridesAttr().has_value()
                           ? parseIntArrayAttr<int64_t>(stridedSlice.getStridesAttr().value())
                           : SmallVector<int64_t>{};

    return StridedSliceInputData{std::move(begins), std::move(ends), std::move(strides)};
}

}  // namespace
// TODO: E-90249 Extend the infer type logic for StridedSlice to support different input / output ranks
mlir::LogicalResult vpux::VPU::StridedSliceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::StridedSliceOpAdaptor slice(operands, attrs, prop);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice);
    const auto beginsShape =
            slice.getBegins() != nullptr
                    ? SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(slice.getBegins().getType()).getShape())
                    : SmallVector<int64_t>{};
    const auto endsShape =
            slice.getEnds() != nullptr
                    ? SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(slice.getEnds().getType()).getShape())
                    : SmallVector<int64_t>{};
    const auto stridesShape =
            slice.getStrides() != nullptr
                    ? SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(slice.getStrides().getType()).getShape())
                    : SmallVector<int64_t>{};

    const auto beginMask = parseIntArrayAttr<int64_t>(slice.getBeginMask());
    const auto endMask = parseIntArrayAttr<int64_t>(slice.getEndMask());
    const auto newAxisMask = parseIntArrayAttr<int64_t>(slice.getNewAxisMask());
    const auto shrinkAxisMask = parseIntArrayAttr<int64_t>(slice.getShrinkAxisMask());
    const auto ellipsisMask = parseIntArrayAttr<int64_t>(slice.getEllipsisMask());

    const auto inDataType = mlir::cast<NDTypeInterface>(slice.getInput().getType());
    const auto inShapeInfo = ShapeInfo::fromNDType(inDataType);
    auto outputShapeInfo = inferStridedSliceOutputShape(inShapeInfo, inputData.begins, inputData.ends,
                                                        inputData.strides, beginsShape, endsShape, stridesShape,
                                                        beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto outShape = Shape(outputShapeInfo.shape);

    auto typeComponents = TypeComponents().setShape(outShape);

    if (auto outBounds = outputShapeInfo.bounds; !outBounds.empty()) {
        assignDynamicTypeComponents(typeComponents, slice.getBoundsRepresentation(), outShape.raw(), outBounds);
    }

    auto outType = inDataType.changeTypeComponents(typeComponents);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

bool vpux::VPU::StridedSliceOp::isSimplified() {
    if (getBegins() != nullptr || getEnds() != nullptr) {
        return false;
    }

    auto isZero = [](auto val) {
        return val == 0;
    };
    auto isPositive = [](auto val) {
        return val >= 0;
    };

    return (llvm::all_of(parseIntArrayAttr<int64_t>(getNewAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getShrinkAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEllipsisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginsAttr().value()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndsAttr().value()), isPositive));
}

InputTiling vpux::VPU::StridedSliceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto inShape = getShape(getInput());
    const auto begins = Shape(parseIntArrayAttr<int64_t>(getBeginsAttrAttr()));
    const auto strides = Shape(parseIntArrayAttr<int64_t>(getStridesAttrAttr()));
    const auto outOrder = DimsOrder::fromValue(getOutput());
    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        auto idx = outOrder.dimAt(ind);
        curTile.shape[idx] = outputTile.shape[idx] * strides[idx];
        curTile.offsets[idx] = outputTile.offsets[idx] * strides[idx] + begins[idx];
        curTile.axis[idx] = outputTile.axis[idx];
    }
    return TilingInfo{curTile};
}

void vpux::VPU::StridedSliceOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    // TODO: [E#115695]: the logic of tiling needs to be updated for
    // the case of non-const begins and ends
    VPUX_THROW_UNLESS(getBeginsAttr().has_value(), "begins_attr is null");
    VPUX_THROW_UNLESS(getEndsAttr().has_value(), "ends_attr is null");

    const auto inShape = getShape(getInput());
    auto ends = parseIntArrayAttr<int64_t>(getEndsAttr().value());
    auto begins = parseIntArrayAttr<int64_t>(getBeginsAttr().value());
    for (auto ind : irange(inShape.size())) {
        begins[ind] = 0;
        ends[ind] = inputTiling.tiles[0].shape[Dim(ind)];
    }
    const auto newEndsAttr = getIntArrayAttr(getContext(), ends);
    const auto newBeginsAttr = getIntArrayAttr(getContext(), begins);
    setEndsAttrAttr(newEndsAttr);
    setBeginsAttrAttr(newBeginsAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::StridedSliceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
