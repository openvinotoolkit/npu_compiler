//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Interfaces/TilingInterface.h>
#include <tuple>

using namespace vpux;

namespace {

using ExtractSliceParams =
        std::tuple<SmallVector<mlir::OpFoldResult>, SmallVector<mlir::OpFoldResult>, SmallVector<mlir::OpFoldResult>>;

mlir::FailureOr<ExtractSliceParams> computeExtractSliceParams(vpux::Shave::TilingRootOp op, mlir::OpBuilder& b,
                                                              mlir::ArrayRef<mlir::OpFoldResult> offsets) {
    if (offsets.empty()) {
        return mlir::failure();
    }

    if (op.getNumResults() != 1) {
        // E#219995 - Add support for more than one result for the root op
        return mlir::failure();
    }

    const auto resultType = mlir::cast<mlir::ShapedType>(op.getOperation()->getResult(0).getType());
    const auto rank = resultType.getRank();
    auto tilingRanks = parseIntArrayAttr<int64_t>(op.getTilingRanks());

    SmallVector<mlir::Type> outTys;
    outTys.assign(tilingRanks.size(), b.getIndexType());

    mlir::Value loopId = op.getLoopId();
    mlir::Value tileNum = mlir::getValueOrCreateConstantIndexOp(b, op.getLoc(), offsets[0]);
    auto sliceInfo = b.create<vpux::Shave::OutSliceInfoOp>(op.getLoc(), outTys, outTys, loopId, tileNum);

    SmallVector<mlir::OpFoldResult> resultOffsets(rank, b.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> resultSizes =
            mlir::tensor::getMixedSizes(b, op.getLoc(), op.getOperation()->getResult(0));

    for (size_t i = 0; i < tilingRanks.size(); ++i) {
        const auto dim = static_cast<size_t>(tilingRanks[i]);
        if (dim >= resultSizes.size()) {
            return mlir::failure();
        }
        resultSizes[dim] = sliceInfo.getSizes()[i];
        resultOffsets[dim] = sliceInfo.getOffsets()[i];
    }

    SmallVector<mlir::OpFoldResult> strides(rank, b.getIndexAttr(1));

    return ExtractSliceParams{std::move(resultOffsets), std::move(resultSizes), std::move(strides)};
}

}  // namespace

mlir::LogicalResult vpux::Shave::TilingRootOp::verify() {
    if (getNumResults() == 0) {
        return emitOpError("expects at least one result tensor");
    }

    const auto firstResultType = mlir::dyn_cast<mlir::ShapedType>(getResult(0).getType());
    if (!firstResultType) {
        return emitOpError("expects first result to be a shaped tensor type");
    }

    const auto firstResultRank = firstResultType.getRank();
    const auto tilingRanks = parseIntArrayAttr<int64_t>(getTilingRanks());
    llvm::SmallSet<int64_t, 4> seenRanks;
    for (const auto rank : tilingRanks) {
        if (rank < 0 || rank >= firstResultRank) {
            return emitOpError() << "has tilingRanks value " << rank << " outside first result rank "
                                 << firstResultRank;
        }

        if (!seenRanks.insert(rank).second) {
            return emitOpError() << "has duplicate tilingRanks value " << rank;
        }
    }

    const auto inputs = getInputs();
    const auto results = getResults();
    if (inputs.size() != results.size()) {
        return emitOpError() << "expects the same number of inputs and outputs, got " << inputs.size() << " inputs and "
                             << results.size() << " outputs";
    }

    for (auto [input, result] : llvm::zip(inputs, results)) {
        if (input.getType() != result.getType()) {
            return emitOpError() << "expects input and output types to match, got " << input.getType() << " and "
                                 << result.getType();
        }
    }

    return mlir::success();
}

mlir::SmallVector<mlir::utils::IteratorType> vpux::Shave::TilingRootOp::getLoopIteratorTypes() {
    return {mlir::utils::IteratorType::parallel};
}

mlir::SmallVector<mlir::Range> vpux::Shave::TilingRootOp::getIterationDomain(mlir::OpBuilder& b) {
    // Create a LoopTripCountOp tied to this op's loop id. The trip count
    // represents the upper bound of the iteration space for the tiling
    // transformation.
    auto tripCountOp = b.create<Shave::LoopTripCountOp>(getLoc(), mlir::TypeRange{b.getIndexType()},
                                                        getOperation()->getOperand(0));
    auto tripCountVal = tripCountOp.getOperation()->getResult(0);

    mlir::SmallVector<mlir::Range> loopBounds(1);
    loopBounds[0].offset = b.getIndexAttr(0);
    loopBounds[0].size = tripCountVal;
    loopBounds[0].stride = b.getIndexAttr(1);

    return loopBounds;
}

mlir::FailureOr<mlir::TilingResult> vpux::Shave::TilingRootOp::getTiledImplementation(
        mlir::OpBuilder& b, mlir::ArrayRef<mlir::OpFoldResult> offsets, mlir::ArrayRef<mlir::OpFoldResult>) {
    auto paramsOrFailure = computeExtractSliceParams(*this, b, offsets);
    if (mlir::failed(paramsOrFailure)) {
        return mlir::failure();
    }

    SmallVector<mlir::OpFoldResult> resultOffsets;
    SmallVector<mlir::OpFoldResult> resultSizes;
    SmallVector<mlir::OpFoldResult> strides;
    std::tie(resultOffsets, resultSizes, strides) = std::move(paramsOrFailure.value());

    auto extractSlice =
            b.create<mlir::tensor::ExtractSliceOp>(getLoc(), this->getInputs()[0], resultOffsets, resultSizes, strides);

    SmallVector<mlir::Operation*> tiledOps;
    SmallVector<mlir::Value> resultValues;
    SmallVector<mlir::Operation*> generatedSlices;

    tiledOps.push_back(extractSlice.getOperation());
    resultValues.push_back(extractSlice->getResult(0));
    generatedSlices.push_back(extractSlice.getOperation());

    return mlir::TilingResult{std::move(tiledOps), std::move(resultValues), std::move(generatedSlices)};
}

mlir::LogicalResult vpux::Shave::TilingRootOp::getResultTilePosition(
        mlir::OpBuilder& b, unsigned resultNumber, mlir::ArrayRef<mlir::OpFoldResult> offsets,
        mlir::ArrayRef<mlir::OpFoldResult>, mlir::SmallVector<mlir::OpFoldResult>& resultOffsets,
        mlir::SmallVector<mlir::OpFoldResult>& resultSizes) {
    if (resultNumber != 0) {
        return mlir::failure();
    }

    auto paramsOrFailure = computeExtractSliceParams(*this, b, offsets);
    if (mlir::failed(paramsOrFailure)) {
        return mlir::failure();
    }

    SmallVector<mlir::OpFoldResult> tmpOffsets;
    SmallVector<mlir::OpFoldResult> tmpSizes;
    SmallVector<mlir::OpFoldResult> tmpStrides;
    std::tie(tmpOffsets, tmpSizes, tmpStrides) = std::move(paramsOrFailure.value());

    VPUX_UNUSED(tmpStrides);

    resultOffsets = std::move(tmpOffsets);
    resultSizes = std::move(tmpSizes);

    return mlir::success();
}
