//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/AffineMap.h>

using namespace vpux;

namespace {

class IESliceToExtractSlice : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    using mlir::OpRewritePattern<IE::SliceOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::SliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult IESliceToExtractSlice::matchAndRewrite(IE::SliceOp op, mlir::PatternRewriter& rewriter) const {
    auto inputNdType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
    auto inputRank = inputNdType.getRank();
    auto outputNdType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());

    auto resultDimsOrder = DimsOrder::fromPermutation(outputNdType.getDimsOrder().toPermutation());
    auto flatResultShape = resultDimsOrder.toMemoryOrder(outputNdType.getShape()).raw();
    auto elTy = inputNdType.getElementType();
    auto flatResultTy = mlir::RankedTensorType::get(flatResultShape, elTy);
    auto inputMemMap = inputNdType.getDimsOrder().toAffineMap(op.getContext());

    // Convert the input to identity layout.
    auto input = ShaveCodeGen::convertToLinalgValue(op->getOperand(0), rewriter);

    // Apply input permutation to slice offsets and sizes
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(op.getStaticOffsets());
    const auto sliceSizes = parseIntArrayAttr<int64_t>(op.getStaticSizes());
    auto memSliceOffsets = mlir::applyPermutationMap<int64_t>(inputMemMap, sliceOffsets);
    auto memSliceSizes = mlir::applyPermutationMap<int64_t>(inputMemMap, sliceSizes);
    SmallVector<int64_t> sliceStrides(inputRank, 1);

    // Build the extract slice op
    auto extractSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
            op.getLoc(), mlir::TypeRange{flatResultTy}, input, mlir::ValueRange{}, mlir::ValueRange{},
            mlir::ValueRange{}, memSliceOffsets, memSliceSizes, sliceStrides);

    rewriter.replaceOp(
            op,
            ShaveCodeGen::convertFromLinalgValue(extractSlice, op->getResult(0).getType(), rewriter).getDefiningOp());

    return mlir::success();
}

}  // namespace

void ShaveCodeGen::populateIEDataMovementToTensorPatterns(mlir::RewritePatternSet& patternSet) {
    auto& ctx = *patternSet.getContext();
    patternSet.add<IESliceToExtractSlice>(&ctx);
}
