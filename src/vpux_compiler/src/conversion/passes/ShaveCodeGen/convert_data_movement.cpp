//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

class IESliceToExtractSlice : public mlir::OpConversionPattern<IE::SliceOp> {
public:
    using mlir::OpConversionPattern<IE::SliceOp>::OpConversionPattern;
    using OpAdaptor = mlir::OpConversionPattern<IE::SliceOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(IE::SliceOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult IESliceToExtractSlice::matchAndRewrite(IE::SliceOp op, OpAdaptor adaptor,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    auto inputNdType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
    auto inputRank = inputNdType.getRank();
    auto inputMemMap = inputNdType.getDimsOrder().toAffineMap(op.getContext());

    auto input = adaptor.getOperands()[0];

    // Apply input permutation to slice offsets and sizes
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(op.getStaticOffsets());
    const auto sliceSizes = parseIntArrayAttr<int64_t>(op.getStaticSizes());
    auto memSliceOffsets = mlir::applyPermutationMap<int64_t>(inputMemMap, sliceOffsets);
    auto memSliceSizes = mlir::applyPermutationMap<int64_t>(inputMemMap, sliceSizes);
    SmallVector<int64_t> sliceStrides(inputRank, 1);

    auto normalizedOutTy = ShaveCodeGen::normalizeType(op.getResult().getType());
    auto inputTy = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto extractResultTy = normalizedOutTy;
    auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputTy.getElementType());

    if (quantType) {
        // Since ExtractSlice cannot change types we need to operate on the storage type.
        // Create a StorageCast to the storage type, we'll cast this back after the ExtractSlice
        auto outQuantType = mlir::cast<mlir::quant::QuantizedType>(normalizedOutTy.getElementType());
        auto storageElemType = outQuantType.getStorageType();
        auto storageInTy = mlir::RankedTensorType::get(inputTy.getShape(), storageElemType, inputTy.getEncoding());
        extractResultTy =
                mlir::RankedTensorType::get(normalizedOutTy.getShape(), storageElemType, normalizedOutTy.getEncoding());
        input = rewriter.create<mlir::quant::StorageCastOp>(op.getLoc(), storageInTy, input);
    }

    // Build the extract slice op
    auto extractSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
            op.getLoc(), mlir::TypeRange{extractResultTy}, input, mlir::ValueRange{}, mlir::ValueRange{},
            mlir::ValueRange{}, memSliceOffsets, memSliceSizes, sliceStrides);

    mlir::Value result = extractSlice.getResult();
    if (quantType) {
        // Recast this back to the output quant type if needed.
        result = rewriter.create<mlir::quant::StorageCastOp>(op.getLoc(), normalizedOutTy, result).getResult();
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
}

}  // namespace

void ShaveCodeGen::populateIEDataMovementToTensorPatterns(mlir::RewritePatternSet& patternSet,
                                                          mlir::TypeConverter& typeConverter) {
    auto& ctx = *patternSet.getContext();
    patternSet.add<IESliceToExtractSlice>(typeConverter, &ctx);
}
