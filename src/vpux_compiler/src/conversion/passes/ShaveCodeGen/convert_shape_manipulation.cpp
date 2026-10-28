//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/jit_utils.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

static mlir::LogicalResult convertViewLikeOp(mlir::Operation* op, mlir::Value input, mlir::PatternRewriter& rewriter) {
    // Collapse to a 1D tensor then expand to the output memory shape.
    // This sequence should then be canonicalized if possible.
    // At the moment we don't do anything more complex since there is no evidence
    // that it would justify the added complexity.
    auto loc = op->getLoc();

    auto numElem = mlir::cast<NDTypeInterface>(input.getType()).getNumElements();
    auto elTy = mlir::cast<NDTypeInterface>(input.getType()).getElementType();
    auto inputRank = mlir::cast<NDTypeInterface>(input.getType()).getRank();

    // For quantized types, cast to storage type before shape operations
    auto ndResultTy = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto shapeOpElementType = elTy;

    bool isQuantized = llvm::isa<mlir::quant::QuantizedType>(elTy);
    if (isQuantized) {
        // Cast input to storage type before collapse/expand
        auto quantTy = llvm::cast<mlir::quant::QuantizedType>(elTy);
        auto storageElTy = quantTy.getStorageType();
        auto inputQuantCastType = mlir::RankedTensorType::get(
                mlir::cast<mlir::RankedTensorType>(input.getType()).getShape(), storageElTy);
        input = mlir::quant::StorageCastOp::create(rewriter, loc, inputQuantCastType, input);
        shapeOpElementType = storageElTy;
    }

    // Collapse to a 1d tensor
    SmallVector<int64_t> collapseResultShape(1, numElem);
    mlir::ReassociationIndices dimCollapse(inputRank);
    std::iota(std::begin(dimCollapse), std::end(dimCollapse), 0);
    SmallVector<mlir::ReassociationIndices> collapseReassocMap;
    collapseReassocMap.emplace_back(dimCollapse);
    auto collapseResultType = mlir::RankedTensorType::get(collapseResultShape, shapeOpElementType);

    input = mlir::tensor::CollapseShapeOp::create(rewriter, loc, collapseResultType, input, collapseReassocMap);

    // Expand to target shape
    auto resultDimsOrder = DimsOrder::fromPermutation(ndResultTy.getDimsOrder().toPermutation());
    auto resultShape = resultDimsOrder.toMemoryOrder(ndResultTy.getShape()).raw();
    auto expandResultType = mlir::RankedTensorType::get(resultShape, shapeOpElementType);

    mlir::ReassociationIndices dimExpand(ndResultTy.getRank());
    std::iota(std::begin(dimExpand), std::end(dimExpand), 0);
    SmallVector<mlir::ReassociationIndices> expandReassocMap;
    expandReassocMap.emplace_back(dimExpand);

    input = mlir::tensor::ExpandShapeOp::create(rewriter, loc, expandResultType, input, expandReassocMap);

    // For quantized types, cast back to the output quantized type
    if (isQuantized) {
        auto outputNormalizedTy =
                ShaveCodeGen::normalizeType(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()));
        input = mlir::quant::StorageCastOp::create(rewriter, loc, outputNormalizedTy, input);
    }

    rewriter.replaceOp(op, input);

    return mlir::success();
}

// Emit a copy for a view-like operation which only adds and/or removes unit dimensions of the
// in-memory shape of its input. Since the operation doesn't move any data, the non-unit
// dimensions of the two memory shapes match one to one, in order, and the copy only needs to
// re-index the unit dimensions.
static mlir::LogicalResult convertUnitDimViewLikeOp(mlir::Operation* op, mlir::Value input,
                                                    mlir::ConversionPatternRewriter& rewriter) {
    auto* ctx = rewriter.getContext();
    const auto loc = op->getLoc();

    const auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    const auto resultType = ShaveCodeGen::normalizeType(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()));

    const auto inputShape = inputType.getShape();
    const auto resultShape = resultType.getShape();

    auto copyElemType = inputType.getElementType();
    const bool isQuantized = mlir::isa<mlir::quant::QuantizedType>(copyElemType);
    if (isQuantized) {
        // The linalg operation cannot work on quantized element types, so the copy operates on the
        // storage type.
        copyElemType = mlir::cast<mlir::quant::QuantizedType>(copyElemType).getStorageType();
        const auto storageInputType = mlir::RankedTensorType::get(inputShape, copyElemType);
        input = mlir::quant::StorageCastOp::create(rewriter, loc, storageInputType, input);
    }

    // Collect the iterators of the non-unit output dimensions, in order, so that they can be
    // matched against the non-unit input dimensions. The iterators of the unit output dimensions
    // are collected separately and are reused for the unit input dimensions.
    SmallVector<unsigned> nonUnitResultDims;
    SmallVector<unsigned> unitResultDims;
    for (auto dim : llvm::enumerate(resultShape)) {
        auto& resultDims = dim.value() != 1 ? nonUnitResultDims : unitResultDims;
        resultDims.push_back(checked_cast<unsigned>(dim.index()));
    }

    const auto resultRank = checked_cast<unsigned>(resultShape.size());

    SmallVector<mlir::AffineExpr> inputExprs;
    inputExprs.reserve(inputShape.size());
    size_t nextNonUnitDim = 0;
    size_t nextUnitDim = 0;
    auto nextExtraUnitDim = resultRank;
    for (auto dim : inputShape) {
        if (dim == 1) {
            // Reuse the iterator of an unmatched unit output dimension if there is one left,
            // otherwise append a new iterator to the iteration space.
            const auto unitDim =
                    nextUnitDim < unitResultDims.size() ? unitResultDims[nextUnitDim++] : nextExtraUnitDim++;
            inputExprs.push_back(mlir::getAffineDimExpr(unitDim, ctx));
            continue;
        }
        VPUX_THROW_UNLESS(nextNonUnitDim < nonUnitResultDims.size(),
                          "The non-unit dimensions of '{0}' and '{1}' don't match at '{2}'", inputShape, resultShape,
                          loc);
        inputExprs.push_back(mlir::getAffineDimExpr(nonUnitResultDims[nextNonUnitDim++], ctx));
    }

    const auto numIterators = nextExtraUnitDim;
    auto resultExprs = llvm::to_vector(llvm::map_range(llvm::seq<unsigned>(0, resultRank), [&](unsigned dim) {
        return mlir::getAffineDimExpr(dim, ctx);
    }));
    const SmallVector<mlir::AffineMap> copyAffineMaps{mlir::AffineMap::get(numIterators, 0, inputExprs, ctx),
                                                      mlir::AffineMap::get(numIterators, 0, resultExprs, ctx)};
    const SmallVector<mlir::utils::IteratorType> copyLoopAttrs(numIterators, mlir::utils::IteratorType::parallel);

    const auto copyResultType = mlir::RankedTensorType::get(resultShape, copyElemType);
    auto outputTensor = mlir::tensor::EmptyOp::create(rewriter, loc, resultShape, copyElemType).getResult();

    auto linalgOp = mlir::linalg::GenericOp::create(
            rewriter, loc, copyResultType, mlir::ValueRange{input}, outputTensor, copyAffineMaps, copyLoopAttrs,
            [](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange blockArgs) {
                mlir::linalg::YieldOp::create(opBuilder, loc, blockArgs.front());
            });

    mlir::Value result = linalgOp.getResult(0);
    if (isQuantized) {
        // If required, perform a cast to the output quantized type.
        result = mlir::quant::StorageCastOp::create(rewriter, loc, resultType, result).getResult();
    }

    rewriter.replaceOp(op, result);

    return mlir::success();
}

template <typename SrcOp>
class IEViewLikeToTensorOps : public mlir::OpConversionPattern<SrcOp> {
public:
    using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;
    using OpAdaptor = typename mlir::OpConversionPattern<SrcOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto rankedOutTy = mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
        if (adaptor.getOperands()[0].getType() == ShaveCodeGen::normalizeType(rankedOutTy)) {
            rewriter.replaceOp(op, adaptor.getOperands()[0]);
            return mlir::success();
        }

        if (ShaveCodeGen::hasMatchingNonUnitMemDims(mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()),
                                                    mlir::cast<vpux::NDTypeInterface>(rankedOutTy))) {
            // Operations which only add and/or remove unit dimensions of the in-memory shape are
            // lowered to a tilable linalg.generic op.
            return convertUnitDimViewLikeOp(op, adaptor.getOperands()[0], rewriter);
        }

        // Default lowering path valid for any reshape using collapse/expand shape ops.
        return convertViewLikeOp(op, adaptor.getOperands()[0], rewriter);
    }
};

}  // namespace

void ShaveCodeGen::populateIEShapeManipulationToTensorPatterns(mlir::RewritePatternSet& patternSet,
                                                               mlir::TypeConverter& typeConverter) {
    auto& ctx = *patternSet.getContext();
    patternSet.add<IEViewLikeToTensorOps<IE::PermuteCastOp>, IEViewLikeToTensorOps<IE::AffineReshapeOp>>(typeConverter,
                                                                                                         &ctx);
}
