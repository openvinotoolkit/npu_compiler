//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

template <typename SrcOp>
mlir::Value emitLinalgRegion(SrcOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(op);
    VPUX_UNUSED(args);
    VPUX_UNUSED(resultTypes);
    VPUX_UNUSED(rewriter);
    VPUX_THROW("emitLinalgRegion not specialized for operator");
}

// Reduce layers
template <>
mlir::Value emitLinalgRegion<IE::ReduceMaxOp>(IE::ReduceMaxOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // The sw layer implementation doesn't handle NaNs or signed zeroes,
        // so it should be okay to have nnan/nsz.
        auto attr = mlir::arith::FastMathFlagsAttr::get(
                rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
        return rewriter.create<mlir::arith::MaximumFOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<mlir::arith::MaxSIOp>(loc, resultTypes, args);
    }
    return rewriter.create<mlir::arith::MaxUIOp>(loc, resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::ReduceMinOp>(IE::ReduceMinOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // The sw layer implementation doesn't handle NaNs or signed zeroes,
        // so it should be okay to have nnan/nsz.
        auto attr = mlir::arith::FastMathFlagsAttr::get(
                rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
        return rewriter.create<mlir::arith::MinimumFOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<mlir::arith::MinSIOp>(loc, resultTypes, args);
    }
    return rewriter.create<mlir::arith::MinUIOp>(loc, resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::ReduceL2Op>(IE::ReduceL2Op op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto in = args[0];
    auto accum = args[1];
    auto loc = op.getLoc();
    auto inTy = in.getType();
    if (mlir::isa<mlir::FloatType>(inTy)) {
        if (inTy != accum.getType()) {
            in = rewriter.create<mlir::arith::ExtFOp>(loc, accum.getType(), in);
        }
        in = rewriter.create<mlir::arith::MulFOp>(loc, accum.getType(), in, in);
        // We need the reassoc flag for auto-vectorization.
        return rewriter.create<mlir::arith::AddFOp>(loc, accum.getType(), accum, in,
                                                    mlir::arith::FastMathFlags::reassoc);
    }
    in = rewriter.create<mlir::arith::MulIOp>(loc, accum.getType(), in, in);
    return rewriter.create<mlir::arith::AddIOp>(loc, accum.getType(), accum, in);
}

template <>
mlir::Value emitLinalgRegion<IE::ReduceL1Op>(IE::ReduceL1Op op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto in = args[0];
    auto accum = args[1];
    auto loc = op.getLoc();
    auto inTy = in.getType();
    if (mlir::isa<mlir::FloatType>(inTy)) {
        in = rewriter.create<mlir::math::AbsFOp>(loc, in.getType(), in);
        if (inTy != accum.getType()) {
            in = rewriter.create<mlir::arith::ExtFOp>(loc, accum.getType(), in);
        }
        // We need the reassoc flag for auto-vectorization.
        return rewriter.create<mlir::arith::AddFOp>(loc, accum.getType(), accum, in,
                                                    mlir::arith::FastMathFlags::reassoc);
    }

    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    if (elTy.isSignedInteger()) {
        in = rewriter.create<mlir::math::AbsIOp>(loc, in.getType(), in);
    }
    if (inTy != accum.getType()) {
        in = rewriter.create<mlir::arith::ExtUIOp>(loc, accum.getType(), in);
    }
    return rewriter.create<mlir::arith::AddIOp>(loc, accum.getType(), accum, in);
}

template <>
mlir::Value emitLinalgRegion<IE::ReduceSumOp>(IE::ReduceSumOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto in = args[0];
    auto accum = args[1];
    auto loc = op.getLoc();
    auto inTy = in.getType();
    if (mlir::isa<mlir::FloatType>(inTy)) {
        if (inTy != accum.getType()) {
            in = rewriter.create<mlir::arith::ExtFOp>(loc, accum.getType(), in);
        }
        // We need the reassoc flag for auto-vectorization.
        return rewriter.create<mlir::arith::AddFOp>(loc, accum.getType(), accum, in,
                                                    mlir::arith::FastMathFlags::reassoc);
    }

    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    if (inTy != accum.getType()) {
        if (elTy.isSignedInteger()) {
            in = rewriter.create<mlir::arith::ExtSIOp>(loc, accum.getType(), in);
        } else {
            in = rewriter.create<mlir::arith::ExtUIOp>(loc, accum.getType(), in);
        }
    }
    return rewriter.create<mlir::arith::AddIOp>(loc, accum.getType(), accum, in);
}

// Infrastructure for reduce ops.

static mlir::Value getNullScalar(mlir::Operation* op, mlir::Type resultTy, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    mlir::TypedAttr constAttr;
    if (mlir::isa<IE::ReduceMaxOp>(op)) {
        if (mlir::isa<mlir::FloatType>(elTy)) {
            constAttr = rewriter.getFloatAttr(
                    resultTy, llvm::APFloat::getInf(mlir::cast<mlir::FloatType>(resultTy).getFloatSemantics(), true));
        } else {
            constAttr = rewriter.getIntegerAttr(
                    resultTy, llvm::APSInt::getMinValue(resultTy.getIntOrFloatBitWidth(), !elTy.isSignedInteger()));
        }
    } else if (mlir::isa<IE::ReduceMinOp>(op)) {
        if (mlir::isa<mlir::FloatType>(elTy)) {
            constAttr = rewriter.getFloatAttr(
                    resultTy, llvm::APFloat::getInf(mlir::cast<mlir::FloatType>(resultTy).getFloatSemantics(), false));
        } else {
            constAttr = rewriter.getIntegerAttr(
                    resultTy, llvm::APSInt::getMaxValue(resultTy.getIntOrFloatBitWidth(), !elTy.isSignedInteger()));
        }
    } else if (mlir::isa<IE::ReduceSumOp>(op) || mlir::isa<IE::ReduceL1Op>(op) || mlir::isa<IE::ReduceL2Op>(op) ||
               mlir::isa<IE::ReduceMeanOp>(op) || mlir::isa<IE::ReduceLogicalOrOp>(op)) {
        if (mlir::isa<mlir::FloatType>(elTy)) {
            constAttr = rewriter.getFloatAttr(
                    resultTy, llvm::APFloat::getZero(mlir::cast<mlir::FloatType>(resultTy).getFloatSemantics()));
        } else {
            constAttr = rewriter.getIntegerAttr(resultTy, llvm::APSInt::getZero(resultTy.getIntOrFloatBitWidth()));
        }
    } else if (mlir::isa<IE::ReduceLogicalAndOp>(op) || mlir::isa<IE::ReduceProdOp>(op)) {
        if (mlir::isa<mlir::FloatType>(elTy)) {
            constAttr = rewriter.getFloatAttr(
                    resultTy, llvm::APFloat::getOne(mlir::cast<mlir::FloatType>(resultTy).getFloatSemantics()));
        } else {
            llvm::APInt one(resultTy.getIntOrFloatBitWidth(), 1);
            constAttr = rewriter.getIntegerAttr(resultTy, one);
        }
    } else {
        VPUX_THROW("unknown null value for reduce op");
    }
    return rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), constAttr);
}

static mlir::AffineMap dropZeroResults(mlir::AffineMap& map) {
    // Should replace with AffineMap::dropZeroResults when this becomes available.
    auto exprs = llvm::to_vector(map.getResults());
    SmallVector<mlir::AffineExpr> newExprs;
    for (auto expr : map.getResults()) {
        auto constExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(expr);
        if (!constExpr || constExpr.getValue() != 0) {
            newExprs.push_back(expr);
        }
    }
    return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), newExprs, map.getContext());
}

std::optional<mlir::ArrayAttr> getInputPadAttr(mlir::Operation* op) {
    if (auto sum = mlir::dyn_cast<IE::ReduceSumOp>(op)) {
        return sum.getInputPadding();
    }

    if (auto mean = mlir::dyn_cast<IE::ReduceMeanOp>(op)) {
        return mean.getInputPadding();
    }
    return std::nullopt;
}

std::optional<mlir::ArrayAttr> getOutputPadAttr(mlir::Operation* op) {
    if (auto sum = mlir::dyn_cast<IE::ReduceSumOp>(op)) {
        return sum.getOutputPadding();
    }

    if (auto mean = mlir::dyn_cast<IE::ReduceMeanOp>(op)) {
        return mean.getOutputPadding();
    }
    return std::nullopt;
}

using EmitBodyCallback = std::function<mlir::Value(mlir::Operation*, mlir::ValueRange, llvm::ArrayRef<mlir::Type>,
                                                   mlir::PatternRewriter&)>;

static mlir::LogicalResult emitLinalgReduceHelper(mlir::Operation* op, EmitBodyCallback callback,
                                                  EmitBodyCallback normalizeCallback, bool accumNeedsF32Precision,
                                                  bool keepDims, SmallVector<int64_t>& axes,
                                                  mlir::PatternRewriter& rewriter) {
    // The reduce operation contains three stages:
    // 1. Producing an initial value for the output tensor filled with the neutral element
    //    for the reduce operation. This neutral element is dependent on the performed
    //    reduction.
    // 2. The actual reduction linalg operation. The operation will use the input memory
    //    order, so the output affine map and loop iterators need to be adjusted accordingly.
    //    The output shape will *not* keep the reduced dimensions, since otherwise the
    //    resulting linalg operation wouldn't be tilable.
    // 3. An element-wise normalization stage, which will perform the required post-reduction
    //    adjustments (e.g. sqrt for L2, or a div for mean). This stage will also add back
    //    any of the dropped reduced dimensions if necessary.
    auto input = op->getOperands().front();
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();
    auto inputPad = getInputPadAttr(op);

    inputType = ShaveCodeGen::getUnpaddedTensorType(inputType, op->getLoc(), inputPad);

    auto result = op->getResult(0);
    auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
    auto resultRank = resultType.getRank();
    mlir::RankedTensorType resultPaddedType = ShaveCodeGen::normalizeType(resultType);
    auto outputPad = getOutputPadAttr(op);
    resultType = ShaveCodeGen::getUnpaddedTensorType(resultType, op->getLoc(), outputPad);

    // Element type from the output of the normalization stage
    auto postNormElTy = ShaveCodeGen::getLinalgElementType(resultType, rewriter.getContext());
    auto forceF32ForReductionOutput = (accumNeedsF32Precision && mlir::isa<mlir::FloatType>(postNormElTy) &&
                                       postNormElTy.getIntOrFloatBitWidth() < 32);
    // Element type from the output of the reduction stage
    auto reduceResultElTy = forceF32ForReductionOutput ? mlir::Float32Type::get(rewriter.getContext()) : postNormElTy;
    auto inputMemMap = mlir::cast<vpux::NDTypeInterface>(op->getOperandTypes().front())
                               .getDimsOrder()
                               .toAffineMap(rewriter.getContext());
    auto outputMemMap =
            mlir::cast<vpux::NDTypeInterface>(result.getType()).getDimsOrder().toAffineMap(rewriter.getContext());
    auto inputLogicalShape = mlir::cast<vpux::NDTypeInterface>(inputType).getShape();

    bool hasNorm = normalizeCallback != nullptr || postNormElTy != reduceResultElTy || keepDims;

    // Reject the dynamic input type, at least for now.
    if (inputLogicalShape.isDynamic()) {
        return mlir::failure();
    }

    auto ndResultTy = mlir::cast<NDTypeInterface>(resultType);
    auto resultDimsOrder = DimsOrder::fromPermutation(ndResultTy.getDimsOrder().toPermutation());
    auto finalMemoryShape = resultDimsOrder.toMemoryOrder(ndResultTy.getShape()).raw();

    // Compute the logical operation affine map for the reduce. This always has a constant zero
    // expression on the reduced dimensions. We can drop the zeros when needed.
    SmallVector<mlir::AffineExpr> logicalAffineExprs;
    for (auto it : llvm::enumerate(inputLogicalShape)) {
        auto isReduce = llvm::any_of(axes, [&](auto dim) {
            return checked_cast<size_t>(dim) == it.index();
        });
        if (isReduce) {
            logicalAffineExprs.push_back(rewriter.getAffineConstantExpr(0));
            continue;
        }
        logicalAffineExprs.push_back(rewriter.getAffineDimExpr(it.index()));
    }
    auto operationLogicalMap = mlir::AffineMap::get(inputRank, 0, logicalAffineExprs, rewriter.getContext());

    // Phase 1, initialize the reduce operation output.

    // Compute the reduction stage's output shape. This is the same shape we need to use when emitting
    // and initializing our empty tensor.
    SmallVector<int64_t> reduceShape = finalMemoryShape;
    if (keepDims) {
        auto logicalShape = ndResultTy.getShape().raw();
        auto outMap = outputMemMap.compose(operationLogicalMap);
        outMap = dropZeroResults(outMap);
        reduceShape = outMap.compose(logicalShape);
    }

    mlir::Value outputEmptyTensor = nullptr;
    mlir::Value padded = nullptr;
    if (outputPad && !hasNorm) {
        std::tie(outputEmptyTensor, padded) =
                ShaveCodeGen::emitTensorSlice(op->getLoc(), reduceShape, resultPaddedType, rewriter);
    } else {
        outputEmptyTensor = rewriter.create<mlir::tensor::EmptyOp>(op->getLoc(), reduceShape, reduceResultElTy);
    }
    auto nullScalar = getNullScalar(op, reduceResultElTy, rewriter);
    auto outputTensor = rewriter.create<mlir::linalg::FillOp>(op->getLoc(), mlir::ValueRange{nullScalar},
                                                              mlir::ValueRange{outputEmptyTensor})
                                .result();

    // Phase 2, emit the linalg reduce operation.
    auto linalgOperands = SmallVector<mlir::Value>(1, ShaveCodeGen::convertToLinalgValue(input, rewriter, inputPad));

    // Compute the affine map for the linalg output operand.
    // This is a composition of the output memory order permutation, the operations
    // logical map and the inverse input memory map (to account for the change of
    // the input memory map to an identity one).
    SmallVector<mlir::AffineMap> reductionAffineMaps;
    auto outputMap = operationLogicalMap.compose(mlir::inversePermutation(inputMemMap));
    if (!keepDims) {
        outputMap = dropZeroResults(outputMap);
    }
    outputMap = outputMemMap.compose(outputMap);
    if (keepDims) {
        outputMap = dropZeroResults(outputMap);
    }

    reductionAffineMaps.push_back(rewriter.getMultiDimIdentityMap(inputRank));
    reductionAffineMaps.push_back(outputMap);

    // Prepare the reduction loop iterators.
    SmallVector<mlir::utils::IteratorType, 5> reductionLoopAttrs;
    {
        llvm::SmallVector<mlir::utils::IteratorType> logicalLoopAttrs;
        for (auto it : llvm::enumerate(inputLogicalShape)) {
            auto isReduce = llvm::any_of(axes, [&](auto dim) {
                return checked_cast<size_t>(dim) == it.index();
            });
            if (isReduce) {
                logicalLoopAttrs.push_back(mlir::utils::IteratorType::reduction);
                continue;
            }
            logicalLoopAttrs.push_back(mlir::utils::IteratorType::parallel);
        }
        // Permute the logical loop attributes to account for the change in
        // the input order from the memory order to an identity mapping.
        for (mlir::AffineExpr expr : inputMemMap.getResults()) {
            auto dimExpr = mlir::cast<mlir::AffineDimExpr>(expr);
            reductionLoopAttrs.push_back(logicalLoopAttrs[dimExpr.getPosition()]);
        }
    }

    auto linalgOp = rewriter.create<mlir::linalg::GenericOp>(
            op->getLoc(), outputTensor.getType(), linalgOperands, outputTensor, reductionAffineMaps, reductionLoopAttrs,
            [&](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange blockArgs) {
                mlir::Value opResult = callback(op, blockArgs, {reduceResultElTy}, rewriter);
                opBuilder.create<mlir::linalg::YieldOp>(loc, opResult);
            });

    // Phase 3, post-reduction element-wise normalization

    if (hasNorm) {
        // The normalization stage element-wise so all loop attributes are parallel.
        SmallVector<mlir::utils::IteratorType> normalizeLoopAttrs(resultRank, mlir::utils::IteratorType::parallel);
        SmallVector<mlir::AffineMap, 2> normalizeAffineMaps;
        if (keepDims) {
            // In the keep_dims case we need to remove the reduced dimensions from
            // our input map to match what the output of the reduction stage.
            auto map = outputMemMap.compose(operationLogicalMap).compose(mlir::inversePermutation(outputMemMap));
            normalizeAffineMaps.push_back(dropZeroResults(map));
        } else {
            normalizeAffineMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));
        }
        normalizeAffineMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));
        // We'll need a new empty tensor if the normalization operation results either
        // changes shape or has e different element type.
        bool needsTruncate = (postNormElTy != reduceResultElTy);
        mlir::Value normalizeOutputTensor = nullptr;
        if (outputPad) {
            std::tie(normalizeOutputTensor, padded) =
                    ShaveCodeGen::emitTensorSlice(op->getLoc(), finalMemoryShape, resultPaddedType, rewriter);
        } else if (needsTruncate || keepDims) {
            normalizeOutputTensor =
                    rewriter.create<mlir::tensor::EmptyOp>(op->getLoc(), finalMemoryShape, postNormElTy);
        } else {
            normalizeOutputTensor = linalgOp.getResult(0);
        }

        linalgOp = rewriter.create<mlir::linalg::GenericOp>(
                op->getLoc(), normalizeOutputTensor.getType(), mlir::ValueRange{linalgOp.getResult(0)},
                normalizeOutputTensor, normalizeAffineMaps, normalizeLoopAttrs,
                [&](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange blockArgs) {
                    mlir::Value opResult = normalizeCallback ? normalizeCallback(op, blockArgs.take_front(1),
                                                                                 {reduceResultElTy}, rewriter)
                                                             : blockArgs.front();
                    if (needsTruncate) {
                        opResult = rewriter.create<mlir::arith::TruncFOp>(op->getLoc(), postNormElTy, opResult);
                    }
                    opBuilder.create<mlir::linalg::YieldOp>(loc, opResult);
                });
    }

    if (outputPad) {
        SmallVector<mlir::OpFoldResult> zeros(resultRank, rewriter.getIndexAttr(0));
        SmallVector<mlir::OpFoldResult> ones(resultRank, rewriter.getIndexAttr(1));
        SmallVector<mlir::OpFoldResult> sizes =
                mlir::tensor::getMixedSizes(rewriter, op->getLoc(), linalgOp->getResult(0));

        auto insertSliceOp = rewriter.create<mlir::tensor::InsertSliceOp>(op->getLoc(), linalgOp->getResult(0), padded,
                                                                          zeros, sizes, ones);

        rewriter.replaceOp(op, ShaveCodeGen::convertFromLinalgValue(insertSliceOp->getResult(0),
                                                                    op->getResult(0).getType(), rewriter)
                                       .getDefiningOp());
        return mlir::success();
    }

    rewriter.replaceOp(
            op, ShaveCodeGen::convertFromLinalgValue(linalgOp->getResult(0), op->getResult(0).getType(), rewriter)
                        .getDefiningOp());
    return mlir::success();
}

template <typename SrcOp>
EmitBodyCallback getNormalizationCallback() {
    if (std::is_same<SrcOp, IE::ReduceL2Op>::value) {
        return [](mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> types,
                  mlir::PatternRewriter& rewriter) -> mlir::Value {
            VPUX_UNUSED(types);
            // Convert to float if integer-like.
            // Type should be f32
            auto argVal = args[0];
            bool isFloat = mlir::isa<mlir::FloatType>(args[0].getType());
            auto origTy = argVal.getType();
            if (!isFloat) {
                mlir::FloatType fpType = mlir::Float32Type::get(rewriter.getContext());
                if (args[0].getType().getIntOrFloatBitWidth() > 32) {
                    fpType = mlir::Float64Type::get(rewriter.getContext());
                }
                // We don't have to look at the signdness, this should be positive or we'll get a NAN
                // from the sqrt.
                argVal = rewriter.create<mlir::arith::UIToFPOp>(op->getLoc(), fpType, argVal);
            }
            argVal = rewriter.create<mlir::math::SqrtOp>(op->getLoc(), argVal.getType(), argVal,
                                                         mlir::arith::FastMathFlags::afn);
            if (isFloat) {
                return argVal;
            }
            return rewriter.create<mlir::arith::FPToUIOp>(op->getLoc(), origTy, argVal);
        };
    }
    if (std::is_same<SrcOp, IE::ReduceMeanOp>::value) {
        VPUX_THROW("ReduceMean not supported");
    }
    return nullptr;
}

template <typename SrcOp>
constexpr bool reduceRequiresF32Accumulator() {
    if (std::is_same<SrcOp, IE::ReduceL2Op>::value || std::is_same<SrcOp, IE::ReduceL1Op>::value ||
        std::is_same<SrcOp, IE::ReduceSumOp>::value || std::is_same<SrcOp, IE::ReduceMeanOp>::value ||
        std::is_same<SrcOp, IE::ReduceProdOp>::value) {
        return true;
    }
    return false;
}

template <typename SrcOp>
class IEReduceToLinalg : public mlir::OpRewritePattern<SrcOp> {
public:
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const final {
        auto emitBody = [](mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> types,
                           mlir::PatternRewriter& rewriter) {
            return emitLinalgRegion<SrcOp>(mlir::cast<SrcOp>(op), args, types, rewriter);
        };
        bool keepDims = op.getKeepDims();
        SmallVector<int64_t> axes = parseIntArrayAttr<int64_t>(op.getAxesValue().value());
        return emitLinalgReduceHelper(op, emitBody, getNormalizationCallback<SrcOp>(),
                                      /*accumNeedsF32Precision=*/reduceRequiresF32Accumulator<SrcOp>(), keepDims, axes,
                                      rewriter);
    }
};

}  // namespace

void ShaveCodeGen::populateIEReduceToLinalgPatterns(mlir::RewritePatternSet& patternSet) {
    auto& ctx = *patternSet.getContext();
    patternSet.add<IEReduceToLinalg<IE::ReduceMaxOp>, IEReduceToLinalg<IE::ReduceMinOp>,
                   IEReduceToLinalg<IE::ReduceL2Op>>(&ctx);
    patternSet.add<IEReduceToLinalg<IE::ReduceSumOp>, IEReduceToLinalg<IE::ReduceL1Op>>(&ctx);
}
