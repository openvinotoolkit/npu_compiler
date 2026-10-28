//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSELOGSOFTMAXVARIANTS
#define GEN_PASS_DEF_FUSELOGSOFTMAXVARIANTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// Helper functions
//

bool isInnerAxis(IE::LogSoftmaxOp logSoftmaxOp) {
    int64_t axis = parseIntAttr<int64_t>(logSoftmaxOp.getAxisIndAttr());

    const auto inOrder = DimsOrder::fromValue(logSoftmaxOp.getInput());

    if (axis < 0) {
        axis += inOrder.numDims();
    }
    MemDim md = inOrder.toMemDim(Dim(axis));

    const auto shape = getShape(logSoftmaxOp.getInput());
    auto nDims = checked_cast<uint32_t>(shape.size());

    // only inner mode is supported
    if (md.ind() != (int32_t)(nDims - 1)) {
        return false;
    }

    return true;
}

inline bool isTopKMaxValuesOutputUsed(IE::TopKOp topKOp) {
    return !topKOp.getOutputValues().use_empty();
}

//
// FuseLogSoftmaxPeakPattern
//

class FuseLogSoftmaxPeakPattern final : public mlir::OpRewritePattern<IE::LogSoftmaxOp> {
public:
    FuseLogSoftmaxPeakPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::LogSoftmaxOp>(ctx), _log(log) {
        setDebugName("FuseLogSoftmaxPeakPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::LogSoftmaxOp logSoftmaxOp, mlir::PatternRewriter& rewriter) const final {
        auto ctx = logSoftmaxOp->getContext();
        auto f16Type = mlir::Float16Type::get(ctx);
        auto f32Type = mlir::Float32Type::get(ctx);
        auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
        auto si64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);

        if (!isInnerAxis(logSoftmaxOp)) {
            return mlir::failure();
        }

        // Find the path to LogSoftmax input
        // Pattern 1: Expand -> LogSoftmax
        // Pattern 2: Expand -> AffineReshape -> LogSoftmax
        mlir::Value logSoftmaxInput = logSoftmaxOp.getInput();
        IE::AffineReshapeOp reshapeBeforeLogSoftmax = nullptr;
        IE::ExpandOp expandOp = nullptr;

        reshapeBeforeLogSoftmax = logSoftmaxInput.getDefiningOp<IE::AffineReshapeOp>();
        if (reshapeBeforeLogSoftmax != nullptr) {
            expandOp = reshapeBeforeLogSoftmax.getInput().getDefiningOp<IE::ExpandOp>();
        } else {
            expandOp = logSoftmaxInput.getDefiningOp<IE::ExpandOp>();
        }

        if (expandOp == nullptr) {
            return mlir::failure();
        }

        auto commonInput = expandOp.getInput();
        auto commonInputType = mlir::cast<vpux::NDTypeInterface>(commonInput.getType());

        // Verify input is f16
        if (commonInputType.getElementType() != f16Type) {
            return mlir::failure();
        }

        const auto numCommonInputUses = std::distance(commonInput.getUses().begin(), commonInput.getUses().end());
        if (numCommonInputUses != 2) {
            return mlir::failure();
        }

        // Find TopK, it could be directly from commonInput or through AffineReshape
        IE::TopKOp topKOp = nullptr;
        IE::AffineReshapeOp reshapeBeforeTopK = nullptr;

        for (auto user : commonInput.getUsers()) {
            if (user == expandOp.getOperation()) {
                continue;
            }

            // Check if user is AffineReshape -> TopK
            if (auto reshape = mlir::dyn_cast<IE::AffineReshapeOp>(user)) {
                if (reshape.getOutput().hasOneUse()) {
                    if (auto topK = mlir::dyn_cast<IE::TopKOp>(*reshape.getOutput().getUsers().begin())) {
                        reshapeBeforeTopK = reshape;
                        topKOp = topK;
                        break;
                    }
                }
            }
            // Check if user is directly TopK
            if (auto topK = mlir::dyn_cast<IE::TopKOp>(user)) {
                topKOp = topK;
                break;
            }
        }

        if (topKOp == nullptr) {
            return mlir::failure();
        }

        if (isTopKMaxValuesOutputUsed(topKOp)) {
            return mlir::failure();
        }

        if (topKOp.getKValue() != 1) {
            return mlir::failure();
        }

        // Validate axis match (also ensures TopK axis is on the inner size)
        auto topKAxis = topKOp.getAxis();
        auto logSoftmaxAxis = parseIntAttr<int64_t>(logSoftmaxOp.getAxisIndAttr());
        if (topKAxis != logSoftmaxAxis) {
            return mlir::failure();
        }

        // Validate LogSoftmax output chain: LogSoftmax -> Slice -> AffineReshape -> GatherElements
        if (!logSoftmaxOp.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(*logSoftmaxOp.getOutput().getUsers().begin());
        if (sliceOp == nullptr || !sliceOp.getResult().hasOneUse()) {
            return mlir::failure();
        }

        auto reshapeBeforeGather = mlir::dyn_cast<IE::AffineReshapeOp>(*sliceOp.getResult().getUsers().begin());
        if (reshapeBeforeGather == nullptr || !reshapeBeforeGather.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto gatherElementsOp =
                mlir::dyn_cast<IE::GatherElementsOp>(*reshapeBeforeGather.getOutput().getUsers().begin());
        if (gatherElementsOp == nullptr || !gatherElementsOp.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        // TopK should have 2 users: one for GatherElements and one for output
        auto topKIndices = topKOp.getTargetShape();
        const auto numTopKIndicesUses = std::distance(topKIndices.getUses().begin(), topKIndices.getUses().end());
        if (numTopKIndicesUses != 2) {
            return mlir::failure();
        }

        // Find the AffineReshape that feeds into GatherElements and the one for output
        IE::AffineReshapeOp indicesReshapeForGather = nullptr;
        IE::AffineReshapeOp indicesReshapeForOutput = nullptr;
        for (auto user : topKIndices.getUsers()) {
            if (auto reshape = mlir::dyn_cast<IE::AffineReshapeOp>(user)) {
                // Check if this reshape feeds into GatherElements or Convert
                if (reshape.getOutput().hasOneUse()) {
                    auto reshapeUser = *reshape.getOutput().getUsers().begin();
                    if (mlir::isa<IE::GatherElementsOp>(reshapeUser)) {
                        indicesReshapeForGather = reshape;
                    } else if (mlir::isa<IE::ConvertOp>(reshapeUser)) {
                        indicesReshapeForOutput = reshape;
                    }
                }
            }
        }

        if (indicesReshapeForGather == nullptr || indicesReshapeForOutput == nullptr) {
            return mlir::failure();
        }

        // Verify the GatherElements indices input matches
        if (gatherElementsOp.getIndices().getDefiningOp() != indicesReshapeForGather) {
            return mlir::failure();
        }

        // Validate GatherElements output chain: GatherElements -> AffineReshape -> Convert(f16->f32)
        auto reshapeAfterGather = mlir::dyn_cast<IE::AffineReshapeOp>(*gatherElementsOp.getOutput().getUsers().begin());
        if (reshapeAfterGather == nullptr || !reshapeAfterGather.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto peakValuesConvert = mlir::dyn_cast<IE::ConvertOp>(*reshapeAfterGather.getOutput().getUsers().begin());
        if (peakValuesConvert == nullptr) {
            return mlir::failure();
        }

        auto peakConvertInputType = mlir::cast<vpux::NDTypeInterface>(peakValuesConvert.getInput().getType());
        auto peakConvertOutputType = mlir::cast<vpux::NDTypeInterface>(peakValuesConvert.getOutput().getType());
        if (peakConvertInputType.getElementType() != f16Type || peakConvertOutputType.getElementType() != f32Type) {
            return mlir::failure();
        }

        // Validate TopK indices output chain: AffineReshape -> Convert(si32->si64)
        auto indicesConvert = mlir::dyn_cast<IE::ConvertOp>(*indicesReshapeForOutput.getOutput().getUsers().begin());
        if (indicesConvert == nullptr) {
            return mlir::failure();
        }

        auto indicesConvertInputType = mlir::cast<vpux::NDTypeInterface>(indicesConvert.getInput().getType());
        auto indicesConvertOutputType = mlir::cast<vpux::NDTypeInterface>(indicesConvert.getOutput().getType());
        if (indicesConvertInputType.getElementType() != si32Type ||
            indicesConvertOutputType.getElementType() != si64Type) {
            return mlir::failure();
        }

        _log.trace("LogSoftmaxPeak pattern matched for operation at {0}", logSoftmaxOp.getLoc());

        mlir::Value fusedInput = logSoftmaxOp.getInput();
        auto inputType = mlir::cast<vpux::NDTypeInterface>(fusedInput.getType());
        auto inputShape = inputType.getShape().raw();

        // Output shape should be the same as input, but the axis dimension set to 1
        SmallVector<int64_t> outputShapeVec(inputShape.begin(), inputShape.end());
        outputShapeVec[logSoftmaxAxis] = 1;
        Shape outputShape(outputShapeVec);

        auto peakValuesInferredType = inputType.changeShapeElemType(outputShape, f32Type);
        auto indicesInferredType = inputType.changeShapeElemType(outputShape, si64Type);

        auto dstElemTypeAttr = mlir::TypeAttr::get(f32Type);

        // Set insertion point after the input to LogSoftmax
        if (reshapeBeforeLogSoftmax != nullptr) {
            rewriter.setInsertionPointAfter(reshapeBeforeLogSoftmax);
        } else {
            rewriter.setInsertionPointAfter(expandOp);
        }

        auto fusedOp = IE::LogSoftmaxPeakOp::create(rewriter, logSoftmaxOp.getLoc(), peakValuesInferredType,
                                                    indicesInferredType, fusedInput, logSoftmaxOp.getAxisIndAttr(),
                                                    logSoftmaxOp.getPadSizeAttr(), dstElemTypeAttr);

        auto finalPeakValuesType = mlir::cast<vpux::NDTypeInterface>(peakValuesConvert.getOutput().getType());
        auto peakValuesReshape = IE::AffineReshapeOp::create(
                rewriter, reshapeAfterGather.getLoc(), finalPeakValuesType, fusedOp.getOutput(),
                reshapeAfterGather.getDimMappingAttr(), reshapeAfterGather.getShapeValueAttr());

        auto finalIndicesType = mlir::cast<vpux::NDTypeInterface>(indicesConvert.getOutput().getType());
        auto indicesReshape = IE::AffineReshapeOp::create(
                rewriter, indicesReshapeForOutput.getLoc(), finalIndicesType, fusedOp.getTopKOutput(),
                indicesReshapeForOutput.getDimMappingAttr(), indicesReshapeForOutput.getShapeValueAttr());

        rewriter.replaceAllUsesWith(peakValuesConvert.getOutput(), peakValuesReshape.getOutput());
        rewriter.replaceAllUsesWith(indicesConvert.getOutput(), indicesReshape.getOutput());

        rewriter.eraseOp(peakValuesConvert);
        rewriter.eraseOp(reshapeAfterGather);
        rewriter.eraseOp(gatherElementsOp);
        rewriter.eraseOp(indicesReshapeForGather);
        rewriter.eraseOp(indicesConvert);
        rewriter.eraseOp(indicesReshapeForOutput);
        rewriter.eraseOp(reshapeBeforeGather);
        rewriter.eraseOp(sliceOp);
        rewriter.eraseOp(logSoftmaxOp);
        rewriter.eraseOp(topKOp);

        return mlir::success();
    }

private:
    Logger _log;
};

//
// Find TopKOp from PermuteCast pattern
//

IE::TopKOp findTopKFromPermuteCastPattern(IE::ExpandOp expandOp) {
    auto expandInput = expandOp.getInput();

    // Check if Expand input is directly from PermuteCast
    auto permuteCastOp = expandInput.getDefiningOp<IE::PermuteCastOp>();
    if (permuteCastOp == nullptr) {
        // Try the old pattern: Expand <- AffineReshape <- PermuteCast
        auto reshapeBeforeExpand = expandInput.getDefiningOp<IE::AffineReshapeOp>();
        if (reshapeBeforeExpand == nullptr) {
            return nullptr;
        }
        permuteCastOp = reshapeBeforeExpand.getInput().getDefiningOp<IE::PermuteCastOp>();
        if (permuteCastOp == nullptr) {
            return nullptr;
        }
    }

    auto numUsers =
            std::distance(permuteCastOp.getOutput().getUsers().begin(), permuteCastOp.getOutput().getUsers().end());
    if (numUsers != 2) {
        return nullptr;
    }

    IE::AffineReshapeOp topKInputReshape = nullptr;
    for (auto user : permuteCastOp.getOutput().getUsers()) {
        if (auto reshape = mlir::dyn_cast<IE::AffineReshapeOp>(user)) {
            topKInputReshape = reshape;
            break;
        }
    }

    if (topKInputReshape == nullptr || !topKInputReshape.getOutput().hasOneUse()) {
        return nullptr;
    }

    return mlir::dyn_cast<IE::TopKOp>(*topKInputReshape.getOutput().getUsers().begin());
}

//
// Find TopKOp from Convert pattern
//

IE::TopKOp findTopKFromConvertPattern(IE::ExpandOp expandOp, mlir::MLIRContext* ctx) {
    auto f16Type = mlir::Float16Type::get(ctx);
    auto f32Type = mlir::Float32Type::get(ctx);

    auto firstConvertOp = expandOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (firstConvertOp == nullptr) {
        return nullptr;
    }

    auto firstConvertInputType = mlir::cast<vpux::NDTypeInterface>(firstConvertOp.getInput().getType());
    auto firstConvertOutputType = mlir::cast<vpux::NDTypeInterface>(firstConvertOp.getOutput().getType());
    if (firstConvertInputType.getElementType() != f32Type || firstConvertOutputType.getElementType() != f16Type) {
        return nullptr;
    }

    const auto numUses =
            std::distance(firstConvertOp.getOutput().getUses().begin(), firstConvertOp.getOutput().getUses().end());
    if (numUses != 2) {
        return nullptr;
    }

    for (auto user : firstConvertOp.getOutput().getUsers()) {
        if (auto topK = mlir::dyn_cast<IE::TopKOp>(user)) {
            return topK;
        }
    }

    return nullptr;
}

//
// FuseLogSoftmaxTopKPattern
//

class FuseLogSoftmaxTopKPattern final : public mlir::OpRewritePattern<IE::LogSoftmaxOp> {
public:
    FuseLogSoftmaxTopKPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::LogSoftmaxOp>(ctx), _log(log) {
        setDebugName("FuseLogSoftmaxTopKPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::LogSoftmaxOp logSoftmaxOp, mlir::PatternRewriter& rewriter) const final {
        auto ctx = logSoftmaxOp->getContext();
        auto f16Type = mlir::Float16Type::get(ctx);
        auto f32Type = mlir::Float32Type::get(ctx);
        auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
        auto si64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);

        if (!isInnerAxis(logSoftmaxOp)) {
            return mlir::failure();
        }

        // Find ExpandOp in the input chain
        auto logSoftmaxInput = logSoftmaxOp.getInput();
        IE::ExpandOp expandOp = nullptr;
        mlir::Value expandOutput = logSoftmaxInput;

        // Pattern 1: LogSoftmax <- AffineReshape <- Expand
        if (auto reshapeBeforeLogSoftmax = logSoftmaxInput.getDefiningOp<IE::AffineReshapeOp>()) {
            expandOp = reshapeBeforeLogSoftmax.getInput().getDefiningOp<IE::ExpandOp>();
            expandOutput = reshapeBeforeLogSoftmax.getOutput();
        }
        // Pattern 2: LogSoftmax <- Expand
        else {
            expandOp = logSoftmaxInput.getDefiningOp<IE::ExpandOp>();
            expandOutput = logSoftmaxInput;
        }

        if (expandOp == nullptr) {
            return mlir::failure();
        }

        // Try to find TopKOp from both patterns
        IE::TopKOp topKOp = findTopKFromPermuteCastPattern(expandOp);
        if (topKOp == nullptr) {
            topKOp = findTopKFromConvertPattern(expandOp, ctx);
        }

        if (topKOp == nullptr) {
            return mlir::failure();
        }

        // Shouldn't fuse if the maximum values output is used by some operation
        if (isTopKMaxValuesOutputUsed(topKOp)) {
            return mlir::failure();
        }

        // Validate output chain of LogSoftmax
        if (!logSoftmaxOp.getOutput().hasOneUse()) {
            return mlir::failure();
        }
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(*logSoftmaxOp.getOutput().getUsers().begin());
        if (sliceOp == nullptr || !sliceOp.getResult().hasOneUse()) {
            return mlir::failure();
        }

        auto sliceConvertOp = mlir::dyn_cast<IE::ConvertOp>(*sliceOp.getResult().getUsers().begin());
        if (sliceConvertOp == nullptr || !sliceConvertOp.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto sliceConvertInputType = mlir::cast<vpux::NDTypeInterface>(sliceConvertOp.getInput().getType());
        auto sliceConvertOutputType = mlir::cast<vpux::NDTypeInterface>(sliceConvertOp.getOutput().getType());
        if (sliceConvertInputType.getElementType() != f16Type || sliceConvertOutputType.getElementType() != f32Type) {
            return mlir::failure();
        }

        // Validate output chain of TopK
        if (!topKOp.getTargetShape().hasOneUse()) {
            return mlir::failure();
        }

        auto topKReshape1 = mlir::dyn_cast<IE::AffineReshapeOp>(*topKOp.getTargetShape().getUsers().begin());
        if (topKReshape1 == nullptr || !topKReshape1.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto topKConvertOp = mlir::dyn_cast<IE::ConvertOp>(*topKReshape1.getOutput().getUsers().begin());
        if (topKConvertOp == nullptr || !topKConvertOp.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto topKConvertInputType = mlir::cast<vpux::NDTypeInterface>(topKConvertOp.getInput().getType());
        auto topKConvertOutputType = mlir::cast<vpux::NDTypeInterface>(topKConvertOp.getOutput().getType());
        if (topKConvertInputType.getElementType() != si32Type || topKConvertOutputType.getElementType() != si64Type) {
            return mlir::failure();
        }

        // Validate axis match (also ensures TopK axis is on the innermost dim)
        auto topKAxis = topKOp.getAxis();
        auto logSoftmaxAxis = parseIntAttr<int64_t>(logSoftmaxOp.getAxisIndAttr());
        if (topKAxis != logSoftmaxAxis) {
            return mlir::failure();
        }

        _log.trace("LogSoftmaxTopK pattern matched for operation at {0}", logSoftmaxOp.getLoc());

        // Create fused operation
        auto dstElemTypeAttr = mlir::TypeAttr::get(f32Type);

        auto inputType = mlir::cast<vpux::NDTypeInterface>(expandOutput.getType());
        auto outputType = inputType.changeElemType(f32Type);

        auto topKOutputOrigType = mlir::cast<vpux::NDTypeInterface>(topKOp.getTargetShape().getType());
        auto topKOutputType = topKOutputOrigType.changeElemType(si64Type);

        rewriter.setInsertionPointAfter(expandOutput.getDefiningOp());

        auto fusedLogSoftmax = IE::LogSoftmaxTopKOp::create(rewriter, logSoftmaxOp.getLoc(), outputType, topKOutputType,
                                                            expandOutput, logSoftmaxOp.getAxisIndAttr(),
                                                            logSoftmaxOp.getPadSizeAttr(), dstElemTypeAttr);

        auto originalSliceType = mlir::cast<vpux::NDTypeInterface>(sliceOp.getResult().getType());
        auto newSliceType = originalSliceType.changeElemType(f32Type);

        auto newSliceOp = IE::SliceOp::create(rewriter, sliceOp.getLoc(), newSliceType, fusedLogSoftmax.getOutput(),
                                              sliceOp.getStaticOffsetsAttr(), sliceOp.getStaticSizesAttr());

        auto topKReshape1OrigType = mlir::cast<vpux::NDTypeInterface>(topKReshape1.getOutput().getType());
        auto newTopKReshape1Type = topKReshape1OrigType.changeElemType(si64Type);

        auto newTopKReshape1 = IE::AffineReshapeOp::create(
                rewriter, topKReshape1.getLoc(), newTopKReshape1Type, fusedLogSoftmax.getTopKOutput(),
                topKReshape1.getDimMappingAttr(), topKReshape1.getShapeValueAttr());

        rewriter.replaceAllUsesWith(sliceConvertOp.getOutput(), newSliceOp.getOutput());
        rewriter.replaceAllUsesWith(topKConvertOp.getOutput(), newTopKReshape1.getOutput());

        rewriter.eraseOp(sliceConvertOp);
        rewriter.eraseOp(sliceOp);

        rewriter.eraseOp(topKConvertOp);
        rewriter.eraseOp(topKReshape1);
        rewriter.eraseOp(topKOp);

        rewriter.eraseOp(logSoftmaxOp);

        return mlir::success();
    }

private:
    Logger _log;
};

//
// FuseLogSoftmaxVariantsPass
//

class FuseLogSoftmaxVariantsPass final : public IE::impl::FuseLogSoftmaxVariantsBase<FuseLogSoftmaxVariantsPass> {
public:
    explicit FuseLogSoftmaxVariantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseLogSoftmaxVariantsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseLogSoftmaxPeakPattern>(&ctx, _log);
    patterns.add<FuseLogSoftmaxTopKPattern>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseLogSoftmaxVariantsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseLogSoftmaxVariantsPass(Logger log) {
    return std::make_unique<FuseLogSoftmaxVariantsPass>(log);
}
