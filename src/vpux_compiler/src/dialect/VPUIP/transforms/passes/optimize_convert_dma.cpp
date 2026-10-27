//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/back_infer_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZECONVERTDMAOP
#define GEN_PASS_DEF_OPTIMIZECONVERTDMAOP
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

VPUIP::LayerOpInterface getConvertDMAOp(mlir::Operation* maybeConvertDMAOperation) {
    if (auto convertDMAOp = mlir::dyn_cast_or_null<VPUIP::ConvertDMAOp>(maybeConvertDMAOperation)) {
        return mlir::cast<VPUIP::LayerOpInterface>(*convertDMAOp);
    }
    return nullptr;
}

VPUIP::LayerOpInterface getCopyOp(mlir::Operation* sourceOp) {
    return mlir::dyn_cast_or_null<VPUIP::CopyOp>(sourceOp);
}

void replaceOpWithNewConvertDMAOp(mlir::PatternRewriter& rewriter, mlir::Value input, mlir::Value outputBuff,
                                  mlir::Operation* opToReplace) {
    rewriter.replaceOpWithNewOp<VPUIP::ConvertDMAOp>(opToReplace, input, outputBuff);
}

bool isCompactStride(vpux::NDTypeInterface typeIf) {
    const auto inReqs = StrideReqs::compact(typeIf.getRank());
    return inReqs.checkStrides(typeIf);
}

mlir::FailureOr<mlir::Value> createNewViewLikeOps(mlir::PatternRewriter& rewriter,
                                                  ArrayRef<mlir::Operation*> viewLikeOps, mlir::Value newInput) {
    SmallVector<mlir::Type> newOutputTypes;
    auto newInputType = VPUIP::extractDataType(newInput);
    for (auto* viewLikeOp : viewLikeOps) {
        const auto newOutputType = VPUIP::BackInferUtils::inferOutputType(viewLikeOp, newInputType);
        if (!newOutputType.has_value()) {
            return mlir::failure();
        }
        newOutputTypes.push_back(newOutputType.value());
        newInputType = newOutputType.value();
    }

    for (auto [viewLikeOp, newOutputType] : llvm::zip_equal(viewLikeOps, newOutputTypes)) {
        mlir::IRMapping mapper;
        mapper.map(viewLikeOp->getOperands(), ArrayRef({newInput}));
        auto* newViewLikeOp = rewriter.clone(*viewLikeOp, mapper);
        newViewLikeOp->getResult(0).setType(newOutputType);
        newInput = newViewLikeOp->getResult(0);
    }

    return newInput;
}

//
// ConvertDMAViewLikeCopy
//

class ConvertDMAViewLikeCopy : public mlir::OpRewritePattern<VPUIP::ConvertDMAOp> {
public:
    ConvertDMAViewLikeCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConvertDMAOp>(ctx), _log(log) {
        setDebugName("ConvertDMAViewLikeCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ConvertDMAOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertDMAViewLikeCopy::matchAndRewrite(VPUIP::ConvertDMAOp convertOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Copy at {1}", getDebugName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    if (!convertOp->hasOneUse()) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> viewLikeOps;
    auto userOp = *convertOp->getUsers().begin();
    while (VPUIP::isPureViewOp(userOp) && userOp->hasOneUse() && userOp->getNumOperands() == 1) {
        viewLikeOps.push_back(userOp);
        userOp = *userOp->getUsers().begin();
    }

    auto userCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(userOp);
    if (userCopyOp == nullptr) {
        return mlir::failure();
    }

    const auto convertInput = convertOp.getInput();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(convertInput));
    const auto newInputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);

    if (!viewLikeOps.empty()) {
        if (!userCopyOp->hasOneUse()) {
            return mlir::failure();
        }

        if (vpux::VPUIP::hasDistributedOperand(convertOp)) {
            return mlir::failure();
        }

        if (!isCompactStride(inputType)) {
            nestedLogger.trace("ConvertDMA input is strided at {0}", convertOp->getLoc());
            return mlir::failure();
        }

        if (llvm::any_of(viewLikeOps, [](mlir::Operation* op) {
                return mlir::isa<Core::ReinterpretCastOp>(op);
            })) {
            // E#179283: ReinterpretCast is a "boundary" that cannot be crossed.
            nestedLogger.trace(
                    "ConvertDMA at {0} has a ReinterpretCast operation after, the optimization is not possible",
                    convertOp->getLoc());
            return mlir::failure();
        }
    }

    const auto outputBuffer = userCopyOp.getOutputBuff();
    const auto newOutputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(outputBuffer));
    if (newInputDistType != nullptr && newOutputDistType != nullptr &&
        mlir::failed(VPU::areDistributionAttrsCompatible(newInputDistType, newOutputDistType,
                                                         /*allowDifferentPerClusterMemoryView = */ false))) {
        nestedLogger.trace("ConvertDMA {0} will have incompatible input and output distributions after fused with copy",
                           convertOp->getLoc());
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(userCopyOp.getOperation());
    const auto newOutput = createNewViewLikeOps(rewriter, viewLikeOps, convertInput);
    if (mlir::failed(newOutput)) {
        return mlir::failure();
    }
    auto newConvertOp = VPUIP::ConvertDMAOp::create(rewriter, appendLoc(convertOp->getLoc(), "fused_dma"),
                                                    newOutput.value(), outputBuffer);
    rewriter.replaceOp(userCopyOp, newConvertOp.getOutput());

    for (auto viewLikeOp : viewLikeOps | reversed) {
        rewriter.eraseOp(viewLikeOp);
    }
    rewriter.eraseOp(convertOp);

    nestedLogger.trace("Successfully optimized ConvertDMA->ViewLike->Copy pattern");
    return mlir::success();
}

//
// CopyConvertDMA
//

class CopyConvertDMA : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyConvertDMA(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
        setDebugName("CopyConvertDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyConvertDMA::matchAndRewrite(VPUIP::CopyOp copy, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Copy at {1}", getDebugName(), copy->getLoc());
    auto nestedLogger = _log.nest();

    auto copyOp = getCopyOp(copy);
    if (copyOp == nullptr) {
        nestedLogger.trace("Couldn't find the copyOp");
        return mlir::failure();
    }

    // Copy op should have only one result
    if (copyOp->getResults().size() != 1) {
        nestedLogger.trace("Copy op should have only one result {0}", copyOp.getLoc());
        return mlir::failure();
    }

    if (!copyOp->hasOneUse()) {
        nestedLogger.trace("Copy op has multiple use {0}", copyOp.getLoc());
        return mlir::failure();
    }

    auto copyOutput = *copyOp->getResult(0).getUsers().begin();
    auto convertDMAOp = getConvertDMAOp(copyOutput);
    if (convertDMAOp == nullptr) {
        nestedLogger.trace("Result ConvertDMAOp not found {0}", copyOutput->getLoc());
        return mlir::failure();
    }

    auto newConvertDMAInput = copyOp->getOperand(0);
    auto outputBuff = convertDMAOp.getOutputs()[0];

    auto newConvertDMAInputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(newConvertDMAInput.getType());
    auto newConvertDMAOutputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputBuff.getType());
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType != nullptr &&
        mlir::failed(VPU::areDistributionAttrsCompatible(newConvertDMAInputDistType, newConvertDMAOutputDistType,
                                                         /*allowDifferentPerClusterMemoryView = */ false))) {
        nestedLogger.trace("ConvertDMA will have incompatible input and output distributions after fused with copy",
                           copyOp.getLoc());
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(convertDMAOp.getOperation());

    replaceOpWithNewConvertDMAOp(rewriter, newConvertDMAInput, outputBuff, convertDMAOp);

    if (copyOp->use_empty()) {
        rewriter.eraseOp(copyOp);
    }
    nestedLogger.trace("Successfully optimized Copy->ClusterConvertDMA pattern");
    return mlir::success();
}

//
// ConvertDMASubViewCopy
//

//
// Scenario 1: A whole-tensor ConvertDMA feeds SubView -> Copy branches through an optional view chain.
//                                                    /- SubView -> Copy
//   Input -> ConvertDMA -> [GenericReshape, PermuteCast]
//                                                    \- SubView -> Copy
// Scenario 2: The whole-tensor conversion is split into matching tiles before a ConcatView. Recover the common
// compact source and apply the same consumer-side rewrite as Scenario 1.
//   Common input
//        |                                               /- SubView -> Copy
//        +-> SubView -> ConvertDMA -\                    |
//        |                             ConcatView -> [GenericReshape, PermuteCast]
//        +-> SubView -> ConvertDMA -/                    |
//                                                        \- SubView -> Copy
//  =>
//                                                   /- SubView -> ConvertDMA
//   Common input -> [GenericReshape, PermuteCast]
//                                                   \- SubView -> ConvertDMA

class ConvertDMASubViewCopy : public mlir::OpRewritePattern<VPUIP::ConvertDMAOp> {
public:
    ConvertDMASubViewCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConvertDMAOp>(ctx), _log(log) {
        setDebugName("ConvertDMASubViewCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ConvertDMAOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertDMASubViewCopy::matchAndRewrite(VPUIP::ConvertDMAOp convertOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] ConvertDMA at {1}", getDebugName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    if (convertOp->use_empty()) {
        return mlir::failure();
    }

    if (vpux::VPUIP::hasDistributedOperand(convertOp)) {
        return mlir::failure();
    }

    auto convertInput = convertOp.getInput();
    auto viewChainInputOp = convertOp.getOperation();
    auto convertLoc = convertOp->getLoc();
    VPUIP::ConcatViewOp concatViewOp = nullptr;
    SmallVector<VPUIP::ConvertDMAOp> partitionConvertOps;

    // The whole-tensor conversion may have been split into matching input/output tiles before a ConcatView. Recover
    // the common compact source so the original rewrite below can handle both forms atomically.
    if (convertOp->hasOneUse()) {
        concatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*convertOp->getUsers().begin());
    }
    if (concatViewOp != nullptr) {
        const auto hasSingleSupportedViewUser =
                concatViewOp->hasOneUse() &&
                mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp>(*concatViewOp->getUsers().begin());
        if (!llvm::is_contained(concatViewOp.getInputs(), convertOp.getOutput()) || !hasSingleSupportedViewUser ||
            concatViewOp.getInputs().size() < 2) {
            return mlir::failure();
        }

        const auto concatOutputBuffer = concatViewOp.getOutputBuff();
        const auto concatOutputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(concatOutputBuffer));
        if (concatOutputType.getMemoryKind() != VPU::MemoryKind::DDR) {
            return mlir::failure();
        }

        mlir::Value commonSource;
        partitionConvertOps.reserve(concatViewOp.getInputs().size());
        for (const auto input : concatViewOp.getInputs()) {
            auto partitionConvertOp = input.getDefiningOp<VPUIP::ConvertDMAOp>();
            if (partitionConvertOp == nullptr || !partitionConvertOp->hasOneUse()) {
                return mlir::failure();
            }

            auto partitionInput = partitionConvertOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
            auto partitionOutput = partitionConvertOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
            if (partitionInput == nullptr || partitionOutput == nullptr || !partitionOutput->hasOneUse() ||
                partitionOutput.getSource() != concatOutputBuffer ||
                partitionInput.getStaticOffsetsAttr() != partitionOutput.getStaticOffsetsAttr() ||
                partitionInput.getStaticSizesAttr() != partitionOutput.getStaticSizesAttr() ||
                partitionInput.getStaticStridesAttr() != partitionOutput.getStaticStridesAttr()) {
                return mlir::failure();
            }

            if (commonSource == nullptr) {
                commonSource = partitionInput.getSource();
            } else if (commonSource != partitionInput.getSource()) {
                return mlir::failure();
            }
            partitionConvertOps.push_back(partitionConvertOp);
        }

        const auto commonSourceType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(commonSource));
        if (commonSourceType.getMemoryKind() != VPU::MemoryKind::DDR ||
            commonSourceType.getShape() != concatOutputType.getShape() ||
            commonSourceType.getDimsOrder() != concatOutputType.getDimsOrder() || !isCompactStride(commonSourceType)) {
            return mlir::failure();
        }

        convertInput = commonSource;
        viewChainInputOp = concatViewOp.getOperation();
        convertLoc = appendLoc(concatViewOp->getLoc(), "fused_convert_dma");
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(convertInput.getType());
    auto lastViewOp = viewChainInputOp;

    SmallVector<mlir::Operation*> viewLikeOps;
    if (lastViewOp->hasOneUse()) {
        auto userOp = *lastViewOp->getUsers().begin();
        while (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp>(userOp)) {
            viewLikeOps.push_back(userOp);
            if (!userOp->hasOneUse()) {
                break;
            }
            userOp = *userOp->getUsers().begin();
        }
    }

    if (!viewLikeOps.empty()) {
        if (!isCompactStride(inputType)) {
            nestedLogger.trace("Convert input has strides at {0}", convertOp->getLoc());
            return mlir::failure();
        }

        nestedLogger.trace("ViewLike ops found at {0}", convertOp->getLoc());
        lastViewOp = viewLikeOps.back();
    }

    mlir::Type newSubViewInputType = inputType;
    for (auto* viewLikeOp : viewLikeOps) {
        const auto newOutputType = VPUIP::BackInferUtils::inferOutputType(viewLikeOp, newSubViewInputType);
        if (!newOutputType.has_value()) {
            return mlir::failure();
        }
        newSubViewInputType = newOutputType.value();
    }

    const auto arch = config::getArch(convertOp.getOperation());
    SmallVector<std::pair<VPUIP::SubViewOp, VPUIP::CopyOp>> subViewCopyPair;
    for (auto user : lastViewOp->getUsers()) {
        auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user);
        if (subViewOp == nullptr || !subViewOp->hasOneUse()) {
            return mlir::failure();
        }

        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subViewOp->getUsers().begin());
        if (copyOp == nullptr) {
            return mlir::failure();
        }

        if (!copyOp->hasOneUse()) {
            const auto newSubViewType = VPUIP::BackInferUtils::inferOutputType(subViewOp, newSubViewInputType);
            const auto newSubViewNDType = newSubViewType.has_value()
                                                  ? mlir::dyn_cast<vpux::NDTypeInterface>(newSubViewType.value())
                                                  : nullptr;
            if (newSubViewNDType == nullptr) {
                return mlir::failure();
            }
            if (newSubViewNDType.getMemoryKind() == VPU::MemoryKind::DDR &&
                VPUIP::isEffectivelyStrided(newSubViewNDType) && !VPUIP::isDDRCopyEfficient(newSubViewNDType, arch)) {
                nestedLogger.trace("Multi-use Copy at {0} would create an inefficient strided DDR ConvertDMA source",
                                   copyOp->getLoc());
                return mlir::failure();
            }
        }
        subViewCopyPair.push_back({subViewOp, copyOp});
    }

    if (subViewCopyPair.empty()) {
        return mlir::failure();
    }

    nestedLogger.trace("Start to rewrite at {0}", convertOp->getLoc());
    const auto newOutput = createNewViewLikeOps(rewriter, viewLikeOps, convertInput);
    if (mlir::failed(newOutput)) {
        return mlir::failure();
    }

    for (auto& [subViewOp, copyOp] : subViewCopyPair) {
        rewriter.setInsertionPointAfter(copyOp.getOperation());
        auto newSubViewOp = VPUIP::SubViewOp::create(rewriter, appendLoc(subViewOp->getLoc(), "convert"),
                                                     newOutput.value(), subViewOp.getStaticOffsetsAttr(),
                                                     subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        auto newConvertOp = VPUIP::ConvertDMAOp::create(rewriter, appendLoc(convertLoc, "fused_dma"),
                                                        newSubViewOp.getResult(), copyOp.getOutputBuff());
        rewriter.replaceOp(copyOp, newConvertOp.getOutput());
        if (subViewOp->use_empty()) {
            rewriter.eraseOp(subViewOp);
        }
    }

    for (auto viewLikeOp : viewLikeOps | reversed) {
        if (viewLikeOp->use_empty()) {
            rewriter.eraseOp(viewLikeOp);
        }
    }

    if (concatViewOp == nullptr) {
        if (convertOp->use_empty()) {
            rewriter.eraseOp(convertOp);
        }
    } else {
        rewriter.eraseOp(concatViewOp);
        for (auto partitionConvertOp : partitionConvertOps) {
            auto partitionInput = partitionConvertOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
            auto partitionOutput = partitionConvertOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
            rewriter.eraseOp(partitionConvertOp);
            if (partitionInput->use_empty()) {
                rewriter.eraseOp(partitionInput);
            }
            rewriter.eraseOp(partitionOutput);
        }
    }

    nestedLogger.trace("Successfully optimized ConvertDMA->ViewLike=>Subview->Copy pattern");

    return mlir::success();
}

//
// OptimizeConvertDMAPass
//

class OptimizeConvertDMAPass final : public VPUIP::impl::OptimizeConvertDMAOpBase<OptimizeConvertDMAPass> {
public:
    explicit OptimizeConvertDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeConvertDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertDMAViewLikeCopy>(&ctx, _log);
    patterns.add<CopyConvertDMA>(&ctx, _log);
    patterns.add<ConvertDMASubViewCopy>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConvertDMAOpPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeConvertDMAOpPass(Logger log) {
    return std::make_unique<OptimizeConvertDMAPass>(log);
}
