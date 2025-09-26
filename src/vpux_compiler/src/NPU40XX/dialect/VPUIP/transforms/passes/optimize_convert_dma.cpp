//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_OPTIMIZECONVERTDMAOP
#define GEN_PASS_DEF_OPTIMIZECONVERTDMAOP
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

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

mlir::Value createNewViewLikeOps(mlir::PatternRewriter& rewriter, ArrayRef<mlir::Operation*> viewLikeOps,
                                 mlir::Value newInput) {
    const auto changeStrides = [](vpux::NDTypeInterface origTypeIf, vpux::NDTypeInterface newTypeIf) {
        const auto origStrides = origTypeIf.getStrides();
        const auto origElemSize = origTypeIf.getElemTypeSize().count();
        const auto newElemSize = newTypeIf.getElemTypeSize().count();
        auto outStrides = SmallVector<Bit>(origStrides.size(), Bit(0));

        std::transform(origStrides.begin(), origStrides.end(), outStrides.begin(), [&](Bit stride) {
            return stride / origElemSize * newElemSize;
        });
        return Strides(outStrides);
    };

    for (auto viewLikeOp : viewLikeOps) {
        mlir::IRMapping mapper;
        mapper.map(viewLikeOp->getOperands(), ArrayRef({newInput}));
        auto* newViewLikeOp = rewriter.clone(*viewLikeOp, mapper);

        const auto viewLikeOutType =
                mlir::cast<vpux::NDTypeInterface>(VPUIP::getCompactBufferType(viewLikeOp->getResult(0).getType()));
        const auto viewLikeInType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(newInput));

        const auto newStrides = changeStrides(viewLikeOutType, viewLikeInType);
        const auto newViewLikeOutType = viewLikeOutType.changeElemType(viewLikeInType.getElementType())
                                                .changeMemSpace(viewLikeInType.getMemSpace())
                                                .changeStrides(newStrides);
        newViewLikeOp->getResult(0).setType(newViewLikeOutType);
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
    auto newOutput = createNewViewLikeOps(rewriter, viewLikeOps, convertInput);
    auto newConvertOp =
            rewriter.create<VPUIP::ConvertDMAOp>(appendLoc(convertOp->getLoc(), "_fused_dma"), newOutput, outputBuffer);
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
    _log.trace("[{0}] Copy at {0}", getDebugName(), copy->getLoc());
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
// Perform the transformation below.
//
//                                           /- SubView->Copy
//   ConvertDMA->[GenericReshape,PermuteCast]
//                                           \- SubView->Copy
//
//  =>
//
//                               /- SubView->ConvertDMA
//   [GenericReshape,PermuteCast]
//                               \- SubView->ConvertDMA
//

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
    _log.trace("[{0}] Copy at {1}", getDebugName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    if (convertOp->use_empty()) {
        return mlir::failure();
    }

    if (vpux::VPUIP::hasDistributedOperand(convertOp)) {
        return mlir::failure();
    }

    const auto convertInput = convertOp.getInput();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(convertInput.getType());
    if (!isCompactStride(inputType)) {
        nestedLogger.trace("Convert input has strides at {0}", convertOp->getLoc());
        return mlir::failure();
    }

    auto lastViewOp = convertOp.getOperation();
    SmallVector<mlir::Operation*> viewLikeOps;
    if (convertOp->hasOneUse()) {
        auto userOp = *convertOp->getUsers().begin();
        while (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp>(userOp)) {
            viewLikeOps.push_back(userOp);
            if (!userOp->hasOneUse()) {
                break;
            }
            userOp = *userOp->getUsers().begin();
        }
    }

    if (!viewLikeOps.empty()) {
        nestedLogger.trace("ViewLike ops found at {0}", convertOp->getLoc());
        lastViewOp = viewLikeOps.back();
    }

    SmallVector<std::pair<VPUIP::SubViewOp, VPUIP::CopyOp>> subViewCopyPair;
    for (auto user : lastViewOp->getUsers()) {
        auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user);
        if (subViewOp == nullptr || !subViewOp->hasOneUse()) {
            return mlir::failure();
        }

        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subViewOp->getUsers().begin());
        if (copyOp == nullptr || !copyOp->hasOneUse()) {
            return mlir::failure();
        }
        subViewCopyPair.push_back({subViewOp, copyOp});
    }

    nestedLogger.trace("Start to rewrite at {0}", convertOp->getLoc());
    auto newOutput = createNewViewLikeOps(rewriter, viewLikeOps, convertInput);

    for (auto& [subViewOp, copyOp] : subViewCopyPair) {
        rewriter.setInsertionPointAfter(copyOp.getOperation());
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(appendLoc(subViewOp->getLoc(), "_convert"), newOutput,
                                                              subViewOp.getStaticOffsets(), subViewOp.getStaticSizes());
        auto newConvertOp = rewriter.create<VPUIP::ConvertDMAOp>(appendLoc(convertOp->getLoc(), "_fused_dma"),
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

    if (convertOp->use_empty()) {
        rewriter.eraseOp(convertOp);
    }

    nestedLogger.trace("Successfully optimized ConvertDMA->ViewLike=>Subview->Copy pattern");

    return mlir::success();
}

//
// OptimizeConvertDMAPass
//

class OptimizeConvertDMAPass final : public VPUIP::arch40xx::impl::OptimizeConvertDMAOpBase<OptimizeConvertDMAPass> {
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
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConvertDMAOpPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createOptimizeConvertDMAOpPass(Logger log) {
    return std::make_unique<OptimizeConvertDMAPass>(log);
}
