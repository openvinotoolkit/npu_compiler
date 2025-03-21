//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_OPTIMIZECONVERTDMAOP
#define GEN_PASS_DEF_OPTIMIZECONVERTDMAOP
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

using CreateAndReplaceWithConvertDMAFunctType =
        FuncRef<void(mlir::PatternRewriter&, mlir::Value, mlir::Value, mlir::Operation*)>;
using GetCopyFunctType = FuncRef<VPUIP::LayerOpInterface(mlir::Operation*)>;
using GetConvertDMAFunctType = FuncRef<VPUIP::LayerOpInterface(mlir::Operation*)>;

VPUIP::LayerOpInterface getConvertDMAOp(mlir::Operation* maybeConvertDMAOperation) {
    if (auto convertDMAOp = mlir::dyn_cast_or_null<VPUIP::ConvertDMAOp>(maybeConvertDMAOperation)) {
        return mlir::cast<VPUIP::LayerOpInterface>(*convertDMAOp);
    }
    return nullptr;
}

VPUIP::LayerOpInterface getClusterConvertDMAOp(mlir::Operation* maybeConvertDMAOperation) {
    if (auto clusterConvertDMAOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(maybeConvertDMAOperation)) {
        if (clusterConvertDMAOp.getInnerTaskOpOfType<VPUIP::ConvertDMAOp>() != nullptr) {
            return mlir::cast<VPUIP::LayerOpInterface>(*clusterConvertDMAOp);
        }
    }
    return nullptr;
}

VPUIP::LayerOpInterface getAnyConvertDMA(mlir::Operation* maybeConvertDMAOperation) {
    if (auto convertDMAOp = getConvertDMAOp(maybeConvertDMAOperation)) {
        return convertDMAOp;
    }

    if (auto convertDMAOp = getClusterConvertDMAOp(maybeConvertDMAOperation)) {
        return convertDMAOp;
    }

    return nullptr;
}

VPUIP::LayerOpInterface getCopyOp(mlir::Operation* sourceOp) {
    return mlir::dyn_cast_or_null<VPUIP::CopyOp>(sourceOp);
}

VPUIP::LayerOpInterface getClusterCopyOp(mlir::Operation* sourceOp) {
    return sourceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
}

void replaceOpWithNewConvertDMAOp(mlir::PatternRewriter& rewriter, mlir::Value input, mlir::Value outputBuff,
                                  mlir::Operation* opToReplace) {
    rewriter.replaceOpWithNewOp<VPUIP::ConvertDMAOp>(opToReplace, input, outputBuff);
}

void replaceOpWithNewClusterConvertDMAOp(mlir::PatternRewriter& rewriter, mlir::Value input, mlir::Value outputBuff,
                                         mlir::Operation* opToReplace) {
    const auto convertOpBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::ConvertDMAOp>(loc, newOperands[0], newOperands[1]);
    };
    SmallVector<mlir::Value> inputsOutputOperands = {input, outputBuff};
    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(opToReplace, outputBuff.getType(), inputsOutputOperands,
                                                           convertOpBodyBuilder);
}

class ConvertDMACopyRewriterBase : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    ConvertDMACopyRewriterBase(mlir::MLIRContext* ctx,
                               CreateAndReplaceWithConvertDMAFunctType createAndReplaceWithNewConvertDMAOp,
                               GetCopyFunctType getCopyOp, GetConvertDMAFunctType getConvertDMAOp, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx),
              _createAndReplaceWithNewConvertDMAOp(createAndReplaceWithNewConvertDMAOp),
              _getCopyOp(getCopyOp),
              _getConvertDMAOp(getConvertDMAOp),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    CreateAndReplaceWithConvertDMAFunctType _createAndReplaceWithNewConvertDMAOp;
    GetCopyFunctType _getCopyOp;
    GetConvertDMAFunctType _getConvertDMAOp;
    Logger _log;
};

mlir::LogicalResult ConvertDMACopyRewriterBase::matchAndRewrite(VPUIP::CopyOp copy,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("ConvertDMACopyRewriterBase: Copy at {0}", copy->getLoc());
    auto nestedLogger = _log.nest();

    auto copyOp = _getCopyOp(copy);
    if (copyOp == nullptr) {
        nestedLogger.trace("Couldn't find the copyOp");
        return mlir::failure();
    }

    auto copyInput = copyOp->getOperand(0);
    auto convertDMAOp = _getConvertDMAOp(copyInput.getDefiningOp());
    if (convertDMAOp == nullptr) {
        nestedLogger.trace("Input ConvertDMAOp not found {0}", copyInput.getLoc());
        return mlir::failure();
    }

    if (!convertDMAOp->hasOneUse()) {
        nestedLogger.trace("ConvertDMA has multiple use {0}", copyOp.getLoc());
        return mlir::failure();
    }

    auto newConvertDMAInput = convertDMAOp->getOperand(0);
    auto parentCopy = copyOp.getOperation();
    auto outputBuff = copyOp.getOutputs()[0];

    auto newConvertDMAInputDistType = newConvertDMAInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto newConvertDMAOutputDistType = outputBuff.getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType != nullptr &&
        mlir::failed(VPU::areDistributionAttrsCompatible(newConvertDMAInputDistType, newConvertDMAOutputDistType,
                                                         /*allowDifferentPerClusterMemoryView = */ false))) {
        nestedLogger.trace("ConvertDMA will have incompatible input and output distributions after fused with copy",
                           copyOp.getLoc());
        return mlir::failure();
    }

    // Temporarily disable fuse of ClusterConvertDMA(from SEGMENDTED) and Copy(toDDR) due to wrong DMA descriptors
    // generated for this case
    // Tracked in: E#101270
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType == nullptr) {
        const auto inDistMode = newConvertDMAInputDistType.getDistribution().getMode().getValue();
        const auto outMemKind = outputBuff.getType().cast<NDTypeInterface>().getMemoryKind();
        if (inDistMode == VPU::DistributionMode::SEGMENTED && outMemKind == VPU::MemoryKind::DDR) {
            return mlir::failure();
        }
    }

    rewriter.setInsertionPointAfter(parentCopy);

    _createAndReplaceWithNewConvertDMAOp(rewriter, newConvertDMAInput, outputBuff, parentCopy);

    if (convertDMAOp->use_empty()) {
        rewriter.eraseOp(convertDMAOp);
    }
    nestedLogger.trace("Successfully optimized ConvertDMA->Copy pattern");
    return mlir::success();
}

class CopyConvertDMARewriterBase : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyConvertDMARewriterBase(mlir::MLIRContext* ctx,
                               CreateAndReplaceWithConvertDMAFunctType createAndReplaceWithNewConvertDMAOp,
                               GetCopyFunctType getCopyOp, GetConvertDMAFunctType getConvertDMAOp, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx),
              _createAndReplaceWithNewConvertDMAOp(createAndReplaceWithNewConvertDMAOp),
              _getCopyOp(getCopyOp),
              _getConvertDMAOp(getConvertDMAOp),
              _log(log) {
        setDebugName("CopyConvertDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    CreateAndReplaceWithConvertDMAFunctType _createAndReplaceWithNewConvertDMAOp;
    GetCopyFunctType _getCopyOp;
    GetConvertDMAFunctType _getConvertDMAOp;
    Logger _log;
};

mlir::LogicalResult CopyConvertDMARewriterBase::matchAndRewrite(VPUIP::CopyOp copy,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Copy at {0}", getDebugName(), copy->getLoc());
    auto nestedLogger = _log.nest();

    auto copyOp = _getCopyOp(copy);
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
    auto convertDMAOp = _getConvertDMAOp(copyOutput);
    if (convertDMAOp == nullptr) {
        nestedLogger.trace("Result ConvertDMAOp not found {0}", copyOutput->getLoc());
        return mlir::failure();
    }

    auto newConvertDMAInput = copyOp->getOperand(0);
    auto outputBuff = convertDMAOp.getOutputs()[0];

    auto newConvertDMAInputDistType = newConvertDMAInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto newConvertDMAOutputDistType = outputBuff.getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType != nullptr &&
        mlir::failed(VPU::areDistributionAttrsCompatible(newConvertDMAInputDistType, newConvertDMAOutputDistType,
                                                         /*allowDifferentPerClusterMemoryView = */ false))) {
        nestedLogger.trace("ConvertDMA will have incompatible input and output distributions after fused with copy",
                           copyOp.getLoc());
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(convertDMAOp.getOperation());

    _createAndReplaceWithNewConvertDMAOp(rewriter, newConvertDMAInput, outputBuff, convertDMAOp);

    if (copyOp->use_empty()) {
        rewriter.eraseOp(copyOp);
    }
    nestedLogger.trace("Successfully optimized Copy->ClusterConvertDMA pattern");
    return mlir::success();
}

//
// ConvertDMAOp                  |
//     |             =>      ConvertDMAOp
//   CopyOp                      |
//

class ConvertDMACopy final : public ConvertDMACopyRewriterBase {
public:
    ConvertDMACopy(mlir::MLIRContext* ctx, Logger log)
            : ConvertDMACopyRewriterBase(ctx, replaceOpWithNewConvertDMAOp, getCopyOp, getConvertDMAOp, log) {
    }
};

//
// Cluster/ConvertDMAOp             |
//     |             =>      ClusterConvertDMAOp
// ClusterCopyOp                    |
//

class ConvertDMAClusterCopy final : public ConvertDMACopyRewriterBase {
public:
    ConvertDMAClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : ConvertDMACopyRewriterBase(ctx, replaceOpWithNewClusterConvertDMAOp, getClusterCopyOp, getAnyConvertDMA,
                                         log) {
    }
};

//
// ClusterConvertDMAOp               |
//     |             =>      ClusterConvertDMAOp
//   CopyOp                          |
//

class ClusterConvertDMACopy final : public ConvertDMACopyRewriterBase {
public:
    ClusterConvertDMACopy(mlir::MLIRContext* ctx, Logger log)
            : ConvertDMACopyRewriterBase(ctx, replaceOpWithNewClusterConvertDMAOp, getCopyOp, getClusterConvertDMAOp,
                                         log) {
    }
};

//
// ClusterCopyOp                          |
//     |                =>      ClusterConvertDMAOp
// ClusterConvertDMAOp                    |
//

class ClusterCopyClusterConvertDMA final : public CopyConvertDMARewriterBase {
public:
    ClusterCopyClusterConvertDMA(mlir::MLIRContext* ctx, Logger log)
            : CopyConvertDMARewriterBase(ctx, replaceOpWithNewClusterConvertDMAOp, getClusterCopyOp,
                                         getClusterConvertDMAOp, log) {
    }
};

//
// ClusterCopyOp                          |
//     |                =>      ClusterConvertDMAOp
// ConvertDMAOp                           |
//

class ClusterCopyConvertDMA final : public CopyConvertDMARewriterBase {
public:
    ClusterCopyConvertDMA(mlir::MLIRContext* ctx, Logger log)
            : CopyConvertDMARewriterBase(ctx, replaceOpWithNewConvertDMAOp, getClusterCopyOp, getConvertDMAOp, log) {
    }
};

//
//  CopyOp                           |
//     |                =>      ConvertDMAOp
// ConvertDMAOp                      |
//

class CopyConvertDMA final : public CopyConvertDMARewriterBase {
public:
    CopyConvertDMA(mlir::MLIRContext* ctx, Logger log)
            : CopyConvertDMARewriterBase(ctx, replaceOpWithNewConvertDMAOp, getCopyOp, getConvertDMAOp, log) {
    }
};

//
//   CopyOp                               |
//     |                =>      ClusterConvertDMAOp
// ClusterConvertDMAOp                    |
//

class CopyClusterConvertDMA final : public CopyConvertDMARewriterBase {
public:
    CopyClusterConvertDMA(mlir::MLIRContext* ctx, Logger log)
            : CopyConvertDMARewriterBase(ctx, replaceOpWithNewClusterConvertDMAOp, getCopyOp, getClusterConvertDMAOp,
                                         log) {
    }
};

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
    patterns.add<ConvertDMACopy>(&ctx, _log);
    patterns.add<ConvertDMAClusterCopy>(&ctx, _log);
    patterns.add<ClusterConvertDMACopy>(&ctx, _log);
    patterns.add<CopyConvertDMA>(&ctx, _log);
    patterns.add<CopyClusterConvertDMA>(&ctx, _log);
    patterns.add<ClusterCopyClusterConvertDMA>(&ctx, _log);
    patterns.add<ClusterCopyConvertDMA>(&ctx, _log);

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
