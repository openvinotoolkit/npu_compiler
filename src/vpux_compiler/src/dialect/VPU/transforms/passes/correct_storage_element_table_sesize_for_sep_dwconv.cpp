//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/utils/core/error.hpp>
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CORRECTSTORAGEELEMENTTABLESESIZEFORSEPDWCONV
#define GEN_PASS_DEF_CORRECTSTORAGEELEMENTTABLESESIZEFORSEPDWCONV
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class SeSizeRewriter final : public mlir::OpRewritePattern<VPU::StorageElementTableOp> {
public:
    SeSizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::StorageElementTableOp>(ctx), _log(log) {
        setDebugName("SeSizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::StorageElementTableOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    bool updateSETableForUser(VPU::StorageElementTableOp origOp, mlir::Operation* user,
                              mlir::PatternRewriter& rewriter) const;
};

bool SeSizeRewriter::updateSETableForUser(VPU::StorageElementTableOp origOp, mlir::Operation* user,
                                          mlir::PatternRewriter& rewriter) const {
    const auto& nestedLog = _log.nest();
    auto groupSparseTensor = mlir::dyn_cast<VPU::GroupSparseTensorOp>(user);
    if (groupSparseTensor == nullptr || !groupSparseTensor->hasOneUse()) {
        nestedLog.trace("User GroupSparseTensorOp does not exist or has multiple uses.");
        return false;
    }

    auto dwConv = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(*groupSparseTensor->user_begin());
    if (dwConv == nullptr || dwConv.getInput().getDefiningOp() != groupSparseTensor) {
        nestedLog.trace("User GroupSparseTensorOp is not a DWConv input.");
        return false;
    }
    auto inputShape = getShape(dwConv.getInput());

    auto origSeSizeAttr = origOp.getSeSize();
    const auto origSeSizes = parseIntArrayAttr<int64_t>(origSeSizeAttr);

    // See @DWConvWithSEPChannelSliceWithDepth1 for example and explanation of scenario.
    if (origSeSizes.size() == 1 && origSeSizes.front() == inputShape[Dims4D::Act::C]) {
        nestedLog.trace("SeSize is equal to input channels size.");
        return false;
    }

    auto strategyAttr = dwConv.getMultiClusterStrategyAttr();
    const auto [seDepth, seSizeAttr] = [&]() -> std::pair<int64_t, mlir::ArrayAttr> {
        const auto seSize = inputShape[Dims4D::Act::C];
        if (strategyAttr == nullptr) {
            return std::make_pair(1, getIntArrayAttr(origOp.getContext(), SmallVector<int64_t>{seSize}));
        }

        const auto strategy = strategyAttr.getValue();
        const auto numClusters = VPU::getOptimalNumClusters(dwConv, getShape(dwConv.getOutput()), strategy);
        auto distributedInputType = mlir::cast<VPU::DistributedTensorType>(
                VPU::getDistributedActivationTypeFromOp(dwConv, dwConv.getInput(), dwConv.getInput().getType(),
                                                        numClusters, strategy)
                        .getDistributedTypes()
                        .front());

        if (VPU::isSegmentedOverC(distributedInputType.getDistribution())) {
            SmallVector<int64_t> seSizes;
            for (auto shape : distributedInputType.getPerClusterMemoryShapes()) {
                seSizes.push_back(shape[Dims4D::Act::C]);
            }
            return std::make_pair(static_cast<int64_t>(seSizes.size()), getIntArrayAttr(origOp.getContext(), seSizes));
        }

        return std::make_pair(1, getIntArrayAttr(origOp.getContext(), SmallVector<int64_t>{seSize}));
    }();

    if (seDepth == static_cast<int64_t>(origSeSizes.size()) && seSizeAttr == origSeSizeAttr) {
        nestedLog.trace("SeSize is correct.");
        return false;
    }

    nestedLog.trace("Updating seSize for {0}", origOp->getLoc());
    auto dataShapeAttr = getIntArrayAttr(origOp.getContext(), getShape(groupSparseTensor.getData()).raw());
    auto newStorageElementTableOp = rewriter.create<VPU::StorageElementTableOp>(
            origOp->getLoc(), dataShapeAttr, origOp.getDataElemTypeAttr(), seSizeAttr,
            getIntAttr(origOp.getContext(), seDepth), groupSparseTensor.getSeAttr().value(), nullptr, nullptr);

    rewriter.replaceUsesWithIf(origOp.getOutput(), newStorageElementTableOp, [&](mlir::OpOperand& operand) {
        return operand.getOwner() == groupSparseTensor;
    });

    vpux::inferReturnTypes(groupSparseTensor, vpux::InferShapedTypeMode::ALL);

    nestedLog.trace("Updated seSize {0} and seDepth {1}", seSizeAttr, seDepth);
    return true;
}

mlir::LogicalResult SeSizeRewriter::matchAndRewrite(VPU::StorageElementTableOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    if (!origOp.getSeAttr().has_value()) {
        return mlir::failure();
    }

    bool wereAnyUsersUpdated = false;
    _log.debug("Found StorageElementTableOp '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    for (auto user : origOp->getUsers()) {
        _log.trace("Analyze user of type {0} at loc {1}", user->getName(), user->getLoc());
        wereAnyUsersUpdated = wereAnyUsersUpdated || updateSETableForUser(origOp, user, rewriter);
    }

    if (!wereAnyUsersUpdated) {
        return matchFailed(_log, rewriter, origOp, "No users of StorageElementTable were updated.");
    }

    return mlir::success();
}

class CorrectStorageElementTableSeSizeForSEPDWConvPass final :
        public VPU::impl::CorrectStorageElementTableSeSizeForSEPDWConvBase<
                CorrectStorageElementTableSeSizeForSEPDWConvPass> {
public:
    explicit CorrectStorageElementTableSeSizeForSEPDWConvPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CorrectStorageElementTableSeSizeForSEPDWConvPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<SeSizeRewriter>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(greedyPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createCorrectStorageElementTableSeSizeForSEPDWConvPass(Logger log) {
    return std::make_unique<CorrectStorageElementTableSeSizeForSEPDWConvPass>(std::move(log));
}
