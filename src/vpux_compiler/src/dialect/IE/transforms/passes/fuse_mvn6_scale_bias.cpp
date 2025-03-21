//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEMVN6SCALEBIAS
#define GEN_PASS_DEF_FUSEMVN6SCALEBIAS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseMvn6ScaleBias
//
//   [MVN6->Multiply->Add]
//   [MVN6->Multiply]
//   [MVN6->Add]
//

class FuseMvn6ScaleBias final : public mlir::OpRewritePattern<IE::MVN6Op> {
public:
    FuseMvn6ScaleBias(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVN6Op>(ctx), _log(log) {
        setDebugName("FuseMvn6ScaleBias");
    }

    mlir::LogicalResult matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    bool fitsIntoCmx(IE::MVN6Op origOp, mlir::Value mulInput, mlir::Value addInput) const {
        const auto mvnType = origOp.getInput().getType().cast<NDTypeInterface>();
        const auto actInput = mulInput ? mulInput : addInput;
        const auto actType = actInput.getType().cast<NDTypeInterface>();
        const auto axesVec = parseIntArrayAttr<int64_t>(origOp.getAxesValueAttr());

        int64_t normSize = 1;
        int64_t actSize = 1;  // for same norm axes
        for (auto axis : axesVec) {
            normSize *= mvnType.getShape()[Dim(axis)];
            actSize *= actType.getShape()[Dim(axis)];
        }
        const auto bpp = mvnType.getElemTypeSize().count() / CHAR_BIT;
        normSize *= bpp;
        actSize *= bpp;
        SmallVector<Byte> buffSizes = {Byte(normSize) /*in*/, Byte(normSize) /*out*/};
        if (mulInput) {
            buffSizes.push_back(Byte(actSize));
        }
        if (addInput) {
            buffSizes.push_back(Byte(actSize));
        }

        const auto arch = VPU::getArch(origOp);
        auto totalAvailCMXSize = vpux::VPU::getTotalCMXSize(origOp).count();
        auto neededCMXSize = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffSizes).count();
        if (neededCMXSize >= totalAvailCMXSize) {
            _log.trace("Normalization space too large (not tileable) {0}", neededCMXSize);
            return false;
        }

        return true;
    }
};

mlir::LogicalResult FuseMvn6ScaleBias::matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getScale() || origOp.getBias()) {
        _log.nest().trace("MVN6 already has scale/bias.");
        return mlir::failure();
    }

    const auto nextOp = *(origOp.getOutput().getUsers().begin());
    if (!mlir::isa_and_nonnull<IE::MultiplyOp, IE::AddOp>(nextOp)) {
        _log.trace("No Multiply or Add found after MVN6 op.");
        return mlir::failure();
    }

    if (!origOp.getResult().hasOneUse()) {
        _log.trace("MVN6 op has multiple users.");
        return mlir::failure();
    }

    // Infrequent config (not implemented on Shave)
    if (!origOp.getNormalizeVarianceAttr().getValue()) {
        _log.trace("MVN6 normalize=false not implemented");
        return mlir::failure();
    }

    const auto rank = origOp.getOutput().getType().getRank();
    if (rank > 4) {
        _log.trace("Fuse supported for rank <=4, got {0}.", rank);
        return mlir::failure();
    }

    auto getFuseIdx = [](mlir::Operation* child, mlir::Operation* parent) -> int64_t {
        VPUX_THROW_UNLESS(child->getOperands().size() == 2, "Child op (Add/Mul) does not have 2 inputs");
        return child->getOperand(0).getDefiningOp() == parent ? 1 : 0;
    };

    mlir::Operation* nextMul = mlir::isa_and_nonnull<IE::MultiplyOp>(nextOp) ? nextOp : nullptr;
    mlir::Operation* nextAdd = mlir::isa_and_nonnull<IE::AddOp>(nextOp) ? nextOp : nullptr;

    if (nextMul && nextMul->getResult(0).hasOneUse()) {
        auto nextMulUser = *nextMul->getResult(0).getUsers().begin();
        nextAdd = mlir::isa_and_nonnull<IE::AddOp>(nextMulUser) ? nextMulUser : nullptr;
    }

    mlir::Value mulInput = nullptr;
    mlir::Value addInput = nullptr;
    mlir::Operation* lastOp = nullptr;

    if (nextMul && nextAdd) {  // [MVN->Mul->Add]
        const auto mulShape = nextMul->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
        const auto addShape = nextAdd->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
        if (mulShape != addShape) {
            _log.trace("Mul/Add have different shapes.");
            return mlir::failure();
        }
        auto mulIdx = getFuseIdx(nextMul, origOp);
        auto addIdx = getFuseIdx(nextAdd, nextMul);
        mulInput = nextMul->getOperand(mulIdx);
        addInput = nextAdd->getOperand(addIdx);
        lastOp = nextAdd;
    } else if (nextMul && !nextAdd) {  // [MVN->Mul]
        auto mulIdx = getFuseIdx(nextMul, origOp);
        mulInput = nextMul->getOperand(mulIdx);
        lastOp = nextMul;
    } else if (!nextMul && nextAdd) {  // [MVN->Add]
        auto addIdx = getFuseIdx(nextAdd, origOp);
        addInput = nextAdd->getOperand(addIdx);
        lastOp = nextAdd;
    } else {
        return mlir::failure();
    }

    if (!fitsIntoCmx(origOp, mulInput, addInput)) {
        // For large instances, rely on [MVN6 -> MVN1 -> DecomposeMVNPass]
        return mlir::failure();
    }

    auto newMvn = rewriter.create<IE::MVN6Op>(origOp.getLoc(), origOp.getInput(), mulInput, addInput, nullptr,
                                              origOp.getAxesValueAttr(), origOp.getNormalizeVarianceAttr(),
                                              origOp.getEpsAttr(), origOp.getEpsModeAttr());
    lastOp->replaceAllUsesWith(newMvn);

    return mlir::success();
}

//
// FuseMvn6ScaleBiasPass
//

class FuseMvn6ScaleBiasPass final : public IE::impl::FuseMvn6ScaleBiasBase<FuseMvn6ScaleBiasPass> {
public:
    explicit FuseMvn6ScaleBiasPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseMvn6ScaleBiasPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseMvn6ScaleBias>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseMvn6ScaleBiasPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMvn6ScaleBiasPass(Logger log) {
    return std::make_unique<FuseMvn6ScaleBiasPass>(log);
}
