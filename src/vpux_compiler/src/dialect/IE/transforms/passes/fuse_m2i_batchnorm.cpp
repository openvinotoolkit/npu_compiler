//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>

namespace vpux::IE {
#define GEN_PASS_DECL_M2IBATCHNORMFUSION
#define GEN_PASS_DEF_M2IBATCHNORMFUSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// M2IBatchNormFusionRewriterBase
//

class M2IBatchNormFusionRewriterBase : public mlir::OpRewritePattern<IE::AddOp> {
public:
    M2IBatchNormFusionRewriterBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual bool checkRequiredPattern(mlir::Operation*, LogCb) const = 0;
    virtual mlir::Operation* getOpForPatternCheck(mlir::Operation*, mlir::Operation*) const = 0;

protected:
    Logger _log;
};
/*
    openVino operation for BatchNormInference (given x input and y output):
        y = f(x, gamma, beta, mean, variance, epsilon) =
        gamma * ((x - mean)/sqrt(var + epsilon) + beta =
        (gamma/sqrt(var + epsilon)) * x + (beta - mean*gamma/sqrt(var + epsilon)) =
        gamma_div_scale * x + (beta - mean * gamma_div_scale) =
        cst0 * x + cst1 =
        Add( Mult(x, cst0), cst1)
    where
        cst0 = gamma_div_scale = gamma/sqrt(var + epsilon)
        cst1 = beta - mean * gamma_div_scale = (beta - mean*gamma/sqrt(var + epsilon))

    M2I Norm operation is instead:
        y = f(x, A, B, C, D) =
        A * ((x - B) / C ) + D =
        A/C * x + (D - A*B/C)

    Then we have
    A = gamma
    C = sqrt(var + epsilon)
    B = mean
    D = beta

    since we only have 2 constant values (cst0/1) in Add/Mult operations the system of equations is indeterminate,
    which means that we can fix some of the values to find one solution.
    For simplicity we can fix epsilon := 0 var := 1 mean := 0

    Then in M2I terms it will become
    A = gamma = cst0
    B = mean = 0
    C = sqrt(var + epsilon) = 1
    D = Beta = cst1

*/
mlir::LogicalResult M2IBatchNormFusionRewriterBase::matchAndRewrite(IE::AddOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AddOp '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Match patterns
    auto* addParentOp0 = origOp.getInput1().getDefiningOp();
    auto* addParentOp1 = origOp.getInput2().getDefiningOp();
    if (addParentOp0 == nullptr || addParentOp1 == nullptr) {
        return mlir::failure();
    }

    mlir::Operation *multOp{}, *constOp1{};
    if (mlir::isa<Const::DeclareOp>(addParentOp0) && mlir::isa<IE::MultiplyOp>(addParentOp1)) {
        multOp = addParentOp1;
        constOp1 = addParentOp0;
    } else if (mlir::isa<IE::MultiplyOp>(addParentOp0) && mlir::isa<Const::DeclareOp>(addParentOp1)) {
        multOp = addParentOp0;
        constOp1 = addParentOp1;
    } else {
        _log.trace("[{0}] M2IBatchNormFusion pattern not matching", getDebugName());
        return mlir::failure();
    }

    auto* multParent0 = multOp->getOperand(0).getDefiningOp();
    auto* multParent1 = multOp->getOperand(1).getDefiningOp();
    if (mlir::isa_and_nonnull<Const::DeclareOp>(multParent0) and mlir::isa_and_nonnull<Const::DeclareOp>(multParent1)) {
        _log.trace("[{0}] Only one parent of MultiplyOp should be a Const::DeclareOp", getDebugName());
        return mlir::failure();
    }

    unsigned inputOpIdx{};
    // one multParent operation can be nullptr in case MultiplyOp is the first operation in the IR
    if (mlir::isa_and_nonnull<Const::DeclareOp>(multParent0)) {
        inputOpIdx = 1;
    } else if (mlir::isa_and_nonnull<Const::DeclareOp>(multParent1)) {
        inputOpIdx = 0;
    } else {
        _log.trace("[{0}] M2IBatchNormFusion pattern not matching", getDebugName());
        return mlir::failure();
    }

    auto* constOp0 = inputOpIdx ? multParent0 : multParent1;
    auto* inputOp = inputOpIdx ? multParent1 : multParent0;
    if (constOp0 == nullptr) {
        return mlir::failure();
    }
    auto multOpInputTensor = multOp->getOperand(inputOpIdx);

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    // Check that also the future conversion from IE::BatchNormInferenceOp to VPU::M2INormOp is supported,
    // otherwise no reason to fuse IE::MultiplyOp + IE::AddOp into IE::BatchNormInferenceOp
    if (!VPU::isM2IBatchNormSupported(multOpInputTensor, origOp.getOutput(), logCb)) {
        _log.trace("[{0}] Batchnorm operation doesn't support the specific ops configs", getDebugName());
        return mlir::failure();
    }

    auto* childOp = *origOp.getOutput().getUsers().begin();
    if (!checkRequiredPattern(getOpForPatternCheck(inputOp, childOp), logCb)) {
        _log.trace("[{0}] No matching pattern for Mult+Add fusion into BatchNormInferenceOp", getDebugName());
        return mlir::failure();
    }

    // Green light to do the pattern substitution IE::Mult + IE::Add -> IE::BatchNormInferenceOp
    auto const0DeclareOp = mlir::dyn_cast_or_null<Const::DeclareOp>(constOp0);
    auto const1DeclareOp = mlir::dyn_cast_or_null<Const::DeclareOp>(constOp1);
    if ((const0DeclareOp == nullptr) || (const1DeclareOp == nullptr)) {
        _log.trace("[{0}] DeclareOp not found", getDebugName());
        return mlir::failure();
    }

    const auto cst0Content = const0DeclareOp.getContent();
    const auto cst1Content = const1DeclareOp.getContent();
    auto ctx = getContext();
    const auto gammaAttr = /* cst0Attr = */ getFPArrayAttr(ctx, cst0Content.getValues<double>());
    const auto betaAttr = /* cst1Attr = */ getFPArrayAttr(ctx, cst1Content.getValues<double>());
    const auto numChannels = gammaAttr.size();
    // Verify number of attributes
    if (numChannels != betaAttr.size() || numChannels != 3) {
        _log.trace("[{0}] Add and Multiply constant inputs must have 3 sized attributes", getDebugName());
        return mlir::failure();
    }
    SmallVector<mlir::Attribute> mean(numChannels, getFPAttr(ctx, 0.));
    SmallVector<mlir::Attribute> var(numChannels, getFPAttr(ctx, 1.));
    const auto meanAttr = mlir::ArrayAttr::get(ctx, mean);
    const auto varAttr = mlir::ArrayAttr::get(ctx, var);
    const auto epsAttr = mlir::FloatAttr::get(mlir::Float64Type::get(ctx), 0.);

    auto batchNormOp = rewriter.create<IE::BatchNormInferenceOp>(origOp->getLoc(), origOp.getType(), multOpInputTensor,
                                                                 nullptr, nullptr, nullptr, nullptr, gammaAttr,
                                                                 betaAttr, meanAttr, varAttr, epsAttr);

    rewriter.replaceOp(origOp, batchNormOp.getOutput());

    return mlir::success();
}

//
// M2IBatchNormFusionParent
//
class M2IBatchNormFusionParent : public M2IBatchNormFusionRewriterBase {
public:
    M2IBatchNormFusionParent(mlir::MLIRContext* ctx, Logger log): M2IBatchNormFusionRewriterBase(ctx, log) {
    }

protected:
    mlir::Operation* getOpForPatternCheck(mlir::Operation* inputOp, mlir::Operation*) const override {
        return inputOp;
    }

    bool checkRequiredPattern(mlir::Operation* inputOp, LogCb logCb) const override {
        if (inputOp == nullptr) {
            return false;
        }
        // before MultiplyOp there could be a sequence of MemPermute/ConvertOp that are still compatible with the
        // pattern
        while (mlir::isa<IE::ConvertOp>(inputOp) || mlir::isa<IE::MemPermuteOp>(inputOp) ||
               mlir::isa<IE::TransposeOp>(inputOp)) {
            if (!inputOp->getResult(0).hasOneUse()) {
                return false;
            }
            // move inputOp upstream in the graph
            inputOp = inputOp->getOperand(0).getDefiningOp();
        }

        if (!inputOp->getResult(0).hasOneUse()) {
            return false;
        }

        // Verify that the inputOp belongs to one of the operations that can be mapped to m2i
        if (auto interpolateOp = mlir::dyn_cast_or_null<IE::InterpolateOp>(inputOp)) {
            // check if the IE::InterpolateOp to VPU::M2IResizeOp is supported (interpolation will be mapped to m2i)
            return VPU::isM2IResizeSupported<IE::InterpolateOp>(interpolateOp, logCb, true /*checkFp16Interleaved*/);
        } else if (auto cscOp = mlir::dyn_cast_or_null<IE::YuvToRgbOp>(inputOp)) {
            return VPU::M2IColorConvertOp::isSupported(cscOp, logCb, /*checkLayout=*/true,
                                                       /*checkChannelAlignment=*/true);
        }
        return false;
    }
};

//
// M2IBatchNormFusionChild
//
class M2IBatchNormFusionChild : public M2IBatchNormFusionRewriterBase {
public:
    M2IBatchNormFusionChild(mlir::MLIRContext* ctx, Logger log): M2IBatchNormFusionRewriterBase(ctx, log) {
    }

protected:
    mlir::Operation* getOpForPatternCheck(mlir::Operation*, mlir::Operation* childOp) const override {
        return childOp;
    }

    bool checkRequiredPattern(mlir::Operation* childOp, LogCb logCb) const override {
        // Check if the pattern is followed by an Interpolation (can't be CSC as it only converts from YUV to RGB)
        if (childOp == nullptr) {
            return false;
        }
        // following AddOp there could be a sequence of MemPermute/ConvertOp that are still compatible with the
        // pattern
        while (mlir::isa<IE::ConvertOp>(childOp) || mlir::isa<IE::MemPermuteOp>(childOp) ||
               mlir::isa<IE::TransposeOp>(childOp)) {
            if (!childOp->getResult(0).hasOneUse()) {
                return false;
            }
            // move op downstream in the graph
            childOp = *childOp->getResult(0).getUsers().begin();
        }

        // Verify that the childOp is an Interpolate, and it's mappable to m2i
        if (auto interpolateOp = mlir::dyn_cast_or_null<IE::InterpolateOp>(childOp)) {
            // check if the IE::InterpolateOp to VPU::M2IResizeOp is supported (interpolation will be mapped to m2i)
            return VPU::isM2IResizeSupported<IE::InterpolateOp>(interpolateOp, logCb, true /*checkFp16Interleaved*/);
        }
        return false;
    }
};

//
// M2IBatchNormFusionPass
//

class M2IBatchNormFusionPass final : public IE::impl::M2IBatchNormFusionBase<M2IBatchNormFusionPass> {
public:
    explicit M2IBatchNormFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void M2IBatchNormFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<M2IBatchNormFusionParent>(&ctx, _log);
    patterns.add<M2IBatchNormFusionChild>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createM2IBatchNormFusionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createM2IBatchNormFusionPass(Logger log) {
    return std::make_unique<M2IBatchNormFusionPass>(log);
}
