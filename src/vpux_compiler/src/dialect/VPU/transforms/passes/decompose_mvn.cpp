//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_DECOMPOSEMVN
#define GEN_PASS_DEF_DECOMPOSEMVN
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// This computes the buffers for the most granular tiling possible for the original MVN op.
// The decomposition happens if not even this tiling scheme fits in CMX.
SmallVector<vpux::NDTypeInterface> getTiledBuffers(vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                                   DimArr nonNormDims) {
    auto inputShape = to_small_vector(input.getShape());
    auto outputShape = to_small_vector(output.getShape());
    for (auto& dim : nonNormDims) {
        inputShape[dim.ind()] = 1;
        outputShape[dim.ind()] = 1;
    }
    return SmallVector<vpux::NDTypeInterface>{input.changeShape(ShapeRef(inputShape)),
                                              output.changeShape(ShapeRef(outputShape))};
}

bool checkInsertReshapeDimOrder(const DimsOrder& dimOrder, bool acrossChannel) {
    if (dimOrder == DimsOrder::HCNW || dimOrder == DimsOrder::HNWC || dimOrder == DimsOrder::CWNH) {
        return false;
    }

    if (acrossChannel == true) {
        return true;
    }

    return dimOrder != DimsOrder::NHCW && dimOrder != DimsOrder::NWCH && dimOrder != DimsOrder::WCHN;
}

bool canDecomposeMVN(VPU::MVNOp op, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    if (inputType.getRank() != 4) {
        log.nest(1).trace("Support for decompose MVN is limited to 4D tensors only");
        return false;
    }

    if (op.getInternalReshape().has_value()) {
        log.nest(1).trace("Real 'internal_reshape' does not fit into CMX");
        return true;
    }

    // Can't get feasible tiling strategy for MVNOp because it will not fit into CMX.
    if (!op.fitIntoCMX(getTiledBuffers(inputType, outputType, op.getNonNormDims()))) {
        log.nest(1).trace("Can't still fit into CMX after tiling. The pass is used to decompose MVNOp.");
        return true;
    }
    return false;
}

//
// DecomposeMVNPass
//

class DecomposeMVNPass final : public VPU::impl::DecomposeMVNBase<DecomposeMVNPass> {
public:
    explicit DecomposeMVNPass(Logger log, bool forceDecompose): _forceDecompose(forceDecompose) {
        Base::initLogger(log, Base::getArgumentName());
    }
    ~DecomposeMVNPass() override {
    }
    // Explicit copy constructor: forceDecomposeOpt re-registers via its in-class initializer.
    DecomposeMVNPass(const DecomposeMVNPass& other)
            : VPU::impl::DecomposeMVNBase<DecomposeMVNPass>(other), _forceDecompose(other._forceDecompose) {
    }
    DecomposeMVNPass& operator=(const DecomposeMVNPass&) = delete;
    DecomposeMVNPass(DecomposeMVNPass&&) = delete;
    DecomposeMVNPass& operator=(DecomposeMVNPass&&) = delete;

public:
    class MVNConverter;

private:
    void safeRunOnFunc() final;
    bool _forceDecompose;
    mlir::Pass::Option<bool> forceDecomposeOpt{
            *this, "force-decompose",
            llvm::cl::desc("Force decompose CMX-fitting MVN ops when layout/type/size guards pass"),
            llvm::cl::init(false)};
};

//
// MVNConverter
//

class DecomposeMVNPass::MVNConverter final : public mlir::OpRewritePattern<VPU::MVNOp> {
public:
    MVNConverter(mlir::MLIRContext* ctx, Logger log, bool forceDecompose)
            : mlir::OpRewritePattern<VPU::MVNOp>(ctx), _log(log), _forceDecompose(forceDecompose) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _forceDecompose;
};

mlir::LogicalResult DecomposeMVNPass::MVNConverter::matchAndRewrite(VPU::MVNOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    if (inputType.getRank() != 4) {
        _log.nest(1).trace("Support for decompose MVN is limited to 4D tensors only");
        return mlir::failure();
    }

    auto module = origOp.getOperation()->getParentOfType<mlir::ModuleOp>();

    const bool fitsInCMX = !canDecomposeMVN(origOp, _log);
    if (fitsInCMX) {
        if (!_forceDecompose) {
            _log.debug("Skipping MVN decomposition: op fits in CMX and forceDecompose is disabled.");
            return mlir::failure();
        }
        // Force decompose only when the downstream DPU-normalize path can apply.
        // RunMVNNormalizeOnDPU requires NHWC layout and f16/bf16 element type for MVN1NormalizeOp -> NCE.MaxPool.
        if (inputType.getDimsOrder() != DimsOrder::NHWC) {
            _log.debug("Force decompose skipped: input layout is not NHWC.");
            return mlir::failure();
        }
        if (inputType.getShape()[Dims4D::Act::N] != 1) {
            _log.debug("Force decompose skipped: batch {0} != 1 is not supported by RunMVNNormalizeOnDPU.",
                       inputType.getShape()[Dims4D::Act::N]);
            return mlir::failure();
        }
        const auto elemType = inputType.getElementType();
        if (!mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(elemType)) {
            _log.debug("Force decompose skipped: element type {0} not supported by DPU normalize path.", elemType);
            return mlir::failure();
        }
        // Guard 1 -- channel alignment: each tile must hold at least VPU_CHANNEL_ALIGNMENT
        // channels so that MVN1Normalize (NCE.MaxPool) can form a valid DPU workload.
        // RunMVNNormalizeOnDPU also requires C % VPU_CHANNEL_ALIGNMENT == 0; skip forceDecompose
        // if that would not hold so a CMX-fitting MVN is not needlessly left on SHAVE decomposed.
        const auto numTiles = static_cast<int64_t>(config::getTileExecutor(module).getCount());
        VPUX_THROW_UNLESS(numTiles > 0, "Tile executor count must be positive, got {0}", numTiles);
        const auto channels = inputType.getShape()[Dims4D::Act::C];
        if (channels % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
            _log.debug("Force decompose skipped: channel count {0} is not aligned to VPU_CHANNEL_ALIGNMENT {1}.",
                       channels, VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
            return mlir::failure();
        }
        if (channels < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT * numTiles) {
            _log.debug("Force decompose skipped: {0} channels / {1} tiles < VPU_CHANNEL_ALIGNMENT {2}.", channels,
                       numTiles, VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
            return mlir::failure();
        }
        // Guard 2 -- per-tile CMX utilization: DPU dispatch overhead dominates when the
        // tensor is small relative to per-tile CMX. Below this threshold SHAVE normalizes the
        // data efficiently in a single CMX-resident pass; N-tile DPU parallelism pays off above
        // this threshold. Shared with RunMVNNormalizeOnDPU's sibling callers -- see
        // VPU::NCEInvariant::isLargeEnoughForDPUOverSHAVE.
        const auto tensorBytes = static_cast<int64_t>(inputType.getTotalAllocSize().count());
        if (!VPU::NCEInvariant::isLargeEnoughForDPUOverSHAVE(origOp.getOperation(), tensorBytes, numTiles)) {
            _log.debug("Force decompose skipped: tensor {0}B is too small relative to per-tile CMX / {1} tiles.",
                       tensorBytes, numTiles);
            return mlir::failure();
        }
        _log.trace("Force decomposing MVNOp at '{0}' (channels/tile >= {1}, tensor >= per-tile CMX)", origOp->getLoc(),
                   VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
    }

    const auto& ctx = origOp.getContext();
    auto inputDimOrder = inputType.getDimsOrder();
    // Use the op's own multiClusterStrategy (if set) to compute the correct tile count.
    // Falling back to the global executor count can create MVN1SumOp with wrong output_height
    // when the op runs on fewer clusters than the hardware maximum.
    const auto totalClusters = static_cast<int64_t>(config::getTileExecutor(module).getCount());
    const auto numClusters = [&]() -> int64_t {
        if (auto strategy = origOp.getMultiClusterStrategy()) {
            return getOptimalNumClusters(origOp.getOperation(), inputType.getShape(), strategy.value());
        }
        return totalClusters;
    }();
    const auto accrossChannels = origOp.getAcrossChannels();

    mlir::Value lastOp = origOp.getInput();
    if (checkInsertReshapeDimOrder(inputDimOrder, accrossChannels)) {
        const auto inputShape = inputType.getShape();
        const auto inputBatch = inputShape[Dims4D::Act::N];
        const auto inputChannel = accrossChannels ? 1 : inputShape[Dims4D::Act::C];
        const auto inputHeight =
                accrossChannels ? inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W] * inputShape[Dims4D::Act::C]
                                : inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W];

        auto newShape = Shape{inputBatch, inputChannel, inputHeight, 1};

        lastOp = rewriter.createOrFold<VPU::ShapeCastOp>(origOp.getLoc(), inputType.changeShape(newShape),
                                                         origOp.getInput(), getIntArrayAttr(ctx, newShape));
    }

    auto tileMVN1SumOp = rewriter.create<VPU::MVN1SumOp>(appendLoc(origOp.getLoc(), "mvn1Sum"), lastOp, accrossChannels,
                                                         origOp.getNormalizeVariance(), numClusters);

    const auto internalReshape =
            origOp.getInternalReshape().has_value() ? origOp.getInternalReshape().value() : nullptr;
    auto tileMVN1MeanVarOp = rewriter.create<VPU::MVN1MeanVarOp>(
            appendLoc(origOp.getLoc(), "mvn1MeanVar"), tileMVN1SumOp->getResult(0),
            getIntArrayAttr(rewriter, inputType.getShape().raw()), accrossChannels, origOp.getNormalizeVariance(),
            origOp.getEps(), inputType.getElementType(), internalReshape);

    auto tileMVN1NormalizeOp = rewriter.create<VPU::MVN1NormalizeOp>(
            appendLoc(origOp.getLoc(), "mvn1Normalize"), lastOp, tileMVN1MeanVarOp.getResult(),
            origOp.getAcrossChannelsAttr(), origOp.getNormalizeVarianceAttr(), origOp.getHighPrecisionNormalizeAttr(),
            rewriter.getBoolAttr(fitsInCMX));

    auto origOpOutType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto reshapeOutOp =
            rewriter.createOrFold<VPU::ShapeCastOp>(origOp.getLoc(), origOpOutType, tileMVN1NormalizeOp.getOutput(),
                                                    getIntArrayAttr(ctx, origOpOutType.getShape()));

    rewriter.replaceOp(origOp, reshapeOutOp);
    return mlir::success();
}

//
// safeRunOnFunc
//

void DecomposeMVNPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MVNConverter>(&ctx, _log, _forceDecompose || forceDecomposeOpt);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createDecomposeMVNPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDecomposeMVNPass(Logger log, bool forceDecompose) {
    return std::make_unique<DecomposeMVNPass>(log, forceDecompose);
}
