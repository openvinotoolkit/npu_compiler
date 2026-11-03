//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSEDEPTHTOSPACETODPUPASS
#define GEN_PASS_DEF_FUSEDEPTHTOSPACETODPUPASS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Once ODU D2S is active, tiling over C is forbidden, so the whole weights-related buffers (filter, weights
// table, scale, bias, zero points) must reside in CMX together with any single spatial (H/W) tile. Reject
// fusion when they alone would already consume too much of the CMX budget to leave room for tiling.
constexpr int64_t CMX_WEIGHTS_FRACTION_DIVISOR = 3;

//
// FuseDepthToSpaceToDpuPass
//

class FuseDepthToSpaceToDpuPass final : public VPU::impl::FuseDepthToSpaceToDpuPassBase<FuseDepthToSpaceToDpuPass> {
public:
    explicit FuseDepthToSpaceToDpuPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool canFuse(VPU::NCEConvolutionOp convOp, VPU::DepthToSpaceOp d2sOp, VPU::S2DD2SConfigAttr& s2dd2sConfig);
    bool checkWeightsSizeForFusion(VPU::NCEConvolutionOp convOp, vpux::NDTypeInterface convOutType);
};

bool FuseDepthToSpaceToDpuPass::checkWeightsSizeForFusion(VPU::NCEConvolutionOp convOp,
                                                          vpux::NDTypeInterface convOutType) {
    const auto filterSize = mlir::cast<vpux::NDTypeInterface>(convOp.getFilter().getType()).getTotalAllocSize();

    SmallVector<Byte> weightTableBuffers;
    // Size of weights is calculated based on the pre-S2DD2S channels count, even if channels count must be expressed as
    // aligned with hardware constraints. So no further rounding up is needed here for now.
    const auto OC = convOutType.getShape()[Dims4D::Act::C];
    if (mlir::failed(VPU::NCEInvariant::getWeightTableBuffers(convOp.getOperation(), weightTableBuffers, OC))) {
        _log.trace("Failed to compute weight table buffer sizes, skipping fusion");
        return false;
    }

    auto weightsSize = filterSize;
    for (const auto& tableSize : weightTableBuffers) {
        weightsSize += tableSize;
    }

    const auto totalCMXSize = VPU::getTotalCMXSize(convOp.getOperation());
    if (weightsSize.count() * CMX_WEIGHTS_FRACTION_DIVISOR > totalCMXSize.count()) {
        _log.trace("NCEConvolutionOp weights and weight tables size '{0}' exceeds 1/{1} of available CMX '{2}', "
                   "skipping fusion since tiling over C is not possible once ODU D2S is active",
                   weightsSize, CMX_WEIGHTS_FRACTION_DIVISOR, totalCMXSize);

        return false;
    }

    return true;
}

bool FuseDepthToSpaceToDpuPass::canFuse(VPU::NCEConvolutionOp convOp, VPU::DepthToSpaceOp d2sOp,
                                        VPU::S2DD2SConfigAttr& s2dd2sConfig) {
    if (convOp.getS2dd2sConfigAttr() != nullptr) {
        _log.trace("NCEConvolutionOp already has an active ODU S2DD2S config, skipping");
        return false;
    }

    if (d2sOp.getPaddedChannels().has_value()) {
        _log.trace("DepthToSpaceOp has padding, cannot be fused");
        return false;
    }

    const auto blkSize = d2sOp.getBlockSize();
    if (blkSize != 2 && blkSize != 4) {
        _log.trace("DepthToSpaceOp has unsupported block size {0}", blkSize);
        return false;
    }

    const auto convOutType = mlir::cast<vpux::NDTypeInterface>(convOp.getOutput().getType());
    const auto d2sOutType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getOutput().getType());
    if (convOutType.getDimsOrder() != DimsOrder::NHWC || d2sOutType.getDimsOrder() != DimsOrder::NHWC) {
        _log.trace("NCEConvolutionOp or DepthToSpaceOp output is not in NHWC layout, skipping fusion");
        return false;
    }

    if (convOutType.getElementType() != d2sOutType.getElementType()) {
        _log.trace("NCEConvolutionOp or DepthToSpaceOp output element type differs, skipping fusion");
        return false;
    }

    if (d2sOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST && !checkWeightsSizeForFusion(convOp, convOutType)) {
        return false;
    }

    auto* ctx = convOp.getContext();
    const auto variant = (d2sOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST) ? VPU::S2DD2SVariant::BLOCK_FIRST
                                                                                 : VPU::S2DD2SVariant::DEPTH_FIRST;
    s2dd2sConfig =
            VPU::S2DD2SConfigAttr::get(ctx, VPU::S2DD2SEnableAttr::get(ctx, VPU::S2DD2SEnable::D2S),
                                       VPU::S2DD2SVariantAttr::get(ctx, variant), vpux::getIntAttr(ctx, blkSize));

    // Verify that going from the NCEConvolutionOp's real (pre-ODU) output shape to the DepthToSpaceOp's
    // output shape is indeed described by the ODU D2S scaling for this config, otherwise the DPU hardware
    // would not reproduce the same result as the standalone DepthToSpaceOp.
    const auto rank = static_cast<int64_t>(d2sOutType.getShape().size());
    const auto transformInfo = VPU::getODUS2DD2STransformInfo(s2dd2sConfig, rank);
    if (!transformInfo.has_value()) {
        _log.trace("DepthToSpaceOp does not provide active ODU D2S transform info");
        return false;
    }

    const auto scales = VPU::getODUS2DD2SScaling(*transformInfo, rank);
    const auto preODUShape = VPU::invertODUScaling(scales, d2sOutType.getShape(), d2sOp->getLoc());
    if (mlir::failed(preODUShape) || ShapeRef(*preODUShape) != convOutType.getShape()) {
        _log.trace("DepthToSpaceOp output shape is not consistent with NCEConvolutionOp output through ODU D2S "
                   "scaling, skipping fusion");
        return false;
    }

    return true;
}

void FuseDepthToSpaceToDpuPass::safeRunOnFunc() {
    auto func = getOperation();

    SmallVector<mlir::Operation*> eraseList;
    // Collect all NCE op -> DepthToSpaceOp pairs and try to fuse them.
    func.walk([&](VPU::NCEOpInterface nceOp) {
        // Early return if the producer is not one of the supported DPU operations.
        // TODO: extend to VPU::NCEMaxPoolOp and other NCE ops once their fusion logic is implemented.
        if (!mlir::isa<VPU::NCEConvolutionOp>(nceOp.getOperation())) {
            _log.trace("Producer of DepthToSpaceOp is not a supported DPU op (NCEConvolutionOp), skipping");
            return;
        }

        if (!nceOp->hasOneUse()) {
            return;
        }

        auto d2sOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(*nceOp->getUsers().begin());
        if (d2sOp == nullptr) {
            return;
        }

        auto* parentOp = nceOp.getOperation();

        if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(parentOp)) {
            VPU::S2DD2SConfigAttr s2dd2sConfig;
            if (!canFuse(convOp, d2sOp, s2dd2sConfig)) {
                return;
            }

            mlir::OpBuilder builder(convOp);
            // Preserve convOp's reduce outputs (result types + axes_value) on the fused op: this pass
            // currently runs before FuseReduceMinMaxToDpu so convOp never has them today, but dropping
            // them unconditionally would silently lose data if the pass order ever changes.
            const auto reduceXyMaxType = convOp.getReduceXyMax() ? convOp.getReduceXyMax().getType() : nullptr;
            const auto reduceXyMinType = convOp.getReduceXyMin() ? convOp.getReduceXyMin().getType() : nullptr;
            const auto reduceTensorMinMaxType =
                    convOp.getReduceTensorMinMax() ? convOp.getReduceTensorMinMax().getType() : nullptr;
            auto newConvOp = VPU::NCEConvolutionOp::create(
                    builder, convOp.getLoc(), d2sOp.getOutput().getType(), reduceXyMaxType, reduceXyMinType,
                    reduceTensorMinMaxType, convOp.getInput(), convOp.getFilter(), convOp.getWeightsTable(),
                    convOp.getWeightTableScale(), convOp.getWeightTableBias(), convOp.getWeightZeroPoints(),
                    convOp.getStridesAttr(), convOp.getPadAttr(), convOp.getPpeAttr(), convOp.getMpeEngineAttr(),
                    convOp.getRawFilterShape(), convOp.getStaticRawFilterShape(), convOp.getMultiClusterStrategyAttr(),
                    convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr(), s2dd2sConfig,
                    convOp.getAxesValueAttr());

            d2sOp.getOutput().replaceAllUsesWith(newConvOp.getOutput());
            if (convOp.getReduceXyMax() != nullptr) {
                convOp.getReduceXyMax().replaceAllUsesWith(newConvOp.getReduceXyMax());
            }
            if (convOp.getReduceXyMin() != nullptr) {
                convOp.getReduceXyMin().replaceAllUsesWith(newConvOp.getReduceXyMin());
            }
            if (convOp.getReduceTensorMinMax() != nullptr) {
                convOp.getReduceTensorMinMax().replaceAllUsesWith(newConvOp.getReduceTensorMinMax());
            }
            eraseList.push_back(d2sOp);
            eraseList.push_back(convOp);

            _log.trace("Fused DepthToSpaceOp '{0}' into NCEConvolutionOp '{1}'", d2sOp->getLoc(), convOp->getLoc());
            return;
        }
    });

    llvm::SmallPtrSet<mlir::Operation*, 8> visited;
    for (auto* op : eraseList) {
        if (op == nullptr || !visited.insert(op).second) {
            continue;
        }
        if (op->use_empty()) {
            op->erase();
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseDepthToSpaceToDpuPass(Logger log) {
    return std::make_unique<FuseDepthToSpaceToDpuPass>(log);
}
