//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_RUNMVNNORMALIZEONDPU
#define GEN_PASS_DEF_RUNMVNNORMALIZEONDPU
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
namespace {

//
// RunMVNNormalizeOnDPUPass
//

class RunMVNNormalizeOnDPU final : public mlir::OpRewritePattern<VPU::MVN1NormalizeOp> {
public:
    RunMVNNormalizeOnDPU(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MVN1NormalizeOp>(ctx), _log(std::move(log)) {
        setDebugName("RunMVNNormalizeOnDPU");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVN1NormalizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    // Extract mean from meanVar, returns shape [1, C, 1, 1] for NCE operations
    mlir::Value extractMean(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                            mlir::PatternRewriter& rewriter) const;
    // Extract scale from meanVar, returns shape [C, 1, 1, 1] for weights table
    mlir::Value extractScale(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                             mlir::Type fp32ElemType, mlir::PatternRewriter& rewriter) const;
    // Compute bias = -mean, input shape [1, C, 1, 1], output shape [C, 1, 1, 1] (after reshape)
    mlir::Value computeBias(mlir::Location origOpLoc, mlir::Value mean, int64_t weightsTableC, const DimsOrder& inOrder,
                            VPU::MVN1NormalizeOp origOp, mlir::PatternRewriter& rewriter) const;
    mlir::Value createWeightsTable(mlir::Location origOpLoc, mlir::Value scale, mlir::Value bias, int64_t weightsTableC,
                                   mlir::Type fp32ElemType, const DimsOrder& inOrder,
                                   mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

// Extract mean from meanVar at W=0 (slice on the last dimension)
// Input:  meanVar with shape [1, C, 1, 2]
// Output: mean with shape [1, C, 1, 1] (keep NCE-friendly format)
mlir::Value RunMVNNormalizeOnDPU::extractMean(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    // meanVar shape: [1, C, 1, 2] with NHWC layout
    // Extract mean at W=0: slice [0, 0, 0, 0] with size [1, C, 1, 1]
    const SmallVector<int64_t> meanOffsets = {0, 0, 0, 0};
    const SmallVector<int64_t> meanSizes = {1, weightsTableC, 1, 1};
    auto mean = rewriter.create<VPU::SliceOp>(appendLoc(origOpLoc, "_extract_mean"), meanVar,
                                              getIntArrayAttr(ctx, meanOffsets), getIntArrayAttr(ctx, meanSizes))
                        .getOutput();

    // Keep [1, C, 1, 1] shape for NCE.Eltwise operation
    return mean;
}

// Extract scale from meanVar at W=1 (slice on the last dimension)
// Input:  meanVar with shape [1, C, 1, 2]
// Output: scale with shape [C, 1, 1, 1] in fp32 (for weights table)
mlir::Value RunMVNNormalizeOnDPU::extractScale(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                                               mlir::Type fp32ElemType, mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    // meanVar shape: [1, C, 1, 2] with NHWC layout
    // Extract scale at W=1: slice [0, 0, 0, 1] with size [1, C, 1, 1]
    const SmallVector<int64_t> scaleOffsets = {0, 0, 0, 1};
    const SmallVector<int64_t> scaleSizes = {1, weightsTableC, 1, 1};
    auto scale = rewriter.create<VPU::SliceOp>(appendLoc(origOpLoc, "_extract_scale"), meanVar,
                                               getIntArrayAttr(ctx, scaleOffsets), getIntArrayAttr(ctx, scaleSizes))
                         .getOutput();

    // Reshape to [C, 1, 1, 1] for weights table concat
    scale = rewriter.create<VPU::ShapeCastOp>(appendLoc(origOpLoc, "__scale_shape_cast"), scale,
                                              getIntArrayAttr(ctx, Shape{weightsTableC, 1, 1, 1}));

    // Convert to fp32 for weights table
    scale = rewriter.create<VPU::ConvertOp>(appendLoc(origOpLoc, "_scale_convert"), scale,
                                            mlir::TypeAttr::get(fp32ElemType));
    return scale;
}

// Compute bias = -mean
// Input:  mean with shape [1, C, 1, 1]
// Output: bias with shape [C, 1, 1, 1] in fp32 (for weights table)
mlir::Value RunMVNNormalizeOnDPU::computeBias(mlir::Location origOpLoc, mlir::Value mean, int64_t weightsTableC,
                                              const DimsOrder& inOrder, VPU::MVN1NormalizeOp origOp,
                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto fp32ElemType = mlir::Float32Type::get(ctx);

    // mean has shape [1, C, 1, 1] with NHWC layout
    // Create constant -1 with same shape [1, C, 1, 1] for NCE.Eltwise broadcast
    const auto negOneShape = Shape({1, weightsTableC, 1, 1});
    const auto negOneType = mlir::RankedTensorType::get(negOneShape.raw(), mlir::Float16Type::get(ctx));
    const auto negOneAttr = Const::createConstContent(negOneType, ArrayRef({-1.0f}));
    const auto negOneContentAttr = Const::ContentAttr::get(negOneAttr).transform().reorder(inOrder).get();
    auto negOne = rewriter.create<Const::DeclareOp>(appendLoc(origOpLoc, "_neg_one"), negOneContentAttr.getType(),
                                                    std::move(negOneContentAttr))
                          .getOutput();

    // Multiply mean by -1 to get -mean
    // Both inputs have shape [1, C, 1, 1]
    const auto opType = VPU::EltwiseType::MULTIPLY;
    auto bias_ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);
    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    auto bias =
            rewriter.create<VPU::NCEEltwiseOp>(appendLoc(origOpLoc, "_compute_bias"), mean.getType(), mean, negOne,
                                               /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
                                               VPU::EltwiseTypeAttr::get(ctx, opType), bias_ppeAttr, mpeEngineModeAttr,
                                               /*multi_cluster_strategy*/ nullptr,
                                               /*is_inplace*/ nullptr, nullptr, nullptr)
                    .getOutput();

    // Convert to fp32
    bias = rewriter.create<VPU::ConvertOp>(appendLoc(origOpLoc, "_bias_convert"), bias,
                                           mlir::TypeAttr::get(fp32ElemType));

    // Reshape from [1, C, 1, 1] to [C, 1, 1, 1] for weights table concat
    bias = rewriter.create<VPU::ShapeCastOp>(appendLoc(origOpLoc, "__bias_reshape_for_concat"), bias,
                                             getIntArrayAttr(ctx, Shape{weightsTableC, 1, 1, 1}));

    return bias;
}

// Create weights table with shape [C, 1, 1, 4]
// Layout: [sparsityPointers(2), scale(1), bias(1)]
mlir::Value RunMVNNormalizeOnDPU::createWeightsTable(mlir::Location origOpLoc, mlir::Value scale, mlir::Value bias,
                                                     int64_t weightsTableC, mlir::Type fp32ElemType,
                                                     const DimsOrder& inOrder, mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    // Create weightsSparsityPointers with shape [C, 1, 1, 2] and initialized to 0
    const auto sparsityPointerShape = Shape({weightsTableC, 1, 1, 2});
    const auto sparsityPointerType = mlir::RankedTensorType::get(sparsityPointerShape.raw(), fp32ElemType);
    const auto baseAttr = Const::createConstContent(sparsityPointerType, ArrayRef({0.0f}));
    const auto contentAttr = Const::ContentAttr::get(baseAttr).transform().reorder(inOrder).get();
    auto weightsSparsityPointers = rewriter.create<Const::DeclareOp>(appendLoc(origOpLoc, "_weights_sparsity_pointers"),
                                                                     contentAttr.getType(), std::move(contentAttr))
                                           .getOutput();

    // Concat {weightsSparsityPointers [C,1,1,2], scale [C,1,1,1], bias [C,1,1,1]} to create weights table [C,1,1,4]
    const auto weightsTableOutType =
            mlir::cast<NDTypeInterface>(mlir::RankedTensorType::get({weightsTableC, 1, 1, 4}, fp32ElemType))
                    .changeDimsOrder(inOrder);
    const SmallVector<SmallVector<int64_t>> staticOffsets = {{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}};
    auto weightsTable = rewriter.create<VPU::ConcatOp>(appendLoc(origOpLoc, "_concat"), weightsTableOutType,
                                                       mlir::ValueRange{weightsSparsityPointers, scale, bias},
                                                       getIntArrayOfArray(ctx, staticOffsets))
                                .getOutput();

    // Reinterpret weights table to si32
    const auto desiredType = mlir::cast<NDTypeInterface>(weightsTable.getType())
                                     .changeElemType(getSInt32Type(ctx))
                                     .changeDimsOrder(inOrder);
    weightsTable =
            rewriter.create<Core::ReinterpretCastOp>(appendLoc(origOpLoc, "_weights_table"), desiredType, weightsTable)
                    .getOutput();

    return weightsTable;
}

mlir::LogicalResult RunMVNNormalizeOnDPU::matchAndRewrite(VPU::MVN1NormalizeOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto origOpLoc = origOp->getLoc();
    _log.trace("Found MVN1NormalizeOp operation '{0}' at '{1}'.", origOp->getName(), origOpLoc);

    auto ctx = origOp.getContext();

    // Validate input
    auto meanVar = origOp.getMeanVar();
    auto meanVarShape = getShape(meanVar);
    auto weightsTableC = meanVarShape[Dims4D::Act::C];
    if (weightsTableC % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
        return mlir::failure();
    }

    // meanVar shape: [1, C, 1, 2] with NHWC layout
    // W=0 contains mean (μ), W=1 contains scale (1/sqrt(σ² + ε))
    const auto inOrder = DimsOrder::fromValue(meanVar);
    auto fp32ElemType = mlir::Float32Type::get(ctx);

    // Extract mean (μ) from meanVar at W=0 (slice on the last dimension)
    // Output shape: [1, C, 1, 1]
    auto mean = extractMean(origOpLoc, meanVar, weightsTableC, rewriter);

    // Extract scale (1/sqrt(σ² + ε)) from meanVar at W=1 (slice on the last dimension)
    // Output shape: [C, 1, 1, 1] in fp32
    auto scale = extractScale(origOpLoc, meanVar, weightsTableC, fp32ElemType, rewriter);

    // Compute bias = -μ
    // Input shape: [1, C, 1, 1], Output shape: [C, 1, 1, 1] in fp32
    auto bias = computeBias(origOpLoc, mean, weightsTableC, inOrder, origOp, rewriter);

    // Create weights table with shape [C, 1, 1, 4]
    auto weightsTable = createWeightsTable(origOpLoc, scale, bias, weightsTableC, fp32ElemType, inOrder, rewriter);

    // Create NCEMaxPoolOp to replace the original operation
    const SmallVector<int64_t> maxPoolStrides = {1, 1};
    const SmallVector<int64_t> maxPoolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads)));
    auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);
    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }
    auto newMaxPoolOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getOutput().getType(),
            /*reduceXyMax */ nullptr,
            /*reduceXyMin */ nullptr,
            /*reduceTensorMinMax */ nullptr, origOp.getInput(), weightsTable, nullptr, nullptr,
            getIntArrayAttr(ctx, maxPoolKernels), getIntArrayAttr(ctx, maxPoolStrides), padAttr, ppeAttr,
            mpeEngineModeAttr, nullptr,
            /*MultiClusterStrategyAttr=*/nullptr, /*maxPoolOp.getOutputPaddingAttr()=*/nullptr,
            /*maxPoolOp.getInputPaddingAttr()=*/nullptr, /*axesValue=*/nullptr);
    rewriter.replaceOp(origOp, newMaxPoolOp.getOutput());

    return mlir::success();
}

class RunMVNNormalizeOnDPUPass final : public VPU::impl::RunMVNNormalizeOnDPUBase<RunMVNNormalizeOnDPUPass> {
public:
    explicit RunMVNNormalizeOnDPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void RunMVNNormalizeOnDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RunMVNNormalizeOnDPU>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createRunMVNNormalizeOnDPUPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createRunMVNNormalizeOnDPUPass(Logger log) {
    return std::make_unique<RunMVNNormalizeOnDPUPass>(log);
}
