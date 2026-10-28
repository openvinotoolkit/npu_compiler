//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/convert_tile_with_eltwise_to_nce_pool.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <llvm/ADT/STLExtras.h>

using namespace vpux;

VPU::ConvertTileWithEltwiseToNCEPoolRewriter::ConvertTileWithEltwiseToNCEPoolRewriter(mlir::MLIRContext* ctx,
                                                                                      Logger log)
        : mlir::OpRewritePattern<VPU::TileOp>(ctx), _log(log) {
    setDebugName("ConvertTileWithEltwiseToNCEPoolPass::ConvertTileWithEltwiseToNCEPool");
}

mlir::LogicalResult VPU::ConvertTileWithEltwiseToNCEPoolRewriter::matchAndRewrite(
        VPU::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TileOp has multiple users at '{0}'", origOp->getLoc());
    }

    auto userOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(*origOp.getOutput().getUsers().begin());
    if (userOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "TileOp has a non-eltwise user at '{0}'", origOp->getLoc());
    }

    if (userOp.getWeightTableScale() != nullptr || userOp.getWeightTableBias() != nullptr) {
        return matchFailed(_log, rewriter, userOp, "Eltwise user already has weight-table scale/bias operands at '{0}'",
                           userOp->getLoc());
    }

    if (userOp.getReduceXyMin() != nullptr || userOp.getReduceXyMax() != nullptr ||
        userOp.getReduceTensorMinMax() != nullptr) {
        return matchFailed(_log, rewriter, userOp, "Eltwise user has unsupported reduce outputs at '{0}'",
                           userOp->getLoc());
    }

    const auto eltType = userOp.getOpType();
    if (eltType != VPU::EltwiseType::ADD && eltType != VPU::EltwiseType::MULTIPLY) {
        return matchFailed(_log, rewriter, origOp, "Eltwise type '{0}' is not supported for TileOp at '{1}'", eltType,
                           origOp->getLoc());
    }

    auto ctx = origOp.getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto scaleBiasAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterScaleBias*>();
    if (scaleBiasAdapter == nullptr) {
        return matchFailed(_log, rewriter, origOp, "PPE scale/bias adapter is not available at '{0}'",
                           origOp->getLoc());
    }

    auto input = origOp.getInput();
    const auto inputShape = getShape(input);
    if (inputShape.isDynamic()) {
        return matchFailed(_log, rewriter, origOp, "TileOp input has dynamic shape at '{0}'", origOp->getLoc());
    }

    if (inputShape.size() != 4) {
        return matchFailed(_log, rewriter, origOp, "TileOp input is not 4D at '{0}'", origOp->getLoc());
    }

    if (inputShape[Dims4D::Act::N] != 1 || inputShape[Dims4D::Act::H] != 1 || inputShape[Dims4D::Act::W] != 1) {
        return matchFailed(_log, rewriter, origOp, "TileOp input is not flattened to [1, C, 1, 1] at '{0}'",
                           origOp->getLoc());
    }

    const auto repeatedValues = parseIntArrayAttr<int64_t>(origOp.getRepeatsValues());
    if (repeatedValues.size() != 4) {
        return matchFailed(_log, rewriter, origOp, "TileOp repeats are not 4D at '{0}'", origOp->getLoc());
    }
    if (Shape(repeatedValues)[Dims4D::Act::N] != 1 || Shape(repeatedValues)[Dims4D::Act::C] != 1) {
        return matchFailed(_log, rewriter, origOp, "TileOp repeats are not supported at '{0}'", origOp->getLoc());
    }

    const auto eltwiseInput1 = userOp.getInput1();
    const auto eltwiseInput2 = userOp.getInput2();
    auto newInput = eltwiseInput1 == origOp.getOutput() ? eltwiseInput2 : eltwiseInput1;
    if (getShape(newInput).isDynamic() || getShape(userOp.getOutput()).isDynamic()) {
        return matchFailed(_log, rewriter, userOp, "Eltwise input or output has dynamic shape at '{0}'",
                           userOp->getLoc());
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inputElemType = inputType.getElementType();

    auto fp16Type = mlir::Float16Type::get(ctx);
    auto fp32Type = mlir::Float32Type::get(ctx);
    if (inputElemType != fp16Type && inputElemType != fp32Type) {
        return matchFailed(_log, rewriter, origOp, "TileOp input element type is not supported at '{0}'",
                           origOp->getLoc());
    }

    auto userPpe = userOp.getPpe();
    const auto scale = scaleBiasAdapter->getScale(userPpe);
    const auto bias = scaleBiasAdapter->getBias(userPpe);
    VPUX_THROW_UNLESS(bias.has_value() && scale.has_value(),
                      "Eltwise user at '{0}' has no per-tensor scale/bias in its PPE", userOp->getLoc());

    const auto allScalesAreNeutral = llvm::all_of(scale.value(), [](double scaleValue) {
        return scaleValue == 1.0;
    });
    if (!allScalesAreNeutral) {
        return matchFailed(_log, rewriter, userOp, "Eltwise user has non-neutral scale at '{0}'", userOp->getLoc());
    }
    if (bias.value() != 0.0) {
        return matchFailed(_log, rewriter, userOp, "Eltwise user has non-neutral bias at '{0}'", userOp->getLoc());
    }

    const auto weightsTableC = inputShape[Dims4D::Act::C];

    const auto createDummyPerChannelConst = [&](llvm::StringRef suffix, float value) {
        const auto tableShape = Shape({weightsTableC, 1, 1, 1});
        const auto tableType = mlir::RankedTensorType::get(tableShape.raw(), fp32Type);
        const SmallVector<float> tableValues = {value};
        return Const::createConst<float>(rewriter, appendLoc(origOp.getLoc(), suffix), tableType, ArrayRef(tableValues),
                                         [](Const::ContentSetup& contentSetup) -> Const::ContentSetup {
                                             return contentSetup.reorder(DimsOrder::NCHW);
                                         });
    };

    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads)));

    auto materializeWeightTableInput = [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
        auto tableInput = value;
        auto tableInputType = mlir::cast<vpux::NDTypeInterface>(tableInput.getType());

        if (tableInputType.getElementType() == fp16Type) {
            const auto fp32OutputType = tableInputType.changeElemType(fp32Type);
            if (auto avgPoolOp = tableInput.getDefiningOp<VPU::NCEAveragePoolOp>()) {
                if (avgPoolOp->hasOneUse()) {
                    rewriter.modifyOpInPlace(avgPoolOp, [&]() {
                        avgPoolOp.getOutput().setType(fp32OutputType);
                    });
                    tableInput = avgPoolOp.getOutput();
                } else {
                    auto* clonedAvgPoolOp = rewriter.clone(*avgPoolOp);
                    rewriter.modifyOpInPlace(clonedAvgPoolOp, [&]() {
                        clonedAvgPoolOp->getResult(0).setType(fp32OutputType);
                    });
                    tableInput = clonedAvgPoolOp->getResult(0);
                }
            } else if (auto maxPoolOp = tableInput.getDefiningOp<VPU::NCEMaxPoolOp>()) {
                if (maxPoolOp->hasOneUse()) {
                    rewriter.modifyOpInPlace(maxPoolOp, [&]() {
                        maxPoolOp.getOutput().setType(fp32OutputType);
                    });
                    tableInput = maxPoolOp.getOutput();
                } else {
                    auto* clonedMaxPoolOp = rewriter.clone(*maxPoolOp);
                    rewriter.modifyOpInPlace(clonedMaxPoolOp, [&]() {
                        clonedMaxPoolOp->getResult(0).setType(fp32OutputType);
                    });
                    tableInput = clonedMaxPoolOp->getResult(0);
                }
            } else {
                auto materializedMaxPoolOp = VPU::NCEMaxPoolOp::create(
                        rewriter, appendLoc(origOp.getLoc(), "weight_table_maxpool"), fp32OutputType,
                        /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr,
                        /*reduceTensorMinMax=*/nullptr, tableInput, /*weightsTable=*/nullptr,
                        /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
                        getIntArrayAttr(ctx, poolKernels), getIntArrayAttr(ctx, poolStrides), padAttr,
                        VPU::PPEStubAttr::get(ctx),
                        // VPU::NCEMaxPoolOp has no MPEEngineInfoOpInterface to derive this independently;
                        // reuse userOp's, which is equivalent here since neither op has zero points.
                        userOp.getMpeEngineAttr(), /*multi_cluster_strategyAttr=*/nullptr,
                        /*output_paddingAttr=*/nullptr, /*input_paddingAttr=*/nullptr,
                        /*s2dd2s_configAttr=*/nullptr, /*axes_valueAttr=*/nullptr);
                rewriter.modifyOpInPlace(materializedMaxPoolOp, [&]() {
                    materializedMaxPoolOp.setPpeAttr(ppeConfig.retrievePPEAttribute(materializedMaxPoolOp));
                });
                tableInput = materializedMaxPoolOp.getOutput();
            }
            tableInputType = mlir::cast<vpux::NDTypeInterface>(tableInput.getType());
        }

        if (tableInputType.getDimsOrder() != DimsOrder::NCHW) {
            const auto memPerm =
                    tryToFindPermutationForPermuteCast(tableInputType, DimsOrder::NCHW, tableInputType.getShape(), ctx);
            if (!memPerm.has_value()) {
                return mlir::failure();
            }

            auto permuteCastOp =
                    VPU::PermuteCastOp::create(rewriter, appendLoc(origOp.getLoc(), "weight_table_permute"), tableInput,
                                               DimsOrder::NCHW.toAffineMap(ctx), memPerm.value());
            tableInput = permuteCastOp.getOutput();
        }

        return tableInput;
    };

    rewriter.setInsertionPoint(userOp);

    auto fp32Input = materializeWeightTableInput(input);
    if (mlir::failed(fp32Input)) {
        return matchFailed(_log, rewriter, origOp, "Cannot materialize weight-table input for TileOp at '{0}'",
                           origOp->getLoc());
    }

    auto shapeCastOp = VPU::AffineReshapeOp::create(
            rewriter, appendLoc(origOp.getLoc(), "weight_table_reshape"), *fp32Input,
            getIntArrayOfArray(ctx, SmallVector<SmallVector<int64_t>>{{0}, {0}, {1}, {2, 3}}),
            getIntArrayAttr(ctx, SmallVector<int64_t>{weightsTableC, 1, 1, 1}));

    mlir::Value scaleTable = nullptr;
    mlir::Value biasTable = nullptr;
    if (eltType == VPU::EltwiseType::ADD) {
        biasTable = shapeCastOp.getOutput();
        scaleTable = createDummyPerChannelConst("scale_table", 1.0f);
    } else if (eltType == VPU::EltwiseType::MULTIPLY) {
        scaleTable = shapeCastOp.getOutput();
        biasTable = createDummyPerChannelConst("bias_table", 0.0f);
    } else {
        VPUX_THROW("Unexpected eltwise type: {0}", eltType);
    }

    const auto weightTables =
            createWeightTableOperands(origOp.getLoc(), scaleTable, biasTable, weightsTableC, rewriter);
    auto newPpe = scaleBiasAdapter->discardScaleBias(userPpe);
    // Eltwise ADD/MULTIPLY always carry a non-null mpe_engine (SCL mode) on 50XX: preserve it on the pool.
    rewriter.replaceOpWithNewOp<VPU::NCEMaxPoolOp>(
            userOp, userOp.getResult(0).getType(),
            /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr, /*reduceTensorMinMax=*/nullptr, newInput,
            weightTables.weightsTable, weightTables.scale, weightTables.bias, getIntArrayAttr(ctx, poolKernels),
            getIntArrayAttr(ctx, poolStrides), padAttr, newPpe, userOp.getMpeEngineAttr(),
            /*multi_cluster_strategyAttr=*/nullptr, userOp.getOutputPaddingAttr(), userOp.getInputPaddingAttr(),
            /*s2dd2s_configAttr=*/nullptr, /*axes_valueAttr=*/nullptr);

    rewriter.eraseOp(origOp);
    return mlir::success();
}
