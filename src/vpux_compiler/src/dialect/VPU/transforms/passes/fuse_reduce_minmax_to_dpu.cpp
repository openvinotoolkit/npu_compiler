//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSEREDUCEMINMAXTODPU
#define GEN_PASS_DEF_FUSEREDUCEMINMAXTODPU
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// FuseReduceMinMaxToDpuPass
//

class FuseReduceMinMaxToDpuPass final : public VPU::impl::FuseReduceMinMaxToDpuBase<FuseReduceMinMaxToDpuPass> {
public:
    explicit FuseReduceMinMaxToDpuPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool checkAxesForFusion(llvm::ArrayRef<int64_t> axesValues, vpux::NDTypeInterface inType,
                            vpux::NDTypeInterface outType, bool& isPerTensorReduction);
};

bool FuseReduceMinMaxToDpuPass::checkAxesForFusion(llvm::ArrayRef<int64_t> axesValues, vpux::NDTypeInterface inType,
                                                   vpux::NDTypeInterface outType, bool& isPerTensorReduction) {
    const auto shape = outType.getShape();
    if (axesValues.size() != 1 && (axesValues.size() != shape.size())) {
        // DPU can only do reduce over Channels or over entire tensor,
        // so this means we only support reduction over one axis (C) or over all axes (NCHW)
        _log.trace("Unsupported number of axes for reduction: {0}, skipping", axesValues.size());
        return false;
    }

    // Reduce output doesn't support F32
    if (outType.getElementType().isF32()) {
        _log.trace("Unsupported element type for reduction output: {0}, skipping", outType.getElementType());
        return false;
    }

    if (axesValues.size() > 1) {
        if (axesValues.size() == shape.size()) {
            // TODO(E#207252): Support per-tensor reduction use case in DPU and remove this workaround
            isPerTensorReduction = true;
            _log.trace("Per-tensor reduction detected, but usecase is not supported for now, skipping");
            return false;
        } else {
            _log.trace("Unsupported number of axes for reduction: {0}, skipping", axesValues.size());
            return false;
        }
    }

    // Check that the reduction targets the channel axis and the DPU op does not permute its output.
    if (!IE::isChannelAxisReductionWithMatchingLayout(inType, outType, axesValues, _log)) {
        return false;
    }

    // Fusing is valid only when the DPU op's input and output are in NHWC layout,
    // so that ODU permute is not activated.
    const auto rank = checked_cast<int64_t>(shape.size());
    const auto expectedLayout = (rank == 5) ? DimsOrder::GNHWC : DimsOrder::NHWC;
    if (inType.getDimsOrder() != expectedLayout || outType.getDimsOrder() != expectedLayout) {
        _log.trace("DPU op input or output is not in NHWC layout, skipping fusion");
        return false;
    }

    return true;
}

void FuseReduceMinMaxToDpuPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    SmallVector<mlir::Operation*> eraseList;
    // Collect all ReduceMaxOp and ReduceMinOp operations in the function and try to fuse them into DPU operations
    func.walk([&](VPU::NCEOpInterface nceOp) {
        // Early return if the producer is not one of the supported DPU operations
        if (!mlir::isa<VPU::NCEConvolutionOp, VPU::NCEMatMulOp, VPU::NCEMaxPoolOp, VPU::NCEEltwiseOp,
                       VPU::NCEDepthConvolutionOp>(nceOp.getOperation())) {
            _log.trace("Producer of Reduce is not a supported DPU op (NCEConvolutionOp, NCEMatMulOp, "
                       "NCEMaxPoolOp, NCEEltwiseOp, NCEDepthConvolutionOp), skipping");
            return;
        }
        VPU::ReduceMaxOp reduceMaxOp = nullptr;
        VPU::ReduceMinOp reduceMinOp = nullptr;
        for (auto consumerOp : nceOp.getOperation()->getUsers()) {
            if (auto reduceMax = mlir::dyn_cast<VPU::ReduceMaxOp>(consumerOp)) {
                if (reduceMaxOp) {
                    _log.trace("Multiple ReduceMax consumers found for one NCE op, skipping");
                    return;
                }
                reduceMaxOp = reduceMax;
            } else if (auto reduceMin = mlir::dyn_cast<VPU::ReduceMinOp>(consumerOp)) {
                if (reduceMinOp) {
                    _log.trace("Multiple ReduceMin consumers found for one NCE op, skipping");
                    return;
                }
                reduceMinOp = reduceMin;
            }
        }

        if (!reduceMaxOp && !reduceMinOp) {
            return;
        }

        if (reduceMaxOp && reduceMinOp) {
            _log.trace("Both ReduceMax and ReduceMin consumers found for one NCE op");
            auto reduceMaxAxes = parseIntArrayAttr<int64_t>(reduceMaxOp.getAxesValue());
            auto reduceMinAxes = parseIntArrayAttr<int64_t>(reduceMinOp.getAxesValue());
            if (reduceMaxAxes != reduceMinAxes) {
                _log.trace("Axes for ReduceMax and ReduceMin are different, can't fuse both");
                return;
            }

            const auto reduceMaxKeepDims = reduceMaxOp.getKeepDims();
            const auto reduceMinKeepDims = reduceMinOp.getKeepDims();
            if (reduceMaxKeepDims != reduceMinKeepDims) {
                _log.trace("keep_dims for ReduceMax and ReduceMin are different, can't fuse both");
                return;
            }
        }

        bool isPerTensorReduction = false;
        bool hasReduceMax = reduceMaxOp != nullptr;
        const auto keepDims = hasReduceMax ? reduceMaxOp.getKeepDims() : reduceMinOp.getKeepDims();
        if (!keepDims) {
            _log.trace("Reduce operation has keep_dims=false, skip fusion");
            return;
        }

        auto axesValue = hasReduceMax ? reduceMaxOp.getAxesValue() : reduceMinOp.getAxesValue();
        const auto axesValues = parseIntArrayAttr<int64_t>(axesValue);

        if (llvm::any_of(axesValues, [](int64_t axis) {
                return axis < 0;
            })) {
            _log.trace("Negative axis value found, skipping fusion");
            return;
        }

        const auto inType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
        const auto outType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
        if (!checkAxesForFusion(axesValues, inType, outType, isPerTensorReduction)) {
            return;
        }

        mlir::OpBuilder builder(&ctx);

        mlir::Type reduceXyMaxType;
        mlir::Type reduceXyMinType;
        SmallVector<mlir::Value> origReduceOutputs;
        if (reduceMaxOp) {
            reduceXyMaxType = reduceMaxOp.getOutput().getType();
            origReduceOutputs.push_back(reduceMaxOp.getOutput());
        }
        if (reduceMinOp) {
            reduceXyMinType = reduceMinOp.getOutput().getType();
            origReduceOutputs.push_back(reduceMinOp.getOutput());
        }
        mlir::Type reduceTensorMinMaxType = nullptr;
        if (isPerTensorReduction) {
            // TODO(E#207252): Unreachable code until we support per-tensor reduction in DPU
            reduceTensorMinMaxType =
                    hasReduceMax ? reduceMaxOp.getOutput().getType() : reduceMinOp.getOutput().getType();
            reduceXyMaxType = nullptr;
            reduceXyMinType = nullptr;
        }

        // Check if we have valid outputs to fuse after all the checks, if not skip fusion
        if (origReduceOutputs.empty()) {
            _log.trace("No valid Reduce outputs to fuse after checks, skipping fusion");
            return;
        }

        auto replaceOpUses = [&](mlir::Value oldNceOutput, mlir::Value newNceOutput,
                                 llvm::ArrayRef<mlir::Value> newReduceOutput) {
            oldNceOutput.replaceAllUsesWith(newNceOutput);
            eraseList.push_back(oldNceOutput.getDefiningOp());

            // origReduceOutputs and newReduceOutput have the same size when min and max are both fused.
            // When only one is fused, newReduceOutput[0] replaces all entries.
            for (size_t newInd = 0, i = 0; i < origReduceOutputs.size(); i++) {
                origReduceOutputs[i].replaceAllUsesWith(newReduceOutput[newInd]);
                eraseList.push_back(origReduceOutputs[i].getDefiningOp());
                if (newReduceOutput.size() > (newInd + 1)) {
                    // Increase only in case we have multiple reduce outputs
                    ++newInd;
                }
            }
        };

        auto selectReduceResult = [&](mlir::Value reduceTensorResult, mlir::Value reduceXyMaxResult,
                                      mlir::Value reduceXyMinResult) -> SmallVector<mlir::Value> {
            SmallVector<mlir::Value> reduceResults = {reduceXyMaxResult, reduceXyMinResult, reduceTensorResult};
            reduceResults.erase(std::remove_if(reduceResults.begin(), reduceResults.end(),
                                               [](mlir::Value v) {
                                                   return !v;
                                               }),
                                reduceResults.end());
            return reduceResults;
        };

        auto* parentOp = nceOp.getOperation();
        builder.setInsertionPoint(parentOp);
        const auto fusedReduceKind = (reduceMaxOp && reduceMinOp) ? "Min & Max" : (reduceMaxOp ? "Max" : "Min");

        if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(parentOp)) {
            _log.trace("Fusing Reduce{0} with NCEConvolutionOp", fusedReduceKind);

            auto newConvOp = builder.create<VPU::NCEConvolutionOp>(
                    convOp.getLoc(), convOp.getOutput().getType(), reduceXyMaxType, reduceXyMinType,
                    reduceTensorMinMaxType, convOp.getInput(), convOp.getFilter(), convOp.getWeightsTable(),
                    convOp.getWeightTableScale(), convOp.getWeightTableBias(), convOp.getWeightZeroPoints(),
                    convOp.getStridesAttr(), convOp.getPadAttr(), convOp.getPpeAttr(), convOp.getMpeEngineAttr(),
                    convOp.getRawFilterShape(), convOp.getStaticRawFilterShape(), convOp.getMultiClusterStrategyAttr(),
                    convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr(), axesValue);

            auto reduceNceOutput = selectReduceResult(newConvOp.getReduceTensorMinMax(), newConvOp.getReduceXyMax(),
                                                      newConvOp.getReduceXyMin());
            replaceOpUses(convOp.getOutput(), newConvOp.getOutput(), reduceNceOutput);

            _log.trace("Successfully fused Reduce{0} into NCEConvolutionOp", fusedReduceKind);
            return;
        }
        if (auto matMulOp = mlir::dyn_cast<VPU::NCEMatMulOp>(parentOp)) {
            _log.trace("Fusing Reduce{0} with NCEMatMulOp", fusedReduceKind);

            auto newMatMulOp = builder.create<VPU::NCEMatMulOp>(
                    matMulOp.getLoc(), matMulOp.getOutput().getType(), reduceXyMaxType, reduceXyMinType,
                    reduceTensorMinMaxType, matMulOp.getInput(), matMulOp.getWeights(), matMulOp.getWeightsTable(),
                    matMulOp.getWeightTableScale(), matMulOp.getWeightTableBias(), matMulOp.getWeightZeroPoints(),
                    matMulOp.getStridesAttr(), matMulOp.getPadAttr(), matMulOp.getPpeAttr(),
                    matMulOp.getMpeEngineAttr(), matMulOp.getRawFilterShape(), matMulOp.getStaticRawFilterShape(),
                    matMulOp.getMultiClusterStrategyAttr(), axesValue);

            auto replacementValue = selectReduceResult(newMatMulOp.getReduceTensorMinMax(),
                                                       newMatMulOp.getReduceXyMax(), newMatMulOp.getReduceXyMin());
            replaceOpUses(matMulOp.getOutput(), newMatMulOp.getOutput(), replacementValue);

            _log.trace("Successfully fused Reduce{0} into NCEMatMulOp", fusedReduceKind);
            return;
        }
        if (auto maxPoolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(parentOp)) {
            _log.trace("Fusing Reduce{0} with NCEMaxPoolOp", fusedReduceKind);

            // Create new NCEMaxPoolOp with reduce output
            auto newMaxPoolOp = builder.create<VPU::NCEMaxPoolOp>(
                    maxPoolOp->getLoc(), maxPoolOp.getOutput().getType(), reduceXyMaxType, reduceXyMinType,
                    reduceTensorMinMaxType, maxPoolOp.getInput(), maxPoolOp.getWeightsTable(),
                    maxPoolOp.getWeightTableScale(), maxPoolOp.getWeightTableBias(), maxPoolOp.getKernelSizeAttr(),
                    maxPoolOp.getStridesAttr(), maxPoolOp.getPadAttr(), maxPoolOp.getPpeAttr(),
                    maxPoolOp.getMpeEngineAttr(), maxPoolOp.getMultiClusterStrategyAttr(),
                    maxPoolOp.getOutputPaddingAttr(), maxPoolOp.getInputPaddingAttr(), maxPoolOp.getS2dd2sConfigAttr(),
                    axesValue);

            auto replacementValue = selectReduceResult(newMaxPoolOp.getReduceTensorMinMax(),
                                                       newMaxPoolOp.getReduceXyMax(), newMaxPoolOp.getReduceXyMin());
            replaceOpUses(maxPoolOp.getOutput(), newMaxPoolOp.getOutput(), replacementValue);

            _log.trace("Successfully fused Reduce{0} into NCEMaxPoolOp", fusedReduceKind);
            return;
        }
        if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(parentOp)) {
            _log.trace("Fusing Reduce{0} with NCEEltwiseOp", fusedReduceKind);

            auto newEltwiseOp = builder.create<VPU::NCEEltwiseOp>(
                    eltwiseOp->getLoc(), eltwiseOp.getOutput().getType(), reduceXyMaxType, reduceXyMinType,
                    reduceTensorMinMaxType, eltwiseOp.getInput1(), eltwiseOp.getInput2(),
                    eltwiseOp.getWeightTableScale(), eltwiseOp.getWeightTableBias(), eltwiseOp.getOpTypeAttr(),
                    eltwiseOp.getPpeAttr(), eltwiseOp.getMpeEngineAttr(), eltwiseOp.getMultiClusterStrategyAttr(),
                    eltwiseOp.getIsInplaceAttr(), eltwiseOp.getOutputPaddingAttr(), eltwiseOp.getInputPaddingAttr(),
                    axesValue);

            auto replacementValue = selectReduceResult(newEltwiseOp.getReduceTensorMinMax(),
                                                       newEltwiseOp.getReduceXyMax(), newEltwiseOp.getReduceXyMin());
            replaceOpUses(eltwiseOp.getOutput(), newEltwiseOp.getOutput(), replacementValue);

            _log.trace("Successfully fused Reduce{0} into NCEEltwiseOp", fusedReduceKind);
            return;
        }
        if (auto dwConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(parentOp)) {
            _log.trace("Fusing Reduce{0} with NCEDepthConvolutionOp", fusedReduceKind);

            auto newDwConvOp = builder.create<VPU::NCEDepthConvolutionOp>(
                    dwConvOp.getLoc(), dwConvOp.getOutput().getType(), reduceXyMaxType, reduceXyMinType,
                    reduceTensorMinMaxType, dwConvOp.getInput(), dwConvOp.getFilter(), dwConvOp.getWeightsTable(),
                    dwConvOp.getWeightTableDataPtr(), dwConvOp.getWeightTableScale(), dwConvOp.getWeightTableBias(),
                    dwConvOp.getStridesAttr(), dwConvOp.getPadAttr(), dwConvOp.getPpeAttr(),
                    dwConvOp.getMpeEngineAttr(), dwConvOp.getRawFilterShape(), dwConvOp.getStaticRawFilterShape(),
                    dwConvOp.getMultiClusterStrategyAttr(), dwConvOp.getOutputPaddingAttr(),
                    dwConvOp.getInputPaddingAttr(), axesValue);

            auto replacementValue = selectReduceResult(newDwConvOp.getReduceTensorMinMax(),
                                                       newDwConvOp.getReduceXyMax(), newDwConvOp.getReduceXyMin());
            replaceOpUses(dwConvOp.getOutput(), newDwConvOp.getOutput(), replacementValue);

            _log.trace("Successfully fused Reduce{0} into NCEDepthConvolutionOp", fusedReduceKind);
            return;
        }
    });
    llvm::SmallPtrSet<mlir::Operation*, 8> visited;
    for (auto* op : llvm::reverse(eraseList)) {
        if (op == nullptr || !visited.insert(op).second) {
            continue;
        }
        if (std::all_of(op->getResults().begin(), op->getResults().end(), [](mlir::Value result) {
                return result.use_empty();
            })) {
            op->erase();
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseReduceMinMaxToDpuPass(Logger log) {
    return std::make_unique<FuseReduceMinMaxToDpuPass>(log);
}
