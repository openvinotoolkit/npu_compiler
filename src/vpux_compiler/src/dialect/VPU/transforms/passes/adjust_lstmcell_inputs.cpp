//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <vpu/utils.h>

#include <utility>
#include <vpux/utils/core/error.hpp>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTLSTMCELLINPUTS
#define GEN_PASS_DEF_ADJUSTLSTMCELLINPUTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class AdjustLSTMCellInputs final : public mlir::OpRewritePattern<VPU::LSTMCellOp> {
public:
    AdjustLSTMCellInputs(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::LSTMCellOp>(ctx), _log(std::move(log)) {
        setDebugName("AdjustLSTMCellInputs");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LSTMCellOp lstmCellOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustLSTMCellInputs::matchAndRewrite(VPU::LSTMCellOp lstmCellOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto weights = lstmCellOp.getWeights().getDefiningOp<Const::DeclareOp>();
    auto recurrenceWeights = lstmCellOp.getRecurrenceWeights().getDefiningOp<Const::DeclareOp>();
    auto biases = lstmCellOp.getBiases().getDefiningOp<Const::DeclareOp>();

    VPUX_THROW_UNLESS(weights != nullptr && recurrenceWeights != nullptr && biases != nullptr,
                      "Only constant weights are supported for LSTMCell");

    const auto weightsShape = getShape(weights).raw();
    const auto recurrenceWeightsShape = getShape(recurrenceWeights).raw();
    const auto biasesShape = getShape(biases).raw();

    const auto weightsOrder = weights.getType().cast<NDTypeInterface>().getDimsOrder();
    const auto recurrenceWeightsOrder = recurrenceWeights.getType().cast<NDTypeInterface>().getDimsOrder();
    const auto biasesOrder = biases.getType().cast<NDTypeInterface>().getDimsOrder();

    const auto newWeightsOrder = DimsOrder::NWHC;
    const auto newBiasesOrder = DimsOrder::NCWH;

    if (weightsOrder == newWeightsOrder && recurrenceWeightsOrder == newWeightsOrder && biasesOrder == newBiasesOrder) {
        return mlir::failure();
    }

    const Shape newWeightsShape{1, 4, weightsShape[2] / 4, weightsShape[3]};
    const Shape newRecurrenceWeightsShape{1, 4, recurrenceWeightsShape[2] / 4, recurrenceWeightsShape[3]};
    const Shape newBiasesShape{1, 1, 4, biasesShape[3] / 4};

    auto newWeightsContent =
            weights.getContentAttr().transform().reshape(newWeightsShape).reorder(newWeightsOrder).get();
    auto newRecurrenceWeightsContent = recurrenceWeights.getContentAttr()
                                               .transform()
                                               .reshape(newRecurrenceWeightsShape)
                                               .reorder(newWeightsOrder)
                                               .get();
    auto newBiasesConent = biases.getContentAttr().transform().reshape(newBiasesShape).reorder(newBiasesOrder).get();

    auto newWeightsOp = rewriter.create<Const::DeclareOp>(weights.getLoc(), newWeightsContent.getType(),
                                                          std::move(newWeightsContent));
    auto newRecurrenceWeightsOp = rewriter.create<Const::DeclareOp>(
            recurrenceWeights.getLoc(), newRecurrenceWeightsContent.getType(), std::move(newRecurrenceWeightsContent));
    auto newBiasesMemPermOp =
            rewriter.create<Const::DeclareOp>(biases.getLoc(), newBiasesConent.getType(), std::move(newBiasesConent));

    rewriter.replaceOpWithNewOp<VPU::LSTMCellOp>(
            lstmCellOp, lstmCellOp.getInputData(), lstmCellOp.getInitialHiddenState(), lstmCellOp.getInitialCellState(),
            newWeightsOp, newRecurrenceWeightsOp, newBiasesMemPermOp, lstmCellOp.getHiddenSizeAttr());

    return mlir::success();
};

class AdjustLSTMCellInputsPass final : public VPU::impl::AdjustLSTMCellInputsBase<AdjustLSTMCellInputsPass> {
public:
    explicit AdjustLSTMCellInputsPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustLSTMCellInputsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<AdjustLSTMCellInputs>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(greedyPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustLSTMCellInputsPass(Logger log) {
    return std::make_unique<AdjustLSTMCellInputsPass>(log);
}
