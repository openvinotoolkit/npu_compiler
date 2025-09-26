//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_WEIGHTSQUANTFUSEDINTOTASK
#define GEN_PASS_DEF_WEIGHTSQUANTFUSEDINTOTASK
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
class WeightsQuantFusedIntoTaskPass final :
        public IE::impl::WeightsQuantFusedIntoTaskBase<WeightsQuantFusedIntoTaskPass> {
public:
    explicit WeightsQuantFusedIntoTaskPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void findWeightElementType(mlir::Operation* op, const Logger& log) {
    if (mlir::isa_and_nonnull<Const::DeclareOp>(op)) {
        const auto tensor = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
        if (const auto quantType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(tensor.getElementType())) {
            log.trace("Weights constant(WAC) has quantized element type for NCE op - {0}", op->getLoc());
        }
    } else if (mlir::isa<mlir::BlockArgument>(op->getOperand(0))) {
        auto blocArgElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
        if (blocArgElemType.isInteger(8) || blocArgElemType.isInteger(4)) {
            log.trace("Weights block argument(WAI) has quantized element type for NCE op - {0} ", op->getLoc());
        }
    } else if (IE::isPureViewOp(op)) {
        findWeightElementType(op->getOperand(0).getDefiningOp(), log);
    }
}

void WeightsQuantFusedIntoTaskPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](mlir::Operation* op) {
        if (mlir::isa<IE::ConvolutionOp, IE::MatMulOp, IE::GroupConvolutionOp>(*op)) {
            findWeightElementType(op->getOperand(1).getDefiningOp(), _log);
        }
    });
}

}  // namespace

//
// createWeightsQuantFusedIntoTaskPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createWeightsQuantFusedIntoTaskPass(Logger log) {
    return std::make_unique<WeightsQuantFusedIntoTaskPass>(log);
}
