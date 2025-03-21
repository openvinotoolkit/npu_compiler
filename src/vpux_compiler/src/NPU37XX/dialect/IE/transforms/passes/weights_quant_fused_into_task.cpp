//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_WEIGHTSQUANTFUSEDINTOTASK
#define GEN_PASS_DEF_WEIGHTSQUANTFUSEDINTOTASK
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {
class WeightsQuantFusedIntoTaskPass final :
        public IE::arch37xx::impl::WeightsQuantFusedIntoTaskBase<WeightsQuantFusedIntoTaskPass> {
public:
    explicit WeightsQuantFusedIntoTaskPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void findWeightElementType(mlir::Operation* op, const Logger& log) {
    if (mlir::isa_and_nonnull<Const::DeclareOp>(op)) {
        const auto tensor = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        if (const auto quantType = tensor.getElementType().dyn_cast_or_null<mlir::quant::QuantizedType>()) {
            log.trace("Weights constant(WAC) has quantized element type for NCE op - {0}", op->getLoc());
        }
    } else if (mlir::isa<mlir::BlockArgument>(op->getOperand(0))) {
        auto blocArgElemType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
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

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createWeightsQuantFusedIntoTaskPass(Logger log) {
    return std::make_unique<WeightsQuantFusedIntoTaskPass>(log);
}
