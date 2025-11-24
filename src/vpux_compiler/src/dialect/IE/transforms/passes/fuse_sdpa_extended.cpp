//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/type/float16.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESDPAEXTENDED
#define GEN_PASS_DEF_FUSESDPAEXTENDED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseSDPAExtendedPass
//

class FuseSDPAExtendedPass final : public IE::impl::FuseSDPAExtendedBase<FuseSDPAExtendedPass> {
public:
    explicit FuseSDPAExtendedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
bool isMatMulAsFC(mlir::Value output, mlir::Value& input1, mlir::Value& input2) {
    auto reshapeOutOp = output.getDefiningOp<IE::ReshapeOp>();
    if (!reshapeOutOp) {
        return false;
    }
    auto fcOp = reshapeOutOp->getOperand(0).getDefiningOp<IE::FullyConnectedOp>();
    if (!fcOp) {
        return false;
    }
    auto reshapeInput1Op = fcOp->getOperand(0).getDefiningOp<IE::ReshapeOp>();
    if (!reshapeInput1Op) {
        return false;
    }
    auto reshapeInput2Op = fcOp->getOperand(1).getDefiningOp<IE::ReshapeOp>();
    if (!reshapeInput2Op) {
        return false;
    }
    input1 = reshapeInput1Op->getOperand(0);
    input2 = reshapeInput2Op->getOperand(0);
    return true;
}

void createMatMulSoftmaxMatMul(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV) {
    auto builder = mlir::OpBuilder(op);

    auto sdpaOp = builder.create<IE::SDPAExtendedOp>(appendLoc(op->getLoc(), "_sdpa_extended"), inputQ, inputK, inputV,
                                                     nullptr, nullptr, nullptr);

    op->replaceAllUsesWith(sdpaOp);
}

bool isLegalSDPAExtended(mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV) {
    auto tensorTypeQ = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto tensorTypeK = mlir::cast<NDTypeInterface>(inputK.getType());
    auto tensorTypeV = mlir::cast<NDTypeInterface>(inputV.getType());
    const auto rankQ = tensorTypeQ.getRank();
    const auto rankK = tensorTypeK.getRank();
    const auto rankV = tensorTypeV.getRank();
    auto shapeQ = tensorTypeQ.getShape().raw();
    auto shapeK = tensorTypeK.getShape().raw();
    if (rankQ < 2 || rankQ > 4) {
        return false;
    }
    if (rankK < 2 || rankK > 4) {
        return false;
    }
    if (rankV < 2 || rankV > 4) {
        return false;
    }

    auto batchSize = shapeQ[0] * shapeQ[1];
    auto L = shapeQ[rankQ - 2];
    auto S = shapeK[rankK - 2];

    // experimental restriction based on current shave implementation
    if ((batchSize < 24) || (L < 128) || (S < 128) || (L > 256) || (S > 256)) {
        return false;
    }

    return true;
}

void FuseSDPAExtendedPass::safeRunOnFunc() {
    auto func = getOperation();
    // Detect MatMul - (Reshapes (+ ReLU)) - Softmax - (Reshapes (+ ReLU)) - MatMul
    func->walk([&](IE::MatMulOp matMul2Op) {
        auto skipOptionalOpsIfPresent = [](mlir::Operation* op) -> mlir::Operation* {
            // No Reshapes
            if (!mlir::isa_and_nonnull<IE::ReshapeOp>(op)) {
                return op;
            }
            auto nextOp = op->getOperand(0).getDefiningOp();
            if (!mlir::isa_and_nonnull<IE::ReshapeOp>(nextOp)) {
                if (!mlir::isa_and_nonnull<IE::ReLUOp>(nextOp)) {
                    return nextOp;
                }
                auto nextNextOp = nextOp->getOperand(0).getDefiningOp();
                if (!mlir::isa_and_nonnull<IE::ReshapeOp>(nextNextOp)) {
                    return nextNextOp;
                }
                return nextNextOp->getOperand(0).getDefiningOp();
            }
            if (!op->hasOneUse() || !nextOp->hasOneUse()) {
                return nullptr;
            }
            // Double Reshapes
            return nextOp->getOperand(0).getDefiningOp();
        };
        auto softmaxOp = mlir::dyn_cast_or_null<IE::SoftMaxOp>(
                skipOptionalOpsIfPresent(matMul2Op.getOperand(0).getDefiningOp()));
        if (!softmaxOp) {
            return;
        }
        auto matMul1Op = mlir::dyn_cast_or_null<IE::MatMulOp>(
                skipOptionalOpsIfPresent(softmaxOp->getOperand(0).getDefiningOp()));
        if (!matMul1Op) {
            return;
        }

        // Create SDPAExtended Operator
        mlir::Value inputQ = matMul1Op->getOperand(0);
        mlir::Value inputK = matMul1Op->getOperand(1);
        mlir::Value inputV = matMul2Op.getOperand(1);
        if ((!isLegalSDPAExtended(inputQ, inputK, inputV))) {
            return;
        }
        createMatMulSoftmaxMatMul(matMul2Op, inputQ, inputK, inputV);
    });

    // Detect [Reshape - FullyConnected - Reshape] - Softmax - [Reshape - FullyConnected - Reshape]
    func->walk([&](IE::ReshapeOp reshape0Op) {
        auto fc0Op = reshape0Op.getOperand(0).getDefiningOp<IE::FullyConnectedOp>();
        if (!fc0Op) {
            return;
        }
        auto reshape1Op = fc0Op->getOperand(0).getDefiningOp<IE::ReshapeOp>();
        if (!reshape1Op) {
            return;
        }
        auto reshapeVOp = fc0Op->getOperand(1).getDefiningOp<IE::ReshapeOp>();
        if (!reshapeVOp) {
            return;
        }
        mlir::Value inputV = reshapeVOp->getOperand(0);
        mlir::Value inputK = mlir::Value();
        mlir::Value inputQ = mlir::Value();
        auto softmaxOp = reshape1Op->getOperand(0).getDefiningOp<IE::SoftMaxOp>();
        if (!softmaxOp) {
            return;
        }
        if (!isMatMulAsFC(softmaxOp->getOperand(0), inputQ, inputK)) {
            return;
        }

        if ((!isLegalSDPAExtended(inputQ, inputK, inputV))) {
            return;
        }
        createMatMulSoftmaxMatMul(reshape0Op, inputQ, inputK, inputV);
    });
}

}  // namespace

//
// createFuseSDPAExtendedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSDPAExtendedPass(Logger log) {
    return std::make_unique<FuseSDPAExtendedPass>(log);
}
