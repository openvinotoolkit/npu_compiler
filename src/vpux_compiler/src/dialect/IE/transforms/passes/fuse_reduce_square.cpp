//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
namespace vpux::IE {

#define GEN_PASS_DECL_FUSEREDUCESQUARE
#define GEN_PASS_DEF_FUSEREDUCESQUARE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseReduceSquarePass
//

class FuseReduceSquarePass final : public IE::impl::FuseReduceSquareBase<FuseReduceSquarePass> {
public:
    explicit FuseReduceSquarePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Operation* getPowerOp(mlir::Operation* op) {
    // detect Pow(x,2) or Multiply(x,x)
    if (auto power = mlir::dyn_cast<IE::PowerOp>(op)) {
        auto constOp = power.getInput2().getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
            return nullptr;
        }
        const auto coefContent = constOp.getContent();
        const auto coefValue = coefContent.getSplatValue<double>();
        return (coefValue == 2.0) ? op : nullptr;
    } else if (auto mul = mlir::dyn_cast<IE::MultiplyOp>(op)) {
        return mul.getInput1() == mul.getInput2() ? op : nullptr;
    }
    return nullptr;
}

std::optional<mlir::FloatAttr> extractEpsilon(IE::AddOp addOp, mlir::OpBuilder& builder, Logger log) {
    auto maybeConstant = addOp->getOperand(0).getDefiningOp();
    auto epsilonConstOp = mlir::dyn_cast<Const::DeclareOp>(mlir::isa_and_nonnull<Const::DeclareOp>(maybeConstant)
                                                                   ? maybeConstant
                                                                   : addOp->getOperand(1).getDefiningOp());

    if (epsilonConstOp == nullptr) {
        log.trace("Epsilon operand is not a Const::DeclareOp, skipping fuse");
        return std::nullopt;
    }

    auto epsilonValue = Const::getSplatValue<float>(epsilonConstOp);
    if (mlir::failed(epsilonValue)) {
        log.trace("Failed to extract epsilon splat value, skipping fuse");
        return std::nullopt;
    }

    return getFPAttr(builder.getContext(), epsilonValue.value());
}

IE::ReduceSquareOp createReduceSquareOp(mlir::OpBuilder& builder, mlir::Value input, mlir::Type outputType,
                                        mlir::FloatAttr epsilon, mlir::ArrayAttr axesAttr, mlir::UnitAttr keepDimsAttr,
                                        mlir::IntegerAttr scaleAttr = nullptr) {
    auto loc = input.getLoc();
    return builder.create<IE::ReduceSquareOp>(appendLoc(loc, "_reduce_square"), outputType, input, epsilon, axesAttr,
                                              keepDimsAttr, scaleAttr);
}

void FuseReduceSquarePass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](mlir::Operation* op) {
        auto powerOp = getPowerOp(op);
        if (powerOp == nullptr || !powerOp->hasOneUse()) {
            _log.trace("Power op not found or has multiple uses, skipping fuse.");
            return;
        }
        auto reduceOp = *powerOp->getUsers().begin();
        auto reduceMeanOp = mlir::dyn_cast<IE::ReduceMeanOp>(reduceOp);
        auto reduceSumOp = mlir::dyn_cast<IE::ReduceSumOp>(reduceOp);

        if (reduceMeanOp == nullptr && reduceSumOp == nullptr) {
            _log.trace("ReduceMean/ReduceSum op not found, skipping fuse.");
            return;
        }
        if (!reduceOp->hasOneUse()) {
            _log.trace("Reduce op has multiple uses, skipping fuse.");
            return;
        }

        // Extract common attributes based on reduce type
        mlir::ArrayAttr axesAttr;
        mlir::UnitAttr keepDimsAttr;
        mlir::Value reduceInput;

        if (reduceMeanOp) {
            axesAttr = reduceMeanOp.getAxesValue();
            keepDimsAttr = reduceMeanOp.getKeepDimsAttr();
            reduceInput = reduceMeanOp.getInput();
        } else {
            axesAttr = reduceSumOp.getAxesValue();
            keepDimsAttr = reduceSumOp.getKeepDimsAttr();
            reduceInput = reduceSumOp.getInput();
        }

        auto userOp = *reduceOp->getUsers().begin();
        auto builder = mlir::OpBuilder(reduceOp);
        mlir::FloatAttr epsilonAttr = nullptr;

        mlir::Operation* sqrtOp = userOp;

        if (auto addOp = mlir::dyn_cast<IE::AddOp>(userOp)) {
            // Epsilon (Add) is only supported for ReduceMean pattern, not for ReduceSum
            if (reduceSumOp != nullptr) {
                _log.trace("Add op (epsilon) is not supported for ReduceSum pattern, skipping fuse.");
                return;
            }

            if (!addOp->hasOneUse()) {
                _log.trace("Add op has multiple uses, skipping fuse.");
                return;
            }

            auto extractedEpsilon = extractEpsilon(addOp, builder, _log);
            if (!extractedEpsilon.has_value()) {
                _log.trace("Failed to extract epsilon, skipping fuse.");
                return;
            }
            epsilonAttr = extractedEpsilon.value();
            const auto axes = parseIntArrayAttr<int64_t>(axesAttr);
            const auto inputRank = mlir::cast<vpux::NDTypeInterface>(reduceInput.getType()).getRank();

            if (axes.size() != 1 || axes[0] != inputRank - 1) {
                _log.trace("Axis is not innermost. Skipping fuse.");
                return;
            }

            sqrtOp = *addOp->getUsers().begin();
        }

        // Accept either IE.Sqrt or IE.Power(x, 0.5) as the final sqrt-like step.
        mlir::Operation* sqrtLikeOp = nullptr;
        if (mlir::isa<IE::SqrtOp>(sqrtOp)) {
            sqrtLikeOp = sqrtOp;
        } else if (auto powOp = mlir::dyn_cast<IE::PowerOp>(sqrtOp)) {
            auto exponent = IE::getExponentSplatVal(powOp);
            if (exponent.has_value() && isFloatEqual(exponent.value(), 0.5f)) {
                sqrtLikeOp = sqrtOp;
            }
        }
        if (sqrtLikeOp == nullptr) {
            _log.trace("Sqrt-like op not found, skipping fuse.");
            return;
        }

        if (reduceMeanOp) {
            auto reduceSquareOp =
                    createReduceSquareOp(builder, powerOp->getOperand(0), sqrtLikeOp->getResult(0).getType(),
                                         epsilonAttr, axesAttr, keepDimsAttr);
            sqrtLikeOp->replaceAllUsesWith(reduceSquareOp);
            _log.trace("[FuseReduceSquare] ReduceMean pattern matched");
        } else {
            const auto inputElementType = mlir::cast<vpux::NDTypeInterface>(reduceInput.getType()).getElementType();
            // For fp32 inputs with ReduceSum pattern, map to ReduceL2 instead of ReduceSquare
            if (inputElementType.isF32()) {
                auto reduceL2Op = builder.create<IE::ReduceL2Op>(appendLoc(reduceInput.getLoc(), "_reduce_l2"),
                                                                 sqrtLikeOp->getResult(0).getType(),
                                                                 powerOp->getOperand(0), axesAttr, keepDimsAttr);
                sqrtLikeOp->replaceAllUsesWith(reduceL2Op);
                _log.trace("[FuseReduceSquare] ReduceSum fp32 pattern matched, replaced with ReduceL2");
                return;
            }

            // ReduceSum pattern: only single innermost axis is supported
            const auto axes = parseIntArrayAttr<int64_t>(axesAttr);
            const auto inputRank = mlir::cast<vpux::NDTypeInterface>(reduceInput.getType()).getRank();
            if (axes.size() != 1 || axes[0] != inputRank - 1) {
                _log.trace("ReduceSum pattern: axis is not single innermost. Skipping fuse.");
                return;
            }
            const auto inputShape = getShape(reduceInput);
            const auto innermostDim = inputShape[Dim(inputShape.size() - 1)];
            const auto scaleAttr = getIntAttr(builder.getContext(), static_cast<int64_t>(innermostDim));

            auto reduceSquareOp =
                    createReduceSquareOp(builder, powerOp->getOperand(0), sqrtLikeOp->getResult(0).getType(),
                                         epsilonAttr, axesAttr, keepDimsAttr, scaleAttr);
            sqrtLikeOp->replaceAllUsesWith(reduceSquareOp);
            _log.trace("[FuseReduceSquare] ReduceSum pattern matched");
        }
    });
}

}  // namespace

//
// createFuseReduceSquarePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseReduceSquarePass(Logger log) {
    return std::make_unique<FuseReduceSquarePass>(log);
}
