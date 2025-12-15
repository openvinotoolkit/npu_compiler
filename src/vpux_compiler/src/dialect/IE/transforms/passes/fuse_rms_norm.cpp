//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
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
#define GEN_PASS_DECL_FUSERMSNORM
#define GEN_PASS_DEF_FUSERMSNORM
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseRMSNormPass
//

class FuseRMSNormPass final : public IE::impl::FuseRMSNormBase<FuseRMSNormPass> {
public:
    explicit FuseRMSNormPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Match pattern
// Input -> IE.Power|IE.Multiply -> IE.ReduceSum -> (optional IE.Add) -> IE.Sqrt -> IE.Divide -> IE.Multiply(Cst_Scale)
//   |                                                                                   ^
//   |                                                                                   |
//    ------------------------------------------------------------------------------------
//                     Fuses to IE.RMS with gamma = Cst_Scale/Sqrt(inputDims[axis])
// Or
// Input -> IE.Power|IE.Multiply -> IE.ReduceSum -> (optional IE.Add) -> IE.Sqrt -> IE.Divide
//   |                                                                                   ^
//   |                                                                                   |
//    ------------------------------------------------------------------------------------
//                     Fuses to IE.RMS with gamma = 1/Sqrt(inputDims[axis])
// Since (X/Sqrt(ReduceMean(X^2, axis)))*(1/Sqrt(inputDims[axis])) = X/Sqrt(ReduceSum(X^2, axis))

mlir::Operation* getPowerOp(mlir::Operation* op) {
    // Check the case of x^2
    auto powerOp = mlir::dyn_cast_or_null<IE::PowerOp>(op);
    if (powerOp != nullptr && powerOp->hasOneUse()) {
        return powerOp;
    }

    // Check the case of x*x
    auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(op);
    if (multiplyOp != nullptr && multiplyOp.getInput1() == multiplyOp.getInput2() && multiplyOp->hasOneUse()) {
        return multiplyOp;
    }

    return nullptr;
}

mlir::Operation* getSqrtAndDivideOps(mlir::Operation* op) {
    // Check the case of 1/Sqrt(x)
    const auto sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(op);
    if (sqrtOp != nullptr && sqrtOp->hasOneUse()) {
        auto divideOp = mlir::dyn_cast_or_null<IE::DivideOp>(*sqrtOp->getUsers().begin());
        if (divideOp != nullptr) {
            return divideOp;
        }
    }

    // Check the case of x^(-0.5)
    auto powerOp = mlir::dyn_cast_or_null<IE::PowerOp>(op);
    if (powerOp != nullptr && powerOp->hasOneUse()) {
        auto exponent = IE::getExponentSplatVal(powerOp);
        if (exponent.has_value() && isFloatEqual(exponent.value(), -0.5)) {
            return powerOp;
        }
    }

    return nullptr;
}

mlir::FailureOr<int64_t> getReducedSize(ArrayRef<int64_t> reduceAxes, vpux::ShapeRef reduceInputShape) {
    // Calculate shape size of reduced axes
    int64_t reduceSize = 1;
    for (const auto& axis : reduceAxes | indexed) {
        // Make sure the gamma related to reduce axes should be the same as the input width
        if (axis.value() != (int64_t)(reduceInputShape.size() - reduceAxes.size() + axis.index())) {
            return mlir::failure();
        }
        reduceSize *= reduceInputShape[Dim(axis.value())];
    }

    return reduceSize;
}

mlir::FailureOr<float> getEpsilon(mlir::Operation* op) {
    if (auto clampOp = mlir::dyn_cast_or_null<IE::ClampOp>(op)) {
        const auto clampMin = clampOp.getMin().convertToDouble();
        const auto clampMax = clampOp.getMax().convertToDouble();

        // Skip match if Clamp isn't single-sided with positive low end. This ensures fusion only happens
        // if the Clamp is effectively performing a max(x, epsilon) operation, which can be fused into an
        // RMS_Norm kernel without affecting accuracy. Other Clamp forms cannot be fused, or are invalid.
        const double almostMax = 0.9;
        if (clampMax < almostMax * static_cast<double>(std::numeric_limits<type::float16>::max()) || clampMin < 0.0) {
            return mlir::failure();
        }
        return static_cast<float>(clampMin);
    } else if (mlir::isa_and_nonnull<IE::AddOp, IE::MaximumOp>(op)) {
        auto epsilonConstOp = mlir::isa_and_nonnull<Const::DeclareOp>(op->getOperand(0).getDefiningOp())
                                      ? op->getOperand(0).getDefiningOp<Const::DeclareOp>()
                                      : op->getOperand(1).getDefiningOp<Const::DeclareOp>();
        if (epsilonConstOp == nullptr) {
            return mlir::failure();
        }

        const auto maybeEpsilonVal = Const::getSplatValue<float>(epsilonConstOp);
        if (mlir::failed(maybeEpsilonVal)) {
            return mlir::failure();
        }
        return maybeEpsilonVal.value();
    }
    return mlir::failure();
}

// Create gamma
mlir::Value createGamma(mlir::OpBuilder& builder, mlir::Operation* op, int64_t size, float gammaScale) {
    const float weightData = 1.0f / gammaScale;
    const auto dataStorageType = mlir::RankedTensorType::get({size}, mlir::Float32Type::get(op->getContext()));
    const auto constLoc = appendLoc(op->getLoc(), "_const");
    return Const::createConst(builder, constLoc, dataStorageType, ArrayRef(weightData));
}

void isReduceSumPattern(mlir::Operation* maybePowerOp, IE::ReduceSumOp reduceSumOp, mlir::MLIRContext& ctx,
                        vpux::Logger /*_log*/) {
    auto powerOp = getPowerOp(maybePowerOp);
    if (powerOp == nullptr) {
        return;
    }
    if (reduceSumOp == nullptr || !reduceSumOp->hasOneUse()) {
        return;
    }

    auto powerInput = powerOp->getOperand(0);
    auto powerInputShape = getShape(powerInput);
    auto epsilon = std::numeric_limits<type::float16>::smallest_mixed_precision_eps;

    auto sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(*reduceSumOp->getUsers().begin());
    if (sqrtOp == nullptr) {
        auto preventDivByZeroOp = *reduceSumOp->getUsers().begin();
        if (!preventDivByZeroOp->hasOneUse()) {
            return;
        }
        auto epsilonResult = getEpsilon(preventDivByZeroOp);
        if (mlir::failed(epsilonResult)) {
            return;
        }
        epsilon = epsilonResult.value();

        sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(*preventDivByZeroOp->getUsers().begin());
        if (sqrtOp == nullptr) {
            return;
        }
    }

    if (!sqrtOp->hasOneUse()) {
        return;
    }

    const auto epsilonAttr = getFPAttr(&ctx, epsilon);

    auto divideOp = mlir::dyn_cast_or_null<IE::DivideOp>(*sqrtOp->getUsers().begin());
    if (divideOp == nullptr) {
        return;
    }
    auto skipFqIfPresent = [](mlir::Operation* op) -> mlir::Operation* {
        if (!mlir::isa_and_nonnull<IE::FakeQuantizeOp>(op)) {
            return op;
        }
        if (!op->hasOneUse()) {
            return nullptr;
        }
        return *op->getUsers().begin();
    };
    auto skipReshapeIfPresent = [](mlir::Operation* op) -> mlir::Operation* {
        if (!mlir::isa_and_nonnull<IE::AffineReshapeOp>(op)) {
            return op;
        }
        if (!op->hasOneUse()) {
            return nullptr;
        }
        return *op->getUsers().begin();
    };

    // Get shape size of reduced axes
    const auto reduceAxes = parseIntArrayAttr<int64_t>(reduceSumOp.getAxesValueAttr());
    auto reduceSumInputShape = getShape(reduceSumOp->getOperand(0));
    auto reduceSizeResult = getReducedSize(reduceAxes, reduceSumInputShape);
    if (mlir::failed(reduceSizeResult)) {
        return;
    }
    int64_t reduceSize = reduceSizeResult.value();
    float gammaScale = sqrtf(reduceSize);

    mlir::Operation* opBeforeScale = divideOp;
    // In case the first input of the DivideOp is 1
    auto divideInput = mlir::dyn_cast_or_null<Const::DeclareOp>(divideOp.getInput1().getDefiningOp());
    if (divideInput != nullptr && divideInput.getContent().getSplatValue<int64_t>() == 1) {
        if (!divideOp->hasOneUse()) {
            return;
        }
        auto selfMultiplyOp =
                mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(*divideOp->getUsers().begin()));
        if (selfMultiplyOp == nullptr) {
            return;
        }

        opBeforeScale = selfMultiplyOp;
    }

    if (opBeforeScale->getOperand(0) != powerInput && opBeforeScale->getOperand(1) != powerInput) {
        return;
    }

    auto builder = mlir::OpBuilder(opBeforeScale);

    auto scaleMultiplyOp =
            opBeforeScale->hasOneUse()
                    ? mlir::dyn_cast_or_null<IE::MultiplyOp>(skipFqIfPresent(*opBeforeScale->getUsers().begin()))
                    : nullptr;

    // Process the pattern of "DivideOp + AffineReshapOp"
    if (divideOp->hasOneUse() && mlir::isa_and_nonnull<IE::AffineReshapeOp>(*divideOp->getUsers().begin()) &&
        scaleMultiplyOp == nullptr) {
        mlir::Value gamma = createGamma(builder, opBeforeScale, reduceSize, gammaScale);

        // Reshape Input
        auto reduceSumOutputShape = getShape(reduceSumOp->getResult(0));
        const SmallVector<int64_t> newInShape = {1, 1, reduceSumOutputShape[Dim(reduceSumOutputShape.size() - 1)],
                                                 reduceSize};

        auto inReshapeOp =
                builder.create<IE::ReshapeOp>(appendLoc(powerOp->getLoc(), "_in_reshape"), powerOp->getOperand(0),
                                              nullptr, false, getIntArrayAttr(&ctx, newInShape));

        // Create RMSOp
        auto rmsOp = builder.create<IE::RMSOp>(appendLoc(powerOp->getLoc(), "_rms"), inReshapeOp, gamma, epsilonAttr);

        // Reshape Output
        auto outReshapeOp =
                builder.create<IE::ReshapeOp>(appendLoc(powerOp->getLoc(), "_out_reshape"), rmsOp->getResult(0),
                                              nullptr, false, getIntArrayAttr(&ctx, powerInputShape));

        opBeforeScale->replaceAllUsesWith(outReshapeOp);

        return;
    }

    mlir::Operation* replaceOp = opBeforeScale;
    if (scaleMultiplyOp != nullptr) {
        if (Const::DeclareOp constOp = scaleMultiplyOp.getOperand(1).getDefiningOp<Const::DeclareOp>()) {
            Const::Content constantContent = constOp.getContent();
            if (constantContent.isSplat() && constantContent.getSplatValue<float>() != 0.0f) {
                // Fold scaleMultiply Op into RMSOp gamma
                gammaScale /= constantContent.getSplatValue<float>();
                replaceOp = scaleMultiplyOp;
                builder = mlir::OpBuilder(replaceOp);
            }
        }
    }

    mlir::Value gamma = createGamma(builder, replaceOp, reduceSize, gammaScale);

    // Create RMSOp
    auto rmsOp =
            builder.create<IE::RMSOp>(appendLoc(powerOp->getLoc(), "_rms"), powerOp->getOperand(0), gamma, epsilonAttr);

    replaceOp->replaceAllUsesWith(rmsOp);
}

mlir::Value createGammaWithShape(mlir::OpBuilder& builder, mlir::Operation* op, ShapeRef shape) {
    const float weightData = 1.0f;
    const auto dataStorageType = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(op->getContext()));
    const auto constLoc = appendLoc(op->getLoc(), "_const");
    return Const::createConst(builder, constLoc, dataStorageType, ArrayRef(weightData));
};

// Match the pattern:
// Input -> IE.Power -> IE.ReduceSum -> IE.Add -> IE.Sqrt -> IE.Divide
//   |                                                          ^
//   |                                                          |
//    -----------------------------------------------------------
bool matchPatternEndsWithDivideOp(mlir::Operation* lastOp, mlir::Operation* headOp) {
    auto divideOp = mlir::dyn_cast<IE::DivideOp>(lastOp);
    if (divideOp == nullptr) {
        return false;
    }

    return divideOp.getInput1() == headOp->getOperand(0);
}

// Determines if the specified operation is a DivideOp that calculates the reciprocal of its input
bool isValidReciprocalDivideOp(mlir::Operation* op) {
    auto divideOp = mlir::dyn_cast<IE::DivideOp>(op);
    if (divideOp == nullptr) {
        return true;
    }

    // The valid DivideOp's 1st input should be a constant with value 1
    auto divideInput = mlir::dyn_cast_or_null<Const::DeclareOp>(divideOp.getInput1().getDefiningOp());
    if (divideInput == nullptr) {
        return false;
    }

    return divideInput.getContent().getSplatValue<int64_t>() == 1;
}

IE::RMSOp createRMSOp(mlir::OpBuilder& builder, mlir::Operation* headOp, mlir::Value gamma, int64_t layerSize,
                      mlir::FloatAttr epsilonAttr) {
    auto gammaRank = mlir::cast<vpux::NDTypeInterface>(gamma.getType()).getRank();
    if (gammaRank != 1) {
        auto reshapeOp =
                builder.create<IE::ReshapeOp>(gamma.getLoc(), gamma, nullptr, false,
                                              getIntArrayAttr(headOp->getContext(), SmallVector<int64_t>({layerSize})));
        gamma = reshapeOp;
    }
    auto rmsOp =
            builder.create<IE::RMSOp>(appendLoc(headOp->getLoc(), "_rms"), headOp->getOperand(0), gamma, epsilonAttr);

    return rmsOp;
}

//
// safeRunOnFunc
//

// Match pattern
// Input -> IE.Power -> IE.ReduceMean -> IE.Add (epsilon) -> IE.Sqrt -> IE.Divide -> IE.Multiply -> IE.Multiply (gamma)
//   |                                                                                   ^
//   |                                                                                   |
//    -----------------------------------------------------------------------------------
// Or
// Input -> IE.Convert -> IE.Power -> IE.ReduceMean -> IE.Add (epsilon) -> IE.Sqrt -> IE.Divide -> IE.Multiply ->
// IE.Convert -> IE.Multiply (gamma)
//   |                                                                                                  ^
//   |                                                                                                  |
//    --------------------------------------------------------------------------------------------------
// Convert to RMS
// RMS = x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
void FuseRMSNormPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    func->walk([&](mlir::Operation* op) {
        auto powerOp = getPowerOp(op);
        if (powerOp == nullptr) {
            return;
        }
        _log.trace("Got PowerOp {0} at {1}", powerOp->getName(), powerOp->getLoc());

        // make sure the op's output has one non-one dimension
        // return the dimension size or 0
        auto getSingleDimSize = [](mlir::Operation* op) {
            auto outputShape = getShape(op->getResult(0));
            auto nonOneDim = getNonOneDim(outputShape);
            if (nonOneDim.empty()) {
                return static_cast<int64_t>(0);
            }
            return outputShape[nonOneDim.back()];
        };
        const auto layerSize = getSingleDimSize(powerOp);
        if (layerSize == 0) {
            _log.nest().trace("PowerOp does not have one single non-one dim");
            return;
        }
        auto reduceMeanOp = mlir::dyn_cast_or_null<IE::ReduceMeanOp>(*powerOp->getUsers().begin());
        if (reduceMeanOp == nullptr || !reduceMeanOp->hasOneUse()) {
            const auto reduceSumOp = mlir::dyn_cast_or_null<IE::ReduceSumOp>(*powerOp->getUsers().begin());
            isReduceSumPattern(powerOp, reduceSumOp, ctx, _log);
            return;
        }

        auto headOp = powerOp;

        mlir::Operation* divideOp = getSqrtAndDivideOps(*reduceMeanOp->getUsers().begin());
        auto epsilon = std::numeric_limits<type::float16>::smallest_mixed_precision_eps;

        if (divideOp == nullptr) {
            auto preventDivByZeroOp = *reduceMeanOp->getUsers().begin();
            if (!preventDivByZeroOp->hasOneUse()) {
                return;
            }
            auto epsilonResult = getEpsilon(preventDivByZeroOp);
            if (mlir::failed(epsilonResult)) {
                return;
            }
            epsilon = epsilonResult.value();
            divideOp = getSqrtAndDivideOps(*preventDivByZeroOp->getUsers().begin());
            if (divideOp == nullptr) {
                return;
            }
        }

        // Get shape size of reduced axes
        const auto reduceAxes = parseIntArrayAttr<int64_t>(reduceMeanOp.getAxesValueAttr());
        auto reduceMeanInputShape = getShape(reduceMeanOp->getOperand(0));
        auto reduceSizeResult = getReducedSize(reduceAxes, reduceMeanInputShape);
        if (mlir::failed(reduceSizeResult)) {
            return;
        }
        int64_t reduceSize = reduceSizeResult.value();
        if (layerSize != reduceSize) {
            return;
        }
        const auto epsilonAttr = getFPAttr(&ctx, epsilon);

        if (matchPatternEndsWithDivideOp(divideOp, headOp)) {
            _log.trace("Match the pattern ends with DivideOp");
            auto builder = mlir::OpBuilder(divideOp);

            // Create default gamma
            auto gamma = createGamma(builder, divideOp, reduceSize, 1.0f);

            auto rmsOp = createRMSOp(builder, headOp, gamma, layerSize, epsilonAttr);

            divideOp->replaceAllUsesWith(rmsOp);
            return;
        }

        // Not the pattern ends with DivideOp, multi-users is not allowed
        if (!divideOp->hasOneUse()) {
            return;
        }

        if (!isValidReciprocalDivideOp(divideOp)) {
            return;
        }

        auto multiplyOp1 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*divideOp->getUsers().begin());
        if (multiplyOp1 == nullptr || !multiplyOp1->hasOneUse() || getSingleDimSize(multiplyOp1) != layerSize) {
            return;
        }

        auto multiplyOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*multiplyOp1->getUsers().begin());
        auto convertOp2 = mlir::dyn_cast_or_null<IE::ConvertOp>(*multiplyOp1->getUsers().begin());
        auto convertOp1 = mlir::dyn_cast_or_null<IE::ConvertOp>(powerOp->getOperand(0).getDefiningOp());
        if (multiplyOp2 == nullptr) {
            // try to match convert case
            // Convert -> Power -> .... -> Multiply1 -> Convert -> Multiply2
            if (convertOp1 != nullptr && convertOp2 != nullptr) {
                multiplyOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*convertOp2->getUsers().begin());
                if (multiplyOp2 != nullptr) {
                    headOp = convertOp1.getOperation();
                }
            }
        }
        auto needCreateGamma = multiplyOp2 == nullptr || getSingleDimSize(multiplyOp2) != layerSize;
        auto builder = needCreateGamma ? mlir::OpBuilder(multiplyOp1) : mlir::OpBuilder(multiplyOp2);
        mlir::Value gamma;
        if (needCreateGamma) {
            auto gammaShape = getShape(multiplyOp1->getResult(0));
            gamma = createGammaWithShape(builder, multiplyOp1, gammaShape);
        } else {
            gamma = multiplyOp2.getInput1().getDefiningOp() == multiplyOp1 ||
                                    (convertOp2 != nullptr && multiplyOp2.getInput1().getDefiningOp() == convertOp2)
                            ? multiplyOp2.getInput2()
                            : multiplyOp2.getInput1();
            auto gammaDims = getShape(gamma);
            auto gammaWidth = gammaDims[Dim(gammaDims.size() - 1)];

            auto inputDims = getShape(powerOp->getOperand(0));
            auto inputWidth = inputDims[Dim(inputDims.size() - 1)];

            // Gamma should have only one non-one dimension, and the width should be the same as the input width
            if (inputWidth != gammaWidth || getNonOneDim(gammaDims).size() != 1) {
                return;
            }
        }

        _log.trace("RMS pattern matched");
        auto rmsOp = createRMSOp(builder, headOp, gamma, layerSize, epsilonAttr);
        if (needCreateGamma) {
            multiplyOp1->replaceAllUsesWith(rmsOp);
        } else {
            multiplyOp2->replaceAllUsesWith(rmsOp);
        }
    });
}

}  // namespace

//
// createFuseRMSNormPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseRMSNormPass(Logger log) {
    return std::make_unique<FuseRMSNormPass>(log);
}
