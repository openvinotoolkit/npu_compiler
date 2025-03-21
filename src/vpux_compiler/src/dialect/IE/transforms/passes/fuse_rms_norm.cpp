//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
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

// Input -> IE.Power -> IE.ReduceSum -> IE.Sqrt -> IE.Divide -> IE.Multiply (sqrt(inputDims[axis]))
//   |                                                                ^
//   |                                                                |
//    -----------------------------------------------------------------

bool compareWithTolerance(float num1, float num2, float tolerance = 0.1f) {
    return std::abs(num1 - num2) <= tolerance;
}

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
        const auto divideOp = mlir::dyn_cast_or_null<IE::DivideOp>(*sqrtOp->getUsers().begin());
        if (divideOp != nullptr && divideOp->hasOneUse()) {
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

void isReduceSumPattern(mlir::Operation* maybePowerOp, IE::ReduceSumOp reduceSumOp, mlir::MLIRContext& ctx,
                        vpux::Logger /*_log*/) {
    auto powerOp = getPowerOp(maybePowerOp);
    if (powerOp == nullptr) {
        return;
    }
    if (reduceSumOp == nullptr || !reduceSumOp->hasOneUse()) {
        return;
    }
    const auto sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(*reduceSumOp->getUsers().begin());
    if (sqrtOp == nullptr || !sqrtOp->hasOneUse()) {
        return;
    }
    const auto divideOp = mlir::dyn_cast_or_null<IE::DivideOp>(*sqrtOp->getUsers().begin());
    if (divideOp == nullptr || !divideOp->hasOneUse()) {
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
    auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(skipFqIfPresent(*divideOp->getUsers().begin()));
    if (multiplyOp == nullptr || !multiplyOp->hasOneUse()) {
        return;
    }

    Const::DeclareOp constOp = multiplyOp.getOperand(1).getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return;
    }
    Const::Content constantContent = constOp.getContent();
    if (!constantContent.isSplat()) {
        return;
    }
    auto constantValue = constantContent.getSplatValue<float>();
    auto inputDims = getShape(powerOp->getOperand(0));
    auto inputWidth = inputDims[Dim(inputDims.size() - 1)];
    if (!compareWithTolerance(constantValue, sqrt(inputWidth))) {
        return;
    }

    // Create default gamma
    mlir::Value gamma;
    auto builder = mlir::OpBuilder(multiplyOp);
    const float weightData = 1.0f;
    const auto dataStorageType =
            mlir::RankedTensorType::get({inputWidth}, mlir::Float32Type::get(powerOp->getContext()));
    const auto constLoc = appendLoc(powerOp->getLoc(), "_const");
    gamma = Const::createConst(builder, constLoc, dataStorageType, ArrayRef(weightData));

    float epsilon = 0.000000001f;
    const auto epsilonAttr = getFPAttr(&ctx, epsilon);
    auto gammaRank = mlir::cast<vpux::NDTypeInterface>(gamma.getType()).getRank();
    if (gammaRank != 1) {
        auto ReshapeOp = builder.create<IE::ReshapeOp>(gamma.getLoc(), gamma, nullptr, false,
                                                       getIntArrayAttr(&ctx, SmallVector<int64_t>({inputWidth})));
        gamma = ReshapeOp;
    }
    auto rmsOp =
            builder.create<IE::RMSOp>(appendLoc(powerOp->getLoc(), "_rms"), powerOp->getOperand(0), gamma, epsilonAttr);
    multiplyOp->replaceAllUsesWith(rmsOp);
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
        const auto reduceMeanOp = mlir::dyn_cast_or_null<IE::ReduceMeanOp>(*powerOp->getUsers().begin());
        if (reduceMeanOp == nullptr || !reduceMeanOp->hasOneUse()) {
            const auto reduceSumOp = mlir::dyn_cast_or_null<IE::ReduceSumOp>(*powerOp->getUsers().begin());
            isReduceSumPattern(powerOp, reduceSumOp, ctx, _log);
            return;
        }
        auto addOp = mlir::dyn_cast_or_null<IE::AddOp>(*reduceMeanOp->getUsers().begin());
        if (addOp == nullptr || !addOp->hasOneUse()) {
            return;
        }

        float epsilon = 0.000000001f;
        auto epsilonConstOp = mlir::isa<Const::DeclareOp>(addOp.getInput1().getDefiningOp())
                                      ? addOp.getInput1().getDefiningOp<Const::DeclareOp>()
                                      : addOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (epsilonConstOp != nullptr) {
            auto epsilonContent = epsilonConstOp.getContent();
            auto epsilonArray = to_small_vector(epsilonContent.getValues<float>());
            VPUX_THROW_WHEN(epsilonArray.size() != 1, "wrong epsilon value");
            epsilon = epsilonArray[0];
        } else {
            _log.trace("use default epsilon value");
        }

        const auto divideOp = getSqrtAndDivideOps(*addOp->getUsers().begin());
        if (divideOp == nullptr) {
            return;
        }
        auto multiplyOp1 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*divideOp->getUsers().begin());
        if (multiplyOp1 == nullptr || !multiplyOp1->hasOneUse() || getSingleDimSize(multiplyOp1) != layerSize) {
            return;
        }

        auto multiplyOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*multiplyOp1->getUsers().begin());
        auto headOp = powerOp;
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
            const float weightData = 1.0f;
            auto weightShape = getShape(multiplyOp1.getOutput());
            const auto dataStorageType =
                    mlir::RankedTensorType::get(weightShape, mlir::Float32Type::get(headOp->getContext()));
            const auto constLoc = appendLoc(headOp->getLoc(), "_const");
            gamma = Const::createConst(builder, constLoc, dataStorageType, ArrayRef(weightData));
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
        const auto epsilonAttr = getFPAttr(&ctx, epsilon);
        auto gammaRank = mlir::cast<vpux::NDTypeInterface>(gamma.getType()).getRank();
        if (gammaRank != 1) {
            auto ReshapeOp = builder.create<IE::ReshapeOp>(gamma.getLoc(), gamma, nullptr, false,
                                                           getIntArrayAttr(&ctx, SmallVector<int64_t>({layerSize})));
            gamma = ReshapeOp;
        }
        auto rmsOp = builder.create<IE::RMSOp>(appendLoc(headOp->getLoc(), "_rms"), headOp->getOperand(0), gamma,
                                               epsilonAttr);
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
