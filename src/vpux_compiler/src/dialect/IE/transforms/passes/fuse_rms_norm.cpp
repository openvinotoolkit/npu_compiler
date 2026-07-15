//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <llvm/ADT/STLExtras.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
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
// Or
// Input -> IE.Power|IE.Multiply -> IE.ReduceSum -> IE.Multiply(Cst_Scale) -> (optional IE.Add)
//   |                                                                                 |
//   |                               --------------------------------------------------
//   |                               |
//   |                               v
//   |                     (optional IE.AffineReshape) -> IE.Power(-0.5) -> IE.Multiply
//   |                                                                           ^
//   |                                                                           |
//    ----------------------------------------------------------------------------
//                     Fuses to IE.RMS with gamma = 1/Sqrt(inputDims[axis]*Cst_Scale)
// Or
// Input -> IE.Power|IE.Multiply -> IE.ReduceSum -> IE.Multiply(Cst_Scale_1) -> (optional IE.Add)
//   |                                                                                 |
//   |                               --------------------------------------------------
//   |                               |
//   |                               v
//   |                     (optional IE.AffineReshape) -> IE.Power(-0.5) -> IE.Multiply -> IE.Multiply(Cst_Scale_2)
//   |                                                                           ^
//   |                                                                           |
//    ----------------------------------------------------------------------------
//                     Fuses to IE.RMS with gamma = Cst_Scale_2/Sqrt(inputDims[axis]*Cst_Scale_1)
// Since (X/Sqrt(ReduceMean(X^2, axis)))*(1/Sqrt(inputDims[axis])) = X/Sqrt(ReduceSum(X^2, axis))

bool isDimExpansionOp(mlir::Operation* op) {
    // Helper to check if op is a dimension expansion operation (unsqueeze-like)
    if (auto affineReshapeOp = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(op)) {
        auto inShape = getShape(affineReshapeOp.getInput());
        auto outShape = getShape(affineReshapeOp.getOutput());
        return !IE::isNotDimExpansionReshape(inShape, outShape);
    }
    return mlir::isa_and_present<IE::UnsqueezeOp>(op);
}

Const::DeclareOp getMultiplyConstOperand(mlir::Operation* op) {
    auto const1 = op->getOperand(0).getDefiningOp<Const::DeclareOp>();
    auto const2 = op->getOperand(1).getDefiningOp<Const::DeclareOp>();
    return const1 ? const1 : const2;
}

mlir::Operation* getPowerOp(mlir::Operation* op) {
    // Check the case of x^2
    auto powerOp = mlir::dyn_cast_or_null<IE::PowerOp>(op);
    if (powerOp != nullptr && powerOp->hasOneUse()) {
        return powerOp;
    }

    auto skipDimExpansionParent = [](mlir::Operation* op) -> mlir::Value {
        if (op == nullptr) {
            return mlir::Value();
        }
        if (isDimExpansionOp(op) && op->hasOneUse()) {
            return op->getOperand(0);
        }
        return op->getResult(0);
    };

    // Check the case of x*x
    auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(op);
    if (multiplyOp != nullptr && multiplyOp->hasOneUse()) {
        if (multiplyOp.getInput1() == multiplyOp.getInput2()) {
            return multiplyOp;
        } else {
            // Support mixed inputs - one is multiplyOp, another is multiplyOp + affineReshapeOp
            mlir::Value input1 = multiplyOp.getInput1();
            mlir::Value input2 = multiplyOp.getInput2();
            mlir::Value parentValue = skipDimExpansionParent(input1.getDefiningOp());
            if (parentValue && parentValue == input2) {
                return multiplyOp;
            }
            parentValue = skipDimExpansionParent(input2.getDefiningOp());
            if (parentValue && parentValue == input1) {
                return multiplyOp;
            }
        }
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
    } else if (auto selectOp = mlir::dyn_cast_or_null<IE::SelectOp>(op)) {
        // Pattern: Select(Equal(x, 0), epsilon, x)
        // Input1 (condition): Equal(x, 0)
        // Input2 (then): epsilon constant
        // Input3 (else): x (the ReduceMean output)

        // Check that condition is EqualOp
        auto equalOp = selectOp.getInput1().getDefiningOp<IE::EqualOp>();
        if (!equalOp) {
            return mlir::failure();
        }

        // Check that "then" (Input2) is a constant - this should be epsilon
        auto epsilonConstOp = selectOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (!epsilonConstOp) {
            return mlir::failure();
        }

        // Check that "else" (Input3) is NOT a constant - it should be 'x' (ReduceMean output)
        if (!selectOp.getInput3().getDefiningOp<IE::ReduceMeanOp>()) {
            return mlir::failure();
        }

        const auto epsilonVal = Const::getSplatValue<float>(epsilonConstOp);
        if (mlir::failed(epsilonVal)) {
            return mlir::failure();
        }

        return epsilonVal.value();
    } else if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(op)) {
        // Handle MultiplyOp(Cst_Scale) + AddOp pattern
        if (!multiplyOp->hasOneUse()) {
            return mlir::failure();
        }

        Const::DeclareOp multiplyConstOp = getMultiplyConstOperand(multiplyOp);
        if (multiplyConstOp) {
            const auto multiplyConstVal = Const::getSplatValue<float>(multiplyConstOp);
            if (mlir::succeeded(multiplyConstVal)) {
                if (auto addOp = mlir::dyn_cast<IE::AddOp>(*multiplyOp->getUsers().begin())) {
                    auto epsilonConstOp = mlir::isa_and_nonnull<Const::DeclareOp>(addOp->getOperand(0).getDefiningOp())
                                                  ? addOp->getOperand(0).getDefiningOp<Const::DeclareOp>()
                                                  : addOp->getOperand(1).getDefiningOp<Const::DeclareOp>();
                    if (epsilonConstOp == nullptr) {
                        return mlir::failure();
                    }

                    const auto maybeEpsilonVal = Const::getSplatValue<float>(epsilonConstOp);
                    if (mlir::failed(maybeEpsilonVal)) {
                        return mlir::failure();
                    }

                    return maybeEpsilonVal.value() / multiplyConstVal.value();
                }
            }
        }
    }
    return mlir::failure();
}

mlir::FailureOr<float> getMultiplyScale(mlir::Operation* op) {
    // Handle MultiplyOp as Cst_Scale
    if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(op)) {
        if (!multiplyOp->hasOneUse()) {
            return mlir::failure();
        }

        Const::DeclareOp multiplyConstOp = getMultiplyConstOperand(multiplyOp);
        if (multiplyConstOp) {
            const auto multiplyConstVal = Const::getSplatValue<float>(multiplyConstOp);
            if (mlir::succeeded(multiplyConstVal)) {
                return multiplyConstVal.value();
            }
        }
    }

    return mlir::failure();
}

// Create gamma from pre-scaled values
mlir::Value createGamma(mlir::OpBuilder& builder, mlir::Operation* op, int64_t size, ArrayRef<float> values) {
    const auto dataStorageType = mlir::RankedTensorType::get({size}, mlir::Float32Type::get(op->getContext()));
    const auto constLoc = appendLoc(op->getLoc(), "const");
    return Const::createConst(builder, constLoc, dataStorageType, values);
}

// Create a splat gamma
mlir::Value createGamma(mlir::OpBuilder& builder, mlir::Operation* op, int64_t size, float gammaScale) {
    VPUX_THROW_WHEN(gammaScale == 0.0f, "gammaScale must be non-zero");
    const float weightData = 1.0f / gammaScale;
    return createGamma(builder, op, size, ArrayRef(weightData));
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
    mlir::Operation* divideOp = nullptr;
    float multiplyScale = 1.0f;

    // Support both Sqrt->Divide and Power(-0.5) patterns
    divideOp = getSqrtAndDivideOps(*reduceSumOp->getUsers().begin());

    // Try getSqrtAndDivideOps again after epsilon operation
    auto tryGetSqrtAndDivideOps = [&divideOp](mlir::Operation* op) -> bool {
        // Skip dimension expansion op (AffineReshape/Unsqueeze) if present
        auto nextOp = *op->getUsers().begin();
        if (isDimExpansionOp(nextOp)) {
            if (!nextOp->hasOneUse()) {
                return false;
            }
            nextOp = *nextOp->getUsers().begin();
        }

        divideOp = getSqrtAndDivideOps(nextOp);
        if (divideOp == nullptr) {
            auto maybeAddOp = mlir::dyn_cast_or_null<IE::AddOp>(nextOp);
            if (!maybeAddOp || !maybeAddOp->hasOneUse()) {
                return false;
            }

            nextOp = *maybeAddOp->getUsers().begin();
            if (isDimExpansionOp(nextOp)) {
                if (!nextOp->hasOneUse()) {
                    return false;
                }
                nextOp = *nextOp->getUsers().begin();
            }

            divideOp = getSqrtAndDivideOps(nextOp);
            return divideOp != nullptr;
        }
        return true;
    };

    if (divideOp == nullptr) {
        auto preventDivByZeroOp = *reduceSumOp->getUsers().begin();
        if (!preventDivByZeroOp->hasOneUse()) {
            return;
        }
        auto epsilonResult = getEpsilon(preventDivByZeroOp);
        if (mlir::failed(epsilonResult)) {
            return;
        }
        epsilon = epsilonResult.value();

        auto multiplyScaleResult = getMultiplyScale(preventDivByZeroOp);
        if (mlir::succeeded(multiplyScaleResult)) {
            multiplyScale = multiplyScaleResult.value();
        }

        if (!tryGetSqrtAndDivideOps(preventDivByZeroOp)) {
            return;
        }
    }

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
    float gammaScale = sqrtf(reduceSize * multiplyScale);
    const auto epsilonAttr = getFPAttr(&ctx, epsilon / reduceSize);

    mlir::Operation* opBeforeScale = divideOp;
    if (auto realDivideOp = mlir::dyn_cast_or_null<IE::DivideOp>(divideOp)) {
        // Sqrt->Divide pattern
        // In case the first input of the DivideOp is 1
        auto divideInput = mlir::dyn_cast_or_null<Const::DeclareOp>(realDivideOp.getInput1().getDefiningOp());
        if (divideInput != nullptr && divideInput.getContent().getSplatValue<int64_t>() == 1) {
            if (!realDivideOp->hasOneUse()) {
                return;
            }
            auto selfMultiplyOp =
                    mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(*realDivideOp->getUsers().begin()));
            if (selfMultiplyOp == nullptr) {
                return;
            }
            opBeforeScale = selfMultiplyOp;
        }
    } else if (auto powerDivideOp = mlir::dyn_cast_or_null<IE::PowerOp>(divideOp)) {
        // Power(-0.5) pattern
        if (!powerDivideOp->hasOneUse()) {
            return;
        }
        auto selfMultiplyOp =
                mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(*powerDivideOp->getUsers().begin()));
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

        auto inReshapeOp = builder.create<IE::ReshapeOp>(appendLoc(powerOp->getLoc(), "in_reshape"),
                                                         powerOp->getOperand(0), getIntArrayAttr(&ctx, newInShape));

        // Create RMSOp
        auto rmsOp = builder.create<IE::RMSOp>(appendLoc(powerOp->getLoc(), "rms"), inReshapeOp, gamma, epsilonAttr);

        // Reshape Output
        auto outReshapeOp = builder.create<IE::ReshapeOp>(appendLoc(powerOp->getLoc(), "out_reshape"),
                                                          rmsOp->getResult(0), getIntArrayAttr(&ctx, powerInputShape));

        opBeforeScale->replaceAllUsesWith(outReshapeOp);

        return;
    }

    mlir::Operation* replaceOp = opBeforeScale;
    std::vector<float> scaledValues;
    if (scaleMultiplyOp != nullptr) {
        if (Const::DeclareOp constOp = scaleMultiplyOp.getOperand(1).getDefiningOp<Const::DeclareOp>()) {
            Const::Content constantContent = constOp.getContent();
            const bool matchSplat = constantContent.isSplat() && constantContent.getSplatValue<float>() != 0.0f;
            const bool matchVec = !constantContent.isSplat() && gammaScale != 0.0f &&
                                  constantContent.getType().getNumElements() == reduceSize;
            if (matchSplat || matchVec) {
                if (matchSplat) {
                    gammaScale /= constantContent.getSplatValue<float>();
                } else {
                    // Non-splat vector gamma: scale each element by 1/gammaScale.
                    scaledValues.resize(reduceSize);
                    for (const auto [i, v] : llvm::enumerate(constantContent.getValues<float>())) {
                        scaledValues[i] = v / gammaScale;
                    }
                }
                // Fold scaleMultiply Op into RMSOp gamma
                replaceOp = scaleMultiplyOp;
                builder = mlir::OpBuilder(replaceOp);
            }
        }
    }

    mlir::Value gamma = !scaledValues.empty() ? createGamma(builder, replaceOp, reduceSize, ArrayRef(scaledValues))
                                              : createGamma(builder, replaceOp, reduceSize, gammaScale);

    // Create RMSOp
    auto rmsOp =
            builder.create<IE::RMSOp>(appendLoc(powerOp->getLoc(), "rms"), powerOp->getOperand(0), gamma, epsilonAttr);

    replaceOp->replaceAllUsesWith(rmsOp);
}

mlir::Value createGammaWithShape(mlir::OpBuilder& builder, mlir::Operation* op, ShapeRef shape) {
    const float weightData = 1.0f;
    const auto dataStorageType = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(op->getContext()));
    const auto constLoc = appendLoc(op->getLoc(), "const");
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
                      mlir::FloatAttr epsilonAttr, bool conditionalEps = false) {
    auto gammaRank = mlir::cast<vpux::NDTypeInterface>(gamma.getType()).getRank();
    if (gammaRank != 1) {
        auto reshapeOp = builder.create<IE::ReshapeOp>(
                gamma.getLoc(), gamma, getIntArrayAttr(headOp->getContext(), SmallVector<int64_t>({layerSize})));
        gamma = reshapeOp;
    }
    auto conditionalEpsAttr = mlir::BoolAttr::get(headOp->getContext(), conditionalEps);
    auto rmsOp = builder.create<IE::RMSOp>(appendLoc(headOp->getLoc(), "rms"), headOp->getOperand(0), gamma,
                                           epsilonAttr, conditionalEpsAttr);

    return rmsOp;
}

//
// safeRunOnFunc
//

// Match pattern
// Input -> IE.Power -> IE.ReduceMean -> IE.Add (epsilon) -> [IE.AffineReshape] -> IE.Sqrt -> IE.Divide -> IE.Multiply
//   \                                                                                             /
//    ---------------------------------------------------------------------------------------------
// -> IE.Multiply (gamma)
// Or
// Input -> IE.Power|IE.Multiply -> IE.ReduceMean -> IE.Add (epsilon) -> [IE.AffineReshape] -> IE.Power(-0.5) ->
// IE.Multiply
//   \                                                                                                               /
//    ---------------------------------------------------------------------------------------------------------------
// -> [IE.Multiply (gamma)]
// Note: IE.Multiply (gamma) is optional; without it a unit gamma is synthesized.
//       The final IE.Multiply may have multiple downstream users.
// Or
// Input -> IE.Convert -> IE.Power -> IE.ReduceMean -> IE.Add (epsilon) -> [IE.AffineReshape] -> IE.Sqrt -> IE.Divide
//   \                                                                                                           /
//    -----------------------------------------------------------------------------------------------------------
// -> IE.Multiply -> IE.Convert -> IE.Multiply (gamma)
// Or
// Input -> IE.Power -> IE.ReduceMean -> IE.Equal -----
//                            |                       | ->
//                            |------------------------ ->   IE.Select -> IE.Sqrt -> IE.Divide -> IE.Multiply
//                                              Const   ->
// Note: [IE.AffineReshape] is optional, present when ReduceMean uses keep_dims=false
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

        // Check if ReduceMean has valid users:
        // - 1 user: normal Add/Clamp/Maximum/Sqrt pattern
        // - 2 users: Equal + Select pattern (both use ReduceMean output)
        auto hasValidReduceMeanUsers = [](IE::ReduceMeanOp reduceOp) -> bool {
            if (reduceOp == nullptr) {
                return false;
            }
            auto numUsers = std::distance(reduceOp->getUsers().begin(), reduceOp->getUsers().end());
            if (numUsers == 1) {
                return true;
            }
            if (numUsers == 2) {
                // Check for Equal + Select pattern
                bool hasEqual = false;
                bool hasSelect = false;
                for (auto user : reduceOp->getUsers()) {
                    if (mlir::isa<IE::EqualOp>(user)) {
                        hasEqual = true;
                    } else if (mlir::isa<IE::SelectOp>(user)) {
                        hasSelect = true;
                    }
                }
                return hasEqual && hasSelect;
            }
            return false;
        };

        if (!hasValidReduceMeanUsers(reduceMeanOp)) {
            const auto reduceSumOp = mlir::dyn_cast_or_null<IE::ReduceSumOp>(*powerOp->getUsers().begin());
            isReduceSumPattern(powerOp, reduceSumOp, ctx, _log);
            return;
        }

        auto headOp = powerOp;

        // Find the preventDivByZeroOp - could be Add, Clamp, Maximum, Sqrt, or Select
        // For Select pattern, we need SelectOp specifically (not EqualOp which is also a user of ReduceMean)
        auto findPreventDivByZeroOp = [](IE::ReduceMeanOp reduceOp) -> mlir::Operation* {
            mlir::Operation* fallback = nullptr;
            for (auto user : reduceOp->getUsers()) {
                if (mlir::isa<IE::SelectOp>(user)) {
                    return user;
                }
                if (mlir::isa<IE::AddOp, IE::ClampOp, IE::MaximumOp, IE::SqrtOp>(user)) {
                    fallback = user;
                }
            }
            return fallback ? fallback : *reduceOp->getUsers().begin();
        };

        auto firstReduceMeanUser = findPreventDivByZeroOp(reduceMeanOp);
        mlir::Operation* divideOp = getSqrtAndDivideOps(firstReduceMeanUser);
        auto epsilon = std::numeric_limits<type::float16>::smallest_mixed_precision_eps;
        bool isConditionalEpsPattern = false;

        if (divideOp == nullptr) {
            auto preventDivByZeroOp = firstReduceMeanUser;
            if (!preventDivByZeroOp->hasOneUse()) {
                return;
            }
            // Check if this is a Select pattern (conditional epsilon)
            isConditionalEpsPattern = mlir::isa<IE::SelectOp>(preventDivByZeroOp);
            auto epsilonResult = getEpsilon(preventDivByZeroOp);
            if (mlir::failed(epsilonResult)) {
                return;
            }
            epsilon = epsilonResult.value();
            auto nextOp = *preventDivByZeroOp->getUsers().begin();

            // Skip dimension expansion op (AffineReshape/Unsqueeze) if present
            if (isDimExpansionOp(nextOp)) {
                if (!nextOp->hasOneUse()) {
                    return;
                }
                nextOp = *nextOp->getUsers().begin();
            }

            divideOp = getSqrtAndDivideOps(nextOp);
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

            auto rmsOp = createRMSOp(builder, headOp, gamma, layerSize, epsilonAttr, isConditionalEpsPattern);

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
        if (multiplyOp1 == nullptr || getSingleDimSize(multiplyOp1) != layerSize) {
            return;
        }

        // Verify that the self-normalize multiply uses the original input on one side.
        // For IE.Multiply (x*x or x*Reshape(x)), both operands are candidates for the
        // original input because the squaring op may carry mixed inputs (e.g. x and
        // AffineReshape(x)), so operand(0) alone is not sufficient.
        auto matchesOrigInput = [&](mlir::Value v) -> bool {
            if (auto mulOp = mlir::dyn_cast<IE::MultiplyOp>(powerOp)) {
                return v == mulOp.getInput1() || v == mulOp.getInput2();
            }
            return v == powerOp->getOperand(0);
        };
        if (!matchesOrigInput(multiplyOp1->getOperand(0)) && !matchesOrigInput(multiplyOp1->getOperand(1))) {
            return;
        }

        // Look for an optional gamma multiply (or a Convert->gamma-multiply) only when
        // multiplyOp1 has exactly one user; otherwise treat it as the pattern endpoint.
        IE::MultiplyOp multiplyOp2 = nullptr;
        IE::ConvertOp convertOp2 = nullptr;
        auto convertOp1 = mlir::dyn_cast_or_null<IE::ConvertOp>(powerOp->getOperand(0).getDefiningOp());
        if (multiplyOp1->hasOneUse()) {
            multiplyOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*multiplyOp1->getUsers().begin());
            convertOp2 = mlir::dyn_cast_or_null<IE::ConvertOp>(*multiplyOp1->getUsers().begin());
            if (multiplyOp2 == nullptr) {
                // try to match convert case
                // Convert -> Power -> .... -> Multiply1 -> Convert -> Multiply2
                if (convertOp1 != nullptr && convertOp2 != nullptr && !convertOp2->use_empty()) {
                    multiplyOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(*convertOp2->getUsers().begin());
                    if (multiplyOp2 != nullptr) {
                        headOp = convertOp1.getOperation();
                    }
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

            if (getNonOneDim(gammaDims).size() > 1 || !isBroadcastable(gammaWidth, inputWidth)) {
                return;
            }

            if (gammaWidth != inputWidth) {
                auto gammaConstOp = gamma.getDefiningOp<Const::DeclareOp>();
                if (gammaConstOp == nullptr) {
                    return;
                }

                SmallVector<int64_t> newGammaShape(gammaDims.begin(), gammaDims.end());
                newGammaShape[gammaDims.size() - 1] = inputWidth;

                auto broadcastedContentAttr =
                        gammaConstOp.transformContentAttr().broadcast(Dim(gammaDims.size() - 1), inputWidth).get();
                auto newGammaType = mlir::RankedTensorType::get(
                        newGammaShape, mlir::cast<mlir::ShapedType>(gamma.getType()).getElementType());
                gamma = builder.create<Const::DeclareOp>(appendLoc(headOp->getLoc(), "gamma_broadcast"), newGammaType,
                                                         broadcastedContentAttr);
            }
        }

        _log.trace("RMS pattern matched");
        auto rmsOp = createRMSOp(builder, headOp, gamma, layerSize, epsilonAttr, isConditionalEpsPattern);
        if (needCreateGamma) {
            // Replacing all uses is safe: multiplyOp1 computes x/rms(x), which is
            // identical to the RMS op output with a unit gamma, regardless of how
            // many downstream consumers exist.
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
