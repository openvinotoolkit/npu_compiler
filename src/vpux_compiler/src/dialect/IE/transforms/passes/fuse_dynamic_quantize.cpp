//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <cmath>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEDYNAMICQUANTIZE
#define GEN_PASS_DEF_FUSEDYNAMICQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool hasSameAxes(mlir::ArrayRef<int64_t> lhs, mlir::ArrayRef<int64_t> rhs) {
    return llvm::equal(lhs, rhs);
}

std::optional<float> getConstSplatValueImpl(mlir::Value operand) {
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    auto convertOp = operand.getDefiningOp<IE::ConvertOp>();
    if (constOp == nullptr && convertOp == nullptr) {
        return std::nullopt;
    }
    if (constOp == nullptr) {
        constOp = convertOp.getInput().getDefiningOp<Const::DeclareOp>();
    }
    if (constOp == nullptr) {
        return std::nullopt;
    }

    auto splatValue = Const::getSplatValue<float>(constOp);
    if (mlir::failed(splatValue)) {
        return std::nullopt;
    }

    return splatValue.value();
}

bool isScalarLike(mlir::Value value) {
    const auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
    if (shapedType == nullptr || !shapedType.hasRank()) {
        return true;
    }

    return llvm::all_of(shapedType.getShape(), [](int64_t dim) {
        return dim == 1;
    });
}

bool hasSameShape(mlir::Value lhs, mlir::Value rhs) {
    return mlir::cast<mlir::ShapedType>(lhs.getType()).getShape() ==
           mlir::cast<mlir::ShapedType>(rhs.getType()).getShape();
}

bool isBroadcastCompatibleWith(mlir::Value tensor, mlir::Value input) {
    const auto tensorType = mlir::dyn_cast<mlir::ShapedType>(tensor.getType());
    const auto inputType = mlir::dyn_cast<mlir::ShapedType>(input.getType());
    if (tensorType == nullptr || inputType == nullptr || !tensorType.hasRank() || !inputType.hasRank()) {
        return true;
    }

    const auto tensorShape = tensorType.getShape();
    const auto inputShape = inputType.getShape();
    if (tensorShape.size() > inputShape.size()) {
        return false;
    }

    auto inputIt = inputShape.rbegin();
    auto tensorIt = tensorShape.rbegin();
    for (; tensorIt != tensorShape.rend(); ++tensorIt, ++inputIt) {
        if (*tensorIt > 1 && *tensorIt != *inputIt) {
            return false;
        }
    }

    return true;
}

bool isUnsignedQuantizeClamp(IE::ClampOp clampOp) {
    const auto minVal = clampOp.getMinAttr().getValueAsDouble();
    const auto maxVal = clampOp.getMaxAttr().getValueAsDouble();

    return isDoubleEqual(minVal, 0.0) && isDoubleEqual(maxVal, 255.0);
}

mlir::Type changeElemType(mlir::Type type, mlir::Type newElemType) {
    if (auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type)) {
        return ndType.changeElemType(newElemType);
    }

    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
        return mlir::RankedTensorType::get(shapedType.getShape(), newElemType);
    }

    return type;
}

mlir::Value matchBoundedValue(mlir::Value input, bool clampNegativeRange) {
    if (!input.hasOneUse()) {
        return input;
    }

    auto* user = *input.getUsers().begin();
    if (auto clampOp = mlir::dyn_cast<IE::ClampOp>(user)) {
        // Walk through any single-use ClampOp — downstream ops (Subtract, etc.) use the clamped
        // value regardless of whether the clamp specifically enforces min≤0 or max≥0.
        return clampOp.getOutput();
    }

    if (clampNegativeRange) {
        auto minimumOp = mlir::dyn_cast<IE::MinimumOp>(user);
        if (minimumOp == nullptr) {
            return input;
        }
        auto otherInput = minimumOp.getInput1() == input ? minimumOp.getInput2() : minimumOp.getInput1();
        const auto otherValue = getConstSplatValueImpl(otherInput);
        if (otherValue.has_value() && isDoubleEqual(otherValue.value(), 0.0)) {
            return minimumOp.getOutput();
        }
        return input;
    }

    auto maximumOp = mlir::dyn_cast<IE::MaximumOp>(user);
    if (maximumOp == nullptr) {
        return input;
    }
    auto otherInput = maximumOp.getInput1() == input ? maximumOp.getInput2() : maximumOp.getInput1();
    const auto otherValue = getConstSplatValueImpl(otherInput);
    if (otherValue.has_value() && isDoubleEqual(otherValue.value(), 0.0)) {
        return maximumOp.getOutput();
    }
    return input;
}

IE::SubtractOp findSubtractWithInputs(mlir::Value input1, mlir::Value input2) {
    for (auto* userOp : input1.getUsers()) {
        auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(userOp);
        if (subtractOp == nullptr) {
            continue;
        }
        if (subtractOp.getInput1() == input1 && subtractOp.getInput2() == input2) {
            return subtractOp;
        }
    }

    return nullptr;
}

bool matchValueOrConvertFrom(mlir::Value candidate, mlir::Value source, SmallVectorImpl<mlir::Operation*>& opsToErase) {
    if (candidate == source) {
        return true;
    }

    auto convertOp = candidate.getDefiningOp<IE::ConvertOp>();
    if (convertOp != nullptr && convertOp.getInput() == source) {
        opsToErase.push_back(convertOp);
        return true;
    }

    return false;
}

mlir::Value matchZpValueForAdd(IE::ClampOp clampQuant, IE::ConvertOp convertZp,
                               SmallVectorImpl<mlir::Operation*>& opsToErase) {
    for (auto* userOp : clampQuant.getOutput().getUsers()) {
        if (mlir::isa<IE::AddOp>(userOp)) {
            return clampQuant.getOutput();
        }
    }

    if (convertZp == nullptr) {
        return nullptr;
    }

    for (auto* userOp : convertZp.getOutput().getUsers()) {
        auto convertBack = mlir::dyn_cast<IE::ConvertOp>(userOp);
        if (convertBack == nullptr) {
            continue;
        }
        opsToErase.push_back(convertBack);
        return convertBack.getOutput();
    }

    return nullptr;
}

bool isValidQuantSpanConstant(float value, bool isMultiplier);

IE::RoundOp matchQuantRound(mlir::Value input, mlir::Value scaleValue, mlir::Value subtractSpanValue,
                            SmallVectorImpl<mlir::Operation*>& opsToErase) {
    for (auto* userOp : input.getUsers()) {
        if (auto divideOp = mlir::dyn_cast<IE::DivideOp>(userOp)) {
            if (divideOp.getInput1() != input || !divideOp->hasOneUse()) {
                continue;
            }

            SmallVector<mlir::Operation*> localOpsToErase;
            if (!matchValueOrConvertFrom(divideOp.getInput2(), scaleValue, localOpsToErase)) {
                continue;
            }

            auto roundOp = mlir::dyn_cast<IE::RoundOp>(*divideOp->getUsers().begin());
            if (roundOp == nullptr || !roundOp->hasOneUse()) {
                continue;
            }

            opsToErase.push_back(divideOp);
            opsToErase.append(localOpsToErase);
            return roundOp;
        }

        auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(userOp);
        if (multiplyOp == nullptr || !multiplyOp->hasOneUse()) {
            continue;
        }

        auto otherInput = multiplyOp.getInput1() == input ? multiplyOp.getInput2() : multiplyOp.getInput1();
        if ((multiplyOp.getInput1() != input && multiplyOp.getInput2() != input) || !isScalarLike(otherInput)) {
            continue;
        }

        // Validate the multiplier K matches a valid quantization span (255, 254, or 127)
        const auto kValue = getConstSplatValueImpl(otherInput);
        if (!kValue.has_value() || !isValidQuantSpanConstant(kValue.value(), /*isMultiplier=*/false)) {
            continue;
        }

        auto divideOp = mlir::dyn_cast<IE::DivideOp>(*multiplyOp->getUsers().begin());
        if (divideOp == nullptr || !divideOp->hasOneUse()) {
            continue;
        }
        if (divideOp.getInput1() != multiplyOp.getOutput() || divideOp.getInput2() != subtractSpanValue) {
            continue;
        }

        auto roundOp = mlir::dyn_cast<IE::RoundOp>(*divideOp->getUsers().begin());
        if (roundOp == nullptr || !roundOp->hasOneUse()) {
            continue;
        }

        opsToErase.append({multiplyOp, divideOp});
        return roundOp;
    }

    return nullptr;
}

IE::ClampOp matchZpClamp(mlir::Value scaleValue, IE::SubtractOp subtractQuantSpanOp, mlir::Value& zpNumerator,
                         SmallVectorImpl<mlir::Operation*>& opsToErase) {
    if (subtractQuantSpanOp != nullptr) {
        auto divideQuant = mlir::dyn_cast<IE::DivideOp>(*subtractQuantSpanOp->getUsers().begin());
        if (divideQuant == nullptr || !divideQuant->hasOneUse()) {
            return nullptr;
        }

        if (divideQuant.getInput1() != subtractQuantSpanOp.getOutput() || divideQuant.getInput2() != scaleValue) {
            return nullptr;
        }

        auto roundQuant = mlir::dyn_cast<IE::RoundOp>(*divideQuant->getUsers().begin());
        if (roundQuant == nullptr || !roundQuant->hasOneUse()) {
            return nullptr;
        }

        auto clampQuant = mlir::dyn_cast<IE::ClampOp>(*roundQuant->getUsers().begin());
        if (clampQuant == nullptr) {
            return nullptr;
        }

        zpNumerator = subtractQuantSpanOp.getOutput();
        opsToErase.append({divideQuant, roundQuant, clampQuant});
        return clampQuant;
    }

    for (auto* userOp : scaleValue.getUsers()) {
        auto divideQuant = mlir::dyn_cast<IE::DivideOp>(userOp);
        if (divideQuant == nullptr || !divideQuant->hasOneUse()) {
            continue;
        }

        if (divideQuant.getInput2() != scaleValue || !hasSameShape(divideQuant.getInput1(), scaleValue)) {
            continue;
        }

        // Try: Divide → Round → Clamp (signed asymmetric)
        // or:  Divide → Subtract(const, result) → Round → Clamp (unsigned asymmetric)
        IE::RoundOp roundQuant = nullptr;
        IE::SubtractOp zpSubtractOp = nullptr;
        auto* soleUser = *divideQuant->getUsers().begin();

        roundQuant = mlir::dyn_cast<IE::RoundOp>(soleUser);
        if (roundQuant == nullptr) {
            zpSubtractOp = mlir::dyn_cast<IE::SubtractOp>(soleUser);
            if (zpSubtractOp != nullptr && zpSubtractOp->hasOneUse() &&
                zpSubtractOp.getInput2() == divideQuant.getOutput() && isScalarLike(zpSubtractOp.getInput1())) {
                roundQuant = mlir::dyn_cast<IE::RoundOp>(*zpSubtractOp->getUsers().begin());
            }
        }

        if (roundQuant == nullptr || !roundQuant->hasOneUse()) {
            continue;
        }

        auto clampQuant = mlir::dyn_cast<IE::ClampOp>(*roundQuant->getUsers().begin());
        if (clampQuant == nullptr) {
            continue;
        }

        zpNumerator = divideQuant.getInput1();
        if (auto* zpNumeratorOp = zpNumerator.getDefiningOp();
            zpNumeratorOp != nullptr && zpNumerator.hasOneUse() && !llvm::is_contained(opsToErase, zpNumeratorOp)) {
            opsToErase.push_back(zpNumeratorOp);
        }
        opsToErase.push_back(divideQuant);
        if (zpSubtractOp != nullptr) {
            opsToErase.push_back(zpSubtractOp);
        }
        opsToErase.append({roundQuant, clampQuant});
        return clampQuant;
    }

    return nullptr;
}

bool isValidQuantSpanConstant(float value, bool isMultiplier) {
    // Valid quantization spans: 255 (unsigned), 254 (signed asymmetric), 127 (signed symmetric).
    // Use std::round on the reciprocal to tolerate float32 precision loss in MLIR constants
    // (e.g., 0.00392156886 ≈ 1/255 but differs from exact double 1.0/255.0 by ~6.8e-8).
    float span;
    if (isMultiplier) {
        if (value == 0.0f) {
            return false;
        }
        span = std::round(1.0f / value);
    } else {
        span = std::round(value);
    }
    return span == 255.0f || span == 254.0f || span == 127.0f;
}

float getQuantSpanFromConstant(float value, bool isMultiplier) {
    if (isMultiplier) {
        return std::round(1.0f / value);
    }
    return std::round(value);
}

mlir::Operation* matchScaleProducer(mlir::Value subtractSpan, SmallVectorImpl<mlir::Operation*>& opsToErase,
                                    mlir::Value& scaleValue, float& detectedQuantSpan) {
    for (auto* userOp : subtractSpan.getUsers()) {
        if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(userOp)) {
            auto otherInput = multiplyOp.getInput1() == subtractSpan ? multiplyOp.getInput2() : multiplyOp.getInput1();
            const auto constVal = getConstSplatValueImpl(otherInput);
            if (!isScalarLike(otherInput) || !constVal.has_value()) {
                continue;
            }
            if (!isValidQuantSpanConstant(constVal.value(), /*isMultiplier=*/true)) {
                continue;
            }
            detectedQuantSpan = getQuantSpanFromConstant(constVal.value(), /*isMultiplier=*/true);
            opsToErase.push_back(multiplyOp);
            scaleValue = multiplyOp.getOutput();
            return multiplyOp.getOperation();
        }

        if (auto divideOp = mlir::dyn_cast<IE::DivideOp>(userOp)) {
            if (divideOp.getInput1() != subtractSpan) {
                continue;
            }
            const auto constVal = getConstSplatValueImpl(divideOp.getInput2());
            if (!isScalarLike(divideOp.getInput2()) || !constVal.has_value()) {
                continue;
            }
            if (!isValidQuantSpanConstant(constVal.value(), /*isMultiplier=*/false)) {
                continue;
            }
            detectedQuantSpan = getQuantSpanFromConstant(constVal.value(), /*isMultiplier=*/false);
            opsToErase.push_back(divideOp);
            scaleValue = divideOp.getOutput();
            return divideOp.getOperation();
        }
    }

    scaleValue = nullptr;
    return nullptr;
}

//
// FuseDynamicQuantizePass
//

class FuseDynamicQuantizePass final : public IE::impl::FuseDynamicQuantizeBase<FuseDynamicQuantizePass> {
public:
    explicit FuseDynamicQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// The op sequence for the decomposed DynamicQuantizeLinear appears as follows when it reaches this point.
//
//  {
//    %0 = IE.ReduceMin(input)
//    %1 = IE.Clamp(%0)
//    %2 = IE.Subtract(%cst_1, %1)
//    %3 = IE.ReduceMax(input)
//    %4 = IE.Clamp(%3)
//    %5 = IE.Subtract(%4, %1)
//    %scale = IE.Multiply(%5, %cst_0)
//
//    %7 = IE.Divide(%2, %scale)
//    %8 = IE.Round(%7)
//    %9 = IE.Clamp(%8)
//    %zp = IE.Convert(%9)
//
//    %11 = IE.Multiply(input, %cst)
//    %12 = IE.Divide(%11, %5)
//    %13 = IE.Round(%12)
//    %14 = IE.Add(%13, %9)
//    %15 = IE.Clamp(%14)
//    %quantOutput = IE.Convert(%15)
//    return %quantOutput, %scale, %zp
//  }
//
// This sequence is similar to the quantization equation.
//    scale = (max-min) / 255
//    zp = (0-min) / scale
//    quantOutput = (x * 255) / (max-min) + zp
//                = x / scale + zp
//

void FuseDynamicQuantizePass::safeRunOnFunc() {
    auto func = getOperation();

    // Backward compatibility migration: update existing DynamicQuantize ops that have signless i8
    // output types (from older model XMLs serialized before the dstElemType attribute was
    // introduced) to use signed si8, matching the current inferReturnTypes. Also remove
    // now-redundant Convert(i8→si8) ops that were previously needed. This migration is
    // co-located with the fusion pass because both deal with DynamicQuantize IR correctness and
    // should run as a single atomic transformation. Only ops with signless-i8 results and
    // dstElemType=si8 are affected; the loop is a no-op when all results already have si8 type.
    bool funcSignatureNeedsUpdate = false;
    for (auto dqOp : func.getOps<IE::DynamicQuantizeOp>()) {
        const auto dstElemType = dqOp.getDstElemType();
        if (!dstElemType.isSignedInteger(8)) {
            continue;
        }
        auto ctx = dqOp.getContext();
        const auto si8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Signed);
        for (auto result : {dqOp.getOutput(), dqOp.getZeroPoint()}) {
            auto resultType = mlir::cast<mlir::RankedTensorType>(result.getType());
            if (!resultType.getElementType().isSignlessInteger(8)) {
                continue;
            }
            auto newType = mlir::RankedTensorType::get(resultType.getShape(), si8Type, resultType.getEncoding());
            result.setType(newType);
            funcSignatureNeedsUpdate = true;
            // Remove redundant Convert ops (i8→si8) that used this result
            for (auto& use : llvm::make_early_inc_range(result.getUses())) {
                if (auto cvtOp = mlir::dyn_cast<IE::ConvertOp>(use.getOwner())) {
                    auto cvtOutType = mlir::cast<mlir::RankedTensorType>(cvtOp.getOutput().getType());
                    if (cvtOutType.getElementType() == si8Type) {
                        cvtOp.getOutput().replaceAllUsesWith(result);
                        cvtOp->erase();
                    }
                }
            }
        }
    }

    // Update function signature to match mutated result types
    if (funcSignatureNeedsUpdate) {
        auto* terminator = func.getBody().back().getTerminator();
        SmallVector<mlir::Type> newResultTypes;
        for (auto operand : terminator->getOperands()) {
            newResultTypes.push_back(operand.getType());
        }
        func.setFunctionType(
                mlir::FunctionType::get(func.getContext(), func.getFunctionType().getInputs(), newResultTypes));
    }

    const auto createConvert = [&](mlir::OpBuilder& builder, mlir::Value newInput, mlir::Value origInput,
                                   mlir::Location loc) -> mlir::Value {
        const auto newEltType = mlir::cast<NDTypeInterface>(newInput.getType()).getElementType();
        const auto origEltType = mlir::cast<NDTypeInterface>(origInput.getType()).getElementType();
        if (newEltType == origEltType) {
            return newInput;
        }
        auto cvtOrigType = builder.create<IE::ConvertOp>(loc, newInput, origEltType);
        return cvtOrigType.getOutput();
    };

    for (auto reduceMinOp : llvm::make_early_inc_range(func.getOps<IE::ReduceMinOp>())) {
        const auto& nestLog = _log.nest();
        _log.trace("Got ReduceMinOp at {0}", reduceMinOp->getLoc());

        SmallVector<mlir::Operation*> opsToErase;
        if (!reduceMinOp->hasOneUse()) {
            nestLog.trace("ReduceMin has multi-users");
            continue;
        }

        const auto dynamicQuantizeInput = reduceMinOp.getInput();
        const auto inputType = dynamicQuantizeInput.getType().getElementType();
        if (!inputType.isF32()) {
            nestLog.trace("ReduceMin has non-FP32 input");
            continue;
        }

        auto reduceMinAxesValue = parseIntArrayAttr<int64_t>(reduceMinOp.getAxesValue());

        IE::ReduceMaxOp reduceMaxOp = nullptr;
        for (const auto userOp : dynamicQuantizeInput.getUsers()) {
            if (userOp == reduceMinOp.getOperation()) {
                continue;
            }
            if (mlir::isa<IE::ReduceMaxOp>(userOp) && reduceMaxOp == nullptr) {
                reduceMaxOp = mlir::cast<IE::ReduceMaxOp>(userOp);
                continue;
            }
        }

        if (reduceMaxOp == nullptr) {
            nestLog.trace("ReduceMaxOp is not found for ReduceMin input");
            continue;
        }

        auto reduceMaxAxesValue = parseIntArrayAttr<int64_t>(reduceMaxOp.getAxesValue());
        if (!hasSameAxes(reduceMinAxesValue, reduceMaxAxesValue) ||
            reduceMinOp.getKeepDims() != reduceMaxOp.getKeepDims()) {
            nestLog.trace("ReduceMin and ReduceMax use different reduction parameters");
            continue;
        }

        auto maxValue = matchBoundedValue(reduceMaxOp.getOutput(), false);
        auto minValue = matchBoundedValue(reduceMinOp.getOutput(), true);
        if (!isBroadcastCompatibleWith(maxValue, dynamicQuantizeInput) ||
            !isBroadcastCompatibleWith(minValue, dynamicQuantizeInput)) {
            nestLog.trace("DynamicQuantize min/max tensors are not broadcast-compatible with input");
            continue;
        }

        if (!hasSameShape(maxValue, minValue)) {
            nestLog.trace("DynamicQuantize min/max tensors have different shapes");
            continue;
        }

        auto subtractSpanOp = findSubtractWithInputs(maxValue, minValue);
        if (subtractSpanOp == nullptr) {
            nestLog.trace("maxValue is not followed by SubtractOp");
            continue;
        }

        IE::SubtractOp subtractQuantSpanOp = nullptr;
        for (const auto userOp : minValue.getUsers()) {
            if (userOp == subtractSpanOp.getOperation()) {
                continue;
            }
            if (mlir::isa<IE::SubtractOp>(userOp) && subtractQuantSpanOp == nullptr) {
                auto candidate = mlir::cast<IE::SubtractOp>(userOp);
                if (candidate.getInput2() != minValue || !isScalarLike(candidate.getInput1())) {
                    continue;
                }
                subtractQuantSpanOp = candidate;
                continue;
            }
            break;
        }

        if (subtractQuantSpanOp != nullptr) {
            // Treat the captured Subtract as a zero-point candidate only if it strictly
            // matches the DQ formula `zp = (0 - min) / scale`. Other forms (e.g. `qMin - min`
            // for signed decompositions) are handled later by matchZpClamp on the general
            // numerator, so simply drop the capture instead of bailing on the whole fusion.
            const auto zeroConstVal = getConstSplatValueImpl(subtractQuantSpanOp.getInput1());
            const bool isZeroMinusMin = subtractQuantSpanOp->hasOneUse() &&
                                        subtractQuantSpanOp.getInput2() == minValue &&
                                        isScalarLike(subtractQuantSpanOp.getInput1()) && zeroConstVal.has_value() &&
                                        isDoubleEqual(zeroConstVal.value(), 0.0);
            if (!isZeroMinusMin) {
                subtractQuantSpanOp = nullptr;
            } else {
                opsToErase.push_back(subtractQuantSpanOp);
            }
        }

        opsToErase.push_back(subtractSpanOp);

        mlir::Value scaleValue;
        float detectedQuantSpan = 0.0f;
        auto scaleProducerOp =
                matchScaleProducer(subtractSpanOp.getOutput(), opsToErase, scaleValue, detectedQuantSpan);
        if (scaleProducerOp == nullptr) {
            nestLog.trace("subtractSpanOp has no supported scale computation");
            continue;
        }

        mlir::Value zpNumerator;
        auto clampQuant = matchZpClamp(scaleValue, subtractQuantSpanOp, zpNumerator, opsToErase);
        if (clampQuant == nullptr) {
            // Symmetric (no zero-point) fusion is intentionally disabled: it triggers a
            // ~6% perf regression on the value_sym_i8 LLM KV-cache pattern. Keep only the
            // asymmetric zero-point path here; the symmetric path will be revisited in a
            // follow-up once the kernel-side cost is understood.
            nestLog.trace("Skipping symmetric DynamicQuantize fusion");
            continue;
        }

        IE::ConvertOp convertZp = nullptr;
        for (const auto userOp : clampQuant->getUsers()) {
            if (mlir::isa<IE::ConvertOp>(userOp) && convertZp == nullptr) {
                convertZp = mlir::cast<IE::ConvertOp>(userOp);
                continue;
            }
        }

        SmallVector<mlir::Operation*> localOpsToErase;
        auto zpValueForAdd = matchZpValueForAdd(clampQuant, convertZp, localOpsToErase);
        if (zpValueForAdd == nullptr) {
            nestLog.trace("Failed to derive fp32 zero-point value for AddOp");
            continue;
        }

        auto roundSpan = matchQuantRound(dynamicQuantizeInput, scaleValue, subtractSpanOp.getOutput(), opsToErase);
        if (roundSpan == nullptr) {
            nestLog.trace("Failed to match quantization branch before zero-point add");
            continue;
        }

        IE::AddOp addQuant = nullptr;
        for (auto* userOp : roundSpan.getOutput().getUsers()) {
            auto candidate = mlir::dyn_cast<IE::AddOp>(userOp);
            if (candidate == nullptr || !candidate->hasOneUse()) {
                continue;
            }
            auto otherInput =
                    candidate.getInput1() == roundSpan.getOutput() ? candidate.getInput2() : candidate.getInput1();
            if (otherInput != zpValueForAdd) {
                continue;
            }
            addQuant = candidate;
            break;
        }
        if (addQuant == nullptr) {
            nestLog.trace("roundSpan is not followed by an AddOp with matched zero-point input");
            continue;
        }

        auto clampQuantZp = mlir::dyn_cast<IE::ClampOp>(*addQuant->getUsers().begin());
        if (clampQuantZp == nullptr || !clampQuantZp->hasOneUse()) {
            nestLog.trace("addQuant is not followed by ClampOp");
            continue;
        }

        IE::ConvertOp outputConvert = nullptr;
        for (auto* userOp : clampQuantZp->getUsers()) {
            auto candidate = mlir::dyn_cast<IE::ConvertOp>(userOp);
            if (candidate == nullptr) {
                continue;
            }
            outputConvert = candidate;
            break;
        }

        opsToErase.append(localOpsToErase);
        opsToErase.append({roundSpan, addQuant, clampQuantZp});
        if (outputConvert != nullptr) {
            opsToErase.push_back(outputConvert);
        }
        if (convertZp != nullptr) {
            opsToErase.push_back(convertZp);
        }

        // Scale
        auto outputScale = scaleValue;
        // Zero point
        auto outputZp = convertZp != nullptr ? convertZp.getOutput() : clampQuant.getOutput();
        // data
        auto outputQuant = outputConvert != nullptr ? outputConvert.getOutput() : clampQuantZp.getOutput();

        SmallVector<mlir::OpOperand*> outputQuantUses;
        for (auto& use : outputQuant.getUses()) {
            outputQuantUses.push_back(&use);
        }

        SmallVector<mlir::OpOperand*> outputZpUses;
        for (auto& use : outputZp.getUses()) {
            outputZpUses.push_back(&use);
        }

        SmallVector<mlir::OpOperand*> outputScaleUses;
        for (auto& use : outputScale.getUses()) {
            outputScaleUses.push_back(&use);
        }

        const auto replaceCapturedUses = [](SmallVectorImpl<mlir::OpOperand*>& uses, mlir::Value newValue) {
            for (auto* use : uses) {
                use->set(newValue);
            }
        };

        const auto useUnsignedQuantizedType =
                isUnsignedQuantizeClamp(clampQuantZp) && isUnsignedQuantizeClamp(clampQuant);

        // Verify that the detected quantSpan from the original graph matches the kernel's
        // hardcoded quantSpan (qMax - qMin): 255 for unsigned, 254 for signed.
        // For signed types, quantSpan=127 is also accepted — handled by doubling the range
        // so the kernel's (max-min)/254 yields the same scale as the original range/127.
        const float expectedKernelQuantSpan = useUnsignedQuantizedType ? 255.0f : 254.0f;
        const bool needsRangeDoubling = !useUnsignedQuantizedType && isDoubleEqual(detectedQuantSpan, 127.0);
        if (!isDoubleEqual(detectedQuantSpan, expectedKernelQuantSpan) && !needsRangeDoubling) {
            nestLog.trace("Detected quantSpan {0} does not match kernel expectation {1}", detectedQuantSpan,
                          expectedKernelQuantSpan);
            continue;
        }

        const auto quantizedElemType =
                useUnsignedQuantizedType
                        ? mlir::Type(mlir::IntegerType::get(clampQuantZp.getContext(), 8,
                                                            mlir::IntegerType::SignednessSemantics::Unsigned))
                        : mlir::Type(mlir::IntegerType::get(clampQuantZp.getContext(), 8,
                                                            mlir::IntegerType::SignednessSemantics::Signed));
        const auto zpElemType =
                useUnsignedQuantizedType
                        ? mlir::Type(mlir::IntegerType::get(clampQuant.getContext(), 8,
                                                            mlir::IntegerType::SignednessSemantics::Unsigned))
                        : mlir::Type(mlir::IntegerType::get(clampQuant.getContext(), 8,
                                                            mlir::IntegerType::SignednessSemantics::Signed));
        const auto dqOutputType = changeElemType(outputQuant.getType(), quantizedElemType);
        const auto dqZpType = changeElemType(outputZp.getType(), zpElemType);

        // Insert the new DynamicQuantizeOp and its result Converts right after the latest of the
        // ops producing the values consumed/replaced (the reduce ops and the min/max bound
        // producers). This guarantees the new results dominate every original use being replaced
        // (scale/zp/quant-output downstream consumers all appear later in the block). Inserting at
        // the tail of the decomposition chain instead would place the new values after early uses
        // and break SSA dominance ('operand does not dominate this use').
        const auto pickLater = [](mlir::Operation* lhs, mlir::Operation* rhs) -> mlir::Operation* {
            if (lhs == nullptr) {
                return rhs;
            }
            if (rhs == nullptr) {
                return lhs;
            }
            return lhs->isBeforeInBlock(rhs) ? rhs : lhs;
        };
        mlir::Operation* insertAfterOp = pickLater(reduceMinOp, reduceMaxOp);
        insertAfterOp = pickLater(insertAfterOp, minValue.getDefiningOp());
        insertAfterOp = pickLater(insertAfterOp, maxValue.getDefiningOp());

        auto builder = mlir::OpBuilder(reduceMinOp);
        builder.setInsertionPointAfter(insertAfterOp);

        const auto loc = clampQuantZp->getLoc();
        const auto dstElemTypeAttr = mlir::TypeAttr::get(quantizedElemType);

        // When needsRangeDoubling: model uses scale=range/127 but kernel uses (max-min)/254.
        // Pass maxArg = 2*max - min (== min + 2*(max-min)) so kernel's (maxArg - min)/254 =
        // 2*range/254 = range/127. Compute it from min/max directly (not from subtractSpanOp,
        // which is erased) so the helper ops only depend on values that dominate the insertion
        // point.
        mlir::Value maxArg = maxValue;
        if (needsRangeDoubling) {
            const auto maxTensorType = mlir::cast<mlir::RankedTensorType>(maxValue.getType());
            // Use splat (size==1) to avoid IR/memory bloat for per-axis min/max tensors.
            auto twoConst = Const::createFloatConst(builder, appendLoc(loc, "two"), maxTensorType, {2.0f});
            auto doubledMax =
                    builder.create<IE::MultiplyOp>(appendLoc(loc, "doubled_max"), maxValue, twoConst,
                                                   IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
            maxArg = builder.create<IE::SubtractOp>(appendLoc(loc, "adjusted_max"), doubledMax.getOutput(), minValue,
                                                    IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr)
                             .getOutput();
        }

        auto dqOp = builder.create<IE::DynamicQuantizeOp>(
                appendLoc(loc, "dq_linear"), mlir::TypeRange{dqOutputType, outputScale.getType(), dqZpType},
                dynamicQuantizeInput, minValue, maxArg, dstElemTypeAttr);
        auto dqZpResult = createConvert(builder, dqOp.getZeroPoint(), outputZp, appendLoc(loc, "zp"));
        auto dqOutputResult = createConvert(builder, dqOp.getOutput(), outputQuant, appendLoc(loc, "output"));
        auto dqScale = createConvert(builder, dqOp.getScale(), outputScale, appendLoc(loc, "scale"));

        replaceCapturedUses(outputQuantUses, dqOutputResult);
        replaceCapturedUses(outputZpUses, dqZpResult);
        replaceCapturedUses(outputScaleUses, dqScale);

        _log.trace("DynamicQuantizeOp fused");

        for (auto op : opsToErase | reversed) {
            op->erase();
        }
    }
}

}  // namespace

//
// createFuseDynamicQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseDynamicQuantizePass(Logger log) {
    return std::make_unique<FuseDynamicQuantizePass>(log);
}
