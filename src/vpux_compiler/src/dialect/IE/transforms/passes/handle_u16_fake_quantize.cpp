//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

namespace vpux::IE {
#define GEN_PASS_DECL_HANDLEU16FAKEQUANTIZE
#define GEN_PASS_DEF_HANDLEU16FAKEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// HandleU16FakeQuantizePass
//

class HandleU16FakeQuantizePass final : public IE::impl::HandleU16FakeQuantizeBase<HandleU16FakeQuantizePass> {
public:
    explicit HandleU16FakeQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class RemoveU16FakeQuantizeRewriter;
    class ReplaceChildU16FakeQuantizeWithReluRewriter;

private:
    void safeRunOnFunc() final;
};

std::pair<SmallVector<float>, SmallVector<float>> getWeightsAndBiases(mlir::Value inputLow, mlir::Value inputHigh,
                                                                      mlir::Value outputLow, mlir::Value outputHigh) {
    auto inLowVals = IE::getConst(inputLow.getDefiningOp<Const::DeclareOp>());
    auto inHighVals = IE::getConst(inputHigh.getDefiningOp<Const::DeclareOp>());
    auto outLowVals = IE::getConst(outputLow.getDefiningOp<Const::DeclareOp>());
    auto outHighVals = IE::getConst(outputHigh.getDefiningOp<Const::DeclareOp>());

    auto resultSize = std::max(inLowVals.size(), outLowVals.size());
    SmallVector<float> weights(resultSize, 0.f), biases(resultSize, 0.f);

    auto getVal = [](SmallVector<float> values, size_t idx) {
        return values.size() > 1 ? values[idx] : values[0];
    };

    for (size_t idx = 0; idx < resultSize; idx++) {
        auto inLow = getVal(inLowVals, idx);
        auto inHigh = getVal(inHighVals, idx);
        auto outLow = getVal(outLowVals, idx);
        auto outHigh = getVal(outHighVals, idx);

        // FakeQuantize output calculation:
        // output = round((x - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high -
        // output_low) + output_low
        // - >
        // output = x * (output_high - output_low) / (input_high - input_low) - ((input_low * output_high - input_high *
        // output_low) / (input_high - input_low))
        // where: weights = (output_high - output_low) / (input_high - input_low)
        // biases = - ((input_low * output_high - input_high * output_low) / (input_high - input_low))
        // FakeQuantize -> ScaleShift: x * weights + biases
        weights[idx] = (outHigh - outLow) / (inHigh - inLow);
        biases[idx] = -((inLow * outHigh - inHigh * outLow) / (inHigh - inLow));
    }

    return {weights, biases};
}

bool areFQValsEqual(mlir::Value inputLow, mlir::Value inputHigh, mlir::Value outputLow, mlir::Value outputHigh) {
    // Check if all FQ input/output values are equal
    auto inLowVals = IE::getConst(inputLow.getDefiningOp<Const::DeclareOp>());
    auto inHighVals = IE::getConst(inputHigh.getDefiningOp<Const::DeclareOp>());
    auto outLowVals = IE::getConst(outputLow.getDefiningOp<Const::DeclareOp>());
    auto outHighVals = IE::getConst(outputHigh.getDefiningOp<Const::DeclareOp>());

    auto areValsEqual = [](SmallVector<float> values, float value) {
        return llvm::all_of(values, [&](float val) {
            return isFloatEqual(val, value);
        });
    };

    if (inLowVals.size() == outLowVals.size()) {
        return std::equal(inLowVals.begin(), inLowVals.end(), outLowVals.begin(), isFloatEqual) &&
               std::equal(inHighVals.begin(), inHighVals.end(), outHighVals.begin(), isFloatEqual);
    } else if (inLowVals.size() > outLowVals.size()) {
        return areValsEqual(std::move(inLowVals), outLowVals[0]) && areValsEqual(std::move(inHighVals), outHighVals[0]);
    } else {
        return areValsEqual(std::move(outLowVals), inLowVals[0]) && areValsEqual(std::move(outHighVals), inHighVals[0]);
    }
    return false;
}

float getConstSplatValue(mlir::Value fqVal) {
    auto fqValDeclareOp = fqVal.getDefiningOp<Const::DeclareOp>();
    return fqValDeclareOp.getContent().getSplatValue<float>();
}

mlir::Value applyU16FakequantizeOnConstant(mlir::PatternRewriter& rewriter, IE::FakeQuantizeOp fqOp,
                                           Const::DeclareOp fqInput) {
    auto inLowValue = getConstSplatValue(fqOp.getInputLow());
    auto inHighValue = getConstSplatValue(fqOp.getInputHigh());
    auto outLowValue = getConstSplatValue(fqOp.getOutputLow());
    auto outHighValue = getConstSplatValue(fqOp.getOutputHigh());

    auto fqInputType = mlir::cast<vpux::NDTypeInterface>(fqInput.getType());
    auto fqInputElementType = fqInputType.getElementType();
    auto storageType = mlir::RankedTensorType::get(fqInputType.getShape(), fqInputElementType);
    auto inputValues = IE::getConst(fqInput);
    auto fqLevels = mlir::dyn_cast<mlir::IntegerAttr>(fqOp.getLevelsOrLowFpTypeAttr());
    for (auto& value : inputValues) {
        value = fakeQuantize(value, inLowValue, inHighValue, outLowValue, outHighValue, fqLevels.getInt());
    }
    return vpux::Const::createFloatConst(rewriter, fqOp.getLoc(), storageType, inputValues);
}
//
// RemoveU16FakeQuantizeRewriter
//

class HandleU16FakeQuantizePass::RemoveU16FakeQuantizeRewriter final :
        public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    RemoveU16FakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("RemoveU16FakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult convertFQToScaleShift(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult HandleU16FakeQuantizePass::RemoveU16FakeQuantizeRewriter::convertFQToScaleShift(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto greaterThanOne = [](auto dim) {
        return dim > 1;
    };
    auto outAxisCount = llvm::count_if(getShape(origOp.getOutputLow()), greaterThanOne);
    auto inAxisCount = llvm::count_if(getShape(origOp.getInputLow()), greaterThanOne);
    // Check if input_low/high shape is <1xNx1x1> and output_low/high shape is <1xMx1x1> - unable to broadcast
    if (outAxisCount > 0 && inAxisCount > 0) {
        return mlir::failure();
    }

    SmallVector<float> weightsVec, biasesVec;
    std::tie(weightsVec, biasesVec) = getWeightsAndBiases(origOp.getInputLow(), origOp.getInputHigh(),
                                                          origOp.getOutputLow(), origOp.getOutputHigh());

    auto maxShape = vpux::details::calcTotalShapeSize(getShape(origOp.getInputLow())) >
                                    vpux::details::calcTotalShapeSize(getShape(origOp.getOutputLow()))
                            ? getShape(origOp.getInputLow())
                            : getShape(origOp.getOutputLow());

    const auto newShape = mlir::RankedTensorType::get(maxShape, mlir::Float32Type::get(rewriter.getContext()));

    // A u16 quantizer's scale `(outHigh-outLow)/(inHigh-inLow)` and bias `outLow - inLow*scale` can both
    // exceed the FP16 range and would then be clamped to the FP16 max once materialized as a constant. To
    // avoid this we split each of them into FP16-representable pieces: the scale into a product of factors
    // (one Multiply per factor) and the bias into a sum of terms (one Add per term).
    const auto fp16Max = checked_cast<float>(std::numeric_limits<vpux::type::float16>::max());
    // Largest per-factor power-of-two exponent that keeps the factor within FP16 (2^maxStageExp <= fp16Max).
    const auto maxStageExp = static_cast<int>(std::floor(std::log2(fp16Max)));

    // Factor one scale into pieces whose magnitudes each fit FP16 and whose product is the original scale. A
    // single power-of-two/residual pair is not always enough: a very large scale (e.g. 1e10) needs a
    // power-of-two factor to bring the residual back into FP16 range.
    const auto factorizeScale = [&](float weight) -> SmallVector<float> {
        const auto magnitude = std::abs(weight);
        if (magnitude <= fp16Max) {
            return {weight};
        }
        // Smallest total power-of-two exponent that pulls the residual back into FP16 range.
        const auto totalExp = static_cast<int>(std::ceil(std::log2(magnitude / fp16Max)));
        SmallVector<float> factors;
        for (int remaining = totalExp; remaining > 0; remaining -= maxStageExp) {
            factors.push_back(std::exp2(static_cast<float>(std::min(remaining, maxStageExp))));
        }
        // The residual now fits FP16; carry the sign here so the product of all factors equals the scale.
        factors.push_back(std::copysign(magnitude / std::exp2(static_cast<float>(totalExp)), weight));
        return factors;
    };

    // Split one bias into equal terms whose magnitudes each fit FP16 and whose sum is the original bias.
    // Biases that already fit FP16 stay a single term.
    const auto splitBias = [&](float bias) -> SmallVector<float> {
        const auto magnitude = std::abs(bias);
        if (magnitude <= fp16Max) {
            return {bias};
        }
        const auto numTerms = static_cast<size_t>(std::ceil(magnitude / fp16Max));
        // magnitude / numTerms <= fp16Max by construction, so every term stays within FP16.
        return SmallVector<float>(numTerms, bias / static_cast<float>(numTerms));
    };

    // Factor every channel, then transpose the per-channel piece lists into one constant per stage, padding
    // shorter factorizations with the operation's identity (1.0 for Multiply, 0.0 for Add) so every stage
    // covers all channels.
    const auto buildStages = [](ArrayRef<float> values, llvm::function_ref<SmallVector<float>(float)> factorize,
                                float identity) -> SmallVector<SmallVector<float>> {
        SmallVector<SmallVector<float>> perChannel(values.size());
        size_t numStages = 1;
        for (size_t idx = 0; idx < values.size(); ++idx) {
            perChannel[idx] = factorize(values[idx]);
            numStages = std::max(numStages, perChannel[idx].size());
        }
        SmallVector<SmallVector<float>> stages(numStages, SmallVector<float>(values.size(), identity));
        for (size_t idx = 0; idx < values.size(); ++idx) {
            for (size_t stage = 0; stage < perChannel[idx].size(); ++stage) {
                stages[stage][idx] = perChannel[idx][stage];
            }
        }
        return stages;
    };

    const auto scaleStages = buildStages(weightsVec, factorizeScale, /*identity=*/1.0f);
    const auto biasStages = buildStages(biasesVec, splitBias, /*identity=*/0.0f);

    // Materialize one constant per stage first so the ScaleShift constants stay grouped ahead of the
    // Multiply/Add chain. The product of the scale constants equals the original scale and the sum of the
    // bias constants equals the original bias.
    SmallVector<mlir::Value> scaleConsts;
    for (const auto& scaleVec : scaleStages) {
        scaleConsts.push_back(Const::createConst(rewriter, origOp->getLoc(), newShape, ArrayRef(scaleVec)));
    }
    SmallVector<mlir::Value> biasConsts;
    for (const auto& biasVec : biasStages) {
        biasConsts.push_back(Const::createConst(rewriter, origOp->getLoc(), newShape, ArrayRef(biasVec)));
    }

    mlir::Value result = origOp.getInput();
    for (auto scaleConst : scaleConsts) {
        result = rewriter.createOrFold<IE::MultiplyOp>(takeOpLoc(origOp, "as_mul"), origOp.getType(), result,
                                                       scaleConst, IE::AutoBroadcastType::NUMPY,
                                                       /*post_op=*/nullptr,
                                                       /*clamp=*/nullptr,
                                                       /*outputPadding=*/nullptr,
                                                       /*inputPadding=*/nullptr);
    }
    for (auto biasConst : biasConsts) {
        result = rewriter.createOrFold<IE::AddOp>(takeOpLoc(origOp, "as_add"), result.getType(), result, biasConst,
                                                  IE::AutoBroadcastType::NUMPY,
                                                  /*post_op=*/nullptr,
                                                  /*clamp=*/nullptr,
                                                  /*outputPadding=*/nullptr,
                                                  /*inputPadding=*/nullptr);
    }
    rewriter.replaceAllOpUsesWith(origOp, result);
    return mlir::success();
}

mlir::LogicalResult HandleU16FakeQuantizePass::RemoveU16FakeQuantizeRewriter::matchAndRewrite(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto moduleOp = getModuleOp(origOp);
    auto setAdaptiveStrippingEnabled = config::hasEnableAdaptiveStripping(moduleOp);
    auto seOpsEnabled = config::hasEnableSEPtrsOperations(moduleOp);
    auto levels = origOp.getLevels();
    if (!levels.has_value()) {
        return mlir::failure();
    }
    auto maxLevels = IE::getMaximumQuantizationLevels(levels.value(), origOp);
    // Maximum number of levels that don't exceeds I8/U8 storage type
    if (levels.value() <= maxLevels) {
        return mlir::failure();
    }

    const auto mappableInfo = IE::classifyMPEEngineMappableConsumers(origOp, seOpsEnabled, _log);
    if (mappableInfo.eligibility == IE::MPEConsumerEligibility::ALL) {
        _log.trace("[{0}] Keeping U16 FQ at '{1}' since all its consumers can be mapped by the MPE engine",
                   getDebugName(), origOp->getLoc());
        return mlir::failure();
    }

    bool consumersSplit = false;
    const auto splitMappableConsumers = [&]() {
        if (consumersSplit) {
            return;
        }

        consumersSplit = IE::splitMPEEngineMappableConsumers(rewriter, origOp, seOpsEnabled, _log);
    };

    auto areFQValsEqualToValue = [](IE::FakeQuantizeOp op, float value) {
        auto inLowValue = IE::getConst(op.getInputLow().getDefiningOp<Const::DeclareOp>())[0];

        return areFQValsEqual(op.getInputLow(), op.getInputHigh(), op.getOutputLow(), op.getOutputHigh()) &&
               isFloatEqual(inLowValue, value);
    };

    auto areFQLowValsEqualToValue = [](IE::FakeQuantizeOp op, float value) {
        const auto inLowValue = IE::getConst(op.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto outLowValue = IE::getConst(op.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];

        return isFloatEqual(inLowValue, outLowValue) && isFloatEqual(inLowValue, value);
    };

    auto fqInput = origOp.getInput();
    if (!mlir::isa<mlir::BlockArgument>(fqInput) && mlir::isa<Const::DeclareOp>(fqInput.getDefiningOp())) {
        // Create a copy of the original constant in case it has more uses
        auto fqInputConst = mlir::cast<Const::DeclareOp>(fqInput.getDefiningOp());
        auto fqInputContent = fqInputConst.getContent();
        auto fqInputContentType = fqInputContent.getType();
        const auto fqInputContentSize = checked_cast<size_t>(fqInputContentType.getTotalAllocSize().count());
        std::vector<char> newContent(fqInputContentSize);
        fqInputContent.copyTo(MutableArrayRef(newContent.data(), fqInputContentSize));
        const auto newFoldedBaseContent =
                Const::createConstContent(mlir::cast<mlir::ShapedType>(fqInputContentType), ArrayRef(newContent));
        Const::ContentSetup newContentAttrSetup(newFoldedBaseContent, fqInputContentType);
        auto newContentAttr = Const::ContentAttr::get(newFoldedBaseContent, newContentAttrSetup);
        auto clonedFoldedConstant = Const::DeclareOp::create(rewriter, origOp.getLoc(), newContentAttr.getType(),
                                                             std::move(newContentAttr));

        // Apply fakeQuantize on the constant
        auto newFqInput = applyU16FakequantizeOnConstant(rewriter, origOp, clonedFoldedConstant);
        splitMappableConsumers();
        rewriter.replaceOp(origOp, newFqInput);
        return mlir::success();
    }

    if (setAdaptiveStrippingEnabled) {
        // Scan all users to find a U16 FQ child
        IE::FakeQuantizeOp childFqOp;
        for (auto* user : origOp.getOutput().getUsers()) {
            if (auto candidate = mlir::dyn_cast<IE::FakeQuantizeOp>(user)) {
                if (auto candidateLevels = candidate.getLevels()) {
                    if (candidateLevels.value() > maxLevels) {
                        childFqOp = candidate;
                        break;
                    }
                }
            }
        }
        if (childFqOp != nullptr) {
            // origOp is the parent in a U16 FQ → U16 FQ chain and the parent itself
            // matches the il=ol=0 and ih=oh ReLU pattern
            if (IE::isPerTensorFQ({origOp}) && areFQValsEqualToValue(origOp, 0.0f) && origOp->hasOneUse()) {
                splitMappableConsumers();
                rewriter.replaceOpWithNewOp<IE::ReLUOp>(origOp, fqInput);
                return mlir::success();
            }
        } else {
            // no U16 FQ child among any user.
            // A per-tensor U16 FQ with inLow==outLow==0 following a LayerWithPostOpInterface op
            // (e.g. a bias-add after a convolution) is an activation clamp semantically
            // equivalent to ReLU in QDQ-quantized CNNs.
            if (IE::isPerTensorFQ({origOp}) && areFQLowValsEqualToValue(origOp, 0.0f) &&
                mlir::isa_and_present<IE::LayerWithPostOpInterface>(fqInput.getDefiningOp()) &&
                !mlir::isa_and_present<IE::ElemTypeInfoOpInterface>(fqInput.getDefiningOp())) {
                splitMappableConsumers();
                rewriter.replaceOpWithNewOp<IE::ReLUOp>(origOp, fqInput);
                return mlir::success();
            }
        }
        // In case the FakeQuantize has values in_low != out_low or in_high != out_high it can be replaced with a
        // ScaleShift op
        if (!areFQValsEqual(origOp.getInputLow(), origOp.getInputHigh(), origOp.getOutputLow(),
                            origOp.getOutputHigh())) {
            splitMappableConsumers();
            const auto scaleShiftResult = convertFQToScaleShift(origOp, rewriter);
            if (mlir::succeeded(scaleShiftResult)) {
                return mlir::success();
            }
            if (!consumersSplit) {
                return mlir::failure();
            }
        }
    } else if (IE::isPerTensorFQ({origOp}) && areFQValsEqualToValue(origOp, 0.0f)) {
        splitMappableConsumers();
        rewriter.replaceOpWithNewOp<IE::ReLUOp>(origOp, fqInput);
        return mlir::success();
    }

    splitMappableConsumers();
    rewriter.replaceOp(origOp, fqInput);
    return mlir::success();
}

//
// ReplaceChildU16FakeQuantizeWithReluRewriter
//
// Mirror of RemoveU16FakeQuantizeRewriter for the ParentFQ U16 -> ChildFQ U16 chain where
// the child matches the il=ol=0 ReLU pattern. The child is matched as root and replaced
// with a ReLU consuming the parent's input.
//

class HandleU16FakeQuantizePass::ReplaceChildU16FakeQuantizeWithReluRewriter final :
        public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    ReplaceChildU16FakeQuantizeWithReluRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("ReplaceChildU16FakeQuantizeWithReluRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp childFqOp, mlir::PatternRewriter& rewriter) const final {
        auto moduleOp = getModuleOp(childFqOp);
        if (!config::hasEnableAdaptiveStripping(moduleOp)) {
            return mlir::failure();
        }
        auto seOpsEnabled = config::hasEnableSEPtrsOperations(moduleOp);

        auto childLevels = childFqOp.getLevelsOrLowFpTypeAttr();
        auto childLevelsInt = mlir::dyn_cast<mlir::IntegerAttr>(childLevels);
        auto childMaxLevels = IE::getMaximumQuantizationLevels(
                childLevelsInt != nullptr ? childLevelsInt.getInt() : QuantizationLevels::QUANT_LEVELS_8BIT, childFqOp);
        if (childLevelsInt == nullptr || childLevelsInt.getInt() <= childMaxLevels) {
            return mlir::failure();
        }

        const auto childMappableInfo = IE::classifyMPEEngineMappableConsumers(childFqOp, seOpsEnabled, _log);
        if (childMappableInfo.eligibility == IE::MPEConsumerEligibility::ALL) {
            _log.trace("[{0}] Keeping child U16 FQ at '{1}' since all its consumers can be mapped by the MPE engine",
                       getDebugName(), childFqOp->getLoc());
            return mlir::failure();
        }

        // Parent must also be a U16 FQ.
        auto parentFqOp = childFqOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
        if (parentFqOp == nullptr) {
            return mlir::failure();
        }
        auto parentLevels = parentFqOp.getLevelsOrLowFpTypeAttr();
        auto parentLevelsInt = mlir::dyn_cast<mlir::IntegerAttr>(parentLevels);
        auto parentMaxLevels = IE::getMaximumQuantizationLevels(
                parentLevelsInt != nullptr ? parentLevelsInt.getInt() : QuantizationLevels::QUANT_LEVELS_8BIT,
                parentFqOp);
        if (parentLevelsInt == nullptr || parentLevelsInt.getInt() <= parentMaxLevels) {
            return mlir::failure();
        }

        // Skip if the parent itself matches the ReLU pattern (handled by the parent rewriter).
        const auto parentInLow = IE::getConst(parentFqOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto parentOutLow = IE::getConst(parentFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto parentInHigh = IE::getConst(parentFqOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
        const auto parentOutHigh = IE::getConst(parentFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];
        const bool parentIsReluPattern = IE::isPerTensorFQ({parentFqOp}) && isFloatEqual(parentInLow, parentOutLow) &&
                                         isFloatEqual(parentInHigh, parentOutHigh) && isFloatEqual(parentInLow, 0.0f) &&
                                         parentFqOp->hasOneUse();
        if (parentIsReluPattern) {
            return mlir::failure();
        }

        // Child must match the ReLU pattern.
        const auto childInLow = IE::getConst(childFqOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto childOutLow = IE::getConst(childFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto childInHigh = IE::getConst(childFqOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
        const auto childOutHigh = IE::getConst(childFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];
        if (!IE::isPerTensorFQ({childFqOp}) || !isFloatEqual(childInLow, childOutLow) ||
            !isFloatEqual(childInHigh, childOutHigh) || !isFloatEqual(childInLow, 0.0f)) {
            return mlir::failure();
        }

        IE::splitMPEEngineMappableConsumers(rewriter, childFqOp, seOpsEnabled, _log);
        _log.trace("[{0}] Replacing child U16 FQ at '{1}' with ReLU consuming parent FQ input", getDebugName(),
                   childFqOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::ReLUOp>(childFqOp, parentFqOp.getInput());
        return mlir::success();
    }

private:
    Logger _log;
};

//
// LowerFakeQuantizeRewriter
//

template <class ConcreteOp>
class LowerFakeQuantizeRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    LowerFakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("LowerFakeQuantizeRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

struct LowHigh {
    mlir::Value low;
    mlir::Value high;
};

enum class Direction {
    ToUpperBound = 1,
    ToLowerBound = -1,
};

std::optional<LowHigh> reduceLevelsZeroPoint(IE::FakeQuantizeOp origOp, float low, float high, int64_t maxLevels,
                                             mlir::PatternRewriter& rewriter) {
    auto loc = origOp->getLoc();
    auto type = origOp.getInputLow().getType();
    auto limit = static_cast<float>(maxLevels - 1);
    auto scale = (high - low) / limit;

    auto lowScaleCalc = [](float low, float zeroPoint) {
        return -low / zeroPoint;
    };
    auto highScaleCalc = [](float high, float limit, float zeroPoint) {
        return high / (limit - zeroPoint);
    };
    auto adjustZeroPoint = [](float zeroPoint, float upperOrLowerBound, Direction direction,
                              bool isBoundaryUpdated) -> std::optional<float> {
        if (isBoundaryUpdated) {
            if (isFloatEqual(zeroPoint, upperOrLowerBound)) {
                return std::nullopt;
            }
            zeroPoint += static_cast<float>(direction);
        }
        return std::optional(zeroPoint);
    };
    if (isFloatEqual(low, high)) {
        return std::nullopt;
    }

    auto zeroPoint = std::optional(std::round(-low / (high - low) * limit));
    // NOTE: In order to shift zeroPoint it must not be 0 or limit, because the scaled values would divide by zero.
    if (!isFloatEqual(zeroPoint.value(), 0.0f) && !isFloatEqual(zeroPoint.value(), limit)) {
        auto zeroPointTowardsLowerBound = zeroPoint.value() < limit / 2.0f;
        if (zeroPointTowardsLowerBound) {
            auto lowScale = lowScaleCalc(low, zeroPoint.value());
            auto newHigh = (limit - zeroPoint.value()) * lowScale;
            auto lowerBound = 1.0f;
            auto isCurrentHighHigher = high > newHigh;
            zeroPoint = adjustZeroPoint(zeroPoint.value(), lowerBound, Direction::ToLowerBound, isCurrentHighHigher);
            if (zeroPoint.has_value()) {
                scale = lowScaleCalc(low, zeroPoint.value());
            }
        } else {
            auto highScale = highScaleCalc(high, limit, zeroPoint.value());
            auto newLow = -zeroPoint.value() * highScale;
            auto upperBound = limit - 1.0f;
            auto isCurrentLowLower = low < newLow;
            zeroPoint = adjustZeroPoint(zeroPoint.value(), upperBound, Direction::ToUpperBound, isCurrentLowLower);
            if (zeroPoint.has_value()) {
                scale = highScaleCalc(high, limit, zeroPoint.value());
            }
        }
    }
    if (!zeroPoint.has_value()) {
        return std::nullopt;
    }

    return LowHigh{Const::createFloatConst(rewriter, loc, type, -zeroPoint.value() * scale),
                   Const::createFloatConst(rewriter, loc, type, (limit - zeroPoint.value()) * scale)};
}

mlir::Operation* getFQInputIfPresentThroughReshapes(mlir::Operation* op) {
    while (mlir::isa_and_nonnull<IE::AffineReshapeOp, IE::ReshapeOp, IE::TransposeOp>(op)) {
        op = op->getOperand(0).getDefiningOp();
    }

    return mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(op);
}

template <class ConcreteOp>
std::pair<mlir::Operation*, mlir::Operation*> getFakeQuantizeInput(ConcreteOp concreteOp) {
    auto leftInput = getFQInputIfPresentThroughReshapes(concreteOp->getOperand(0).getDefiningOp());
    auto rightInput = getFQInputIfPresentThroughReshapes(concreteOp->getOperand(1).getDefiningOp());

    return {leftInput, rightInput};
}

bool lowerFakeQuantizeU16ToU8(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) {
    const auto levelsOrLowFpTypeAttr = origOp.getLevelsOrLowFpTypeAttr();
    const auto maxLevels = QuantizationLevels::QUANT_LEVELS_8BIT;

    if (!mlir::dyn_cast<mlir::IntegerAttr>(levelsOrLowFpTypeAttr) ||
        mlir::cast<mlir::IntegerAttr>(levelsOrLowFpTypeAttr).getInt() <= maxLevels) {
        return false;
    }

    if (!origOp.getOutput().hasOneUse()) {
        return false;
    }

    auto maxLevelsAttr = rewriter.getI64IntegerAttr(maxLevels);
    auto lowFpTypeAttr = origOp.getLowFpTypeAttr();

    rewriter.setInsertionPointAfter(origOp);

    auto inLowValue = Const::getSplatValue<float>(origOp.getInputLow());
    auto inHighValue = Const::getSplatValue<float>(origOp.getInputHigh());
    auto outLowValue = Const::getSplatValue<float>(origOp.getOutputLow());
    auto outHighValue = Const::getSplatValue<float>(origOp.getOutputHigh());

    if (mlir::failed(inLowValue) || mlir::failed(inHighValue) || mlir::failed(outLowValue) ||
        mlir::failed(outHighValue)) {
        return false;
    }

    auto input = reduceLevelsZeroPoint(origOp, inLowValue.value(), inHighValue.value(), maxLevels, rewriter);
    auto output = reduceLevelsZeroPoint(origOp, outLowValue.value(), outHighValue.value(), maxLevels, rewriter);

    if (!input.has_value() || !output.has_value()) {
        return false;
    }

    const mlir::Attribute levelsOrFpTypeAttr =
            maxLevelsAttr ? mlir::cast<mlir::Attribute>(maxLevelsAttr) : mlir::cast<mlir::Attribute>(lowFpTypeAttr);

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, origOp.getInput(), input->low, input->high, output->low,
                                                    output->high, levelsOrFpTypeAttr, IE::AutoBroadcastType::NUMPY);

    return true;
}

template <class ConcreteOp>
mlir::LogicalResult LowerFakeQuantizeRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp op,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto inputs = getFakeQuantizeInput<ConcreteOp>(op);

    // Supported operations with both non-constant FQ inputs has a significant accuracy drop when lowering
    auto areBothFQ = inputs.first != nullptr && inputs.second != nullptr;
    auto areBothNotFQ = inputs.first == nullptr && inputs.second == nullptr;
    if (areBothFQ || areBothNotFQ) {
        return mlir::failure();
    }

    if (inputs.first != nullptr) {
        return lowerFakeQuantizeU16ToU8(mlir::cast<IE::FakeQuantizeOp>(inputs.first), rewriter) ? mlir::success()
                                                                                                : mlir::failure();
    }

    return lowerFakeQuantizeU16ToU8(mlir::cast<IE::FakeQuantizeOp>(inputs.second), rewriter) ? mlir::success()
                                                                                             : mlir::failure();
}

//
// U16FQConsolidationRewriter
//

class U16FQConsolidationRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    struct MatchResult {
        IE::FakeQuantizeOp fqOp = nullptr;
        IE::ConvertOp convertToFloatOp = nullptr;
        mlir::Operation* nonComputeOp = nullptr;
        IE::ConvertOp convertToIntOp = nullptr;
        IE::MultiplyOp multiplyOp = nullptr;
    };

    U16FQConsolidationRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        this->setDebugName("U16FQConsolidationRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), addOp->getName(), addOp->getLoc());

        auto matchResult = matchPattern(addOp);
        if (mlir::failed(matchResult)) {
            _log.trace("[{0}] Pattern matching failed", getDebugName());
            return mlir::failure();
        }

        auto [fqOp, convertToFloatOp, nonComputeOp, convertToIntOp, multiplyOp] = matchResult.value();

        if (mlir::failed(validateFqOp(fqOp, addOp))) {
            _log.trace("[{0}] FakeQuantizeOp validation failed", getDebugName());
            return mlir::failure();
        }
        if (mlir::failed(validateConverts(convertToFloatOp, convertToIntOp))) {
            _log.trace("[{0}] ConvertOps validation failed", getDebugName());
            return mlir::failure();
        }
        if (mlir::failed(validateScaleAndBias(multiplyOp, addOp))) {
            _log.trace("[{0}] Scale and Bias validation failed", getDebugName());
            return mlir::failure();
        }

        auto scaleVals = IE::getConst(multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>());
        auto biasVals = IE::getConst(addOp.getInput2().getDefiningOp<Const::DeclareOp>());
        mlir::Operation* lastOp = createNewFqOp(rewriter, addOp->getLoc(), fqOp, scaleVals.front(), biasVals.front());

        if (nonComputeOp != nullptr) {
            rewriter.modifyOpInPlace(nonComputeOp, [nonComputeOp = nonComputeOp, &lastOp, &rewriter]() {
                mlir::IRMapping mapper;
                mapper.map(nonComputeOp->getOperand(0), lastOp->getResult(0));
                lastOp = rewriter.clone(*nonComputeOp, mapper);
                extendOpLoc(lastOp, "copy_of_non_compute");
                inferReturnTypes(lastOp, InferShapedTypeMode::ELEM_TYPE);
            });
        }

        // Insert Convert if the new FQ output element type differs from the original Add output
        auto addOutElemType = mlir::cast<vpux::NDTypeInterface>(addOp.getType()).getElementType();
        auto lastOpElemType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType()).getElementType();
        if (addOutElemType != lastOpElemType) {
            lastOp = IE::ConvertOp::create(rewriter, addOp->getLoc(), lastOp->getResult(0), addOutElemType);
        }

        rewriter.replaceAllUsesWith(addOp, lastOp->getResult(0));
        return mlir::success();
    }

private:
    mlir::FailureOr<MatchResult> matchPattern(IE::AddOp addOp) const {
        MatchResult result;

        result.multiplyOp = addOp.getInput1().getDefiningOp<IE::MultiplyOp>();
        if (result.multiplyOp == nullptr) {
            _log.trace("[{0}] MultiplyOp not found", getDebugName());
            return mlir::failure();
        }

        mlir::Operation* currentOp = result.multiplyOp;

        if (auto convertToFloatOp = currentOp->getOperand(0).getDefiningOp<IE::ConvertOp>()) {
            currentOp = result.convertToFloatOp = convertToFloatOp;
        }

        auto nextOp = currentOp->getOperand(0).getDefiningOp();
        if (mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::TransposeOp>(nextOp)) {
            result.nonComputeOp = nextOp;
            currentOp = nextOp;
        }

        if (auto convertToIntOp = currentOp->getOperand(0).getDefiningOp<IE::ConvertOp>()) {
            currentOp = result.convertToIntOp = convertToIntOp;
        }

        result.fqOp = currentOp->getOperand(0).getDefiningOp<IE::FakeQuantizeOp>();
        if (result.fqOp == nullptr) {
            _log.trace("[{0}] FakeQuantizeOp not found", getDebugName());
            return mlir::failure();
        }

        return result;
    }

    mlir::LogicalResult validateConverts(IE::ConvertOp convertToFloatOp, IE::ConvertOp convertToIntOp) const {
        if (convertToFloatOp == nullptr && convertToIntOp == nullptr) {
            // Nothing to validate
            return mlir::success();
        }
        if (convertToFloatOp == nullptr || convertToIntOp == nullptr) {
            // Allow single convert (e.g. only convertToInt without reshape, or only convertToFloat)
            return mlir::success();
        }
        if (!convertToIntOp.getDstElemType().isUnsignedInteger(16)) {
            _log.trace("[{0}] Convert is not 16-bit", getDebugName());
            return mlir::failure();
        }
        // Allow FP16 destination (f32→ui16→f16 pattern from Q/DQ with FP16 dequantize)
        auto dstElemType = convertToFloatOp.getDstElemType();
        if (!dstElemType.isF16() && !dstElemType.isF32()) {
            _log.trace("[{0}] Convert destination is not float type", getDebugName());
            return mlir::failure();
        }
        return mlir::success();
    }

    mlir::LogicalResult validateFqOp(IE::FakeQuantizeOp fqOp, mlir::Operation* lastOp) const {
        auto levelsOrLowFpType = fqOp.getLevelsOrLowFpTypeAttr();
        auto levelsAttr = mlir::dyn_cast<mlir::IntegerAttr>(levelsOrLowFpType);
        if (levelsAttr == nullptr) {
            _log.trace("[{0}] FakeQuantize is not going to be removed (has lowFpType), skipping", getDebugName());
            return mlir::failure();
        }
        if (levelsAttr.getInt() <= IE::getMaximumQuantizationLevels(levelsAttr.getInt(), lastOp)) {
            _log.trace("[{0}] FakeQuantize is not going to be removed, skipping", getDebugName());
            return mlir::failure();
        }

        auto outLowVals = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHighVals = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        if (outLowVals == nullptr || outHighVals == nullptr) {
            _log.trace("[{0}] Output low or high values not found", getDebugName());
            return mlir::failure();
        }
        if (getShape(outLowVals).totalSize() != 1 || getShape(outHighVals).totalSize() != 1) {
            _log.trace("[{0}] Output low or high values are not scalars", getDebugName());
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult validateScaleAndBias(IE::MultiplyOp multiplyOp, IE::AddOp addOp) const {
        auto scale = multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>();
        auto bias = addOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (scale == nullptr || bias == nullptr) {
            _log.trace("[{0}] Scale or bias not found", getDebugName());
            return mlir::failure();
        }
        if (getShape(scale).totalSize() != 1 || getShape(bias).totalSize() != 1) {
            _log.trace("[{0}] Scale or bias are not scalars", getDebugName());
            return mlir::failure();
        }
        if (getShape(multiplyOp.getInput1()) != getShape(multiplyOp.getOutput())) {
            _log.trace("[{0}] Multiply input and output shapes do not match", getDebugName());
            return mlir::failure();
        }
        if (getShape(addOp.getInput1()) != getShape(addOp.getOutput())) {
            _log.trace("[{0}] Add input and output shapes do not match", getDebugName());
            return mlir::failure();
        }

        return mlir::success();
    }

    IE::FakeQuantizeOp createNewFqOp(mlir::OpBuilder& builder, mlir::Location loc, IE::FakeQuantizeOp oldFqOp,
                                     float scale, float bias) const {
        auto origOutLow = IE::getConst(oldFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>());
        auto origOutHigh = IE::getConst(oldFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>());

        auto levelsOrLowFpType = oldFqOp.getLevelsOrLowFpTypeAttr();
        auto levelsAttr = mlir::dyn_cast<mlir::IntegerAttr>(levelsOrLowFpType);
        // When FQ output range is the integer quantization grid [0, levels-1], the Mul+Add
        // following it is a dequantize. The combined Q+DQ is identity on [inputLow, inputHigh].
        // Produce an FQ with output == input to enable stripping downstream.
        if (levelsAttr != nullptr && isFloatEqual(origOutLow.front(), 0.0f) &&
            isFloatEqual(origOutHigh.front(), static_cast<float>(levelsAttr.getInt() - 1))) {
            return IE::FakeQuantizeOp::create(builder, loc, oldFqOp.getInput(), oldFqOp.getInputLow(),
                                              oldFqOp.getInputHigh(), oldFqOp.getInputLow(), oldFqOp.getInputHigh(),
                                              levelsOrLowFpType, oldFqOp.getAutoBroadcastAttr());
        }

        // General case: compute new output range from scale and bias
        auto newOutputLow = origOutLow.front() * scale + bias;
        auto newOutputHigh = origOutHigh.front() * scale + bias;

        auto outLowConstType =
                mlir::cast<mlir::RankedTensorType>(oldFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>().getType());
        auto newOutLowConst = Const::createConst(builder, oldFqOp.getLoc(), outLowConstType, ArrayRef(newOutputLow));
        auto newOutHighConst = Const::createConst(builder, oldFqOp.getLoc(), outLowConstType, ArrayRef(newOutputHigh));

        return IE::FakeQuantizeOp::create(builder, loc, oldFqOp.getInput(), oldFqOp.getInputLow(),
                                          oldFqOp.getInputHigh(), newOutLowConst, newOutHighConst,
                                          oldFqOp.getLevelsOrLowFpTypeAttr(), oldFqOp.getAutoBroadcastAttr());
    }

private:
    Logger _log;
};

class U16FQConvertToQuantizeRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    U16FQConvertToQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        this->setDebugName("U16FQConvertToQuantizeRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp op, mlir::PatternRewriter& rewriter) const final {
        const auto dstElemType = op.getDstElemType();
        if (!dstElemType.isUnsignedInteger(16)) {
            _log.trace("[{0}] Destination element type is not unsigned 16-bit integer", getDebugName());
            return mlir::failure();
        }
        auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op.getInput().getDefiningOp());
        if (mlir::failed(validateFqOp(fqOp))) {
            return mlir::failure();
        }

        auto [scale, bias] =
                getWeightsAndBiases(fqOp.getInputLow(), fqOp.getInputHigh(), fqOp.getOutputLow(), fqOp.getOutputHigh());
        const auto storageParams = getStorageParams(dstElemType);
        if (mlir::failed(storageParams)) {
            return mlir::failure();
        }
        const auto [storageMin, storageMax, storageType] = *storageParams;
        const auto convertInputElemType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
        auto quantType = mlir::quant::UniformQuantizedType::get(
                dstElemType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
                convertInputElemType, 1.0 / scale.front(), bias.front(), storageMin, storageMax);
        auto quantizeOp = IE::QuantizeOp::create(rewriter, takeOpLoc(op, "quant"), fqOp.getInput(), quantType);
        auto quantizeCast =
                IE::QuantizeCastOp::create(rewriter, takeOpLoc(op, "quant_cast"), quantizeOp.getResult(), dstElemType);
        rewriter.replaceOp(op, quantizeCast.getResult());
        return mlir::success();
    }

private:
    mlir::LogicalResult validateFqOp(IE::FakeQuantizeOp fqOp) const {
        if (fqOp == nullptr) {
            return mlir::failure();
        }
        const auto levelsOrLowFpType = fqOp.getLevelsOrLowFpTypeAttr();

        auto levels = mlir::dyn_cast<mlir::IntegerAttr>(levelsOrLowFpType);
        if (levels == nullptr) {
            return mlir::failure();
        }
        if (levels.getInt() <= QuantizationLevels::QUANT_LEVELS_8BIT) {
            _log.trace("[{0}] Levels are not greater than 8-bit quantization levels", getDebugName());
            return mlir::failure();
        }
        if (getShape(fqOp.getOutputLow()).totalSize() != 1 || getShape(fqOp.getOutputHigh()).totalSize() != 1) {
            _log.trace("[{0}] Output low or high values are not scalars", getDebugName());
            return mlir::failure();
        }
        if (areFQValsEqual(fqOp.getInputLow(), fqOp.getInputHigh(), fqOp.getOutputLow(), fqOp.getOutputHigh())) {
            _log.trace("[{0}] FakeQuantize input and output low/high values are equal", getDebugName());
            return mlir::failure();
        }
        return mlir::success();
    }

private:
    Logger _log;
};

void HandleU16FakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<U16FQConsolidationRewriter>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<U16FQConvertToQuantizeRewriter>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    auto module = getModuleOp(func);

    auto enableU16ToU8Lowering = config::hasEnableQDQOptimizationAggressive(module);
    if (enableU16ToU8Lowering) {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<LowerFakeQuantizeRewriter<IE::MatMulOp>>(&ctx, _log);
        patterns.add<LowerFakeQuantizeRewriter<IE::FullyConnectedOp>>(&ctx, _log);
        patterns.add<LowerFakeQuantizeRewriter<IE::ConvolutionOp>>(&ctx, _log);
        patterns.add<LowerFakeQuantizeRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    // RemoveU16FakeQuantizeRewriter uses a separate walk so it runs after the u16-to-u8
    // lowering above (which starts from other ops and would otherwise apply this pattern first).
    //
    // Must run before RemoveU16FakeQuantizeRewriter, which could otherwise simplify the
    // parent FQ away and make the chain unmatchable.
    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<ReplaceChildU16FakeQuantizeWithReluRewriter>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveU16FakeQuantizeRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createHandleU16FakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleU16FakeQuantizePass(Logger log) {
    return std::make_unique<HandleU16FakeQuantizePass>(log);
}
