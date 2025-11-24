//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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
    for (auto& value : inputValues) {
        value = fakeQuantize(value, inLowValue, inHighValue, outLowValue, outHighValue, *(fqOp.getLevels()));
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
    const auto weightsConst = Const::createConst(rewriter, origOp->getLoc(), newShape, ArrayRef(weightsVec));
    const auto biasesConst = Const::createConst(rewriter, origOp->getLoc(), newShape, ArrayRef(biasesVec));

    auto multiplyOp = rewriter.create<IE::MultiplyOp>(takeOpLoc(origOp, "as_mul"), origOp.getType(), origOp.getInput(),
                                                      weightsConst, IE::AutoBroadcastType::NUMPY,
                                                      /*post_op=*/nullptr,
                                                      /*clamp=*/nullptr,
                                                      /*outputPadding=*/nullptr,
                                                      /*inputPadding=*/nullptr);
    auto addOp = rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, multiplyOp.getType(), multiplyOp.getOutput(),
                                                        biasesConst, IE::AutoBroadcastType::NUMPY,
                                                        /*post_op=*/nullptr,
                                                        /*clamp=*/nullptr,
                                                        /*outputPadding=*/nullptr,
                                                        /*inputPadding=*/nullptr);
    extendOpLoc(addOp, "as_add");
    return mlir::success();
}

mlir::LogicalResult HandleU16FakeQuantizePass::RemoveU16FakeQuantizeRewriter::matchAndRewrite(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto moduleOp = getModuleOp(origOp);
    auto setAdaptiveStrippingEnabled = config::hasEnableAdaptiveStripping(moduleOp);
    auto levels = origOp.getLevels();
    auto maxLevels = QuantizationLevels::QUANT_LEVELS_8BIT;
    // Maximum number of levels that don't exceeds I8/U8 storage type
    if (!levels.has_value() || *levels <= maxLevels) {
        return mlir::failure();
    }

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
        Const::ContentSetup newContentAttrSetup(fqInputContentType);
        auto newContentAttr = Const::ContentAttr::get(newFoldedBaseContent, newContentAttrSetup);
        auto clonedFoldedConstant =
                rewriter.create<Const::DeclareOp>(origOp.getLoc(), newContentAttr.getType(), std::move(newContentAttr));

        // Apply fakeQuantize on the constant
        auto newFqInput = applyU16FakequantizeOnConstant(rewriter, origOp, clonedFoldedConstant);
        rewriter.replaceOp(origOp, newFqInput);
        return mlir::success();
    }

    if (setAdaptiveStrippingEnabled) {
        auto childOp = *origOp.getOutput().getUsers().begin();
        auto childFqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(childOp);
        if (childFqOp != nullptr && *childFqOp.getLevels() > maxLevels) {
            const auto inLowValue = IE::getConst(origOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
            const auto outLowValue = IE::getConst(origOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
            const auto inHighValue = IE::getConst(origOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
            const auto outHighValue = IE::getConst(origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];
            const auto childInLowValue = IE::getConst(childFqOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
            const auto childOutLowValue = IE::getConst(childFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
            const auto childInHighValue = IE::getConst(childFqOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
            const auto childOutHighValue = IE::getConst(childFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];

            // ParentOp -> FQ1 U16 (il=ol=0) -> FQ2 U16 -> Op => ParentOp -> ReLU -> Op
            // ParentOp -> FQ1 U16 -> FQ2 U16 (il=ol=0) -> Op => ParentOp -> ReLU -> Op
            if (IE::isPerTensorFQ({origOp}) && isFloatEqual(inLowValue, outLowValue) &&
                isFloatEqual(inHighValue, outHighValue) && isFloatEqual(inLowValue, 0.0f) && origOp->hasOneUse()) {
                rewriter.replaceOpWithNewOp<IE::ReLUOp>(origOp, fqInput);
                return mlir::success();
            } else {
                if (IE::isPerTensorFQ({childFqOp}) && isFloatEqual(childInLowValue, childOutLowValue) &&
                    isFloatEqual(childInHighValue, childOutHighValue) && isFloatEqual(childInLowValue, 0.0f)) {
                    rewriter.replaceOpWithNewOp<IE::ReLUOp>(childFqOp, fqInput);
                    return mlir::success();
                }
            }
        }
        // In case the FakeQuantize has values in_low != out_low or in_high != out_high it can be replaced with a
        // ScaleShift op
        if (!areFQValsEqual(origOp.getInputLow(), origOp.getInputHigh(), origOp.getOutputLow(),
                            origOp.getOutputHigh())) {
            return convertFQToScaleShift(origOp, rewriter);
        }
    } else {
        // In case the FakeQuantize is per tensor and the input and output low is equal to 0 it is replaced with a
        // ReLu activation function otherwise the FakeQuantize is completely removed
        if (IE::isPerTensorFQ({origOp})) {
            const auto inLowValue = IE::getConst(origOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
            const auto outLowValue = IE::getConst(origOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
            const auto inHighValue = IE::getConst(origOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
            const auto outHighValue = IE::getConst(origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];
            if (isFloatEqual(inLowValue, outLowValue) && isFloatEqual(inHighValue, outHighValue) &&
                isFloatEqual(inLowValue, 0.0f)) {
                rewriter.replaceOpWithNewOp<IE::ReLUOp>(origOp, fqInput);
                return mlir::success();
            }
        }
    }

    rewriter.replaceOp(origOp, fqInput);
    return mlir::success();
}

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
    const auto levels = origOp.getLevels();
    const auto maxLevels = QuantizationLevels::QUANT_LEVELS_8BIT;

    if (!levels.has_value() || *levels <= maxLevels) {
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

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, origOp.getInput(), input->low, input->high, output->low,
                                                    output->high, maxLevelsAttr, lowFpTypeAttr,
                                                    IE::AutoBroadcastType::NUMPY);

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

void HandleU16FakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = getModuleOp(func);

    auto failedToPatternMatch = false;
    auto enableU16ToU8Lowering = config::hasEnableQDQOptimizationAggressive(module);
    if (enableU16ToU8Lowering) {
        // NOTE: LowerFakeQuantizeRewriter must take priority over RemoveU16FakeQuantizeRewriter
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<LowerFakeQuantizeRewriter<IE::MatMulOp>>(&ctx, _log);
        patterns.add<LowerFakeQuantizeRewriter<IE::FullyConnectedOp>>(&ctx, _log);
        patterns.add<LowerFakeQuantizeRewriter<IE::ConvolutionOp>>(&ctx, _log);
        patterns.add<LowerFakeQuantizeRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
        if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
            failedToPatternMatch = true;
        }
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveU16FakeQuantizeRewriter>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        failedToPatternMatch = true;
    }

    if (failedToPatternMatch) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleU16FakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleU16FakeQuantizePass(Logger log) {
    return std::make_unique<HandleU16FakeQuantizePass>(log);
}
