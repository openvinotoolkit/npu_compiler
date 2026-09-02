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

#include <mlir/Dialect/Quant/IR/Quant.h>
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

SmallVector<mlir::Operation*> getMPEEngineMappableConsumers(mlir::Operation* op) {
    const auto& strategyFactory = IE::getIEStrategyFactory(op->getContext());
    if (strategyFactory == nullptr) {
        return {};
    }
    const auto multiConsumerStrategy = strategyFactory->getMultiConsumerStrategy();
    if (multiConsumerStrategy == nullptr) {
        return {};
    }
    return multiConsumerStrategy->getMPEEngineMappableConsumers(op);
}

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

    auto multiplyOpResult =
            rewriter.createOrFold<IE::MultiplyOp>(takeOpLoc(origOp, "as_mul"), origOp.getType(), origOp.getInput(),
                                                  weightsConst, IE::AutoBroadcastType::NUMPY,
                                                  /*post_op=*/nullptr,
                                                  /*clamp=*/nullptr,
                                                  /*outputPadding=*/nullptr,
                                                  /*inputPadding=*/nullptr);
    auto addOpResult = rewriter.createOrFold<IE::AddOp>(takeOpLoc(origOp, "as_add"), multiplyOpResult.getType(),
                                                        multiplyOpResult, biasesConst, IE::AutoBroadcastType::NUMPY,
                                                        /*post_op=*/nullptr,
                                                        /*clamp=*/nullptr,
                                                        /*outputPadding=*/nullptr,
                                                        /*inputPadding=*/nullptr);
    rewriter.replaceAllOpUsesWith(origOp, addOpResult);
    return mlir::success();
}

mlir::LogicalResult HandleU16FakeQuantizePass::RemoveU16FakeQuantizeRewriter::matchAndRewrite(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto moduleOp = getModuleOp(origOp);
    auto setAdaptiveStrippingEnabled = config::hasEnableAdaptiveStripping(moduleOp);
    auto levels = origOp.getLevels();
    auto maxLevels = IE::getMaximumQuantizationLevels(levels.value_or(QuantizationLevels::QUANT_LEVELS_8BIT), origOp);
    // Maximum number of levels that don't exceeds I8/U8 storage type
    if (!levels.has_value() || *levels <= maxLevels) {
        return mlir::failure();
    }

    // Analyze consumers that the target's MPE engine can map. Such consumers may require the U16 FQ to be preserved.
    const auto mpeEngineMappableConsumers = getMPEEngineMappableConsumers(origOp);
    if (!mpeEngineMappableConsumers.empty()) {
        const bool allUsersMappable = llvm::all_of(origOp.getOutput().getUsers(), [&](mlir::Operation* userOp) {
            return llvm::is_contained(mpeEngineMappableConsumers, userOp);
        });
        if (allUsersMappable) {
            // Every consumer can be mapped, so the U16 FQ must be kept for all of them.
            _log.trace("[{0}] Keeping U16 FQ at '{1}' since all its consumers can be mapped by the MPE engine",
                       getDebugName(), origOp->getLoc());
            return mlir::failure();
        }
    }

    bool consumersSplit = false;
    const auto splitMappableConsumers = [&]() {
        if (mpeEngineMappableConsumers.empty() || consumersSplit) {
            return;
        }
        // Mixed consumers - clone a dedicated U16 FQ for different MPE engine-mappable users and redirect them
        // to it, origOp is then left feeding only the non-mappable users, so it falls through to
        // the regular removal/conversion path below
        const auto isMappableUser = [&](mlir::OpOperand& opOperand) {
            return llvm::is_contained(mpeEngineMappableConsumers, opOperand.getOwner());
        };
        rewriter.setInsertionPoint(origOp);
        auto clonedFqOp = mlir::cast<IE::FakeQuantizeOp>(rewriter.clone(*origOp.getOperation()));
        extendOpLoc(clonedFqOp, "mappable_consumers");
        rewriter.replaceUsesWithIf(origOp, clonedFqOp.getOutput(), isMappableUser);

        if (clonedFqOp.getOutput().use_empty()) {
            _log.trace("[{0}] No mappable consumers redirected at '{1}'; discarding cloned U16 FQ", getDebugName(),
                       origOp->getLoc());
            rewriter.eraseOp(clonedFqOp);
            return;
        }
        consumersSplit = true;
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
        auto clonedFoldedConstant =
                rewriter.create<Const::DeclareOp>(origOp.getLoc(), newContentAttr.getType(), std::move(newContentAttr));

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
                if (candidate.getLevels().has_value() && *candidate.getLevels() > maxLevels) {
                    childFqOp = candidate;
                    break;
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

        auto childLevels = childFqOp.getLevels();
        auto childMaxLevels = IE::getMaximumQuantizationLevels(
                childLevels.value_or(QuantizationLevels::QUANT_LEVELS_8BIT), childFqOp);
        if (!childLevels.has_value() || *childLevels <= childMaxLevels) {
            return mlir::failure();
        }

        if (!getMPEEngineMappableConsumers(childFqOp).empty()) {
            _log.trace("[{0}] Keeping child U16 FQ at '{1}' since a consumer can be mapped by the MPE engine",
                       getDebugName(), childFqOp->getLoc());
            return mlir::failure();
        }

        // Parent must also be a U16 FQ.
        auto parentFqOp = childFqOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
        if (parentFqOp == nullptr) {
            return mlir::failure();
        }
        auto parentLevels = parentFqOp.getLevels();
        auto parentMaxLevels = IE::getMaximumQuantizationLevels(
                parentLevels.value_or(QuantizationLevels::QUANT_LEVELS_8BIT), parentFqOp);
        if (!parentLevels.has_value() || *parentLevels <= parentMaxLevels) {
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
            lastOp = rewriter.create<IE::ConvertOp>(addOp->getLoc(), lastOp->getResult(0), addOutElemType);
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
        auto levels = fqOp.getLevels();
        if (!levels.has_value() || *levels <= IE::getMaximumQuantizationLevels(*levels, lastOp)) {
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

        auto levels = oldFqOp.getLevels();
        // When FQ output range is the integer quantization grid [0, levels-1], the Mul+Add
        // following it is a dequantize. The combined Q+DQ is identity on [inputLow, inputHigh].
        // Produce an FQ with output == input to enable stripping downstream.
        if (levels.has_value() && isFloatEqual(origOutLow.front(), 0.0f) &&
            isFloatEqual(origOutHigh.front(), static_cast<float>(*levels - 1))) {
            return builder.create<IE::FakeQuantizeOp>(loc, oldFqOp.getInput(), oldFqOp.getInputLow(),
                                                      oldFqOp.getInputHigh(), oldFqOp.getInputLow(),
                                                      oldFqOp.getInputHigh(), oldFqOp.getLevelsAttr(),
                                                      oldFqOp.getLowFpTypeAttr(), oldFqOp.getAutoBroadcastAttr());
        }

        // General case: compute new output range from scale and bias
        auto newOutputLow = origOutLow.front() * scale + bias;
        auto newOutputHigh = origOutHigh.front() * scale + bias;

        auto outLowConstType =
                mlir::cast<mlir::RankedTensorType>(oldFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>().getType());
        auto newOutLowConst = Const::createConst(builder, oldFqOp.getLoc(), outLowConstType, ArrayRef(newOutputLow));
        auto newOutHighConst = Const::createConst(builder, oldFqOp.getLoc(), outLowConstType, ArrayRef(newOutputHigh));

        return builder.create<IE::FakeQuantizeOp>(
                loc, oldFqOp.getInput(), oldFqOp.getInputLow(), oldFqOp.getInputHigh(), newOutLowConst, newOutHighConst,
                oldFqOp.getLevelsAttr(), oldFqOp.getLowFpTypeAttr(), oldFqOp.getAutoBroadcastAttr());
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
        auto quantizeOp = rewriter.create<IE::QuantizeOp>(takeOpLoc(op, "quant"), fqOp.getInput(), quantType);
        auto quantizeCast =
                rewriter.create<IE::QuantizeCastOp>(takeOpLoc(op, "quant_cast"), quantizeOp.getResult(), dstElemType);
        rewriter.replaceOp(op, quantizeCast.getResult());
        return mlir::success();
    }

private:
    mlir::LogicalResult validateFqOp(IE::FakeQuantizeOp fqOp) const {
        if (fqOp == nullptr) {
            return mlir::failure();
        }
        const auto levels = fqOp.getLevels();
        if (!levels.has_value() || *levels <= QuantizationLevels::QUANT_LEVELS_8BIT) {
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
