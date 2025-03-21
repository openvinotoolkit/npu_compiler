//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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
    explicit HandleU16FakeQuantizePass(const IE::TransformOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);
        initializeFromOptions();
    }

public:
    class RemoveU16FakeQuantizeRewriter;

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    // Initialize fields from pass options
    void initializeFromOptions();

    void safeRunOnFunc() final;

private:
    bool _enableU16FQToScaleShiftConversion = false;
};

mlir::LogicalResult HandleU16FakeQuantizePass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void HandleU16FakeQuantizePass::initializeFromOptions() {
    if (enableU16FQToScaleShiftConversion.hasValue()) {
        _enableU16FQToScaleShiftConversion = enableU16FQToScaleShiftConversion.getValue();
    }
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
    RemoveU16FakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log, bool enableU16FQToScaleShiftConversion)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx),
              _log(log),
              _enableU16FQToScaleShiftConversion(enableU16FQToScaleShiftConversion) {
        setDebugName("RemoveU16FakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult convertFQToScaleShift(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    bool _enableU16FQToScaleShiftConversion = false;
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
                                                      /*output_channels=*/nullptr,
                                                      /*input_channels=*/nullptr);
    auto addOp = rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, multiplyOp.getType(), multiplyOp.getOutput(),
                                                        biasesConst, IE::AutoBroadcastType::NUMPY,
                                                        /*post_op=*/nullptr,
                                                        /*clamp=*/nullptr,
                                                        /*output_channels=*/nullptr,
                                                        /*input_channels=*/nullptr);
    extendOpLoc(addOp, "as_add");
    return mlir::success();
}

mlir::LogicalResult HandleU16FakeQuantizePass::RemoveU16FakeQuantizeRewriter::matchAndRewrite(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto levels = origOp.getLevels();

    // Maximum number of levels that don't exceeds I8/U8 storage type
    if (!levels.has_value() || *levels <= MAX_LEVELS) {
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

    if (_enableU16FQToScaleShiftConversion) {
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

void HandleU16FakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveU16FakeQuantizeRewriter>(&ctx, _log, _enableU16FQToScaleShiftConversion);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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

std::unique_ptr<mlir::Pass> vpux::IE::createHandleU16FakeQuantizePass(const IE::TransformOptions& options, Logger log) {
    return std::make_unique<HandleU16FakeQuantizePass>(options, log);
}
