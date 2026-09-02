//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE::arch50xx {
#define GEN_PASS_DECL_CONVERTFAKECONVERTTODYNAMICDEQUANTIZE
#define GEN_PASS_DEF_CONVERTFAKECONVERTTODYNAMICDEQUANTIZE
#include "vpux/compiler/NPU50XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch50xx

using namespace vpux;

namespace {

SmallVector<float> getContentValues(const Const::Content& content) {
    return content.isSplat() ? SmallVector<float>{content.getSplatValue<float>()}
                             : SmallVector<float>(content.getValues<float>());
}

size_t getBroadcastIndex(size_t inputIndex, ArrayRef<int64_t> inputShape, ArrayRef<int64_t> parameterShape) {
    VPUX_THROW_UNLESS(parameterShape.size() <= inputShape.size(), "Parameter rank exceeds input rank");

    size_t parameterIndex = 0;
    size_t parameterStride = 1;
    size_t inputStride = 1;
    for (size_t reverseDim = 0; reverseDim < inputShape.size(); ++reverseDim) {
        const auto inputDim = inputShape.size() - reverseDim - 1;
        const auto inputCoordinate = (inputIndex / inputStride) % checked_cast<size_t>(inputShape[inputDim]);
        inputStride *= checked_cast<size_t>(inputShape[inputDim]);

        if (reverseDim >= parameterShape.size()) {
            continue;
        }

        const auto parameterDim = parameterShape.size() - reverseDim - 1;
        const auto parameterDimSize = parameterShape[parameterDim];
        VPUX_THROW_UNLESS(parameterDimSize == 1 || parameterDimSize == inputShape[inputDim],
                          "Parameter shape is not broadcastable to input shape");
        const auto parameterCoordinate = parameterDimSize == 1 ? 0 : inputCoordinate;
        parameterIndex += parameterCoordinate * parameterStride;
        parameterStride *= checked_cast<size_t>(parameterDimSize);
    }

    return parameterIndex;
}

float getBroadcastValue(ArrayRef<float> values, size_t inputIndex, ArrayRef<int64_t> inputShape,
                        ArrayRef<int64_t> parameterShape) {
    return values.size() == 1 ? values.front() : values[getBroadcastIndex(inputIndex, inputShape, parameterShape)];
}

mlir::LogicalResult rewriteWeightsAsDynamicDequantize(IE::FakeConvertOp origOp, Const::DeclareOp inputConst,
                                                      mlir::PatternRewriter& rewriter, const Logger& log) {
    auto scaleConst = origOp.getScale().getDefiningOp<Const::DeclareOp>();
    auto shiftConst = origOp.getShift() != nullptr
                              ? mlir::dyn_cast_if_present<Const::DeclareOp>(origOp.getShift().getDefiningOp())
                              : nullptr;
    if (scaleConst == nullptr || (origOp.getShift() != nullptr && shiftConst == nullptr)) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto inputElemType = inputType.getElementType();
    const auto scaleShape = getShape(scaleConst.getOutput()).raw();

    const auto inputValues = getContentValues(inputConst.getContent());
    const auto scaleValues = getContentValues(scaleConst.getContent());
    SmallVector<float> shiftValues{0.0f};
    ArrayRef<int64_t> shiftShape;
    if (shiftConst != nullptr) {
        shiftValues = getContentValues(shiftConst.getContent());
        shiftShape = getShape(shiftConst.getOutput()).raw();
    }

    // DynamicDequantize dequantizes as (input - zp) * scale, so it must store the *inverse* of the FakeConvert
    // scale. The inverse-scale is kept in f32 (the op's scale operand accepts f32) to store the reciprocal as
    // faithfully as the type allows.
    auto* ctx = rewriter.getContext();
    const auto f32Type = mlir::Float32Type::get(ctx);

    SmallVector<float> inverseScaleValues(scaleValues.size());
    for (auto index : irange(scaleValues.size())) {
        if (scaleValues[index] == 0.0f) {
            // A zero scale cannot be inverted; leave the FakeConvert untouched for a later, non-inverting lowering.
            log.trace("FakeConvert has a zero scale; cannot form an inverse-scale DynamicDequantize");
            return mlir::failure();
        }
        inverseScaleValues[index] = 1.0f / scaleValues[index];
    }

    Const::ContentAttr zpContentAttr = nullptr;
    mlir::RankedTensorType zpType = nullptr;
    if (shiftConst != nullptr) {
        SmallVector<int64_t> alignedZpShape(inputShape.size(), 1);
        std::copy(shiftShape.begin(), shiftShape.end(), alignedZpShape.end() - shiftShape.size());
        zpType = mlir::RankedTensorType::get(alignedZpShape, origOp.getDstType());

        SmallVector<float> requestedZpValues(shiftValues.size());
        std::transform(shiftValues.begin(), shiftValues.end(), requestedZpValues.begin(), [](float value) {
            return -value;
        });
        zpContentAttr = Const::createFloatContentAttr(rewriter, takeOpLoc(origOp, "ddq_zp"), zpType, requestedZpValues);

        // Read the f8-rounded zero-point back and require it to be bit-exact to `-shift`.
        const auto roundedZpValues = getContentValues(zpContentAttr.fold());
        for (auto index : irange(requestedZpValues.size())) {
            const auto rounded = roundedZpValues.size() == 1 ? roundedZpValues.front() : roundedZpValues[index];
            if (rounded != requestedZpValues[index]) {
                log.trace("FakeConvert shift is not exactly representable in {0}; deferring to FakeQuantize fallback",
                          origOp.getDstType());
                return mlir::failure();
            }
        }
    }

    // Raw weights = fp8_round_and_clamp(value * scale - shift).
    const auto inputCount = checked_cast<size_t>(inputShape.totalSize());
    SmallVector<float> quantizedValues(
            inputValues.size() == 1 && scaleValues.size() == 1 && shiftValues.size() == 1 ? 1 : inputCount);
    for (auto inputIndex : irange(quantizedValues.size())) {
        const auto value = inputValues.size() == 1 ? inputValues.front() : inputValues[inputIndex];
        const auto scale = getBroadcastValue(scaleValues, inputIndex, inputShape.raw(), scaleShape);
        const auto shift =
                shiftConst == nullptr ? 0.0f : getBroadcastValue(shiftValues, inputIndex, inputShape.raw(), shiftShape);
        quantizedValues[inputIndex] = value * scale - shift;
    }

    const auto rawWeightsType = mlir::cast<mlir::RankedTensorType>(inputType.changeElemType(origOp.getDstType()));
    auto rawWeightsContent =
            Const::createFloatContentAttr(rewriter, takeOpLoc(origOp, "ddq_weights"), rawWeightsType, quantizedValues);
    auto rawWeights = rewriter.create<Const::DeclareOp>(takeOpLoc(origOp, "ddq_weights"), rawWeightsType,
                                                        std::move(rawWeightsContent));

    const auto inverseScaleType = mlir::RankedTensorType::get(scaleShape, f32Type);
    auto inverseScale =
            Const::createFloatConst(rewriter, takeOpLoc(origOp, "ddq_scale"), inverseScaleType, inverseScaleValues);
    mlir::Value zp = nullptr;
    if (zpContentAttr != nullptr) {
        zp = rewriter.create<Const::DeclareOp>(takeOpLoc(origOp, "ddq_zp"), zpType, std::move(zpContentAttr));
    }

    auto dynamicDequantize =
            rewriter.replaceOpWithNewOp<IE::DynamicDequantizeOp>(origOp, rawWeights, inverseScale, zp, inputElemType);
    dynamicDequantize->setAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR, mlir::UnitAttr::get(rewriter.getContext()));
    return mlir::success();
}

class ConvertFakeConvertToDynamicDequantize final :
        public IE::arch50xx::impl::ConvertFakeConvertToDynamicDequantizeBase<ConvertFakeConvertToDynamicDequantize> {
public:
    explicit ConvertFakeConvertToDynamicDequantize(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    class FakeConvertRewriter;
};

class ConvertFakeConvertToDynamicDequantize::FakeConvertRewriter final :
        public mlir::OpRewritePattern<IE::FakeConvertOp> {
public:
    FakeConvertRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeConvertOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::FakeConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        if (!IE::isOnWeightsPath(origOp, origOp.getDstType(), _log.nest())) {
            return mlir::failure();
        }

        auto inputConst = origOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (inputConst == nullptr) {
            return mlir::failure();
        }

        return rewriteWeightsAsDynamicDequantize(origOp, inputConst, rewriter, _log.nest());
    }

private:
    Logger _log;
};

void ConvertFakeConvertToDynamicDequantize::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FakeConvertRewriter>(&ctx, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::arch50xx::createConvertFakeConvertToDynamicDequantizePass(Logger log) {
    return std::make_unique<ConvertFakeConvertToDynamicDequantize>(log);
}
