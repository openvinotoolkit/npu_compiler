//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE::arch50xx {
#define GEN_PASS_DECL_CONVERTFAKECONVERTTOFAKEQUANTIZE
#define GEN_PASS_DEF_CONVERTFAKECONVERTTOFAKEQUANTIZE
#include "vpux/compiler/NPU50XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch50xx

using namespace vpux;

namespace {

mlir::FailureOr<std::tuple<double, double>> getMinMax(IE::FakeConvertOp origOp, const Logger& log) {
    const auto destinationType = origOp.getDstType();
    if (!vpux::isFloat8(destinationType)) {
        log.error("Unsupported FP data type");
        return mlir::failure();
    }
    return getRepresentableRange(destinationType);
}

Const::ContentAttr getScale(IE::FakeConvertOp origOp) {
    const auto scale = origOp.getScale();
    if (scale == nullptr) {
        return {};
    }

    auto scaleCst = scale.getDefiningOp<Const::DeclareOp>();
    return scaleCst == nullptr ? Const::ContentAttr{} : scaleCst.getContentAttr();
}

Const::ContentAttr getShift(IE::FakeConvertOp origOp) {
    const auto shift = origOp.getShift();
    if (shift == nullptr) {
        return {};
    }

    auto shiftCst = shift.getDefiningOp<Const::DeclareOp>();
    return shiftCst == nullptr ? Const::ContentAttr{} : shiftCst.getContentAttr();
}

bool isFakeConvertToFakeQuantizeBeneficial(IE::FakeConvertOp origOp) {
    const auto scale = origOp.getScale();
    const auto scaleCst = scale.getDefiningOp<Const::DeclareOp>();
    if (scaleCst == nullptr) {
        return false;
    }

    if (const auto shift = origOp.getShift()) {
        const auto shiftCst = shift.getDefiningOp<Const::DeclareOp>();
        if (shiftCst == nullptr) {
            return false;
        }
    }

    return true;
}

//
// ConvertFakeConvertToFakeQuantize
//

class ConvertFakeConvertToFakeQuantize final :
        public IE::arch50xx::impl::ConvertFakeConvertToFakeQuantizeBase<ConvertFakeConvertToFakeQuantize> {
public:
    explicit ConvertFakeConvertToFakeQuantize(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    class FakeConvertRewriter;
};

class ConvertFakeConvertToFakeQuantize::FakeConvertRewriter final : public mlir::OpRewritePattern<IE::FakeConvertOp> {
public:
    FakeConvertRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeConvertOp>(ctx), _log(log) {
    }

private:
    Logger _log;

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeConvertOp origOp, mlir::PatternRewriter& rewriter) const final;
};

/*
                                              inL = fp8_min / scale + shift
       data  scale shift  fp8_dst_type     data   |   inH = fp8_max / scale + shift
        |     |     |      |                |     |    |   outL = inL
        |     |     |      |                |     |    |    |   outH = inH
        |     |     |      |                |     |    |    |     |     low_fp8_type
        |     |     |      |                |     |    |    |     |      |
        v     v     v      v                v     v    v    v     v      v
      +-----------------------+            +------------------------------+
      |      FakeConvert      |   =====>   |         FakeQuantize         |
      +-----------------------+            +------------------------------+
                  |                                        |
                  v                                        v
*/

mlir::LogicalResult ConvertFakeConvertToFakeQuantize::FakeConvertRewriter::matchAndRewrite(
        IE::FakeConvertOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

    // Note: weights-path FakeConverts are handled first by ConvertFakeConvertToDynamicDequantize. It only converts
    // the cases it can represent exactly (e.g. an f8-representable shift) and leaves the rest as FakeConvert, which
    // this pass then converts to FakeQuantize.
    if (!isFakeConvertToFakeQuantizeBeneficial(origOp)) {
        return mlir::failure();
    }

    // Get fp8 min and max
    const auto minMaxTensors = getMinMax(origOp, _log.nest());
    if (mlir::failed(minMaxTensors)) {
        return mlir::failure();
    }
    const auto [inLow, inHigh] = minMaxTensors.value();

    const auto inputElemType = origOp.getInput().getType().getElementType();
    const auto inputRank = origOp.getInput().getType().getShape().size();
    const auto inputShape = SmallVector<int64_t>(inputRank, 1);
    const auto storageType = mlir::RankedTensorType::get(inputShape, inputElemType);

    const auto scaleShifted = IE::applyScaleShift(origOp.getContext(), getScale(origOp), getShift(origOp), inLow,
                                                  inHigh, storageType, _log.nest());
    if (mlir::failed(scaleShifted)) {
        _log.error("Failed to apply scale-shift");
        return mlir::failure();
    }
    const auto& [outLow, outHigh] = scaleShifted.value();

    // Create FakeQuantize intervals as:
    // (low, high) = (fp8_min, fp8_max) / scale + shift
    const auto outLowValues = to_small_vector(outLow.getValues<float>());
    const auto outHighValues = to_small_vector(outHigh.getValues<float>());
    const auto outStorageType = mlir::cast<mlir::RankedTensorType>(outLow.getType());
    const auto inOutLowConst =
            Const::createFloatConst(rewriter, takeOpLoc(origOp, "out_low"), outStorageType, ArrayRef(outLowValues));
    const auto inOutHighConst =
            Const::createFloatConst(rewriter, takeOpLoc(origOp, "out_high"), outStorageType, ArrayRef(outHighValues));

    // Create a dummy FakeQuantize, with equal input and output ranges
    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(origOp.getContext(), IE::AutoBroadcastType::NUMPY);
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, origOp.getInput(), inOutLowConst, inOutHighConst,
                                                    inOutLowConst, inOutHighConst, nullptr, origOp.getDstTypeAttr(),
                                                    broadCastAttr);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertFakeConvertToFakeQuantize::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FakeConvertRewriter>(&ctx, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createConvertFakeConvertToFakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch50xx::createConvertFakeConvertToFakeQuantizePass(Logger log) {
    return std::make_unique<ConvertFakeConvertToFakeQuantize>(log);
}
