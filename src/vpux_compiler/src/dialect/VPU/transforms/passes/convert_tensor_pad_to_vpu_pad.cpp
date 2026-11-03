//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APFloat.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTTENSORPADTOVPUPAD
#define GEN_PASS_DEF_CONVERTTENSORPADTOVPUPAD
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// Helpers
//

// Dynamic padding (runtime low/high operands, or ShapedType::kDynamic static entries) keeps the
// upstream tensor bufferization path. Only static-padding pads are rewritten.
bool hasDynamicPadding(mlir::tensor::PadOp padOp) {
    return !padOp.getLow().empty() || !padOp.getHigh().empty() ||
           llvm::any_of(padOp.getStaticLow(), mlir::ShapedType::isDynamic) ||
           llvm::any_of(padOp.getStaticHigh(), mlir::ShapedType::isDynamic);
}

// nofold forbids removing a pad whose source and result types are equal, which is exactly the
// condition under which VPU::PadOp::fold removes the rewritten op.
bool hasUnexpressibleNofold(mlir::tensor::PadOp padOp) {
    return padOp.getNofold() && padOp.getSourceType() == padOp.getResultType();
}

// The input must be a ranked tensor that is either static or carries an upper-bound encoding.
// Unbounded dynamic input is rejected because the padded result bounds cannot be inferred.
bool hasRewritableInput(mlir::tensor::PadOp padOp) {
    const auto inType = mlir::dyn_cast<mlir::RankedTensorType>(padOp.getSource().getType());
    if (inType == nullptr) {
        return false;
    }
    return inType.hasStaticShape() || !vpux::getBounds(inType).empty();
}

// Derives the value that fills the padded area, as the f64 VPU.Pad expects. Only IE::PadMode::CONSTANT
// is expressible: the pad body must yield a value independent of its index block arguments, so an
// index-dependent body (mirror/reflect/symmetric-style padding) is rejected.
//
// SCF tiling emits a quantized padding value as a signless integer arith.constant wrapped in a
// builtin.unrealized_conversion_cast to the quantized element type (createQuantizedPaddingValue).
// Only that exact form carries the semantics that the integer input IS the pad storage value:
// unrealized_conversion_cast is otherwise a semantics-free type bridge, so any other shape -- or a
// storage constant that is not the quantized zero point -- is rejected. The storage constant is
// signless, so its signedness is taken from the quantized type.
mlir::FailureOr<double> deriveConstantPadValue(mlir::tensor::PadOp padOp, mlir::PatternRewriter& rewriter, Logger log) {
    const auto padValue = padOp.getConstantPaddingValue();
    if (padValue == nullptr) {
        return matchFailed(log, rewriter, padOp, "Pad body does not yield an index-independent value");
    }

    mlir::Value cstValue = padValue;
    std::optional<bool> quantStorageIsUnsigned;
    if (auto castOp = padValue.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() != 1 || castOp->getNumResults() != 1) {
            return matchFailed(log, rewriter, padOp, "Pad value cast '{0}' is not one-to-one", castOp);
        }
        const auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(padValue.getType());
        const auto storageType = mlir::dyn_cast<mlir::IntegerType>(castOp.getInputs().front().getType());
        if (qType == nullptr || storageType == nullptr || !storageType.isSignless() ||
            storageType.getWidth() != qType.getStorageTypeIntegralWidth()) {
            return matchFailed(log, rewriter, padOp, "Pad value cast '{0}' is not the quantized-storage form", castOp);
        }
        cstValue = castOp.getInputs().front();
        // createQuantizedPaddingValue emits exactly the storage zero point; the cast carries no numeric
        // semantics, so any other integer behind it must not be reinterpreted as pad data.
        mlir::Attribute storageAttr;
        if (!mlir::matchPattern(cstValue, mlir::m_Constant(&storageAttr)) ||
            storageAttr != mlir::IntegerAttr::get(storageType, qType.getZeroPoint())) {
            return matchFailed(log, rewriter, padOp, "Quantized pad value is not the storage zero point of '{0}'",
                               qType);
        }
        quantStorageIsUnsigned = !qType.isSigned();
    }

    mlir::Attribute cstAttr;
    if (!mlir::matchPattern(cstValue, mlir::m_Constant(&cstAttr))) {
        return matchFailed(log, rewriter, padOp, "Pad value '{0}' is not a constant", cstValue);
    }

    if (const auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(cstAttr)) {
        // pad_value_attr is an f64; a constant wider than f64 (e.g. f80/f128) must round-trip exactly.
        auto value = floatAttr.getValue();
        bool losesInfo = false;
        value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        if (losesInfo) {
            return matchFailed(log, rewriter, padOp, "Pad value '{0}' is not exactly representable in f64", floatAttr);
        }
        return value.convertToDouble();
    }

    const auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstAttr);
    if (intAttr == nullptr) {
        return matchFailed(log, rewriter, padOp, "Pad value attribute '{0}' is neither float nor integer", cstAttr);
    }
    // Signedness comes from the element type: a sign-extending read corrupts unsigned values
    // (e.g. ui8 255 -> -1).
    const auto intType = mlir::dyn_cast<mlir::IntegerType>(intAttr.getType());
    const bool isUnsigned = quantStorageIsUnsigned.value_or(intType != nullptr && intType.isUnsigned());
    // pad_value_attr is an f64; convert the APInt at full width so wider-than-64-bit values are
    // neither narrowed nor rejected when they are exactly representable.
    llvm::APFloat value(llvm::APFloat::IEEEdouble());
    const auto status = value.convertFromAPInt(intAttr.getValue(), !isUnsigned, llvm::APFloat::rmNearestTiesToEven);
    if (status != llvm::APFloat::opOK) {
        return matchFailed(log, rewriter, padOp, "Pad value '{0}' is not exactly representable in f64", intAttr);
    }
    return value.convertToDouble();
}

//
// ConvertTensorPadToVPUPadPass
//

class ConvertTensorPadToVPUPadPass final :
        public VPU::impl::ConvertTensorPadToVPUPadBase<ConvertTensorPadToVPUPadPass> {
public:
    explicit ConvertTensorPadToVPUPadPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class PadOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// PadOpConverter
//
// Rewrites a single SCF-tiling `tensor.pad` into `VPU.Pad` so the padding is handled by the existing
// VPU.Pad infrastructure (bounds/dynamic/SW-kernel support + bufferization) instead of a tensor.pad
// bufferization that materialized generic linalg ops. Supported subset: CONSTANT mode, static
// low/high padding, static or bounded-dynamic input. Anything else keeps the upstream tensor
// bufferization path. The result type is not recomputed here -- VPU::PadOp::inferReturnTypes derives
// it and the rewrite only proceeds when it reproduces the tensor.pad result type exactly.
class ConvertTensorPadToVPUPadPass::PadOpConverter final : public mlir::OpRewritePattern<mlir::tensor::PadOp> {
public:
    PadOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<mlir::tensor::PadOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::tensor::PadOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertTensorPadToVPUPadPass::PadOpConverter::matchAndRewrite(
        mlir::tensor::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp.getContext();
    const auto log = _log.nest();

    if (hasDynamicPadding(origOp)) {
        return matchFailed(log, rewriter, origOp, "Dynamic low/high padding is not expressible on VPU.Pad");
    }

    if (hasUnexpressibleNofold(origOp)) {
        return matchFailed(log, rewriter, origOp, "nofold is not expressible on a foldable VPU.Pad");
    }

    if (!hasRewritableInput(origOp)) {
        return matchFailed(log, rewriter, origOp, "Input '{0}' is neither static nor bounded-dynamic",
                           origOp.getSource().getType());
    }

    // E#235709 - Support SW-kernel lowering for dynamic tensor.pad generated by SCF tiling
    const auto lowPad = origOp.getStaticLow();
    const auto highPad = origOp.getStaticHigh();
    if (lowPad.empty() || highPad.empty()) {
        return matchFailed(log, rewriter, origOp,
                           "Empty low/high would select PadOp::inferReturnTypes' output-shape branch");
    }

    const auto padValue = deriveConstantPadValue(origOp, rewriter, log);
    if (mlir::failed(padValue)) {
        return mlir::failure();
    }

    log.trace("Rewriting '{0}' at '{1}' to VPU.Pad", origOp->getName(), origOp->getLoc());

    const auto padsBeginAttr = getIntArrayAttr(ctx, lowPad);
    const auto padsEndAttr = getIntArrayAttr(ctx, highPad);
    const auto padValueAttr = rewriter.getF64FloatAttr(*padValue);
    const auto modeAttr = IE::PadModeAttr::get(ctx, IE::PadMode::CONSTANT);
    const auto boundsRepr = VPU::BoundsRepresentationAttr::get(ctx, VPU::BoundsRepresentation::BOUNDS);

    // Builder A: pads/pad_value come in as attrs (the runtime operands stay null); the bounded result
    // type is computed by VPU::PadOp::inferReturnTypes.
    auto newPad =
            VPU::PadOp::create(rewriter, origOp.getLoc(), origOp.getSource(), /*pads_begin=*/mlir::Value(),
                               /*pads_end=*/mlir::Value(), /*pad_value=*/mlir::Value(), padsBeginAttr, padsEndAttr,
                               padValueAttr, modeAttr, /*output_padding=*/nullptr, /*input_padding=*/nullptr,
                               /*output_shape=*/nullptr, /*output_bounds=*/nullptr, boundsRepr);

    // The inferred VPU.Pad result must match the tensor.pad result (same shape + bounds encoding),
    // otherwise the pad's users would see a mismatched type.
    const auto inferredType = newPad.getOutput().getType();
    if (inferredType != origOp.getResult().getType()) {
        rewriter.eraseOp(newPad);
        return matchFailed(log, rewriter, origOp, "Inferred VPU.Pad type '{0}' != tensor.pad result '{1}'",
                           inferredType, origOp.getResult().getType());
    }

    rewriter.replaceOp(origOp, newPad.getOutput());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertTensorPadToVPUPadPass::safeRunOnFunc() {
    auto func = getOperation();

    if (config::isPureHostCompileFunc(func)) {
        return;
    }

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PadOpConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertTensorPadToVPUPadPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertTensorPadToVPUPadPass(Logger log) {
    return std::make_unique<ConvertTensorPadToVPUPadPass>(log);
}
