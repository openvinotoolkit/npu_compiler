//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTCONSTANTDYNAMICDEQUANTIZETODEQUANTIZE
#define GEN_PASS_DEF_CONVERTCONSTANTDYNAMICDEQUANTIZETODEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertConstantDynamicDequantizeToDequantize
//

// Rewrites IE.DynamicDequantize whose scale and zero-point
// operands are constants into:
//
//   IE.QuantizeCast  -- embeds constants into the quant.uniform element type
//   IE.Dequantize    -- static dequantization as used by the downstream pipeline
//
// This is a bridge-adaptation pass: it ensures that higher-level passes can
// express weights quantization via DynamicDequantize while the rest of the
// pipeline continues to use the static Dequantize representation.

class ConvertConstantDynamicDequantizeToDequantize final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    ConvertConstantDynamicDequantizeToDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
        setDebugName("ConvertConstantDynamicDequantizeToDequantize");
    }

    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Build a new quantized element type with the given scales and zero-points embedded, preserving all other
// attributes (flags, storage type, range) from baseType.
mlir::quant::QuantizedType buildQuantTypeWithEmbeddedParams(mlir::quant::QuantizedType baseType,
                                                            ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
                                                            int32_t quantAxis) {
    VPUX_THROW_UNLESS(scales.size() == zeroPoints.size(),
                      "scales and zeroPoints must have the same size, got {0} vs {1}", scales.size(),
                      zeroPoints.size());

    if (scales.size() == 1) {
        return mlir::quant::UniformQuantizedType::get(baseType.getFlags(), baseType.getStorageType(),
                                                      baseType.getExpressedType(), scales.front(), zeroPoints.front(),
                                                      baseType.getStorageTypeMin(), baseType.getStorageTypeMax());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(baseType.getFlags(), baseType.getStorageType(),
                                                         baseType.getExpressedType(), scales, zeroPoints, quantAxis,
                                                         baseType.getStorageTypeMin(), baseType.getStorageTypeMax());
}

mlir::LogicalResult ConvertConstantDynamicDequantizeToDequantize::matchAndRewrite(
        IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // Both scale and optional ZP must be compile-time constants.
    auto scaleOp = origOp.getScale().getDefiningOp<Const::DeclareOp>();
    if (scaleOp == nullptr) {
        return matchFailed(rewriter, origOp, "scale is not a compile-time constant");
    }

    auto zpOp = origOp.getZp() != nullptr ? origOp.getZp().getDefiningOp<Const::DeclareOp>() : nullptr;
    if (origOp.getZp() != nullptr && zpOp == nullptr) {
        return matchFailed(rewriter, origOp, "Both scale and zp must be compile-time constant");
    }

    // Input elemnt type must be quantized
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto baseQType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputType.getElementType());
    if (baseQType == nullptr) {
        return matchFailed(rewriter, origOp, "input element type is not QuantizedType");
    }

    // Derive the quantization axis from the scale shape
    const auto scaleShape = mlir::cast<vpux::NDTypeInterface>(origOp.getScale().getType()).getShape();
    SmallVector<int32_t> nonOneAxes;
    for (auto [i, dim] : llvm::enumerate(scaleShape)) {
        if (dim != 1) {
            nonOneAxes.push_back(static_cast<int32_t>(i));
        }
    }
    if (nonOneAxes.size() > 1) {
        return matchFailed(rewriter, origOp, "per-channel quantization on multiple axes is not supported");
    }

    const int32_t quantAxis = nonOneAxes.empty() ? 0 : nonOneAxes.front();

    auto scales = vpux::IE::readConstAsDoubles(scaleOp);

    // Determine zero-points: if ZP is not present, or is a single scalar, broadcast it to match the number of scales.
    // If it's already the same size as scales, use as is. Otherwise, fail.
    SmallVector<int64_t> zeroPoints(scales.size(), 0);
    if (zpOp != nullptr) {
        zeroPoints = vpux::IE::readConstAsInt64(zpOp);

        // The ZP tensor element type is signless i8/i4/i2 (required by DynamicDequantize op verifier).
        // dispatchByElemType maps signless integers to unsigned C++ types (uint8_t, etc.),
        // so negative values like -10 are read as 246. If the quantized storage type is signed,
        // sign-extend from the ZP element bit width to recover the correct negative value.
        if (baseQType.isSigned()) {
            const auto zpElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getZp().getType()).getElementType();
            const auto bitWidth = zpElemType.getIntOrFloatBitWidth();
            if (bitWidth < 64) {
                const int64_t shift = 64 - static_cast<int64_t>(bitWidth);
                for (auto& zp : zeroPoints) {
                    zp = (zp << shift) >> shift;
                }
            }
        }

        if (zeroPoints.size() == 1) {
            zeroPoints.assign(scales.size(), zeroPoints.front());
        } else if (zeroPoints.size() != scales.size()) {
            return matchFailed(rewriter, origOp, "ZP element count {0} does not match scale count {1}",
                               zeroPoints.size(), scales.size());
        }
    }

    const auto newQType = buildQuantTypeWithEmbeddedParams(baseQType, scales, zeroPoints, quantAxis);

    auto quantCastOp =
            rewriter.create<IE::QuantizeCastOp>(takeOpLoc(origOp, "quant_cast"), origOp.getInput(), newQType);

    auto dequantizeOp =
            rewriter.create<IE::DequantizeOp>(origOp->getLoc(), quantCastOp.getOutput(), origOp.getDstElemType());
    rewriter.replaceOp(origOp, dequantizeOp.getOutput());
    return mlir::success();
}

//
// ConvertConstantDynamicDequantizeToDequantizePass
//

class ConvertConstantDynamicDequantizeToDequantizePass final :
        public IE::impl::ConvertConstantDynamicDequantizeToDequantizeBase<
                ConvertConstantDynamicDequantizeToDequantizePass> {
public:
    explicit ConvertConstantDynamicDequantizeToDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertConstantDynamicDequantizeToDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertConstantDynamicDequantizeToDequantize>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertConstantDynamicDequantizeToDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConstantDynamicDequantizeToDequantizePass(Logger log) {
    return std::make_unique<ConvertConstantDynamicDequantizeToDequantizePass>(log);
}
