//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

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

// Build a new quantized element type with the given scales and zero-points embedded.
mlir::quant::QuantizedType buildQuantTypeWithEmbeddedParams(unsigned flags, mlir::Type storageType,
                                                            mlir::Type expressedType, int64_t storageMin,
                                                            int64_t storageMax, ArrayRef<double> scales,
                                                            ArrayRef<int64_t> zeroPoints, int32_t quantAxis) {
    VPUX_THROW_UNLESS(scales.size() == zeroPoints.size(),
                      "scales and zeroPoints must have the same size, got {0} vs {1}", scales.size(),
                      zeroPoints.size());

    if (scales.size() == 1) {
        return mlir::quant::UniformQuantizedType::get(flags, storageType, expressedType, scales.front(),
                                                      zeroPoints.front(), storageMin, storageMax);
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(flags, storageType, expressedType, scales, zeroPoints,
                                                         quantAxis, storageMin, storageMax);
}

mlir::LogicalResult ConvertConstantDynamicDequantizeToDequantize::matchAndRewrite(
        IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // Only convert DynamicDequantize ops that were synthetically created by the compiler
    // (e.g., from SplitFakeQuant or ConsolidateWeightsDequantization). Model-original
    // DynamicDequantize ops must not be touched.
    if (!origOp->hasAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR)) {
        return matchFailed(rewriter, origOp, "not a synthetic DynamicDequantize (missing marker)");
    }

    // Both scale and optional ZP must be compile-time constants.
    auto scaleOp = origOp.getScale().getDefiningOp<Const::DeclareOp>();
    if (scaleOp == nullptr) {
        return matchFailed(rewriter, origOp, "scale is not a compile-time constant");
    }

    auto zpOp = origOp.getZp() != nullptr ? origOp.getZp().getDefiningOp<Const::DeclareOp>() : nullptr;
    if (origOp.getZp() != nullptr && zpOp == nullptr) {
        return matchFailed(rewriter, origOp, "Both scale and zp must be compile-time constant");
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputElemType = inputType.getElementType();
    const auto expressedType = origOp.getDstElemType();

    // Extract storage parameters from either an already-quantized input or a raw storage type.

    unsigned flags = 0;
    mlir::Type storageType;
    int64_t storageMin = 0;
    int64_t storageMax = 0;

    const auto storageParamsResult =
            mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputElemType)
                    .Case<mlir::quant::QuantizedType>([&](mlir::quant::QuantizedType qType) {
                        flags = qType.getFlags();
                        storageType = qType.getStorageType();
                        storageMin = qType.getStorageTypeMin();
                        storageMax = qType.getStorageTypeMax();
                        return mlir::success();
                    })
                    .Case<mlir::IntegerType>([&](mlir::IntegerType intType) {
                        const bool isSigned = intType.isSigned();
                        flags = isSigned ? mlir::quant::QuantizationFlags::Signed : 0;
                        const auto signedness = isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
                        storageType = mlir::IntegerType::get(intType.getContext(), intType.getWidth(), signedness);
                        storageMin =
                                mlir::quant::QuantizedType::getDefaultMinimumForInteger(isSigned, intType.getWidth());
                        storageMax =
                                mlir::quant::QuantizedType::getDefaultMaximumForInteger(isSigned, intType.getWidth());
                        return mlir::success();
                    })
                    .Case<vpux::type::QuantileType>([&](vpux::type::QuantileType quantileType) {
                        const bool isSigned = quantileType.shouldDefaultToSigned();
                        flags = isSigned ? mlir::quant::QuantizationFlags::Signed : 0;
                        storageType = quantileType;
                        const auto bitWidth = quantileType.getStorageWidth();
                        storageMin = mlir::quant::QuantizedType::getDefaultMinimumForInteger(isSigned, bitWidth);
                        storageMax = mlir::quant::QuantizedType::getDefaultMaximumForInteger(isSigned, bitWidth);
                        return mlir::success();
                    })
                    .Default([&](mlir::Type elemType) -> mlir::LogicalResult {
                        // Raw float storage type (e.g., f8E5M2, f8E4M3FN, f4E2M1FN)
                        const auto storageParamsFP = vpux::getStorageParams(elemType);
                        if (mlir::failed(storageParamsFP)) {
                            return matchFailed(rewriter, origOp, "unsupported input element type {0}", elemType);
                        }
                        storageType = elemType;
                        storageMin = static_cast<int64_t>(std::get<0>(*storageParamsFP));
                        storageMax = static_cast<int64_t>(std::get<1>(*storageParamsFP));
                        return mlir::success();
                    });
    if (mlir::failed(storageParamsResult)) {
        return storageParamsResult;
    }

    // Derive the quantization axis from the scale shape, falling back to the ZP shape when
    // scale is per-tensor (all dimensions are 1) but ZP is per-channel.
    const auto scaleShape = mlir::cast<vpux::NDTypeInterface>(origOp.getScale().getType()).getShape();
    SmallVector<int32_t> nonOneAxes;
    for (auto [i, dim] : llvm::enumerate(scaleShape)) {
        if (dim != 1) {
            nonOneAxes.push_back(static_cast<int32_t>(i));
        }
    }
    if (zpOp != nullptr && nonOneAxes.empty()) {
        // Only derive the quant axis from ZP shape when ZP is non-splat.
        // A splat ZP (all values identical) is effectively per-tensor and does not
        // contribute axis information.
        const auto zpContent = zpOp.getContentAttr().fold();
        if (!zpContent.isSplat()) {
            const auto zpShape = mlir::cast<vpux::NDTypeInterface>(origOp.getZp().getType()).getShape();
            for (auto [i, dim] : llvm::enumerate(zpShape)) {
                if (dim != 1) {
                    nonOneAxes.push_back(static_cast<int32_t>(i));
                }
            }
        }
    }
    if (nonOneAxes.size() > 1) {
        return matchFailed(rewriter, origOp, "per-channel quantization on multiple axes is not supported");
    }

    const int32_t quantAxis = nonOneAxes.empty() ? 0 : nonOneAxes.front();

    auto scales = vpux::IE::readConstAsDoubles(scaleOp);

    // Determine zero-points from the explicit ZP tensor operand (primary path) or, as a
    // fallback, inherit from the input quant type when no ZP tensor is provided.
    // A single scalar on either side is broadcast to match the other's count.
    // Mismatched counts beyond that are rejected.
    const auto baseQType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputElemType);
    SmallVector<int64_t> zeroPoints;
    if (zpOp != nullptr) {
        zeroPoints = vpux::IE::readConstAsInt64(zpOp);

        // dispatchByElemType maps signless integers to unsigned C++ types (uint8_t, etc.),
        // so negative values like -10 are read as 246. If the quantized storage type is signed,
        // sign-extend from the ZP element bit width to recover the correct negative value.
        // Sign-extension only applies to integer ZP types; float ZP values are already correct.
        const auto zpElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getZp().getType()).getElementType();
        const bool isSigned = (flags & mlir::quant::QuantizationFlags::Signed) != 0;
        if (isSigned && zpElemType.isInteger()) {
            const auto bitWidth = zpElemType.getIntOrFloatBitWidth();
            if (bitWidth < 64) {
                const int64_t shift = 64 - static_cast<int64_t>(bitWidth);
                for (auto& zp : zeroPoints) {
                    zp = (zp << shift) >> shift;
                }
            }
        }

        // Collapse all-equal ZP arrays to a single value so per-tensor broadcast works
        // correctly (e.g. ZP tensor has full input shape but is splat).
        if (zeroPoints.size() > 1 && llvm::all_equal(zeroPoints)) {
            zeroPoints.resize(1);
        }

        if (zeroPoints.size() == 1) {
            zeroPoints.assign(scales.size(), zeroPoints.front());
        } else if (scales.size() == 1) {
            scales.assign(zeroPoints.size(), scales.front());
        } else if (zeroPoints.size() != scales.size()) {
            return matchFailed(rewriter, origOp, "ZP element count {0} does not match scale count {1}",
                               zeroPoints.size(), scales.size());
        }
    } else {
        // No explicit ZP tensor — inherit zero points from the input's quantized element type
        // or default to zero for raw input types.
        if (baseQType) {
            if (const auto perAxis = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(baseQType)) {
                zeroPoints.assign(perAxis.getZeroPoints().begin(), perAxis.getZeroPoints().end());
            } else if (const auto perTensor = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(baseQType)) {
                zeroPoints.assign(scales.size(), perTensor.getZeroPoint());
            } else {
                zeroPoints.assign(scales.size(), 0);
            }
        } else {
            zeroPoints.assign(scales.size(), 0);
        }
    }

    const auto newQType = buildQuantTypeWithEmbeddedParams(flags, storageType, expressedType, storageMin, storageMax,
                                                           scales, zeroPoints, quantAxis);

    // If the input is already a quantized type with real scale/ZP (not the identity placeholder
    // inserted by ConvertDQRawDataTypeToQuantized), apply Dequantize directly without a QuantizeCast.
    // A QuantizeCast here would introduce a type mismatch caused by f32 precision loss in the
    // scale/ZP round-trip, preventing FuseOutstandingDequant from removing the stray Dequantize
    // after FuseQuantizedOps fuses the NCE → Quantize pair.
    if (baseQType != nullptr) {
        const auto perTensorQType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedType>(baseQType);
        const bool isIdentityPlaceholder =
                perTensorQType && (perTensorQType.getScale() == 1.0) && (perTensorQType.getZeroPoint() == 0);
        if (!isIdentityPlaceholder) {
            rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, origOp.getInput(), origOp.getDstElemType());
            return mlir::success();
        }
    }

    // For raw storage inputs (integer, QuantileType, low-precision float) and identity-placeholder
    // quantized inputs, embed the reconstructed quant type via QuantizeCast before Dequantize.
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
