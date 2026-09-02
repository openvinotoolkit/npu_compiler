//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDQRAWDATATYPETOQUANTIZED
#define GEN_PASS_DEF_CONVERTDQRAWDATATYPETOQUANTIZED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertDQRawDataTypeToQuantized
//

// For each IE.DynamicDequantize whose input carries a raw (non-quantized) element type
// (e.g. si4, ui4, QuantileType, f8E4M3FN), inserts an IE.QuantizeCast that attaches an
// identity quant.uniform type (scale=1.0, zp=0) to the input tensor.
//
// This is a bridge-adaptation pattern: it allows DynamicDequantize to be used with raw storage types in the
// higher-level parts of the pipeline, while ensuring that downstream passes can uniformly expect quantized types with
// embedded parameters.

class ConvertDQRawDataTypeToQuantized final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    using mlir::OpRewritePattern<IE::DynamicDequantizeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final {
        const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();

        if (mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
            return mlir::failure();
        }

        unsigned qFlags = 0;

        const auto storageParams = vpux::getStorageParams(inputElemType);
        if (mlir::failed(storageParams)) {
            return matchFailed(rewriter, origOp, "unsupported raw element type {0}", inputElemType);
        }

        const auto [storageMin, storageMax, storageType] = *storageParams;
        if (const auto intType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
            qFlags = intType.isSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
        } else if (const auto quantileStorageType = mlir::dyn_cast_if_present<vpux::type::QuantileType>(storageType)) {
            qFlags = quantileStorageType.shouldDefaultToSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
        } else if (!isLowFpType(storageType)) {
            return matchFailed(rewriter, origOp, "unsupported raw storage type {0}", storageType);
        }

        const auto quantType =
                mlir::quant::UniformQuantizedType::get(qFlags, storageType, origOp.getDstElemType(),
                                                       /*scale=*/1.0, /*zeroPoint=*/0, storageMin, storageMax);

        auto quantCast = rewriter.create<IE::QuantizeCastOp>(origOp->getLoc(), origOp.getInput(), quantType);
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.getInputMutable().assign(quantCast.getOutput());
        });
        return mlir::success();
    }
};

//
// ConvertDQRawDataTypeToQuantizedPass
//

class ConvertDQRawDataTypeToQuantizedPass final :
        public IE::impl::ConvertDQRawDataTypeToQuantizedBase<ConvertDQRawDataTypeToQuantizedPass> {
public:
    explicit ConvertDQRawDataTypeToQuantizedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDQRawDataTypeToQuantizedPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertDQRawDataTypeToQuantized>(&ctx);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertDQRawDataTypeToQuantizedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDQRawDataTypeToQuantizedPass(Logger log) {
    return std::make_unique<ConvertDQRawDataTypeToQuantizedPass>(log);
}
