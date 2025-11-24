//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/convert_quantize_ops_to_nce_ops_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTQUANTIZEOPSTONCEOPS
#define GEN_PASS_DEF_CONVERTQUANTIZEOPSTONCEOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace vpux::IE {

bool isQuantizedPerAxis(mlir::Value val) {
    auto elemType = mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType();
    return mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);
}

bool isLegalQuantizeOp(IE::QuantizeOp quantizeOp, bool canUseCMajor) {
    auto outputLayerUsers = quantizeOp.getOutput().getUsers();
    auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
        return mlir::isa<IE::ConvolutionOp>(user);
    });

    return anyUserIsConv && canUseCMajor;
};

bool isPerChannelQuantizedType(mlir::Value val) {
    auto elemType = mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType();
    auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    if (perAxisType == nullptr) {
        return false;
    }

    auto rank = mlir::cast<vpux::NDTypeInterface>(val.getType()).getRank();
    if (rank != 4) {
        return false;
    }

    auto quantizeDim = perAxisType.getQuantizedDimension();
    return quantizeDim == Dims4D::Act::C.ind();
}

bool is16BitStorageType(vpux::NDTypeInterface type) {
    auto quantType = mlir::cast<mlir::quant::QuantizedType>(type.getElementType());
    return quantType.getStorageTypeIntegralWidth() == 16;
}

bool is8BitStorageType(vpux::NDTypeInterface type) {
    auto quantType = mlir::cast<mlir::quant::QuantizedType>(type.getElementType());
    return quantType.getStorageTypeIntegralWidth() == 8;
}

}  // namespace vpux::IE

namespace {

//
// ConvertQuantizeOpsToNceOpsPass
//

class ConvertQuantizeOpsToNceOpsPass final :
        public IE::impl::ConvertQuantizeOpsToNceOpsBase<ConvertQuantizeOpsToNceOpsPass> {
public:
    explicit ConvertQuantizeOpsToNceOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertQuantizeOpsToNceOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto strategy = IE::createConvertQuantizeOpsToNceOpsStrategy(func);

    mlir::ConversionTarget toAvgPoolTarget(ctx);
    mlir::RewritePatternSet toAvgPoolPatterns(&ctx);
    strategy->prepareAvgPool(toAvgPoolTarget, toAvgPoolPatterns, ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, toAvgPoolTarget, std::move(toAvgPoolPatterns)))) {
        signalPassFailure();
    }

    // perTensor quantize/dequantize convert to add or and
    mlir::ConversionTarget toEltwiseTarget(ctx);
    mlir::RewritePatternSet toEltwisePatterns(&ctx);
    strategy->prepareEltwise(toEltwiseTarget, toEltwisePatterns, ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, toEltwiseTarget, std::move(toEltwisePatterns)))) {
        signalPassFailure();
    }

    // per-axis scales and per-tensor zero points quantize/dequantize convert to DW conv
    mlir::ConversionTarget quantToConvTarget(ctx);
    mlir::RewritePatternSet quantToConvPatterns(&ctx);
    strategy->prepareQuantToConv(quantToConvTarget, quantToConvPatterns, ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, quantToConvTarget, std::move(quantToConvPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertQuantizeOpsToNceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertQuantizeOpsToNceOpsPass(Logger log) {
    return std::make_unique<ConvertQuantizeOpsToNceOpsPass>(log);
}
