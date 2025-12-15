//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTODEQUANTIZE
#define GEN_PASS_DEF_CONVERTTODEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertToDequantizePass
//

class ConvertToDequantizePass final : public IE::impl::ConvertToDequantizeBase<ConvertToDequantizePass> {
public:
    explicit ConvertToDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit ConvertToDequantizePass(const IE::LowPrecisionOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);
    }

public:
    class ConvertOpConverter;

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

    void safeRunOnFunc() final;
};

mlir::LogicalResult ConvertToDequantizePass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// ConvertOpConverter
//

class ConvertToDequantizePass::ConvertOpConverter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    ConvertOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// It matches pattern non-const -> Convert -> ViewLikeOp/TransposeOp -> Convolution/GroupConvolution,
// then replace Convert with QuantizeCast -> Dequantize.
// We expect that Dequantize op will then be propagated to the Convolution/GroupConvolution
mlir::LogicalResult ConvertToDequantizePass::ConvertOpConverter::matchAndRewrite(
        IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const {
    auto constInput = convertOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (constInput != nullptr) {
        return mlir::failure();
    }

    auto inputElemType = convertOp.getInput().getType().getElementType();
    auto outputElemType = convertOp.getOutput().getType().getElementType();
    if (!outputElemType.isF16()) {
        return mlir::failure();
    }

    // Currently we're supporting on DPU only INT8 and INT4 quantized weight types
    const auto supportedIntBitWidth = SmallVector<int64_t>({8, 4});
    const auto inputElemTypeSize = getElemTypeSize(inputElemType).count();
    if (llvm::find(supportedIntBitWidth, inputElemTypeSize) == supportedIntBitWidth.end()) {
        return mlir::failure();
    }

    if (!convertOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    mlir::Operation* preOp = convertOp;
    auto postOp = *convertOp.getResult().getUsers().begin();
    while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface, IE::TransposeOp, IE::QuantizeOp, IE::DequantizeOp>(postOp)) {
        if (!postOp->hasOneUse()) {
            return mlir::failure();
        }

        preOp = postOp;
        postOp = *postOp->getUsers().begin();
    }

    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(postOp)) {
        return mlir::failure();
    }

    if (preOp->getResult(0) != postOp->getOperand(1)) {
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();
    mlir::quant::QuantizedType outQuantizeElemType;
    mlir::IntegerType integerType;
    if (inputElemType.isSignedInteger()) {
        integerType = mlir::IntegerType::get(ctx, inputElemTypeSize, mlir::IntegerType::Signed);
        // Map integer type in max representable range; example for INT8 [-128, 127]
        // Attention, below logic does not cover also I1 integer types
        outQuantizeElemType = mlir::quant::UniformQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, integerType, mlir::Float16Type::get(ctx), /*scale=*/1,
                /*zero_point=*/0, -1 * (1 << (inputElemTypeSize - 1)), (1 << (inputElemTypeSize - 1)) - 1);
    } else if (auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(inputElemType)) {
        integerType = mlir::IntegerType::get(ctx, inputElemTypeSize, mlir::IntegerType::Unsigned);
        outQuantizeElemType = mlir::quant::QuantileQuantizedType::get(
                0, quantileFloatType.getStorageType(), quantileFloatType.getQuantileType(), mlir::Float16Type::get(ctx),
                quantileFloatType.getQuantiles(), /*scale=*/1,
                /*zero_point=*/0, 0, (1 << inputElemTypeSize) - 1);
    } else {
        integerType = mlir::IntegerType::get(ctx, inputElemTypeSize, mlir::IntegerType::Unsigned);
        // Map integer type in max representable range; example for UINT8 [0, 255]
        // Attention, below logic does not cover also I1 integer types
        outQuantizeElemType =
                mlir::quant::UniformQuantizedType::get(0, integerType, mlir::Float16Type::get(ctx), /*scale=*/1,
                                                       /*zero_point=*/0, 0, (1 << inputElemTypeSize) - 1);
    }

    auto quantizeCastOp =
            rewriter.create<IE::QuantizeCastOp>(convertOp.getLoc(), convertOp.getInput(), outQuantizeElemType);

    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(convertOp, quantizeCastOp.getResult(), outputElemType);

    return mlir::success();
}  // namespace

//
// safeRunOnFunc
//

void ConvertToDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToDequantizePass(Logger log) {
    return std::make_unique<ConvertToDequantizePass>(log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToDequantizePass(const IE::LowPrecisionOptions& options,
                                                                    Logger log) {
    return std::make_unique<ConvertToDequantizePass>(options, log);
}
