//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOQUANTIZEDOPS
#define GEN_PASS_DEF_CONVERTTOQUANTIZEDOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertToQuantizedOpsPass
//

class ConvertToQuantizedOpsPass final : public IE::impl::ConvertToQuantizedOpsBase<ConvertToQuantizedOpsPass> {
public:
    explicit ConvertToQuantizedOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConvertToDequantize;
    class ConvertToQuantize;

private:
    void safeRunOnFunc() final;
};

//
// ConvertToDequantize
//

class ConvertToQuantizedOpsPass::ConvertToDequantize final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    ConvertToDequantize(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// It matches pattern non-const -> Convert -> ViewLikeOp/TransposeOp -> Convolution/GroupConvolution,
// then replace Convert with QuantizeCast -> Dequantize.
// We expect that Dequantize op will then be propagated to the Convolution/GroupConvolution
mlir::LogicalResult ConvertToQuantizedOpsPass::ConvertToDequantize::matchAndRewrite(
        IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const {
    assert(convertOp.getInput().getDefiningOp<Const::DeclareOp>() == nullptr &&
           "const.Declare -> IE.Convert must be folded before this rewriter runs");

    auto outputElemType = convertOp.getOutput().getType().getElementType();
    if (!outputElemType.isF16()) {
        return mlir::failure();
    }

    auto inputElemType = convertOp.getInput().getType().getElementType();
    auto quantileType = mlir::dyn_cast<mlir::quant::QuantileType>(inputElemType);
    if (!inputElemType.isInteger() && quantileType == nullptr) {
        return mlir::failure();
    }

    // Currently we're supporting on DPU only 8-bit and 4-bit quantized weight types
    const auto supportedBitWidth = SmallVector<int64_t>({8, 4});
    const auto inputElemTypeSize = getElemTypeSize(inputElemType).count();
    if (llvm::find(supportedBitWidth, inputElemTypeSize) == supportedBitWidth.end()) {
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
    } else if (quantileType != nullptr) {
        const auto newQuantileStorageType = mlir::quant::QuantileType::get(
                ctx, quantileType.getStorageType(), quantileType.getQuantileType(), quantileType.getQuantiles(),
                quantileType.getStorageMin(), quantileType.getStorageMax());

        outQuantizeElemType =
                mlir::quant::UniformQuantizedType::get(0, newQuantileStorageType, mlir::Float16Type::get(ctx),
                                                       /*scale=*/1,
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
            IE::QuantizeCastOp::create(rewriter, convertOp.getLoc(), convertOp.getInput(), outQuantizeElemType);

    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(convertOp, quantizeCastOp.getResult(), outputElemType);

    return mlir::success();
}

//
// ConvertToQuantize
//

class ConvertToQuantizedOpsPass::ConvertToQuantize final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    ConvertToQuantize(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    template <typename Op1Type, typename Op2Type>
    bool matchPattern(mlir::Value input, Op1Type& firstOp, Op2Type& secondOp) const;

    Logger _log;
};

// Helper function to match the pattern Op1 -> Op2 -> input
template <typename Op1Type, typename Op2Type>
bool ConvertToQuantizedOpsPass::ConvertToQuantize::matchPattern(mlir::Value input, Op1Type& firstOp,
                                                                Op2Type& secondOp) const {
    if (auto op2 = input.template getDefiningOp<Op2Type>()) {
        if (auto op1 = op2.getInput().template getDefiningOp<Op1Type>()) {
            firstOp = op1;
            secondOp = op2;
            return true;
        }
    }
    return false;
}

// It matches pattern [GroupConvolution] -> Round -> Clamp -> Convert or [GroupConvolution] -> Clamp -> Round ->
// Convert, then replace with Quantize -> QuantizeCast.
mlir::LogicalResult ConvertToQuantizedOpsPass::ConvertToQuantize::matchAndRewrite(
        IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const {
    auto outputElemType = convertOp.getOutput().getType().getElementType();
    if (!outputElemType.isInteger()) {
        return mlir::failure();
    }

    // Currently we're supporting only 8-bit and 4-bit quantized types
    const auto supportedBitWidth = SmallVector<int64_t>({8, 4});
    const auto outputElemTypeSize = getElemTypeSize(outputElemType).count();
    bool isSupportedBitWidth = llvm::find(supportedBitWidth, outputElemTypeSize) != supportedBitWidth.end();
    if (!isSupportedBitWidth) {
        return mlir::failure();
    }

    // Verify clamp range matches the Convert type range
    int32_t expectedMin, expectedMax;
    if (outputElemType.isSignedInteger()) {
        expectedMin = -1 * (1 << (outputElemTypeSize - 1));
        expectedMax = (1 << (outputElemTypeSize - 1)) - 1;
    } else {
        expectedMin = 0;
        expectedMax = (1 << outputElemTypeSize) - 1;
    }

    IE::ClampOp clampOp = nullptr;
    IE::RoundOp roundOp = nullptr;
    mlir::Value patternInput;

    // Try to match two patterns:
    // Pattern 1: ... -> Round -> Clamp -> Convert
    // Pattern 2: ... -> Clamp -> Round -> Convert
    auto convertInput = convertOp.getInput();
    if (matchPattern<IE::RoundOp, IE::ClampOp>(convertInput, roundOp, clampOp)) {
        patternInput = roundOp.getInput();
    } else if (matchPattern<IE::ClampOp, IE::RoundOp>(convertInput, clampOp, roundOp)) {
        patternInput = clampOp.getInput();
    } else {
        return mlir::failure();
    }

    // Extract clamp range
    const auto clampMin = static_cast<int32_t>(clampOp.getMin().convertToDouble());
    const auto clampMax = static_cast<int32_t>(clampOp.getMax().convertToDouble());
    if (clampMin != expectedMin || clampMax != expectedMax) {
        return mlir::failure();
    }

    // Check if patternInput is a fusible GroupConvolution
    IE::GroupConvolutionOp grConvOp = nullptr;
    double scale = 1.0;
    int64_t zeroPoint = 0;

    if (auto groupConv = mlir::dyn_cast_or_null<IE::GroupConvolutionOp>(patternInput.getDefiningOp())) {
        if (!IE::hasPPE(groupConv) && IE::isEltwiseGroupConv(groupConv, /*isConstFilter=*/true)) {
            const auto filterValue = Const::getSplatValue<double>(groupConv.getFilter()).value();
            if (!isDoubleEqual(filterValue, 0.0)) {
                scale = 1 / filterValue;

                if (auto bias = groupConv.getBias()) {
                    const auto biasValue = Const::getSplatValue<double>(bias).value();
                    // Use the same rounding mode as the Round op in the pattern
                    if (roundOp.getMode() == IE::RoundMode::HALF_TO_EVEN) {
                        zeroPoint = static_cast<int64_t>(std::rint(biasValue));
                    } else {
                        zeroPoint = static_cast<int64_t>(std::round(biasValue));
                    }
                }

                // GroupConv can be fused - it's a single-value multiply (and optional add)
                grConvOp = groupConv;
                patternInput = groupConv.getInput();
            }
        }
    }

    // Create Quantize op
    auto ctx = rewriter.getContext();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(patternInput.getType());
    auto inputElemType = inputType.getElementType();

    mlir::quant::QuantizedType outQuantizeElemType;
    mlir::IntegerType integerType;

    if (outputElemType.isSignedInteger()) {
        integerType = mlir::IntegerType::get(ctx, outputElemTypeSize, mlir::IntegerType::Signed);
        outQuantizeElemType =
                mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, integerType,
                                                       inputElemType, scale, zeroPoint, expectedMin, expectedMax);
    } else {
        integerType = mlir::IntegerType::get(ctx, outputElemTypeSize, mlir::IntegerType::Unsigned);
        outQuantizeElemType = mlir::quant::UniformQuantizedType::get(0, integerType, inputElemType, scale, zeroPoint,
                                                                     expectedMin, expectedMax);
    }

    auto quantizeOp = IE::QuantizeOp::create(rewriter, convertOp.getLoc(), patternInput, outQuantizeElemType);
    auto quantizeCastOp =
            IE::QuantizeCastOp::create(rewriter, convertOp.getLoc(), quantizeOp.getResult(), outputElemType);

    rewriter.replaceOp(convertOp, quantizeCastOp.getResult());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertToQuantizedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToDequantize>(&ctx, _log);
    patterns.add<ConvertToQuantize>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertToQuantizedOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToQuantizedOpsPass(Logger log) {
    return std::make_unique<ConvertToQuantizedOpsPass>(log);
}
