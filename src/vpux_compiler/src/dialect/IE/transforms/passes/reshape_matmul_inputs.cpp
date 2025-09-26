//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_RESHAPEMATMULINPUTS
#define GEN_PASS_DEF_RESHAPEMATMULINPUTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
void InputsTo2D(IE::MatMulOp origOp) {
    const auto lhs = origOp.getInput1();
    const auto rhs = origOp.getInput2();
    const auto out = origOp.getOutput();
    const auto lhsShape = getShape(lhs);
    const auto rhsShape = getShape(rhs);
    const auto outShape = getShape(out);
    // Transpose attributes are ignored for 1D tensors.
    // However, transpose attributes apply to 2D tensors after the reshape.
    //
    // transposeA = false, transposeB = false
    // IE.MatMul(tensor<32xf16>, tensor<32x64xf16>) -> IE.MatMul(tensor<1x32xf16>, tensor<32x64xf16>)
    // IE.MatMul(tensor<64x32xf16>, tensor<32xf16>) -> IE.MatMul(tensor<64x32xf16>, tensor<32x1xf16>)
    // IE.MatMul(tensor<32xf16>, tensor<32xf16>) -> IE.MatMul(tensor<1x32xf16>, tensor<32x1xf16>)
    //
    // transposeA = false, transposeB = true
    // IE.MatMul(tensor<32xf16>, tensor<64x32xf16>) -> IE.MatMul(tensor<1x32xf16>, tensor<64x32xf16>)
    // IE.MatMul(tensor<64x32xf16>, tensor<32xf16>) -> IE.MatMul(tensor<64x32xf16>, tensor<1x32xf16>)
    // IE.MatMul(tensor<32xf16>, tensor<32xf16>) -> IE.MatMul(tensor<1x32xf16>, tensor<1x32xf16>)
    //
    // transposeA = true, transposeB = false
    // IE.MatMul(tensor<32xf16>, tensor<32x64xf16>) -> IE.MatMul(tensor<32x1xf16>, tensor<32x64xf16>)
    // IE.MatMul(tensor<32x64xf16>, tensor<32xf16>) -> IE.MatMul(tensor<32x64xf16>, tensor<32x1xf16>)
    // IE.MatMul(tensor<32xf16>, tensor<32xf16>) -> IE.MatMul(tensor<32x1xf16>, tensor<32x1xf16>)
    //
    // transposeA = true, transposeB = true
    // IE.MatMul(tensor<32xf16>, tensor<64x32xf16>) -> IE.MatMul(tensor<32x1xf16>, tensor<64x32xf16>)
    // IE.MatMul(tensor<32x64xf16>, tensor<32xf16>) -> IE.MatMul(tensor<32x64xf16>, tensor<1x32xf16>)
    // IE.MatMul(tensor<32xf16>, tensor<32xf16>) -> IE.MatMul(tensor<32x1xf16>, tensor<1x32xf16>)
    const auto lhsRank = lhsShape.size();
    const auto rhsRank = rhsShape.size();
    if (lhsRank > 1 && rhsRank > 1) {
        return;
    }
    const auto lhsOrder = DimsOrder::fromValue(lhs);
    const auto rhsOrder = DimsOrder::fromValue(rhs);
    const auto outOrder = DimsOrder::fromValue(out);
    if (!lhsOrder.isIdentity() || !rhsOrder.isIdentity() || !outOrder.isIdentity()) {
        return;
    }
    Shape newLhsShape = lhsShape.toValues();
    if (lhsRank == 1) {
        if (origOp.getTransposeA()) {
            newLhsShape = {lhsShape.front(), 1};
        } else {
            newLhsShape = {1, lhsShape.front()};
        }
    }
    Shape newRhsShape = rhsShape.toValues();
    if (rhsRank == 1) {
        if (origOp.getTransposeB()) {
            newRhsShape = {1, rhsShape.front()};
        } else {
            newRhsShape = {rhsShape.front(), 1};
        }
    }
    auto ctx = origOp.getContext();
    mlir::OpBuilder builder(origOp);

    auto reshapeLhs = builder.createOrFold<IE::ReshapeOp>(appendLoc(lhs.getLoc(), "reshape_lhs"), lhs,
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, newLhsShape));
    auto reshapeRhs = builder.createOrFold<IE::ReshapeOp>(appendLoc(rhs.getLoc(), "reshape_rhs"), rhs,
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, newRhsShape));
    auto newMatMul = builder.create<IE::MatMulOp>(origOp.getLoc(), reshapeLhs, reshapeRhs, origOp.getTransposeA(),
                                                  origOp.getTransposeB(), origOp.getPostOpAttr());
    auto reshapeOut = builder.createOrFold<IE::ReshapeOp>(appendLoc(out.getLoc(), "reshape_out"), newMatMul.getOutput(),
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, outShape));

    origOp.getOutput().replaceAllUsesWith(reshapeOut);
    origOp.erase();
}

void TransposeInputs(IE::MatMulOp matmulOp) {
    if (IE::isMatmulWithRHSTransposition(matmulOp)) {
        return;
    }

    auto ctx = matmulOp.getContext();
    auto transposedOrderAttr = [&](int64_t inputRank) -> mlir::AffineMapAttr {
        SmallVector<unsigned> perm(inputRank, 0);
        std::iota(perm.begin(), perm.end(), 0);
        if (inputRank < 2) {
            return nullptr;
        }
        std::iter_swap(perm.end() - 1, perm.end() - 2);
        return mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, ctx));
    };

    mlir::OpBuilder builder(matmulOp);
    auto getInput = [&](bool isTranspose, mlir::Value input, StringRef locSuffix) -> mlir::Value {
        if (isTranspose) {
            return input;
        }
        const auto inputShape = getShape(input);
        const auto inputRank = inputShape.size();

        if (transposedOrderAttr(inputRank) != nullptr) {
            auto orderAttr = transposedOrderAttr(inputRank);
            auto transpose = builder.create<IE::TransposeOp>(appendLoc(matmulOp->getLoc(), locSuffix), input, nullptr,
                                                             orderAttr);
            return transpose.getOutput();
        }

        return nullptr;
    };

    auto input1 = getInput(!matmulOp.getTransposeA(), matmulOp.getInput1(), "input_1");
    if (input1 == nullptr) {
        return;
    }
    auto input2 = getInput(matmulOp.getTransposeB(), matmulOp.getInput2(), "input_2");
    if (input2 == nullptr) {
        return;
    }

    auto newMatMul = builder.create<IE::MatMulOp>(matmulOp.getLoc(), input1, input2, /*transpose_a=*/false,
                                                  /*transpose_b=*/true, matmulOp.getPostOpAttr());
    matmulOp.getOutput().replaceAllUsesWith(newMatMul.getOutput());
    matmulOp.erase();
}

void CollapseBatch(IE::MatMulOp origOp) {
    if (origOp.getTransposeA() || !origOp.getTransposeB()) {
        return;
    }
    const auto lhs = origOp.getInput1();
    const auto rhs = origOp.getInput2();
    const auto out = origOp.getOutput();
    const auto lhsShape = getShape(lhs);
    const auto rhsShape = getShape(rhs);
    const auto outShape = getShape(out);
    // Convert
    // IE.MatMul(%arg0, %arg1) : [1, batch, inRows, inCols] * [1, 1, outCols, inCols]
    // into
    // IE.FullyConnected(%arg0, %arg1) : [batch * inRows, inCols] * [outCols, inCols]
    // For example consider the following IE.MatMul(2x3x4, 5x4) {transpose_b}:
    //
    // Input A:                     Input B:                Output:
    // [[[  1   2   3   4]      *   [[301 302 303 304]  =   [[[  3030   3130   3230   3330   3430]
    //   [ 11  12  13  14]           [311 312 313 314]        [ 15130  15630  16130  16630  17130]
    //   [ 21  22  23  24]]          [321 322 323 324]        [ 27230  28130  29030  29930  30830]]
    //                               [331 332 333 334]
    //  [[101 102 103 104]           [341 342 343 344]]      [[124030 128130 132230 136330 140430]
    //   [111 112 113 114]                                    [136130 140630 145130 149630 154130]
    //   [121 122 123 124]]]                                  [148230 153130 158030 162930 167830]]]
    //
    // Now let's compare the result with IE.MatMul(6x5, 5x4) {transpose_b}:
    //
    // Input A:                     Input B:                Output:
    // [[  1   2   3   4]           [[301 302 303 304]      [[  3030   3130   3230   3330   3430]
    //  [ 11  12  13  14]            [311 312 313 314]       [ 15130  15630  16130  16630  17130]
    //  [ 21  22  23  24]            [321 322 323 324]       [ 27230  28130  29030  29930  30830]
    //  [101 102 103 104]            [331 332 333 334]       [124030 128130 132230 136330 140430]
    //  [111 112 113 114]            [341 342 343 344]]      [136130 140630 145130 149630 154130]
    //  [121 122 123 124]]                                   [148230 153130 158030 162930 167830]]
    //
    // The output of IE.MatMul(6x5, 5x4) -> 6x5 can be reshaped to 2x3x5

    // Exclude row and column dimensions from the list, multiply only batches.
    const auto rhsBatch = std::accumulate(rhsShape.begin(), rhsShape.end() - 2, 1, std::multiplies<int64_t>());
    if (rhsBatch != 1) {
        return;
    }
    // Multiply every LHS dimension except for the last one.
    const auto lhsBatch = std::accumulate(lhsShape.begin(), lhsShape.end() - 1, 1, std::multiplies<int64_t>());
    const auto lhsOrder = DimsOrder::fromValue(lhs);
    const auto rhsOrder = DimsOrder::fromValue(rhs);
    const auto outOrder = DimsOrder::fromValue(out);
    if (!lhsOrder.isIdentity() || !rhsOrder.isIdentity() || !outOrder.isIdentity()) {
        return;
    }
    const auto rhsRank = rhsShape.size();
    const Shape newLhsShape = {lhsBatch, lhsShape.back()};
    const Shape newRhsShape = {rhsShape[Dim(rhsRank - 2)], rhsShape[Dim(rhsRank - 1)]};

    auto ctx = origOp.getContext();
    mlir::OpBuilder builder(origOp);
    auto reshapeLhs = builder.createOrFold<IE::ReshapeOp>(appendLoc(lhs.getLoc(), "reshape_lhs"), lhs,
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, newLhsShape));
    auto reshapeRhs = builder.createOrFold<IE::ReshapeOp>(appendLoc(rhs.getLoc(), "reshape_rhs"), rhs,
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, newRhsShape));
    auto fullyConnected =
            builder.create<IE::FullyConnectedOp>(origOp.getLoc(), reshapeLhs, reshapeRhs, /*bias=*/nullptr);
    auto reshapeOut = builder.createOrFold<IE::ReshapeOp>(
            appendLoc(out.getLoc(), "reshape_out"), fullyConnected.getOutput(), /*shape=*/nullptr,
            /*special_zero=*/false, /*shape_value=*/getIntArrayAttr(ctx, outShape));
    origOp.getOutput().replaceAllUsesWith(reshapeOut);
    origOp.erase();
}

void To4D(IE::MatMulOp origOp) {
    const auto lhs = origOp.getInput1();
    const auto rhs = origOp.getInput2();
    const auto out = origOp.getOutput();
    const auto lhsShape = getShape(lhs);
    const auto rhsShape = getShape(rhs);
    const auto outShape = getShape(out);
    // Convert
    // IE.MatMul(%arg0, %arg1) : [batch1, batch2, ..., rows, columns]
    // into
    // IE.MatMul(%arg0, %arg1) : [1, batch1 * batch2 * ..., rows, columns]
    const auto lhsRank = lhsShape.size();
    const auto rhsRank = rhsShape.size();
    const auto outRank = outShape.size();
    if (lhsRank == 4 && rhsRank == 4 && outRank == 4 && outShape[Dims4D::Act::N] == 1) {
        return;
    }
    if (lhsRank < 2 || rhsRank < 2) {
        return;
    }
    const auto lhsOrder = DimsOrder::fromValue(lhs);
    const auto rhsOrder = DimsOrder::fromValue(rhs);
    const auto outOrder = DimsOrder::fromValue(out);
    if (!lhsOrder.isIdentity() || !rhsOrder.isIdentity() || !outOrder.isIdentity()) {
        return;
    }
    // Exclude row and column dimensions from the list, multiply only batches.
    const auto lhsBatch = std::accumulate(lhsShape.begin(), lhsShape.end() - 2, 1, std::multiplies<int64_t>());
    const auto rhsBatch = std::accumulate(rhsShape.begin(), rhsShape.end() - 2, 1, std::multiplies<int64_t>());
    const Shape newLhsShape = {1, lhsBatch, lhsShape[Dim(lhsRank - 2)], lhsShape[Dim(lhsRank - 1)]};
    const Shape newRhsShape = {1, rhsBatch, rhsShape[Dim(rhsRank - 2)], rhsShape[Dim(rhsRank - 1)]};

    auto ctx = origOp.getContext();
    mlir::OpBuilder builder(origOp);
    auto reshapeLhs = builder.createOrFold<IE::ReshapeOp>(appendLoc(lhs.getLoc(), "reshape_lhs"), lhs,
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, newLhsShape));
    auto reshapeRhs = builder.createOrFold<IE::ReshapeOp>(appendLoc(rhs.getLoc(), "reshape_rhs"), rhs,
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, newRhsShape));
    auto newMatMul = builder.create<IE::MatMulOp>(origOp.getLoc(), reshapeLhs, reshapeRhs, origOp.getTransposeA(),
                                                  origOp.getTransposeB(), origOp.getPostOpAttr());
    auto reshapeOut = builder.createOrFold<IE::ReshapeOp>(appendLoc(out.getLoc(), "reshape_out"), newMatMul.getOutput(),
                                                          /*shape=*/nullptr, /*special_zero=*/false,
                                                          /*shape_value=*/getIntArrayAttr(ctx, outShape));

    origOp.getOutput().replaceAllUsesWith(reshapeOut);
    origOp.erase();
}

void SoftMaxTo4D(IE::SoftMaxOp origOp) {
    // To maintain SDP(MatMul-SoftMax-MatMul) fusion, we need to reshape SoftMax to 4D aligned with reshaped MatMul.
    auto skipFakeQuantizeIfPresent = [](mlir::Operation* op) -> mlir::Operation* {
        if (!mlir::isa_and_nonnull<IE::FakeQuantizeOp>(op)) {
            return op;
        }
        if (!op->hasOneUse()) {
            return nullptr;
        }
        return op->getOperand(0).getDefiningOp();
    };
    auto inputReshapeOp =
            mlir::dyn_cast_or_null<IE::ReshapeOp>(skipFakeQuantizeIfPresent(origOp->getOperand(0).getDefiningOp()));
    if (inputReshapeOp == nullptr) {
        return;
    }
    if (!mlir::isa_and_nonnull<IE::MatMulOp>(inputReshapeOp.getInput().getDefiningOp())) {
        return;
    }

    const auto input = origOp.getInput();
    const auto out = origOp.getOutput();
    const auto inShape = getShape(input);
    const auto outShape = getShape(out);
    // Convert
    // IE.SoftMax(%arg0) : [batch1, batch2, ..., rows, columns]
    // into
    // IE.SoftMax(%arg0) : [1, batch1 * batch2 * ..., rows, columns]
    const auto inputRank = inShape.size();
    const auto outRank = outShape.size();
    if (inputRank == 4 && outRank == 4 && outShape[Dims4D::Act::N] == 1) {
        return;
    }
    if (inputRank < 2) {
        return;
    }
    auto axisValue = origOp.getAxisInd();
    axisValue = axisValue < 0 ? axisValue + inputRank : axisValue;
    if (axisValue < int64_t(inputRank - 2)) {
        return;
    }
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(out);
    if (!inOrder.isIdentity() || !outOrder.isIdentity()) {
        return;
    }
    // Exclude row and column dimensions from the list, multiply only batches.
    const auto inBatch = std::accumulate(inShape.begin(), inShape.end() - 2, 1, std::multiplies<int64_t>());
    const Shape newInShape = {1, inBatch, inShape[Dim(inputRank - 2)], inShape[Dim(inputRank - 1)]};
    axisValue += int64_t(newInShape.size() - inputRank);

    auto ctx = origOp.getContext();
    mlir::OpBuilder builder(origOp);
    auto reshapeInput = builder.createOrFold<IE::ReshapeOp>(appendLoc(input.getLoc(), "reshape_in"), input,
                                                            /*shape=*/nullptr, /*special_zero=*/false,
                                                            /*shape_value=*/getIntArrayAttr(ctx, newInShape));
    auto newSoftMax = builder.create<IE::SoftMaxOp>(origOp->getLoc(), reshapeInput, getIntAttr(ctx, axisValue),
                                                    origOp.getPadSizeAttr());
    auto reshapeOut =
            builder.createOrFold<IE::ReshapeOp>(appendLoc(out.getLoc(), "reshape_out"), newSoftMax.getOutput(),
                                                /*shape=*/nullptr, /*special_zero=*/false,
                                                /*shape_value=*/getIntArrayAttr(ctx, outShape));

    origOp.getOutput().replaceAllUsesWith(reshapeOut);
    origOp.erase();
}

class ReshapeMatMulInputsPass final : public IE::impl::ReshapeMatMulInputsBase<ReshapeMatMulInputsPass> {
public:
    explicit ReshapeMatMulInputsPass(const bool enableGroupedMatMul, Logger log)
            : _enableGroupedMatMul(enableGroupedMatMul) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableGroupedMatMul = false;
};

mlir::LogicalResult ReshapeMatMulInputsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!enableGroupedMatMul.hasValue()) {
        return mlir::success();
    }

    _enableGroupedMatMul = enableGroupedMatMul;
    return mlir::success();
}

void ReshapeMatMulInputsPass::safeRunOnFunc() {
    auto func = getOperation();
    func.walk(InputsTo2D);
    if (_enableGroupedMatMul) {
        func.walk(TransposeInputs);
        func.walk(CollapseBatch);
        func.walk(To4D);
    }
    func.walk(SoftMaxTo4D);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createReshapeMatMulInputsPass(const bool enableGroupedMatMul, Logger log) {
    return std::make_unique<ReshapeMatMulInputsPass>(enableGroupedMatMul, log);
}
