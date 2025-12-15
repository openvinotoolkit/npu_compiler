//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LogicalResult.h>
#include <climits>
#include <cstdint>

namespace vpux::IE {
#define GEN_PASS_DECL_MATMULINPUTSTO2D
#define GEN_PASS_DEF_MATMULINPUTSTO2D
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 3;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

//
// MatMulInputsTo2dPass
//

class MatMulInputsTo2dPass final : public IE::impl::MatMulInputsTo2dBase<MatMulInputsTo2dPass> {
public:
    explicit MatMulInputsTo2dPass(const bool enableGroupedMatMul, Logger log)
            : _enableGroupedMatMul(enableGroupedMatMul) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class SwapInputsConverter;
    class ReshapeNDInputConverter;
    class MatMulOpConverter;

private:
    void safeRunOnFunc() final;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult MatMulInputsTo2dPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!enableGroupedMatMul.hasValue()) {
        return mlir::success();
    }

    _enableGroupedMatMul = enableGroupedMatMul;
    return mlir::success();
}

//
// MatMulOpConverter
//

class MatMulInputsTo2dPass::MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

static SmallVector<mlir::Value> sliceTensor(const mlir::Value tensorToSplit, const mlir::Location location,
                                            mlir::PatternRewriter& rewriter, const std::string& tensorName) {
    const auto tensorShape = getShape(tensorToSplit);
    int64_t batch = 1;
    int64_t width = 1;
    int64_t height = 1;
    auto channelDim = Dim(0);
    if (tensorShape.size() == 3) {
        batch = tensorShape[Dim(0)];
        height = tensorShape[Dim(1)];
        width = tensorShape[Dim(2)];
        channelDim = Dim(0);
    } else if (tensorShape.size() == 4) {
        batch = tensorShape[Dim(1)];
        height = tensorShape[Dim(2)];
        width = tensorShape[Dim(3)];
        channelDim = Dim(1);
    } else if (tensorShape.size() == 2) {
        return {tensorToSplit};
    }
    SmallVector<mlir::Value> weightSlices;
    Shape rhsShape2D{height, width};
    const auto rhsShape2DAttr = getIntArrayAttr(rewriter.getContext(), rhsShape2D);
    if (batch > 1) {
        for (int64_t sliceIdx = 0; sliceIdx < batch; sliceIdx++) {
            Shape sliceOffsets = Shape(tensorShape.size(), 0);
            sliceOffsets[channelDim] = checked_cast<int64_t>(sliceIdx);
            auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffsets);

            Shape sliceSizes = tensorShape.raw();
            sliceSizes[channelDim] = 1;
            auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSizes);
            auto newSubViewOp = rewriter.create<IE::SliceOp>(appendLoc(location, "{0}_slice_{1}", tensorName, sliceIdx),
                                                             tensorToSplit, staticOffsetsAttr, staticSizesAttr);

            auto rhs2d = rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape_{1}", tensorName, sliceIdx),
                                                        newSubViewOp, nullptr, false, rhsShape2DAttr);
            weightSlices.push_back(rhs2d);
        }
    } else {
        auto rhs2d = rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape", tensorName), tensorToSplit,
                                                    nullptr, false, rhsShape2DAttr);
        weightSlices.push_back(rhs2d);
    }

    return weightSlices;
}

mlir::LogicalResult MatMulInputsTo2dPass::MatMulOpConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    // E-122051:
    // MatMulInputsTo2dPass should be moved to a new pass `ConvertMatMulToFullyConnected`.
    // This check should be moved in a `addDynamicallyLegalOp<IE::MatMulOp>`.
    // Transpose should be done after `ReshapeNDInputConverter` (not in canonicalizer), experiments show that it
    // is faster when batch dimensions are merged.
    if (VPU::MatMulOp::isSupported(matmulOp)) {
        if (matmulOp.getTransposeB()) {
            auto input2 = matmulOp.getInput2();
            auto input2Rank = getShape(input2).size();
            VPUX_THROW_UNLESS(input2Rank > 2,
                              "VPU::MatMulOp only supports input 2 rank bigger than 2. "
                              "If that changes, this code needs update. Input 2 rank = '{0}'",
                              input2Rank);
            SmallVector<uint32_t> perm(input2Rank, 0);
            std::iota(perm.begin(), perm.end(), 0);
            std::iter_swap(perm.end() - 1, perm.end() - 2);
            const auto orderAttr =
                    mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
            input2 = rewriter.create<IE::TransposeOp>(takeOpLoc(matmulOp, "input_b_transpose"), input2, nullptr,
                                                      orderAttr)
                             .getOutput();
            rewriter.replaceOpWithNewOp<IE::MatMulOp>(matmulOp, matmulOp.getInput1(), input2, false, false,
                                                      matmulOp.getPostOpAttr());
            return mlir::success();
        }
        return mlir::failure();
    }

    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    // 1. Cover 3D input or weights.
    // 2. Cover 4D input and weights without batch.
    if (!(input1Shape.size() == 3 && input2Shape.size() == 3) &&
        !(input1Shape.size() == 4 &&
          ((input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1))) {
        return mlir::failure();
    }

    // Ideally this should be skipped using calculation from ReshapeNDInputConverter
    if (_enableGroupedMatMul && IE::isGroupedMatMulBeneficial(matmulOp, input1Shape, input2Shape)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> activationSlices =
            sliceTensor(matmulOp.getInput1(), matmulOp->getLoc(), rewriter, "activation");
    SmallVector<mlir::Value> weightSlices = sliceTensor(matmulOp.getInput2(), matmulOp->getLoc(), rewriter, "weights");

    // Handle broadcasting by replicating the slices of the broadcasted input to match
    // the number of slices of the non-broadcasted input.
    if (activationSlices.size() != weightSlices.size()) {
        if (activationSlices.size() == 1) {
            activationSlices = SmallVector<mlir::Value>(weightSlices.size(), activationSlices[0]);
        } else if (weightSlices.size() == 1) {
            weightSlices = SmallVector<mlir::Value>(activationSlices.size(), weightSlices[0]);
        } else {
            VPUX_THROW("Mismatch activationSlices number '{0}' with weightSlices number '{1}'", activationSlices.size(),
                       weightSlices.size());
        }
    }

    SmallVector<mlir::Value> matmulSlices;
    for (size_t sliceIdx = 0; sliceIdx < activationSlices.size(); sliceIdx++) {
        auto lhs2d = activationSlices[sliceIdx];
        auto rhs2d = weightSlices[weightSlices.size() == 1 ? 0 : sliceIdx];
        auto op = rewriter.create<IE::MatMulOp>(takeOpLoc(matmulOp, llvm::StringLiteral("slice_{0}"), sliceIdx), lhs2d,
                                                rhs2d, matmulOp.getTransposeA(), matmulOp.getTransposeB(),
                                                matmulOp.getPostOpAttr());
        matmulSlices.push_back(op.getOutput());
    }

    VPUX_THROW_WHEN(matmulSlices.empty(), "Cannot slice MatMul operation with input shape {0}, weights' shape {1}",
                    input1Shape, input2Shape);

    auto newOp = matmulSlices.size() != 1
                         ? rewriter.create<IE::ConcatOp>(takeOpLoc(matmulOp, "slice_gather"), matmulSlices, 0)
                         : matmulSlices.front();

    const auto outShape4D = getShape(matmulOp.getOutput());
    const auto outShape4DAttr = getIntArrayAttr(rewriter.getContext(), outShape4D);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newOp, nullptr, false, outShape4DAttr);

    return mlir::success();
}

//
// ReshapeNDInputConverter
//

class MatMulInputsTo2dPass::ReshapeNDInputConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    ReshapeNDInputConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult MatMulInputsTo2dPass::ReshapeNDInputConverter::matchAndRewrite(
        IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), matmulOp->getName(), matmulOp->getLoc());

    auto transposeA = matmulOp.getTransposeA();
    auto transposeB = matmulOp.getTransposeB();
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    auto adjustTo3DShape = [](ShapeRef origShape, const bool isFirstInput) {
        if (origShape.size() == 1) {
            return isFirstInput ? Shape{1, 1, origShape.front()} : Shape{1, origShape.front(), 1};
        }

        auto batchSize = std::accumulate(origShape.begin(), origShape.end() - 2, 1, std::multiplies<int64_t>());
        return Shape{batchSize, origShape[Dim(origShape.size() - 2)], origShape[Dim(origShape.size() - 1)]};
    };

    // Adjust second input
    // Step 1: Adjust input shape to a tensor rank of 3D [Batch, Height, Width]
    //  If the tensor is 1D, the size is assigned to Height, and both Batch and Width are set to 1
    //    - For example: [6] -> [1, 6, 1]
    //  If the tensor is larger or equal than 2D, the last two dimensions are assigned to Height and Width
    //  and Batch is set to the product of the remaining dimensions
    //    - For example: [2, 3] -> [1, 2, 3]; [1, 1, 6] -> [1, 1, 6]; [3, 1, 6, 4, 2] -> [18, 4, 2]
    // Step 2: The batch dimension can be removed if its size equals 1
    //    - For example: [1, 1, 1, 8] -> [1, 1, 8] -> [1, 8]
    //                   [1, 6, 1, 8] -> [6, 1, 8] -> [6, 1, 8]
    auto newIn2Shape = adjustTo3DShape(input2Shape, /*isFirstInput=*/false);
    if (newIn2Shape.front() == 1) {
        newIn2Shape.erase(newIn2Shape.begin());
    }

    // Adjust first input
    // Step 1: Adjust input shape to a tensor rank of 3D [Batch, Height, Width]
    //  If the tensor is 1D, the size is assigned to Width, and both Batch and Height are set to 1
    //    - For example: [6] -> [1, 1, 6]
    //  If the tensor is larger or equal than 2D, the last two dimensions are assigned to Height and Width
    //  and Batch is set to the product of the remaining dimensions
    //    - For example: [2, 3] -> [1, 2, 3]; [1, 2, 6] -> [1, 2, 6]; [3, 1, 6, 4, 2] -> [18, 4, 2]
    // Step 2: If transposeA is set to false or batch equal 1 and the new second input shape lacks a batch dimension
    //         the Batch can be integrated into the Height dimension.
    // For example:
    //   MatMul(2x3x6x4, 4x8) {transposeA = false} collapses to MatMul(36x4, 4x8) {transposeA = false}
    //   MatMul(2x3x4x6, 4x8) {transposeA = true} collapses to MatMul(6x4x6, 4x8) {transposeA = true}
    //   MatMul(1x2x6x4, 2x4x8) {transposeA = false} collapses to MatMul(2x6x4, 2x4x8) {transposeA = false}
    auto newIn1Shape = adjustTo3DShape(input1Shape, /*isFirstInput=*/true);
    if (newIn2Shape.size() == 2 && (!transposeA || newIn1Shape.front() == 1)) {
        auto batchSize = newIn1Shape.front();
        newIn1Shape.erase(newIn1Shape.begin());
        newIn1Shape[Dim(0)] = newIn1Shape[Dim(0)] * batchSize;
    }
    if (newIn1Shape == input1Shape && newIn2Shape == input2Shape) {
        return mlir::failure();
    }

    if (_enableGroupedMatMul && newIn1Shape.size() > 2 && newIn2Shape.size() > 2 && newIn1Shape.front() != 1 &&
        newIn2Shape.front() != 1) {
        if (IE::isGroupedMatMulBeneficial(matmulOp, newIn1Shape, newIn2Shape)) {
            return mlir::failure();
        }
    }

    // Check if the original input shapes are either both 3D or both 4D without batch
    // If they can be converted to MatMul without batch, the shapes will be adjusted
    // If not, the following conversion logic can directly slice them without the need for reshaping
    const auto isNewShapeWithoutBatch = (newIn1Shape.size() == 2) && (newIn2Shape.size() == 2);
    const auto is3DInput = (input1Shape.size() == 3) && (input2Shape.size() == 3);
    const auto is4DInputWithoutBatch =
            input1Shape.size() == 4 && (input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1;
    if (!isNewShapeWithoutBatch && (is3DInput || is4DInputWithoutBatch)) {
        return mlir::failure();
    }

    // Adjust MatMul inputs
    auto adjustInputTensor = [&](mlir::Value input, ShapeRef newShape, mlir::Location newLoc) {
        return rewriter.createOrFold<IE::ReshapeOp>(newLoc, input, nullptr, false,
                                                    getIntArrayAttr(rewriter.getContext(), newShape));
    };

    auto reshapeInput1 = adjustInputTensor(matmulOp.getInput1(), newIn1Shape, takeOpLoc(matmulOp, "in1_reshape"));
    auto reshapeInput2 = adjustInputTensor(matmulOp.getInput2(), newIn2Shape, takeOpLoc(matmulOp, "in2_reshape"));

    auto newMatMul = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), reshapeInput1, reshapeInput2, transposeA,
                                                   transposeB, matmulOp.getPostOpAttr());

    const auto origOutShape = getShape(matmulOp.getOutput());
    const auto origOutShapeAttr = getIntArrayAttr(rewriter.getContext(), origOutShape);
    auto newOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newMatMul.getOutput(), nullptr, false,
                                                            origOutShapeAttr);
    extendOpLoc(newOp, "out_reshape");
    return mlir::success();
}

//
// SwapInputsConverter
//

/*
Swap input1 with input2 when input1 is 2D after invalid dimensions removed and dimensions of input2 are bigger than 2D
For example:
    IE.MatMul(input1, input2) {transpose_b} : tensor<1x1x32x64xf32>, tensor<1x6336x1x64xf32> -> tensor<1x6336x32x1xf32>
-->
    IE.MatMul(input2, input1) {transpose_b} : tensor<1x6336x1x64xf32>, tensor<32x64xf32> -> tensor<1x6336x1x32xf32>

Then using ReshapeNDInputConverter pattern to convert the new input1 to 2D as:
    IE.MatMul(input1, input2) {transpose_b} : tensor<6336x64xf32>, tensor<32x64xf32> -> tensor<6336x32xf32>
*/
class MatMulInputsTo2dPass::SwapInputsConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    SwapInputsConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MatMulInputsTo2dPass::SwapInputsConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), matmulOp->getName(), matmulOp->getLoc());
    auto ctx = rewriter.getContext();

    auto transposeA = matmulOp.getTransposeA();
    auto transposeB = matmulOp.getTransposeB();
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    auto adjustTo2DShape = [](ShapeRef origShape) -> std::optional<Shape> {
        if (origShape.size() == 1) {
            return std::nullopt;
        }

        auto batchSize = std::accumulate(origShape.begin(), origShape.end() - 2, 1, std::multiplies<int64_t>());
        if (batchSize != 1) {
            return std::nullopt;
        }

        return Shape{origShape[Dim(origShape.size() - 2)], origShape[Dim(origShape.size() - 1)]};
    };

    auto newIn1ShapeValue = adjustTo2DShape(input1Shape);
    if (!newIn1ShapeValue.has_value()) {
        return mlir::failure();
    }
    auto newIn1Shape = newIn1ShapeValue.value();

    // When both input1 and input2 can be reshaped to 2D, matmulOp can directly be rewrited by ReshapeNDInputConverter
    auto inShape2Size = input2Shape.size();
    if (inShape2Size <= 2 || adjustTo2DShape(input2Shape).has_value()) {
        return mlir::failure();
    }

    const auto dimToCheck = transposeB ? input2Shape[Dim(inShape2Size - 2)] : input2Shape[Dim(inShape2Size - 1)];
    if (dimToCheck != 1) {
        return mlir::failure();
    }

    auto layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(matmulOp.getOperation());
    if (layerWithPostOp) {
        const auto postOp = layerWithPostOp.getPostOp();
        if (postOp != nullptr && !postOp.isChannelAgnostic()) {
            return mlir::failure();
        }
    }

    auto newTransposeB = transposeA ? false : true;
    auto newTransposeA = transposeB ? false : true;

    auto in2ReshapeOp = rewriter.create<IE::ReshapeOp>(matmulOp->getLoc(), matmulOp.getInput1(), nullptr, false,
                                                       getIntArrayAttr(ctx, newIn1Shape));

    auto newMatMulOp = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), matmulOp.getInput2(), in2ReshapeOp.getOutput(),
                                                     newTransposeA, newTransposeB, matmulOp.getPostOpAttr());

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newMatMulOp.getOutput(), nullptr, false,
                                               getIntArrayAttr(ctx, getShape(matmulOp.getOutput())));
    return mlir::success();
}

//
// safeRunOnFunc
//

void MatMulInputsTo2dPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapInputsConverter>(&ctx, benefitLevels[0], _log);
    patterns.add<ReshapeNDInputConverter>(&ctx, benefitLevels[1], _log, _enableGroupedMatMul);
    patterns.add<MatMulOpConverter>(&ctx, benefitLevels[2], _log, _enableGroupedMatMul);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMatMulInputsTo2dPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMatMulInputsTo2dPass(const bool enableGroupedMatMul, Logger log) {
    return std::make_unique<MatMulInputsTo2dPass>(enableGroupedMatMul, log);
}
