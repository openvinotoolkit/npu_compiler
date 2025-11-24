//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Support/LogicalResult.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSECONCATMATMUL
#define GEN_PASS_DEF_DECOMPOSECONCATMATMUL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Constants for optimization thresholds
constexpr int64_t MAX_SLICE_CHANNELS = 4;

//
// DecomposeConcatMatMulPass
//

class DecomposeConcatMatMulPass final : public IE::impl::DecomposeConcatMatMulBase<DecomposeConcatMatMulPass> {
public:
    explicit DecomposeConcatMatMulPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    bool isEfficientConcatMatMul(IE::MatMulOp matmulOp, IE::ConcatOp concatOp) const;
    void decomposeConcatMatMul(IE::MatMulOp matmulOp) const;
};

bool DecomposeConcatMatMulPass::isEfficientConcatMatMul(IE::MatMulOp matmulOp, IE::ConcatOp concatOp) const {
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    // Check if input shapes have at least 2 dimensions (for matrix multiplication)
    if (input1Shape.size() < 2 || input2Shape.size() < 2) {
        return false;
    }

    // Check concat axis is the last dimension (innermost)
    auto concatAxis = IE::getConcatAxis(concatOp);
    if (!concatAxis.has_value() || concatAxis.value().ind() != static_cast<int64_t>(input1Shape.size() - 1)) {
        return false;
    }

    // Check if all concat inputs have the same shape except for the concat dimension
    auto concatInputs = concatOp.getInputs();
    if (concatInputs.empty()) {
        return false;
    }

    auto firstInputShape = getShape(concatInputs[0]);
    if (firstInputShape.size() < 2) {
        return false;
    }

    // Check that the last dimension of each input is 1
    for (auto input : concatInputs) {
        auto inputShape = getShape(input);
        if (inputShape.size() != firstInputShape.size() || inputShape[Dim(inputShape.size() - 1)] != 1) {
            return false;
        }

        // Check that all dimensions except the last are the same
        for (size_t i = 0; i < inputShape.size() - 1; ++i) {
            if (inputShape[Dim(i)] != firstInputShape[Dim(i)]) {
                return false;
            }
        }
    }

    // Check MatMul has transposeB attribute
    if (!matmulOp.getTransposeB()) {
        return false;
    }

    // Check if the common dimension size equals number of concat inputs
    // With transposeB=true, the common dimension for matmul is:
    // - input1's last dimension (after concat): will be concatInputs.size()
    // - input2's last dimension (before transpose): input2Shape[last]
    auto input2CommonDim = input2Shape[Dim(input2Shape.size() - 1)];  // Last dimension before transpose
    if (static_cast<int64_t>(concatInputs.size()) != input2CommonDim) {
        return false;
    }

    // Get input and output channel dimensions
    auto inputChannels = input2CommonDim;  // K dimension
    auto outputChannels =
            input2Shape[Dim(input2Shape.size() - 2)];  // Output dimension (second-to-last before transpose)

    // Limit the number of slices to avoid generating too many operations
    // Both input and output channels should be small for efficient decomposition
    return inputChannels <= MAX_SLICE_CHANNELS && outputChannels <= MAX_SLICE_CHANNELS;
}

void DecomposeConcatMatMulPass::decomposeConcatMatMul(IE::MatMulOp matmulOp) const {
    _log.trace("Got '{0}' at '{1}'", matmulOp->getName(), matmulOp->getLoc());

    // Check if input1 is a ConcatOp
    auto concatOp = matmulOp.getInput1().getDefiningOp<IE::ConcatOp>();
    if (!concatOp) {
        return;
    }

    // Check if this is an inefficient Concat+MatMul pattern
    if (!isEfficientConcatMatMul(matmulOp, concatOp)) {
        return;
    }

    mlir::OpBuilder rewriter(matmulOp);
    auto input2Shape = getShape(matmulOp.getInput2());
    auto concatInputs = concatOp.getInputs();
    auto numInputs = concatInputs.size();

    _log.trace("Optimizing Concat+MatMul pattern with {0} inputs", numInputs);

    // Step 1: Slice the weight tensor (input2) along the common dimension
    // With transposeB=true, input2 shape is [1, 256, 2, 4], we need to slice along last dim (K dimension)
    SmallVector<mlir::Value> weightSlices;
    for (size_t i = 0; i < numInputs; ++i) {
        Shape sliceOffsets(input2Shape.size(), 0);
        sliceOffsets[Dim(input2Shape.size() - 1)] = static_cast<int64_t>(i);  // Slice along K dimension (last)
        auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffsets);

        Shape sliceSizes = input2Shape.raw();
        sliceSizes[Dim(input2Shape.size() - 1)] = 1;  // Take 1 slice along K dimension
        auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSizes);

        auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(matmulOp.getLoc(), "weight_slice_{0}", i),
                                                    matmulOp.getInput2(), offsetsAttr, sizesAttr);
        weightSlices.push_back(sliceOp.getOutput());
    }

    // Step 2: Further slice each weight slice along the output dimension
    // Each weight slice has shape [1, 256, 2, 1], slice along second-to-last dim to get [1, 256, 1, 1]
    auto outputDim = input2Shape[Dim(input2Shape.size() - 2)];  // Output dimension (second-to-last)
    SmallVector<SmallVector<mlir::Value>> weightSubSlices(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        for (int64_t j = 0; j < outputDim; ++j) {
            Shape sliceOffsets(input2Shape.size(), 0);
            sliceOffsets[Dim(input2Shape.size() - 2)] = j;
            auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffsets);

            auto sliceShape = getShape(weightSlices[i]);
            Shape sliceSizes = sliceShape.raw();
            sliceSizes[Dim(sliceShape.size() - 2)] = 1;
            auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSizes);

            auto subSliceOp =
                    rewriter.createOrFold<IE::SliceOp>(appendLoc(matmulOp.getLoc(), "weight_subslice_{0}_{1}", i, j),
                                                       weightSlices[i], offsetsAttr, sizesAttr);
            weightSubSlices[i].push_back(subSliceOp);
        }
    }

    // Step 3: Perform element-wise multiplication for each output channel
    SmallVector<mlir::Value> outputChannels;

    for (int64_t outputIdx = 0; outputIdx < outputDim; ++outputIdx) {
        SmallVector<mlir::Value> products;

        for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
            auto multiplyOp = rewriter.create<IE::MultiplyOp>(
                    appendLoc(matmulOp.getLoc(), "multiply_{0}_{1}", inputIdx, outputIdx), concatInputs[inputIdx],
                    weightSubSlices[inputIdx][outputIdx], IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr,
                    nullptr);
            products.push_back(multiplyOp.getOutput());
        }

        // Add all products together
        VPUX_THROW_UNLESS(!products.empty(), "Products vector should not be empty");
        mlir::Value result = products[0];
        for (size_t i = 1; i < products.size(); ++i) {
            result = rewriter.create<IE::AddOp>(appendLoc(matmulOp.getLoc(), "add_{0}_{1}", outputIdx, i), result,
                                                products[i], IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr,
                                                nullptr)
                             .getOutput();
        }

        outputChannels.push_back(result);
    }

    // Step 4: Concatenate output channels
    mlir::Value finalResult;
    if (outputChannels.size() == 1) {
        finalResult = outputChannels[0];
    } else {
        // Concatenate along the last dimension
        auto concatOutputOp =
                rewriter.create<IE::ConcatOp>(appendLoc(matmulOp.getLoc(), "final_concat"), outputChannels,
                                              static_cast<int64_t>(input2Shape.size() - 1));
        finalResult = concatOutputOp.getOutput();
    }

    _log.trace("Successfully optimized Concat+MatMul pattern");

    // Replace the MatMul operation with the decomposed result
    matmulOp.getResult().replaceAllUsesWith(finalResult);
    matmulOp.erase();
}

//
// safeRunOnFunc
//

void DecomposeConcatMatMulPass::safeRunOnFunc() {
    auto func = getOperation();

    SmallVector<IE::MatMulOp> matmulOps;
    func.walk([&](IE::MatMulOp matmulOp) {
        matmulOps.push_back(matmulOp);
    });

    for (auto matmulOp : llvm::make_early_inc_range(matmulOps)) {
        decomposeConcatMatMul(matmulOp);
    }
}

}  // namespace

//
// createDecomposeConcatMatMulPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeConcatMatMulPass(const Logger& log) {
    return std::make_unique<DecomposeConcatMatMulPass>(log);
}
