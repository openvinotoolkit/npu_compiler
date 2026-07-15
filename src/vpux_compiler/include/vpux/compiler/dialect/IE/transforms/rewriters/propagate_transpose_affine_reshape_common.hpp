//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

bool doesAffineReshapeChangeRank(IE::AffineReshapeOp reshape);

// Check if operation is a valid compute op (Conv/GroupConv or Elementwise)
bool isValidComputeOp(mlir::Operation* op);

// Find single operation of type OpType from the largest input on concat axis
// Returns the operation if found, nullptr otherwise
// This function prevents infinite loops by ensuring:
// 1. Only one input has maximum size on concat axis
// 2. Only the largest input has the target operation
// 3. All other inputs don't have the target operation
template <typename OpType>
OpType findSingleOpFromLargestInput(mlir::OperandRange inputs, const mlir::DenseSet<int64_t>& modifiedAxes,
                                    const Logger& log) {
    // Only support single dimension concat
    if (modifiedAxes.size() != 1) {
        return nullptr;
    }

    const auto concatAxis = *modifiedAxes.begin();

    struct InputInfo {
        mlir::Value value;
        int64_t concatAxisSize;
    };

    SmallVector<InputInfo> inputInfos;
    inputInfos.reserve(inputs.size());

    for (const auto& input : inputs) {
        const auto inputShape = getShape(input);
        inputInfos.push_back({input, inputShape[Dim(concatAxis)]});
    }

    // Sort by concat axis size in descending order
    llvm::sort(inputInfos, [](const InputInfo& a, const InputInfo& b) {
        return a.concatAxisSize > b.concatAxisSize;
    });

    // Check if only the largest input on concat axis exists
    const int64_t maxConcatAxisSize = inputInfos[0].concatAxisSize;
    size_t maxSizeInputCount = 0;

    for (const auto& info : inputInfos) {
        if (info.concatAxisSize == maxConcatAxisSize) {
            maxSizeInputCount++;
        }
    }

    // Prevent infinite loop: only apply if there's exactly one max-size input on concat axis
    if (maxSizeInputCount > 1) {
        log.trace("Multiple inputs with same max concat axis size, avoiding potential infinite loop");
        return nullptr;
    }

    // Check if the largest input has the target operation
    auto targetOp = mlir::dyn_cast_if_present<OpType>(inputInfos[0].value.getDefiningOp());
    if (targetOp == nullptr) {
        return nullptr;
    }

    // Check that other inputs don't have the target operation
    for (size_t i = 1; i < inputInfos.size(); i++) {
        if (mlir::isa_and_present<OpType>(inputInfos[i].value.getDefiningOp())) {
            log.trace("Other inputs also have the target operation, avoiding transformation");
            return nullptr;
        }
    }

    return targetOp;
}

SmallVector<int64_t> invertDimMappingWithAxesNotSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping,
                                                              ShapeRef affineInShape, ShapeRef affineOutShape);

bool areModifiedAxesSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping, ShapeRef affineInShape,
                                  ShapeRef affineOutShape, const mlir::DenseSet<int64_t>& modifiedAxes, bool swapOrder,
                                  Logger log);

std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithAffineReshape(IE::SoftMaxOp softmaxOp,
                                                                       IE::AffineReshapeOp affineReshapeOp,
                                                                       const Logger& log);

std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithShapeCast(IE::SoftMaxOp softmaxOp, IE::ShapeCastOp shapeCastOp,
                                                                   const Logger& log);

std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithShapeCast(int64_t softmaxLogicalAxis,
                                                                   IE::ShapeCastOp shapeCastOp, const Logger& log);

//
// MoveTransposeAffineReshapeThroughAdd
//

/* Rewrite the pattern from:

        Transpose
            |
      AffineReshape
            |
           Add
            |
      (QuantizeCast)

    to:
           Add
            |
      (QuantizeCast)
            |
        Transpose
            |
      AffineReshape
*/

class MoveTransposeAffineReshapeThroughAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    MoveTransposeAffineReshapeThroughAdd(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx, benefit), _log(log) {
        this->setDebugName("MoveTransposeAffineReshapeThroughAdd");
    }
    enum InputsMode {
        Asymmetry = 0,
        Symmetry = 1,
        Unsupported = 2,
    };

private:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isBeneficialConversion(IE::AddOp origOp, InputsMode mode) const;
    std::tuple<InputsMode, IE::TransposeOp, IE::AffineReshapeOp> checkAddInputsMode(IE::AddOp origOp) const;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
