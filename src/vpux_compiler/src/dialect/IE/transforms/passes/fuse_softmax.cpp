//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <optional>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESOFTMAX
#define GEN_PASS_DEF_FUSESOFTMAX
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <typename OpType>
OpType getOpSkippingFakeQuantize(mlir::Value value) {
    auto op = value.getDefiningOp<OpType>();
    if (op != nullptr) {
        return op;
    }

    auto fakeQuantOp = value.getDefiningOp<IE::FakeQuantizeOp>();
    if (fakeQuantOp != nullptr) {
        return fakeQuantOp.getInput().getDefiningOp<OpType>();
    }

    return nullptr;
}

template <typename ReduceOpType>
bool validateReduceAxes(ReduceOpType reduceOp, int64_t rank) {
    auto axesValue = reduceOp.getAxesValue();
    if (!axesValue.has_value()) {
        return false;
    }

    auto axesValues = parseIntArrayAttr<int64_t>(axesValue.value());
    if (axesValues.empty()) {
        return false;
    }

    for (auto& axis : axesValues) {
        axis = axis < 0 ? axis + rank : axis;
        if (axis < 0 || axis >= rank) {
            return false;
        }
    }
    return true;
}

struct AxisInfo {
    SmallVector<int64_t> axes;
    bool isSingleAxis;
    bool isConsecutive;
    bool isInnermost;
    int64_t softmaxAxis;
    bool needsTranspose() const {
        return !isSingleAxis && (!isConsecutive || !isInnermost);
    }
};

bool areAxesConsecutive(const SmallVector<int64_t>& axes) {
    for (size_t i = 1; i < axes.size(); ++i) {
        if (axes[i] != axes[i - 1] + 1) {
            return false;
        }
    }
    return true;
}

bool areAxesInnermost(const SmallVector<int64_t>& axes, int64_t rank) {
    if (axes.empty()) {
        return false;
    }

    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] != rank - static_cast<int64_t>(axes.size()) + static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

template <typename ReduceOpType>
std::optional<AxisInfo> getAxisInfo(ReduceOpType reduceOp, int64_t rank) {
    auto axesValue = reduceOp.getAxesValue();
    if (!axesValue.has_value()) {
        return std::nullopt;  // Return null when no valid axes are available
    }

    AxisInfo info;
    info.axes = parseIntArrayAttr<int64_t>(axesValue.value());
    for (auto& axis : info.axes) {
        axis = axis < 0 ? axis + rank : axis;  // Normalize negative axes
    }
    std::sort(info.axes.begin(), info.axes.end());

    info.isSingleAxis = (info.axes.size() == 1);
    info.isConsecutive = areAxesConsecutive(info.axes);
    info.softmaxAxis = info.axes[0];
    info.isInnermost = areAxesInnermost(info.axes, rank);

    return info;
}

class FuseSoftmaxPass final : public IE::impl::FuseSoftmaxBase<FuseSoftmaxPass> {
public:
    explicit FuseSoftmaxPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    // Helper functions for better code organization
    SmallVector<int64_t> createTransposeOrder(const AxisInfo& axisInfo, int64_t rank);
    SmallVector<int64_t> createReshapeDims(const AxisInfo& axisInfo, vpux::ShapeRef originalShape,
                                           int64_t concatenatedDim);
    int64_t calculateConcatenatedDim(const AxisInfo& axisInfo, vpux::ShapeRef originalShape);
    mlir::Value createTransposeWithReshape(mlir::OpBuilder& builder, IE::DivideOp divideOp, mlir::Value input,
                                           const AxisInfo& axisInfo, vpux::ShapeRef originalShape);
    mlir::Value createReshapeOnly(mlir::OpBuilder& builder, IE::DivideOp divideOp, mlir::Value input,
                                  const AxisInfo& axisInfo, vpux::ShapeRef originalShape);
};

SmallVector<int64_t> FuseSoftmaxPass::createTransposeOrder(const AxisInfo& axisInfo, int64_t rank) {
    SmallVector<int64_t> transposeOrder;
    for (int64_t i = 0; i < rank; ++i) {
        if (std::find(axisInfo.axes.begin(), axisInfo.axes.end(), i) == axisInfo.axes.end()) {
            transposeOrder.push_back(i);
        }
    }
    for (auto axis : axisInfo.axes) {
        transposeOrder.push_back(axis);
    }

    return transposeOrder;
}

int64_t FuseSoftmaxPass::calculateConcatenatedDim(const AxisInfo& axisInfo, vpux::ShapeRef originalShape) {
    int64_t concatenatedDim = 1;
    for (auto axis : axisInfo.axes) {
        concatenatedDim *= originalShape[vpux::Dim(axis)];
    }
    return concatenatedDim;
}

SmallVector<int64_t> FuseSoftmaxPass::createReshapeDims(const AxisInfo& axisInfo, vpux::ShapeRef originalShape,
                                                        int64_t concatenatedDim) {
    const int64_t numOnesNeeded = static_cast<int64_t>(axisInfo.axes.size()) - 1;
    auto finalDims = SmallVector<int64_t>(numOnesNeeded, 1);

    if (axisInfo.needsTranspose()) {
        for (int64_t i = 0; i < static_cast<int64_t>(originalShape.size()); ++i) {
            if (std::find(axisInfo.axes.begin(), axisInfo.axes.end(), i) == axisInfo.axes.end()) {
                finalDims.push_back(originalShape[vpux::Dim(i)]);
            }
        }
        finalDims.push_back(concatenatedDim);
    } else {
        for (int64_t i = 0; i < axisInfo.axes[0]; ++i) {
            finalDims.push_back(originalShape[vpux::Dim(i)]);
        }
        finalDims.push_back(concatenatedDim);
        for (int64_t i = axisInfo.axes.back() + 1; i < static_cast<int64_t>(originalShape.size()); ++i) {
            finalDims.push_back(originalShape[vpux::Dim(i)]);
        }
    }

    return finalDims;
}

mlir::Value FuseSoftmaxPass::createTransposeWithReshape(mlir::OpBuilder& builder, IE::DivideOp divideOp,
                                                        mlir::Value input, const AxisInfo& axisInfo,
                                                        vpux::ShapeRef originalShape) {
    auto rank = static_cast<int64_t>(originalShape.size());
    auto transposeOrder = createTransposeOrder(axisInfo, rank);
    auto concatenatedDim = calculateConcatenatedDim(axisInfo, originalShape);

    // Create forward transpose
    auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transposeOrder, builder.getContext()));
    auto transposeOp = builder.create<IE::TransposeOp>(takeOpLoc(divideOp, "transpose_in"), input, nullptr, orderAttr);

    // Reshape for softmax
    auto reshapeDims = createReshapeDims(axisInfo, originalShape, concatenatedDim);
    auto inputShapeAttr = getIntArrayAttr(builder.getContext(), reshapeDims);
    auto reshapeInputOp = builder.create<IE::ReshapeOp>(takeOpLoc(divideOp, "reshape_in"), transposeOp.getOutput(),
                                                        nullptr, false, inputShapeAttr);

    // Apply softmax
    int64_t softmaxAxis = static_cast<int64_t>(reshapeDims.size()) - 1;
    auto softmaxAxisAttr = getIntAttr(builder.getContext(), softmaxAxis);
    auto softmaxOp = builder.create<IE::SoftMaxOp>(takeOpLoc(divideOp, "softmax"), reshapeInputOp.getOutput(),
                                                   softmaxAxisAttr, nullptr);

    // Reshape back to transposed shape
    SmallVector<int64_t> transposedShape;
    for (auto idx : transposeOrder) {
        transposedShape.push_back(originalShape[vpux::Dim(idx)]);
    }
    auto transposedShapeAttr = getIntArrayAttr(builder.getContext(), transposedShape);
    auto reshapeOutputOp = builder.create<IE::ReshapeOp>(takeOpLoc(divideOp, "reshape_out"), softmaxOp.getOutput(),
                                                         nullptr, false, transposedShapeAttr);

    // Transpose back to original order
    auto forwardOrderMap = mlir::AffineMap::getPermutationMap(transposeOrder, builder.getContext());
    auto inverseOrderMap = mlir::inversePermutation(forwardOrderMap);
    auto inverseOrderAttr = mlir::AffineMapAttr::get(inverseOrderMap);
    auto transposeBackOp = builder.create<IE::TransposeOp>(takeOpLoc(divideOp, "transpose_out"),
                                                           reshapeOutputOp.getOutput(), nullptr, inverseOrderAttr);

    return transposeBackOp.getOutput();
}

mlir::Value FuseSoftmaxPass::createReshapeOnly(mlir::OpBuilder& builder, IE::DivideOp divideOp, mlir::Value input,
                                               const AxisInfo& axisInfo, vpux::ShapeRef originalShape) {
    auto concatenatedDim = calculateConcatenatedDim(axisInfo, originalShape);

    auto reshapeDims = createReshapeDims(axisInfo, originalShape, concatenatedDim);
    auto inputShapeAttr = getIntArrayAttr(builder.getContext(), reshapeDims);
    auto reshapeInputOp =
            builder.create<IE::ReshapeOp>(takeOpLoc(divideOp, "reshape_in"), input, nullptr, false, inputShapeAttr);

    int64_t numOnesNeeded = static_cast<int64_t>(axisInfo.axes.size()) - 1;
    int64_t softmaxAxis = numOnesNeeded + axisInfo.axes[0];
    auto softmaxAxisAttr = getIntAttr(builder.getContext(), softmaxAxis);
    auto softmaxOp = builder.create<IE::SoftMaxOp>(takeOpLoc(divideOp, "softmax"), reshapeInputOp.getOutput(),
                                                   softmaxAxisAttr, nullptr);
    auto originalShapeAttr = getIntArrayAttr(builder.getContext(), originalShape);
    auto reshapeOutputOp = builder.create<IE::ReshapeOp>(takeOpLoc(divideOp, "reshape_out"), softmaxOp.getOutput(),
                                                         nullptr, false, originalShapeAttr);

    return reshapeOutputOp.getOutput();
}

//
// FuseSoftmax Pass
//
// This pass detects decomposed softmax patterns and fuses them into a
// single IE.SoftMax operation for improved performance. The pass identifies
// the standard softmax decomposition pattern:
//   ReduceMax -> Subtract -> Exp -> ReduceSum -> Divide
//
// The pattern may optionally include FakeQuantize operations at various points.
// When detected, the entire sequence is replaced with optimized operations based
// on the reduction axes configuration:
//
// AXIS HANDLING:
// - Single axis: SoftMax operation directly with the specified axis (no reshapes needed)
// - Multiple consecutive axes at the end: Reshape -> SoftMax -> Reshape (no transpose needed)
// - Multiple non-consecutive axes: Transpose -> Reshape -> SoftMax -> Reshape -> Transpose
//
void FuseSoftmaxPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk([&](IE::DivideOp divideOp) {
        auto builder = mlir::OpBuilder(divideOp);
        auto reduceSumOp = getOpSkippingFakeQuantize<IE::ReduceSumOp>(divideOp.getInput2());
        if (!reduceSumOp) {
            return;
        }

        auto inputType = mlir::cast<vpux::NDTypeInterface>(reduceSumOp.getInput().getType());
        int64_t rank = inputType.getShape().size();

        if (!validateReduceAxes(reduceSumOp, rank)) {
            return;
        }
        auto axisInfoOpt = getAxisInfo(reduceSumOp, rank);
        if (!axisInfoOpt.has_value()) {
            return;
        }
        auto axisInfo = axisInfoOpt.value();

        auto expOp = getOpSkippingFakeQuantize<IE::ExpOp>(divideOp.getInput1());
        if (!expOp || getOpSkippingFakeQuantize<IE::ExpOp>(reduceSumOp.getInput()) != expOp) {
            return;
        }

        auto subtractOp = getOpSkippingFakeQuantize<IE::SubtractOp>(expOp.getInput());
        if (!subtractOp) {
            return;
        }

        auto reduceMaxOp = getOpSkippingFakeQuantize<IE::ReduceMaxOp>(subtractOp.getInput2());
        if (!reduceMaxOp || !validateReduceAxes(reduceMaxOp, rank)) {
            return;
        }

        auto reduceMaxAxisInfoOpt = getAxisInfo(reduceMaxOp, rank);
        if (!reduceMaxAxisInfoOpt.has_value()) {
            return;
        }
        auto reduceMaxAxisInfo = reduceMaxAxisInfoOpt.value();
        if (axisInfo.axes != reduceMaxAxisInfo.axes || subtractOp.getInput1() != reduceMaxOp.getInput()) {
            return;
        }
        _log.trace("FuseSoftmax pattern detected at {0}", divideOp->getLoc());

        auto originalShape = inputType.getShape();
        mlir::Value softmaxOutput;

        if (axisInfo.isSingleAxis) {
            auto softmaxAxisAttr = getIntAttr(builder.getContext(), axisInfo.softmaxAxis);
            auto softmaxOp = builder.create<IE::SoftMaxOp>(takeOpLoc(divideOp, "softmax"), subtractOp.getInput1(),
                                                           softmaxAxisAttr, nullptr);
            softmaxOutput = softmaxOp.getOutput();
        } else if (axisInfo.needsTranspose()) {
            softmaxOutput =
                    createTransposeWithReshape(builder, divideOp, subtractOp.getInput1(), axisInfo, originalShape);
        } else {
            softmaxOutput = createReshapeOnly(builder, divideOp, subtractOp.getInput1(), axisInfo, originalShape);
        }

        divideOp.replaceAllUsesWith(softmaxOutput);
    });
}

}  // namespace

//
// createFuseSoftmaxPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSoftmaxPass(Logger log) {
    return std::make_unique<FuseSoftmaxPass>(log);
}
