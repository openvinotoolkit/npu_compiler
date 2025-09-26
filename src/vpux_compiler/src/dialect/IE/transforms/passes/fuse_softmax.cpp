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
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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

    auto axesArrayAttr = axesValue.value();
    SmallVector<int64_t> axesValues;
    for (auto attr : axesArrayAttr) {
        axesValues.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }

    if (axesValues.size() != 2) {
        return false;
    }

    auto normalizeAxis = [rank](int64_t axis) {
        return axis < 0 ? axis + rank : axis;
    };

    int64_t axis1 = normalizeAxis(axesValues[0]);
    int64_t axis2 = normalizeAxis(axesValues[1]);

    return (axis1 == rank - 2 && axis2 == rank - 1) || (axis1 == rank - 1 && axis2 == rank - 2);
}

class FuseSoftmaxPass final : public IE::impl::FuseSoftmaxBase<FuseSoftmaxPass> {
public:
    explicit FuseSoftmaxPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// FuseSoftmax Pass
//
// This pass detects decomposed softmax patterns in the IR and fuses them into
// a single IE.SoftMax operation for improved performance. The pass identifies
// the standard softmax decomposition pattern:
//   ReduceMax -> Subtract -> Exp -> ReduceSum -> Divide
//
// The pattern may optionally include FakeQuantize operations at various points.
// When detected, the entire sequence is replaced with:
//   [FakeQuantize] -> Reshape -> SoftMax -> Reshape
//
// LIMITATION: This pass only applies when both ReduceMax and ReduceSum operate on
// the last 2 dimensions of the input tensor.
// Examples of supported patterns:
//   - 3D tensor [D0, D1, D2]: Both operations must use axes [-2, -1] or [1, 2]
//   - 4D tensor [D0, D1, D2, D3]: Both operations must use axes [-2, -1] or [2, 3]
//   - 5D tensor [D0, D1, D2, D3, D4]: Both operations must use axes [-2, -1] or [3, 4]
//

void FuseSoftmaxPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk([&](IE::DivideOp divideOp) {
        auto builder = mlir::OpBuilder(divideOp);

        auto reduceSumOp = getOpSkippingFakeQuantize<IE::ReduceSumOp>(divideOp.getInput2());
        if (reduceSumOp == nullptr) {
            return;
        }

        auto inputType = mlir::cast<vpux::NDTypeInterface>(reduceSumOp.getInput().getType());
        auto inputShape = inputType.getShape();
        int64_t rank = inputShape.size();

        if (!validateReduceAxes(reduceSumOp, rank)) {
            return;
        }

        auto expOp = getOpSkippingFakeQuantize<IE::ExpOp>(divideOp.getInput1());
        if (expOp == nullptr) {
            return;
        }

        auto expOpFromReduceSum = getOpSkippingFakeQuantize<IE::ExpOp>(reduceSumOp.getInput());
        if (expOpFromReduceSum != expOp) {
            return;
        }

        auto subtractOp = getOpSkippingFakeQuantize<IE::SubtractOp>(expOp.getInput());
        if (subtractOp == nullptr) {
            return;
        }

        auto reduceMaxOp = getOpSkippingFakeQuantize<IE::ReduceMaxOp>(subtractOp.getInput2());
        if (reduceMaxOp == nullptr) {
            return;
        }

        if (!validateReduceAxes(reduceMaxOp, rank)) {
            return;
        }

        mlir::Value subtractInput1 = subtractOp.getInput1();
        mlir::Value reduceMaxInput = reduceMaxOp.getInput();
        if (subtractInput1 != reduceMaxInput) {
            return;
        }

        // Pattern matched successfully - create the fused SoftMax operation
        auto originalInputType = mlir::cast<vpux::NDTypeInterface>(subtractInput1.getType());
        auto originalShape = originalInputType.getShape();

        // Calculate the reshaped dimensions
        // Reshape from [D0, D1, ..., D(n-2), D(n-1)] to [1, D0, D1, ..., D(n-2)*D(n-1)]
        SmallVector<int64_t> reshapeInputDims;
        reshapeInputDims.push_back(1);
        for (int64_t i = 0; i < rank - 2; ++i) {
            reshapeInputDims.push_back(originalShape[vpux::Dim(i)]);
        }
        int64_t flattenedDim = originalShape[vpux::Dim(rank - 2)] * originalShape[vpux::Dim(rank - 1)];
        reshapeInputDims.push_back(flattenedDim);

        const auto inputShapeAttr = getIntArrayAttr(builder.getContext(), reshapeInputDims);
        auto reshapeInputOp = builder.create<IE::ReshapeOp>(takeOpLoc(divideOp, "reshape_in"), subtractInput1, nullptr,
                                                            false, inputShapeAttr);
        const auto optimalAxis = -1;
        const auto optimalAxisAttr = getIntAttr(builder.getContext(), optimalAxis);
        auto newSoftMaxOp = builder.create<IE::SoftMaxOp>(takeOpLoc(divideOp, "softmax"), reshapeInputOp.getOutput(),
                                                          optimalAxisAttr, nullptr);
        const auto outputShapeAttr = getIntArrayAttr(builder.getContext(), originalShape.raw());
        auto reshapeOutputOp = builder.create<IE::ReshapeOp>(takeOpLoc(divideOp, "reshape_out"),
                                                             newSoftMaxOp.getOutput(), nullptr, false, outputShapeAttr);
        divideOp->replaceAllUsesWith(reshapeOutputOp);
    });
}

}  // namespace

//
// createFuseSoftmaxPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSoftmaxPass(Logger log) {
    return std::make_unique<FuseSoftmaxPass>(log);
}
