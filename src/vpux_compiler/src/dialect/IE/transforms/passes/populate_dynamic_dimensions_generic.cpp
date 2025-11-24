//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reify_shape.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_POPULATEDYNAMICDIMENSIONSGENERIC
#define GEN_PASS_DEF_POPULATEDYNAMICDIMENSIONSGENERIC
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// temporary limit the pass to use only limited number of operations
bool isSupportedOp(mlir::Operation* op) {
    return mlir::isa<IE::SoftMaxOp, IE::MinimumOp, IE::MaximumOp, IE::LSTMSequenceOp, IE::LessOp, IE::LessEqualOp,
                     IE::SubtractOp>(op);
}

bool supportsStridedAccess(mlir::Operation* op) {
    return mlir::isa<IE::SoftMaxOp, IE::MinimumOp, IE::MaximumOp, IE::LSTMSequenceOp, IE::LessOp, IE::LessEqualOp,
                     IE::SubtractOp>(op);
}

void populateDynamicResult(mlir::Operation* op, const unsigned resultIdx) {
    mlir::Value result{op->getResult(resultIdx)};
    auto resultType = mlir::cast<NDTypeInterface>(result.getType());
    const auto resultShape = resultType.getShape();
    if (resultShape.isStatic()) {
        return;
    }

    SmallVector<mlir::OpOperand*> oldUses;
    for (auto& use : result.getUses()) {
        oldUses.push_back(&use);
    }

    SmallVector<mlir::Value> dynamicResults{};
    IE::DynamicDimOpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    mlir::bufferization::populateDynamicDimSizes(builder, op->getLoc(), result, dynamicResults);

    auto concat = buildConcat(op->getLoc(), builder, resultShape, dynamicResults);

    auto newResult = [&]() -> mlir::Value {
        if (supportsStridedAccess(op)) {
            const SmallVector<int64_t> outputShape{resultShape.raw()};
            auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(op->getResult(0).getType());
            VPUX_THROW_UNLESS(boundedType != nullptr, "Expected to get BoundedTensorType at {0}",
                              op->getResult(0).getLoc());

            auto reshape = builder.create<IE::DynamicReshapeOp>(
                    appendLoc(op->getLoc(), "reshape"),
                    /*data=*/op->getResult(0),
                    /*shape=*/concat.getOutput(),
                    /*output_shape=*/getIntArrayAttr(builder.getContext(), outputShape),
                    /*output_bounds=*/getIntArrayAttr(builder.getContext(), boundedType.getBounds()),
                    /*only_set_shape*/ true);

            return reshape.getResult();
        }

        return repackDynamicTensor(builder, op, resultType, concat);
    }();

    for (auto oldUse : oldUses) {
        oldUse->set(newResult);
    }
}

void populateDynamicSizes(mlir::ReifyRankedShapedTypeOpInterface op) {
    if (!isSupportedOp(op)) {
        return;
    }

    for (const unsigned idx : irange(op->getNumResults())) {
        populateDynamicResult(op, idx);
    }
}

}  // namespace

namespace {

class PopulateDynamicDimensionsGenericPass final :
        public IE::impl::PopulateDynamicDimensionsGenericBase<PopulateDynamicDimensionsGenericPass> {
public:
    explicit PopulateDynamicDimensionsGenericPass(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PopulateDynamicDimensionsGenericPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk(populateDynamicSizes);
}

};  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPopulateDynamicDimensionsGenericPass(Logger log) {
    return std::make_unique<PopulateDynamicDimensionsGenericPass>(std::move(log));
}
