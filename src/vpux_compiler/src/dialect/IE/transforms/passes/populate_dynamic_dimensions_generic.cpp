//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <utility>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reify_shape.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace vpux::IE {
#define GEN_PASS_DECL_POPULATEDYNAMICDIMENSIONSGENERIC
#define GEN_PASS_DEF_POPULATEDYNAMICDIMENSIONSGENERIC
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// temporary limit the pass to use only limited number of operations
bool isSupportedOp(mlir::Operation* op) {
    return mlir::isa<IE::SoftMaxOp>(op);
}

bool supportsStridedAccess(mlir::Operation* op) {
    return mlir::isa<IE::SoftMaxOp>(op);
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
    mlir::OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    mlir::bufferization::populateDynamicDimSizes(builder, op->getLoc(), result, dynamicResults);

    auto concat = buildConcat(op->getLoc(), builder, resultShape, dynamicResults);

    auto newResult = [&]() -> mlir::Value {
        if (supportsStridedAccess(op)) {
            const SmallVector<int64_t> outputShape{resultShape.raw()};
            auto reshape = builder.create<IE::DynamicReshapeOp>(
                    appendLoc(op->getLoc(), "reshape"),
                    /*data=*/op->getResult(0),
                    /*shape=*/concat.getOutput(),
                    /*output_shape=*/getIntArrayAttr(builder.getContext(), outputShape),
                    /*output_bounds=*/getBounds(op->getResult(0)),
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
