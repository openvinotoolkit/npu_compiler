//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PADDYNAMICINPUTS
#define GEN_PASS_DEF_PADDYNAMICINPUTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
void padInput(mlir::Operation* firstOp) {
    mlir::OpBuilder builder{firstOp};
    for (unsigned i = 0; i < firstOp->getNumOperands(); ++i) {
        auto operand = firstOp->getOperand(i);
        if (IE::hasDynamicShapeAttr(operand)) {
            auto expand = builder.create<IE::DynamicExpandOp>(
                    appendLoc(firstOp->getLoc(), "_expand_" + std::to_string(i)), operand);
            firstOp->setOperand(i, expand.getOutput());
        }
    }
}

SmallVector<mlir::Operation*> getDynamicOperations(mlir::Operation* op, Logger log) {
    SmallVector<mlir::Operation*> dynamicOps;
    SmallVector<mlir::Operation*> current = {op};
    while (!current.empty()) {
        SmallVector<mlir::Operation*> next;
        for (const auto opIdx : irange(current.size())) {
            if (!IE::needsStaticShape(current[opIdx])) {
                continue;
            }
            auto origType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
            auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(origType);
            if (boundedType == nullptr) {
                log.trace("Op {0} at loc {1} does not have output bounds.", current[opIdx]->getName(),
                          current[opIdx]->getLoc());
                return {};
            }
            dynamicOps.push_back(current[opIdx]);
            for (const auto operandIdx : irange(current[opIdx]->getNumOperands())) {
                auto currentOp = current[opIdx]->getOperand(operandIdx);
                if (getShape(currentOp).isDynamic()) {
                    next.push_back(currentOp.getDefiningOp());
                }
            }
        }
        current = std::move(next);
    }
    return dynamicOps;
}

void freezeOutputShape(mlir::Operation* op) {
    auto origType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(origType);
    VPUX_THROW_UNLESS(boundedType != nullptr, "Expected to get BoundedTensorType at {0}", op->getResult(0).getLoc());
    auto bounds = boundedType.getBounds();
    const auto newShape = bounds.raw();
    const auto newType = mlir::RankedTensorType::get(newShape, origType.getElementType(),
                                                     getTensorAttr(op->getContext(), origType.getDimsOrder(), nullptr));

    op->getResult(0).setType(newType);
    // TODO(#157061): outputType of DynamicDataMaskOp depends on an attribute
    if (auto generateDynGarbageOp = mlir::dyn_cast<IE::DynamicDataMaskOp>(op)) {
        generateDynGarbageOp.setOutputTensorType(newType);
    }
}

void traverseDynamicSubgraph(IE::DynamicReshapeOp dynReshape, Logger log) {
    log.debug("Got '{0}' at '{1}'", dynReshape->getName(), dynReshape->getLoc());

    auto nestedLog = log.nest();
    mlir::Value dynReshapeInput{dynReshape.getInput()};
    if (mlir::isa<mlir::BlockArgument>(dynReshapeInput)) {
        nestedLog.trace("DynamicReshape has BlockArg input");
        return;
    }
    auto dynReshapeParent = dynReshapeInput.getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::StridedSliceOp>(dynReshapeParent) && !IE::needsStaticShape(dynReshapeParent)) {
        nestedLog.trace("DynamicReshape's parent is not StridedSlice.");
        return;
    }
    auto producer = mlir::isa<IE::StridedSliceOp>(dynReshapeParent) ? dynReshapeParent->getOperand(0).getDefiningOp()
                                                                    : dynReshapeParent;
    const auto dynamicOps = getDynamicOperations(producer, nestedLog);
    if (dynamicOps.empty()) {
        nestedLog.trace("No dynamic ops found.");
        return;
    }
    std::for_each(dynamicOps.begin(), dynamicOps.end(), freezeOutputShape);

    nestedLog.debug("Adding dynamic padding to the input of the first op in chain.");
    std::for_each(dynamicOps.begin(), dynamicOps.end(), padInput);
}

class PadDynamicInputsPass final : public IE::impl::PadDynamicInputsBase<PadDynamicInputsPass> {
public:
    explicit PadDynamicInputsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PadDynamicInputsPass::safeRunOnFunc() {
    getOperation()->walk([&](IE::DynamicReshapeOp dynReshape) {
        traverseDynamicSubgraph(dynReshape, _log);
    });
}
};  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPadDynamicInputsPass(Logger log) {
    return std::make_unique<PadDynamicInputsPass>(log);
}
