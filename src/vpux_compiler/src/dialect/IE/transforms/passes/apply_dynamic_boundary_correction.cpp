//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reify_shape.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_APPLYDYNAMICBOUNDARYCORRECTION
#define GEN_PASS_DEF_APPLYDYNAMICBOUNDARYCORRECTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool needsClearGarbage(mlir::Operation* op) {
    if (mlir::isa_and_nonnull<StaticShapeOpInterface>(op) && !IE::hasDynamicTensors(op)) {
        return false;
    }
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        const auto filterShape = mlir::cast<NDTypeInterface>(convOp.getFilter().getType()).getShape();
        // TODO: E#160397
        // We actually should check the kernel size by the dynamic dimension only
        // if input.W is dynamic && filterShape[Dims4D::Filter::KX] > 1 ||
        //    input.H is dynamic && filterShape[Dims4D::Filter::KY] > 1
        return filterShape[Dims4D::Filter::KX] > 1 || filterShape[Dims4D::Filter::KY] > 1;
    } else if (auto maxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(op)) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(maxPoolOp.getKernelSize());
        // TODO: E#160397
        // We actually should check the kernel size by the dynamic dimension only
        // if input.W is dynamic && kernelSize[Dims4D::Kernel::X.ind()] > 1 ||
        //    input.H is dynamic && kernelSize[Dims4D::Kernel::Y.ind()] > 1
        return kernelSize[Dims4D::Kernel::X.ind()] > 1 || kernelSize[Dims4D::Kernel::Y.ind()] > 1;
    }
    return false;
}

bool producesDynamicGarbage(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::AddOp>(op) && IE::hasDynamicTensors(op);
}

bool checkForDynamicGarbage(mlir::Operation* startOp) {
    // Start from the given operation and traverse upwards
    mlir::Operation* currentOp = startOp;
    while (currentOp != nullptr) {
        // Check if the current operation produces dynamic garbage
        if (!IE::hasDynamicTensors(currentOp)) {
            // operation can produce dynamic garbage only if it has at least one dynamic tensor
            return false;
        }
        // specific ops only can produce the dynamic garbage
        if (producesDynamicGarbage(currentOp)) {
            return true;
        }
        // Move to the parent operation
        // TODO: E#160643 only lhs is processed. Have to care about all operand instead of lhs only
        currentOp = currentOp->getOperand(0).getDefiningOp();
    }
    return false;
}

void traverseDynamicSubgraph(mlir::Operation* op) {
    if (!needsClearGarbage(op)) {
        return;
    }

    if (checkForDynamicGarbage(op)) {
        IE::DynamicDimOpBuilder builder(op);
        for (auto operand : op->getOperands() | indexed) {
            if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(operand.value().getType())) {
                if (boundedType.getBounds().empty()) {
                    continue;
                }
                auto producer{operand.value().getDefiningOp()};

                builder.setInsertionPointAfter(producer);
                SmallVector<mlir::Value> dynamicOperands{};
                mlir::bufferization::populateDynamicDimSizes(builder, op->getLoc(), operand.value(), dynamicOperands);
                auto newShapeValue = buildConcat(
                        appendLoc(op->getLoc(), "clear_dyn_garbage_{0}_operand_{1}", op->getName(), operand.index()),
                        builder, getShape(operand.value()), dynamicOperands);

                auto clearGarbageMask = builder.create<IE::DynamicDataMaskOp>(
                        appendLoc(op->getLoc(), "clear_garbage_Generate_Mask_operand_{0}", operand.index()),
                        newShapeValue.getResult(), operand.value().getType());
                auto cleanOperand = builder.create<IE::MultiplyOp>(
                        appendLoc(op->getLoc(), "clear_garbage_Apply_Mask_operand_{0}", operand.index()),
                        operand.value(), clearGarbageMask.getResult(), vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT,
                        nullptr, nullptr, nullptr, nullptr);
                op->setOperand(operand.index(), cleanOperand.getResult());
            }
        }
    }
}  // namespace

class ApplyDynamicBoundaryCorrectionPass final :
        public IE::impl::ApplyDynamicBoundaryCorrectionBase<ApplyDynamicBoundaryCorrectionPass> {
public:
    explicit ApplyDynamicBoundaryCorrectionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ApplyDynamicBoundaryCorrectionPass::safeRunOnFunc() {
    getOperation()->walk(traverseDynamicSubgraph);
}

};  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createApplyDynamicBoundaryCorrectionPass(Logger log) {
    return std::make_unique<ApplyDynamicBoundaryCorrectionPass>(log);
}
