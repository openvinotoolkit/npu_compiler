//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/Pass.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_PREPAREDYNAMICKERNELALLOCATIONSIZES
#define GEN_PASS_DEF_PREPAREDYNAMICKERNELALLOCATIONSIZES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Find the latest safe insertion point for the helper reify sequence.
// We want to place the generated values right before the consuming op so we do not break
// dominance for any values they produce or use.
// Some SW DMA ops consume an auxiliary buffer materialized as VPU.Empty. When that buffer is
// placed after the other same-block inputs, we anchor to it so the reify sequence is inserted
// before the buffer and stays away from the consumer. The opposite ordering is unsupported,
// because the helpers would end up between the buffer and the consumer and can break outlining.
void setInsertionPointAfterInputs(mlir::OpBuilder& builder, mlir::Operation* op) {
    mlir::Operation* anchor = nullptr;
    mlir::Operation* emptyAnchor = nullptr;

    for (auto operand : op->getOperands()) {
        auto definingOp = operand.getDefiningOp();
        if (definingOp == nullptr) {
            continue;
        }

        if (mlir::isa<VPU::EmptyOp>(definingOp)) {
            emptyAnchor = definingOp;
            continue;
        }

        if (definingOp->getBlock() != op->getBlock()) {
            continue;
        }

        if (anchor == nullptr || anchor->isBeforeInBlock(definingOp)) {
            anchor = definingOp;
        }
    }

    if (anchor != nullptr && emptyAnchor != nullptr) {
        VPUX_THROW_UNLESS(mlir::isa<VPU::SwIoDmaOpInterface>(op), "Unexpected VPU.Empty aux buffer on non-SW DMA op");
        // VPU.Empty must come after the other same-block inputs; otherwise we would place the
        // helper sequence between the empty buffer and its consumer
        VPUX_THROW_UNLESS(anchor->isBeforeInBlock(emptyAnchor), "Unexpected op between VPU.Empty and its consumer");
        builder.setInsertionPoint(emptyAnchor);
        return;
    }

    if (anchor != nullptr) {
        builder.setInsertionPointAfter(anchor);
        return;
    }

    if (emptyAnchor != nullptr && emptyAnchor->getBlock() == op->getBlock()) {
        builder.setInsertionPoint(emptyAnchor);
        return;
    }

    builder.setInsertionPoint(op);
}

//
// PrepareDynamicKernelAllocationSizesPass
//
class PrepareDynamicKernelAllocationSizesPass final :
        public VPU::impl::PrepareDynamicKernelAllocationSizesBase<PrepareDynamicKernelAllocationSizesPass> {
public:
    explicit PrepareDynamicKernelAllocationSizesPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() override {
        auto module = getOperation();

        SmallVector<mlir::Operation*> opsToRewrite;
        module.walk([&](mlir::Operation* op) {
            if (op->getName().getDialectNamespace() != VPU::VPUDialect::getDialectNamespace()) {
                return;
            }

            // At this point, any remaining operations that were not placed inside an scf.for can be
            // treated as truly dynamic
            if (op->getParentOfType<mlir::scf::ForOp>() != nullptr) {
                return;
            }

            const bool hasDynamicTensorResult = llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
                const auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type);
                return shapedType != nullptr && !shapedType.hasStaticShape();
            });

            if (!hasDynamicTensorResult) {
                return;
            }

            if (!mlir::isa<mlir::ReifyRankedShapedTypeOpInterface>(op)) {
                VPUX_THROW("Unsupported dynamic VPU op outside scf.for does not implement "
                           "ReifyRankedShapedTypeOpInterface: '{0}'",
                           *op);
            }

            opsToRewrite.push_back(op);
        });

        auto builder = mlir::OpBuilder(module.getContext());
        for (auto* op : opsToRewrite) {
            const auto loc = op->getLoc();

            setInsertionPointAfterInputs(builder, op);

            mlir::ReifiedRankedShapedTypeDims reifiedShapes;
            if (mlir::failed(mlir::reifyResultShapes(builder, op, reifiedShapes))) {
                VPUX_THROW("Failed to reify result shapes for op '{0}'", *op);
            }

            for (auto resultIdx : llvm::seq<unsigned>(0, op->getNumResults())) {
                auto result = op->getResult(resultIdx);
                const auto resultType = mlir::dyn_cast<mlir::ShapedType>(result.getType());
                if (resultType == nullptr || resultType.hasStaticShape()) {
                    continue;
                }

                SmallVector<mlir::Value> dynamicDims;
                dynamicDims.reserve(resultType.getNumDynamicDims());
                for (int64_t dimIdx = 0; dimIdx < resultType.getRank(); ++dimIdx) {
                    if (!resultType.isDynamicDim(dimIdx)) {
                        continue;
                    }

                    dynamicDims.push_back(
                            mlir::getValueOrCreateConstantIndexOp(builder, loc, reifiedShapes[resultIdx][dimIdx]));
                }

                auto emptyOp = mlir::tensor::EmptyOp::create(builder, loc, resultType, dynamicDims);

                builder.setInsertionPointAfter(op);

                SmallVector<mlir::OpFoldResult> offsets(resultType.getRank(), builder.getIndexAttr(0));
                SmallVector<mlir::OpFoldResult> strides(resultType.getRank(), builder.getIndexAttr(1));
                auto insertSliceOp = mlir::tensor::InsertSliceOp::create(builder, loc, result, emptyOp.getResult(),
                                                                         offsets, reifiedShapes[resultIdx], strides);

                result.replaceAllUsesExcept(insertSliceOp->getResult(0), insertSliceOp);
            }
        }
    }

    Logger _log;
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createPrepareDynamicKernelAllocationSizesPass(Logger log) {
    return std::make_unique<PrepareDynamicKernelAllocationSizesPass>(std::move(log));
}
