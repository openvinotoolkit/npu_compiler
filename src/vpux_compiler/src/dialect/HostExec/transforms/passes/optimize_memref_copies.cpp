//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace vpux::HostExec {
#define GEN_PASS_DECL_OPTIMIZEMEMREFCOPIES
#define GEN_PASS_DEF_OPTIMIZEMEMREFCOPIES
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

//
// OptimizeMemRefCopiesPass
//

class OptimizeMemRefCopiesPass final : public HostExec::impl::OptimizeMemRefCopiesBase<OptimizeMemRefCopiesPass> {
public:
    explicit OptimizeMemRefCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeMemRefCopiesPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());

    //
    // pattern to optimize memref.copy operations that copy from a call result to a subview
    //
    func.walk([&](mlir::memref::CopyOp copyOp) {
        mlir::Value src = copyOp.getSource();
        mlir::Value dst = copyOp.getTarget();

        auto callOp = src.getDefiningOp<mlir::CallOpInterface>();
        if (!callOp || callOp->getNumResults() != 1) {
            return;
        }

        auto subviewOp = dst.getDefiningOp<mlir::memref::SubViewOp>();
        if (!subviewOp) {
            return;
        }

        if (callOp->getNumOperands() < 1) {
            return;
        }

        mlir::Value oldDest = callOp->getOperand(callOp->getNumOperands() - 1);

        // Check if the old destination was from an alloc (we want to remove it)
        auto allocOp = oldDest.getDefiningOp<mlir::memref::AllocOp>();
        if (!allocOp) {
            return;
        }

        // Replace call operands to use the subview instead of the alloc
        SmallVector<mlir::Value> newOperands(callOp->getOperands().begin(), callOp->getOperands().end());
        builder.setInsertionPointAfter(subviewOp);
        if (src.getType() != dst.getType()) {
            // E169895: If the types are different, we need to cast the result of the subview to match the call operand
            // type
            auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(callOp->getLoc(), src.getType(), dst);
            newOperands.back() = castOp.getResult(0);
        } else {
            newOperands.back() = subviewOp.getResult();
        }

        auto* newCallOp = callOp->clone();
        newCallOp->setOperands(newOperands);
        builder.insert(newCallOp);

        callOp->getResult(0).replaceAllUsesWith(newCallOp->getResult(0));

        copyOp.erase();
        callOp.erase();
        if (allocOp->use_empty()) {
            allocOp.erase();
        }
    });
    //
    // pattern to optimize memref.copy operations that copy a result from scf.index_switch to a subview. It avoid the
    // buffer copies by replacing all allocations instance with the subview operations matching original destination
    // subview. It also adds UnrealizedConversionCastOp to handle types mismatch
    //
    func.walk([&](mlir::memref::CopyOp copyOp) {
        mlir::Value src = copyOp.getSource();
        mlir::Value dst = copyOp.getTarget();

        auto indexSwitchOp = src.getDefiningOp<mlir::scf::IndexSwitchOp>();
        if (!indexSwitchOp || indexSwitchOp.getNumResults() != 1) {
            return;
        }

        auto subviewOp = dst.getDefiningOp<mlir::memref::SubViewOp>();
        if (!subviewOp) {
            return;
        }

        SmallVector<mlir::memref::AllocOp> allocOpsToErase;
        for (mlir::Region* region : indexSwitchOp.getRegions()) {
            auto& block = region->front();
            for (auto& op : llvm::make_early_inc_range(block)) {
                if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
                    for (auto operand : callOp.getOperands()) {
                        auto allocOp = operand.getDefiningOp<mlir::memref::AllocOp>();
                        if (allocOp) {
                            builder.setInsertionPointAfter(allocOp);
                            auto newSubviewOp = builder.create<mlir::memref::SubViewOp>(
                                    subviewOp.getLoc(), subviewOp.getSource(), subviewOp.getMixedOffsets(),
                                    subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
                            mlir::Value newDst = newSubviewOp.getResult();
                            if (newDst.getType() != operand.getType()) {
                                // E169895: If the types are different, we need to cast the result of the subview to
                                // match the call operand type
                                auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
                                        callOp.getLoc(), operand.getType(), newDst);
                                newDst = castOp.getResult(0);
                            }
                            allocOp.getResult().replaceAllUsesWith(newDst);
                            allocOpsToErase.push_back(allocOp);
                            break;
                        }
                    }
                }
            }
        }

        copyOp.erase();
        subviewOp.erase();
        for (auto allocOp : llvm::make_early_inc_range(allocOpsToErase)) {
            if (allocOp->use_empty()) {
                allocOp.erase();
            }
        }
    });
    //
    // pattern to optimize memref.copy operations that copy from an alloc to a block argument
    //
    func.walk([&](mlir::memref::CopyOp copyOp) {
        mlir::Value src = copyOp.getSource();
        mlir::Value dst = copyOp.getTarget();

        auto allocOp = src.getDefiningOp<mlir::memref::AllocOp>();
        if (!allocOp) {
            return;
        }

        if (!mlir::isa<mlir::BlockArgument>(dst)) {
            return;
        }

        SmallVector<mlir::memref::SubViewOp> subviewOps;
        for (auto user : allocOp->getUsers()) {
            if (auto subviewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(user)) {
                subviewOps.push_back(subviewOp);
            }
        }
        for (auto subviewOp : subviewOps) {
            builder.setInsertionPointAfter(subviewOp);
            auto newSubview =
                    builder.create<mlir::memref::SubViewOp>(subviewOp.getLoc(), dst, subviewOp.getMixedOffsets(),
                                                            subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
            subviewOp.getResult().replaceAllUsesWith(newSubview.getResult());
            subviewOp.erase();
        }

        allocOp.getResult().replaceAllUsesWith(dst);
        copyOp.erase();
        if (allocOp->use_empty()) {
            allocOp.erase();
        }
    });
}

}  // namespace

//
// createOptimizeMemRefCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createOptimizeMemRefCopiesPass(Logger log) {
    return std::make_unique<OptimizeMemRefCopiesPass>(log);
}
