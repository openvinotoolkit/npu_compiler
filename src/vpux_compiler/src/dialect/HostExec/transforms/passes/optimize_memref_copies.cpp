//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"

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

mlir::memref::AllocOp replaceAllocWithSubview(mlir::Value value, mlir::memref::SubViewOp subviewOp,
                                              mlir::OpBuilder& builder) {
    auto* definingOp = value.getDefiningOp();
    if (auto castOp = mlir::dyn_cast_or_null<mlir::memref::CastOp>(definingOp)) {
        definingOp = castOp.getSource().getDefiningOp();
    }
    auto allocOp = mlir::dyn_cast_or_null<mlir::memref::AllocOp>(definingOp);
    if (allocOp == nullptr) {
        return nullptr;
    }
    builder.setInsertionPointAfter(allocOp);
    auto newSubviewOp = mlir::memref::SubViewOp::create(builder, subviewOp.getLoc(), subviewOp.getSource(),
                                                        subviewOp.getMixedOffsets(), subviewOp.getMixedSizes(),
                                                        subviewOp.getMixedStrides());
    allocOp.getResult().replaceAllUsesWith(newSubviewOp.getResult());
    return allocOp;
}

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
        if (callOp == nullptr || callOp->getNumResults() != 1) {
            return;
        }

        auto subviewOp = dst.getDefiningOp<mlir::memref::SubViewOp>();
        if (subviewOp == nullptr) {
            return;
        }

        if (callOp->getNumOperands() < 1) {
            return;
        }

        mlir::Value oldDest = callOp->getOperand(callOp->getNumOperands() - 1);

        // Check if the old destination was from an alloc (we want to remove it),
        // looking through a memref.cast if present.
        auto allocOp = replaceAllocWithSubview(oldDest, subviewOp, builder);
        if (allocOp == nullptr) {
            return;
        }

        copyOp.erase();
        subviewOp.erase();
        if (allocOp->use_empty()) {
            allocOp.erase();
        }
    });
    //
    // pattern to optimize memref.copy operations that copy a result from scf.index_switch to a subview. It avoid the
    // buffer copies by replacing all allocations instance with the subview operations matching original destination
    // subview
    //
    func.walk([&](mlir::memref::CopyOp copyOp) {
        mlir::Value src = copyOp.getSource();
        mlir::Value dst = copyOp.getTarget();

        auto indexSwitchOp = src.getDefiningOp<mlir::scf::IndexSwitchOp>();
        if (indexSwitchOp == nullptr || indexSwitchOp.getNumResults() != 1) {
            return;
        }

        auto subviewOp = dst.getDefiningOp<mlir::memref::SubViewOp>();
        if (subviewOp == nullptr) {
            return;
        }

        SmallVector<mlir::memref::AllocOp> allocOpsToErase;
        for (mlir::Region* region : indexSwitchOp.getRegions()) {
            auto& block = region->front();
            for (auto& op : llvm::make_early_inc_range(block)) {
                if (auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
                    for (auto operand : callOp.getArgOperands()) {
                        auto allocOp = replaceAllocWithSubview(operand, subviewOp, builder);
                        if (allocOp == nullptr) {
                            continue;
                        }
                        allocOpsToErase.push_back(allocOp);
                        break;
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
        if (allocOp == nullptr) {
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
                    mlir::memref::SubViewOp::create(builder, subviewOp.getLoc(), dst, subviewOp.getMixedOffsets(),
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
