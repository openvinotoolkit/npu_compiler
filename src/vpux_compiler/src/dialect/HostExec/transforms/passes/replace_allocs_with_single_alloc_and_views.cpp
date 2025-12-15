//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::HostExec {
#define GEN_PASS_DECL_REPLACEALLOCSWITHSINGLEALLOCANDVIEWS
#define GEN_PASS_DEF_REPLACEALLOCSWITHSINGLEALLOCANDVIEWS
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

//
// ReplaceAllocsWithSingleAllocAndViewsPass
//

class ReplaceAllocsWithSingleAllocAndViewsPass final :
        public HostExec::impl::ReplaceAllocsWithSingleAllocAndViewsBase<ReplaceAllocsWithSingleAllocAndViewsPass> {
public:
    explicit ReplaceAllocsWithSingleAllocAndViewsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ReplaceAllocsWithSingleAllocAndViewsPass::safeRunOnFunc() {
    auto func = getOperation();

    bool found = false;
    func->walk([&](mlir::memref::DeallocOp) {
        found = true;
    });
    if (found) {
        func->emitError(
                "ReplaceAllocsWithSingleAllocAndViewsPass cannot be applied to functions with dealloc operations");
        signalPassFailure();
    }

    mlir::PatternRewriter rewriter(func.getContext());

    SmallVector<mlir::memref::AllocOp> allocsToReplace;
    SmallVector<SmallVector<mlir::Value>> allocDynamicSizes;
    SmallVector<ArrayRef<int64_t>> allocShapes;
    SmallVector<mlir::Type> allocElemTypes;

    func.walk([&](mlir::memref::AllocOp allocOp) {
        allocsToReplace.push_back(allocOp);
        allocDynamicSizes.push_back(allocOp.getDynamicSizes());
        allocShapes.push_back(allocOp.getType().getShape());
        allocElemTypes.push_back(allocOp.getType().getElementType());
    });

    if (allocsToReplace.empty()) {
        return;
    }

    // compute total size and offsets (in bytes)

    // Check if the parent of the first alloc is in the main function scope and is not inside a nested scope
    // If we are in highest scope, we can insert the scratch alloc before the first alloc
    // Otherwise, we need to insert the scratch alloc at the beginning of the parent op
    auto parentOfFirstAllocToReplace = allocsToReplace.front()->getParentOp();
    if (mlir::isa<mlir::func::FuncOp>(parentOfFirstAllocToReplace)) {
        rewriter.setInsertionPoint(allocsToReplace.front());
    } else {
        rewriter.setInsertionPoint(parentOfFirstAllocToReplace);
    }

    auto loc = func.getLoc();

    SmallVector<mlir::Value> offsets;
    SmallVector<mlir::Value> sizes;

    for (size_t i = 0; i < allocsToReplace.size(); ++i) {
        mlir::Value size = nullptr;
        for (size_t d = 0, dynIdx = 0; d < allocShapes[i].size(); ++d) {
            mlir::Value dimSize = [&] {
                if (allocShapes[i][d] != mlir::ShapedType::kDynamic) {
                    return rewriter.create<mlir::arith::ConstantIndexOp>(loc, allocShapes[i][d]).getResult();
                }
                return allocDynamicSizes[i][dynIdx++];
            }();

            if (size == nullptr) {
                size = dimSize;
            } else {
                size = rewriter.create<mlir::arith::MulIOp>(loc, size, dimSize);
            }
        }

        auto elemSize = allocElemTypes[i].getIntOrFloatBitWidth() / 8;
        auto elemSizeValue = rewriter.create<mlir::arith::ConstantIndexOp>(loc, elemSize);
        auto allocSize = rewriter.create<mlir::arith::MulIOp>(loc, size, elemSizeValue);

        sizes.push_back(allocSize);
        if (i == 0) {
            offsets.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
        } else {
            auto offset = rewriter.create<mlir::arith::AddIOp>(loc, offsets.back(), sizes.back());
            offsets.push_back(offset);
        }
    }

    if (offsets.empty() || sizes.empty()) {
        return;
    }

    assert(offsets.size() == sizes.size() && sizes.size() == allocsToReplace.size() &&
           "Offsets, sizes and allocsToReplace should have the same size");

    mlir::Value totalSize = sizes.front();
    for (size_t i = 1; i < sizes.size(); ++i) {
        totalSize = rewriter.create<mlir::arith::AddIOp>(loc, totalSize, sizes[i]);
    }

    auto scratchBufferType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIntegerType(8));
    // align to DEFAULT_CMX_ALIGNMENT (64) as it is done for DDR scratch buffer on NPU
    mlir::Value scratchBufferAlloc = rewriter.create<mlir::memref::AllocOp>(
            loc, scratchBufferType, totalSize, mlir::ValueRange{}, rewriter.getI64IntegerAttr(DEFAULT_CMX_ALIGNMENT));

    for (size_t i = 0; i < allocsToReplace.size(); ++i) {
        auto allocType = allocsToReplace[i].getType();
        mlir::Value view = rewriter.create<mlir::memref::ViewOp>(loc, allocType, scratchBufferAlloc, offsets[i],
                                                                 allocDynamicSizes[i]);

        allocsToReplace[i].replaceAllUsesWith(view);
    }

    for (size_t i = 0; i < allocsToReplace.size(); ++i) {
        rewriter.eraseOp(allocsToReplace[i]);
    }
}

}  // namespace

//
// createReplaceAllocsWithSingleAllocAndViewsPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createReplaceAllocsWithSingleAllocAndViewsPass(Logger log) {
    return std::make_unique<ReplaceAllocsWithSingleAllocAndViewsPass>(log);
}
