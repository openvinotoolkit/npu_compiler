//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <unordered_map>

namespace vpux {
#define GEN_PASS_DECL_DEDUPLICATEANDLOWERIMMREGISTEROPS
#define GEN_PASS_DEF_DEDUPLICATEANDLOWERIMMREGISTEROPS
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
namespace {

bool isDstUse(mlir::OpOperand& use) {
    if (auto instr = mlir::dyn_cast<bytecode::InstructionOpInterface>(use.getOwner())) {
        return instr.isDstOperand(use);
    }
    return false;
}

// Create a (VVR + SetImmOp) pair before `insertBefore` and return the VVR result.
mlir::Value materialize(mlir::Operation* insertBefore, mlir::IntegerAttr valueAttr, mlir::MLIRContext* ctx) {
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPoint(insertBefore);
    auto reg = bytecode::VirtualValueRegisterOp::create(builder, insertBefore->getLoc());
    bytecode::SetImmOp::create(builder, insertBefore->getLoc(), reg.getResult(), valueAttr);
    return reg.getResult();
}

void lowerBlock(mlir::Block& block, const std::unordered_map<uint64_t, mlir::Value>& entryBlockVvrs,
                std::unordered_map<uint64_t, mlir::Value>& blockVvrs, mlir::MLIRContext* ctx) {
    auto immRegisterOps = to_small_vector(block.getOps<bytecode::ImmRegisterOp>());

    for (auto op : immRegisterOps) {
        auto immValue = op.getValue();

        SmallVector<mlir::OpOperand*> srcUses;
        SmallVector<mlir::OpOperand*> dstUses;
        for (auto& use : op.getResult().getUses()) {
            if (isDstUse(use)) {
                dstUses.push_back(&use);
            } else {
                srcUses.push_back(&use);
            }
        }

        // Source uses: share one VVR per value, preferring entry-block VVRs.
        if (!srcUses.empty()) {
            mlir::Value sharedVvr;
            auto entryIt = entryBlockVvrs.find(immValue);
            if (entryIt != entryBlockVvrs.end()) {
                sharedVvr = entryIt->second;
            } else {
                auto blockIt = blockVvrs.find(immValue);
                if (blockIt != blockVvrs.end()) {
                    sharedVvr = blockIt->second;
                } else {
                    sharedVvr = materialize(op, op.getValueAttr(), ctx);
                    blockVvrs[immValue] = sharedVvr;
                }
            }
            for (auto* use : srcUses) {
                use->set(sharedVvr);
            }
        }

        // Destination uses: each gets a fresh VVR to prevent corrupting the shared one.
        for (auto* use : dstUses) {
            mlir::Value freshVvr = materialize(op, op.getValueAttr(), ctx);
            use->set(freshVvr);
        }

        op.erase();
    }
}

void lowerInRegion(mlir::Region& body, mlir::MLIRContext* ctx) {
    if (body.empty()) {
        return;
    }

    // Pass 1: entry block — VVRs created here are reused in all successor blocks.
    std::unordered_map<uint64_t, mlir::Value> entryBlockVvrs;
    lowerBlock(body.front(), entryBlockVvrs, entryBlockVvrs, ctx);

    // Pass 2: non-entry blocks — reuse entry-block VVRs for source uses where
    // available; otherwise deduplicate within the current block.
    for (auto& block : llvm::drop_begin(body)) {
        std::unordered_map<uint64_t, mlir::Value> blockVvrs;
        lowerBlock(block, entryBlockVvrs, blockVvrs, ctx);
    }
}
}  // namespace

namespace vpux {

//
// DeduplicateAndLowerImmRegisterOpsPass
//
// Lowers every bytecode.imm_register op to a (VirtualValueRegisterOp + SetImmOp) pair.
//
// For each use of an ImmRegisterOp result:
//   - Source use (isDstOperand returns false): share one (VVR + SetImmOp) per constant
//     value. Entry-block VVRs are reused in all successor blocks (entry dominates all).
//     Within a non-entry block, same-value source uses share a block-scoped VVR.
//   - Destination use (isDstOperand returns true): always create a fresh (VVR + SetImmOp)
//     so the shared source VVR is never overwritten.
//

class DeduplicateAndLowerImmRegisterOpsPass final :
        public impl::DeduplicateAndLowerImmRegisterOpsBase<DeduplicateAndLowerImmRegisterOpsPass> {
private:
    void safeRunOnModule() final {
        mlir::MLIRContext* ctx = &getContext();
        getOperation().walk([ctx](mlir::Operation* op) {
            if (auto funcOp = mlir::dyn_cast<bytecode::FuncOp>(op)) {
                lowerInRegion(funcOp.getBody(), ctx);
            } else if (auto extFuncOp = mlir::dyn_cast<bytecode::ExtFuncOp>(op)) {
                lowerInRegion(extFuncOp.getBody(), ctx);
            }
        });
    }
};

}  // namespace vpux

std::unique_ptr<mlir::Pass> vpux::bytecode::createDeduplicateAndLowerImmRegisterOpsPass() {
    return std::make_unique<DeduplicateAndLowerImmRegisterOpsPass>();
}
