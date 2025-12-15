//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/SmallVector.h>
#include <numeric>

using namespace vpux;

std::vector<mlir::Operation*> vpux::collectOpsForPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet& patterns) {
    // Collect both concrete op TypeIDs (for OpRewritePattern<OpT>) and interface TypeIDs
    // (for OpInterfaceRewritePattern<InterfaceT>) from the provided patterns.
    llvm::DenseSet<mlir::TypeID> targetOpTypeIDs;
    llvm::DenseSet<mlir::TypeID> targetInterfaceIDs;
    auto& patternVec = patterns.getNativePatterns();
    for (auto& patternPtr : patternVec) {
        auto rootKind = patternPtr->getRootKind();
        if (rootKind) {
            targetOpTypeIDs.insert(rootKind->getTypeID());
        }
        auto rootInterfaceId = patternPtr->getRootInterfaceID();
        if (rootInterfaceId) {
            targetInterfaceIDs.insert(*rootInterfaceId);
        }
    }

    // Collect operations matching the pattern types automatically
    std::vector<mlir::Operation*> relevantOps;
    func.walk([&](mlir::Operation* op) {
        bool matches = false;
        // Match by concrete op type (OpRewritePattern)
        if (!targetOpTypeIDs.empty()) {
            if (auto info = op->getRegisteredInfo()) {
                if (targetOpTypeIDs.contains(info->getTypeID())) {
                    matches = true;
                }
            }
        }
        // Match by interface (OpInterfaceRewritePattern)
        if (!matches && !targetInterfaceIDs.empty()) {
            auto name = op->getName();
            for (auto ifaceID : targetInterfaceIDs) {
                if (name.hasInterface(ifaceID)) {
                    matches = true;
                    break;
                }
            }
        }

        if (matches) {
            relevantOps.push_back(op);
        }
    });

    return relevantOps;
}

void vpux::applyPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet&& patterns, ArrayRef<mlir::Operation*> ops) {
    auto* ctx = func->getContext();
    mlir::PatternRewriter rewriter(ctx);
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    mlir::PatternApplicator patternApplicator(frozenPatterns);
    patternApplicator.applyDefaultCostModel();

    for (auto op : ops) {
        (void)patternApplicator.matchAndRewrite(op, rewriter);
    }
}

void vpux::collectOpsAndApplyPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet&& patterns) {
    auto relevantOps = collectOpsForPatterns(func, patterns);
    applyPatterns(func, std::move(patterns), relevantOps);
    runLocalDCE(func);
}

void vpux::runLocalDCE(mlir::func::FuncOp func) {
    llvm::SetVector<mlir::Operation*> worklist;

    // Seed with currently-dead ops
    func->walk([&](mlir::Operation* op) {
        if (mlir::isOpTriviallyDead(op)) {
            worklist.insert(op);
        }
    });

    // Erase dead ops, propagating to producers that may become dead
    while (!worklist.empty()) {
        mlir::Operation* op = worklist.pop_back_val();

        // Capture producers before erase
        SmallVector<mlir::Operation*, 8> producers;
        for (auto operand : op->getOperands()) {
            if (auto* definingOp = operand.getDefiningOp()) {
                producers.push_back(definingOp);
            }
        }

        op->erase();

        for (auto* definingOp : producers) {
            if (mlir::isOpTriviallyDead(definingOp)) {
                worklist.insert(definingOp);
            }
        }
    }
}
