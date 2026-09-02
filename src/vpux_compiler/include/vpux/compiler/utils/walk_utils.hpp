//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>

namespace mlir {
class Operation;
}  // namespace mlir

namespace mlir::func {
class FuncOp;
}  // namespace mlir::func

namespace vpux {

// Traverse patterns first and collect which ops/interfaces need to be collected.
// Then iterate over IR using func.walk and collect all ops related for patterns
std::vector<mlir::Operation*> collectOpsForPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet& patterns);

using OrderedPatternSet = SmallVector<mlir::RewritePatternSet>;

// Apply patterns to provided inputs using mlir pattern applicator
void applyPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet&& patterns, ArrayRef<mlir::Operation*> ops);

// Iterate over IR to collect ops once and then apply patterns to already collected ops
// This is very fast, downside is it only works on frozen view of the IR before any pattern is
// applied. Unlike ApplyPatternsAndFoldGreedily this does not do any folding or dead code (ops) elimination
void collectOpsAndApplyPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet&& patterns);

// Go over whole IR using func.walk and remove all dead code(ops)
void runLocalDCE(mlir::func::FuncOp func);

// Returns the first operation of type Op reachable from `current` through single-use chains,
// skipping pure view ops. Returns nullptr if the chain branches or the target is absent.
template <typename Op>
Op findUnfusedConsumer(const Logger& log, mlir::Operation* current, bool checkPostOp = true) {
    if (!current->hasOneUse()) {
        log.trace("findUnfusedConsumer: '{0}' at '{1}' has multiple uses, aborting", current->getName(),
                  current->getLoc());
        return nullptr;
    }

    auto user = *current->getUsers().begin();

    while (user != nullptr && IE::isPureViewOp(user)) {
        if (!user->hasOneUse()) {
            log.trace("findUnfusedConsumer: view op '{0}' at '{1}' has multiple uses, aborting", user->getName(),
                      user->getLoc());
            return nullptr;
        }
        user = *user->getUsers().begin();
    }

    auto targetOp = mlir::dyn_cast_or_null<Op>(user);
    if (checkPostOp) {
        if (targetOp == nullptr || targetOp->hasAttr(vpux::OperationAttrName::POST_OP) ||
            targetOp->hasAttr(vpux::OperationAttrName::CLAMP)) {
            log.trace("findUnfusedConsumer: target not found or has post-op/clamp attr");
            return nullptr;
        }
    }

    return targetOp;
}

// Returns an "origin" operation of the specified type (one of Ops), ignoring
// pure view ops, for the given operation. This procedure assumes IR in question
// is a chain of single-use operations.
template <typename Op>
Op findUnfusedProducer(const Logger& log, mlir::Operation* current, bool checkPostOp = true,
                       bool skipTranspose = false) {
    // skip "pure view ops"
    while (current && (IE::isPureViewOp(current) || (skipTranspose && mlir::isa<IE::TransposeOp>(current)))) {
        // Note: for the sake of this pass, only single-use op chains are
        // considered
        auto operands = current->getOperands();
        if (operands.size() != 1) {
            log.trace("findUnfusedProducer: '{0}' at '{1}' has unexpected number of operands {2}, expected 1",
                      current->getName(), current->getLoc(), operands.size());
            return nullptr;
        }

        // ViewOp should have single user
        if (!current->hasOneUse()) {
            log.trace("findUnfusedProducer: view op '{0}' at '{1}' has multiple uses, aborting", current->getName(),
                      current->getLoc());
            return nullptr;
        }

        current = operands[0].getDefiningOp();
    }

    auto originOp = mlir::dyn_cast_or_null<Op>(current);
    if (checkPostOp) {
        if (originOp == nullptr || originOp->hasAttr(vpux::OperationAttrName::POST_OP) ||
            originOp->hasAttr(vpux::OperationAttrName::CLAMP)) {
            log.trace("findUnfusedProducer: origin not found or has post-op/clamp attr");
            return nullptr;
        }
    }

    return originOp;
}

}  // namespace vpux
