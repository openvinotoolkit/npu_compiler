//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>

/// EditPlan is a utility for managing and applying atomic edits to MLIR operations, specifically for copy-result reuse
/// scenarios. The reason not to directly modify operations using MLIR's standard mutation APIs is every time IR
/// changed, MLIR rewriter driver collect and traverse the IR once. EditPlan allows batching these edits and applying
/// them in a single shot, improving efficiency and ensuring correctness. EditPlan also enables the multi-thread
/// opportunity by allowing edits to be collected concurrently and applied in a controlled manner.
///
namespace vpux::VPUIP::DmaOptimization {

enum class EditKind : uint8_t {
    ReplaceAllUsesWith,
    EraseOp,
};

struct EditOp final {
    EditKind kind = EditKind::EraseOp;
    mlir::Operation* op = nullptr;
    mlir::Value lhsValue;
    mlir::Value rhsValue;
};

class EditPlan final {
public:
    void replaceAllUsesWith(mlir::Value lhs, mlir::Value rhs);
    void eraseOp(mlir::Operation* op);

    llvm::ArrayRef<EditOp> ops() const {
        return _ops;
    }

    bool empty() const {
        return _ops.empty();
    }

    size_t size() const {
        return _ops.size();
    }

    mlir::LogicalResult verify(Logger log) const;

private:
    /// Ordered edit list applied by EditPlanApplier: all replacements first, erases second.
    SmallVector<EditOp, 0> _ops;
    /// Operations that already have an erase edit in _ops.
    llvm::DenseSet<mlir::Operation*> _erasedSet;
    /// Source values that already have a ReplaceAllUsesWith edit in _ops.
    llvm::DenseSet<mlir::Value> _replaceAllUsesSources;
};

/// Applies a verified CopyReuse EditPlan without rediscovering legality.
class EditPlanApplier final {
public:
    explicit EditPlanApplier(Logger log): _log(log) {
    }

    mlir::LogicalResult run(mlir::IRRewriter& rewriter, const EditPlan& plan);

private:
    Logger _log;
};

}  // namespace vpux::VPUIP::DmaOptimization
