//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinOps.h>
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {
class BatchingSplitter {
public:
    BatchingSplitter(Logger log): _log(log) {
    }
    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp mainFunction);

private:
    Logger _log;
};

SmallVector<OutliningInstance> BatchingSplitter::getOutliningInstances(mlir::func::FuncOp mainFunction) {
    OutliningInstance splits(1);
    std::set<mlir::Operation*> coveredOps;

    mainFunction.walk([&](mlir::Operation* op) {
        if (mlir::isa<mlir::func::ReturnOp>(op) || op->getParentOp() != mainFunction) {
            return;
        }

        // a regular operation must be outlined rather than artificial `UnrealizedConversionCastOp` inserted by the
        // DebatcherPass
        auto& currentSplit = splits.front();
        if (!mlir::isa<mlir::UnrealizedConversionCastOp>(op)) {
            if (currentSplit.inputs.empty()) {
                VPUX_THROW_UNLESS(mlir::isa<Const::DeclareOp>(op),
                                  "Operations encountered at the beginning of a main function block right before  a "
                                  "first `unrealized cast` must be `Const::DeclareOp`");
                return;
            }
            currentSplit.operations.push_back(op);

            // update inputs & outputs of the slice
            llvm::copy_if(op->getOperands(), std::back_inserter(currentSplit.inputs), [](auto operand) {
                return mlir::isa<mlir::BlockArgument>(operand);
            });
            llvm::copy_if(op->getResults(), std::back_inserter(currentSplit.outputs), [](auto result) {
                return llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
                    return mlir::isa<mlir::func::ReturnOp>(userOp);
                });
            });
            return;
        }

        // Sort `UnrealizedConversionCastOp` out onto two kinds: initial & terminational
        bool areInitialCasts = llvm::any_of(mainFunction.getArguments(), [op](auto blockArg) {
            return llvm::any_of(blockArg.getUsers(), [op](auto userOp) {
                return userOp == op;
            });
        });

        // Gather inputs of our single split produced by initial `UnrealizedConversionCastOp` operations
        bool isDebatchedConstant = op->getOperand(0).getDefiningOp<Const::DeclareOp>();
        if (areInitialCasts || (currentSplit.operations.empty() && isDebatchedConstant)) {
            // avoid remembering stale operands as a split input
            if (!op->getUsers().empty()) {
                llvm::copy(op->getResults(), std::back_inserter(currentSplit.inputs));
            }
            return;
        }

        bool areFinalCasts = llvm::all_of(op->getUses(), [](auto& u) {
            return mlir::isa<mlir::func::ReturnOp>(u.getOwner());
        });

        VPUX_THROW_UNLESS(areFinalCasts || isDebatchedConstant,
                          "Only three category of UnrealizedConversionCastOp are supported: {0}, owner: {1}", *op,
                          *op->getParentOp());

        // Gather outputs of our single split to consume by terminational `UnrealizedConversionCastOp` operations
        llvm::copy(op->getOperands(), std::back_inserter(currentSplit.outputs));
    });

    return SmallVector<OutliningInstance>{splits};
}
}  // namespace

namespace vpux {
namespace IE {

FunctionOutlinerBatching::FunctionOutlinerBatching(Logger log): _log(log) {
    _log.setName("function-outliner-batching");
}

SmallVector<OutliningInstance> FunctionOutlinerBatching::getOutliningTargets(mlir::func::FuncOp mainFunction) {
    _log.debug("Searching for outlining targets with a batching split strategy");

    BatchingSplitter splitter(_log);
    const auto outliningInstances = splitter.getOutliningInstances(mainFunction);
    VPUX_THROW_UNLESS(outliningInstances.size() == 1, "Only one outlining instance is expected");
    printOutliningInstances(outliningInstances, _log);
    return outliningInstances;
}

}  // namespace IE
}  // namespace vpux
