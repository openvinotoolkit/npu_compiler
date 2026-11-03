//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

#include <deque>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_DEDUPLICATESWKERNELS
#define GEN_PASS_DEF_DEDUPLICATESWKERNELS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// DuplicateFuncOpEquivalenceInfo
//

// Copied from the upstream MLIR DuplicateFunctionElimination implementation
// (mlir/lib/Dialect/Func/Transforms/DuplicateFunctionElimination.cpp).
//
// Define a notion of function equivalence that allows for reuse. Ignore the
// symbol name for this purpose.
struct DuplicateFuncOpEquivalenceInfo : public llvm::DenseMapInfo<mlir::func::FuncOp> {
    static unsigned getHashValue(const mlir::func::FuncOp cFunc) {
        if (!cFunc) {
            return DenseMapInfo<mlir::func::FuncOp>::getHashValue(cFunc);
        }

        // Aggregate attributes, ignoring the symbol name.
        llvm::hash_code hash = {};
        mlir::func::FuncOp func = const_cast<mlir::func::FuncOp&>(cFunc);
        mlir::StringAttr symNameAttrName = func.getSymNameAttrName();
        for (mlir::NamedAttribute namedAttr : cFunc->getAttrs()) {
            mlir::StringAttr attrName = namedAttr.getName();
            if (attrName == symNameAttrName) {
                continue;
            }
            hash = llvm::hash_combine(hash, namedAttr);
        }

        // Also hash the func body.
        func.getBody().walk([&](mlir::Operation* op) {
            hash = llvm::hash_combine(hash, mlir::OperationEquivalence::computeHash(
                                                    op,
                                                    /*hashOperands=*/mlir::OperationEquivalence::ignoreHashValue,
                                                    /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
                                                    mlir::OperationEquivalence::IgnoreLocations));
        });

        return hash;
    }

    static bool isEqual(mlir::func::FuncOp lhs, mlir::func::FuncOp rhs) {
        if (lhs == rhs) {
            return true;
        }
        if (lhs == getTombstoneKey() || lhs == getEmptyKey() || rhs == getTombstoneKey() || rhs == getEmptyKey()) {
            return false;
        }

        if (lhs.isDeclaration() || rhs.isDeclaration()) {
            return false;
        }

        // Check discardable attributes equivalence
        if (lhs->getDiscardableAttrDictionary() != rhs->getDiscardableAttrDictionary()) {
            return false;
        }

        // Check properties equivalence, ignoring the symbol name.
        // Make a copy, so that we can erase the symbol name and perform the
        // comparison.
        auto pLhs = lhs.getProperties();
        auto pRhs = rhs.getProperties();
        pLhs.sym_name = nullptr;
        pRhs.sym_name = nullptr;
        if (pLhs != pRhs) {
            return false;
        }

        // Compare inner workings.
        return mlir::OperationEquivalence::isRegionEquivalentTo(&lhs.getBody(), &rhs.getBody(),
                                                                mlir::OperationEquivalence::IgnoreLocations);
    }
};

//
// getEnclosingSwKernel
//

// Returns the VPU.SW kernel carrying the given symbol reference.
mlir::func::FuncOp getEnclosingSwKernel(mlir::Operation* user, mlir::ModuleOp swModule) {
    auto func = mlir::dyn_cast<mlir::func::FuncOp>(user);
    if (func == nullptr) {
        func = user->getParentOfType<mlir::func::FuncOp>();
    }

    if (func == nullptr || func->getParentOp() != swModule.getOperation()) {
        return nullptr;
    }

    return func;
}

//
// DeduplicateSwKernelsPass
//

class DeduplicateSwKernelsPass final :
        public vpux::ShaveCodeGen::impl::DeduplicateSwKernelsBase<DeduplicateSwKernelsPass> {
public:
    explicit DeduplicateSwKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void DeduplicateSwKernelsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

    mlir::DenseSet<mlir::func::FuncOp, DuplicateFuncOpEquivalenceInfo> uniqueFuncOps;
    mlir::DenseSet<mlir::Operation*> deadFuncOps;
    std::deque<mlir::func::FuncOp> worklist;
    mlir::DenseSet<mlir::func::FuncOp> queued;

    const auto enqueue = [&](mlir::func::FuncOp func) {
        if (queued.insert(func).second) {
            worklist.push_back(func);
        }
    };

    for (auto func : swModule.getOps<mlir::func::FuncOp>()) {
        if (!func.isExternal()) {
            enqueue(func);
        }
    }

    // Built once over the whole module: the call sites (VPU::GenericSwLayerOp) live outside of
    // VPU.SW, and SymbolUserMap transfers the user list of a symbol on every replacement, so it
    // stays accurate for the whole run.
    mlir::SymbolTableCollection symbolTables;
    mlir::SymbolUserMap userMap(symbolTables, moduleOp);

    while (!worklist.empty()) {
        auto func = worklist.front();
        worklist.pop_front();
        queued.erase(func);
        if (deadFuncOps.contains(func.getOperation())) {
            continue;
        }

        const auto [it, inserted] = uniqueFuncOps.insert(func);
        if (inserted) {
            continue;
        }
        auto representative = *it;

        // getUsers hands out a reference to the user list that replaceAllUsesWith moves away,
        // so the users have to be copied before the replacement.
        const SmallVector<mlir::Operation*> users(userMap.getUsers(func.getOperation()));

        // Re-enqueue changed kernel representatives that get changed by the RAUW.
        // The uniqueFuncOps entry needs to be removed as well to enable re-hashing.
        // In practice kernels don't reference each other so we do at most two passes
        // over all kernels at most.
        for (auto* user : users) {
            auto userFunc = getEnclosingSwKernel(user, swModule);
            if (userFunc == nullptr || deadFuncOps.contains(userFunc.getOperation())) {
                continue;
            }

            const auto uniqueIt = uniqueFuncOps.find(userFunc);
            if (uniqueIt != uniqueFuncOps.end() && (*uniqueIt).getOperation() == userFunc.getOperation()) {
                uniqueFuncOps.erase(uniqueIt);
                enqueue(userFunc);
            }
        }

        _log.trace("Replacing duplicated kernel '{0}' with '{1}'", func.getSymName(), representative.getSymName());

        userMap.replaceAllUsesWith(func.getOperation(), representative.getSymNameAttr());
        deadFuncOps.insert(func.getOperation());
    }

    // Dead functions need to be erased at the end in order to not invalidate userMap.
    for (auto* func : deadFuncOps) {
        func->erase();
    }
}

}  // namespace

//
// createDeduplicateSwKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createDeduplicateSwKernelsPass(Logger log) {
    return std::make_unique<DeduplicateSwKernelsPass>(log);
}
