//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/analysis.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <deque>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace vpux;

/*
Simple Chain - represents linear dataflow sequences:
___________           ___________           ___________           ___________           ___________
|         |           |         |           |         |           |         |           |         |
|   OP1   |  -------  |   OP2   |  -------  |   OP3   |  -------  |   OP4   |  -------  |   OP5   |
|_________|           |_________|           |_________|           |_________|           |_________|

Forking Chain - represents chains in which dataflow forks at a particular point:

                                             ___________           ___________
                                             |         |           |         |
                                      |----- |   OP3   |  -------  |   OP4   |
                                      |      |_________|           |_________|
___________           ___________     |      ___________           ___________           ___________
|         |           |         |     |      |         |           |         |           |         |
|   OP1   |  -------  |   OP2   |  --------- |   OP5   |  -------  |   OP6   |  -------  |   OP7   |
|_________|           |_________|     |      |_________|           |_________|           |_________|
                                      |      ___________
                                      |      |         |
                                      |----- |   OP8   |
                                             |_________|


Analysis objective is to identify such situations and split into multiple simple (linear) chains:

     ___________           ___________
     |         |           |         |
C1:  |   OP1   |  -------  |   OP2   |
     |_________|           |_________|

     ___________           ___________
     |         |           |         |
C2:  |   OP3   |  -------  |   OP4   |
     |_________|           |_________|

     ___________           ___________           ___________
     |         |           |         |           |         |
C3:  |   OP5   |  -------  |   OP6   |  -------  |   OP7   |
     |_________|           |_________|           |_________|

     ___________
     |         |
C4:  |   OP8   |
     |_________|

//////
Joining Chain - represents situations in which multiple parallel dataflows join in an op

___________           ___________
|         |           |         |
|   OP1   |  -------  |   OP2   |  -----|
|_________|           |_________|       |
                                        |
___________           ___________       |        ___________           ___________
|         |           |         |       |        |         |           |         |
|   OP3   |  -------  |   OP4   |  -----|-----   |   OP5   |  -------  |   OP6   |
|_________|           |_________|       |        |_________|           |_________|
                                        |
___________           ___________       |
|         |           |         |       |
|   OP7   |  -------  |   OP8   |  -----|
|_________|           |_________|

Analysis objective is to identify such situations and split into multiple simple (linear) chains:

     ___________           ___________
     |         |           |         |
C1:  |   OP1   |  -------  |   OP2   |
     |_________|           |_________|

     ___________           ___________
     |         |           |         |
C2:  |   OP3   |  -------  |   OP4   |
     |_________|           |_________|

     ___________           ___________
     |         |           |         |
C1:  |   OP5   |  -------  |   OP6   |
     |_________|           |_________|

     ___________           ___________
     |         |           |         |
C2:  |   OP7   |  -------  |   OP8   |
     |_________|           |_________|

TODO: E#160644 Add heuristics & logic to support the longest possible chain
e.g.: (for forking chain) OP1 -> OP2 -> OP5 -> OP6 -> OP7 with additional results to accomodate for C2 & C4 inputs

*/

namespace {

// Out of line to provide ease of access for potential validity requirement updates
bool isValidChainNode(mlir::Operation* op) {
    if (auto scgOp = mlir::dyn_cast<IE::ShaveCodeGenSupportedOpInterface>(op)) {
        return scgOp.shouldJITCompile();
    }
    return false;
}

void extendChain(mlir::Operation* head, std::vector<mlir::Operation*>& currChain,
                 std::deque<mlir::Operation*>& forkHeads) {
    // Only valid chain heads are expected
    // So we can safely push it to the chain
    currChain.push_back(head);

    auto currOp = head;

    // Returns the counts of users which
    //      pair.first -> are potential chain nodes, as in they pass the validity check
    //      pair.second -> no condition, as in a count of total number of users
    auto getUserCounts = [](mlir::Operation* op) -> std::pair<size_t, size_t> {
        size_t potentialNodeCount = 0;
        size_t genericUserCount = 0;
        llvm::for_each(op->getUsers(), [&](mlir::Operation* user) {
            genericUserCount++;
            if (isValidChainNode(user)) {
                potentialNodeCount++;
            }
        });
        return {potentialNodeCount, genericUserCount};
    };

    auto [potentialNodeCount, genericUserCount] = getUserCounts(currOp);
    while (potentialNodeCount != 0) {
        if (genericUserCount == 1) {
            auto prevOp = currOp;

            // Safe as in this branch both scgUserCount & genericUserCount == 1
            currOp = *(currOp->getUsers().begin());

            bool hasExternalDeps = llvm::any_of(currOp->getOperands(), [&prevOp](mlir::Value val) {
                auto definingOp = val.getDefiningOp();
                if (definingOp) {
                    return definingOp != prevOp;
                }
                return false;
            });

            // If op has any external dependencies, break the chain at this point
            // and further flag node as a possible chain head
            if (hasExternalDeps) {
                forkHeads.push_back(currOp);
                break;
            } else {
                currChain.push_back(currOp);
                std::tie(potentialNodeCount, genericUserCount) = getUserCounts(currOp);
            }
        } else {
            for (auto user : currOp->getUsers()) {
                if (isValidChainNode(user)) {
                    forkHeads.push_back(user);
                }
            }
            break;
        }
    }
}
}  // namespace

ShaveCodeGen::FusionChainAnalysis::FusionChainAnalysis(mlir::Operation* op) {
    auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);
    VPUX_THROW_UNLESS(funcOp, "Fusion Chain Analysis is expected to be run on funcOps");
    auto codeGenOps = funcOp.getOps<IE::ShaveCodeGenSupportedOpInterface>();

    std::deque<mlir::Operation*> forkHeads;
    std::unordered_set<mlir::Operation*> processedHeads;

    // Attempt to find chain heads and start extending the chains starting with them
    for (auto codeGenOp : codeGenOps) {
        // For a node to be considered a chain head, it needs to not have any SCG produced operands
        auto isProducedByCodeGenOp = [](mlir::Value val) {
            auto definingOp = val.getDefiningOp();
            return definingOp && isValidChainNode(definingOp);
        };
        auto isNotChainHead = llvm::any_of(codeGenOp->getOperands(), isProducedByCodeGenOp);

        if (!isValidChainNode(codeGenOp) || isNotChainHead) {
            continue;
        }

        std::vector<mlir::Operation*> newChain;
        extendChain(codeGenOp, newChain, forkHeads);
        processedHeads.insert(newChain[0]);
        _computeOpChains.push_back(std::move(newChain));
    }

    while (!forkHeads.empty()) {
        auto forkHead = forkHeads[0];
        forkHeads.pop_front();
        if (processedHeads.find(forkHead) != processedHeads.end()) {
            continue;
        }
        std::vector<mlir::Operation*> newChain;
        extendChain(forkHead, newChain, forkHeads);
        processedHeads.insert(newChain[0]);
        _computeOpChains.push_back(std::move(newChain));
    }

    setState(State::ComputeOpChains);
}

std::vector<std::vector<mlir::Operation*>> ShaveCodeGen::FusionChainAnalysis::getComputeOpChains() const {
    assert(_state == State::ComputeOpChains);
    return _computeOpChains;
}

std::vector<std::vector<mlir::Operation*>> ShaveCodeGen::FusionChainAnalysis::getCodeGenCapsulesChains() const {
    assert(_state == State::CodeGenCapsuleChains);
    return _codeGenCapsulesChains;
}

void ShaveCodeGen::FusionChainAnalysis::appendCodeGenCapsuleChain(std::vector<mlir::Operation*>& newChain) {
    _codeGenCapsulesChains.push_back(newChain);
}

void ShaveCodeGen::FusionChainAnalysis::invalidate() {
    _state = State::Invalidated;
}

void ShaveCodeGen::FusionChainAnalysis::setState(State newState) {
    _state = newState;
}

ShaveCodeGen::FusionChainAnalysis::State ShaveCodeGen::FusionChainAnalysis::getState() const {
    return _state;
}

bool ShaveCodeGen::FusionChainAnalysis::isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&) {
    return _state == State::Invalidated;
}
