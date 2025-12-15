//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;
/**
 * @brief For large-scale models simplified graph optimization is performed due to N^2 memory
 * complexity required by the algorithm for fast and full optimization of the dependencies graph.
 * The threshold is set so as to enable simplified (partial) optimization for very large models.
 *
 * For the number of operations above the threshold, the optimization of the graph is partial with
 * benefit of smaller memory usage. At the optimization threshold the largest allocated structure
 * size will be of the order 800MB.
 */
#define SIMPLIFIED_OPTIMIZATION_OP_COUNT_THRESHOLD 10000

//
// Constructor
//

vpux::AsyncDepsInfo::AsyncDepsInfo(mlir::func::FuncOp func)
        : _log(Logger::global().nest("async-deps-info", 0)),
          _indexAttrName(mlir::StringAttr::get(func->getContext(), "async-deps-index")) {
    buildDepsMap(func);
}

//
// setIndex/getIndex
//

void vpux::AsyncDepsInfo::setIndex(mlir::async::ExecuteOp execOp, uint64_t index) {
    execOp->setAttr(_indexAttrName, getIntAttr(execOp.getContext(), index));
}

SmallVector<size_t> vpux::AsyncDepsInfo::getDepsVec(const llvm::DenseSet<size_t>& deps) const {
    SmallVector<size_t> vec(deps.begin(), deps.end());
    /* The order of dependencies may impact order of prefetched DMA.
     * Experiments show that arbitrary order of these dependencies may impact
     * inference accuracy (E102917) hence sorting.
     */
    llvm::sort(vec.begin(), vec.end());
    return vec;
}

uint32_t vpux::AsyncDepsInfo::getIndex(mlir::async::ExecuteOp execOp) const {
    const auto attr = execOp->getAttrOfType<mlir::IntegerAttr>(_indexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Attribute '{0}' was not set for '{1}' operation at '{2}'", _indexAttrName,
                      execOp->getName(), execOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

mlir::async::ExecuteOp vpux::AsyncDepsInfo::getExecuteOpAtIndex(size_t opIdx) const {
    VPUX_THROW_WHEN(opIdx >= _execOpCount, "Invalid index '{0}' for _allExecOps", opIdx);
    return _allExecOps[opIdx];
}

size_t vpux::AsyncDepsInfo::getExecOpCount() const {
    return _execOpCount;
}

//
// buildDepsMap
//

void vpux::AsyncDepsInfo::buildDepsMap(mlir::func::FuncOp func) {
    _log.trace("Collect initial dependencies maps");
    _log = _log.nest();

    _allExecOps = to_small_vector(func.getOps<mlir::async::ExecuteOp>());
    for (const auto& p : _allExecOps | indexed) {
        setIndex(p.value(), p.index());
    }

    _execOpCount = _allExecOps.size();
    _depsMap.resize(_execOpCount);

    for (auto& op : func.getOps()) {
        if (auto execOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            addExecOp(execOp);
        } else if (auto waitOp = mlir::dyn_cast<mlir::async::AwaitOp>(op)) {
            _log.trace("Found 'async.await' Operation at '{0}'", op.getLoc());

            if (waitOp.getResult() != nullptr) {
                for (auto* user : waitOp.getResult().getUsers()) {
                    VPUX_THROW_WHEN(
                            user->getParentOfType<mlir::async::ExecuteOp>() != nullptr,
                            "Got 'async.await' Operation at '{0}', which has users inside 'async.execute' region",
                            op.getLoc());
                }
            }
        }
    }
    _log = _log.unnest();
}

void vpux::AsyncDepsInfo::addExecOp(mlir::async::ExecuteOp execOp) {
    _log.trace("Found 'async.execute' Operation at '{0}'", execOp->getLoc());
    _log = _log.nest();

    const auto execInd = getIndex(execOp);

    for (auto arg : execOp->getOperands()) {
        auto argExecOp = mlir::dyn_cast<mlir::async::ExecuteOp>(arg.getDefiningOp());
        VPUX_THROW_UNLESS(argExecOp != nullptr,
                          "'async.execute' at '{0}' has operand '{1}' produced by unsupported Operation",
                          execOp->getLoc(), arg);

        _log.trace("It has a dependency from other 'async.execute' Operation at '{0}'", argExecOp->getLoc());

        const auto argExecInd = getIndex(argExecOp);
        _depsMap[execInd].insert(argExecInd);
    }

    _log = _log.unnest();
}

//
// addDependency
//

void vpux::AsyncDepsInfo::addDependency(mlir::async::ExecuteOp from, mlir::async::ExecuteOp to) {
    const auto fromInd = getIndex(from);
    const auto toInd = getIndex(to);
    _depsMap[toInd].insert(fromInd);
    if (!_consumerMap.empty()) {
        // also update consumer map if build
        _consumerMap[fromInd].insert(toInd);
    }
}

//
// optimizeDepsMap
//

void vpux::AsyncDepsInfo::optimizeDepsMap() {
    //
    // A -> B -> C
    //
    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    //
    // Algorithm is divided into two steps:
    //  step 1 - transitive closure
    //  step 2 - transitive reduction
    // Worst case complexity is O(N^3) but expected time will be proportional to ~N*E*k, where k represents the size of
    // _curDeps, E denotes the size of _depsMapClosure, and N is the number of operations. So in case of sparse graphs
    // which is a usual case for NN models expected time shouldn't be as bad as N^3

    // Step 1: Transitive closure
    // Update all dependencies in a new depsMapClosure to represent transitive closure
    // of initial dependencies graph. For each node starting from beginning it will go
    // though its dependencies, take their dependencies and update its own.
    // In this step even if initial graph was sparse, after it, it will be dense for the case of
    // full optimization. Partial optimization does not build cumulatively the full history of all tasks ancestors
    // but limits the scope to second order dependencies (E102917).
    auto depsMapClosure = _depsMap;
    bool performSimplifiedOptimization = _depsMap.size() > SIMPLIFIED_OPTIMIZATION_OP_COUNT_THRESHOLD;
    SmallVector<llvm::DenseSet<size_t>>& refDeps = performSimplifiedOptimization ? _depsMap : depsMapClosure;
    unsigned i = 0;
    for (auto& curDeps : depsMapClosure) {
        if (i == depsMapClosure.size() - 1) {
            break;  // during optimization we don't use the last entry because an operation cannot depend on itself.
        }
        for (auto curDepInd : llvm::DenseSet<size_t>(curDeps)) {
            const auto& depOfDeps = refDeps[curDepInd];
            curDeps.insert(depOfDeps.begin(), depOfDeps.end());
        }
        i++;
    }

    // Step 2: Transitive reduction
    // Remove all unnecessary edges.
    // Go through each node starting from the end and remove dependencies if those dependencies
    // that are already represented in its dependant nodes
    for (int depInd = (static_cast<int>(_depsMap.size()) - 1); depInd >= 0; depInd--) {
        auto& curDeps = _depsMap[depInd];

        // If node does not have any dependency or it has only one dependency then skip
        if (curDeps.size() <= 1) {
            continue;
        }

        for (auto curDepInd : llvm::DenseSet<size_t>(curDeps)) {
            const auto& depOfDeps = depsMapClosure[curDepInd];
            // In the context of neural network (NN) models, the size of the _depsMapClosure (E) significantly surpasses
            // that of _curDeps (k). By strategically constraining the traversal of _curDeps to a maximum of k edges per
            // operation, we have achieved a refined computational complexity of ~N×E×k.
            for (auto dep : llvm::DenseSet<size_t>(curDeps)) {
                if (depOfDeps.count(dep) == 1) {
                    curDeps.erase(dep);
                }
            }
        }
    }

    if (!_consumerMap.empty()) {
        // re-build consumer map using new deps map if build
        _consumerMap.clear();
        buildConsMap();
    }
}

//
// buildConsMap
//

void vpux::AsyncDepsInfo::buildConsMap() {
    _consumerMap.resize(_depsMap.size());

    for (size_t idx = 0; idx < _depsMap.size(); idx++) {
        for (auto bit : _depsMap[idx]) {
            _consumerMap[checked_cast<size_t>(bit)].insert(checked_cast<uint32_t>(idx));
        }
    }
}

//
// updateTokenDependencies
//

void vpux::AsyncDepsInfo::updateTokenDependencies() {
    _log.trace("Add explicit '!async.token' based dependencies between 'async.execute' operations");
    _log = _log.nest();

    for (auto* execOpIt = _allExecOps.begin(); execOpIt != _allExecOps.begin() + _execOpCount; ++execOpIt) {
        _log.trace("Process 'async.execute' Operation at '{0}'", execOpIt->getLoc());

        const auto execInd = getIndex(*execOpIt);
        const auto execDeps = getDepsVec(_depsMap[execInd]);

        SmallVector<mlir::Value> depsVec;
        for (auto depInd : execDeps) {
            depsVec.push_back(_allExecOps[depInd].getToken());
        }

        _log.nest().trace("Use the following explicit dependencies : {0}", depsVec);
        execOpIt->getDependenciesMutable().assign(ArrayRef(depsVec));
    }

    _log = _log.unnest();
}

size_t vpux::AsyncDepsInfo::insertNewExecOpToDepsMap(mlir::async::ExecuteOp execOp) {
    auto dataStructSize = _allExecOps.size();
    VPUX_THROW_WHEN(_execOpCount > dataStructSize, "Invalid execOp count '{0}'", _execOpCount);

    if (_execOpCount == dataStructSize) {
        preAllocateForNewOps(1);
    }

    _allExecOps[_execOpCount] = execOp;
    setIndex(execOp, _execOpCount);
    addExecOp(execOp);

    return _execOpCount++;
}

/* Adds more space to internal structures, it only resizes all internal structures in advance
   to avoid loss on single operation insertion.
   Use only if you know in advance how many insetions are nesessary. */
void vpux::AsyncDepsInfo::preAllocateForNewOps(size_t numOfNewOps) {
    auto newSize = _allExecOps.size() + numOfNewOps;
    _allExecOps.resize(newSize);
    _depsMap.resize(newSize);
    _consumerMap.resize(newSize);
}

const llvm::SmallVector<size_t> vpux::AsyncDepsInfo::getOpDeps(size_t opIdx) const {
    VPUX_THROW_WHEN(opIdx >= _execOpCount, "Invalid index '{0}' for _depsMap", opIdx);
    return getDepsVec(_depsMap[opIdx]);
}

const llvm::SmallVector<size_t> vpux::AsyncDepsInfo::getConsumerOps(size_t opIdx) const {
    VPUX_THROW_WHEN(_consumerMap.empty(), "Consumer map was not build");
    VPUX_THROW_WHEN(opIdx >= _execOpCount, "Invalid index '{0}' for _consumerMap", opIdx);
    return getDepsVec(_consumerMap[opIdx]);
}

std::unordered_map<size_t, size_t> vpux::AsyncDepsInfo::calculateOpInDegreeTable() const {
    std::unordered_map<size_t, size_t> opInDegree;
    for (size_t i = 0; i < _execOpCount; ++i) {
        opInDegree[i] = static_cast<size_t>(_depsMap[i].size());
    }
    return opInDegree;
}

std::unordered_map<size_t, size_t> vpux::AsyncDepsInfo::calculateOpOutDegreeTable() const {
    VPUX_THROW_WHEN(_consumerMap.empty(), "Consumer map was not build");
    std::unordered_map<size_t, size_t> opOutDegree;
    for (size_t i = 0; i < _execOpCount; ++i) {
        opOutDegree[i] = static_cast<size_t>(_consumerMap[i].size());
    }
    return opOutDegree;
}
