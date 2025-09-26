//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include <vpux/compiler/dialect/core/transforms/passes.hpp>
#include <vpux/compiler/utils/func_dialect.hpp>
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <queue>

namespace vpux::Core {
#define GEN_PASS_DECL_PACKNESTEDMODULES
#define GEN_PASS_DEF_PACKNESTEDMODULES
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

// Make this pass deterministic by
// 1. Sorting the operations in every cluster by the order in which the operations appear in the block
// 2. Sorting the clusters by the order in which their first operations appear in the blocks. Since
//    clusters don't share any operations, this is well defined.
//
// To achieve this we use std::map and std::set instead of the unordered versions and order the ops
// using isBeforeInBlock.

// Note: Do not use FuncOpComparator in another context as it will crash or cause UB if 2 ops appear in different
// blocks.
struct FuncOpComparator {
    bool operator()(const mlir::func::FuncOp& a, const mlir::func::FuncOp& b) const {
        return a->isBeforeInBlock(b);
    }
};

using CallGraph = std::map<mlir::func::FuncOp, SmallVector<mlir::func::FuncOp>, FuncOpComparator>;
using FuncOpSet = std::set<mlir::func::FuncOp, FuncOpComparator>;

bool belongsIntoTopCluster(mlir::func::FuncOp caller, mlir::func::FuncOp mainFuncOp) {
    return caller == mainFuncOp || caller->hasAttrOfType<mlir::UnitAttr>("do_not_nest");
}

/// Returns true if there exists a FuncOp in from that has an edge to any FuncOp in to according to callGraph.
bool hasSpanningEdges(const FuncOpSet& from, const FuncOpSet& to, const CallGraph& callGraph) {
    auto result = false;
    for (const auto& u : from) {
        if (const auto it = callGraph.find(u); it != callGraph.end()) {
            for (const auto& v : it->second) {
                result |= (to.find(v) != to.end());
            }
        }
    }
    return result;
}

/// Merges the set src into dst.
void mergeInto(FuncOpSet& dst, const FuncOpSet& src) {
    dst.insert(src.begin(), src.end());
}

std::string getSubModuleName(size_t index) {
    return formatv("Module{0}", index);
}

mlir::Operation* getFirstClusterInsertionPoint(const FuncOpSet& topCluster) {
    assert(!topCluster.empty());
    return *topCluster.begin();
}

//
// PackNestedModules
//

struct Clusters {
    FuncOpSet topCluster;
    SmallVector<FuncOpSet> nestedClusters;
};

class PackNestedModules final : public Core::impl::PackNestedModulesBase<PackNestedModules> {
public:
    explicit PackNestedModules(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    CallGraph collectCallGraph(mlir::func::FuncOp root);
    SmallVector<FuncOpSet> createClusters(const CallGraph& callGraph, mlir::func::FuncOp mainFuncOp);
    Clusters collectClusters(mlir::func::FuncOp mainFuncOp);
    void patchCallSites(const Clusters& clusters);
    mlir::ModuleOp nestCluster(mlir::OpBuilder& builder, FuncOpSet&& cluster, size_t clusterIndex);

    void safeRunOnModule() final;
};

/// Returns a map of FuncOp -> Vector<FuncOp> that models the call graph using cycle-aware BFS.
CallGraph PackNestedModules::collectCallGraph(mlir::func::FuncOp root) {
    auto topModuleOp = getOperation();

    CallGraph callGraph;

    mlir::DenseSet<mlir::func::FuncOp> visited;

    std::queue<mlir::func::FuncOp> workList;
    workList.push(root);

    while (!workList.empty()) {
        auto funcOp = workList.front();
        workList.pop();

        if (visited.contains(funcOp)) {
            continue;
        }

        auto& calleeVec = callGraph[funcOp];

        funcOp.walk([&](mlir::func::CallOp callOp) {
            const auto calleeFuncOp = topModuleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            assert(calleeFuncOp != nullptr);
            calleeVec.push_back(calleeFuncOp);
            workList.push(calleeFuncOp);
        });

        visited.insert(funcOp);
    }

    return callGraph;
}

/// Returns a vector of sets of FuncOps.
///
///     Assume we have the following call graph:
///
///         main* -> [f, h]
///
///         f -> g*
///
///         h -> i
///
///         i -> j
///         j -> i
///
///     Functions annotated with (*) will be put into the "top" cluster (i.e. they will not be nested into submodules),
///     either because that function is the entry point or it contains the attribute "do_not_nest".
/// 1)
///     The result will be initialized as follows. All FuncOps that belong into the top cluster are put into the first
///     set of the vector first. All other functions are then put into their own respective singleton set. In this
///     example:
///         [{main*, g*}, {f}, {h}, {i}, {j}]
/// 2)
///     Then all singleton sets are merged into the top cluster set when they have an edge into the top cluster set.
///     In our example, because of (f -> g*), we would get {main*, g*, f}. To account for transitive edges, this step
///     is done until a full iteration is performed without any merges.
/// 3)
///     Then the remaining clusters (except for the top cluster) are merged pairwise if they have edges from on set to
///     the other or vice versa. This is done until no more clusters can be merged In this example:
///         [{main*, g*, f}, {h}, {i}, {j}]     {i} and {j}    are merged because of i -> j and j -> i
///         [{main*, g*, f}, {h}, {i, j}]       {h} and {i, j} are merged because of h -> i
///         [{main*, g*, f}, {h, i, j}]         Nothing to merge
/// 4)
///     Done
///
SmallVector<FuncOpSet> PackNestedModules::createClusters(const CallGraph& callGraph, mlir::func::FuncOp mainFuncOp) {
    SmallVector<FuncOpSet> clusters{{}};

    for (auto [caller, callees] : callGraph) {
        if (belongsIntoTopCluster(caller, mainFuncOp)) {
            clusters[0].insert(caller);
        } else {
            clusters.push_back({});
            clusters.back().insert(caller);
        }
    }

    // merge singletons into top cluster
    auto mergeSingletonsIntoTopCluster = [&]() -> bool {
        for (auto src = clusters.begin() + 1; src < clusters.end(); src++) {
            auto dst = clusters.begin();
            const auto doMerge = hasSpanningEdges(*src, *dst, callGraph);
            if (doMerge) {
                mergeInto(*dst, *src);
                clusters.erase(src);
                return true;
            }
        }
        return false;
    };
    while (mergeSingletonsIntoTopCluster()) {
    }

    // merge pairwise until fixed point is reached (ignore top cluster)
    auto mergePairwise = [&]() -> bool {
        for (auto dst = std::next(clusters.begin()); dst < clusters.end(); dst++) {
            for (auto src = std::next(dst); src < clusters.end(); src++) {
                const auto doMerge = hasSpanningEdges(*dst, *src, callGraph) || hasSpanningEdges(*src, *dst, callGraph);
                if (doMerge) {
                    mergeInto(*dst, *src);
                    clusters.erase(src);
                    return true;
                }
            }
        }
        return false;
    };
    while (mergePairwise()) {
    }

    return clusters;
}

/// Returns a vector of vector of FuncOps. Every vector of FuncOps represents one set of operations that will
/// be nested inside a submodule. The resulting vectors will be sorted to ensure deterministic IR.
Clusters PackNestedModules::collectClusters(mlir::func::FuncOp mainFuncOp) {
    const auto callGraph = collectCallGraph(mainFuncOp);
    const auto clustersSet = createClusters(callGraph, mainFuncOp);

    return {clustersSet.front(), SmallVector<FuncOpSet>(std::next(clustersSet.begin()), clustersSet.end())};
}

/// Replaces all mlir::CallOps in the top cluster with Core::NestedCallOps that point to the newly nested functions.
void PackNestedModules::patchCallSites(const Clusters& clusters) {
    auto topModuleOp = getOperation();

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);

    DenseMap<mlir::func::FuncOp, size_t> nestedClusterMap;
    for (const auto& [index, nestedCluster] : clusters.nestedClusters | indexed) {
        for (auto funcOp : nestedCluster) {
            nestedClusterMap[funcOp] = index;
        }
    }

    for (auto funcOp : clusters.topCluster) {
        funcOp.walk([&](mlir::func::CallOp callOp) {
            const auto callee = topModuleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            assert(callee != nullptr);
            auto it = nestedClusterMap.find(callee);
            // callOp does not point to a future nested function
            if (it == nestedClusterMap.end()) {
                return;
            }

            const auto clusterIndex = it->getSecond();
            const auto moduleName = getSubModuleName(clusterIndex);

            const auto nestedSymbolAttr = mlir::SymbolRefAttr::get(
                    &getContext(), moduleName, {mlir::FlatSymbolRefAttr::get(&getContext(), callOp.getCallee())});

            builder.setInsertionPoint(callOp);
            const auto coreCallOp = builder.create<Core::NestedCallOp>(callOp.getLoc(), nestedSymbolAttr,
                                                                       callOp->getResultTypes(), callOp.getOperands());

            callOp->replaceAllUsesWith(coreCallOp);
            callOp->erase();
        });
    }
}

/// Creates a new ModuleOp at the builder's current location and moves all FuncOps in the cluster into it.
mlir::ModuleOp PackNestedModules::nestCluster(mlir::OpBuilder& builder, FuncOpSet&& cluster, size_t clusterIndex) {
    const auto moduleName = getSubModuleName(clusterIndex);

    auto topModuleOp = getOperation();
    auto subModuleOp = builder.create<mlir::ModuleOp>(appendLoc(topModuleOp.getLoc(), "_{0}", moduleName), moduleName);
    auto& targetOps = subModuleOp.getBody()->getOperations();

    for (auto funcOp : cluster) {
        targetOps.splice(targetOps.end(), funcOp->getBlock()->getOperations(), funcOp->getIterator());
    }

    return subModuleOp;
}

void PackNestedModules::safeRunOnModule() {
    const auto topModuleOp = getOperation();
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(topModuleOp, netInfo, mainFuncOp);

    auto clusters = collectClusters(mainFuncOp);
    patchCallSites(clusters);

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    builder.setInsertionPoint(getFirstClusterInsertionPoint(clusters.topCluster));
    for (auto&& [index, nestedCluster] : clusters.nestedClusters | indexed) {
        // move because the FuncOps in cluster will be invalidated
        auto subModuleOp = nestCluster(builder, std::move(nestedCluster), index);
        // nestCluster invalidates IR so we have to update the insertion point
        builder.setInsertionPointAfter(subModuleOp);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createPackNestedModulesPass(Logger log) {
    return std::make_unique<PackNestedModules>(log);
}
