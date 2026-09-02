//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/RegionUtils.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

#include <deque>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_WRAPINKERNELREGION
#define GEN_PASS_DEF_WRAPINKERNELREGION
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// WrapInKernelRegionPass
//

class WrapInKernelRegionPass final : public ShaveCodeGen::impl::WrapInKernelRegionBase<WrapInKernelRegionPass> {
public:
    explicit WrapInKernelRegionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void runOnCapsule(IE::CodeGenCapsuleOp capsule);
    void safeRunOnFunc() final;
};

// Returns true when all results of an op are non-shaped (scalar) types such as
// index or integer/float types. Vacuously false for zero-result ops.
static bool producesOnlyScalars(mlir::Operation* op) {
    return !op->getResults().empty() && llvm::all_of(op->getResults(), [](mlir::Value v) {
        return !mlir::isa<mlir::ShapedType>(v.getType());
    });
}

// Returns true if an op captured from outside the kernel region should be
// cloned into the body rather than passed as an explicit block argument.
//
// Cloned ops:
//   - Constant-like ops: pure, zero-cost to duplicate.
//   - Shave::LoopTripCountOp: part of the tiling descriptor; belongs inside the
//     kernel region together with the scf.for it bounds.
//   - tensor.empty ops: pure allocation placeholders with no data dependency.
//   - Any op whose results are all scalar
static bool shouldCloneOpIntoKernelRegion(mlir::Operation* op) {
    if (mlir::isa<mlir::tensor::DimOp>(op)) {
        return false;
    }
    if (op->hasTrait<mlir::OpTrait::ConstantLike>() || mlir::isa<Shave::LoopTripCountOp, mlir::tensor::EmptyOp>(op)) {
        return true;
    }
    if (producesOnlyScalars(op)) {
        assert(mlir::isMemoryEffectFree(op) && "unexpected side-effectful scalar op");
        return true;
    }
    return false;
}

// Returns true for ops that should be skipped during root candidate selection.
// These are either already-wrapped regions, the capsule terminator, or ops that
// will be pulled in by the isolation step of wrapSubgraphInKernelRegion.
static bool isNotARootCandidate(mlir::Operation* op) {
    return mlir::isa<IE::CGCYieldOp, Shave::KernelRegionOp>(op) || shouldCloneOpIntoKernelRegion(op) ||
           producesOnlyScalars(op);
}

// Builds the subgraph reachable backward from `root` by including producers
// whose entire set of block-level users is already in the subgraph. The walk
// stops at:
//   - block arguments (external inputs),
//   - Shave::KernelRegionOp boundaries (opaque; already-wrapped subgraphs), and
//   - clonable or scalar-producing ops (constants, LoopTripCountOp, tensor.dim,
//     tensor.empty, and any pure scalar op -- handled by the isolation step inside
//     wrapSubgraphInKernelRegion).
//
// Ops are returned in IR order (topological order for a single-block DAG).
static llvm::SmallVector<mlir::Operation*> collectSubgraphFromRoot(mlir::Operation* root) {
    llvm::DenseSet<mlir::Operation*> inSubgraph;
    llvm::SmallVector<mlir::Operation*> subgraph;

    auto addToSubgraph = [&](mlir::Operation* op) {
        inSubgraph.insert(op);
        subgraph.push_back(op);
    };

    addToSubgraph(root);

    auto* rootBlock = root->getBlock();
    for (auto it = root->getIterator(); it != rootBlock->begin();) {
        auto* op = &*--it;
        if (mlir::isa<Shave::KernelRegionOp>(op) || shouldCloneOpIntoKernelRegion(op) || producesOnlyScalars(op)) {
            continue;
        }

        if (op->use_empty() || !llvm::all_of(op->getUsers(), [&](mlir::Operation* user) {
                return inSubgraph.count(user) > 0;
            })) {
            continue;
        }
        addToSubgraph(op);
    }

    // Reverse into IR order: the backwards walk appended ops in reverse IR order.
    std::reverse(subgraph.begin(), subgraph.end());
    return subgraph;
}

// Wraps a subgraph of operations (given in topological order) with a
// Shave::KernelRegionOp. The op is inserted immediately before the last op of
// the subgraph in IR order. All values that the subgraph uses from outside are
// therefore already defined at the insertion point, so the KernelRegionOp
// operands dominate the op itself.
//
// It is the caller's responsibility to ensure that:
//   - all subgraph ops live in the same block, and
//   - no op outside the subgraph that is defined after the last subgraph op (in
//     IR order) is used by consumers of the subgraph outputs, so that the
//     KernelRegionOp results correctly replace those uses without creating
//     dominance violations.
//
// Output values -- results of subgraph ops used outside the subgraph -- become
// the KernelRegionOp results and are yielded by a KernelRegion.Yield terminator.
//
// External operands are handled in two ways:
//   - ops for which shouldCloneOpIntoKernelRegion returns true are cloned into
//     the body with their own operands resolved transitively.
//   - tensor.dim ops are cloned when the source tensor is a graph input or was
//     produced inside the subgraph; otherwise they become block arguments.
//   - everything else becomes a block argument / KernelRegionOp input.
static Shave::KernelRegionOp wrapSubgraphInKernelRegion(mlir::IRRewriter& rewriter,
                                                        llvm::ArrayRef<mlir::Operation*> subgraph) {
    assert(!subgraph.empty() && "subgraph must be non-empty");

    llvm::DenseSet<mlir::Operation*> subgraphSet(subgraph.begin(), subgraph.end());

    // Determine output values: results of subgraph ops with at least one use outside
    // the subgraph.
    llvm::SetVector<mlir::Value> outputValues;
    for (auto* op : subgraph) {
        for (auto result : op->getResults()) {
            if (llvm::any_of(result.getUsers(), [&](mlir::Operation* user) {
                    return !subgraphSet.contains(user);
                })) {
                outputValues.insert(result);
            }
        }
    }

    // Create the KernelRegionOp before the last subgraph op in IR order.
    // This guarantees that every external value the subgraph consumes is already
    // defined at the insertion point, keeping KernelRegionOp operands valid.
    auto* lastInIROrder = subgraph.front();
    for (auto* op : subgraph) {
        if (lastInIROrder->isBeforeInBlock(op)) {
            lastInIROrder = op;
        }
    }
    rewriter.setInsertionPoint(lastInIROrder);
    auto outputTypes = llvm::to_vector(llvm::map_range(outputValues, [](mlir::Value v) {
        return v.getType();
    }));
    auto kernelOp =
            rewriter.create<Shave::KernelRegionOp>(lastInIROrder->getLoc(), outputTypes, /*inputs=*/mlir::ValueRange{});
    kernelOp.getBody().emplaceBlock();
    auto* bodyBlock = &kernelOp.getBody().front();

    // Replace uses of output values that are outside the subgraph with the
    // corresponding KernelRegionOp results. Uses between subgraph ops are left
    // unchanged; they remain valid after the ops are moved into the body.
    for (auto [outVal, kernelRes] : llvm::zip(outputValues, kernelOp.getResults())) {
        rewriter.replaceUsesWithIf(outVal, kernelRes, [&](mlir::OpOperand& use) {
            return !subgraphSet.contains(use.getOwner());
        });
    }

    // Clone subgraph ops into the body in topological order, then add the yield.
    // Cloning rather than moving keeps the function correct when a subgraph op has
    // uses outside the subgraph.
    rewriter.setInsertionPointToEnd(bodyBlock);
    mlir::IRMapping bodyMap;
    for (auto* op : subgraph) {
        rewriter.clone(*op, bodyMap);
    }
    auto clonedOutputValues = llvm::to_vector(llvm::map_range(outputValues, [&](mlir::Value v) {
        return bodyMap.lookupOrDefault(v);
    }));
    rewriter.create<Shave::KernelRegionYieldOp>(kernelOp.getLoc(), clonedOutputValues);

    // Isolate the kernel region from above. Collect all values captured from above, then extend the
    // set to include clonable ops.
    llvm::SetVector<mlir::Value> initialCaptures;
    mlir::getUsedValuesDefinedAbove(kernelOp.getBody(), initialCaptures);

    // Determine what ops need to be cloned and what values should become block arguments.
    // We're rematerializing some values (e.g. constants, scalars) in the kernel
    // region through cloning, so we need to walk up the def-use chains to
    // decide what ops get cloned.
    std::deque<mlir::Value> worklist(initialCaptures.begin(), initialCaptures.end());
    llvm::DenseSet<mlir::Value> visited;
    llvm::DenseSet<mlir::Operation*> visitedOps;
    llvm::DenseSet<mlir::Operation*> opsMarkedForClone;
    llvm::SetVector<mlir::Value> externalInputs;
    llvm::SmallVector<mlir::Operation*> opsToClone;

    auto markForClone = [&](mlir::Operation* op) {
        opsMarkedForClone.insert(op);
        opsToClone.push_back(op);
        for (auto operand : op->getOperands()) {
            if (!visited.count(operand)) {
                worklist.push_back(operand);
            }
        }
    };

    while (!worklist.empty()) {
        auto val = worklist.front();
        worklist.pop_front();
        if (!visited.insert(val).second) {
            continue;
        }

        auto* defOp = val.getDefiningOp();
        if (!defOp || visitedOps.count(defOp)) {
            // Block argument, or a result of an op already visited.
            // Skip if the op is being cloned -- its results will be remapped by the clone.
            if (!defOp || !opsMarkedForClone.count(defOp)) {
                externalInputs.insert(val);
            }
            continue;
        }
        visitedOps.insert(defOp);

        if (auto dimOp = mlir::dyn_cast<mlir::tensor::DimOp>(defOp)) {
            // Clone tensor.dim only when its source tensor is a graph input
            // (block argument) or was produced within the subgraph. In the latter
            // case the source may already have been replaced by a KernelRegion
            // result.
            auto src = dimOp.getSource();
            auto* srcDefOp = src.getDefiningOp();
            // Clone tensor.dim if its source tensor is:
            //   - produced by an op in the subgraph (non-output case),
            //   - a subgraph output that was replaced by a KernelRegion result, or
            //   - produced outside the subgraph but consumed by a subgraph op
            //     (i.e., a subgraph input tensor, which will already become a
            //     block arg of the kernel region for other reasons).
            //   - a value already captured by the kernel region body (e.g. a
            //     capsule block argument used only inside a nested scf.for body),
            //     in which case the source will become a block arg of the kernel
            //     region anyway, so tensor.dim belongs inside too.
            bool srcFromSubgraph = srcDefOp && (subgraphSet.contains(srcDefOp) || srcDefOp == kernelOp.getOperation() ||
                                                llvm::any_of(src.getUsers(), [&](mlir::Operation* user) {
                                                    return subgraphSet.contains(user);
                                                }));
            bool srcIsCapture = initialCaptures.contains(src);
            if (srcFromSubgraph || srcIsCapture) {
                markForClone(defOp);
            } else {
                externalInputs.insert(val);
            }
            continue;
        }

        if (shouldCloneOpIntoKernelRegion(defOp)) {
            markForClone(defOp);
            continue;
        }

        externalInputs.insert(val);
    }

    // Sort external inputs: shaped (tensor/memref) arguments first, scalars last.
    auto sortedExternalInputs = llvm::to_vector(externalInputs);
    llvm::stable_sort(sortedExternalInputs, [](mlir::Value lhs, mlir::Value rhs) {
        return mlir::isa<mlir::ShapedType>(lhs.getType()) && !mlir::isa<mlir::ShapedType>(rhs.getType());
    });

    // Clone ops in topological order at the start of the body.
    // IRMapping remaps operands of later clones to already-cloned earlier results.
    // Extend bodyMap with kernelOp result -> in-body value entries so that
    // clonable ops (e.g. tensor.dim) whose source was a subgraph output --
    // and therefore had its outside use replaced by a kernelOp result via
    // replaceUsesWithIf -- resolve to the correct in-body clone.
    for (auto [outVal, kernelRes] : llvm::zip(outputValues, kernelOp.getResults())) {
        bodyMap.map(kernelRes, bodyMap.lookupOrDefault(outVal));
    }
    if (!mlir::computeTopologicalSorting(opsToClone)) {
        VPUX_THROW("cycle detected among ops-to-clone in wrapSubgraphInKernelRegion");
    }
    rewriter.setInsertionPointToStart(bodyBlock);
    for (auto* op : opsToClone) {
        auto* newOp = rewriter.clone(*op, bodyMap);
        for (auto [origRes, clonedRes] : llvm::zip(op->getResults(), newOp->getResults())) {
            rewriter.replaceUsesWithIf(origRes, clonedRes, [&](mlir::OpOperand& use) {
                return kernelOp.getBody().isAncestor(use.getOwner()->getParentRegion());
            });
        }
    }

    // Add one block arg per external input and replace all uses inside the region
    // with the new block arguments.
    for (auto extVal : sortedExternalInputs) {
        auto blockArg = bodyBlock->addArgument(extVal.getType(), extVal.getLoc());
        rewriter.replaceUsesWithIf(extVal, blockArg, [&](mlir::OpOperand& use) {
            return kernelOp.getBody().isAncestor(use.getOwner()->getParentRegion());
        });
    }

    // Wire the KernelRegionOp operands with the added block arguments.
    rewriter.modifyOpInPlace(kernelOp, [&] {
        kernelOp->setOperands(sortedExternalInputs);
    });
    return kernelOp;
}

void WrapInKernelRegionPass::runOnCapsule(IE::CodeGenCapsuleOp capsule) {
    mlir::IRRewriter rewriter(capsule);

    llvm::SmallVector<mlir::scf::ForOp> tiledLoops;
    // IE::CodeGenCapsuleOp is guaranteed to always have exactly one block.
    for (auto& op : capsule.getContent().front()) {
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(&op)) {
            if (mlir::isa_and_nonnull<Shave::LoopTripCountOp>(forOp.getUpperBound().getDefiningOp())) {
                tiledLoops.push_back(forOp);
            }
        }
    }

    // All tilable regions must be wrapped first in KernelRegions.
    for (auto forOp : tiledLoops) {
        wrapSubgraphInKernelRegion(rewriter, {forOp.getOperation()});
        rewriter.eraseOp(forOp);
    }

    // Wrap remaining ops in KernelRegions.
    auto& capsuleBlock = capsule.getContent().front();
    for (auto& op : llvm::make_early_inc_range(llvm::reverse(capsuleBlock))) {
        if (mlir::isOpTriviallyDead(&op)) {
            rewriter.eraseOp(&op);
            continue;
        }
        if (isNotARootCandidate(&op)) {
            continue;
        }
        auto subgraph = collectSubgraphFromRoot(&op);
        wrapSubgraphInKernelRegion(rewriter, subgraph);
        rewriter.eraseOp(&op);
    }
}

void WrapInKernelRegionPass::safeRunOnFunc() {
    auto func = getOperation();

    auto capsules = llvm::to_vector(func.getOps<IE::CodeGenCapsuleOp>());
    for (auto capsule : capsules) {
        runOnCapsule(capsule);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createWrapInKernelRegionPass(Logger log) {
    return std::make_unique<WrapInKernelRegionPass>(log);
}
