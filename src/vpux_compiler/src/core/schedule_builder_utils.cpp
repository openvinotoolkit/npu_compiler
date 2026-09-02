//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include "vpux/utils/logger/logger.hpp"

using namespace vpux;

namespace {
VPURT::TaskQueueType getQueueType(mlir::async::ExecuteOp execOp) {
    VPURT::TaskQueueType queueType = {};
    queueType.type = VPUIP::getExecutorType(execOp);

    if (auto dmaTask = VPUIP::getDmaTypeOp(execOp)) {
        queueType.id = getDMAQueueIdEncoding(dmaTask.getChannelType());
    }
    return queueType;
}

AllocationType getAllocationType(mlir::async::ExecuteOp execOp, bool hasValidDep) {
    // skip memory movement ops
    if (VPUIP::isDmaDDR2CMX(execOp)) {
        return AllocationType::DATA_IN;
    }

    if (hasValidDep && VPUIP::isDmaCMX2DDR(execOp)) {
        // valid dependency needed for DMA->DMA patterns
        return AllocationType::DATA_OUT;
    }

    return AllocationType::COMPUTE;
}

std::pair<SmallVector<mlir::Value>, SmallVector<mlir::Value>> getOperationBuffers(mlir::async::ExecuteOp execOp,
                                                                                  AliasesInfo& aliasInfo) {
    SmallVector<mlir::Value> inBuffers;
    SmallVector<mlir::Value> outBuffers;

    auto isTargetMemType = [&](mlir::Value buf) {
        auto bufType = getAsyncValueType(buf);
        auto bufNDType = mlir::dyn_cast<vpux::NDTypeInterface>(bufType);

        if (bufNDType == nullptr) {
            return false;
        }

        return bufNDType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
    };

    auto updateBufferStorage = [&](const SmallVector<mlir::Value>& buffers, SmallVector<mlir::Value>& bufferStorage,
                                   llvm::DenseSet<mlir::Value>& cache) {
        bufferStorage.reserve(bufferStorage.size() + buffers.size());
        for (const auto& buffer : buffers) {
            if (!isTargetMemType(buffer)) {
                continue;
            }
            const auto& roots = aliasInfo.getRoots(buffer);
            if (roots.size() == 1 && roots.front() == buffer) {
                if (cache.count(buffer)) {
                    continue;
                }
                cache.insert(buffer);
                bufferStorage.push_back(buffer);
            } else {
                const auto root = aliasInfo.getRoot(buffer);
                if (cache.count(root)) {
                    continue;
                }
                cache.insert(root);
                bufferStorage.push_back(root);
            }
        }
    };

    auto* bodyBlock = execOp.getBody();

    llvm::DenseSet<mlir::Value> inBuffersCache;
    llvm::DenseSet<mlir::Value> outBuffersCache;

    for (auto& innerOp : bodyBlock->getOperations()) {
        if (auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(innerOp)) {
            auto inputs = VPUIP::getInputsSanitized(layerOp);
            auto outputs = layerOp.getOutputs();

            updateBufferStorage(inputs, inBuffers, inBuffersCache);
            updateBufferStorage(outputs, outBuffers, outBuffersCache);
        }
    }

    return {std::move(inBuffers), std::move(outBuffers)};
}

bool hasNonDmaDependency(ArrayRef<size_t> depInd, const llvm::DenseSet<size_t>& nonDmaOps) {
    for (auto& depIdx : depInd) {
        if (nonDmaOps.find(depIdx) != nonDmaOps.end()) {
            return true;
        }
    }
    return false;
}

std::unique_ptr<SchedulingLoop> makeSchedulingLoopFromIterations(std::vector<LoopBody> iterations, LoopType type) {
    auto loop = std::make_unique<SchedulingLoop>();
    loop->type = type;
    loop->loopBodies = std::move(iterations);
    return loop;
}

OpAllocationInfo createAllocInfo(size_t opIdx, AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo,
                                 [[maybe_unused]] Logger log) {
    const auto execOp = depsInfo.getExecuteOpAtIndex(opIdx);
    const auto dependencies = depsInfo.getOpDeps(opIdx);
    // has non-dma dependency
    bool hasNonDmaDep = false;
    hasNonDmaDep = std::any_of(dependencies.begin(), dependencies.end(), [&](size_t depIdx) {
        return VPUIP::getExecutorType(depIdx, depsInfo) != config::ExecutorKind::DMA_NN;
    });

    const auto queueType = getQueueType(execOp);
    const auto consumers = depsInfo.getConsumerOps(opIdx);
    auto [inBuffers, outBuffers] = getOperationBuffers(execOp, aliasInfo);
    const auto allocationType = getAllocationType(execOp, hasNonDmaDep);

#ifdef VPUX_DEVELOPER_BUILD
    log.trace("createAllocInfo: Op {0} executor {1} type {2} inBuffers {3} outBuffers {4} "
              "deps {5} cons {6}",
              opIdx, queueType.type, toString(allocationType), inBuffers, outBuffers, dependencies, consumers);
#endif

    return OpAllocationInfo(opIdx, queueType, inBuffers, outBuffers, allocationType);
}

/**
 *   @name createInnerLoopsFromIterations
 *   @brief create compute regions from loop iterations based on shared dependencies and consumers
 *
 *   @param loop - vector of LoopBody representing iterations
 *   @param allLoopOperations - set of all operations inside the loop
 *   @param depsInfo - async dependencies info
 *   @param nonDmaOps - set of all non-dma operation indexes
 *   @return map of compute regions grouped by insertion points,
 *           insertion point is the first compute operation index in the loop
 *
 *   @details
 *   Algorithm:
 *   for: every iteration in the loop
 *       get global deps and cons for the iteration
 *       for every other iteration in the loop
 *           compare local buffer counts with current iteration
 *           if they match add other iteration to matching iterations
 *       create compute region from matching iterations
 *
 *   Matching criteria considerations:
 *       - Same number of compute ops per iteration
 *       - Same number of local buffers (raw and deduplicated)
 *
 *   Limitations and assumptions:
 *       - Matching criteria met for all merged iterations
 *       - Minimum number of operations in the loop: MIN_TILING_LOOP_OPS, MIN_VF_LOOP_OPS
 *       - Loops are created inplace of first compute operation in the loop
 */

llvm::DenseMap<size_t, ComputeRegionVec> createInnerLoopsFromIterations(
        ArrayRef<LoopBody> loop, llvm::DenseSet<size_t>& allLoopOperations, LoopType loopType, AsyncDepsInfo& depsInfo,
        [[maybe_unused]] llvm::DenseSet<size_t>& nonDmaOps, Logger log) {
    // Precompute global deps for each iteration to avoid recomputing inside matching loop
    SmallVector<std::set<size_t>> cachedGlobalDeps(loop.size());

    for (size_t iterIdx = 0; iterIdx < loop.size(); ++iterIdx) {
        // deps
        llvm::DenseSet<size_t> loopBodyOpsIdxs;
        for (const auto& op : loop[iterIdx]) {
            loopBodyOpsIdxs.insert(op.opIdx);
        }
        std::set<size_t> loopGlobalDeps;
        for (const auto& op : loop[iterIdx]) {
            for (const auto& dep : depsInfo.getOpDeps(op.opIdx)) {
                if (loopBodyOpsIdxs.count(dep) == 0) {
                    loopGlobalDeps.insert(dep);
                }
            }
        }
        cachedGlobalDeps[iterIdx] = std::move(loopGlobalDeps);

#ifdef VPUX_DEVELOPER_BUILD
        // Log the DDR2DDR consumer cases for performance debug hints
        // DDR2DDR consumer after loop region is possible to cause schedule difference
        // This doesn't cause any real performance regression so only keep this log for developer debug
        for (const auto& op : loop[iterIdx]) {
            const auto execOp = depsInfo.getExecuteOpAtIndex(op.opIdx);
            const auto deps = depsInfo.getOpDeps(op.opIdx);
            const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, nonDmaOps));
            if (allocationType == AllocationType::DATA_IN) {
                continue;
            }
            for (const auto& con : depsInfo.getConsumerOps(op.opIdx)) {
                if (loopType == LoopType::Tiling && VPUIP::isDmaDDR2DDR(depsInfo.getExecuteOpAtIndex(con))) {
                    //   COMPUTE         COMPUTE
                    //     |                |
                    // DMA(CMX2DDR)     DMA(CMX2DDR)
                    //      \            /
                    //       DMA(DDR2DDR)
                    // with loop logic such DDR2DDR DMAs are scheduled after loop
                    log.debug("DDR2DDR DMA consumer detected in loop iteration {0}/{1}", iterIdx, loop.size());
                    break;
                }
            }
        }
#endif
    }

    auto getGlobalDepsForLoop = [&](const std::vector<LoopBody>& loop, const std::set<size_t>& matchingIters) {
        // Collect all op indices across merged iterations
        llvm::DenseSet<size_t> loopOps;
        for (const auto& iteration : loop) {
            for (const auto& op : iteration) {
                loopOps.insert(op.opIdx);
            }
        }
        // Reuse cached per-iteration global deps, filter out cross-iteration references
        llvm::DenseSet<size_t> loopGlobalDeps;
        for (auto iterIdx : matchingIters) {
            for (auto dep : cachedGlobalDeps[iterIdx]) {
                if (loopOps.count(dep) == 0) {
                    loopGlobalDeps.insert(dep);
                }
            }
        }
        return loopGlobalDeps;
    };

    auto getDedupBufferCount = [](ArrayRef<mlir::Value> buffers) -> size_t {
        llvm::DenseSet<mlir::Value> uniqueBuffers;
        for (const auto& buf : buffers) {
            uniqueBuffers.insert(buf);
        }
        return uniqueBuffers.size();
    };

    // Cache compute ops for each iteration to avoid recomputing O(N^2) times inside loop
    SmallVector<SmallVector<OpAllocationInfo>> cachedComputeAllocs(loop.size());
    for (size_t i = 0; i < loop.size(); ++i) {
        cachedComputeAllocs[i].reserve(loop[i].size());
        for (const auto& op : loop[i]) {
            if (op.allocationType == AllocationType::COMPUTE) {
                cachedComputeAllocs[i].push_back(op);
            }
        }
    }

    auto getComputeAlloc = [&](size_t iterationIdx) -> const SmallVector<OpAllocationInfo>& {
        VPUX_THROW_UNLESS(iterationIdx < cachedComputeAllocs.size(), "Invalid loop iteration index {0}", iterationIdx);
        return cachedComputeAllocs[iterationIdx];
    };

    log.trace("Creating compute regions from iterations = {0}", loop.size());
    // Merge individual iterations into 1D loops based on shared dependencies and consumers
    llvm::DenseMap<size_t, ComputeRegionVec> computeRegions;
    llvm::DenseSet<size_t> handledIterations;
    for (size_t currentIdx = 0; currentIdx < loop.size(); ++currentIdx) {
        if (handledIterations.count(currentIdx) > 0) {
            log.trace("handled iteration idx {0}", currentIdx);
            continue;
        }
        const auto& computeAllocVec = getComputeAlloc(currentIdx);

        // Topology of current iteration compute ops does not depend on otherIdx,
        // so cache it once per currentIdx and reuse in the nested comparison loop.
        SmallVector<BufferEqualityTopology> currentComputeTopologies;
        currentComputeTopologies.reserve(computeAllocVec.size());
        for (const auto& currentCompute : computeAllocVec) {
            currentComputeTopologies.push_back(getBufferEqualityTopology(currentCompute));
        }

        std::set<size_t> matchingIterations;
        matchingIterations.insert(currentIdx);

        // Check if any other iteration uses these deps.
        for (size_t otherIdx = 0; otherIdx < loop.size(); ++otherIdx) {
            if (handledIterations.count(otherIdx) > 0 || otherIdx == currentIdx) {
                log.trace("skip otherIdx {0}", otherIdx);
                continue;
            }
            const auto& otherComputeAllocVec = getComputeAlloc(otherIdx);
            if (computeAllocVec.size() != otherComputeAllocVec.size()) {
                log.trace("Inconsistent number of compute ops between iterations");
                continue;
            }
            bool consistentOperands = true;
            for (size_t computeIndex : irange(computeAllocVec.size())) {
                const auto& currentCompute = computeAllocVec[computeIndex];
                const auto& otherCompute = otherComputeAllocVec[computeIndex];

                // Use deduplicated counts to handle repeated operands that alias the same buffer.
                const auto currentDedupInCount = getDedupBufferCount(currentCompute.inBuffers);
                const auto currentDedupOutCount = getDedupBufferCount(currentCompute.outBuffers);
                const auto otherDedupInCount = getDedupBufferCount(otherCompute.inBuffers);
                const auto otherDedupOutCount = getDedupBufferCount(otherCompute.outBuffers);
                if (currentDedupInCount != otherDedupInCount || currentDedupOutCount != otherDedupOutCount) {
                    log.trace("skip otherIdx {0} different dedup local buffers, current in/out {1}/{2}, other in/out "
                              "{3}/{4}",
                              otherIdx, currentDedupInCount, currentDedupOutCount, otherDedupInCount,
                              otherDedupOutCount);
                    consistentOperands = false;
                    break;
                }

                // Iterations must also be isomorphic by in/out equality topology (e.g. in-place overlap).
                // Example mismatch to reject:
                //   iter A: in=[A,B], out=[C]  -> in=[0,1], out=[2]
                //   iter B: in=[X,Y], out=[X]  -> in=[0,1], out=[0]
                const auto& currentTopology = currentComputeTopologies[computeIndex];
                const auto otherTopology = getBufferEqualityTopology(otherCompute);
                if (currentTopology.inClassIds != otherTopology.inClassIds ||
                    currentTopology.outClassIds != otherTopology.outClassIds) {
                    log.trace("skip otherIdx {0} different buffer equality topology", otherIdx);
                    consistentOperands = false;
                    break;
                }
            }

            if (!consistentOperands) {
                continue;
            }

            // loops can be merged
            matchingIterations.insert(otherIdx);
        }

        if ((loopType == LoopType::VF && matchingIterations.size() < MIN_VF_LOOP_OPS) ||
            (loopType == LoopType::Tiling && matchingIterations.size() < MIN_TILING_LOOP_OPS)) {
            log.trace("skip currentIdx {0} only {1} matching iterations", currentIdx, matchingIterations.size());
            continue;
        }

        // Merge matching iterations into a single loop and mark their ops as handled
        std::vector<LoopBody> mergedIterations;
        mergedIterations.reserve(matchingIterations.size());
        for (auto matchIdx : matchingIterations) {
            handledIterations.insert(matchIdx);
            LoopBody tempIteration;
            for (auto& op : loop[matchIdx]) {
                tempIteration.push_back(op);
                allLoopOperations.insert(op.opIdx);
            }
            log.nest().trace("Add op to loop {0}", depsInfo.getExecuteOpAtIndex(tempIteration[0].opIdx).getLoc());
            mergedIterations.push_back(std::move(tempIteration));
        }

        // Insert inplace of first compute
        size_t insertionPoint = std::numeric_limits<size_t>::max();
        for (auto& op : mergedIterations[0]) {
            if (op.allocationType == AllocationType::COMPUTE) {
                insertionPoint = std::min(insertionPoint, op.opIdx);
            }
        }

        const auto loopGlobalDeps = getGlobalDepsForLoop(mergedIterations, matchingIterations);
        // Ensure loop inserted after all deps
        auto lastDepOpIdx = std::max_element(loopGlobalDeps.begin(), loopGlobalDeps.end());
        if (lastDepOpIdx != loopGlobalDeps.end()) {
            insertionPoint = std::max(insertionPoint, *lastDepOpIdx);
        }
        SmallVector<size_t> globalDeps(loopGlobalDeps.begin(), loopGlobalDeps.end());
        llvm::sort(globalDeps);

        // Create compute region
        const auto iterations = matchingIterations.size();
        log.nest().debug("Created sub-loop with iterations = {0}", iterations);
        auto schedulingLoop = makeSchedulingLoopFromIterations(std::move(mergedIterations), loopType);
        computeRegions[insertionPoint].emplace_back(std::move(schedulingLoop), std::move(globalDeps));
    }  // end current iteration loop

    return computeRegions;
}

/**
    Improves determinism and keeps data-move ops aligned with the compute op’s operand order,
    which can simplify later allocation and scheduling logic.
 */
void sortDataOpsAroundComputeOp(LoopBody& allocInfos, size_t computeOpPosition) {
    // Sort data in allocations
    // 1. lookup based on compute op
    llvm::DenseMap<mlir::Value, size_t> position;
    auto recordPositions = [&](ArrayRef<mlir::Value> buffers) {
        position.reserve(position.size() + buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            // keeps the first (earliest, since i iterates upwards) occurrence; duplicates are ignored.
            position.try_emplace(buffers[i], i);
        }
    };

    recordPositions(allocInfos[computeOpPosition].inBuffers);
    // 2. Sort based on compute op buffer order
    auto getOrderKey = [&](const OpAllocationInfo& a, bool useInputs) {
        size_t best = std::numeric_limits<size_t>::max();
        const auto& buffers = useInputs ? a.inBuffers : a.outBuffers;
        for (const auto& buf : buffers) {
            auto it = position.find(buf);
            if (it != position.end()) {
                best = std::min(best, it->second);
            }
        }
        return best;
    };
    // stable_sort keeps the original relative order when no compute buffer matches
    std::stable_sort(allocInfos.begin(), allocInfos.begin() + computeOpPosition,
                     [&](const OpAllocationInfo& a, const OpAllocationInfo& b) {
                         return getOrderKey(a, /*useInputs=*/false) < getOrderKey(b, /*useInputs=*/false);
                     });

    // Sort data out allocations
    // 1. lookup based on compute op
    position.clear();
    recordPositions(allocInfos[computeOpPosition].outBuffers);

    // 2. sort based on compute op buffer order
    if (computeOpPosition + 1 < allocInfos.size()) {
        std::stable_sort(allocInfos.begin() + computeOpPosition + 1, allocInfos.end(),
                         [&](const OpAllocationInfo& a, const OpAllocationInfo& b) {
                             return getOrderKey(a, /*useInputs=*/true) < getOrderKey(b, /*useInputs=*/true);
                         });
    }
}

// Create minimalistic description of tiled operations including unique/local data in/out dependencies and consumers
SmallVector<LoopBody> createTiledOpDepsConsDescriptor(llvm::DenseMap<size_t, SmallVector<size_t>>& loopOps,
                                                      LoopType loopType, AliasesInfo& aliasInfo,
                                                      AsyncDepsInfo& depsInfo, const llvm::DenseSet<size_t>& nonDmaOps,
                                                      Logger log) {
    log.trace("createTiledOpDepsConsDescriptor called for {0} tiles", loopOps.size());
    auto iterations = loopOps.size();
    SmallVector<LoopBody> loop;
    auto computeOps = nonDmaOps;

    for (size_t i = 0; i < iterations; ++i) {
        // In the first version we'll only consider compute ops with loop attributes (loopOps is built based on loop
        // attributes)
        // TODO E#211215: consider pulling in also compute ops directly linked to vf but without vf loop attributes

        SmallVector<size_t> computeIndices = loopOps[i];
        llvm::DenseSet<size_t> processed(computeIndices.begin(), computeIndices.end());
        LoopBody currentIteration;

        if (loopType == LoopType::VF) {
            // If op is CMX2CMX DMA treat is as compute loop op
            auto handleCmxToCmxDmaOpAsCompute =
                    [&depsInfo, &nonDmaOps](llvm::DenseSet<size_t>& processed, SmallVector<size_t>& computeIndices,
                                            llvm::DenseSet<size_t>& computeOps, size_t candidateOpIdx) {
                        if (processed.count(candidateOpIdx)) {
                            return;
                        }
                        processed.insert(candidateOpIdx);

                        const auto execOp = depsInfo.getExecuteOpAtIndex(candidateOpIdx);
                        if (!VPUIP::isDmaCMX2CMX(execOp)) {
                            return;
                        }
                        const auto deps = depsInfo.getOpDeps(candidateOpIdx);
                        const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, nonDmaOps));
                        if (allocationType == AllocationType::COMPUTE) {
                            computeIndices.push_back(candidateOpIdx);
                            // Since CMX2CMX DMAs within the loop are treated as compute ops, include them in the
                            // compute ops list to ensure they are included when looking for DATA_IN and DATA_OUT ops
                            computeOps.insert(candidateOpIdx);
                        }
                    };

            // Iterate over compute op dependencies and consumers and find
            // CMX2CMX DMAs. Since they will not be treated as DATA_IN and DATA_OUT ops handle
            // such data transfers as loop compute ops.
            for (const auto opIdx : loopOps[i]) {
                for (auto depIdx : depsInfo.getOpDeps(opIdx)) {
                    handleCmxToCmxDmaOpAsCompute(processed, computeIndices, computeOps, depIdx);
                }

                for (auto conIdx : depsInfo.getConsumerOps(opIdx)) {
                    handleCmxToCmxDmaOpAsCompute(processed, computeIndices, computeOps, conIdx);
                }
            }
            llvm::sort(computeIndices);
            processed.clear();
            processed.insert(computeIndices.begin(), computeIndices.end());
        }

        for (const auto& opIdx : computeIndices) {
            LoopBody currentOperation;
            for (auto depIdx : depsInfo.getOpDeps(opIdx)) {
                if (processed.count(depIdx)) {
                    continue;
                }

                processed.insert(depIdx);

                const auto execOp = depsInfo.getExecuteOpAtIndex(depIdx);
                const auto deps = depsInfo.getOpDeps(depIdx);
                const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, computeOps));

                if (allocationType != AllocationType::DATA_IN) {
                    // only include data in ops
                    continue;
                }
                currentOperation.push_back(createAllocInfo(depIdx, depsInfo, aliasInfo, log));
            }

            currentOperation.push_back(createAllocInfo(opIdx, depsInfo, aliasInfo, log));
            const size_t computeOpPosition = currentOperation.size() - 1;

            for (auto conIdx : depsInfo.getConsumerOps(opIdx)) {
                if (processed.count(conIdx)) {
                    continue;
                }

                processed.insert(conIdx);

                const auto deps = depsInfo.getOpDeps(conIdx);

                auto lastDepOpIndex = std::max_element(deps.begin(), deps.end());
                // ensure data out fully unlocked at this stage
                if (lastDepOpIndex != deps.end() && *lastDepOpIndex > opIdx) {
                    continue;
                }
                const auto execOp = depsInfo.getExecuteOpAtIndex(conIdx);
                const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, computeOps));
                if (allocationType != AllocationType::DATA_OUT) {
                    // only include data out ops
                    continue;
                }

                if (auto dmaTask = VPUIP::getDmaTypeOp(execOp)) {
                    // Skip profiling management DMA as they are not related to compute path and loop regions and where
                    // not considered during strategy manager and tiling Including profiling DMA in loop region may
                    // create coupling between loop and nonLoop ops and result in scheduler failure due to cyclic
                    // dependency between compute region and nonLoop op.
                    // TODO: Profiling handling at memory scheduler will no longer be needed after E#193554
                    if (auto nnDmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(dmaTask.getOperation())) {
                        if (nnDmaOp.getProfilingBufferMgmt()) {
                            continue;
                        }
                    }
                }

                currentOperation.push_back(createAllocInfo(conIdx, depsInfo, aliasInfo, log));
            }

            sortDataOpsAroundComputeOp(currentOperation, computeOpPosition);
            for (auto& op : currentOperation) {
                currentIteration.push_back(std::move(op));
            }
        }
        loop.push_back(std::move(currentIteration));
    }

    // Count how many iterations each op appears in
    llvm::DenseMap<size_t, size_t> loopOpCounts;
    for (const auto& iterationBody : loop) {
        for (const auto& op : iterationBody) {
            ++loopOpCounts[op.opIdx];
        }
    }

    // re-create loops without global-shared ops
    // An op appearing in all iterations (global-shared) is factored out of the iteration body
    // so that createInnerLoopsFromIterations treats it as a global dependency. Partially-shared
    // ops (appearing in more than one but not all iterations) are kept inside their iteration
    // bodies so that createInnerLoopsFromIterations can use them for correct matching:
    // iterations sharing a partially-shared op (e.g., a weight DMA used by one C-tile's
    // H-tile group) will match on that kept op, while iterations with different partially-shared
    // ops form separate loops.
    SmallVector<LoopBody> filteredLoop;
    const auto numIterations = loop.size();
    for (auto& iterationBody : loop) {
        LoopBody filteredIteration;
        for (auto& op : iterationBody) {
            VPUX_THROW_WHEN(loopOpCounts[op.opIdx] > 1 && op.allocationType == AllocationType::COMPUTE &&
                                    !VPUIP::isDmaCMX2CMX(depsInfo.getExecuteOpAtIndex(op.opIdx)),
                            "Compute op {0} count {1} is shared between iterations, but it should not be", op.opIdx,
                            loopOpCounts[op.opIdx]);

            if (loopOpCounts[op.opIdx] == numIterations) {
                // op appears in all the iterations of the loop, consider it as shared op and filter it out
                continue;
            }
            filteredIteration.push_back(std::move(op));
        }
        filteredLoop.push_back(std::move(filteredIteration));
    }

    return filteredLoop;
}

}  // namespace

BufferEqualityTopology vpux::getBufferEqualityTopology(const OpAllocationInfo& op) {
    BufferEqualityTopology topo;
    llvm::DenseMap<mlir::Value, uint32_t> classByValue;
    uint32_t nextClassId = 0;

    auto encode = [&](ArrayRef<mlir::Value> buffers, SmallVector<uint32_t>& outIds) {
        outIds.reserve(buffers.size());
        for (auto value : buffers) {
            auto [it, inserted] = classByValue.try_emplace(value, nextClassId);
            if (inserted) {
                ++nextClassId;
            }
            outIds.push_back(it->second);
        }
    };

    encode(op.inBuffers, topo.inClassIds);
    encode(op.outBuffers, topo.outClassIds);
    return topo;
}

std::string vpux::stringifyClassIds(ArrayRef<uint32_t> classIds) {
    // Keeps topology mismatch logs compact and easy to compare across iterations.
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "[";
    for (size_t i = 0; i < classIds.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << classIds[i];
    }
    os << "]";
    return os.str();
}

// Extract compute regions from async.execute ops based on loop attributes
ComputeRegionVec vpux::getComputeRegionsFromAsyncExec(AliasesInfo& aliasInfo, AsyncDepsInfo& depsInfo, Logger log) {
    depsInfo.buildConsMap();

    // 1. Scan all async.exec ops to collect loop regions (via loop attributes) and track non-DMA ops.

    // Use std::map so iteration is ordered (by LoopType enum value, then by loop index) ensuring
    // deterministic ComputeRegion ordering when multiple loops share the same insertion point (good for
    // reproducibility and debugging).
    // Usage: loopRegions[loopType][loopIndex][loopTileIndex] = SmallVector<size_t: opIdx in the loop iteration>
    std::map<LoopType, std::map<size_t, llvm::DenseMap<size_t, SmallVector<size_t>>>> loopRegions;
    llvm::DenseSet<size_t> nonDmaOps;

    log.debug("Scanning async.exec ops for tiled regions, total ops {0}", depsInfo.getExecOpCount());
    for (size_t opIdx = 0; opIdx < depsInfo.getExecOpCount(); ++opIdx) {
        auto execOp = depsInfo.getExecuteOpAtIndex(opIdx);
        auto* bodyBlock = execOp.getBody();

        for (auto& op : bodyBlock->getOperations()) {
            const auto loopAttributes = vpux::getLoopAttributes(&op);

            if (loopAttributes.tilingLoopIndex != nullptr && mlir::isa<I64Attr>(loopAttributes.tilingLoopIndex)) {
                auto tilingLoopIndex = mlir::cast<mlir::IntegerAttr>(loopAttributes.tilingLoopIndex).getInt();
                const auto tileIdx = loopRegions[LoopType::Tiling][tilingLoopIndex].size();
                loopRegions[LoopType::Tiling][tilingLoopIndex][tileIdx].push_back(opIdx);
            }
            if (loopAttributes.vfLoopIndex != nullptr && mlir::isa<I64Attr>(loopAttributes.vfLoopIndex) &&
                loopAttributes.vfLoopTileIndex != nullptr && mlir::isa<I64Attr>(loopAttributes.vfLoopTileIndex)) {
                auto vfLoopIndex = mlir::cast<mlir::IntegerAttr>(loopAttributes.vfLoopIndex).getInt();
                auto vfLoopTileIdx = mlir::cast<mlir::IntegerAttr>(loopAttributes.vfLoopTileIndex).getInt();
                loopRegions[LoopType::VF][vfLoopIndex][vfLoopTileIdx].push_back(opIdx);
            }
        }

        // keep track of non-DMA ops even if they are not tiled or vf
        const auto execKind = VPUIP::getExecutorType(execOp);
        if (execKind != config::ExecutorKind::DMA_NN) {
            nonDmaOps.insert(opIdx);
        }
    }

    // If no loop regions are found, return an empty result
    if (loopRegions.empty()) {
        log.debug("No loop regions found in async.exec ops");
        return {};
    }
    log.debug("Found {0} loop regions", loopRegions.size());

    ComputeRegionVec computeRegionVec;
    llvm::DenseMap<size_t, ComputeRegionVec> insertionPoints;
    llvm::DenseSet<size_t> allLoopOperations;

    // Process loop regions and create ComputeRegion per detected loop
    for (auto& [loopType, loopRegion] : loopRegions) {
        for (auto& [loopIndex, loopOps] : loopRegion) {
            // For each loop region with enough iterations
            const auto iterations = loopOps.size();
            if ((loopType == LoopType::VF && iterations < MIN_VF_LOOP_OPS) ||
                (loopType == LoopType::Tiling && iterations < MIN_TILING_LOOP_OPS)) {
                continue;
            }

            if (loopType == LoopType::VF) {
                // every loop iterations needs to have at least MIN_VF_LOOP_BODY_COMPUTE_OPS compute ops to be
                // considered for vf loop-based scheduling
                const bool hasInsufficientComputeOpsInIteration = llvm::any_of(loopOps, [](const auto& iterOps) {
                    return iterOps.second.size() < MIN_VF_LOOP_BODY_COMPUTE_OPS;
                });
                if (hasInsufficientComputeOpsInIteration) {
                    log.trace("Skip VF loop index {0} with iteration having less than {1} compute ops", loopIndex,
                              MIN_VF_LOOP_BODY_COMPUTE_OPS);
                    continue;
                }
            }

            // build per-iteration descriptors of compute + data-in/out ops (excluding shared ops)
            SmallVector<LoopBody> loop =
                    createTiledOpDepsConsDescriptor(loopOps, loopType, aliasInfo, depsInfo, nonDmaOps, log);

            // Merge per-iteration LoopBody entries into schedulable loops
            llvm::DenseMap<size_t, ComputeRegionVec> thisOpInsertionPoints =
                    createInnerLoopsFromIterations(loop, allLoopOperations, loopType, depsInfo, nonDmaOps, log);

            for (auto& [insertionPoint, regions] : thisOpInsertionPoints) {
                auto& dest = insertionPoints[insertionPoint];
                dest.insert(dest.end(), std::make_move_iterator(regions.begin()),
                            std::make_move_iterator(regions.end()));
            }
        }
    }  // end process tiled regions loop

    for (size_t opIdx = 0; opIdx < depsInfo.getExecOpCount(); ++opIdx) {
        // handle non-loop operations. Insert them into computeRegionVec as trivial region with 1 iteration
        if (!allLoopOperations.count(opIdx)) {
            log.trace("Add opIdx {0} non-loop operation", opIdx);
            const auto alloc = createAllocInfo(opIdx, depsInfo, aliasInfo, log);
            auto schedulingLoop = makeSchedulingLoopFromIterations({{alloc}}, LoopType::None);
            computeRegionVec.emplace_back(std::move(schedulingLoop));
        }
        // insert loops at proper position
        if (insertionPoints.count(opIdx)) {
            auto& computeRegions = insertionPoints[opIdx];
            for (auto& computeRegion : computeRegions) {
                computeRegionVec.push_back(std::move(computeRegion));
                log.trace("Add tiled compute region: {0}", computeRegion);
            }
        }
    }

    log.debug("Total compute regions created: {0}", computeRegionVec.size());

    return computeRegionVec;
}
