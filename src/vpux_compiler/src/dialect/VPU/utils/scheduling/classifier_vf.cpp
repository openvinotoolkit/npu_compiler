//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/classifier_vf.hpp"
#include "vpux/compiler/core/scheduling/utils.hpp"

#include <llvm/Support/FormatVariadic.h>
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>

using namespace vpux;

//===----------------------------------------------------------------------===//
//                          CLASSIFIER VF
//===----------------------------------------------------------------------===//
//
// Pure analysis over a single ComputeRegion of LoopType::VF
//
// Phase 1 deliberately does not depend on AliasesInfo: that analysis is not
// threaded through ITemporalTilingScenario today. The classifier dedupes buffers by
// SSA-value identity only, same model used by UndefinedTiling.
// View chains (SubView/GenericReshape/DistributedCast) and sparsity components may produce
// duplicate records; switching to alias-rooted keying is tracked under TODO E#220339.

//
// 5 passes over the template iteration:
//   Pass 1 — iteration-identity check
//   Pass 2 — buffer collection + structural in-place
//   Pass 3 — liveness span on the template
//   Pass 4 — classification (loop-invariant / loop-carried / temporary)
//   Pass 5 — aggregates + persistent-fit estimate
//

namespace {

VFPrefetchHint getBestPrefetchHint(const ClassifierVFResult& result) {
    const uint64_t persistentCandidateBytes = result.persistentCandidateTotalBytes;
    const uint64_t loopBodyNoSpillPeakBytes = result.loopBodyNoSpillPeakBytes;
    const uint64_t inputBytes = result.inputBytes;
    const uint64_t outputBytes = result.outputBytes;
    const uint64_t cmxLimit = result.cmxMemoryLimitBytes;

    if (persistentCandidateBytes + loopBodyNoSpillPeakBytes + inputBytes + outputBytes <= cmxLimit) {
        return VFPrefetchHint::FULL_PREFETCHING;
    }
    if (persistentCandidateBytes + loopBodyNoSpillPeakBytes + outputBytes <= cmxLimit) {
        return VFPrefetchHint::LASTOP_PREFETCHING;
    }
    if (persistentCandidateBytes + loopBodyNoSpillPeakBytes <= cmxLimit) {
        return VFPrefetchHint::WEIGHTS_PREFETCHING;
    }
    return VFPrefetchHint::MINIMAL;
}

// True if every iteration body has the same op count and matching per-position
// AllocationType + queueType + in/out buffer arity + in/out equality topology.
bool checkIterationIdentity(const vpux::SchedulingLoop& loop, Logger& log) {
    if (loop.loopBodies.size() <= 1) {
        return true;
    }
    const auto& templateBody = loop.loopBodies.front();
    for (size_t i = 1; i < loop.loopBodies.size(); ++i) {
        const auto& body = loop.loopBodies[i];
        if (body.size() != templateBody.size()) {
            log.trace("iteration-identity FAIL: iter[{0}] op count {1} != template {2}", i, body.size(),
                      templateBody.size());
            return false;
        }
        for (size_t k = 0; k < body.size(); ++k) {
            if (body[k].allocationType != templateBody[k].allocationType ||
                body[k].queueType != templateBody[k].queueType) {
                log.trace("iteration-identity FAIL: iter[{0}] op[{1}] allocType/queueType mismatch "
                          "(iter alloc={2} queue={3}:{4} vs template alloc={5} queue={6}:{7})",
                          i, k, static_cast<int>(body[k].allocationType), stringifyEnum(body[k].queueType.type),
                          body[k].queueType.id, static_cast<int>(templateBody[k].allocationType),
                          stringifyEnum(templateBody[k].queueType.type), templateBody[k].queueType.id);
                return false;
            }
            if (body[k].inBuffers.size() != templateBody[k].inBuffers.size() ||
                body[k].outBuffers.size() != templateBody[k].outBuffers.size()) {
                log.trace("iteration-identity FAIL: iter[{0}] op[{1}] buffer arity mismatch "
                          "(iter in/out={2}/{3} vs template in/out={4}/{5})",
                          i, k, body[k].inBuffers.size(), body[k].outBuffers.size(), templateBody[k].inBuffers.size(),
                          templateBody[k].outBuffers.size());
                return false;
            }

            const auto iterTopo = getBufferEqualityTopology(body[k]);
            const auto templTopo = getBufferEqualityTopology(templateBody[k]);
            if (iterTopo.inClassIds != templTopo.inClassIds || iterTopo.outClassIds != templTopo.outClassIds) {
                log.trace("iteration-identity FAIL: iter[{0}] op[{1}] equality-topology mismatch "
                          "(iter in={2} out={3} vs template in={4} out={5})",
                          i, k, stringifyClassIds(iterTopo.inClassIds), stringifyClassIds(iterTopo.outClassIds),
                          stringifyClassIds(templTopo.inClassIds), stringifyClassIds(templTopo.outClassIds));
                return false;
            }
        }
    }
    return true;
}

// Collect buffers and detect in-place use.
//
// Deduplicate by SSA value: one SSA value -> one BufferRecord.
// If the same SSA value appears in both inBuffers and outBuffers of one op,
// mark that record as isInPlaceAlias = true.
//
// TODO E#220339: support index-based alias-root dedup and attribute-based in-place for
// different SSA values that alias the same memory. This needs AliasesInfo or
// Operation* in OpAllocationInfo, which ITemporalTilingScenario does not expose yet.
void collectBuffers(const vpux::SchedulingLoop& loop, ClassifierVFResult& result) {
    const auto& templateBody = loop.loopBodies.front();

    auto getOrInsertRecord = [&](mlir::Value buf) -> size_t {
        auto it = result.valueToRecord.find(buf);
        if (it != result.valueToRecord.end()) {
            return it->second;
        }
        const size_t idx = result.buffers.size();
        result.valueToRecord[buf] = idx;
        BufferRecord bufferRecord;
        bufferRecord.rootValue = buf;
        bufferRecord.size = getRawBufferSize(buf);
        bufferRecord.alignment = vpux::getAlignment(buf);
        bufferRecord.firstUseOp = std::numeric_limits<size_t>::max();
        bufferRecord.lastUseOp = 0;
        result.buffers.push_back(std::move(bufferRecord));
        return idx;
    };

    // Collect buffers from the template iteration only. Non-template iterations carry
    // distinct SSA values (per-tile outlined buffers); inserting records for them produces
    // phantom entries with no template liveness window (firstUseOp == SIZE_MAX) that
    // pollute downstream allocator reservations. Cross-iteration semantics (loop-invariant,
    // loop-carried) only require SSA-identity matches against template values, performed
    // via valueToRecord.lookup() in later passes.
    for (const auto& info : templateBody) {
        for (auto buf : info.inBuffers) {
            getOrInsertRecord(buf);
        }
        for (auto buf : info.outBuffers) {
            getOrInsertRecord(buf);
        }
    }

    // Mark structural in-place on the template iteration.
    for (const auto& info : templateBody) {
        for (auto inBuf : info.inBuffers) {
            for (auto outBuf : info.outBuffers) {
                if (inBuf == outBuf) {
                    const auto idx = result.valueToRecord[inBuf];
                    result.buffers[idx].isInPlaceAlias = true;
                }
            }
        }
    }

    // Iterations must have the same buffer layout per op (same in/out counts).
    // If counts differ, we throw.
    // Buffer sizes/alignment may still differ across iterations.
    // We take the maximum size/alignment per buffer position so template offsets
    // are safe for all iterations.
    auto bumpRecordSize = [&](mlir::Value templateVal, mlir::Value bodyVal) {
        const auto recIt = result.valueToRecord.find(templateVal);
        if (recIt == result.valueToRecord.end()) {
            return;
        }
        auto& bufferRecord = result.buffers[recIt->second];
        bufferRecord.size = std::max(bufferRecord.size, getRawBufferSize(bodyVal));
        bufferRecord.alignment = std::max(bufferRecord.alignment, vpux::getAlignment(bodyVal));
    };
    for (size_t iteration = 1; iteration < loop.loopBodies.size(); ++iteration) {
        const auto& bodyIteration = loop.loopBodies[iteration];
        VPUX_THROW_UNLESS(bodyIteration.size() == templateBody.size(),
                          "Iteration {0} body op count {1} does not match template's {2}, cannot compute "
                          "max-over-bodies size",
                          iteration, bodyIteration.size(), templateBody.size());
        for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
            const auto& templateIterOpInfo = templateBody[opPos];
            const auto& iterationOpInfo = bodyIteration[opPos];
            VPUX_THROW_UNLESS(iterationOpInfo.inBuffers.size() == templateIterOpInfo.inBuffers.size(),
                              "Iteration {0}, op {1}: in-buffer count {2} does not match template's {3}", iteration,
                              opPos, iterationOpInfo.inBuffers.size(), templateIterOpInfo.inBuffers.size());
            const size_t inN = templateIterOpInfo.inBuffers.size();
            for (size_t s = 0; s < inN; ++s) {
                bumpRecordSize(templateIterOpInfo.inBuffers[s], iterationOpInfo.inBuffers[s]);
            }
            VPUX_THROW_UNLESS(iterationOpInfo.outBuffers.size() == templateIterOpInfo.outBuffers.size(),
                              "Iteration {0}, op {1}: out-buffer count {2} does not match template's {3}", iteration,
                              opPos, iterationOpInfo.outBuffers.size(), templateIterOpInfo.outBuffers.size());
            const size_t outN = templateIterOpInfo.outBuffers.size();
            for (size_t s = 0; s < outN; ++s) {
                bumpRecordSize(templateIterOpInfo.outBuffers[s], iterationOpInfo.outBuffers[s]);
            }
        }
    }
}

// Compute one conservative live range per buffer in the template iteration.
// Each input or output use extends the range to [firstUse, lastUse]. If the
// buffer is used in separate chunks, the gap between them is kept live instead
// of modeling multiple smaller ranges.
void computeLivenessSpan(const LoopBody& templateBody, ClassifierVFResult& result) {
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& info = templateBody[opPos];
        auto updateLiveInterval = [&](mlir::Value buf) {
            auto it = result.valueToRecord.find(buf);
            if (it == result.valueToRecord.end()) {
                return;  // appears only in non-template iterations
            }
            auto& bufferRecord = result.buffers[it->second];
            bufferRecord.firstUseOp = std::min(bufferRecord.firstUseOp, opPos);
            bufferRecord.lastUseOp = std::max(bufferRecord.lastUseOp, opPos);
        };
        for (auto buf : info.inBuffers) {
            updateLiveInterval(buf);
        }
        for (auto buf : info.outBuffers) {
            updateLiveInterval(buf);
        }
    }
}

// Classification.
//
// Phase 1 derives the class purely from buffer-flow within the region (no Operation*
// available on OpAllocationInfo).
//   - Produced never consumed => PERSISTENT_CANDIDATE: produced by every iteration's
//                       template op (producerIterCount == numIterations) and not
//                       consumed inside the loop. Shared output leaving the
//                       region (e.g. accumulator sink); Partially-shared outputs
//                       are out of scope for initial phase.
//   - External consumed  => PERSISTENT_CANDIDATE: no in-loop producer AND consumed
//                       by every iteration (consumerIterCount == numIterations).
//                       Not loaded by any DATA_IN DMA, so there is no per-iteration
//                       reload path. Partially-shared inputs are out of scope for initial phase.
//   - Otherwise      => TEMPORARY.
//   - isPrefetchableWeight: PERSISTENT_CANDIDATE whose every in-loop producer (if any)
//                       is DATA_IN, OR which has no in-loop producer at all (external).
//
// When iteration-identity is broken we cannot rely on cross-iteration semantics, so
// every record stays TEMPORARY.
void classifyBuffers(const vpux::SchedulingLoop& loop, ClassifierVFResult& result, const Logger& log) {
    if (!result.iterationIdentityHolds) {
        log.trace("classifyBuffers: iteration-identity broken - leaving all {0} buffers as TEMPORARY",
                  result.buffers.size());
        return;
    }

    const size_t numBuffers = result.buffers.size();
    SmallVector<size_t> consumerIterCount(numBuffers, 0);
    SmallVector<size_t> producerIterCount(numBuffers, 0);
    SmallVector<bool> producerOnlyDataIn(numBuffers, true);
    SmallVector<bool> hasAnyProducer(numBuffers, false);

    llvm::DenseSet<size_t> consumedThisIter;
    llvm::DenseSet<size_t> producedThisIter;
    for (const auto& body : loop.loopBodies) {
        consumedThisIter.clear();
        producedThisIter.clear();
        for (const auto& info : body) {
            for (auto buf : info.inBuffers) {
                auto it = result.valueToRecord.find(buf);
                if (it == result.valueToRecord.end()) {
                    continue;  // non-template SSA value, no record
                }
                consumedThisIter.insert(it->second);
            }
            for (auto buf : info.outBuffers) {
                auto it = result.valueToRecord.find(buf);
                if (it == result.valueToRecord.end()) {
                    continue;  // non-template SSA value, no record
                }
                const auto idx = it->second;
                hasAnyProducer[idx] = true;
                producedThisIter.insert(idx);
                if (info.allocationType != AllocationType::DATA_IN) {
                    producerOnlyDataIn[idx] = false;
                }
            }
        }
        for (size_t idx : consumedThisIter) {
            ++consumerIterCount[idx];
        }
        for (size_t idx : producedThisIter) {
            ++producerIterCount[idx];
        }
    }

    const size_t numIterations = loop.loopBodies.size();
    for (size_t i = 0; i < numBuffers; ++i) {
        // Detect partial-external buffers: buffers that structurally look like an
        // external input or output but are not used by every iteration. The classifier
        // cannot model this soundly today (each iteration would need its own persistent
        // slot with distinct SSA, breaking the shared-buffer contract). Flag the region
        // as unsupported and let downstream stages fall back to a safe path.
        //
        //   - External input, partial:  no in-loop producer AND 0 < consumerIterCount < N
        //   - External output, partial: has producer(s) AND 0 == consumerIterCount AND
        //                               0 < producerIterCount < N
        if (numIterations >= 2) {
            const bool partialExternalInput =
                    !hasAnyProducer[i] && consumerIterCount[i] > 0 && consumerIterCount[i] < numIterations;
            const bool partialExternalOutput = hasAnyProducer[i] && consumerIterCount[i] == 0 &&
                                               producerIterCount[i] > 0 && producerIterCount[i] < numIterations;
            if (partialExternalInput || partialExternalOutput) {
                result.classifierSupportedRegion = false;
                log.trace("classifyBuffers: unsupported partial-external buffer buf[{0}] "
                          "(kind={1}, producers={2}/{3}, consumers={4}/{5}, size={6}, root value={7}). "
                          "Marking region as classifier-unsupported.",
                          i, partialExternalInput ? "external-input" : "external-output", producerIterCount[i],
                          numIterations, consumerIterCount[i], numIterations, result.buffers[i].size,
                          result.buffers[i].rootValue);
            }
        }

        // Persistent externals require the loop to actually iterate (>=2 bodies)
        // and the SSA identity to be shared across every iteration — under
        // iteration-identity, a template-body SSA that appears in every iter is
        // truly shared, whereas one that appears in only body0 represents a
        // per-iteration slot with its own SSA per iter (must stay TEMPORARY).
        //
        // Produced inside the loop by every iteration and never consumed inside:
        // a shared output leaving the region (e.g. accumulator sink). Mark as PERSISTENT_CANDIDATE.
        const bool producedNeverConsumed =
                numIterations >= 2 && producerIterCount[i] == numIterations && consumerIterCount[i] == 0;
        // Consumed by every iteration and never produced by any in-loop DATA_IN
        // DMA (no in-loop producer at all). Truly shared external input with no
        // per-iteration reload path — must stay persistent for the whole loop.
        const bool externalConsumed = numIterations >= 2 && !hasAnyProducer[i] && consumerIterCount[i] == numIterations;
        if (producedNeverConsumed || externalConsumed) {
            result.buffers[i].category = BufferCategory::PERSISTENT_CANDIDATE;
            // Prefetchable when externally produced (no in-loop producer) or every in-loop
            // producer is DATA_IN.
            result.buffers[i].isPrefetchableWeight = !hasAnyProducer[i] || producerOnlyDataIn[i];
            if (producedNeverConsumed) {
                log.trace("buf[{0}] PERSISTENT_CANDIDATE: produced by all {1} iters and never consumed "
                          "(shared loop output; size={2}, prefetchableWeight={3}), root value={4})",
                          i, numIterations, result.buffers[i].size, result.buffers[i].isPrefetchableWeight,
                          result.buffers[i].rootValue);
            } else {
                log.trace("buf[{0}] PERSISTENT_CANDIDATE: no in-loop producer, consumed by all {1} iters "
                          "(shared external input, not loaded by DATA_IN; size={2}, prefetchableWeight={3}), "
                          "root value={4})",
                          i, numIterations, result.buffers[i].size, result.buffers[i].isPrefetchableWeight,
                          result.buffers[i].rootValue);
            }
        } else if (hasAnyProducer[i] && producerOnlyDataIn[i]) {
            // TEMPORARY whose every producer is DATA_IN a per-iteration prefetched tile
            // (typical for tiled VF regions where each iteration DMAs in its own weight
            // slice).
            result.buffers[i].isPrefetchableWeight = true;
            log.trace("buf[{0}] TEMPORARY (prefetchable): in-loop producer(s) all DATA_IN, consumed in "
                      "{1}/{2} iters (size={3}), root value={4})",
                      i, consumerIterCount[i], numIterations, result.buffers[i].size, result.buffers[i].rootValue);
        } else {
            std::string reason = "default";
            if (hasAnyProducer[i] && !producerOnlyDataIn[i]) {
                reason = "produced in-loop by a non-DATA_IN op";
            } else if (!hasAnyProducer[i] && consumerIterCount[i] < numIterations) {
                reason = "external but only partially shared (not consumed by all iters)";
            }
            log.trace("buf[{0}] TEMPORARY: {1} (hasProducer={2}, producersOnlyDataIn={3}, consumers={4}/{5}, "
                      "size={6}), root value={7})",
                      i, reason, hasAnyProducer[i], producerOnlyDataIn[i], consumerIterCount[i], numIterations,
                      result.buffers[i].size, result.buffers[i].rootValue);
        }
    }
}

// Scan template COMPUTE ops and find the largest per-op footprint.
//
// Per COMPUTE op footprint is defined as:
//   sum(aligned inBuffers) + sum(aligned outBuffers).
//
// Return value:
// - Sum of aligned sizes of PERSISTENT_CANDIDATE input buffers for the op that
//   established loopBodyNoSpillPeakBytes. This is used to avoid double-counting
//   those weights when computing peakBaselineBytes.
vpux::AddressType computeLargestComputeOpPeakAndWeights(const LoopBody& templateBody,
                                                        const SmallVector<vpux::AddressType>& alignedBufferSizes,
                                                        ClassifierVFResult& result) {
    vpux::AddressType largestComputeOpWeights = 0;
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& info = templateBody[opPos];
        if (info.allocationType != AllocationType::COMPUTE) {
            continue;
        }

        vpux::AddressType opPeakMemory = 0;
        vpux::AddressType opWeights = 0;

        for (auto buf : info.inBuffers) {
            auto it = result.valueToRecord.find(buf);
            if (it != result.valueToRecord.end()) {
                const auto idx = it->second;
                const auto& bufferRecord = result.buffers[idx];
                opPeakMemory += alignedBufferSizes[idx];
                if (bufferRecord.category == BufferCategory::PERSISTENT_CANDIDATE) {
                    opWeights += alignedBufferSizes[idx];
                }
            }
        }

        for (auto buf : info.outBuffers) {
            auto it = result.valueToRecord.find(buf);
            if (it != result.valueToRecord.end()) {
                opPeakMemory += alignedBufferSizes[it->second];
            }
        }

        if (opPeakMemory > result.loopBodyNoSpillPeakBytes) {
            result.loopBodyNoSpillPeakBytes = opPeakMemory;
            largestComputeOpWeights = opWeights;
        }
    }
    return largestComputeOpWeights;
}

// Compute the maximum live TEMPORARY footprint across template-op positions.
//
// At each op position, sum aligned sizes of buffers classified as TEMPORARY
// whose conservative liveness interval [firstUseOp, lastUseOp] covers that
// position. The largest such sum becomes temporaryPeakBytes.
void computeTemporaryPeak(const LoopBody& templateBody, const SmallVector<vpux::AddressType>& alignedBufferSizes,
                          ClassifierVFResult& result) {
    const auto noTemplateUse = std::numeric_limits<size_t>::max();
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        vpux::AddressType liveTemp = 0;
        for (size_t idx = 0; idx < result.buffers.size(); ++idx) {
            const auto& bufferRecord = result.buffers[idx];
            if (bufferRecord.firstUseOp == noTemplateUse) {
                continue;
            }
            const bool aliveTemp = (bufferRecord.category == BufferCategory::TEMPORARY) &&
                                   (opPos >= bufferRecord.firstUseOp) && (opPos <= bufferRecord.lastUseOp);
            if (aliveTemp) {
                liveTemp += alignedBufferSizes[idx];
            }
        }
        result.temporaryPeakBytes = std::max(result.temporaryPeakBytes, liveTemp);
    }
}

void computeRegionBoundaryBytes(const LoopBody& templateBody, const SmallVector<vpux::AddressType>& alignedBufferSizes,
                                ClassifierVFResult& result) {
    llvm::DenseSet<size_t> regionInputBuffs;
    llvm::DenseSet<size_t> regionOutputBuffs;

    for (const auto& info : templateBody) {
        if (info.allocationType == AllocationType::DATA_IN) {
            for (auto buf : info.outBuffers) {
                auto it = result.valueToRecord.find(buf);
                if (it != result.valueToRecord.end()) {
                    regionInputBuffs.insert(it->second);
                }
            }
        } else if (info.allocationType == AllocationType::DATA_OUT) {
            for (auto buf : info.inBuffers) {
                auto it = result.valueToRecord.find(buf);
                if (it != result.valueToRecord.end()) {
                    regionOutputBuffs.insert(it->second);
                }
            }
        }
    }

    for (size_t idx : regionInputBuffs) {
        result.inputBytes += alignedBufferSizes[idx];
    }
    for (size_t idx : regionOutputBuffs) {
        result.outputBytes += alignedBufferSizes[idx];
    }
}

// Compute VF memory statistics needed before persistent-fit ordering.
//
// Produces aligned buffer sizes and updates in `result`:
// - persistentCandidateTotalBytes
// - loopBodyNoSpillPeakBytes
// - temporaryPeakBytes
// - peakBaselineBytes
// - inputBytes / outputBytes (region boundary bytes)
SmallVector<vpux::AddressType> computeMemoryStats(const LoopBody& templateBody, ClassifierVFResult& result) {
    SmallVector<vpux::AddressType> alignedBufferSizes(result.buffers.size());
    for (size_t idx = 0; idx < result.buffers.size(); ++idx) {
        const auto& bufferRecord = result.buffers[idx];
        alignedBufferSizes[idx] = vpux::alignValUp(bufferRecord.size, bufferRecord.alignment);
    }

    for (size_t idx = 0; idx < result.buffers.size(); ++idx) {
        const auto& bufferRecord = result.buffers[idx];
        if (bufferRecord.category == BufferCategory::PERSISTENT_CANDIDATE) {
            result.persistentCandidateTotalBytes += alignedBufferSizes[idx];
        }
    }

    // Note: we may have some branches where peak memory could be more than the memory used by largest op.
    // For simplicity we use the largest compute op to approximate the loop-body peak.
    const auto largestComputeOpWeights =
            computeLargestComputeOpPeakAndWeights(templateBody, alignedBufferSizes, result);
    computeTemporaryPeak(templateBody, alignedBufferSizes, result);

    // peakBaselineBytes = loopBodyNoSpillPeakBytes + (persistentCandidateTotalBytes - largestComputeOpWeights)
    result.peakBaselineBytes = result.loopBodyNoSpillPeakBytes +
                               saturatingSub(result.persistentCandidateTotalBytes, largestComputeOpWeights);

    computeRegionBoundaryBytes(templateBody, alignedBufferSizes, result);
    return alignedBufferSizes;
}

// Estimate how many persistent buffers can fit in the remaining memory.
//
// persistentFitBudgetBytes = memoryLimit - temporaryPeakBytes.
//
// Sort persistent candidates by size.
// Larger size comes first. Ties: earlier firstUseOp.
//
// Add candidates in that order while they fit in the budget.
// If one candidate does not fit, continue checking the rest so smaller
// ones can still be admitted.
void addPersistentCandidatesInFittingOrder(const LoopBody& templateBody, vpux::AddressType memoryLimit,
                                           ClassifierVFResult& result, Logger& log) {
    const auto alignedBufferSizes = computeMemoryStats(templateBody, result);

    for (size_t i = 0; i < result.buffers.size(); ++i) {
        if (result.buffers[i].category == BufferCategory::PERSISTENT_CANDIDATE) {
            result.persistentFitOrder.push_back(i);
        }
    }
    std::sort(result.persistentFitOrder.begin(), result.persistentFitOrder.end(), [&](size_t a, size_t b) {
        const auto& ra = result.buffers[a];
        const auto& rb = result.buffers[b];
        const auto alignedA = alignedBufferSizes[a];
        const auto alignedB = alignedBufferSizes[b];
        if (alignedA != alignedB) {
            return alignedA > alignedB;
        }
        return ra.firstUseOp < rb.firstUseOp;
    });

    result.persistentFitBudgetBytes = saturatingSub(memoryLimit, result.temporaryPeakBytes);
    SmallVector<PersistentFitBufferInfo> persistentFitBufferInfo(result.buffers.size());
    for (size_t idx = 0; idx < result.buffers.size(); ++idx) {
        persistentFitBufferInfo[idx] = {result.buffers[idx].size, result.buffers[idx].alignment};
    }
    const auto usePersistentFitSearch =
            shouldUsePersistentFitSearch(result.persistentFitOrder, persistentFitBufferInfo);
    if (usePersistentFitSearch) {
        result.persistentFitOrder = getWorstCasePersistentFitOrder(result.persistentFitOrder, persistentFitBufferInfo);
        const auto requiredBytes = getPackedPersistentSize(result.persistentFitOrder, persistentFitBufferInfo);
        if (requiredBytes > result.persistentFitBudgetBytes) {
            log.warning(
                    "VF persistent candidates do not fit in the worst mixed-alignment order: required={0}, budget={1}",
                    requiredBytes, result.persistentFitBudgetBytes);
        }
    }

    log.trace("persistentFit: budget={0} bytes (cmxLimit={1} - tempPeak={2}), {3} candidates, mode={4}",
              result.persistentFitBudgetBytes, memoryLimit, result.temporaryPeakBytes, result.persistentFitOrder.size(),
              usePersistentFitSearch ? "worst-case-search" : "greedy");
    for (size_t idx : result.persistentFitOrder) {
        const auto& bufferRecord = result.buffers[idx];
        const auto candidateEnd =
                usePersistentFitSearch ? vpux::alignValUp(result.persistentFitReservedBytes, bufferRecord.alignment) +
                                                 bufferRecord.size
                                       : result.persistentFitReservedBytes + alignedBufferSizes[idx];
        if (candidateEnd <= result.persistentFitBudgetBytes) {
            result.persistentFitInitial.push_back(idx);
            result.persistentFitReservedBytes = candidateEnd;
            log.trace("  buf[{0}] ADMITTED: aligned={1}, reservedAfter={2}/{3}, root value={4}", idx,
                      alignedBufferSizes[idx], result.persistentFitReservedBytes, result.persistentFitBudgetBytes,
                      bufferRecord.rootValue);
        } else {
            log.trace("  buf[{0}] REJECTED: aligned={1} would push reserved to {2} > budget {3}, root value={4}", idx,
                      alignedBufferSizes[idx], candidateEnd, result.persistentFitBudgetBytes, bufferRecord.rootValue);
            // Early return as currently there is no support for not accepting PERSISTENT buffer as is.
            // Support will be extended as part of E#222690
            result.classifierSupportedRegion = false;
            return;
        }
    }
}

}  // namespace

namespace vpux {

ClassifierVFResult classifyVFRegion(const ComputeRegion& region, vpux::AddressType memoryLimit, Logger& log) {
    VPUX_THROW_UNLESS(region.schedulingLoop != nullptr, "ComputeRegion has no scheduling loop");
    VPUX_THROW_UNLESS(region.schedulingLoop->type == LoopType::VF, "classifyVFRegion invoked on non-VF loop type: {0}",
                      toString(region.schedulingLoop->type));

    log.setName("undefined-vf-classifier");

    ClassifierVFResult result;
    result.cmxMemoryLimitBytes = memoryLimit;
    const auto& loop = *region.schedulingLoop;
    result.numIterations = loop.loopBodies.size();
    result.opsPerIteration = 0;

    if (loop.loopBodies.empty() || result.numIterations < 2) {
        log.trace("VF region has no iteration bodies or less than 2 iterations - returning empty classification");
        result.classifierSupportedRegion = false;
        return result;
    }

    const auto& templateBody = loop.loopBodies.front();
    result.opsPerIteration = templateBody.size();
    if (result.opsPerIteration == 0) {
        log.trace("VF region has no ops in the template iteration - returning empty classification");
        result.classifierSupportedRegion = false;
        return result;
    }

    log.trace("VF region: iterations={0}, opsPerIteration={1}, cmxLimit={2}", result.numIterations,
              result.opsPerIteration, memoryLimit);

    // Check that all iterations have the same structure.
    //
    // If this check fails, we treat all buffers as TEMPORARY
    // (no PERSISTENT_CANDIDATE across iterations).
    result.iterationIdentityHolds = checkIterationIdentity(loop, log);
    // Non identity VF loops fall fast instead of silently disabling persistent
    if (!result.iterationIdentityHolds) {
        result.classifierSupportedRegion = false;
        log.trace("VF region: iteration-identity invariant violated, unsupported non-isomorphic iteration "
                  "bodies");
        return result;
    }

    collectBuffers(loop, result);
    log.trace("collectBuffers: collected {0} unique template-buffer records", result.buffers.size());

    computeLivenessSpan(templateBody, result);

    classifyBuffers(loop, result, log);

    // Early return if classifier detected an unsupported structural pattern (see
    // classifierSupportedRegion). Downstream stages depend on the classifier being
    // able to model the region soundly; continuing here would produce invalid
    // aggregates and admission decisions.
    if (!result.classifierSupportedRegion) {
        log.trace("Classifier reported unsupported region, skipping persistent-fit estimate");
        return result;
    }

    addPersistentCandidatesInFittingOrder(templateBody, memoryLimit, result, log);
    if (!result.classifierSupportedRegion) {
        log.trace("Not able to fit persistent candidates in the budget");
        return result;
    }

    result.prefetchHint = getBestPrefetchHint(result);

    log.trace("VF region classified: buffers={0}, persistents={1}, peak={2}, tempPeak={3}, "
              "fitReserved={4}, identity={5}, loopNoSpill={6}, inBytes={7}, outBytes={8}, hint={9}",
              result.buffers.size(), result.persistentFitOrder.size(), result.peakBaselineBytes,
              result.temporaryPeakBytes, result.persistentFitReservedBytes, result.iterationIdentityHolds,
              result.loopBodyNoSpillPeakBytes, result.inputBytes, result.outputBytes,
              stringifyPrefetchHint(result.prefetchHint));

    return result;
}

llvm::StringRef stringifyBufferCategory(BufferCategory category) {
    switch (category) {
    case BufferCategory::PERSISTENT_CANDIDATE:
        return "PERSISTENT_CANDIDATE";
    case BufferCategory::TEMPORARY:
        return "TEMPORARY";
    }
    return "UNKNOWN";
}

llvm::StringRef stringifyPrefetchHint(VFPrefetchHint hint) {
    switch (hint) {
    case VFPrefetchHint::FULL_PREFETCHING:
        return "FULL_PREFETCHING";
    case VFPrefetchHint::LASTOP_PREFETCHING:
        return "LASTOP_PREFETCHING";
    case VFPrefetchHint::WEIGHTS_PREFETCHING:
        return "WEIGHTS_PREFETCHING";
    case VFPrefetchHint::MINIMAL:
        return "MINIMAL";
    }
    return "UNKNOWN";
}

void ClassifierVFResult::print(llvm::raw_ostream& os) const {
    const auto persistentCount = llvm::count_if(buffers, [](const auto& b) {
        return b.category == BufferCategory::PERSISTENT_CANDIDATE;
    });

    os << "ClassifierVFResult summary:\n";
    os << "  iterations=" << numIterations << ", opsPerIteration=" << opsPerIteration
       << ", identity=" << (iterationIdentityHolds ? "true" : "false") << "\n";
    os << "  buffers=" << buffers.size() << " (persistents=" << persistentCount
       << ", temporaries=" << (buffers.size() - persistentCount) << ")\n";
    os << "  peakBaselineBytes=" << peakBaselineBytes << ", temporaryPeakBytes=" << temporaryPeakBytes
       << ", persistentCandidateTotalBytes=" << persistentCandidateTotalBytes << "\n";
    os << "  memoryBreakdown: loopBodyNoSpillPeak=" << loopBodyNoSpillPeakBytes << ", inputBytes=" << inputBytes
       << ", outputBytes=" << outputBytes << "\n";
    os << "  cmxLimit=" << cmxMemoryLimitBytes << ", hint=" << stringifyPrefetchHint(prefetchHint) << "\n";
    os << "  persistentFit: reserved=" << persistentFitReservedBytes << ", budget=" << persistentFitBudgetBytes
       << ", initialAdmitted=" << persistentFitInitial.size() << "/" << persistentFitOrder.size() << "\n";
    os << "  per-buffer records:\n";
    for (size_t i = 0; i < buffers.size(); ++i) {
        const auto& b = buffers[i];
        os << "    [" << i << "] category=" << stringifyBufferCategory(b.category) << " size=" << b.size
           << " align=" << b.alignment << " firstUse=" << b.firstUseOp << " lastUse=" << b.lastUseOp
           << " prefetchableWeight=" << (b.isPrefetchableWeight ? "true" : "false")
           << " inPlace=" << (b.isInPlaceAlias ? "true" : "false") << "\n";
    }
}

std::string ClassifierVFResult::summarize() const {
    return llvm::formatv("ops/iter={0} iters={1} buffers={2} (persistents={3}/{4}) "
                         "tempPeak={5} persistTotal={6} peakBaseline={7} "
                         "loopNoSpillPeak={8} inBytes={9} outBytes={10} "
                         "persistFitBudget={11} hint={12}",
                         opsPerIteration, numIterations, buffers.size(), persistentFitInitial.size(),
                         persistentFitOrder.size(), temporaryPeakBytes, persistentCandidateTotalBytes,
                         peakBaselineBytes, loopBodyNoSpillPeakBytes, inputBytes, outputBytes, persistentFitBudgetBytes,
                         stringifyPrefetchHint(prefetchHint))
            .str();
}

}  // namespace vpux
