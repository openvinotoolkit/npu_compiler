//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_vf.hpp"
#include "vpux/compiler/core/scheduling/classifier_vf.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_utils.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPU {

namespace {

// Walk over strategies in priority order and return first feasible
// The MINIMAL acts as a fallback and should be always feasible
// if VF tiling was done sufficiently with respect to memory constraints
constexpr VfStrategyRecipe vfStrategyRecipes[] = {
        {VPU::VFScenario::VF_PIPELINING, VfSchedStrategyDescriptor::IntraIterFetch::ENABLED,
         VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::DISABLED},
        {VPU::VFScenario::LASTOP_PREFETCHING, VfSchedStrategyDescriptor::IntraIterFetch::ENABLED,
         VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED},
        // Minimal is last so any earlier feasible recipe short-circuits the search.
        {VPU::VFScenario::MINIMAL, VfSchedStrategyDescriptor::IntraIterFetch::ENABLED,
         VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::DISABLED},
};

// Create a map which associates each buffer value in the template body with
// its position (opPos, slot, isInput) in the template body.
// This will be later used to find the corresponding buffer value of other iteration bodies
// when replicating the template schedule across iterations.
// Outputs are recorded first so a producer's position wins over a consumer's; input entries
// act as a fallback for buffers that have no in-loop producer (e.g. a persistent candidate
// that got downgraded to temporary and thus appears only in inBuffers of consumer ops).
struct BufferPosition {
    size_t opPos;
    size_t slot;
    bool isInput;
};
llvm::DenseMap<mlir::Value, BufferPosition> buildBufferPositionMap(const LoopBody& templateBody) {
    llvm::DenseMap<mlir::Value, BufferPosition> map;
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& outBufs = templateBody[opPos].outBuffers;
        for (size_t slot = 0; slot < outBufs.size(); ++slot) {
            // First-seen position wins — duplicates would imply structural
            // in-place aliasing; they map to their original allocation site.
            map.insert({outBufs[slot], BufferPosition{opPos, slot, /*isInput=*/false}});
        }
    }
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& inBufs = templateBody[opPos].inBuffers;
        for (size_t slot = 0; slot < inBufs.size(); ++slot) {
            // Fallback only — do not overwrite a producer position recorded above.
            map.insert({inBufs[slot], BufferPosition{opPos, slot, /*isInput=*/true}});
        }
    }
    return map;
}

// Substitute one template value with its iteration-k counterpart.
// `posMap` is built from loopBodies[0]; `bodyK` is loopBodies[k].
mlir::Value findIterationBufferMatchingTemplateBuffer(mlir::Value templateValue,
                                                      const llvm::DenseMap<mlir::Value, BufferPosition>& posMap,
                                                      const LoopBody& bodyK) {
    const auto it = posMap.find(templateValue);
    VPUX_THROW_WHEN(it == posMap.end(),
                    "Template schedule references value {0} not produced or consumed by any template op. All "
                    "buffers are expected to appear in inBuffers or outBuffers of loop body ops",
                    templateValue);
    const auto [opPos, slot, isInput] = it->second;
    VPUX_THROW_UNLESS(opPos < bodyK.size(),
                      "Iteration body shape diverges from template at op {0} slot {1} (isInput={2})", opPos, slot,
                      isInput);
    const auto& bufList = isInput ? bodyK[opPos].inBuffers : bodyK[opPos].outBuffers;
    VPUX_THROW_UNLESS(slot < bufList.size(),
                      "Iteration body shape diverges from template at op {0} slot {1} (isInput={2})", opPos, slot,
                      isInput);
    return bufList[slot];
}

// Replicate the single iteration template emitted by VfAllocateLinear across
// every iteration body in the region. Use same offsets as in template body but use
// given iteration buffers
PredefinedSchedule replicateTemplateScheduleAcrossAllLoopIterations(const VfAllocateResult& result,
                                                                    const SchedulingLoop& loop,
                                                                    AddressType oddIterationBankOffset) {
    PredefinedSchedule out;
    if (loop.loopBodies.empty() || result.iterationSchedule.empty()) {
        return out;
    }

    const auto& templateBody = loop.loopBodies.front();
    VPUX_THROW_UNLESS(result.iterationSchedule.size() == templateBody.size(),
                      "Template iteration schedule size {0} != template body size {1}", result.iterationSchedule.size(),
                      templateBody.size());

    const auto posMap = buildBufferPositionMap(templateBody);

    out.reserve(loop.loopBodies.size());
    for (size_t k = 0; k < loop.loopBodies.size(); ++k) {
        const auto& bodyK = loop.loopBodies[k];
        VPUX_THROW_UNLESS(bodyK.size() == templateBody.size(),
                          "Iteration {0} body size {1} != template body size {2} (iteration-identity "
                          "invariant violated)",
                          k, bodyK.size(), templateBody.size());

        IterationSchedule entryK;
        entryK.reserve(templateBody.size());
        for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
            const auto& templateEntry = result.iterationSchedule[opPos];

            SmallVector<mlir::Value> deallocs;
            deallocs.reserve(templateEntry.deallocations.size());
            for (auto v : templateEntry.deallocations) {
                deallocs.push_back(findIterationBufferMatchingTemplateBuffer(v, posMap, bodyK));
            }

            SmallVector<std::pair<mlir::Value, vpux::AddressType>> allocs;
            allocs.reserve(templateEntry.allocations.size());
            // Pipelining bank placement happens here: odd iterations are moved
            // to Bank B by +oddIterationBankOffset, even iterations stay in
            // Bank A. Shared externals are owned by outer scheduling and keep
            // their base addresses.
            const bool useUpperBank = oddIterationBankOffset > 0 && (k % 2 == 1);
            for (const auto& [v, addr] : templateEntry.allocations) {
                const auto mapped = findIterationBufferMatchingTemplateBuffer(v, posMap, bodyK);
                const bool isSharedExternal =
                        result.sharedExternalBuffers.find(mapped) != result.sharedExternalBuffers.end();
                const auto adjustedAddr = (useUpperBank && !isSharedExternal) ? (addr + oddIterationBankOffset) : addr;
                allocs.push_back({mapped, adjustedAddr});
            }

            // Note: the iteration-identity invariant promises positional
            // outBuffer-slot equality between iterations Both `allocInfo` and the substituted
            // `allocations` must come from the SAME `bodyK[templatePos]` so
            // the two halves agree on the per-iter SSA values they reference.
            const size_t srcPos = templateEntry.templatePos;
            VPUX_THROW_UNLESS(srcPos < bodyK.size(),
                              "TemplatePos {0} out of range (bodyK size {1}) at iter {2} opPos {3}", srcPos,
                              bodyK.size(), k, opPos);
            const auto& srcAlloc = bodyK[srcPos];

            entryK.emplace_back(/*allocInfo=*/srcAlloc, std::move(deallocs), std::move(allocs));
        }
        out.push_back(std::move(entryK));
    }
    return out;
}

LoopScheduleResult buildLoopScheduleResult(const VfAllocateResult& result, const ComputeRegion& region,
                                           const SearchOutcome& outcome, const Logger& log) {
    LoopScheduleResult out;

    // Derive baseAlignment from the strictest operand alignment requirement up front: oddIterationBankOffset
    // below must be rounded up to it, otherwise addr + oddIterationBankOffset can violate a buffer's own
    // alignment even though addr itself was correctly aligned.
    vpux::AddressType baseAlignment = vpux::DEFAULT_CMX_ALIGNMENT;
    for (const auto& entry : result.iterationSchedule) {
        for (const auto& [v, _] : entry.allocations) {
            const auto bufAlign = vpux::getAlignment(v);
            baseAlignment = baseAlignment < bufAlign ? bufAlign : baseAlignment;
        }
    }

    AddressType oddIterationBankOffset = 0;
    auto selectedVFStrategy = outcome.selectedVFStrategy;
    if (selectedVFStrategy.has_value() && selectedVFStrategy.value() == VPU::VFScenario::VF_PIPELINING) {
        // For pipelining, tryStrategy() stores doubled footprint as
        // persistent + 2 * bankSize. Convert that back to bankSize so
        // replication can place odd iterations into the upper bank.
        VPUX_THROW_UNLESS(result.peakUsedBytes >= result.persistentReservedBytes,
                          "UndefinedVF pipelining invariant violated: peakUsedBytes {0} < persistentReservedBytes {1}",
                          result.peakUsedBytes, result.persistentReservedBytes);
        const AddressType volatileBytes = result.peakUsedBytes - result.persistentReservedBytes;
        VPUX_THROW_UNLESS((volatileBytes % 2) == 0,
                          "UndefinedVF pipelining invariant violated: volatile footprint {0} must be even",
                          volatileBytes);
        // bankSize (volatileBytes / 2) is a raw high-water mark, not aligned to any boundary; round the
        // offset up so odd-iteration buffers keep their own alignment after the shift.
        oddIterationBankOffset = vpux::alignValUp(volatileBytes / 2, baseAlignment);
    }

    out.schedule =
            replicateTemplateScheduleAcrossAllLoopIterations(result, *region.schedulingLoop, oddIterationBankOffset);
    if (selectedVFStrategy.has_value() && selectedVFStrategy.value() == VPU::VFScenario::VF_PIPELINING) {
        log.trace("UndefinedVF: VF_PIPELINING selected — applying pairwise hoisting rewrite "
                  "(oddIterationBankOffset={0}, iterations={1})",
                  oddIterationBankOffset, out.schedule.size());
        log.trace("UndefinedVF: pipelining feasible and applied for the first VF region");
        // A successful hoist moves stages between adjacent iterations, so the
        // rewritten pairwise schedule already enables iteration overlap. Keep
        // the regular overlap path enabled only when no hoisting was applied.
        out.enableIterationOverlap = !rewriteVFPipelinedScheduleIntoPairwiseSteps(out.schedule);
        // Note: dumpScheduleASCII can be used for visualization as below
        // dumpScheduleASCII(out.schedule, log, "after pairwise rewrite", regionIdx);
    }

    // `reservedSize` is the local working-area size the outer scheduler must
    // carve out for the loop body — i.e. the bytes occupied by per-iteration
    // TEMPORARY allocations. Confirmed persistents flow separately through
    // `sharedExternalBuffers` (allocated contiguously by the outer scheduler),
    // so the persistent zone bytes must be excluded here. Mirrors
    // `UndefinedTiling::usedMemory` semantics consumed at
    // feasible_memory_scheduler.cpp `prepareLoopRegion`.
    VPUX_THROW_UNLESS(result.peakUsedBytes >= result.persistentReservedBytes,
                      "UndefinedVF: peakUsedBytes {0} < persistentReservedBytes {1} (allocator invariant violated)",
                      result.peakUsedBytes, result.persistentReservedBytes);
    out.reservedSize = result.peakUsedBytes - result.persistentReservedBytes;
    out.sharedExternalBuffers = result.sharedExternalBuffers;
    for (auto shared : region.externalBuffers) {
        out.sharedExternalBuffers.insert(shared);
    }

    out.baseAlignment = baseAlignment;
    return out;
}

void verifyLocalSchedule(const LoopScheduleResult& scheduleResult) {
    struct LocalRange {
        vpux::AddressType offset = 0;
        vpux::AddressType size = 0;
        size_t producerOp = 0;
    };

    for (size_t iteration = 0; iteration < scheduleResult.schedule.size(); ++iteration) {
        llvm::DenseMap<mlir::Value, LocalRange> liveLocalRanges;
        const auto& iterationSchedule = scheduleResult.schedule[iteration];
        for (const auto& entry : iterationSchedule) {
            for (const auto& dealloc : entry.deallocations) {
                liveLocalRanges.erase(dealloc);
            }

            for (const auto& [buffer, offset] : entry.allocations) {
                const auto size = checked_cast<vpux::AddressType>(getTotalSize(buffer).count());
                for (const auto& [liveBuffer, liveRange] : liveLocalRanges) {
                    VPUX_THROW_WHEN(Partitioner::intersects(offset, size, liveRange.offset, liveRange.size),
                                    "UndefinedVF verifier: local offset overlap in iteration {0}: opIdx={1} "
                                    "allocates buffer {2} at [{3}, {4}) but live buffer {5} from opIdx={6} occupies "
                                    "[{7}, {8})",
                                    iteration, entry.allocInfo.opIdx, buffer, offset, offset + size, liveBuffer,
                                    liveRange.producerOp, liveRange.offset, liveRange.offset + liveRange.size);
                }
                liveLocalRanges[buffer] = {offset, size, entry.allocInfo.opIdx};
            }

            for (auto outBuffer : entry.allocInfo.outBuffers) {
                const auto hasLocalRange = liveLocalRanges.find(outBuffer) != liveLocalRanges.end();
                const auto isSharedExternal = scheduleResult.sharedExternalBuffers.find(outBuffer) !=
                                              scheduleResult.sharedExternalBuffers.end();
                VPUX_THROW_UNLESS(hasLocalRange || isSharedExternal,
                                  "UndefinedVF verifier: output buffer {0} produced by opIdx={1} in iteration {2} has "
                                  "no local allocation and is not listed as a shared external buffer",
                                  outBuffer, entry.allocInfo.opIdx, iteration);
            }
        }
    }
}

// Run allocator for given strategy recipe for VF compute region
// Create IterationSchedule if feasible, otherwise return infeasible result
VfAllocateResult tryStrategy(const VfStrategyRecipe& recipe, const ComputeRegion& region,
                             const ClassifierVFResult& classifier, AddressType memoryLimit, Logger log) {
    VfSchedStrategyDescriptor params{recipe.intraIterFetch, recipe.lastCompOverlapNextIterIn};
    if (recipe.id == VPU::VFScenario::VF_PIPELINING) {
        // Analytical feasibility gate for double-buffered VF pipelining:
        //   2*largestOpLocalBytes + persistentCandidateBytes <= memoryLimit
        // largestOpLocalBytes is the largest COMPUTE op's own (non-shared) footprint;
        // persistentCandidateBytes is the total size of buffers invariant across iterations.
        // Both values map onto classifier fields already computed
        // for that scenario:
        //   persistentCandidateBytes = persistentCandidateTotalBytes
        //   largestOpLocalBytes      = peakBaselineBytes - persistentCandidateBytes (peakBaselineBytes
        //             already excludes the largest op's own persistent candidates from
        //             persistentCandidateTotalBytes
        //             to avoid double counting; see computeMemoryStats() in classifier_vf.cpp)
        // Reserved space is already reflected in memoryLimit.
        const auto persistentCandidateBytes = classifier.persistentCandidateTotalBytes;
        const auto largestOpLocalBytes = saturatingSub(classifier.peakBaselineBytes, persistentCandidateBytes);
        if (2 * largestOpLocalBytes + persistentCandidateBytes > memoryLimit) {
            log.debug("VF_PIPELINING infeasible by analytical budget: 2*largestOpLocalBytes({0}) + "
                      "persistentCandidateBytes({1}) > memoryLimit({2})",
                      largestOpLocalBytes, persistentCandidateBytes, memoryLimit);
            return VfAllocateResult{};
        }

        // Pipelining is MINIMAL allocation on one bank + bank alternation at replication.
        // The allocator only answers "does one iteration fit in the one-bank budget?". Cross-iteration
        // double-buffering is handled entirely by the bank offset applied in
        // buildLoopScheduleResult — not by the allocator params
        //
        // The allocator places persistent candidates (single-copy, ~persistentCandidateBytes) and volatile
        // (to-be-doubled) buffers inside the same memory-limit budget. Splitting memoryLimit in half ignores
        // persistentCandidateBytes and either under- or over-constrains
        // the volatile bank relative to the formula above. Derive the one-bank budget so that
        // persistent + bankSize <= oneBankBudget implies 2*bankSize + persistent <= memoryLimit once
        // doubled:
        //   oneBankBudget = (memoryLimit + persistentCandidateBytes) / 2
        const AddressType oneBankBudget = static_cast<AddressType>((memoryLimit + persistentCandidateBytes) / 2);
        auto oneBankResult = VfAllocateLinear(region, classifier, oneBankBudget, params, log).performAllocation();
        if (!oneBankResult.feasible) {
            return oneBankResult;
        }

        VPUX_THROW_UNLESS(oneBankResult.peakUsedBytes >= oneBankResult.persistentReservedBytes,
                          "UndefinedVF pipelining invariant violated: one-bank peak {0} < persistent {1}",
                          oneBankResult.peakUsedBytes, oneBankResult.persistentReservedBytes);
        // buildLoopScheduleResult rounds the odd-iteration bank offset up to the strictest buffer
        // alignment, so bankSize must be pre-aligned here too — otherwise the aligned shift exceeds
        // bankSize and odd-iteration offsets land outside the reserved footprint computed below.
        AddressType baseAlignment = vpux::DEFAULT_CMX_ALIGNMENT;
        for (const auto& entry : oneBankResult.iterationSchedule) {
            for (const auto& [v, _] : entry.allocations) {
                const auto bufAlign = vpux::getAlignment(v);
                baseAlignment = baseAlignment < bufAlign ? bufAlign : baseAlignment;
            }
        }
        // Convert one-bank peak to pipelined total footprint:
        // peak = persistent + 2 * bankSize.
        const AddressType bankSize =
                vpux::alignValUp(oneBankResult.peakUsedBytes - oneBankResult.persistentReservedBytes, baseAlignment);
        oneBankResult.peakUsedBytes = oneBankResult.persistentReservedBytes + 2 * bankSize;

        // Decompose the analytical-vs-real gap into its independently measurable sources — only
        // computed/logged when the doubled peak actually exceeds memoryLimit below:
        //   classifier error    = peakLiveTemporaryBytes - largestOpLocalBytes (estimate too low)
        //   scan fragmentation  = bankSize - peakLiveTemporaryBytes (address-space gaps in the real allocator)
        //     strandedPersistent = persistentCandidateBytes - oneBankResult.persistentReservedBytes: bytes admitted
        //       as persistent but evicted mid-schedule and never reclaimed by the bump allocator (see
        //       findAndSpillPersistentBuffer in vf_allocate_linear.cpp) — usually the dominant term.
        //     residualFragmentation = whatever scan fragmentation remains after removing stranded-persistent
        //       bytes — real address-order-dependent packing gaps (e.g. a small dead buffer's hole being too
        //       small to reuse for a later, larger one).
        //
        // The analytical gate above is an idealized-packing estimate; the real linear-scan
        // allocator can still exceed it (fragmentation from address-based placement, eviction
        // order, etc.). Fail soft into the next recipe instead of aborting compilation.
        if (oneBankResult.peakUsedBytes > memoryLimit) {
            if (log.isActive(LogLevel::Trace) && isDeveloperBuild()) {
                const AddressType scanFragmentation = saturatingSub(bankSize, oneBankResult.peakLiveTemporaryBytes);
                const AddressType strandedPersistentBytes =
                        saturatingSub(persistentCandidateBytes, oneBankResult.persistentReservedBytes);
                const AddressType analyticalTotal = 2 * largestOpLocalBytes + persistentCandidateBytes;
                log.trace("VF_PIPELINING peak breakdown: analytical largestOpLocalBytes={0}, real ideal-packing "
                          "peak={1} (classifier error={2}), real bankSize={3} (scan fragmentation={4} = "
                          "strandedPersistent{5} + residual{6})",
                          largestOpLocalBytes, oneBankResult.peakLiveTemporaryBytes,
                          saturatingSub(oneBankResult.peakLiveTemporaryBytes, largestOpLocalBytes), bankSize,
                          scanFragmentation, strandedPersistentBytes,
                          saturatingSub(scanFragmentation, strandedPersistentBytes));
                log.trace("VF_PIPELINING infeasible after allocation: doubled peak {0} exceeds memoryLimit {1}; "
                          "reconciled as analytical({2}) + 2*scanFragmentation({3}) - strandedPersistent({4}) = "
                          "{5}",
                          oneBankResult.peakUsedBytes, memoryLimit, analyticalTotal, scanFragmentation,
                          strandedPersistentBytes, analyticalTotal + 2 * scanFragmentation - strandedPersistentBytes);
            }
            log.debug("VF_PIPELINING infeasible after allocation: doubled peak {0} exceeds memoryLimit {1}",
                      oneBankResult.peakUsedBytes, memoryLimit);
            return VfAllocateResult{};
        }

        return oneBankResult;
    }

    return VfAllocateLinear(region, classifier, memoryLimit, params, log).performAllocation();
}

}  // namespace

UndefinedVF::UndefinedVF(): _log(Logger::global()) {
    _log.setName("UndefinedVF");
}

llvm::StringRef UndefinedVF::getName() const {
    return "UndefinedVF";
}

// Perform search for a feasible strategy and build a schedule for the given VF loop region and memory size
SearchOutcome UndefinedVF::runStrategySearch(const AllocatorFn& allocator) const {
    VPUX_THROW_UNLESS(static_cast<bool>(allocator), "Allocator callback is empty");
    // Walk recipes in priority order and pick the FIRST feasible one.
    // TODO: In future more sophisticated approach could be tried where multiple strategies
    // are evaluated and the best one from cost perspective is picked
    SearchOutcome out;
    for (const auto& recipe : vfStrategyRecipes) {
        _log.trace("UndefinedVF: trying strategy {0} in VF recipe priority order", stringifyVFScenario(recipe.id));
        out.result = allocator(recipe);
        if (out.result.feasible) {
            _log.trace("UndefinedVF: first feasible strategy selected: {0} (peakUsed={1})",
                       stringifyVFScenario(recipe.id), out.result.peakUsedBytes);
            out.selectedVFStrategy = recipe.id;
            return out;
        }

        _log.trace("UndefinedVF: strategy {0} is infeasible or unsupported (peakUsed={1})",
                   stringifyVFScenario(recipe.id), out.result.peakUsedBytes);
    }
    return out;
}

LoopScheduleResult UndefinedVF::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                    vpux::AddressType memorySize) const {
    VPUX_THROW_WHEN(loopRegion.schedulingLoop == nullptr, "ComputeRegion has no scheduling loop");
    VPUX_THROW_UNLESS(loopRegion.schedulingLoop->type == LoopType::VF, "UndefinedVF invoked on non-VF loop type: {0}",
                      toString(loopRegion.schedulingLoop->type));

    const auto numIterations = loopRegion.schedulingLoop->loopBodies.size();

    LoopScheduleResult scheduleResult;

    if (numIterations == 0) {
        return scheduleResult;
    }

    // Step 1 — Classify the region.
    auto classifyLog = _log.nest();
    const auto classifier = classifyVFRegion(loopRegion, memorySize, classifyLog);
    // Classifier reported the region contains a structural pattern it cannot model
    // soundly. Early return from loop allocator
    if (!classifier.classifierSupportedRegion) {
        _log.warning("Classifier marked region as unsupported (reason={0}), returning empty schedule",
                     stringifyClassifierVFFallbackReason(classifier.fallbackReason));
        return scheduleResult;
    }

    // Currently VF loop allocator does not accept external buffers which are distinct between iterations
    // TODO: This will be enabled once E#229223 is done.
    for (const auto& group : classifier.explicitExternalGroups) {
        if (group.iterationValues.size() > 1) {
            _log.warning(
                    "Explicit external group has multiple iteration values. This is not supported yet by allocator");
            return scheduleResult;
        }
    }

    // Pipelining requires mixed DPU+SHV compute in the template body.
    // This is the strategy feasibility gate before allocation is attempted.
    const bool canUsePipelining = supportsVFPipelining(loopRegion);
    _log.trace("UndefinedVF: VF region dump (iterations={0}, memoryLimit={1}, pipeliningSupported={2})", numIterations,
               memorySize, canUsePipelining);

    // Step 2 — Search best strategy.
    // Allocator callback: evaluates a recipe against the region's constraints.
    const AllocatorFn allocator = [&](const VfStrategyRecipe& recipe) {
        // If pipelining is structurally unsupported, mark only this recipe as
        // infeasible and continue searching for the next strategy.
        if (recipe.id == VPU::VFScenario::VF_PIPELINING && !canUsePipelining) {
            _log.trace("Strategy {0} skipped: region has no mixed DPU+SHV compute", stringifyVFScenario(recipe.id));
            return VfAllocateResult{};
        }
        return tryStrategy(recipe, loopRegion, classifier, memorySize, _log);
    };

    auto outcome = runStrategySearch(allocator);

    auto& chosen = outcome.result;
    _log.trace(
            "UndefinedVF: strategy search result: feasible={0}, selectedStrategy={1}, peakUsed={2}, "
            "memoryLimit={3}",
            chosen.feasible,
            outcome.selectedVFStrategy.has_value() ? stringifyVFScenario(outcome.selectedVFStrategy.value()) : "NONE",
            chosen.peakUsedBytes, memorySize);
    if (!chosen.feasible) {
        _log.warning("No feasible schedule found (MINIMAL infeasible) for region (peakUsed={0}, memoryLimit={1}). "
                     "Single-iteration fit precondition violated.",
                     chosen.peakUsedBytes, memorySize);
        return scheduleResult;
    }

    // Step 3 — Replicate template for all iterations.
    scheduleResult = buildLoopScheduleResult(chosen, loopRegion, outcome, _log);

    // Step 4 — Verify.
    verifyLocalSchedule(scheduleResult);

    return scheduleResult;
}

}  // namespace vpux::VPU
