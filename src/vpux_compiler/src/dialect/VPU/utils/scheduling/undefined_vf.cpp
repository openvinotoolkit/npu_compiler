//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_vf.hpp"
#include "vpux/compiler/core/scheduling/classifier_vf.hpp"

namespace vpux::VPU {

namespace {

// Walk over strategies in priority order and return first feasible
// The MINIMAL acts as a fallback and should be always feasible
// if VF tiling was done sufficiently with respect to memory constraints
constexpr VfStrategyRecipe vfStrategyRecipes[] = {
        {VPU::VFScenario::FULL_PREFETCHING, VfSchedStrategyDescriptor::FetchScope::ALL_INPUTS,
         VfSchedStrategyDescriptor::OutputResidency::KEEP},
        {VPU::VFScenario::LASTOP_PREFETCHING, VfSchedStrategyDescriptor::FetchScope::ALL_WEIGHTS,
         VfSchedStrategyDescriptor::OutputResidency::KEEP},
        {VPU::VFScenario::WEIGHTS_PREFETCHING, VfSchedStrategyDescriptor::FetchScope::ALL_WEIGHTS,
         VfSchedStrategyDescriptor::OutputResidency::DROP},
        // Minimal is last so any earlier feasible recipe short-circuits the search.
        {VPU::VFScenario::MINIMAL, VfSchedStrategyDescriptor::FetchScope::NONE,
         VfSchedStrategyDescriptor::OutputResidency::DROP},
};

// Create a map which associates each buffer value in the template body with
// its position (opPos, slot) in the template body.
// This will be later used to find the corresponding output buffer value of other iteration bodies
// when replicating the template schedule across iterations.
using BufferPosition = std::pair<size_t, size_t>;
llvm::DenseMap<mlir::Value, BufferPosition> buildBufferPositionMap(const LoopBody& templateBody) {
    llvm::DenseMap<mlir::Value, BufferPosition> map;
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& outBufs = templateBody[opPos].outBuffers;
        for (size_t slot = 0; slot < outBufs.size(); ++slot) {
            // First-seen position wins — duplicates would imply structural
            // in-place aliasing; they map to their original allocation site.
            map.insert({outBufs[slot], BufferPosition{opPos, slot}});
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
    VPUX_THROW_UNLESS(it != posMap.end(),
                      "Template schedule references value {0} not produced by any template op outBuffer. All buffers "
                      "are expected to be produced by loop body ops",
                      templateValue);
    const auto [opPos, slot] = it->second;
    VPUX_THROW_UNLESS(opPos < bodyK.size() && slot < bodyK[opPos].outBuffers.size(),
                      "Iteration body shape diverges from template at op {0} slot {1}", opPos, slot);
    return bodyK[opPos].outBuffers[slot];
}

// Replicate the single iteration template emitted by VfAllocateLinear across
// every iteration body in the region. Use same offsets as in template body but use
// given iteration buffers
PredefinedSchedule replicateTemplateScheduleAcrossAllLoopIterations(const VfAllocateResult& result,
                                                                    const SchedulingLoop& loop) {
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
            for (const auto& [v, addr] : templateEntry.allocations) {
                allocs.push_back({findIterationBufferMatchingTemplateBuffer(v, posMap, bodyK), addr});
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

LoopScheduleResult buildLoopScheduleResult(const VfAllocateResult& result, const ComputeRegion& region) {
    LoopScheduleResult out;
    out.schedule = replicateTemplateScheduleAcrossAllLoopIterations(result, *region.schedulingLoop);
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
    for (auto shared : region.sharedExternalBuffers) {
        out.sharedExternalBuffers.insert(shared);
    }

    // Derive baseAlignment from the strictest operand alignment requirement. This is needed to ensure
    // that the outer scheduler allocates memory range for VF block at an address that satisfies all buffer alignment.
    // If base address is aligned to the strictest alignment, all other buffers withing this block will be aligned as
    // well based on their relative offsets and per buffer alignment requirements.
    vpux::AddressType baseAlignment = vpux::DEFAULT_CMX_ALIGNMENT;
    for (const auto& entry : result.iterationSchedule) {
        for (const auto& [v, _] : entry.allocations) {
            const auto bufAlign = vpux::getAlignment(v);
            baseAlignment = baseAlignment < bufAlign ? bufAlign : baseAlignment;
        }
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
                    VPUX_THROW_UNLESS(!Partitioner::intersects(offset, size, liveRange.offset, liveRange.size),
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
    VfSchedStrategyDescriptor params{recipe.fetchScp, recipe.outputRes};
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
        out.result = allocator(recipe);
        if (out.result.feasible) {
            _log.trace("Strategy {0} is first feasible, use it", stringifyVFScenario(recipe.id));
            out.selectedVFStrategy = recipe.id;
            return out;
        }

        _log.trace("Strategy {0} is infeasible or not supported, peakUsed={1}", stringifyVFScenario(recipe.id),
                   out.result.peakUsedBytes);
    }
    return out;
}

LoopScheduleResult UndefinedVF::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                    vpux::AddressType memorySize) const {
    VPUX_THROW_UNLESS(loopRegion.schedulingLoop != nullptr, "ComputeRegion has no scheduling loop");
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
        _log.warning("Classifier marked region as unsupported, returning empty schedule");
        return scheduleResult;
    }

    // Step 2 — Search best strategy.
    // Allocator callback: evaluates a recipe against the region's constraints.
    const AllocatorFn allocator = [&](const VfStrategyRecipe& recipe) {
        return tryStrategy(recipe, loopRegion, classifier, memorySize, _log);
    };

    auto outcome = runStrategySearch(allocator);

    auto& chosen = outcome.result;
    if (!chosen.feasible) {
        _log.warning("No feasible schedule found (MINIMAL infeasible) for region (peakUsed={0}, memoryLimit={1}). "
                     "Single-iteration fit precondition violated.",
                     chosen.peakUsedBytes, memorySize);
        return scheduleResult;
    }

    // Step 3 — Replicate template for all iterations.
    scheduleResult = buildLoopScheduleResult(chosen, loopRegion);

    // Step 4 — Verify.
    verifyLocalSchedule(scheduleResult);

    return scheduleResult;
}

}  // namespace vpux::VPU
