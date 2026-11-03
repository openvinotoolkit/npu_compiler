//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_utils.hpp"

#include <algorithm>
#include "vpux/compiler/dialect/config/IR/resources.hpp"
namespace vpux::VPU {

namespace {

enum class ComputeKind { DPU, SHV, OTHER };

// A group of entries moved from one iteration to the preceding iteration.
struct PipelineHoistSegment {
    size_t startPos = 0;
    size_t endPos = 0;
    SmallVector<ComputeExplicitSchedule, 4> entries;
};

}  // namespace

bool supportsVFPipelining(const ComputeRegion& region) {
    if (region.schedulingLoop == nullptr || region.schedulingLoop->loopBodies.empty()) {
        return false;
    }

    bool hasDPU = false;
    bool hasSHV = false;
    for (const auto& info : region.schedulingLoop->loopBodies.front()) {
        hasDPU |= info.queueType.type == config::ExecutorKind::DPU;
        hasSHV |= info.queueType.type == config::ExecutorKind::SHAVE_ACT;
        if (hasDPU && hasSHV) {
            return true;
        }
    }

    return false;
}

namespace {

// Find the first compute entry in a step.
std::optional<size_t> findFirstComputeEntry(const IterationSchedule& step) {
    for (size_t idx = 0; idx < step.size(); ++idx) {
        if (step[idx].allocInfo.allocationType == AllocationType::COMPUTE) {
            return idx;
        }
    }
    return std::nullopt;
}

// Map a compute entry to its executor kind used by pipelining placement logic.
ComputeKind getComputeKind(const ComputeExplicitSchedule& entry) {
    const auto kind = entry.allocInfo.queueType.type;
    if (kind == config::ExecutorKind::DPU) {
        return ComputeKind::DPU;
    }
    if (kind == config::ExecutorKind::SHAVE_ACT) {
        return ComputeKind::SHV;
    }
    return ComputeKind::OTHER;
}

// True for DMA schedule entries (both DATA_IN and DATA_OUT).
bool isDmaEntry(const ComputeExplicitSchedule& entry) {
    return entry.allocInfo.allocationType == AllocationType::DATA_IN ||
           entry.allocInfo.allocationType == AllocationType::DATA_OUT;
}

// Check if a DMA output buffer matches any of the compute's input buffers.
bool isDmaForComputeInput(const ComputeExplicitSchedule& dmaEntry, const ComputeExplicitSchedule& computeEntry) {
    for (const auto& dmaProd : dmaEntry.allocInfo.outBuffers) {
        for (const auto& computeConsume : computeEntry.allocInfo.inBuffers) {
            if (dmaProd == computeConsume) {
                return true;
            }
        }
    }
    return false;
}

// Find the range [startIdx, endIdx] of operations to hoist for pipelining:
// - Input DMAs immediately before the compute (whose outputs feed compute inputs)
// - The compute operation itself
// Returns {startIdx, endIdx} for [input DMAs..., compute].
std::optional<std::pair<size_t, size_t>> findPipelineHoistRange(const IterationSchedule& step, size_t computePos) {
    const auto& computeEntry = step[computePos];

    size_t startIdx = computePos;
    for (size_t idx = computePos; idx > 0; --idx) {
        const auto& prevEntry = step[idx - 1];
        if (isDmaEntry(prevEntry) && isDmaForComputeInput(prevEntry, computeEntry)) {
            startIdx = idx - 1;
        } else {
            break;
        }
    }

    return std::make_pair(startIdx, computePos + 1);
}

SmallVector<size_t> getComputePositions(const IterationSchedule& step, std::optional<ComputeKind> kind = std::nullopt) {
    SmallVector<size_t> computePositions;
    computePositions.reserve(step.size());
    for (size_t idx = 0; idx < step.size(); ++idx) {
        if (step[idx].allocInfo.allocationType != AllocationType::COMPUTE) {
            continue;
        }
        if (kind.has_value() && getComputeKind(step[idx]) != kind.value()) {
            continue;
        }
        computePositions.push_back(idx);
    }
    return computePositions;
}

bool isAlternatingComputePattern(const IterationSchedule& step) {
    const auto computePositions = getComputePositions(step);
    if (computePositions.size() < 2) {
        return false;
    }

    auto prevKind = getComputeKind(step[computePositions.front()]);
    if (prevKind == ComputeKind::OTHER) {
        return false;
    }

    for (size_t idx = 1; idx < computePositions.size(); ++idx) {
        const auto kind = getComputeKind(step[computePositions[idx]]);
        if (kind == ComputeKind::OTHER || kind == prevKind) {
            return false;
        }
        prevKind = kind;
    }

    return true;
}

bool consumesOutputOfEntry(const ComputeExplicitSchedule& consumer, const ComputeExplicitSchedule& producer) {
    for (const auto& input : consumer.allocInfo.inBuffers) {
        for (const auto& output : producer.allocInfo.outBuffers) {
            if (input == output) {
                return true;
            }
        }
    }
    return false;
}

bool dependsOnUnhoistedEarlierCompute(const IterationSchedule& step, size_t computePos,
                                      ArrayRef<size_t> selectedComputePositions) {
    const auto& consumer = step[computePos];
    for (size_t idx = 0; idx < computePos; ++idx) {
        const auto& producer = step[idx];
        if (producer.allocInfo.allocationType != AllocationType::COMPUTE) {
            continue;
        }
        if (llvm::is_contained(selectedComputePositions, idx)) {
            continue;
        }
        if (consumesOutputOfEntry(consumer, producer)) {
            return true;
        }
    }
    return false;
}

// A hoist segment is one lead-executor compute and the input DMA entries that
// immediately precede it. For example, [DMA(input) -> DPU] is moved as one
// segment from iteration k+1 into iteration k; the DMA must move with the DPU
// so the compute still receives the buffer it consumes.
SmallVector<PipelineHoistSegment, 2> collectPipelineHoistSegments(const IterationSchedule& nextStep,
                                                                  ComputeKind leadKind) {
    SmallVector<PipelineHoistSegment, 2> segments;

    const auto computePositions = getComputePositions(nextStep);
    if (computePositions.size() == 2 && isAlternatingComputePattern(nextStep)) {
        return segments;
    }

    SmallVector<size_t> selectedComputePositions;
    for (const auto computePos : computePositions) {
        if (getComputeKind(nextStep[computePos]) != leadKind) {
            continue;
        }
        if (dependsOnUnhoistedEarlierCompute(nextStep, computePos, selectedComputePositions)) {
            continue;
        }

        const auto hoistRange = findPipelineHoistRange(nextStep, computePos);
        if (!hoistRange.has_value()) {
            continue;
        }

        const auto [startPos, endPos] = hoistRange.value();
        PipelineHoistSegment segment;
        segment.startPos = startPos;
        segment.endPos = endPos;
        segment.entries.reserve(endPos - startPos);
        for (size_t idx = startPos; idx < endPos; ++idx) {
            segment.entries.push_back(nextStep[idx]);
        }
        segments.push_back(std::move(segment));
        selectedComputePositions.push_back(computePos);
    }

    return segments;
}

// Assigns segments to anchor groups. With enough anchors this is a 1:1 mapping
// (unchanged from before); when segments outnumber anchors, the first
// (anchorCount - 1) segments each keep their own anchor and the remaining overflow
// segments are all merged into the last anchor's group. The full overflow chain is
// always merged as one unit - partially hoisting it can split a buffer's allocation
// from its consumer across the iteration boundary (see UndefinedVF verifier failures).
SmallVector<SmallVector<size_t, 4>, 2> groupSegmentsByAnchor(ArrayRef<PipelineHoistSegment> segments,
                                                             size_t anchorCount) {
    SmallVector<SmallVector<size_t, 4>, 2> groups;
    const size_t segmentCount = segments.size();
    if (segmentCount <= anchorCount) {
        for (size_t idx = 0; idx < segmentCount; ++idx) {
            groups.push_back({idx});
        }
        return groups;
    }

    const size_t keepIndividual = anchorCount - 1;
    for (size_t idx = 0; idx < keepIndividual; ++idx) {
        groups.push_back({idx});
    }

    SmallVector<size_t, 4> mergedGroup;
    for (size_t idx = keepIndividual; idx < segmentCount; ++idx) {
        mergedGroup.push_back(idx);
    }
    if (!mergedGroup.empty()) {
        groups.push_back(std::move(mergedGroup));
    }
    return groups;
}

bool insertPipelineSegmentsBeforeOppositeKind(IterationSchedule& currStep, IterationSchedule& nextStep,
                                              SmallVectorImpl<PipelineHoistSegment>& segments, ComputeKind leadKind) {
    if (segments.empty()) {
        return false;
    }

    const auto oppositeKind = leadKind == ComputeKind::DPU ? ComputeKind::SHV : ComputeKind::DPU;
    const auto oppositeComputePositions = getComputePositions(currStep, oppositeKind);
    if (oppositeComputePositions.empty()) {
        return false;
    }

    const auto groups = groupSegmentsByAnchor(segments, oppositeComputePositions.size());
    if (groups.empty()) {
        return false;
    }

    // Anchor before the opposite kind's own input DMAs (not between them and its compute),
    // so the hoisted lead-kind work is fully in place before the opposite kind starts fetching.
    SmallVector<size_t> anchorPositions;
    anchorPositions.reserve(groups.size());
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        const auto hoistRange = findPipelineHoistRange(currStep, oppositeComputePositions[idx]);
        anchorPositions.push_back(hoistRange.has_value() ? hoistRange->first : oppositeComputePositions[idx]);
    }

    // Erase-mark using each original segment's own [startPos, endPos) range rather than
    // a merged span, so any un-hoisted entries interleaved between merged segments
    // (e.g. a compute's own trailing output DMA) are correctly left in place.
    SmallVector<bool> eraseMask(nextStep.size(), false);
    size_t hoistedEntryCount = 0;
    for (const auto& group : groups) {
        for (const auto segIdx : group) {
            const auto& segment = segments[segIdx];
            hoistedEntryCount += segment.entries.size();
            for (size_t idx = segment.startPos; idx < segment.endPos; ++idx) {
                eraseMask[idx] = true;
            }
        }
    }

    IterationSchedule rewrittenCurrent;
    rewrittenCurrent.reserve(currStep.size() + hoistedEntryCount);
    size_t groupIdx = 0;
    for (size_t idx = 0; idx < currStep.size(); ++idx) {
        if (groupIdx < groups.size() && anchorPositions[groupIdx] == idx) {
            for (const auto segIdx : groups[groupIdx]) {
                for (const auto& entry : segments[segIdx].entries) {
                    rewrittenCurrent.push_back(entry);
                }
            }
            ++groupIdx;
        }
        rewrittenCurrent.push_back(currStep[idx]);
    }

    IterationSchedule rewrittenNext;
    rewrittenNext.reserve(nextStep.size() - hoistedEntryCount);
    for (size_t idx = 0; idx < nextStep.size(); ++idx) {
        if (!eraseMask[idx]) {
            rewrittenNext.push_back(nextStep[idx]);
        }
    }

    currStep = std::move(rewrittenCurrent);
    nextStep = std::move(rewrittenNext);
    return true;
}

}  // namespace

bool rewriteVFPipelinedScheduleIntoPairwiseSteps(PredefinedSchedule& schedule) {
    // Turn per-iteration replay into a wavefront by moving selected compute
    // stages from iteration k+1 into iteration k together with their input DMAs.
    // For bodies with more than two compute stages, dependency-safe stages on the
    // lead executor are moved before the opposite executor's input DMAs:
    //   - lead DPU  -> before current SHAVE input DMAs
    //   - lead SHAVE -> before current DPU input DMAs
    // This makes the moved stages eligible for overlap; the scheduler still
    // determines the final execution timing. When lead-kind segments outnumber
    // available opposite-kind anchors, all of them are merged into the last anchor
    // together (see groupSegmentsByAnchor). Two-stage alternating bodies keep their
    // original replay and rely on double buffering plus loop overlap instead.
    bool hoisted = false;
    for (size_t iter = 0; iter + 1 < schedule.size(); ++iter) {
        auto& currStep = schedule[iter];
        auto& nextStep = schedule[iter + 1];

        const auto firstComputePos = findFirstComputeEntry(nextStep);
        if (!firstComputePos.has_value()) {
            continue;
        }

        const auto leadKind = getComputeKind(nextStep[firstComputePos.value()]);
        if (leadKind == ComputeKind::OTHER) {
            continue;
        }

        auto segments = collectPipelineHoistSegments(nextStep, leadKind);
        if (segments.empty()) {
            continue;
        }

        hoisted |= insertPipelineSegmentsBeforeOppositeKind(currStep, nextStep, segments, leadKind);
    }
    return hoisted;
}

}  // namespace vpux::VPU
