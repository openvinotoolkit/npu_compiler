//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_tiling.hpp"
#include "vpux/compiler/core/scheduling/utils.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

using BufferId = size_t;

namespace {

// The scheduling loop stores mlir::Value buffers.
// Allocation/frequency logic operates on integral ids (BufferId) for fast lookup and stable indexing.
// Raw size/alignment are queried once and cached to avoid repeated type inspection.
struct BufferInfoCache {
    mlir::DenseMap<mlir::Value, BufferId> bufferIds;
    std::vector<vpux::BufferDesc> buffers;
};

BufferInfoCache buildBufferInfoCache(const vpux::SchedulingLoop& schedulingLoop) {
    vpux::ValueOrderedSet orderedBuffers;
    for (const auto& loopBody : schedulingLoop.loopBodies) {
        for (const auto& allocInfo : loopBody) {
            orderedBuffers.insert(allocInfo.inBuffers.begin(), allocInfo.inBuffers.end());
            orderedBuffers.insert(allocInfo.outBuffers.begin(), allocInfo.outBuffers.end());
        }
    }

    BufferInfoCache bufferInfoCache;
    bufferInfoCache.buffers.reserve(orderedBuffers.size());

    for (auto buffer : orderedBuffers) {
        const size_t bufferId = bufferInfoCache.buffers.size();
        bufferInfoCache.bufferIds[buffer] = bufferId;
        bufferInfoCache.buffers.emplace_back(buffer, vpux::getRawBufferSize(buffer), vpux::getAlignment(buffer));
    }

    return bufferInfoCache;
}

}  // namespace
namespace vpux::VPU {

using OperandIndex = size_t;
using IterationIndex = size_t;
using ComputeOpIndex = size_t;
using AddressRange = std::pair<vpux::AddressType, vpux::AddressType>;

// Terminology used in this module:
// - Iteration: outer loop body index in SchedulingLoop::loopBodies.
// - Compute op: one COMPUTE op inside an iteration.
//
// schedulingLoop.loopBodies
// ├─ OuterIterationIndex = 0 (if 2D tiling)
// │  ├─ InnerIterationIndex = 0
// │  │  ├─ ComputeOpIndex = 0  -> COMPUTE op in inner loop body 0
// │  ├─ InnerIterationIndex = 1
// │  │  ├─ ComputeOpIndex = 0  -> COMPUTE op in inner loop body 1
// |─ OuterIterationIndex = 1 (if 2D tiling)
// │ ├─ InnerIterationIndex = 0
// │ │ ├─ ComputeOpIndex = 0  -> COMPUTE op in inner loop body 0
// ... and so on for more loop bodies and inner loops.
//
// Note: current implementation only supports 1 compute op per loop body, so ComputeOpIndex is always 0 for now.
// Add more support when enabling VF tiling case
struct ComputeOpKey {
    IterationIndex iterationIdx = 0;
    ComputeOpIndex computeOpIdx = 0;

    bool operator==(const ComputeOpKey& other) const noexcept {
        return iterationIdx == other.iterationIdx && computeOpIdx == other.computeOpIdx;
    }
};

struct ComputeOpKeyHash {
    size_t operator()(const ComputeOpKey& key) const noexcept {
        return static_cast<size_t>(llvm::hash_combine(key.iterationIdx, key.computeOpIdx));
    }
};

using ComputeOpBuffers = std::vector<std::pair<ComputeOpKey, std::vector<BufferId>>>;
using SharedBufferIds = std::set<BufferId>;
using OperandAllocationOrder = std::vector<std::pair<OperandIndex, size_t>>;
using FrequencyTable = std::unordered_map<OperandIndex, std::unordered_map<BufferId, size_t>>;
using ComputeOpAllocationMap = std::unordered_map<ComputeOpKey, std::map<BufferId, AddressRange>, ComputeOpKeyHash>;
using OperandSlotRequirements = std::vector<std::pair<vpux::AddressType, vpux::AddressType>>;

UndefinedTiling::UndefinedTiling(): _log(Logger::global()) {
    _log.setName("UndefinedTiling");
}

llvm::StringRef UndefinedTiling::getName() const {
    return "UndefinedTiling";
}

struct CollectResult {
    ComputeOpBuffers computeOpBuffers;
    FrequencyTable frequencyTable;
};

CollectResult collectComputeOpBuffers(const SchedulingLoop& schedulingLoop, const BufferInfoCache& bufferInfoCache) {
    ComputeOpBuffers computeOpBuffers;
    // Collect buffers for each compute op.
    for (const auto& iterationIdx : irange(schedulingLoop.loopBodies.size())) {
        const auto& loopBody = schedulingLoop.loopBodies[iterationIdx];
        ComputeOpIndex computeOpIdx = 0;

        for (const auto& allocInfo : loopBody) {
            if (allocInfo.allocationType != AllocationType::COMPUTE) {
                continue;
            }

            std::vector<size_t> used;
            for (auto input : allocInfo.inBuffers) {
                const auto inputIt = bufferInfoCache.bufferIds.find(input);
                VPUX_THROW_UNLESS(inputIt != bufferInfoCache.bufferIds.end(), "Buffer is missing in bufferInfoCache");
                const auto id = inputIt->second;
                used.push_back(id);
            }

            for (auto output : allocInfo.outBuffers) {
                const auto outputIt = bufferInfoCache.bufferIds.find(output);
                VPUX_THROW_UNLESS(outputIt != bufferInfoCache.bufferIds.end(), "Buffer is missing in bufferInfoCache");
                const auto id = outputIt->second;

                auto existing = std::find_if(used.begin(), used.end(), [&](size_t itemId) {
                    return itemId == id;
                });
                if (existing != used.end()) {
                    // Buffer appears as both input and output.
                    continue;
                }
                used.push_back(id);
            }

            computeOpBuffers.emplace_back(ComputeOpKey{iterationIdx, computeOpIdx}, std::move(used));
            ++computeOpIdx;
        }
    }

    if (computeOpBuffers.empty()) {
        return {std::move(computeOpBuffers), {}};
    }

    const size_t numOperands = computeOpBuffers.front().second.size();

    // Build frequency table per operand position:
    // frequencyTable[operand][bufferId] = usage count.
    FrequencyTable frequencyTable;

    for (const auto& [computeOpKey, buffers] : computeOpBuffers) {
        VPUX_THROW_UNLESS(buffers.size() == numOperands,
                          "Inconsistent operand count in compute op ({0}, {1}): expected={2}, got={3}",
                          computeOpKey.iterationIdx, computeOpKey.computeOpIdx, numOperands, buffers.size());
        for (size_t operand = 0; operand < numOperands; ++operand) {
            frequencyTable[operand][buffers[operand]]++;
        }
    }

    // E-208299: Investigate more sophisticated heuristics considering buffer size, liveness etc.
    return {std::move(computeOpBuffers), std::move(frequencyTable)};
}

OperandSlotRequirements getOperandSlotRequirements(const ComputeOpBuffers& computeOpBuffers,
                                                   const BufferInfoCache& bufferInfoCache,
                                                   const SharedBufferIds& sharedBuffers) {
    const auto numOperands = computeOpBuffers.front().second.size();
    OperandSlotRequirements slotRequirements(numOperands, {0, 1});

    for (const auto& [computeOpKey, buffers] : computeOpBuffers) {
        VPUX_THROW_UNLESS(buffers.size() == numOperands,
                          "Inconsistent operand count in compute op ({0}, {1}): expected={2}, got={3}",
                          computeOpKey.iterationIdx, computeOpKey.computeOpIdx, numOperands, buffers.size());

        for (const auto& operandIdx : irange(buffers.size())) {
            const auto bufferId = buffers[operandIdx];
            if (sharedBuffers.count(bufferId) > 0) {
                continue;
            }
            const auto rawSize = bufferInfoCache.buffers[bufferId].rawSize;
            const auto rawAlignment = bufferInfoCache.buffers[bufferId].rawAlignment;

            slotRequirements[operandIdx].first = std::max(slotRequirements[operandIdx].first, rawSize);
            slotRequirements[operandIdx].second = std::max(slotRequirements[operandIdx].second, rawAlignment);
        }
    }

    return slotRequirements;
}

// Identify buffers that are shared across all compute ops and
// pre-reserve CMX space for them from top to down.
// Returns:
//   vpux::AddressType  — remaining available CMX memory after reserving shared buffers.
//   SharedBufferIds — set of buffer ids that are globally shared and should be resident in CMX.
std::pair<vpux::AddressType, SharedBufferIds> reserveGloballySharedBuffers(const FrequencyTable& frequencyTable,
                                                                           size_t computeOpCount,
                                                                           vpux::AddressType memorySize,
                                                                           const BufferInfoCache& bufferInfoCache) {
    vpux::AddressType availableMemory = memorySize;
    SharedBufferIds sharedBuffers;

    for (const auto& [_, bufferFrequency] : frequencyTable) {
        for (const auto& [bufferId, frequency] : bufferFrequency) {
            if (frequency >= computeOpCount) {
                sharedBuffers.insert(bufferId);
            }
        }
    }

    // Reserve CMX in deterministic order
    for (const auto bufferId : sharedBuffers) {
        const auto rawSize = bufferInfoCache.buffers[bufferId].rawSize;
        const auto rawAlignment = bufferInfoCache.buffers[bufferId].rawAlignment;
        VPUX_THROW_UNLESS(rawSize <= availableMemory,
                          "Insufficient CMX for shared buffer {0}: requested={1}, available={2}", bufferId, rawSize,
                          availableMemory);
        availableMemory -= rawSize;
        availableMemory = alignValDown(availableMemory, rawAlignment);
    }

    return {availableMemory, sharedBuffers};
}

OperandAllocationOrder getOperandAllocationOrder(const FrequencyTable& frequencyTable) {
    OperandAllocationOrder operandReloads;
    operandReloads.reserve(frequencyTable.size());

    for (const auto& [operandIdx, bufferFrequency] : frequencyTable) {
        operandReloads.emplace_back(operandIdx, bufferFrequency.size());
    }

    // Sort operands by minimum reloads first.
    // E-208301: Investigate more sophisticated heuristics considering buffer size, liveness etc.
    std::sort(operandReloads.begin(), operandReloads.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });

    return operandReloads;
}

ComputeOpAllocationMap getMemoryAllocations(const ComputeOpBuffers& computeOpBuffers,
                                            const SharedBufferIds& sharedBuffers,
                                            const OperandAllocationOrder& operandAllocationOrder,
                                            const OperandSlotRequirements& operandSlotRequirements,
                                            vpux::AddressType availableMemory, vpux::AddressType& usedMemory,
                                            vpux::AddressType& baseAlignment) {
    // Allocate local buffers using two-tier behavior:
    //  1) PIPELINING-like reuse when slots remain valid/alive.
    //  2) PREFETCHING fallback by reusing oldest allocation for the same operand on overflow.
    ComputeOpAllocationMap memoryAllocations;

    // Verify operand slot requirements.
    for (const auto& [computeOpKey, usedBuffers] : computeOpBuffers) {
        VPUX_THROW_UNLESS(usedBuffers.size() == operandSlotRequirements.size(),
                          "Inconsistent operand slot requirements for compute op ({0}, {1}): expected={2}, got={3}",
                          computeOpKey.iterationIdx, computeOpKey.computeOpIdx, usedBuffers.size(),
                          operandSlotRequirements.size());
    }

    // Derive baseAlignment from the strictest operand alignment requirement.
    for (const auto& [_, alignment] : operandSlotRequirements) {
        baseAlignment = std::max(baseAlignment, alignment);
    }

    vpux::AddressType nextFreeOffset = 0;
    std::unordered_map<BufferId, AddressRange> aliveBuffers;
    using SlotEntry = std::pair<BufferId, AddressRange>;
    std::unordered_map<size_t, SmallVector<SlotEntry, 2>> operandAllocations;
    // Once any operand cannot be double-buffered due to memory exhaustion,
    // disable double-buffering for all remaining operands. Partial double-buffering
    // wastes CMX without enabling pipelining and can cause spills elsewhere.
    bool reachedMemoryLimit = false;

    for (const auto& [computeOpKey, usedBuffers] : computeOpBuffers) {
        // Store memory allocations for each compute op.
        std::map<BufferId, AddressRange> computeOpAllocation;

        VPUX_THROW_UNLESS(operandAllocationOrder.size() == usedBuffers.size(), "Inconsistent allocation order");

        for (const auto& [operandIdx, _] : operandAllocationOrder) {
            const auto bufferId = usedBuffers[operandIdx];
            if (sharedBuffers.count(bufferId) > 0) {
                continue;
            }

            if (aliveBuffers.count(bufferId) > 0) {
                // Buffer already alive at the same address range.
                computeOpAllocation[bufferId] = aliveBuffers[bufferId];
                continue;
            }

            VPUX_THROW_UNLESS(operandIdx < operandSlotRequirements.size(),
                              "Operand index {0} exceeds slot requirements size {1}", operandIdx,
                              operandSlotRequirements.size());
            const auto [size, alignment] = operandSlotRequirements[operandIdx];
            VPUX_THROW_UNLESS(alignment > 0, "Invalid alignment {0} for operand {1}", alignment, operandIdx);

            auto begin = alignValUp(nextFreeOffset, alignment);
            VPUX_THROW_UNLESS(begin <= std::numeric_limits<vpux::AddressType>::max() - size,
                              "Address overflow while allocating operand {0}: begin={1}, size={2}", operandIdx, begin,
                              size);
            auto end = begin + size;

            if (end > availableMemory || reachedMemoryLimit || operandAllocations[operandIdx].size() > 1) {
                // Memory exhausted, already hit limit, or operand already has two slots:
                // reuse the oldest allocation slot for this operand.
                if (end > availableMemory) {
                    reachedMemoryLimit = true;
                }
                VPUX_THROW_UNLESS(!operandAllocations[operandIdx].empty(),
                                  "No previous allocation to overwrite for operand {0}", operandIdx);
                const auto [oldBufferId, oldAddress] = operandAllocations[operandIdx].front();
                operandAllocations[operandIdx].erase(operandAllocations[operandIdx].begin());
                begin = oldAddress.first;
                end = oldAddress.second;
                VPUX_THROW_UNLESS(aliveBuffers.count(oldBufferId) > 0,
                                  "Previous allocation for operand {0} (buffer {1}) is not alive", operandIdx,
                                  oldBufferId);
                aliveBuffers.erase(oldBufferId);
            } else {
                // Sequential allocation from offset 0.
                nextFreeOffset = end;
                usedMemory = std::max(usedMemory, nextFreeOffset);
            }

            operandAllocations[operandIdx].push_back({bufferId, {begin, end}});
            aliveBuffers[bufferId] = {begin, end};
            computeOpAllocation[bufferId] = {begin, end};
        }

        memoryAllocations[computeOpKey] = std::move(computeOpAllocation);
    }

    return memoryAllocations;
}

void addAllocationForBuffer(mlir::Value buffer, const std::map<BufferId, AddressRange>& allocationMap,
                            const BufferInfoCache& bufferInfoCache, const SharedBufferIds& sharedBuffers,
                            std::map<BufferId, AddressRange>& aliveBuffers, SmallVector<mlir::Value>& deallocations,
                            SmallVector<std::pair<mlir::Value, vpux::AddressType>>& allocations) {
    const auto idIt = bufferInfoCache.bufferIds.find(buffer);
    VPUX_THROW_UNLESS(idIt != bufferInfoCache.bufferIds.end(), "No buffer id found for value");
    const auto bufferId = idIt->second;

    if (sharedBuffers.count(bufferId) > 0) {
        return;
    }

    VPUX_THROW_UNLESS(allocationMap.count(bufferId) > 0, "No allocation found for buffer id {0}", bufferId);
    const auto targetRange = allocationMap.at(bufferId);

    if (aliveBuffers.count(bufferId) > 0) {
        if (aliveBuffers[bufferId].first == targetRange.first) {
            return;
        }

        deallocations.push_back(buffer);
        aliveBuffers.erase(bufferId);
    } else {
        for (const auto& [aliveId, aliveRange] : llvm::make_early_inc_range(aliveBuffers)) {
            if (aliveRange.first < targetRange.second && targetRange.first < aliveRange.second) {
                deallocations.push_back(bufferInfoCache.buffers[aliveId].value);
                aliveBuffers.erase(aliveId);
            }
        }
    }

    aliveBuffers[bufferId] = targetRange;
    allocations.push_back({buffer, targetRange.first});
}

PredefinedSchedule buildPredefinedSchedule(const SchedulingLoop& schedulingLoop,
                                           const ComputeOpAllocationMap& memoryAllocations,
                                           const SharedBufferIds& sharedBuffers,
                                           const BufferInfoCache& bufferInfoCache) {
    PredefinedSchedule predefinedSchedule;
    std::map<BufferId, AddressRange> aliveBuffers;

    // Build an iteration-level fallback view only for non-COMPUTE ops (e.g. DATA_IN).
    // COMPUTE ops must always use exact (iterationIdx, computeOpIdx) allocation maps.
    std::unordered_map<IterationIndex, std::map<BufferId, AddressRange>> iterationFallbackAllocationMap;
    for (const auto& [computeOpKey, allocationMap] : memoryAllocations) {
        auto& fallbackMap = iterationFallbackAllocationMap[computeOpKey.iterationIdx];
        for (const auto& [bufferId, range] : allocationMap) {
            fallbackMap.try_emplace(bufferId, range);
        }
    }

    for (const auto& iterationIdx : irange(schedulingLoop.loopBodies.size())) {
        const auto& iterationOps = schedulingLoop.loopBodies[iterationIdx];
        ComputeOpIndex computeOpIdx = 0;

        IterationSchedule explicitSchedule;

        for (const auto& allocInfo : iterationOps) {
            SmallVector<std::pair<mlir::Value, vpux::AddressType>> allocations;
            SmallVector<mlir::Value> deallocations;

            const std::map<BufferId, AddressRange>* allocationMapPtr = nullptr;
            if (allocInfo.allocationType == AllocationType::COMPUTE) {
                const auto computeOpKey = ComputeOpKey{iterationIdx, computeOpIdx};
                VPUX_THROW_UNLESS(memoryAllocations.count(computeOpKey) > 0,
                                  "No allocation map found for compute op ({0}, {1})", iterationIdx, computeOpIdx);
                allocationMapPtr = &memoryAllocations.at(computeOpKey);
                ++computeOpIdx;
            } else {
                VPUX_THROW_UNLESS(iterationFallbackAllocationMap.count(iterationIdx) > 0,
                                  "No fallback allocation map found for iteration {0}", iterationIdx);
                allocationMapPtr = &iterationFallbackAllocationMap.at(iterationIdx);
            }

            const auto& allocationMap = *allocationMapPtr;

            for (auto input : allocInfo.inBuffers) {
                addAllocationForBuffer(input, allocationMap, bufferInfoCache, sharedBuffers, aliveBuffers,
                                       deallocations, allocations);
            }
            for (auto output : allocInfo.outBuffers) {
                addAllocationForBuffer(output, allocationMap, bufferInfoCache, sharedBuffers, aliveBuffers,
                                       deallocations, allocations);
            }

            VPUX_THROW_WHEN(!deallocations.empty() && allocations.empty(),
                            "Deallocation buffers are non-empty while allocation buffers are empty");
            if (allocInfo.allocationType == AllocationType::DATA_IN && allocations.empty()) {
                continue;
            }

            explicitSchedule.push_back(
                    ComputeExplicitSchedule(allocInfo, std::move(deallocations), std::move(allocations)));
        }

        IterationSchedule orderedSchedule;
        std::unordered_set<size_t> orderedIndices;
        // Order DATA_IN DMAs first when no deallocation is required.
        for (const auto& index : irange(explicitSchedule.size())) {
            const auto& [allocInfo, deallocations, _allocations, _templatePos] = explicitSchedule[index];
            if (allocInfo.allocationType != AllocationType::DATA_IN || !deallocations.empty()) {
                continue;
            }
            orderedSchedule.push_back(std::move(explicitSchedule[index]));
            orderedIndices.insert(index);
        }
        for (const auto& index : irange(explicitSchedule.size())) {
            if (orderedIndices.count(index) > 0) {
                continue;
            }
            orderedSchedule.push_back(std::move(explicitSchedule[index]));
        }

        predefinedSchedule.push_back(std::move(orderedSchedule));
    }

    return predefinedSchedule;
}

// Verify that all buffers used by non-COMPUTE ops (e.g. DATA_IN) are also tracked by at least one COMPUTE op.
// The loop allocator only assigns CMX addresses for buffers referenced by COMPUTE ops. If a non-COMPUTE op
// uses a buffer that no COMPUTE op references (e.g. GatherDMA indices buffers), the allocator cannot
// produce a valid address and the region must fall back to standard scheduling.
bool verifyComputeOpBuffers(const SchedulingLoop& schedulingLoop, const ComputeOpBuffers& computeOpBuffers,
                            const BufferInfoCache& bufferInfoCache, Logger log) {
    std::set<BufferId> computeBufferIds;
    for (const auto& [_, buffers] : computeOpBuffers) {
        computeBufferIds.insert(buffers.begin(), buffers.end());
    }

    for (const auto& loopBody : schedulingLoop.loopBodies) {
        for (const auto& allocInfo : loopBody) {
            if (allocInfo.allocationType == AllocationType::COMPUTE) {
                continue;
            }
            const auto checkBuffers = [&](const auto& buffers) -> bool {
                for (auto buf : buffers) {
                    const auto it = bufferInfoCache.bufferIds.find(buf);
                    if (it != bufferInfoCache.bufferIds.end() && computeBufferIds.count(it->second) == 0) {
                        log.warning("Non-COMPUTE op {0} uses buffer id {1} not tracked by any COMPUTE op. "
                                    "Falling back to standard scheduling.",
                                    allocInfo.opIdx, it->second);
                        return false;
                    }
                }
                return true;
            };

            if (!checkBuffers(allocInfo.inBuffers) || !checkBuffers(allocInfo.outBuffers)) {
                return false;
            }
        }
    }

    return true;
}

// Verify that buffers fit in memory
bool verifyOperandSlotRequirements(const OperandAllocationOrder& operandAllocationOrder,
                                   const OperandSlotRequirements& operandSlotRequirements,
                                   vpux::AddressType availableMemory, Logger log) {
    vpux::AddressType nextOffset = 0;

    for (const auto& [operandIdx, _] : operandAllocationOrder) {
        // Allocate memory for each operand.
        const auto [size, alignment] = operandSlotRequirements[operandIdx];
        nextOffset = alignValUp(nextOffset, alignment) + size;
        if (nextOffset > availableMemory) {
            log.trace("Insufficient memory for operand {0}: size {1} alignment {2}", operandIdx, size, alignment);
            return false;
        }
    }

    return true;
}

LoopScheduleResult UndefinedTiling::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                        vpux::AddressType memorySize) const {
    VPUX_THROW_UNLESS(loopRegion.schedulingLoop != nullptr, "ComputeRegion has no scheduling loop");

    const auto& schedulingLoop = *loopRegion.schedulingLoop;
    _log.trace("Scheduling loop: type={0}, iterations={1}", toString(schedulingLoop.type),
               schedulingLoop.loopBodies.size());

    const auto bufferInfoCache = buildBufferInfoCache(schedulingLoop);

    const auto [computeOpBuffers, frequencyTable] = collectComputeOpBuffers(schedulingLoop, bufferInfoCache);

    if (computeOpBuffers.empty()) {
        _log.trace("No compute ops found, return empty schedule result");
        return {/*schedule=*/{}, /*reservedSize=*/0, /*sharedExternalBuffers=*/{},
                /*baseAlignment=*/vpux::DEFAULT_CMX_ALIGNMENT};
    }

    _log.trace("Built frequency table: size={0}", frequencyTable.size());
    for (const auto& [buffer, frequency] : frequencyTable) {
        _log.trace("Buffer {0}: frequency={1}", buffer, frequency);
    }

    if (!verifyComputeOpBuffers(schedulingLoop, computeOpBuffers, bufferInfoCache, _log)) {
        return {/*schedule=*/{}, /*reservedSize=*/0, /*sharedExternalBuffers=*/{},
                /*baseAlignment=*/vpux::DEFAULT_CMX_ALIGNMENT};
    }

    // Shared criterion is compute-op based:
    // a buffer is treated as globally shared only if it appears in every compute op.
    const auto computeOpCount = computeOpBuffers.size();
    const auto [availableMemory, sharedBuffers] =
            reserveGloballySharedBuffers(frequencyTable, computeOpCount, memorySize, bufferInfoCache);
    _log.trace("Reserved shared buffers: sharedCount={0}, availableMemory={1}", sharedBuffers.size(), availableMemory);

    // Allocate operands in ascending reload count.
    const auto operandAllocationOrder = getOperandAllocationOrder(frequencyTable);
    _log.trace("Determined operand allocation order: size={0}", operandAllocationOrder.size());
    for (const auto& [operand, reloads] : operandAllocationOrder) {
        _log.trace("Operand {0}: reloads={1}", operand, reloads);
    }

    const auto operandSlotRequirements = getOperandSlotRequirements(computeOpBuffers, bufferInfoCache, sharedBuffers);
    if (!verifyOperandSlotRequirements(operandAllocationOrder, operandSlotRequirements, availableMemory, _log)) {
        return {/*schedule=*/{}, /*reservedSize=*/0, /*sharedExternalBuffers=*/{},
                /*baseAlignment=*/vpux::DEFAULT_CMX_ALIGNMENT};
    }

    vpux::AddressType usedMemory = 0;
    vpux::AddressType baseAlignment = vpux::DEFAULT_CMX_ALIGNMENT;
    const auto memoryAllocations =
            getMemoryAllocations(computeOpBuffers, sharedBuffers, operandAllocationOrder, operandSlotRequirements,
                                 availableMemory, usedMemory, baseAlignment);

    auto predefinedSchedule =
            buildPredefinedSchedule(schedulingLoop, memoryAllocations, sharedBuffers, bufferInfoCache);

    ValueOrderedSet sharedExternalBuffers;
    for (auto bufferId : sharedBuffers) {
        sharedExternalBuffers.insert(bufferInfoCache.buffers[bufferId].value);
    }

    _log.trace(
            "Undefined tiling generated schedule: iterations={0}, usedMemory={1}, sharedBuffers={2}, baseAlignment={3}",
            predefinedSchedule.size(), usedMemory, sharedBuffers.size(), baseAlignment);

    return {std::move(predefinedSchedule), usedMemory, std::move(sharedExternalBuffers), baseAlignment};
}

bool UndefinedTiling::satisfyMemoryConstraint(mlir::Operation* /*op*/, const Shape& /*nTilesOnDim*/,
                                              VPU::LayerCostModel& /*costModel*/, Logger /*log*/) const {
    VPUX_THROW("Not supported");
}

CostInfo UndefinedTiling::calculateCost(mlir::Operation* /*op*/, const OutputTiling& /*tiling*/,
                                        VPU::LayerCostModel& /*costModel*/) const {
    VPUX_THROW("Not supported");
}

uint64_t UndefinedTiling::computeTileOverallCost(uint64_t /*stallCost*/, uint64_t /*dmaInCost*/,
                                                 uint64_t /*computeCost*/, uint64_t /*dmaOutCost*/,
                                                 bool /*isFirstTile*/, bool /*isLastTile*/) const {
    VPUX_THROW("Not supported");
}

Byte UndefinedTiling::calculatePeakMemory(mlir::Operation* /*op*/, const OutputTiling& /*tiling*/,
                                          VPU::LayerCostModel& /*costModel*/) const {
    VPUX_THROW("Not supported");
}

void UndefinedTiling::classifyOperandSize(PeakMemorySizeBuckets& /*buckets*/, Byte /*alignedSize*/,
                                          size_t /*operandIndex*/, size_t /*numOperands*/, bool /*isActivationShared*/,
                                          bool /*isWeightShared*/) const {
    VPUX_THROW("Not supported");
}

Byte UndefinedTiling::computeTilePeakMemory(const PeakMemorySizeBuckets& /*buckets*/, Byte /*weightTableSize*/,
                                            Byte /*sparsityMapSize*/, bool /*isWeightShared*/,
                                            bool /*isWeightTableShared*/) const {
    VPUX_THROW("Not supported");
}

}  // namespace vpux::VPU
