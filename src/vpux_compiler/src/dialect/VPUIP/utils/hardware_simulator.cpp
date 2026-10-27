//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/hardware_simulator.hpp"
#include "vpux/compiler/utils/dma.hpp"

namespace std {
using ScheduledOpInfoPtr = vpux::VPUIP::scheduling::simulator::HardwareSimulator::ScheduledOpInfoPtr;
template <>
struct less<ScheduledOpInfoPtr> final {
    bool operator()(const ScheduledOpInfoPtr& a, const ScheduledOpInfoPtr& b) const {
        return a->getCycleEnd() < b->getCycleEnd();
    }
};
}  // namespace std

namespace vpux::VPUIP::scheduling::simulator {

HardwareSimulator::HardwareFIFO::HardwareFIFO(size_t executorCount): scheduledOpInfosPerExecutor(executorCount) {
    assert(1 <= executorCount && executorCount <= MAX_EXECUTOR_COUNT &&
           "Number of executors must be between 1 and 64 (inclusive)");
}

HardwareSimulator::Slot HardwareSimulator::HardwareFIFO::findNextAvailableSlot(size_t requiredExecutorCount) const {
    VPUX_THROW_UNLESS(requiredExecutorCount == 1 || requiredExecutorCount == getExecutorCount(),
                      "Invalid requiredExecutorCount {0}, must be 1 or {1}", requiredExecutorCount, getExecutorCount());

    const auto executorComparator = [](const std::vector<ScheduledOpInfoPtr>& a,
                                       const std::vector<ScheduledOpInfoPtr>& b) {
        const auto aEnd = a.empty() ? 0 : a.back()->getCycleEnd();
        const auto bEnd = b.empty() ? 0 : b.back()->getCycleEnd();
        return aEnd < bEnd;
    };

    // If we only need a single executor, we pick the one that becomes available the earliest. If we need all executors,
    // we pick the time when all executors are available the earliest. This is the maximum of the individual executor
    // end times.
    if (requiredExecutorCount == 1) {
        const auto it = std::min_element(scheduledOpInfosPerExecutor.begin(), scheduledOpInfosPerExecutor.end(),
                                         executorComparator);
        const auto cycle = it->empty() ? 0 : it->back()->getCycleEnd();
        const auto mask = getExecutorMask(std::distance(scheduledOpInfosPerExecutor.begin(), it));
        return {cycle, mask};
    } else {
        const auto it = std::max_element(scheduledOpInfosPerExecutor.begin(), scheduledOpInfosPerExecutor.end(),
                                         executorComparator);
        const auto cycle = it->empty() ? 0 : it->back()->getCycleEnd();
        const auto mask = getFullExecutorMask();
        return {cycle, mask};
    }
}

HardwareSimulator::ScheduledOpInfo* HardwareSimulator::HardwareFIFO::scheduleOp(size_t cycleCost, const Slot& slot) {
    ScheduledOpInfo* scheduledOpInfoPtr = nullptr;

    const size_t popCount = slot.executorMask.count();
    VPUX_THROW_UNLESS(popCount == 1 || popCount == getExecutorCount(),
                      "Invalid executor count in slot.executorMask {0}, must be 1 or {1}", popCount,
                      getExecutorCount());

    for (size_t i = 0; i < getExecutorCount(); ++i) {
        const auto isSet = slot.executorMask.test(i);
        if (!isSet) {
            continue;
        }

        auto scheduledOpInfo =
                std::make_unique<ScheduledOpInfo>(ScheduledOpInfo{slot.cycleBegin, cycleCost, slot.executorMask});
        // We create a new ScheduledOpInfo for every executor i if slot.executorMask[i] == 1. However, we return
        // the pointer to only one of them. The user should not care about which one.
        if (scheduledOpInfoPtr == nullptr) {
            scheduledOpInfoPtr = scheduledOpInfo.get();
        }

        const auto it =
                std::upper_bound(scheduledOpInfosPerExecutor.at(i).begin(), scheduledOpInfosPerExecutor.at(i).end(),
                                 scheduledOpInfo, std::less<ScheduledOpInfoPtr>());
        scheduledOpInfosPerExecutor.at(i).insert(it, std::move(scheduledOpInfo));
    }

    assert(scheduledOpInfoPtr != nullptr && "scheduledOpInfoPtr should not be null");
    return scheduledOpInfoPtr;
}

void HardwareSimulator::HardwareFIFO::insertStall(size_t cycleBegin, size_t cycleCost) {
    for (auto& scheduledOpInfos : scheduledOpInfosPerExecutor) {
        for (auto& scheduledOpInfo : scheduledOpInfos) {
            if (scheduledOpInfo->cycleBegin >= cycleBegin) {
                scheduledOpInfo->cycleBegin += cycleCost;
            }
        }
    }
}

HardwareSimulator::HardwareSimulator(int64_t dmaCount, ArrayRef<VPUIP::DmaChannelType> dmaChannelTypes) {
    _FIFOs.emplace(FIFOType(config::ExecutorKind::NCE), HardwareFIFO(1));
    _FIFOs.emplace(FIFOType(config::ExecutorKind::DPU), HardwareFIFO(1));
    _FIFOs.emplace(FIFOType(config::ExecutorKind::SHAVE_NN), HardwareFIFO(1));
    _FIFOs.emplace(FIFOType(config::ExecutorKind::SHAVE_ACT), HardwareFIFO(1));
    _FIFOs.emplace(FIFOType(config::ExecutorKind::M2I), HardwareFIFO(1));

    for (const auto& dmaChannelType : dmaChannelTypes) {
        FIFOType queueType(config::ExecutorKind::DMA_NN, static_cast<uint8_t>(getDMAQueueIdEncoding(dmaChannelType)));
        _FIFOs.emplace(queueType, HardwareFIFO(dmaCount));
    }

    // TODO E#-193891: Create FIFOs for spilling here.
}

HardwareSimulator::Slot HardwareSimulator::findNextAvailableSlot(const FIFOType& fifoType,
                                                                 size_t requiredExecutorCount) const {
    const auto it = _FIFOs.find(fifoType);
    VPUX_THROW_WHEN(it == _FIFOs.end(), "No FIFO of type {0} found!", fifoType);
    const auto [cycle, executorMask] = it->second.findNextAvailableSlot(requiredExecutorCount);
    return {cycle, executorMask};
}

void HardwareSimulator::scheduleOp(OpIndex opIndex, size_t cycleCost, const FIFOType& fifoType, const Slot& slot) {
    auto scheduledOpInfo = _FIFOs.at(fifoType).scheduleOp(cycleCost, slot);
    VPUX_THROW_UNLESS(_scheduledOpInfos.count(opIndex) == 0, "OpIndex {0} is already scheduled!", opIndex);
    _scheduledOpInfos[opIndex] = scheduledOpInfo;
}

void HardwareSimulator::insertStall(size_t cycleBegin, size_t cycleCost) {
    for (auto& [_, fifo] : _FIFOs) {
        fifo.insertStall(cycleBegin, cycleCost);
    }
}

HardwareSimulator::ScheduledOpInfo HardwareSimulator::getScheduledOpInfo(OpIndex opIndex) const {
    const auto it = _scheduledOpInfos.find(opIndex);
    VPUX_THROW_WHEN(it == _scheduledOpInfos.end(), "OpIndex {0} is not scheduled!", opIndex);
    return ScheduledOpInfo{it->second->cycleBegin, it->second->cycleCost, it->second->executorMask};
}
}  // namespace vpux::VPUIP::scheduling::simulator

namespace llvm {

constexpr size_t FIFO_TYPE_COLUMN_WIDTH = 20;
constexpr size_t INT_COLUMN_WIDTH = 12;
constexpr size_t EXECUTOR_MASK_COLUMN_WIDTH = vpux::VPUIP::scheduling::simulator::MAX_EXECUTOR_COUNT;
constexpr size_t SPACE_COUNT = FIFO_TYPE_COLUMN_WIDTH + 1 + INT_COLUMN_WIDTH;

void format_provider<vpux::VPUIP::scheduling::simulator::HardwareSimulator>::format(
        const vpux::VPUIP::scheduling::simulator::HardwareSimulator& hardwareSimulator, raw_ostream& os, StringRef) {
    // clang-format off
    // "{0, +$FIFO_TYPE_COLUMN_WIDTH} {1, +$INT_COLUMN_WIDTH} {2, +$INT_COLUMN_WIDTH} {3, +$EXECUTOR_MASK_COLUMN_WIDTH}\n"
    // clang-format on
    const std::string headerFormatString =
            formatv("{{0, +{0}} {{1, +{1}} {{2, +{2}} {{3, +{3}}\n", FIFO_TYPE_COLUMN_WIDTH, INT_COLUMN_WIDTH,
                    INT_COLUMN_WIDTH, EXECUTOR_MASK_COLUMN_WIDTH)
                    .str();
    // "{0, +$FIFO_TYPE_COLUMN_WIDTH}\n"
    const std::string queueFormatString = formatv("{{0, +{0}}\n", FIFO_TYPE_COLUMN_WIDTH).str();
    // "{0, +$INT_COLUMN_WIDTH} {1, +$INT_COLUMN_WIDTH} {2, +$EXECUTOR_MASK_COLUMN_WIDTH}\n"
    const std::string fifoFormatString =
            formatv("{{0, +{0}} {{1, +{1}} {{2, +{2}}\n", SPACE_COUNT, INT_COLUMN_WIDTH, EXECUTOR_MASK_COLUMN_WIDTH)
                    .str();

    os << formatv(headerFormatString.c_str(), "FIFOType[ExecutorId]", "BeginCycle", "EndCycle", "ExecutorMask");

    for (const auto& [type, fifo] : hardwareSimulator._FIFOs) {
        for (size_t executorId = 0; executorId < fifo.getExecutorCount(); ++executorId) {
            auto queueString = formatv("{0}[{1}]", type, executorId);
            assert(queueString.str().length() <= 20 && "Queue string longer than expected");
            os << formatv(queueFormatString.c_str(), std::move(queueString));

            if (fifo.scheduledOpInfosPerExecutor[executorId].empty()) {
                os << formatv(fifoFormatString.c_str(), "-", "-", "-");
                continue;
            }

            for (const auto& scheduledOpInfoPtr : fifo.scheduledOpInfosPerExecutor[executorId]) {
                const auto& [cycleBegin, cycleEnd, executorMask] = *scheduledOpInfoPtr;
                os << formatv(fifoFormatString.c_str(), cycleBegin, cycleEnd, executorMask.to_string());
            }
        }
    }
}
}  // namespace llvm
