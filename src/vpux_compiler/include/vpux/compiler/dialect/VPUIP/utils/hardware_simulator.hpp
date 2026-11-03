//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include <bitset>

namespace vpux::VPUIP::scheduling::simulator {

using OpIndex = size_t;
using FIFOType = VPURT::TaskQueueType;

constexpr size_t MAX_EXECUTOR_COUNT = 64;
using ExecutorMask = std::bitset<MAX_EXECUTOR_COUNT>;

// clang-format off
///
///    The Hardware Model:
///
///    - We assume that most operations operate on tensors big enough to be tiled across all tiles.
///    - Based on that, we assume that every compute resource (DPU, Shave) exists exactly once, except for DMA.
///    - DMA is split into the DMA type (DDR, CMX or unspecified (only NPU3XX)) and a number of channels per DMA type.
///    - All DMA channels can run in parallel.
///    - Jobs can overlap on the same executor. Consequently, the hardware simulator does not ensure legality of the
///      schedule.
///
///    A schematic representation of the hardware model in form of a Gantt chart that displays a possible scheduling:
///
///                ┌──────────────────────────────────────────────┐
///    DPU         │ DPU Task 0 │ DPU Task 1 │ STALL │ DPU Task 2 │
///                ├──────────────────────────────────────────────┤
///    Shave       │ Shave Task 0 │     STALL      │ Shave Task 1 │
///                ├──────────────────────────────────────────────┤
///    DMA DDR [0] │ DMA Task 0 │ DMA Task 1 │    DMA Task 2      │
///                │            ├────────────┼────────────────────┤
///    DMA DDR [1] │ DMA Task 0 │   STALL    │    DMA Task 3      │
///                ├──────────────────────────────────────────────┤
///    DMA CMX [0] │ DMA Task 4 │   STALL    │    DMA Task 5      │
///                ├─────────────────────────┤                    │
///    DMA CMX [1] │       DMA Task 6        │    DMA Task 5      │
///                └──────────────────────────────────────────────┘
///
///    It illustrates the main points. DMAs can have multiple executors and DMA tasks can span 1 or N executors, where N is
///    the number of executors. Executor counts between 1 and N are not supported at the moment. All other executor kinds have exactly 1 executor.
///
// clang-format on
class HardwareSimulator final {
public:
    // Expose inner workings to the unit test.
    friend struct ::llvm::format_provider<HardwareSimulator>;

    struct Slot {
        size_t cycleBegin{0};
        // executorMask[i] == 1 means that executor i is used.
        ExecutorMask executorMask;
    };

    struct ScheduledOpInfo {
        size_t cycleBegin{0};
        size_t cycleCost{0};
        ExecutorMask executorMask;

        size_t getCycleEnd() const noexcept {
            return cycleBegin + cycleCost;
        }
    };

    using ScheduledOpInfoPtr = std::unique_ptr<ScheduledOpInfo>;

    // Note: This class is only exposed for testing purposed but there should be no need for the user to interact with
    // it directly.
    struct HardwareFIFO {
        HardwareFIFO(size_t executorCount);
        Slot findNextAvailableSlot(size_t requiredExecutorCount) const;
        ScheduledOpInfo* scheduleOp(size_t cycleCost, const Slot& slot);
        void insertStall(size_t cycleBegin, size_t cycleCost);

        size_t getExecutorCount() const noexcept {
            return scheduledOpInfosPerExecutor.size();
        }

        ExecutorMask getExecutorMask(size_t i) const {
            return ExecutorMask().set(i);
        }

        ExecutorMask getFullExecutorMask() const {
            ExecutorMask mask;
            for (size_t i = 0; i < getExecutorCount(); ++i) {
                mask.set(i);
            }
            return mask;
        }

        // The jobs are sorted by end cycle.
        // TODO E#-193894: Since we allocate very small objects, we might have to look into a better allocator to reduce
        // fragmentation.
        std::vector<std::vector<ScheduledOpInfoPtr>> scheduledOpInfosPerExecutor;
    };

public:
    /// @brief Construct a HardwareSimulator with the given number of DMA channels per DMA type.
    /// Other executor kinds (DPU, Shave) are assumed to exist exactly once.
    /// @param dmaCount The number of DMA channels per DMA type.
    /// @param dmaChannelTypes The types of DMA channels to create.
    HardwareSimulator(int64_t dmaCount, ArrayRef<VPUIP::DmaChannelType> dmaChannelTypes);

    /// @brief Find the next available slot on the given FIFO type that can run on the required number of executors. It
    /// always starts looking from the back.
    /// @param fifoType The FIFO type to find the slot on.
    /// @param requiredExecutorCount The number of executors required for the operation - must be 1 or the number of
    /// executors.
    /// @return The next available slot.
    Slot findNextAvailableSlot(const FIFOType& fifoType, size_t requiredExecutorCount) const;

    /// @brief Schedule an operation on the given FIFO type at the given slot. It is NOT checked if the slot overlaps
    /// with other previously scheduled operations. Use findNextAvailableSlot to find an empty slot.
    /// @param opIndex The operation index.
    /// @param cycleCost The cycle cost of the operation.
    /// @param fifoType The FIFO type to schedule the operation on.
    /// @param slot The slot to schedule the operation at.
    void scheduleOp(OpIndex opIndex, size_t cycleCost, const FIFOType& fifoType, const Slot& slot);

    /// @brief Insert a stall on all FIFOs starting from cycleBegin for cycleCost cycles. All operations that
    /// start at or after cycleBegin will be shifted by cycleCost cycles. Use findNextAvailableSlot to find an empty
    /// slot.
    /// @param cycleBegin The starting cycle of the stall.
    /// @param cycleCost The cycle cost of the stall.
    void insertStall(size_t cycleBegin, size_t cycleCost);

    ScheduledOpInfo getScheduledOpInfo(OpIndex opIndex) const;

private:
    // Mapping from FIFO type to the actual FIFO.
    std::unordered_map<FIFOType, HardwareFIFO> _FIFOs;
    // Mapping from OpIndex to information about the scheduled job. This raw pointer is safe because every job is
    // allocated on the heap by using unique_ptr. This should not be exposed to the outside of this class.
    std::unordered_map<OpIndex, ScheduledOpInfo*> _scheduledOpInfos;
};

}  // namespace vpux::VPUIP::scheduling::simulator

namespace llvm {
/// Warning: This printer is quite verbose and expensive, use it sparingly.
template <>
struct format_provider<vpux::VPUIP::scheduling::simulator::HardwareSimulator> final {
    static void format(const vpux::VPUIP::scheduling::simulator::HardwareSimulator& hardwareSimulator, raw_ostream& os,
                       StringRef);
};
}  // namespace llvm
