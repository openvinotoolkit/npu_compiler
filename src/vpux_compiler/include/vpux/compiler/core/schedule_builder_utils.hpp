//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>

#include <vector>

namespace vpux {

// It is beneficial to have at least this many iterations in a loop to consider it for loop-based scheduling
static constexpr size_t MIN_TILING_LOOP_OPS = 5;
static constexpr size_t MIN_VF_LOOP_OPS = 2;
// Only consider vf loops when there are at least this many compute ops in each loop body
static constexpr size_t MIN_VF_LOOP_BODY_COMPUTE_OPS = 2;

enum class AllocationType { COMPUTE = 0, DATA_IN = 1, DATA_OUT = 2 };
inline const char* toString(AllocationType state) {
    switch (state) {
    case AllocationType::COMPUTE:
        return "COMPUTE";
    case AllocationType::DATA_IN:
        return "DATA_IN";
    case AllocationType::DATA_OUT:
        return "DATA_OUT";
    }
    return "Unknown";
}

// Describes a single buffer's properties for allocation and scheduling.
struct BufferDesc {
    BufferDesc(mlir::Value value, vpux::AddressType rawSize, vpux::AddressType rawAlignment)
            : value(value), rawSize(rawSize), rawAlignment(rawAlignment) {
    }

    mlir::Value value;
    vpux::AddressType rawSize;
    vpux::AddressType rawAlignment;
};

// OpAllocationInfo represents internal scheduler representation for operation
struct OpAllocationInfo {
    OpAllocationInfo() = default;

    explicit OpAllocationInfo(size_t opIdx, const VPURT::TaskQueueType& queueType,
                              const SmallVector<mlir::Value>& inBuffers, const SmallVector<mlir::Value>& outBuffers,
                              AllocationType allocationType = AllocationType::COMPUTE, size_t executorDemand = 1UL)
            : opIdx(opIdx),
              queueType(queueType),
              inBuffers(inBuffers),
              outBuffers(outBuffers),
              allocationType(allocationType),
              executorDemand(executorDemand) {
    }

    void printFormat(llvm::raw_ostream& os) const {
        os << "Alloc {"
           << " opIdx_: " << opIdx << ", queueType_: " << queueType.type << ", inBuffers_: [";
        for (const auto& buf : inBuffers) {
            os << buf << " ";
        }
        os << "], outBuffers_: [";
        for (const auto& buf : outBuffers) {
            os << buf << " ";
        }
        os << "],\n        allocationType_: " << toString(allocationType) << " }";
    }

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const OpAllocationInfo& info) {
        info.printFormat(os);
        return os;
    }

    size_t opIdx;
    VPURT::TaskQueueType queueType;
    SmallVector<mlir::Value> inBuffers;
    SmallVector<mlir::Value> outBuffers;
    AllocationType allocationType;
    size_t executorDemand;
};

struct ComputeExplicitSchedule {
    ComputeExplicitSchedule() = default;

    explicit ComputeExplicitSchedule(const OpAllocationInfo& allocInfo, SmallVector<mlir::Value> deallocations,
                                     SmallVector<std::pair<mlir::Value, vpux::AddressType>> allocations)
            : allocInfo(allocInfo), deallocations(std::move(deallocations)), allocations(std::move(allocations)) {
    }

    OpAllocationInfo allocInfo;
    SmallVector<mlir::Value> deallocations;
    SmallVector<std::pair<mlir::Value, vpux::AddressType>> allocations;
};

using IterationSchedule = std::vector<ComputeExplicitSchedule>;
using PredefinedSchedule = std::vector<IterationSchedule>;

// Result of predefined schedule generation for a single compute region.
// Contains the schedule alongside memory layout metadata (size, shared buffers, alignment)
// that the scheduler needs for allocation during loop execution.
struct LoopScheduleResult {
    PredefinedSchedule schedule;
    vpux::AddressType reservedSize = 0;
    ValueOrderedSet sharedExternalBuffers;
    vpux::AddressType baseAlignment = 1;

    bool empty() const {
        return schedule.empty();
    }
};

enum class LoopType { None, Tiling, VF };

inline const char* toString(LoopType state) {
    switch (state) {
    case LoopType::None:
        return "None";
    case LoopType::Tiling:
        return "Tiling";
    case LoopType::VF:
        return "VF";
    default:
        return "Unknown";
    }
}

/*
SchedulingLoop represents repeating blocks of compute graph.
Note: Until SCF introduction loops are stored unrolled since tile sizes cannot be calculated
based on iteration indexes.

The structure is as follows:
1D case:
    Compute region <- Always represents 1 tiled operation with all or subset of tiles
    | - Global common cons and deps for all operations in the region <- deps/cons outside compute region
    \ - SchedulingLoop <- outer loop, in 1D case it is the only loop
        | - Global common cons and deps for all operations in the Loop Plan (same as compute region in 1D case)
        \ - loopBody
            | - LoopBody 0
            |   | - OpAllocationInfo <- Final type describing allocation info of an operation
            |   | - OpAllocationInfo
            |   \ - ...
            \
              - LoopBody 1
                | - OpAllocationInfo
                | - OpAllocationInfo
                \ - ...

For 2D case the structure can be self-nested.
*/

using LoopBody = std::vector<OpAllocationInfo>;
struct SchedulingLoop {
    LoopType type = LoopType::None;
    std::vector<LoopBody> loopBodies;  // Contains operations only if it is inner loop
};

// ComputeRegion is a subgraph of loop iterations that can be scheduled together.
// Loops can be nested inside compute region to represent multi dimensional tiling.
struct ComputeRegion {
    ComputeRegion() = default;

    explicit ComputeRegion(std::unique_ptr<SchedulingLoop> schedulingLoop, SmallVector<size_t> dependencies = {},
                           SmallVector<size_t> consumers = {}, size_t size = 0)
            : schedulingLoop(std::move(schedulingLoop)),
              dependencies(std::move(dependencies)),
              consumers(std::move(consumers)),
              size(size),
              baseAlignment(1) {
    }

    void printFormat(llvm::raw_ostream& os) const {
        os << "ComputeRegion {\n"
           << "  size : " << size;
        os << ", dependencies: [ ";
        for (size_t dep : dependencies) {
            os << dep << " ";
        }
        os << " ], consumers: [ ";
        for (size_t cons : consumers) {
            os << cons << " ";
        }
        os << " ]\n";

        if (schedulingLoop) {
            os << "  SchedulingLoop:\n";
            printSchedulingLoop(os, schedulingLoop, 2);
        } else {
            os << "  No scheduling loop\n";
        }

        os << "}\n";
    }

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ComputeRegion& region) {
        region.printFormat(os);
        return os;
    }

    LoopType getLoopType() const {
        VPUX_THROW_UNLESS(schedulingLoop != nullptr, "ComputeRegion has no scheduling loop");
        return schedulingLoop->type;
    }

    /**
    @brief get loop body by flattened index
    @param index - flattened index of the loop body
    @return reference to LoopBody
    */
    const LoopBody& getLoopBodyAtIndex(size_t index) const {
        VPUX_THROW_UNLESS(schedulingLoop != nullptr, "ComputeRegion has no scheduling loop");
        SchedulingLoop const* currentLoop = schedulingLoop.get();

        VPUX_THROW_UNLESS(index < currentLoop->loopBodies.size(), "Index {0} out of range for loop bodies size {1}",
                          index, currentLoop->loopBodies.size());
        return currentLoop->loopBodies[index];
    }

    static size_t getTotalIterations(const SchedulingLoop* loop) {
        return loop->loopBodies.size();
    }

private:
    static void printSchedulingLoop(llvm::raw_ostream& os, const std::unique_ptr<SchedulingLoop>& loop, int indent) {
        std::string indentStr(indent, ' ');
        os << indentStr << "{ type: " << toString(loop->type) << ", iterations: " << getTotalIterations(loop.get());

        if (!loop->loopBodies.empty()) {
            os << indentStr << "  loopBody: [\n";
            for (size_t i = 0; i < loop->loopBodies.size(); ++i) {
                os << indentStr << "    Iteration " << i << ": [\n";
                for (const auto& allocInfo : loop->loopBodies[i]) {
                    os << indentStr << "      ";
                    allocInfo.printFormat(os);
                    os << "\n";
                }
                os << indentStr << "    ]\n";
            }
            os << indentStr << "  ]\n";
        }
        os << indentStr << "}\n";
    }

public:
    // Nested structure for representing multidimensional tiling
    std::unique_ptr<SchedulingLoop> schedulingLoop;

    // dependencies and consumers outside of region
    SmallVector<size_t> dependencies;
    SmallVector<size_t> consumers;

    vpux::AddressType size;

    // Buffer addresses for every iteration:
    // 0 - first iteration
    // 1 - prefetch
    // for example if everything can be prefetched then both will be filled in with same number of addresses
    // but addresses cannot overlap
    std::pair<SmallVector<vpux::AddressType>, SmallVector<vpux::AddressType>> bufferAddressVec;
    // Common buffers among all iterations in the compute region
    ValueOrderedSet sharedExternalBuffers;
    vpux::AddressType baseAlignment;
    size_t prefetchOpCount = 0;
};

using ComputeRegionVec = std::vector<ComputeRegion>;

// Aggregates all loop scheduling data needed by FeasibleMemoryScheduler.
// Produced externally (before scheduler construction) and passed in as a value.
// Separates schedule generation from schedule execution.
struct ComputeRegionsSchedule {
    // Predefined schedules per region index (only regions with valid schedules are present)
    llvm::DenseMap<size_t, LoopScheduleResult> scheduleResults;
    // Operation indices wrapped in loop regions (require loop-based scheduling)
    llvm::DenseSet<size_t> loopRegionInd;
    // Data-in operation indices eligible for prefetching
    llvm::DenseSet<size_t> loopPrefetchInd;
};
ComputeRegionVec getComputeRegionsFromAsyncExec(AliasesInfo& aliasInfo, AsyncDepsInfo& depsInfo,
                                                Logger log = Logger::global());

}  // namespace vpux
