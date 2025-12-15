//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

namespace vpux {
namespace VPUMI40XX {
enum class DmaNnSrcType { DDR, CMX_NN, Count };

//
// TaskMetadata Utils
//

// Sizes are completely governed by compiler
// E#121935: If the following configs vary between arches, then all this info should be moved to arch-specific
// dialects
template <VPURegMapped::TaskType type>
struct MetadataBufferSize {};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::DPUInvariant> {
    static constexpr size_t listCount = 1;
    static constexpr std::array<size_t, listCount> defaultTaskCount = {64};
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::DPUVariant> {
    static constexpr size_t listCount = 1;
    static constexpr std::array<size_t, listCount> defaultTaskCount = {128};
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::ActKernelRange> {
    static constexpr size_t listCount = 2;
    static constexpr std::array<size_t, listCount> defaultTaskCount = {64};
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::ActKernelInvocation> {
    static constexpr size_t listCount = 2;
    static constexpr std::array<size_t, listCount> defaultTaskCount = {64};
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::DMA> {
    static constexpr size_t listCount = 2;
    // DMA does not have default task counts, just a collection of value pairs, out of which one is chosen by logic
    // present in compute_dma_split() max size_t value used as placeholders in order to aid the computation and to
    // preserve uniformity of logic between task types
    static constexpr std::array<size_t, listCount> defaultTaskCount = {std::numeric_limits<size_t>::max(),
                                                                       std::numeric_limits<size_t>::max()};
};

uint64_t computeMaskHi(mlir::ArrayAttr barriers);
uint64_t computeMaskLo(mlir::ArrayAttr barriers);
bool isConfigureBarrierOpType(const mlir::Operation::operand_range& barriers);

constexpr std::pair<size_t, size_t> compute_dma_split(size_t ddr_tasks, size_t cmx_tasks) {
    // [task with CMX src]                       //
    // ^         /                           . ' //
    // |        /                        . '     //
    // |   C   /                     . '         //
    // |      /                  . '             //
    // |     /      B        . '                 //
    // |    /            . '                     //
    // |   /         . '                         //
    // |  /      . '             A               //
    // | /   . '                                 //
    // |/. '                                     //
    // +---------------------------------------> //
    //                      [tasks with DDR src] //

    return ddr_tasks > 2 * cmx_tasks ? std::make_pair(size_t{64}, size_t{16}) :         // A
                   cmx_tasks > ddr_tasks * 2 ? std::make_pair(size_t{16}, size_t{64})   // C
                                             : std::make_pair(size_t{32}, size_t{32});  // B
}

MappedInferenceOp getMPI(mlir::func::FuncOp mainFunc);

// Update indexes in list of operations
size_t reindexList(VPURegMapped::TaskOpInterface head);
template <typename TaskOpType>
void reindexList(VPUMI40XX::MappedInferenceOp mpi, TaskOpType firstTask, size_t fetchTaskTileIdx,
                 size_t fetchTaskListIdx);

VPURegMapped::ExecutionGroupOp getNextGroup(VPURegMapped::ExecutionGroupOp op);

void printIndex(llvm::raw_ostream& os, VPURegMapped::IndexType index, llvm::StringRef head, llvm::StringRef middle,
                llvm::StringRef end);
bool checkBarrierProductionRelationship(mlir::Operation* barr, VPUMI40XX::ExecutableTaskOpInterface exec);

template <typename T>
T getNextOp(T op) {
    auto users = op.getResult().getUsers();
    auto nextOpIt = llvm::find_if(users, [&op](mlir::Operation* user) {
        auto nextTask = mlir::dyn_cast<T>(user);
        return nextTask && (nextTask.getPreviousTask() == op);
    });

    return llvm::cast_if_present<T>(nextOpIt == users.end() ? nullptr : *nextOpIt);
}

size_t reindexEnqueueList(VPURegMapped::EnqueueOp head);
constexpr StringLiteral lastSecondaryTaskInExecutionGroup = "lastSecondaryTaskInExecutionGroup";

uint32_t generateTileMask(mlir::ArrayRef<uint32_t> usedTileIndexes);

void reindexTaskLinkAttrForDMA(VPURegMapped::TaskOpInterface head);

//
// Resolve Task Location utils
//
size_t getTaskBinarySize(VPURegMapped::TaskType taskType, config::ArchKind arch);

VPURegMapped::TaskType convertExecutorKindToExecutableTaskType(VPU::ExecutorKind kind);

}  // namespace VPUMI40XX
}  // namespace vpux
