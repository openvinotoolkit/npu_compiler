//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <npu_37xx_nnrt.hpp>

using namespace vpux;
using namespace npu37xx;

namespace vpux {
namespace VPUMI37XX {

constexpr uint32_t defaultTotalStackSize = static_cast<uint32_t>((16_KB).to<vpux::Byte>().count());
constexpr uint32_t defaultActRtCodeSectionSize = static_cast<uint32_t>((1_MB).to<vpux::Byte>().count());
constexpr uint32_t defaultActRtEntry = 0x1C000000;

// ActShave profiling metric mask
//  [0]  BIT_STALL_CYCLE_CNT_EN
//  [1]  BIT_EXEC_INST_CNT_EN
//  [2]  BIT_CLK_CYCLE_CNT_EN
//  [3]  BIT_BRANCH_TAKEN_CNT_EN
//  [4]  BIT_INST_BRKP0_CNT_EN
//  [5]  BIT_INST_BRKP1_CNT_EN
//  [6]  BIT_DATA_BRKP0_CNT_EN
//  [7]  BIT_DATA_BRKP1_CNT_EN
//  [8]  BIT_GO_COUNT_EN
//  [9]  BIT_LSU0_RBYTE_CNT_EN
//  [10] BIT_LSU0_WBYTE_CNT_EN
//  [11] BIT_LSU1_RBYTE_CNT_EN
//  [12] BIT_LSU1_WBYTE_CNT_EN
//  Stall count instructions:
//  [16] SWIH
//  [17] Other interrupts
//  [18] LSU0 Stall (waiting for data)
//  [19] LSU1 Stall (waiting for data)
//  [20] LSU0 Access Stall
//  [21] LSU1 Access Stall
//  [22] Instruction buffer Low Stall
//  [23] Discontinuity Fetch Stall
//  [24] Discontinuity Decode Stall (too much data in instruction buffer at end of delay slots)
//  [25] Discontinuity Starve Stall
//  [26] Instruction buffer Low during discontinuity
//
//  [27] FRC_DURATION_EN
//  [28] FRC_TIMESTAMP_EN
constexpr uint32_t defaultPerfMetricsMask = 0x183C000F;

}  // namespace VPUMI37XX
}  // namespace vpux

//
// MappedInferenceOp
//

void vpux::VPUMI37XX::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();

    nn_public::VpuMappedInference mi = {};

    mi.vpu_nnrt_api_ver = VPU_NNRT_37XX_API_VER;

    auto dmaCountVec = parseIntArrayAttr<int64_t>(getDmaCount());
    VPUX_THROW_WHEN(dmaCountVec.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA lists");
    // in case of 1 tile scheduling lets duplicate count for both lists - we will update it during relocation process
    if (dmaCountVec.size() == 1) {
        for (size_t listIdx = 0; listIdx < nn_public::VPU_MAX_DMA_ENGINES; ++listIdx) {
            mi.dma_tasks[listIdx].count = dmaCountVec[0];
        }
    } else {
        for (size_t listIdx = 0; listIdx < dmaCountVec.size(); ++listIdx) {
            mi.dma_tasks[listIdx].count = dmaCountVec[listIdx];
        }
    }
    mi.invariants.count = getInvariantCount();
    mi.variants.count = getVariantCount();
    mi.act_kernel_ranges.count = getActKernelRangesCount();
    mi.act_kernel_invocations.count = getActKernelInvocationsCount();
    mi.barrier_configs.count = getBarrierCount();

    mi.shv_rt_configs.dpu_perf_mode = nn_public::VpuHWPStatMode::MODE0;
    if (mi.act_kernel_invocations.count) {
        mi.shv_rt_configs.stack_size = VPUMI37XX::defaultTotalStackSize;
        mi.shv_rt_configs.use_schedule_embedded_rt = false;
        mi.shv_rt_configs.code_window_buffer_size = VPUMI37XX::defaultActRtCodeSectionSize;
        mi.shv_rt_configs.runtime_version = 0;
        mi.shv_rt_configs.runtime_entry = VPUMI37XX::defaultActRtEntry;
        mi.shv_rt_configs.perf_metrics_mask = isActShaveProfilingEnabled ? VPUMI37XX::defaultPerfMetricsMask : 0;

        auto actShvRtOp = mlir::dyn_cast<VPUMI37XX::ActShaveRtOp>(getActShaveRt().getDefiningOp());
        if (actShvRtOp) {
            mi.shv_rt_configs.use_schedule_embedded_rt = true;
            mi.shv_rt_configs.code_window_buffer_size = checked_cast<uint32_t>(actShvRtOp.getBinarySize());
            mi.shv_rt_configs.runtime_version = actShvRtOp.getVersion();
            mi.shv_rt_configs.runtime_entry = actShvRtOp.getKernelEntry();
        }

        mi.shv_rt_configs.stack_size = 0;
        for (auto actShaveStack : getActShaveStacks()) {
            mi.shv_rt_configs.stack_size += checked_cast<uint32_t>(
                    mlir::cast<vpux::NDTypeInterface>(actShaveStack.getType()).getTotalAllocSize().count());
        }
    }

    mi.task_storage_counts_.dma_count = nn_public::VPU_DMA_TASK_COUNT;
    mi.task_storage_counts_.dpu_invariant_count = nn_public::VPU_INVARIANT_COUNT;
    mi.task_storage_counts_.dpu_variant_count = nn_public::VPU_VARIANT_COUNT;
    mi.task_storage_counts_.act_range_count = nn_public::VPU_KERNEL_RANGE_COUNT;
    mi.task_storage_counts_.act_invo_count = nn_public::VPU_KERNEL_INVO_COUNT;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&mi);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::MappedInferenceOp::getBinarySize() {
    return sizeof(nn_public::VpuMappedInference);
}

size_t vpux::VPUMI37XX::MappedInferenceOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuMappedInference);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::MappedInferenceOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::MappedInferenceOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::MappedInferenceOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

size_t vpux::VPUMI37XX::MappedInferenceOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == getBarrierTasks()) {
        return offsetof(nn_public::VpuMappedInference, barrier_configs) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuBarrierCountConfig>, address);
    } else if (val == getActKernelInvocations()) {
        return offsetof(nn_public::VpuMappedInference, act_kernel_invocations) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelInvocation>, address);
    } else if (val == getActKernelRanges()) {
        return offsetof(nn_public::VpuMappedInference, act_kernel_ranges) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelRange>, address);
    } else if (val == getVariantTasks()) {
        return offsetof(nn_public::VpuMappedInference, variants) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUVariant>, address);
    } else if (val == getInvariantTasks()) {
        return offsetof(nn_public::VpuMappedInference, invariants) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUInvariant>, address);
    } else if (val == getActShaveRt()) {
        return offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
               offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base);
    }
    for (auto actShaveStack : getActShaveStacks() | indexed) {
        if (val == actShaveStack.value()) {
            const auto index = actShaveStack.index();
            return offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
                   offsetof(nn_public::VpuNNShaveRuntimeConfigs, stack_frames) +
                   index * sizeof(nn_public::VpuNNShaveRuntimeConfigs::stack_frames[0]);
        }
    }
    for (auto listHead : getDmaTasks()) {
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getPort();
        if (listHead == val) {
            return offsetof(nn_public::VpuMappedInference, dma_tasks) +
                   (sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>) * listIdx) +
                   offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address);
        }
    }

    VPUX_THROW("Provided Value is not linked to the MappedInference Op or getOffset does not support it");
}
