//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux {
namespace NPUReg40XX {

constexpr auto defaultActRtCodeSectionSize = static_cast<uint32_t>((1_MB).to<vpux::Byte>().count());
constexpr uint64_t defaultActRtEntry = 0x1C000000;

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
//  Aggregated stall counters:
//  [13] LSU0_STALLS_CNT_EN
//  [14] LSU1_STALLS_CNT_EN
//  [15] INST_STALLS_CNT_EN
//  Stall count instructions (for BIT_STALL_CYCLE_CNT_EN):
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
constexpr uint32_t defaultPerfMetricsMask = 0x1800E006;

uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareBufferOp buffer);
uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareTaskBufferOp taskBuffer);

template <class OpType>
OpType getOpFrom(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr);

uint32_t getKernelEntry(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr);
uint64_t getKernelTextSize(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr);
llvm::StringRef getKernelPath(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> kernelPath,
                              mlir::SymbolRefAttr taskType);
npu40xx::nn_public::VpuActWLType getActWLType(mlir::SymbolRefAttr taskType);

template <typename OpType>
void fillNNrtConfig(npu40xx::nn_public::VpuNNShaveRuntimeConfigs& shv_rt_configs, mlir::Operation* op,
                    std::optional<mlir::SymbolRefAttr> getActShaveRt, std::optional<uint64_t> shaveStacksSize,
                    bool isActShaveProfilingEnabled, bool getIsActKernelInvocations,
                    std::optional<std::pair<uint32_t, uint32_t>> stackFrames) {
    shv_rt_configs.dpu_perf_mode = npu40xx::nn_public::VpuHWPStatMode::MODE3;
    if (getIsActKernelInvocations) {
        shv_rt_configs.use_schedule_embedded_rt = false;
        shv_rt_configs.code_window_buffer_size = NPUReg40XX::defaultActRtCodeSectionSize;
        // TODO: E#74314 nnActEntry.40xx.elf has a .versiondata section that contains a single uint32_t
        // This should be read and set in mi.shvRtConfigs_.runtimeVersion_
        shv_rt_configs.runtime_version = 0;
        shv_rt_configs.runtime_entry = NPUReg40XX::defaultActRtEntry;
        shv_rt_configs.perf_metrics_mask = isActShaveProfilingEnabled ? NPUReg40XX::defaultPerfMetricsMask : 0;

        if (getActShaveRt.has_value()) {
            auto actShaveRtRef = mlir::SymbolTable::lookupNearestSymbolFrom(op, getActShaveRt.value());
            auto actShaveRtOp = mlir::cast<OpType>(actShaveRtRef);

            shv_rt_configs.use_schedule_embedded_rt = true;
            shv_rt_configs.code_window_buffer_size =
                    checked_cast<uint32_t>(actShaveRtOp.getBinarySize(VPU::ArchKind::UNKNOWN));
            shv_rt_configs.runtime_entry = actShaveRtOp.getKernelEntry();
            shv_rt_configs.runtime_version = actShaveRtOp.getVersion();
        }
    }

    if (stackFrames.has_value()) {
        shv_rt_configs.stack_frames[0] = stackFrames->first;
        shv_rt_configs.stack_frames[1] = stackFrames->second;
    }

    shv_rt_configs.stack_size = shaveStacksSize.value_or(0);
}

}  // namespace NPUReg40XX
}  // namespace vpux
