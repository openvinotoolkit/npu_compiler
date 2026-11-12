//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <cstdint>
#include <npu_60xx_nnrt.hpp>

namespace vpux {
namespace NPUReg60XX {

constexpr uint64_t DEFAULT_ACT_RT_ENTRY = 0x1C000000;
constexpr uint32_t DEFAULT_PERF_METRICS_MASK = 0x1800E006;

template <typename OpType>
void fillNNrtConfig(npu60xx::nn_public::VpuNNShaveRuntimeConfigs& shvRtConfigs, mlir::Operation* op,
                    std::optional<mlir::SymbolRefAttr> getActShaveRt, std::optional<uint64_t> shaveStacksSize,
                    bool isActShaveProfilingEnabled, bool getIsActKernelInvocations, ArrayRef<uint32_t> stackFrames) {
    if (getIsActKernelInvocations) {
        // TODO: E#-74314 nnActEntry.60xx.elf has a .versiondata section that contains a single uint32_t
        // This should be read and set in mi.shvRtConfigs_.runtimeVersion_
        shvRtConfigs.runtime_version = 0;
        shvRtConfigs.runtime_entry = NPUReg60XX::DEFAULT_ACT_RT_ENTRY;
        shvRtConfigs.perf_metrics_mask = isActShaveProfilingEnabled ? NPUReg60XX::DEFAULT_PERF_METRICS_MASK : 0;

        if (getActShaveRt.has_value()) {
            auto actShaveRtRef = ELF::lookupNearestSymbolFrom(op, getActShaveRt.value());
            auto actShaveRtOp = mlir::cast<OpType>(actShaveRtRef);

            shvRtConfigs.runtime_entry = actShaveRtOp.getKernelEntry();
            shvRtConfigs.runtime_version = actShaveRtOp.getVersion();
        }
    }

    for (auto stackFrame : enumerate(stackFrames)) {
        shvRtConfigs.stack_frames[stackFrame.index()] = stackFrame.value();
    }

    shvRtConfigs.stack_size = checked_cast<uint32_t>(shaveStacksSize.value_or(0));
}

VPUMI40XX::TaskDescriptorConstraintsPassArgs getTaskDescriptorConstraintsArgs();

}  // namespace NPUReg60XX
}  // namespace vpux
