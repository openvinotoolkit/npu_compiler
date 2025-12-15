//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/pipeline_options.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux {
void setupPWLMParams50XX(DefaultHWOptions50XX& compilationOptions, LogLevel logLevel) {
    Logger log("wlm-options-parser", logLevel);

    if (compilationOptions.workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES) {
        log.warning("Unsupported compilation option value '{0}' for option '{1}'. Reset to '{2}'.",
                    stringifyEnum(compilationOptions.workloadManagementMode.getValue()),
                    compilationOptions.workloadManagementMode.ArgStr,
                    stringifyEnum(WorkloadManagementMode::FWLM_V1_PAGES));

        compilationOptions.workloadManagementMode = WorkloadManagementMode::FWLM_V1_PAGES;
        compilationOptions.workloadManagementBarrierProgrammingMode =
                WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED;
    }

    bool isWorkloadManagementBarrierProgrammingModeSet =
            compilationOptions.workloadManagementBarrierProgrammingMode.hasValue();

    if (!isWorkloadManagementBarrierProgrammingModeSet) {
        switch (compilationOptions.workloadManagementMode) {
        case WorkloadManagementMode::PWLM_V0_LCA:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::LEGACY;
            break;
        case WorkloadManagementMode::PWLM_V1_BARRIER_FIFO:
        case WorkloadManagementMode::PWLM_V2_PAGES:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::INITIAL_BARRIER_DMAS_SCHEDULED;
            break;
        case WorkloadManagementMode::FWLM_V1_PAGES:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED;
            break;
        default:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::UNKNOWN;
            break;
        }
    }
}
}  // namespace vpux
