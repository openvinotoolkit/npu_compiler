//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

namespace vpux {

void setupParamsAccordingToOptimizationLevel(int optimizationLevel, DefaultHWOptions40XX& compilationOptions,
                                             bool useWlm) {
    //
    // Non-WLM params
    //

    switch (optimizationLevel) {
    case 0:
    case 1:
    case 2: {
        break;
    }
    case 3: {
        overwriteIfUnset(compilationOptions.enableReduceNumTilesForSmallModelsPass, true);
        break;
    }
    default:
        VPUX_THROW("Unexpected optimization-level. Actual value = {0}\n"
                   "Possible values: 0 - optimization for compilation time, "
                   "1 - optimization for execution time (default), 2 - high optimization for execution time, 3 - "
                   "optimization for maximaze HW utilization, may affect compilation time and memory footprint",
                   optimizationLevel);
        break;
    }

    //
    // WLM-related params
    //

    if (!useWlm) {
        compilationOptions.workloadManagementEnable = false;
        return;
    }
    bool isworkloadManagementEnableSet = compilationOptions.workloadManagementEnable.hasValue() ? true : false;

    if (!isworkloadManagementEnableSet) {
        std::optional<int> originalValueBarrierCountThreshold = std::nullopt;
        std::optional<WorkloadManagementMode> originalValueWorkloadManagementMode = std::nullopt;

        if (compilationOptions.workloadManagementMode.hasValue()) {
            originalValueWorkloadManagementMode = compilationOptions.workloadManagementMode;
        }

        if (compilationOptions.workloadManagementBarrierCountThreshold.hasValue()) {
            originalValueBarrierCountThreshold = compilationOptions.workloadManagementBarrierCountThreshold;
        }

        // we do not have default value for workloadManagementMode on NPU40XX. Need to set it explicitly
        // E166333
        switch (optimizationLevel) {
        case 0:
            compilationOptions.workloadManagementEnable = false;
            compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
            break;
        case 1:
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
            break;
        case 2: {
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
            compilationOptions.workloadManagementBarrierCountThreshold = std::numeric_limits<int>::max();
            break;
        }
        case 3: {
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementBarrierCountThreshold = std::numeric_limits<int>::max();
            compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V1_BARRIER_FIFO;
            compilationOptions.workloadManagementDmaFifoType = DMAFifoType::HW;
            break;
        }
        default:
            VPUX_THROW("Unexpected optimization-level. Actual value = {0}\n"
                       "Possible values: 0 - optimization for compilation time, "
                       "1 - optimization for execution time (default), 2 - high optimization for execution time, 3 - "
                       "optimization for maximaze HW utilization, may affect compilation time and memory footprint",
                       optimizationLevel);
            break;
        }

        if (originalValueWorkloadManagementMode.has_value()) {
            compilationOptions.workloadManagementMode = originalValueWorkloadManagementMode.value();
        }

        if (originalValueBarrierCountThreshold.has_value()) {
            compilationOptions.workloadManagementBarrierCountThreshold = originalValueBarrierCountThreshold.value();
        }
    }
}

void setupPWLMParams(DefaultHWOptions40XX& compilationOptions, LogLevel logLevel) {
    Logger log("wlm-options-parser", logLevel);

    if (compilationOptions.workloadManagementMode != WorkloadManagementMode::PWLM_V0_LCA &&
        compilationOptions.workloadManagementMode != WorkloadManagementMode::PWLM_V1_BARRIER_FIFO) {
        log.warning("Unsupported compilation option value '{0}' for option '{1}'. Reset to '{2}'.",
                    stringifyEnum(compilationOptions.workloadManagementMode.getValue()),
                    compilationOptions.workloadManagementMode.ArgStr,
                    stringifyEnum(WorkloadManagementMode::PWLM_V0_LCA));
        compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
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
        default:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::UNKNOWN;
            break;
        }
    }
}

}  // namespace vpux
