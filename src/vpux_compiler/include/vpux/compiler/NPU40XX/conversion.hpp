//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"

#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace arch40xx {

//
// passes
//

//
// pipelines
//

void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                 const BackendCompilationOptions40XX& backendCompilationOptions,
                                 Logger log = Logger::global(),
                                 VPU::DPUDryRunMode dpuDryRunMode = VPU::DPUDryRunMode::NONE);

void elfSubsetPipelineVPUMI(mlir::OpPassManager& pm, bool workloadManagementEnable,
                            WorkloadManagementMode workloadManagementMode, bool enableDumpStatisticsOfWlmOps,
                            const Logger& log);

void elfSubsetPipelineVPUASM(mlir::OpPassManager& pm, bool workloadManagementEnable, const Logger& log);

//
// Registration
//

void registerConversionPipeline();
void registerConversionPasses();

}  // namespace arch40xx
}  // namespace vpux
