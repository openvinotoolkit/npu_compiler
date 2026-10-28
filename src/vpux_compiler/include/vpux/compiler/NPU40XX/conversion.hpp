//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"

#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace arch40xx {

//
// pipelines
//

void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                 const BackendCompilationOptions40XX& backendCompilationOptions,
                                 Logger log = Logger::global(),
                                 VPU::DPUDryRunMode dpuDryRunMode = VPU::DPUDryRunMode::NONE);

void elfSubsetPipelineVPUMI(mlir::OpPassManager& pm, config::ArchKind arch,
                            WorkloadManagementMode workloadManagementMode, bool enableDumpStatisticsOfWlmOps,
                            WorkloadManagementBarrierProgrammingMode workloadManagementBarrierProgrammingMode,
                            const Logger& log);

void elfSubsetPipelineVPUASM(mlir::OpPassManager& pm, bool disableDmaSwFifo, const Logger& log);

//
// Registration
//

void registerConversionPipeline();

}  // namespace arch40xx
}  // namespace vpux
