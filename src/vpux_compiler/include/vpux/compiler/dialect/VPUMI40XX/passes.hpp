//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUMI40XX {

//
// Passes
//

std::unique_ptr<mlir::Pass> createSetupProfilingVPUMI40XXPass(const std::string& enableDmaProfiling = "false",
                                                              Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierComputationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> reorderMappedInferenceOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveTaskLocationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierTopologicalMappingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupExecutionOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnGroupExecutionOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddFetchOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveWLMTaskLocationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateFinalBarrierPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddEnqueueOpsPass(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFetchTaskOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinkEnqueueTargetsPass(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinkAllOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollEnqueueOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinkEnqueueOpsForSameBarrierPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitEnqueueOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddBootstrapBarriersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddBootstrapWorkItemsPass(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNextSameIdAssignmentPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddPlatformInfoPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpStatisticsOfWlmOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddMappedInferenceVersionOpPass(Logger log = Logger::global(),
                                                                  uint32_t versionMajor = 0, uint32_t versionMinor = 0,
                                                                  uint32_t versionPatch = 0);
std::unique_ptr<mlir::Pass> createAddBarrierConfigurationOps(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        WorkloadManagementBarrierProgrammingMode WorkloadManagementBarrierProgrammingMode =
                WorkloadManagementBarrierProgrammingMode::LEGACY,
        Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace VPUMI40XX
}  // namespace vpux
