//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU50XX/conversion.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/abi_version.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

#include <npu_40xx_nnrt.hpp>

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// buildLowerVPUIP2ELFPipeline
//

void vpux::arch50xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                                 const BackendCompilationOptions50XX& backendCompilationOptions,
                                                 Logger log) {
    log.info("BackendCompilationOptions:\n"
             "  workloadManagementEnable = {0}\n"
             "  workloadManagementMode = V{1}\n"
             "  workloadManagementBarrierCountThreshold = {2}\n"
             "  enableMemorySideCache = {3}\n"
             "  enableDMAProfiling = {4}\n"
             "  enableShaveDDRAccessOptimization = {5}\n"
             "  workloadManagementBarrierProgrammingMode = {6}\n"
             "  workloadManagementDmaFifoType = {7}\n",
             backendCompilationOptions.workloadManagementEnable,
             static_cast<int>(backendCompilationOptions.workloadManagementMode.getValue()),
             backendCompilationOptions.workloadManagementBarrierCountThreshold,
             backendCompilationOptions.enableMemorySideCache, backendCompilationOptions.enableDMAProfiling,
             backendCompilationOptions.enableShaveDDRAccessOptimization,
             stringifyEnum(backendCompilationOptions.workloadManagementBarrierProgrammingMode),
             stringifyEnum(backendCompilationOptions.workloadManagementDmaFifoType));

    pm.addPass(VPUMI40XX::createAddPlatformInfoPass(log));

    // Below VPURT WLM passes are placed in LowerVPUIP2ELF pipeline to leverage BarrierInfo
    // working on VPURT dialect and order barriers in a way suitable for WLM
    // Those can be moved to the end of VPURT once WLM rollback does not happen.
    // Currently only ELF backend is retriggered during rollback and IR after VPURT
    // needs to be left in a state suitable for nonWLM flow.
    if (backendCompilationOptions.workloadManagementEnable &&
        backendCompilationOptions.workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES) {
        if (backendCompilationOptions.workloadManagementMode != WorkloadManagementMode::PWLM_V0_LCA) {
            pm.addPass(VPURT::createFindWlmEnqueueBarrierPass(
                    backendCompilationOptions.workloadManagementMode,
                    backendCompilationOptions.workloadManagementDmaFifoType == DMAFifoType::HW, log));
        } else {
            pm.addPass(VPURT::createOrderBarriersForWlmPass(log));
        }
    }

    pm.addPass(createConvertVPUIP2VPUMI40XXPass(log, backendCompilationOptions.enableMemorySideCache,
                                                backendCompilationOptions.allocateShaveStackFrames));
    pm.addPass(VPUMI40XX::createSetupProfilingVPUMI40XXPass(backendCompilationOptions.enableDMAProfiling, log));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ELF::createAddABIVersionPass(log, NPUReg50XX::ABI_VERSION_MAJOR, NPUReg50XX::ABI_VERSION_MINOR,
                                            NPUReg50XX::ABI_VERSION_PATCH));
    arch40xx::elfSubsetPipelineVPUMI(pm, backendCompilationOptions.workloadManagementEnable,
                                     backendCompilationOptions.workloadManagementMode,
                                     backendCompilationOptions.enableDumpStatisticsOfWlmOps,
                                     backendCompilationOptions.workloadManagementBarrierProgrammingMode, log);

    pm.addPass(VPUMI40XX::createAddMappedInferenceVersionOpPass(
            log, VPU_NNRT_40XX_API_VER_MAJOR, VPU_NNRT_40XX_API_VER_MINOR, VPU_NNRT_40XX_API_VER_PATCH));

    arch40xx::elfSubsetPipelineVPUASM(pm, backendCompilationOptions.workloadManagementEnable,
                                      backendCompilationOptions.workloadManagementDmaFifoType == DMAFifoType::HW, log);

    pm.addPass(VPUIPDPU::createExpandDPUConfigPass(log, backendCompilationOptions.npu5PPEBackwardsCompatibilityMode));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(log, backendCompilationOptions.enableShaveDDRAccessOptimization));
    pm.addPass(createConvertVPUASM2NPUReg50XXPass(log, backendCompilationOptions.modelIdentifier));
    pm.addPass(createConvertVPUIPDPU2NPUReg50XXPass(log, backendCompilationOptions.npu5PPEBackwardsCompatibilityMode));

    pm.addPass(VPURegMapped::createDeduceDynamicMappedInferenceVersionPass(log));

    pm.addPass(ELF::createHandleAlignmentRequirementsPass(log));
    pm.addPass(ELF::createSetOpOffsetsPass(log));
    pm.addPass(ELF::createSetCMXSymbolValuePass(
            log, npu40xx::nn_public::VPU_WORKSPACE_ADDR, npu40xx::nn_public::VPU_WORKSPACE_SIZE,
            npu40xx::VPU_METADATA_STORAGE_START, npu40xx::nn_public::VPU_METADATA_SIZE));
    pm.addPass(ELF::createAddELFRelocationsPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
}

void vpux::arch50xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<BackendCompilationOptions50XX>(
            "lower-VPUIP-to-ELF", "Performs full lowering from the VPUIP Dialect to ELF for NPU50XX arch IR",
            [](mlir::OpPassManager& pm, const BackendCompilationOptions50XX& options) {
                vpux::arch50xx::buildLowerVPUIP2ELFPipeline(pm, options);
            });
}
