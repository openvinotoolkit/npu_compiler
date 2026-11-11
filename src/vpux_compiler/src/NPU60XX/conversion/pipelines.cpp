//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU60XX/conversion.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/utils.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

#include <mlir/Transforms/Passes.h>
#include <npu_60xx_nnrt.hpp>

using namespace vpux;

//
// buildLowerVPUIP2ELFPipeline
//

void vpux::arch60xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                                 const BackendCompilationOptions60XX& backendCompilationOptions,
                                                 Logger log) {
    log.info("BackendCompilationOptions:\n"
             "  workloadManagementMode = {0}\n"
             "  workloadManagementBarrierProgrammingMode = {1}\n"
             "  enableMemorySideCache = {2}\n"
             "  enableDMAProfiling = {3}\n"
             "  enableShaveDDRAccessOptimization = {4}\n",
             stringifyEnum(backendCompilationOptions.workloadManagementMode),
             stringifyEnum(backendCompilationOptions.workloadManagementBarrierProgrammingMode),
             backendCompilationOptions.enableMemorySideCache, backendCompilationOptions.enableDMAProfiling,
             backendCompilationOptions.enableShaveDDRAccessOptimization);

    pm.addPass(VPUMI40XX::createAddPlatformInfoPass(log));
    pm.addPass(
            VPUMI40XX::createSetupTaskDescriptorConstraintsPass(NPUReg60XX::getTaskDescriptorConstraintsArgs(), log));

    pm.addPass(createConvertVPUIP2VPUMI40XXPass(log, backendCompilationOptions.enableMemorySideCache,
                                                backendCompilationOptions.allocateDDRStackFrames));
    pm.addPass(VPUMI40XX::createSetupProfilingVPUMI40XXPass(backendCompilationOptions.enableDMAProfiling, log));
    pm.addPass(mlir::createCanonicalizerPass());
    arch40xx::elfSubsetPipelineVPUMI(pm, config::ArchKind::NPU60XX, backendCompilationOptions.workloadManagementMode,
                                     backendCompilationOptions.enableDumpStatisticsOfWlmOps,
                                     backendCompilationOptions.workloadManagementBarrierProgrammingMode, log);

    pm.addPass(VPUMI40XX::createAddMappedInferenceVersionOpPass(
            log, VPU_NNRT_40XX_API_VER_MAJOR, VPU_NNRT_40XX_API_VER_MINOR, VPU_NNRT_40XX_API_VER_PATCH));

    arch40xx::elfSubsetPipelineVPUASM(
            pm, backendCompilationOptions.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES, log);

    // Emit the HostParsedInference structure as a dedicated ELF section (NPU6 and newer).
    pm.addPass(VPUASM::createAddHostParsedInferencePass(log));

    pm.addPass(VPUIPDPU::createExpandDPUConfigPass(log));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(
            log, std::string(backendCompilationOptions.enableShaveDDRAccessOptimization) == "true"));
    pm.addPass(createConvertVPUASM2NPUReg60XXPass(log, backendCompilationOptions.modelIdentifier));
    pm.addPass(createConvertVPUIPDPU2NPUReg60XXPass(log));

    pm.addPass(VPURegMapped::createDeduceDynamicMappedInferenceVersionPass(log));
    pm.addPass(ELF::createAddCompatibilityStringPass(log));

    pm.addPass(ELF::createHandleAlignmentRequirementsPass(log));
    pm.addPass(ELF::createSetOpOffsetsPass(log));
    pm.addPass(ELF::createSetCMXSymbolValuePass(log));
    pm.addPass(ELF::createAddRelocationsForDynamicStridesDMAsPass(log));
    pm.addPass(ELF::createAddELFRelocationsPass(log));
    pm.addPass(ELF::createRemoveRedundantRegisterSectionsPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
}

void vpux::arch60xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<BackendCompilationOptions60XX>(
            "lower-VPUIP-to-ELF", "Performs full lowering from the VPUIP Dialect to ELF for NPU60XX arch IR",
            [](mlir::OpPassManager& pm, const BackendCompilationOptions60XX& options) {
                vpux::arch60xx::buildLowerVPUIP2ELFPipeline(pm, options);
            });
}
