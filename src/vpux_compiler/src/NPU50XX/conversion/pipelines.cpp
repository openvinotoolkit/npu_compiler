//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU50XX/conversion.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

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
             "  workloadManagementMode = {1}\n"
             "  workloadManagementBarrierProgrammingMode = {2}\n"
             "  workloadManagementDmaFifoType = {3}\n"
             "  enableMemorySideCache = {4}\n"
             "  enableDMAProfiling = {5}\n"
             "  enableShaveDDRAccessOptimization = {6}\n",
             backendCompilationOptions.workloadManagementEnable,
             stringifyEnum(backendCompilationOptions.workloadManagementMode),
             stringifyEnum(backendCompilationOptions.workloadManagementBarrierProgrammingMode),
             stringifyEnum(backendCompilationOptions.workloadManagementDmaFifoType),
             backendCompilationOptions.enableMemorySideCache, backendCompilationOptions.enableDMAProfiling,
             backendCompilationOptions.enableShaveDDRAccessOptimization);

    pm.addPass(VPUMI40XX::createAddPlatformInfoPass(log));

    pm.addPass(createConvertVPUIP2VPUMI40XXPass(log, backendCompilationOptions.enableMemorySideCache,
                                                backendCompilationOptions.allocateDDRStackFrames));
    pm.addPass(VPUMI40XX::createSetupProfilingVPUMI40XXPass(backendCompilationOptions.enableDMAProfiling, log));
    pm.addPass(mlir::createCanonicalizerPass());
    arch40xx::elfSubsetPipelineVPUMI(pm, backendCompilationOptions.workloadManagementMode,
                                     backendCompilationOptions.enableDumpStatisticsOfWlmOps,
                                     backendCompilationOptions.workloadManagementBarrierProgrammingMode, log);

    pm.addPass(VPUMI40XX::createAddMappedInferenceVersionOpPass(log, npu40xx::NNRT_API_UD2025_38_MAJOR_VERSION,
                                                                npu40xx::NNRT_API_UD2025_38_MINOR_VERSION,
                                                                npu40xx::NNRT_API_UD2025_38_PATCH_VERSION));

    arch40xx::elfSubsetPipelineVPUASM(pm, backendCompilationOptions.workloadManagementDmaFifoType == DMAFifoType::HW,
                                      log);

    pm.addPass(VPUIPDPU::createExpandDPUConfigPass(log, backendCompilationOptions.npu5PPEBackwardsCompatibilityMode));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(
            log, std::string(backendCompilationOptions.enableShaveDDRAccessOptimization) == "true"));
    pm.addPass(createConvertVPUASM2NPUReg50XXPass(log, backendCompilationOptions.modelIdentifier));
    pm.addPass(createConvertVPUIPDPU2NPUReg50XXPass(log, backendCompilationOptions.npu5PPEBackwardsCompatibilityMode));

    pm.addPass(VPURegMapped::createDeduceDynamicMappedInferenceVersionPass(log));
    pm.addPass(ELF::createAddCompatibilityStringPass(log));

    pm.addPass(ELF::createHandleAlignmentRequirementsPass(log));
    pm.addPass(ELF::createSetOpOffsetsPass(log));
    pm.addPass(ELF::createSetCMXSymbolValuePass(log));
    pm.addPass(ELF::createAddRelocationsForDynamicStridesDMAsPass(log));
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
