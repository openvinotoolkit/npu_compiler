//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/optional.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// Common utilities
//

namespace {

//
// Common utilities
//

template <VPU::MemoryKind KIND>
mlir::Attribute getMemSpace(mlir::MLIRContext* ctx, StringRef) {
    return VPU::MemoryKindAttr::get(ctx, KIND);
}

VPU::ArchKind getArchKind(const StrOption& archKind) {
    VPUX_THROW_UNLESS(archKind.hasValue(), "Platform architecture is not provided. Please try 'vpu-arch=KMB'");
    const auto archKindStr = archKind.getValue();
    const auto parsed = VPU::symbolizeArchKind(archKindStr);
    VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported platform architecture '{0}'", archKindStr);
    return parsed.getValue();
}

Optional<int> getNumOfDPUGroups(const IntOption& numOfDPUGroups) {
    if (numOfDPUGroups.hasValue()) {
        return numOfDPUGroups.getValue();
    }
    return None;
}

}  // namespace

//
// ReferenceSWMode
//

void vpux::buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    IE::buildAdjustForVPUPipeline(pm, log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    // Lowering

    buildLowerIE2IERTPipeline(pm, log);

    pm.addPass(createConvertSWLayers2VPUIPPass(log));

    // Level 2 : Abstract RunTime

    if (options.enableProfiling) {
        pm.addPass(IERT::createTimestampProfilingPass(getMemSpace<VPU::MemoryKind::DDR>, log));
    }

    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPU::MemoryKind::DDR>, log));

    IERT::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPU::MemoryKind::DDR>, log));
    pm.addPass(IERT::createOptimizeAsyncDepsPass(log));

    pm.addPass(IERT::createBreakDataFlowPass(log));

    // Lowering

    buildLowerIERT2VPUIPPipeline(pm, log);

    // Level 1 : VPU RunTime

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// ReferenceHWMode
//

void vpux::buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }
    if (options.enableConvertScaleShiftDW) {
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
    }
    if (options.enableSplitConvWithMultipleFQ) {
        pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustForVPUPipeline(pm, log);

    if (options.enableHandleLargeStrides) {
        pm.addPass(IE::createHandleLargeStridesPass(log));
    }
    if (options.enableHandleAsymmetricStrides) {
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    }

    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, log);
    }
    pm.addPass(IE::createResolvePWLPostOpsPass(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    pm.addPass(IE::createIsolatedTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering

    buildLowerIE2IERTPipeline(pm, log);

    pm.addPass(createConvertToNCEOpsPass(log));
    pm.addPass(createConvertSWLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 2 : Abstract RunTime

    if (options.enableProfiling) {
        pm.addPass(IERT::createTimestampProfilingPass(getMemSpace<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPU::MemoryKind::DDR>, log));

    IERT::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPU::MemoryKind::DDR>, log));
    pm.addPass(IERT::createOptimizeAsyncDepsPass(log));

    pm.addPass(IERT::createBreakDataFlowPass(log));

    // Lowering

    pm.addPass(VPUIP::createConvertWeightsTableOp2ConstPass(log));
    buildLowerIERT2VPUIPPipeline(pm, log);

    // Level 1 : VPU RunTime

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createUnrollBatchPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }
    if (options.enableConvertScaleShiftDW) {
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
    }
    if (options.enableSplitConvWithMultipleFQ) {
        pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustForVPUPipeline(pm, log);

    if (options.enableHandleLargeStrides) {
        pm.addPass(IE::createHandleLargeStridesPass(log));
    }
    if (options.enableHandleAsymmetricStrides) {
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    }

    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, log);
    }
    pm.addPass(IE::createResolvePWLPostOpsPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    pm.addPass(IE::createIsolatedTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering

    buildLowerIE2IERTPipeline(pm, log);

    pm.addPass(createConvertToNCEOpsPass(log));
    pm.addPass(createConvertSWLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 2 : Abstract RunTime

    if (options.enableProfiling) {
        pm.addPass(IERT::createTimestampProfilingPass(getMemSpace<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPU::MemoryKind::DDR>, log));

    if (options.enableOptimizeCopies) {
        pm.addPass(IERT::createOptimizeCopiesPass(log));
        pm.addPass(IERT::createCopyOpHoistingPass(log));
    }

    IERT::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(IERT::createFeasibleAllocationPass(getMemSpace<VPU::MemoryKind::CMX_NN>,
                                                  getMemSpace<VPU::MemoryKind::DDR>, log));

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(IERT::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPU::MemoryKind::DDR>, log));
    pm.addPass(IERT::createOptimizeAsyncDepsPass(log));

    pm.addPass(IERT::createBreakDataFlowPass(log));

    // Lowering

    pm.addPass(VPUIP::createConvertWeightsTableOp2ConstPass(log));
    buildLowerIERT2VPUIPPipeline(pm, log);

    // Level 1 : VPU RunTime

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// registerPipelines
//

void vpux::registerPipelines() {
    mlir::PassPipelineRegistration<ReferenceSWOptions>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution)",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::ReferenceSW));

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceHWOptions>(
            "reference-hw-mode", "Compile IE Network in Reference Hardware mode (HW and SW execution)",
            [](mlir::OpPassManager& pm, const ReferenceHWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                const auto numOfDPUGroups = getNumOfDPUGroups(options.numberOfDPUGroups);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::ReferenceHW, numOfDPUGroups));

                buildReferenceHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution)",
            [](mlir::OpPassManager& pm, const DefaultHWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                const auto numOfDPUGroups = getNumOfDPUGroups(options.numberOfDPUGroups);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::DefaultHW, numOfDPUGroups));

                buildDefaultHWModePipeline(pm, options);
            });
}
