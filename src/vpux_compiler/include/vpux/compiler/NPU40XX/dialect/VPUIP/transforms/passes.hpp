//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/core/pipelines_options.hpp"

#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/pipelines_options.hpp"

namespace vpux {
namespace VPUIP {
namespace arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createComputeTaskStrippingPass(
        Logger log = Logger::global(), VPU::DPUDryRunMode dryRunStripTarget = VPU::DPUDryRunMode::NONE,
        bool shaveDryRun = false);

std::unique_ptr<mlir::Pass> createComputeHaloRegionForDPUTaskOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingHwDdrPass(const std::string& enableDMAProfiling = "true",
                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConstantDpuProfHwpBasePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressSpillDmaPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMAOutOfOrderOptimizationPass(
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createUnrollDistributedOpsPass(Logger log = Logger::global(),
                                                           bool enableSegmentedDmaFusion = false);
std::unique_ptr<mlir::Pass> createOptimizeConvertDMAOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddStartBarrierPass(
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectDMASplitCandidatePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitDMAToBalanceLoadPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSegmentedDmaPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeScheduleForPartialWlmFetchDmasPass(
        const int virtualBarrierThreshold = VIRTUAL_BARRIER_THRESHOLD_WLM, Logger log = Logger::global());
//
// Memory allocation pipeline
//

struct MemoryAllocationOptions final : public VPUIP::MemoryAllocationOptionsBase {
    BoolOption enableCompressActivationSpill{*this, "compress-activation-spill",
                                             ::llvm::cl::desc("Enable compress-activation-spill feature"),
                                             ::llvm::cl::init(false)};

    MemoryAllocationOptions() = default;

    template <class OtherOptions>
    MemoryAllocationOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

void buildMemoryAllocationPipeline(mlir::OpPassManager& pm, const MemoryAllocationOptions& options,
                                   Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions :
        public VPUIP::DefaultHWOptionsDialectBase,
        virtual vpux::arch40xx::DefaultHWOptionsDeviceBase {
    BoolOption enableCompressWeightsBTC{*this, "compress-weights-btc", ::llvm::cl::desc("Enable compress-weights pass"),
                                        ::llvm::cl::init(false)};

    BoolOption enableActivationSwizzling{*this, "enable-activation-swizzling",
                                         ::llvm::cl::desc("Enable activation swizzling"), ::llvm::cl::init(true)};

    // TODO: E#118871 Switch this option to true by default
    BoolOption enableBarrierSchedWithFunctionOutlining{
            *this, "barrier-sched-with-function-outlining",
            llvm::cl::desc("Enable barrier scheduling passes with IR split into multiple functions"),
            llvm::cl::init(false)};

    BoolOption configureUpdateBarriersForSwPrefetch{
            *this, "configure-update-barriers-for-sw-prefetch",
            llvm::cl::desc("Configure update barrier to block shave execution until prefetch finishes"),
            llvm::cl::init(true)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());
void buildReferenceSWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// Registration
//

void registerVPUIPPipelines();
void registerPasses();

}  // namespace arch40xx
}  // namespace VPUIP
}  // namespace vpux
