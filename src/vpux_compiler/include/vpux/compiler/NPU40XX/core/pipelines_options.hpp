//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux {
namespace arch40xx {

//
// DefaultHWOptionsDeviceBase (for all dialects in 40xx)
// This class must be inherited by all dialect-base options
// to avoid confusion when we have the same option for IE and the VPU dialect, but with a different value
//

struct DefaultHWOptionsDeviceBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableConvertToPalletizationLUT{*this, "enable-convert-to-palletization-lut",
                                               llvm::cl::desc("Enable conversion of certain types to palletized LUT"),
                                               llvm::cl::init(true)};

    BoolOption enableExplicitDistributionInfoAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributionInfoAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(true)};

    BoolOption workloadManagementEnable{*this, "workload-management-enable",
                                        llvm::cl::desc("Enable partial workload management"), llvm::cl::init(true)};

    mlir::detail::PassOptions::Option<DMAFifoType> workloadManagementDmaFifoType{
            *this, "workload-management-dma-fifo-type",
            ::llvm::cl::desc("Option to switch behaviour between software and hardware DMA FIFO types"),
            ::llvm::cl::init(DMAFifoType::SW),
            ::llvm::cl::values(clEnumValN(DMAFifoType::SW, "SW", "Enable SW DMA FIFO upfront HW DMA FIFO type"),
                               clEnumValN(DMAFifoType::HW, "HW", "Use HW DMA FIFO directly"))};

    BoolOption enableSwKernelFifoPerShaveEngine{*this, "enable-sw-kernel-fifo-per-shave-engine",
                                                llvm::cl::desc("Enable dedicated FIFO for each ActShave engine"),
                                                llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(true)};

    BoolOption enableReorderConcatBranches{
            *this, "enable-reorder-concat-branches",
            llvm::cl::desc("Reorder branches of concat to make sure it is executed branch by branch"),
            llvm::cl::init(true)};

    BoolOption enableSegmentedDmaFusion{*this, "enable-segmented-dma-fusion",
                                        llvm::cl::desc("Enable fusion of segmented DMAs"), llvm::cl::init(false)};
    // VPUIP option shared with VPU pass
    BoolOption enableWeightsSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                      ::llvm::cl::init(true)};

    mlir::detail::PassOptions::Option<WorkloadManagementBarrierProgrammingMode>
            workloadManagementBarrierProgrammingMode{
                    *this, "workload-management-barrier-programming-mode",
                    ::llvm::cl::desc(
                            "Option for enabling different barrier programming algorithms. To be used only for "
                            "experiments."),
                    ::llvm::cl::values(
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::LEGACY, "LEGACY", "Legacy Mode"),
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED,
                                       "ALL_BARRIER_DMAS_SCHEDULED",
                                       "Compiler generates DMAs to program all barriers"))};

    IntOption modelIdentifier{
            *this, "model-identifier",
            llvm::cl::desc("Unique identifier for the compiled model, used for debugging and troubleshooting"),
            llvm::cl::init(0)};

    BoolOption enableRunMVNNormalizeOnDPU{*this, "enable-run-mvn-normalize-on-dpu",
                                          llvm::cl::desc("Enable RunMVNNormalizeOnDPU pass on DPU"),
                                          llvm::cl::init(false)};
};

//
// MCAndTilingOptionsDevice options
//

struct MCAndTilingOptionsDevice : public vpux::MCAndTilingOptionsBase {
    MCAndTilingOptionsDevice() {
        enableExplicitDistributionInfoAttr = true;
    }
};

}  // namespace arch40xx
}  // namespace vpux
