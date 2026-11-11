//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux {
namespace arch60xx {

//
// DefaultHWOptionsDeviceBase (for all dialects in 60xx)
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

    BoolOption enableSprLUT{*this, "enable-sprlut", llvm::cl::desc("Enable sprLUT"), llvm::cl::init(true)};

    BoolOption enableDCIM{*this, "enable-dcim", llvm::cl::desc("Enable DCIM"), llvm::cl::init(true)};

    BoolOption enableSoftmaxMaskAware{*this, "enable-softmax-mask-aware", ::llvm::cl::desc("Enable Softmax mask aware"),
                                      ::llvm::cl::init(false)};
    DoubleOption softmaxMaskAwareThreshold{
            *this, "softmax-mask-aware-threshold", ::llvm::cl::desc("Softmax mask aware threshold"),
            ::llvm::cl::init(static_cast<double>(std::numeric_limits<vpux::type::float16>::lowest()) +
                             SOFTMAX_MASK_DISABLED_NOISE_MARGIN)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(true)};

    BoolOption enableReorderConcatBranches{
            *this, "enable-reorder-concat-branches",
            llvm::cl::desc("Reorder branches of concat to make sure it is executed branch by branch"),
            llvm::cl::init(true)};

    // Temporary disable fusion for 50XX+ because of regressions, see E#148923
    BoolOption enableSegmentedDmaFusion{*this, "enable-segmented-dma-fusion",
                                        llvm::cl::desc("Enable fusion of segmented DMAs"), llvm::cl::init(false)};
    // VPUIP option shared with VPU pass
    BoolOption enableWeightsSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                      ::llvm::cl::init(false)};

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

    BoolOption enableAsymmetricPerTensorZP{
            *this, "enable-asymmetric-per-tensor-zp",
            llvm::cl::desc("Enable asymmetric weights with per tensor zp to run on NCE without dequantization"),
            llvm::cl::init(true)};

    BoolOption enableAsymmetricPerChannelZP{
            *this, "enable-asymmetric-per-channel-zp",
            llvm::cl::desc("Enable asymmetric weights with per channel zp to run on NCE without dequantization"),
            llvm::cl::init(true)};

    BoolOption enableRunMVNNormalizeOnDPU{*this, "enable-run-mvn-normalize-on-dpu",
                                          llvm::cl::desc("Enable RunMVNNormalizeOnDPU pass on DPU"),
                                          llvm::cl::init(true)};

    BoolOption enableSoftmaxDecomposition{*this, "enable-softmax-decomposition",
                                          llvm::cl::desc("Enable Softmax decomposition pass"), llvm::cl::init(true)};
};

}  // namespace arch60xx
}  // namespace vpux
