//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux {
namespace arch37xx {

//
// DefaultHWOptionsDeviceBase (for all dialects in 37xx)
// This class must be inherited by all dialect-base options
// to avoid confusion when we have the same option for IE and the VPU dialect, but with a different value
//

struct DefaultHWOptionsDeviceBase : public virtual vpux::DefaultHWOptionsBase, public vpux::BatchCompileOptionsAdapter {
    DefaultHWOptionsDeviceBase(): vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
    }

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableExplicitDistributionInfoAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributionInfoAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(false)};

    BoolOption enableConvertToPalletizationLUT{*this, "enable-convert-to-palletization-lut",
                                               llvm::cl::desc("Enable conversion of certain types to palletized LUT"),
                                               llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption enableReorderConcatBranches{
            *this, "enable-reorder-concat-branches",
            llvm::cl::desc("Reorder branches of concat to make sure it is executed branch by branch"),
            llvm::cl::init(false)};

    // VPUIP option shared with VPU pass
    BoolOption enableWeightsSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                      ::llvm::cl::init(true)};
};

//
// MCAndTilingOptionsDevice options
//

struct MCAndTilingOptionsDevice : public vpux::MCAndTilingOptionsBase {};

}  // namespace arch37xx
}  // namespace vpux
