//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/core/pipelines_options.hpp"
#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {
namespace VPU {
namespace arch37xx {

//
// Passes
//

void buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                              Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions : public VPU::DefaultHWOptionsDialectBase, virtual vpux::arch37xx::DefaultHWOptionsDeviceBase {
    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("S0")};

    BoolOption enableVPUNNCostForTiling{*this, "enable-vpunn-cost-for-tiling",
                                        llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                                        llvm::cl::init(true)};
    BoolOption enableVPUNNPreSplit{*this, "enable-vpunn-pre-split", llvm::cl::desc("Enable VPUNN LayersPreSplit API"),
                                   llvm::cl::init(false)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());
void buildReferenceSWPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// Registration
//

void registerVPUPipelines();
void registerPasses();

}  // namespace arch37xx
}  // namespace VPU
}  // namespace vpux
