//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/core/pipelines_options.hpp"
#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux::VPU::arch50xx {

//
// DefaultHWOptions
//

struct DefaultHWOptions : public VPU::DefaultHWOptionsDialectBase, virtual vpux::arch50xx::DefaultHWOptionsDeviceBase {
    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("S0")};

    BoolOption enableVPUNNCostForTiling{*this, "enable-vpunn-cost-for-tiling",
                                        llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                                        llvm::cl::init(true)};

    BoolOption enableVPUNNPreSplit{*this, "enable-vpunn-pre-split", llvm::cl::desc("Enable VPUNN LayersPreSplit API"),
                                   llvm::cl::init(true)};

    BoolOption enableDequantWeightEnsuranceBeforeStrategy{
            *this, "enable-dequant-weight-ensurance-before-strategy",
            llvm::cl::desc("Enable dequantize weight op size ensurance before strategy is assigned in "
                           "EnsureNCEOpsSizeRequirements pass"),
            llvm::cl::init(true)};
};

//
// vpux::VPU::arch50XX::MCAndTilingOptionsDevice
//

struct MCAndTilingOptionsDevice : public vpux::MCAndTilingOptionsBase {
    template <class OtherOptions>
    explicit MCAndTilingOptionsDevice(const OtherOptions& options): vpux::MCAndTilingOptionsBase(options) {
        enableVPUNNCostForTiling = false;
        enableOutputPipelining = false;
    }
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

void buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                              Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createAutopadChannelsPass(const Logger& log = Logger::global());

//
// Registration
//

void registerVPUPipelines();
void registerPasses();

}  // namespace vpux::VPU::arch50xx
