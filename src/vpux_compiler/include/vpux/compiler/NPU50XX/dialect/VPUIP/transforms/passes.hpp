//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/pipelines_options.hpp"

namespace vpux {
namespace VPUIP {
namespace arch50xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createInsertDelayDPUVariantPass(bool dpuProfilingEnabled = false,
                                                            Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions :
        public VPUIP::DefaultHWOptionsDialectBase,
        virtual vpux::arch50xx::DefaultHWOptionsDeviceBase {
    BoolOption enableCompressWeightsBTC{*this, "compress-weights-btc", ::llvm::cl::desc("Enable compress-weights pass"),
                                        ::llvm::cl::init(false)};

    BoolOption enableActivationSwizzling{*this, "enable-activation-swizzling",
                                         ::llvm::cl::desc("Enable activation swizzling"), ::llvm::cl::init(false)};

    // Should only be enabled when accurate VPUNN cost is supported
    BoolOption enableMultiScheduleHeuristic{
            *this, "enable-multi-schedule-heuristic",
            ::llvm::cl::desc("Enables compiler to schedule with different heuristic logics and compare costs"),
            ::llvm::cl::init(true)};

    // TODO: E#118871 Switch this option to true by default
    BoolOption enableBarrierSchedWithFunctionOutlining{
            *this, "barrier-sched-with-function-outlining",
            llvm::cl::desc("Enable barrier scheduling passes with IR split into multiple functions"),
            llvm::cl::init(false)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());
void buildReferenceSWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// Registration
//

void registerVPUIPPipelines();
void registerPasses();

}  // namespace arch50xx
}  // namespace VPUIP
}  // namespace vpux
