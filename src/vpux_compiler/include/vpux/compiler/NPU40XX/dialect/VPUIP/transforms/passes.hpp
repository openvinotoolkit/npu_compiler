//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/core/pipelines_options.hpp"

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/pipelines_options.hpp"

namespace vpux {
namespace VPUIP {
namespace arch40xx {

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

    // Should only be enabled when accurate VPUNN cost is supported
    BoolOption enableMultiScheduleHeuristic{
            *this, "enable-multi-schedule-heuristic",
            ::llvm::cl::desc("Enables compiler to schedule with different heuristic logics and compare costs"),
            ::llvm::cl::init(false)};
    BoolOption enableLoopAllocation{*this, "enable-loop-allocation",
                                    ::llvm::cl::desc("Enables loop allocation for tiling and vertical fusion regions"),
                                    ::llvm::cl::init(false)};

    // TODO: E#216928 - remove flag when VF region scheduler feature fully enabled.
    BoolOption enableVfUndefinedScheduler{
            *this, "enable-vf-undefined-scheduler",
            ::llvm::cl::desc("Route VF regions through UndefinedVF (currently returns an empty schedule and falls back "
                             "to the legacy VF allocation path)"),
            ::llvm::cl::init(false)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());
void buildReferenceSWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// Registration
//

void registerVPUIPPipelines();

}  // namespace arch40xx
}  // namespace VPUIP
}  // namespace vpux
