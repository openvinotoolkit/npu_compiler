//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Pass/PassManager.h>

namespace vpux {
namespace VPUIP {

struct OptimizeCopiesOptionsBase : mlir::PassPipelineOptions<OptimizeCopiesOptionsBase> {
    OptimizeCopiesOptionsBase() = default;

    BoolOption enableOptimizeConstCopies{*this, "optimize-const-copies", llvm::cl::desc("Enable optimize-const-copies"),
                                         llvm::cl::init(true)};
    BoolOption enableOpsAsDMA{*this, "enable-ops-as-dma",
                              llvm::cl::desc("Force using DMA transformations instead of SW ops"),
                              llvm::cl::init(true)};

    template <class OtherOptions>
    explicit OptimizeCopiesOptionsBase(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

struct MemoryAllocationOptionsBase : mlir::PassPipelineOptions<MemoryAllocationOptionsBase> {
    BoolOption linearizeSchedule{*this, "linearize-schedule", llvm::cl::desc("Linearize tasks on all engines"),
                                 llvm::cl::init(false)};

    BoolOption enableLoopAllocation{*this, "enable-loop-allocation",
                                    ::llvm::cl::desc("Enables loop allocation for tiling and vertical fusion regions"),
                                    ::llvm::cl::init(false)};

    BoolOption enablePrefetching{*this, "prefetching",
                                 llvm::cl::desc("Enable prefetch tiling pass and prefetch scheduling"),
                                 llvm::cl::init(true)};

    BoolOption enablePipelining{*this, "pipelining",
                                llvm::cl::desc("Enable vertical fusion pipelining pass and schedule pipelining"),
                                llvm::cl::init(true)};

    BoolOption optimizeFragmentation{*this, "optimize-fragmentation",
                                     ::llvm::cl::desc("Enables compiler to optimize CMX fragmentation"),
                                     ::llvm::cl::init(true)};

    BoolOption optimizeDynamicSpilling{*this, "optimize-dynamic-spilling",
                                       ::llvm::cl::desc("Enables compiler to optimize dynamic spilling DMAs"),
                                       ::llvm::cl::init(true)};

    BoolOption enableMultiScheduleHeuristic{
            *this, "enable-multi-schedule-heuristic",
            ::llvm::cl::desc("Enables compiler to schedule with different heuristic logics and compare costs"),
            ::llvm::cl::init(false)};

    BoolOption enableVfUndefinedScheduler{
            *this, "enable-vf-undefined-scheduler",
            ::llvm::cl::desc("Route VF regions through UndefinedVF (currently returns an empty schedule and falls back "
                             "to the legacy VF allocation path)"),
            ::llvm::cl::init(false)};

    BoolOption enablePrintStatistics{*this, "enable-print-statistics", ::llvm::cl::desc("Enable print statistics"),
                                     ::llvm::cl::init(vpux::isDeveloperBuild())};

    MemoryAllocationOptionsBase() = default;

    template <class OtherOptions>
    explicit MemoryAllocationOptionsBase(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

}  // namespace VPUIP
}  // namespace vpux
