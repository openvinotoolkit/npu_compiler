//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/core/pipelines_options.hpp"

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/pipelines_options.hpp"

namespace vpux {
namespace VPUIP {
namespace arch37xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAddSwKernelCacheHandlingOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDistributedOpsPass(Logger log = Logger::global());

//
// Optimize copies pipeline
//

struct OptimizeCopiesOptions final : public VPUIP::OptimizeCopiesOptionsBase {
    OptimizeCopiesOptions() = default;

    template <class OtherOptions>
    explicit OptimizeCopiesOptions(const OtherOptions& options): OptimizeCopiesOptionsBase(options) {
    }
};

void buildOptimizeCopiesPipeline(mlir::OpPassManager& pm, const OptimizeCopiesOptions& options,
                                 Logger log = Logger::global());

//
// Memory allocation pipeline
//

struct MemoryAllocationOptions final : public VPUIP::MemoryAllocationOptionsBase {
    MemoryAllocationOptions() = default;

    template <class OtherOptions>
    explicit MemoryAllocationOptions(const OtherOptions& options): MemoryAllocationOptionsBase(options) {
    }
};

void buildMemoryAllocationPipeline(mlir::OpPassManager& pm, const MemoryAllocationOptions& options,
                                   Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions :
        public VPUIP::DefaultHWOptionsDialectBase,
        virtual vpux::arch37xx::DefaultHWOptionsDeviceBase {
    BoolOption enableCompressWeightsBTC{*this, "compress-weights-btc", ::llvm::cl::desc("Enable compress-weights pass"),
                                        ::llvm::cl::init(false)};

    BoolOption enableActivationSwizzling{*this, "enable-activation-swizzling",
                                         ::llvm::cl::desc("Enable activation swizzling"), ::llvm::cl::init(true)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());
void buildReferenceSWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// Registration
//

void registerVPUIPPipelines();
void registerPasses();

}  // namespace arch37xx
}  // namespace VPUIP
}  // namespace vpux
