//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// ReferenceSWOptions40XX
//

struct ReferenceSWOptions40XX final : public ReferenceSWOptions<ReferenceSWOptions40XX> {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(false)};
};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions40XX& options,
                                  Logger log = Logger::global());

//
// DefaultHWOptions40XX
//

struct DefaultHWOptions40XX final :
        public IE::arch40xx::DefaultHWOptions,
        VPU::arch40xx::DefaultHWOptions,
        VPUIP::arch40xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions40XX> {
    // Due to multiple inheritance, 'DefaultHWOptions40XX' has multiple definitions of 'createFromString' method
    // here we assume that we are interested in a "final" method that includes parameters from all parent classes
    using mlir::PassPipelineOptions<DefaultHWOptions40XX>::createFromString;
};

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions40XX& options,
                                Logger log = Logger::global());

//
// BackendCompilationOptions40XX
//

struct BackendCompilationOptions40XX final : public BackendCompilationOptionsBase<BackendCompilationOptions40XX> {};

//
// ShaveCodeGenPipeline
//

struct ShaveCodeGenOptions40XX final : public ShaveCodeGenOptionsBase<ShaveCodeGenOptions40XX> {};

void buildShaveCodeGenPipeline(mlir::OpPassManager& pm, const ShaveCodeGenOptions40XX& options,
                               Logger log = Logger::global());

}  // namespace vpux
