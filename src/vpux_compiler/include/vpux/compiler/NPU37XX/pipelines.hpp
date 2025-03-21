//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// ReferenceSWOptions37XX
//

struct ReferenceSWOptions37XX final : public ReferenceSWOptions<ReferenceSWOptions37XX> {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(false)};
};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions37XX& options,
                                  Logger log = Logger::global());

//
// DefaultHWOptions37XX
//

struct DefaultHWOptions37XX final :
        public IE::arch37xx::DefaultHWOptions,
        VPU::arch37xx::DefaultHWOptions,
        VPUIP::arch37xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions37XX> {
    // Due to multiple inheritance, 'DefaultHWOptions37XX' has multiple definitions of 'createFromString' method
    // here we assume that we are interested in a "final" method that includes parameters from all parent classes
    using mlir::PassPipelineOptions<DefaultHWOptions37XX>::createFromString;
};

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options,
                                Logger log = Logger::global());

//
// ShaveCodeGenPipeline
//

struct ShaveCodeGenOptions37XX final : public ShaveCodeGenOptionsBase<ShaveCodeGenOptions37XX> {};

void buildShaveCodeGenPipeline(mlir::OpPassManager& pm, const ShaveCodeGenOptions37XX& options,
                               Logger log = Logger::global());

}  // namespace vpux
