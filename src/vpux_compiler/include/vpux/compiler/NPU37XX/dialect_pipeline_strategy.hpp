//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/pipeline_options.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/dialect_pipeline_strategy.hpp"
#include "vpux/utils/ov/config.hpp"

namespace vpux {

//
// This version is used for production purposes
//

std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy37XX(config::CompilationMode compilationMode,
                                                                            const intel_npu::Config& config);

//
// This version is used for testing purposes
// The main difference is that it does not set any special option values
// No definition in the header to avoid extra dependencies defined here
// Template definition is provided in source file
//

template <class OptionsType>
extern std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy37XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options);

template <class OptionsType>
extern std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy37XXReferenceSW(
        const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options);

template <class OptionsType>
std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<OptionsType>> createOptionsDefaultHW(
        const intel_npu::Config& config);

}  // namespace vpux
