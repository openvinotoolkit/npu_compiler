//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/pipelines/dialect_pipeline_strategy.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "vpux/utils/IE/config.hpp"

namespace vpux {

//
// This version is used for production purposes
//

std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy40XX(config::CompilationMode compilationMode,
                                                                            const intel_npu::Config& config);

//
// This version is used for testing purposes
// The main difference is that it does not set any special option values
// No definition in the header to avoid extra dependencies defined here
// Template definition is provided in source file
//

template <class OptionsType>
std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options);

/// @brief This method creates a pipeline strategy for ReferenceSW compilation.
template <class OptionsType>
std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy40XXReferenceSW(
        const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options);

/// @brief This method creates a pipeline strategy for Monolithic WS compilation.
template <class OptionsType>
std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy40XXWS(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const OptionsType* options);

/// @brief This method creates a pipeline strategy for HostCompile compilation.
template <class OptionsType>
std::unique_ptr<IDialectPipelineStrategy> createDialectPipelineStrategy40XXHostCompile(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const OptionsType* options);

}  // namespace vpux
