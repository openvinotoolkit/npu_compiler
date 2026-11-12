//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU60XX/pipeline_options.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace arch60xx {

//
// Pipelines
//

void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                 const BackendCompilationOptions60XX& backendCompilationOptions,
                                 Logger log = Logger::global());

//
// Registration
//

void registerConversionPipeline();

}  // namespace arch60xx
}  // namespace vpux
