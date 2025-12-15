//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/pipeline_options.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace arch50xx {

//
// Pipelines
//

void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                 const BackendCompilationOptions50XX& backendCompilationOptions,
                                 Logger log = Logger::global());

//
// Registration
//

void registerConversionPipeline();

}  // namespace arch50xx
}  // namespace vpux
