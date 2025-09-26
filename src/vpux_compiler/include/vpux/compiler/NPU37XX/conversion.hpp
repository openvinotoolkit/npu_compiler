//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {
namespace arch37xx {

//
// Pipelines
//

void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowerIE2VPUPipelineReferenceSW(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// Registration
//

void registerConversionPipeline();

}  // namespace arch37xx
}  // namespace vpux
