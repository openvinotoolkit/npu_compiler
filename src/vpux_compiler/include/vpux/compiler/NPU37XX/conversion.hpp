//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace arch37xx {

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertIEToVPUNCEPass(Logger log = Logger::global());

//
// Pipelines
//

void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, bool enableInPlaceBufferization,
                                 Logger log = Logger::global());

//
// Registration
//

void registerConversionPipeline();
void registerConversionPasses();

}  // namespace arch37xx
}  // namespace vpux
