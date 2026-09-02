//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUASM {

//
// Passes
//

std::unique_ptr<mlir::Pass> createHoistInputOutputsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddProfilingSectionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddPerformanceMetricsSectionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddCompilerHashPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddHostParsedInferencePass(Logger log = Logger::global());
//
// Registration
//

void registerPasses();

}  // namespace VPUASM
}  // namespace vpux
