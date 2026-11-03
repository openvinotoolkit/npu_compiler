//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>

#include <string>

namespace vpux {
namespace Shave {

//
// Passes
//

std::unique_ptr<mlir::Pass> createLoadExternalKernelResourcesPass(std::string customKernelConfigPath = "",
                                                                  Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyExternalKernelConfigsPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createMaterializeScratchBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleExternalKernelTilingRequirementsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExternalKernelMemSpaceAdjustmentPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace Shave
}  // namespace vpux
