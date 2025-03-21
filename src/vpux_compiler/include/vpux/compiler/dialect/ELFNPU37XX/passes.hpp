//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace ELFNPU37XX {

//
// Passes
//

std::unique_ptr<mlir::Pass> createRemoveEmptyELFSectionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpdateELFSectionFlagsPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace ELFNPU37XX
}  // namespace vpux
