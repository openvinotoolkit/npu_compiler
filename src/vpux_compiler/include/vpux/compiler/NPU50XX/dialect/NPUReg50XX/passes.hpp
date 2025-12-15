//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace NPUReg50XX {

//
// Passes
//

std::unique_ptr<mlir::Pass> createSetupIduCmxMuxModePass(Logger log = Logger::global(), uint8_t iduCmxMuxMode = 0);

//
// Registration
//

void registerPasses();

}  // namespace NPUReg50XX
}  // namespace vpux
