//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/singleton_initializer.hpp"

namespace vpux::VPU::arch60xx {

/** @brief Creates and sets 60XX factories and utilities in the singleton cache for the given MLIR context. */
void initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform> platform);

}  // namespace vpux::VPU::arch60xx
