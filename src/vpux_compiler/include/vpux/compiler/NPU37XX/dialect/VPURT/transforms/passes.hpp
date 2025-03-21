//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

namespace vpux {
namespace VPURT {
namespace arch37xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAddFinalBarrierPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace arch37xx
}  // namespace VPURT
}  // namespace vpux
