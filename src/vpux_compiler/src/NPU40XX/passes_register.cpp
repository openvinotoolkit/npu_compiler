//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/passes_register.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

using namespace vpux;

//
// PassesRegistry40XX::registerPasses
//

void PassesRegistry40XX::registerPasses() {
    vpux::VPURegMapped::registerPasses();
}
