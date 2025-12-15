//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/passes_register.hpp"
#include "vpux/compiler/NPU50XX/passes_register.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

// Note: Keep this implementation empty for architectures which do not have architecture specific passes
void EmptyPassesRegistry::registerPasses() {
}

//
// createPassesRegistry
//

std::unique_ptr<IPassesRegistry> vpux::createPassesRegistry(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<EmptyPassesRegistry>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<PassesRegistry40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<PassesRegistry50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}
