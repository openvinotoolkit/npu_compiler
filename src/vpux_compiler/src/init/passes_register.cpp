//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/passes_register.hpp"
#include "vpux/compiler/NPU50XX/passes_register.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

// Note: Keep this implementation empty for architectures which do not have architecture specific passes
void EmptyPassesRegistry::registerPasses() {
}

//
// createPassesRegistry
//

std::unique_ptr<IPassesRegistry> vpux::createPassesRegistry(config::Platform platform) {
    const auto arch = config::getArch(platform);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<EmptyPassesRegistry>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<PassesRegistry40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<PassesRegistry50XX>();
    default:
        VPUX_THROW("Unsupported platform: {0}", platform);
    }
}
