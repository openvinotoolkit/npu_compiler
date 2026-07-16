//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/interfaces_registry.hpp"

#include "vpux/compiler/NPU37XX/interfaces_registry.hpp"
#include "vpux/compiler/NPU40XX/interfaces_registry.hpp"
#include "vpux/compiler/NPU50XX/interfaces_registry.hpp"

#include <memory>

#include "vpux/utils/core/error.hpp"

namespace vpux {

//
// createInterfaceRegistry
//

std::unique_ptr<IInterfaceRegistry> createInterfacesRegistry(config::Platform platform) {
    switch (platform) {
    case config::Platform::NPU3720:
        return std::make_unique<InterfacesRegistry37XX>();
    case config::Platform::NPU4000:
        return std::make_unique<InterfacesRegistry40XX>();
    case config::Platform::NPU5010:
        return std::make_unique<InterfacesRegistry50XX>();
    case config::Platform::NPU5020:
        return std::make_unique<InterfacesRegistry50XX>();
    default:
        VPUX_THROW("Unsupported platform: {0}", platform);
    }
}

}  // namespace vpux
