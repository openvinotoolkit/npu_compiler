//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/interfaces_registry.hpp"

#include "vpux/compiler/NPU37XX/interfaces_registry.hpp"
#include "vpux/compiler/NPU40XX/interfaces_registry.hpp"
#include "vpux/compiler/NPU50XX/interfaces_registry.hpp"

#include <memory>

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

//
// createInterfaceRegistry
//

std::unique_ptr<IInterfaceRegistry> createInterfacesRegistry(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<InterfacesRegistry37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<InterfacesRegistry40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<InterfacesRegistry50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

}  // namespace vpux
