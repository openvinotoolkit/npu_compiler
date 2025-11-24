//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dynamic_rewriter/rewriters_register.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/rewriters.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"

namespace vpux {

RewriterRegistry& createRewriterRegistry() {
    auto& registry = RegistryManager::getGlobalRegistry();

    VPUIP::registerVPUIPRewriters(registry);

    return registry;
}

}  // namespace vpux
