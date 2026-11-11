//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/interfaces_registry.hpp"

namespace vpux {

//
// IntefacesRegistry60XX
//

class InterfacesRegistry60XX final : public IInterfaceRegistry {
public:
    void registerInterfaces(mlir::DialectRegistry& registry) override;
};

}  // namespace vpux
