//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/DialectRegistry.h>
#include <memory>

namespace vpux {

//
// IInterfaceRegister
//

class IInterfaceRegistry {
public:
    virtual void registerInterfaces(mlir::DialectRegistry& registry) = 0;
    virtual ~IInterfaceRegistry() = default;
};

//
// createInterface
//

std::unique_ptr<IInterfaceRegistry> createInterfacesRegistry(config::Platform platform);

}  // namespace vpux
