//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// IPassesRegistry
//

class IPassesRegistry {
public:
    virtual void registerPasses() = 0;

    virtual ~IPassesRegistry() = default;
};

class EmptyPassesRegistry final : public IPassesRegistry {
public:
    void registerPasses() override;
};

//
// createPassesRegistry
//

std::unique_ptr<IPassesRegistry> createPassesRegistry(config::ArchKind archKind);

}  // namespace vpux
