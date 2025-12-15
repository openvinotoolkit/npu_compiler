//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <cstdint>
#include <memory>

namespace vpux {
namespace config {

//
// IConstraintsInitializer
//

class IConstraintsInitializer {
public:
    virtual void initialize(mlir::MLIRContext* context) = 0;
    virtual ~IConstraintsInitializer() = default;
};

//
// registerConstraints
//

enum class ArchKind : uint64_t;
void registerConstraints(mlir::DialectRegistry& registry, config::ArchKind arch);

}  // namespace config
}  // namespace vpux
