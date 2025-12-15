//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"

#include "vpux/compiler/NPU37XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/config/constraints_initializer.hpp"
#include "vpux/utils/core/error.hpp"

#include <vpux/compiler/dialect/config/enums.hpp.inc>

#include <functional>
#include <memory>

using namespace vpux;

namespace {

std::unique_ptr<config::IConstraintsInitializer> createConstraintsInitializer(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<config::ConstraintsInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<config::ConstraintsInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<config::ConstraintsInitializer50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

}  // namespace

namespace vpux {
namespace config {

// Extension class to register constraints for a specific architecture
class ConstraintsExtension : public mlir::DialectExtension<ConstraintsExtension, ConfigDialect> {
public:
    explicit ConstraintsExtension(ArchKind arch): arch(arch) {
    }

    void apply(mlir::MLIRContext* context, ConfigDialect* /*dialect*/) const override {
        auto constraintsInitializer = createConstraintsInitializer(arch);
        constraintsInitializer->initialize(context);
    }

private:
    ArchKind arch;
};

void registerConstraints(mlir::DialectRegistry& registry, ArchKind arch) {
    registry.addExtension(mlir::TypeID::get<ConstraintsExtension>(), std::make_unique<ConstraintsExtension>(arch));
}

}  // namespace config
}  // namespace vpux
