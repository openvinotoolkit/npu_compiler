//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"

#include "vpux/compiler/NPU37XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/strategies_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/strategies_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/strategies_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIP/strategies_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/config/constraints_initializer.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <memory>

using namespace vpux;

namespace {

std::unique_ptr<config::IConstraintsInitializer> createConstraintsInitializer(config::Platform platform) {
    const auto arch = config::getArch(platform);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<config::ConstraintsInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<config::ConstraintsInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<config::ConstraintsInitializer50XX>(platform);
    default:
        VPUX_THROW("Unsupported arch: {0}", arch);
    }
}

}  // namespace

namespace vpux {

IStrategiesInitializer::~IStrategiesInitializer() = default;

namespace config {

// Extension class to register constraints for a specific architecture
class ConstraintsExtension : public mlir::DialectExtension<ConstraintsExtension, ConfigDialect> {
public:
    explicit ConstraintsExtension(config::Platform target): _platform(target) {
    }

    void apply(mlir::MLIRContext* context, ConfigDialect* /*dialect*/) const override {
        auto constraintsInitializer = createConstraintsInitializer(_platform);
        constraintsInitializer->initialize(context);
    }

private:
    config::Platform _platform;
};

void registerConstraints(mlir::DialectRegistry& registry, config::Platform platform) {
    registry.addExtension(mlir::TypeID::get<ConstraintsExtension>(), std::make_unique<ConstraintsExtension>(platform));
}

}  // namespace config

namespace IE {

namespace {

std::unique_ptr<IStrategiesInitializer> createStrategiesInitializer(config::Platform platform) {
    const auto arch = config::getArch(platform);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<IE::StrategiesInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<IE::StrategiesInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<IE::StrategiesInitializer50XX>();
    default:
        VPUX_THROW("Unsupported arch: {0}", arch);
    }
}

}  // namespace

class StrategiesExtension : public mlir::DialectExtension<StrategiesExtension, IEDialect> {
public:
    explicit StrategiesExtension(config::Platform platform): _platform(platform) {
    }

    void apply(mlir::MLIRContext* context, IEDialect*) const override {
        auto strategiesInitializer = createStrategiesInitializer(_platform);
        strategiesInitializer->initialize(context);
    }

private:
    config::Platform _platform;
};

void registerStrategies(mlir::DialectRegistry& registry, config::Platform platform) {
    registry.addExtension(mlir::TypeID::get<StrategiesExtension>(), std::make_unique<StrategiesExtension>(platform));
}

}  // namespace IE

namespace VPU {

namespace {

std::unique_ptr<IStrategiesInitializer> createStrategiesInitializer(config::Platform platform) {
    const auto arch = config::getArch(platform);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<VPU::StrategiesInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<VPU::StrategiesInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<VPU::StrategiesInitializer50XX>();
    default:
        VPUX_THROW("Unsupported arch: {0}", arch);
    }
}

}  // namespace

class StrategiesExtension : public mlir::DialectExtension<StrategiesExtension, VPUDialect> {
public:
    explicit StrategiesExtension(config::Platform platform): _platform(platform) {
    }

    void apply(mlir::MLIRContext* context, VPUDialect*) const override {
        auto strategiesInitializer = VPU::createStrategiesInitializer(_platform);
        strategiesInitializer->initialize(context);
    }

private:
    config::Platform _platform;
};

void registerStrategies(mlir::DialectRegistry& registry, config::Platform platform) {
    registry.addExtension(mlir::TypeID::get<VPU::StrategiesExtension>(),
                          std::make_unique<VPU::StrategiesExtension>(platform));
}

}  // namespace VPU

namespace VPUIP {

namespace {

std::unique_ptr<IStrategiesInitializer> createStrategiesInitializer(config::Platform platform) {
    const auto arch = config::getArch(platform);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<VPUIP::StrategiesInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<VPUIP::StrategiesInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<VPUIP::StrategiesInitializer50XX>();
    default:
        VPUX_THROW("Unsupported arch: {0}", arch);
    }
}

}  // namespace

class StrategiesExtension : public mlir::DialectExtension<StrategiesExtension, VPUIPDialect> {
public:
    explicit StrategiesExtension(config::Platform platform): _platform(platform) {
    }

    void apply(mlir::MLIRContext* context, VPUIPDialect*) const override {
        auto strategiesInitializer = VPUIP::createStrategiesInitializer(_platform);
        strategiesInitializer->initialize(context);
    }

private:
    config::Platform _platform;
};

void registerStrategies(mlir::DialectRegistry& registry, config::Platform platform) {
    registry.addExtension(mlir::TypeID::get<StrategiesExtension>(), std::make_unique<StrategiesExtension>(platform));
}

}  // namespace VPUIP
}  // namespace vpux
