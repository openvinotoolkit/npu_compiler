//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/singleton_initializer.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <memory>

using namespace vpux;

namespace {

struct SingletonInitializers {
    std::optional<config::Platform> platform;
    FuncRef<void(mlir::MLIRContext*, std::optional<config::Platform>)> singletonCacheFn;
    FuncRef<void(mlir::MLIRContext*)> ppeVersionConfigFn;

    void initializeSingletonCache(mlir::MLIRContext* context) const {
        singletonCacheFn(context, platform);
    }

    void initializePPEVersionConfig(mlir::MLIRContext* context) const {
        ppeVersionConfigFn(context);
    }
};

SingletonInitializers getSingletonInitializer(config::Platform platform) {
    switch (platform) {
    case config::Platform::NPU3720:
        return {platform, VPU::arch37xx::initializeSingletonCache, VPU::arch37xx::initializePPEVersionConfig};
    case config::Platform::NPU4000:
        return {platform, VPU::arch40xx::initializeSingletonCache, VPU::arch37xx::initializePPEVersionConfig};
    case config::Platform::NPU5010:
    case config::Platform::NPU5020:
        return {platform, VPU::arch50xx::initializeSingletonCache, VPU::arch50xx::initializePPEVersionConfig};
    default:
        VPUX_THROW("Unsupported platform: {0}", platform);
    }
}

}  // namespace

namespace vpux::VPU {

// Extension class to register constraints for a specific architecture
class SingletonExtension : public mlir::DialectExtension<SingletonExtension, VPUDialect> {
public:
    explicit SingletonExtension(config::Platform platform): _platform(platform) {
    }

    void apply(mlir::MLIRContext* context, VPUDialect* /*dialect*/) const override {
        auto singletonInitializer = getSingletonInitializer(_platform);
        singletonInitializer.initializeSingletonCache(context);
        singletonInitializer.initializePPEVersionConfig(context);
    }

private:
    config::Platform _platform;
};

void initializeSingletons(mlir::DialectRegistry& registry, config::Platform platform) {
    registry.addExtension(mlir::TypeID::get<SingletonExtension>(), std::make_unique<SingletonExtension>(platform));
}

}  // namespace vpux::VPU
