//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/performance_metrics_utils.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <npu_40xx_nnrt.hpp>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr float CONFIGURED_ACTIVITY_FACTOR = 0.25F;
constexpr llvm::StringLiteral INPUT_IR = R"(
    module @test {
        net.NetworkInfo {inferenceTiming = 1 : i64} entryPoint : @main inputsInfo : {
            DataInfo "input" : tensor<1xf16>
        } outputsInfo : {
            DataInfo "output" : tensor<1xf16>
        }

        func.func @main(%input: tensor<1xf16>) -> tensor<1xf16> {
            return %input : tensor<1xf16>
        }
    }
)";

class PerformanceMetricsUtilsTest : public testing::TestWithParam<config::Platform> {};

TEST_P(PerformanceMetricsUtilsTest, PopulatesActivityFactorForPlatform) {
    const auto platform = GetParam();
    auto registry = createDialectRegistry();
    auto interfacesRegistry = createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, platform);
    VPU::registerStrategies(registry, platform);

    mlir::MLIRContext context(registry);
    mlir::OpBuilder builder(&context);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &context);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager passManager(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    VPU::buildInitCompilerPipeline(passManager, VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW),
                                   Logger::global());
    ASSERT_TRUE(mlir::succeeded(passManager.run(module.get())));

    auto tileResources = config::getTileExecutor(module.get());
    ASSERT_EQ(config::getKindValue<config::ExecutorKind>(tileResources), config::ExecutorKind::NCE);
    tileResources.setActivityFactorAttr(builder.getF64FloatAttr(CONFIGURED_ACTIVITY_FACTOR));

    npu40xx::VpuPerformanceMetrics metrics{};
    VPUASM::populatePerformanceMetrics(metrics, module.get());

    const auto expectedActivityFactor = CONFIGURED_ACTIVITY_FACTOR;
    EXPECT_FLOAT_EQ(metrics.activity_factor, expectedActivityFactor);
}

INSTANTIATE_TEST_SUITE_P(NPU50XX, PerformanceMetricsUtilsTest, testing::Values(config::Platform::NPU5010));

}  // namespace
