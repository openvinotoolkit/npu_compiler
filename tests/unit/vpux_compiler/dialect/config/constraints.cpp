//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_ConstraintsTest : public MLIR_UnitBase {  // NOLINT case style
public:
    MLIR_ConstraintsTest(): MLIR_UnitBase() {
        registry.insert<vpux::config::ConfigDialect>();
        context.appendDialectRegistry(registry);
        context.loadDialect<vpux::config::ConfigDialect>();
        vpux::config::registerConstraints(registry, config::Platform::NPU5010);
        registry.applyExtensions(&context);
    }

    MLIR_ConstraintsTest(const MLIR_ConstraintsTest&) = delete;
    MLIR_ConstraintsTest& operator=(const MLIR_ConstraintsTest&) = delete;
    MLIR_ConstraintsTest(MLIR_ConstraintsTest&&) = delete;
    MLIR_ConstraintsTest& operator=(MLIR_ConstraintsTest&&) = delete;

    mlir::MLIRContext context;
};

TEST_F(MLIR_ConstraintsTest, MaxKernelSize) {
    auto npuConstraints = vpux::config::getNPUConstraints(&context);
    // check default value for max kernel size
    EXPECT_EQ(npuConstraints.maxKernelSize, 15);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.CustomConstraints @Constraints {
                config.Constraint @constraint.MaxKernelSize : 17 : si64
            }
        }
    )";

    auto moduleRef = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &context);
    ASSERT_TRUE(moduleRef) << "Failed to parse test module";
    auto module = moduleRef.get();

    // Validate that the option is correctly parsed and stored in the module.
    auto customConstraintsOp = module.lookupSymbol<config::CustomConstraintsOp>("Constraints");
    auto maxKernelSizeOption = customConstraintsOp.lookupSymbol<config::ConstraintOp>("constraint.MaxKernelSize");
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(maxKernelSizeOption.getConstraintValue());
    EXPECT_EQ(intAttr.getSInt(), 17);

    // Validate the pass that relies.
    mlir::PassManager pm(moduleRef.get()->getName());
    pm.addPass(VPU::createSetupCustomConstraintsPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(moduleRef.get())));

    // Validate that the pass correctly updates the NPU constraints in the context based on the module options.
    auto npuConstraintsUpdated = vpux::config::getNPUConstraints(&context);
    EXPECT_EQ(npuConstraintsUpdated.maxKernelSize, 17);
}
