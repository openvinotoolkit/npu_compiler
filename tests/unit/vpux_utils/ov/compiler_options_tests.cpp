//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/public_options.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/options.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace vpux;

namespace {

// Reusable fixture to verify that a compiler option is correctly set into the configuration object
class CompilerOptionTest : public testing::Test {
public:
    std::shared_ptr<OV::OptionsDesc> options;
    OV::Config config;

    CompilerOptionTest(): options(std::make_shared<OV::OptionsDesc>()), config(options) {
    }
};

// Reusable fixture to verify that a compiler option is propagated into the IR by the
// init-compiler pipeline (which includes SetTargetIndependentOptions)
class CompilerOptionInIRTest : public testing::Test {
private:
    mlir::MLIRContext _ctx;
    mlir::DialectRegistry _registry;
    mlir::OwningOpRef<mlir::ModuleOp> _module;

    class MinimalOptionsSetup final : public OptionsSetup<MinimalOptionsSetup, PublicOptions> {
    public:
        using Base = OptionsSetup<MinimalOptionsSetup, PublicOptions>;
        using Base::Base;
        using Base::setupLitTestOptionsImpl;

        static void setupOptionsImpl(PublicOptions&, VPU::InitCompilerOptions&, const OV::Config&) {
        }
    };

    static constexpr llvm::StringLiteral INPUT_IR = R"(
    module @test {
        func.func @main() {
            return
        }
    }
)";

public:
    CompilerOptionInIRTest() {
        _ctx.appendDialectRegistry(_registry);
        _ctx.loadDialect<config::ConfigDialect>();
        _ctx.loadDialect<mlir::func::FuncDialect>();
    }

protected:
    // Create a Config object, populated with the option specified via OptT set to the requested value
    // The PLATFORM option is also set to the requested value; it is required to specify the platform,
    // as the OptionsSetup infrastructure relies on it
    template <typename OptT, typename OptValueT>
    OV::Config makeConfig(OptValueT value, config::Platform platform = config::Platform::NPU5010) {
        auto optionsDesc = std::make_shared<OV::OptionsDesc>();
        optionsDesc->add<OV::PLATFORM>();
        optionsDesc->add<OptT>();

        OV::Config config(optionsDesc);
        const auto platformStr = std::string(stringifyPlatform(platform).trim("NPU"));
        config.update({{std::string(OV::PLATFORM::key()), platformStr}, {std::string(OptT::key()), value}});
        return config;
    }

    // Drives the full production path: Config -> InitCompilerOptions -> pipeline -> IR
    void runInitCompilerPipeline(const OV::Config& ovConfig) {
        _module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &_ctx);
        ASSERT_TRUE(_module.get() != nullptr);

        MinimalOptionsSetup optionsSetup(ovConfig);

        mlir::PassManager pm(_module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        VPU::buildInitCompilerPipeline(pm, optionsSetup.getInitCompilerOptions(), Logger::global());
        ASSERT_TRUE(mlir::succeeded(pm.run(_module.get())));
    }

    mlir::ModuleOp getModule() {
        return _module.get();
    }
};

}  // namespace

TEST_F(CompilerOptionTest, PowerOptimized) {
    options->add<OV::POWER_OPTIMIZED>();

    bool found = false;
    options->walk([&](const OV::details::OptionConcept& opt) {
        if (opt.key() == OV::POWER_OPTIMIZED::key()) {
            found = true;
            EXPECT_EQ(opt.isPublic(), false);
            EXPECT_EQ(opt.mode(), OV::OptionMode::CompileTime);
        }
    });
    EXPECT_TRUE(found);

    EXPECT_FALSE(config.has<OV::POWER_OPTIMIZED>());

    config.update({{std::string(OV::POWER_OPTIMIZED::key()), "YES"}});
    ASSERT_TRUE(config.has<OV::POWER_OPTIMIZED>());
    EXPECT_EQ(config.get<OV::POWER_OPTIMIZED>(), true);

    config.update({{std::string(OV::POWER_OPTIMIZED::key()), "NO"}});
    ASSERT_TRUE(config.has<OV::POWER_OPTIMIZED>());
    EXPECT_EQ(config.get<OV::POWER_OPTIMIZED>(), false);
}

TEST_F(CompilerOptionTest, OptimizationLevel) {
    options->add<OV::OPTIMIZATION_LEVEL>();

    bool found = false;
    options->walk([&](const OV::details::OptionConcept& opt) {
        if (opt.key() == OV::OPTIMIZATION_LEVEL::key()) {
            found = true;
            EXPECT_EQ(opt.isPublic(), false);
            EXPECT_EQ(opt.mode(), OV::OptionMode::CompileTime);
        }
    });
    EXPECT_TRUE(found);

    EXPECT_FALSE(config.has<OV::OPTIMIZATION_LEVEL>());

    config.update({{std::string(OV::OPTIMIZATION_LEVEL::key()), "0"}});
    ASSERT_TRUE(config.has<OV::OPTIMIZATION_LEVEL>());
    EXPECT_EQ(config.get<OV::OPTIMIZATION_LEVEL>(), 0u);

    config.update({{std::string(OV::OPTIMIZATION_LEVEL::key()), "2"}});
    ASSERT_TRUE(config.has<OV::OPTIMIZATION_LEVEL>());
    EXPECT_EQ(config.get<OV::OPTIMIZATION_LEVEL>(), 2u);

    config.update({{std::string(OV::OPTIMIZATION_LEVEL::key()), "3"}});
    ASSERT_TRUE(config.has<OV::OPTIMIZATION_LEVEL>());
    EXPECT_EQ(config.get<OV::OPTIMIZATION_LEVEL>(), 3u);

    EXPECT_THROW(config.update({{std::string(OV::OPTIMIZATION_LEVEL::key()), "4"}}), vpux::Exception);
    EXPECT_EQ(config.get<OV::OPTIMIZATION_LEVEL>(), 3u);
}

TEST_F(CompilerOptionTest, OptimizationLevelOverridesCompilationModeParams) {
    options->add<OV::OPTIMIZATION_LEVEL>();
    options->add<OV::COMPILATION_MODE_PARAMS>();

    bool foundOptLevel = false;
    bool foundCompilationModeParams = false;
    options->walk([&](const OV::details::OptionConcept& opt) {
        if (opt.key() == OV::OPTIMIZATION_LEVEL::key()) {
            foundOptLevel = true;
        }
        if (opt.key() == OV::COMPILATION_MODE_PARAMS::key()) {
            foundCompilationModeParams = true;
        }
    });
    EXPECT_TRUE(foundOptLevel);
    EXPECT_TRUE(foundCompilationModeParams);

    EXPECT_FALSE(config.has<OV::OPTIMIZATION_LEVEL>());
    EXPECT_FALSE(config.has<OV::COMPILATION_MODE_PARAMS>());

    config.update({{std::string(OV::COMPILATION_MODE_PARAMS::key()), std::string("optimization-level=0")},
                   {std::string(OV::OPTIMIZATION_LEVEL::key()), std::string("2")}});

    ASSERT_TRUE(config.has<OV::OPTIMIZATION_LEVEL>());
    EXPECT_EQ(config.get<OV::OPTIMIZATION_LEVEL>(), 2u);
}

TEST_F(CompilerOptionInIRTest, PowerOptimizedEnabled) {
    runInitCompilerPipeline(makeConfig<OV::POWER_OPTIMIZED>("YES"));

    auto moduleOp = getModule();
    ASSERT_TRUE(moduleOp != nullptr);
    EXPECT_TRUE(config::isPowerOptimized(moduleOp));
}

TEST_F(CompilerOptionInIRTest, PowerOptimizedDisabled) {
    runInitCompilerPipeline(makeConfig<OV::POWER_OPTIMIZED>("NO"));

    auto moduleOp = getModule();
    ASSERT_TRUE(moduleOp != nullptr);
    EXPECT_FALSE(config::isPowerOptimized(moduleOp));
}
