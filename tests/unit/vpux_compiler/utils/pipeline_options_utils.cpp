//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;
using MLIR_GetAttributeFromOption = MLIR_UnitBase;

using vpux::config::getAttributeFromOption;
namespace {

struct ManyOptionsPassOptions {
    bool boolOption = false;
    int64_t intOption = 42;
    std::string strOption = "hello";
    double doubleOption = 3.14;
};

class ManyOptionsPass : public vpux::ModulePass {
public:
    ManyOptionsPass(): vpux::ModulePass(::mlir::TypeID::get<ManyOptionsPass>()) {
    }
    ManyOptionsPass(const ManyOptionsPass& other): vpux::ModulePass(other) {
    }
    ManyOptionsPass(const ManyOptionsPassOptions& options): vpux::ModulePass(::mlir::TypeID::get<ManyOptionsPass>()) {
        boolOption = options.boolOption;
        intOption = options.intOption;
        strOption = options.strOption;
        doubleOption = options.doubleOption;
    }

    ::llvm::StringRef getName() const override {
        return "ManyOptionsPass";
    }
    void safeRunOnModule() override final {
    }
    std::unique_ptr<::mlir::Pass> clonePass() const override {
        return std::make_unique<ManyOptionsPass>(*static_cast<const ManyOptionsPass*>(this));
    }
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManyOptionsPass)

    mlir::Pass::Option<bool> boolOption{*this, "", ::llvm::cl::desc(""), ::llvm::cl::init(false)};
    mlir::Pass::Option<int64_t> intOption{*this, "", ::llvm::cl::desc(""), ::llvm::cl::init(42)};
    mlir::Pass::Option<std::string> strOption{*this, "", ::llvm::cl::desc(""), ::llvm::cl::init("hello")};
    mlir::Pass::Option<double> doubleOption{*this, "", ::llvm::cl::desc(""), ::llvm::cl::init(3.14)};
};

}  // namespace

TEST(MLIR_GetAttributeFromOption, BoolOption) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    ManyOptionsPassOptions options;
    options.boolOption = true;
    auto pass = std::make_unique<ManyOptionsPass>(options);
    auto attr = getAttributeFromOption(&ctx, pass->boolOption);
    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr);
    ASSERT_TRUE(boolAttr != nullptr);
    EXPECT_TRUE(boolAttr.getValue());
}

TEST(MLIR_GetAttributeFromOption, Int64Option) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    ManyOptionsPassOptions options;
    options.intOption = 73;
    auto pass = std::make_unique<ManyOptionsPass>(options);
    auto attr = getAttributeFromOption(&ctx, pass->intOption);
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
    ASSERT_TRUE(intAttr != nullptr);
    EXPECT_EQ(intAttr.getValue().getSExtValue(), 73);
}

TEST(MLIR_GetAttributeFromOption, StringOption) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    ManyOptionsPassOptions options;
    options.strOption = "bye";
    auto pass = std::make_unique<ManyOptionsPass>(options);
    auto attr = getAttributeFromOption(&ctx, pass->strOption);
    auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    ASSERT_TRUE(strAttr != nullptr);
    EXPECT_EQ(strAttr.getValue(), "bye");
}

TEST(MLIR_GetAttributeFromOption, DoubleOption) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    ManyOptionsPassOptions options;
    options.doubleOption = 2.71828;
    auto pass = std::make_unique<ManyOptionsPass>(options);
    auto attr = getAttributeFromOption(&ctx, pass->doubleOption);
    auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr);
    ASSERT_TRUE(floatAttr != nullptr);
    EXPECT_DOUBLE_EQ(floatAttr.getValueAsDouble(), 2.71828);
}
