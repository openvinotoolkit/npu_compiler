//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"

#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

void extractOpName(SmallVector<std::string>& names, mlir::Operation* op) {
    std::string name = op->getName().getStringRef().str();
    if (auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
        name += ":" + sym.getValue().str();
    }
    names.push_back(name);
}

struct CollectModuleNamesPass : public mlir::PassWrapper<CollectModuleNamesPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectModuleNamesPass)

    explicit CollectModuleNamesPass(SmallVector<std::string>& moduleNames): _moduleNames(moduleNames) {
    }

    void runOnOperation() override {
        extractOpName(_moduleNames, getOperation());
    }

private:
    SmallVector<std::string>& _moduleNames;
};

struct CollectFuncNamesPass : public mlir::PassWrapper<CollectFuncNamesPass, mlir::OperationPass<mlir::func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectFuncNamesPass)

    explicit CollectFuncNamesPass(SmallVector<std::string>& funcNames): _funcNames(funcNames) {
    }

    void runOnOperation() override {
        extractOpName(_funcNames, getOperation());
    }

private:
    SmallVector<std::string>& _funcNames;
};
}  // namespace

// This is not a real test, but rather an example of how to use mlir::PassPipelineOptions
TEST(MLIR_PassManagerTest, ImplicitBehaviourTests) {
    constexpr StringLiteral inputIR = R"(
        module @top_module {
          func.func @top_func() {
            func.func @top_func_nested() {
                return
            }
            return
          }
          module @inner_module {
            func.func @inner_func() {
              return
            }
          }
        }
    )";

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    SmallVector<std::string> moduleNames;
    SmallVector<std::string> funcNames;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(std::make_unique<CollectModuleNamesPass>(moduleNames));
    pm.addPass(std::make_unique<CollectFuncNamesPass>(funcNames));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    // In other words implicit nesting is not "recursive"
    EXPECT_TRUE(funcNames.size() == 1);
    EXPECT_TRUE(funcNames[0] == "func.func:top_func");

    // Implicit nesting does not work for the same type of operations that the pass manager works with.
    EXPECT_TRUE(moduleNames.size() == 1);
    EXPECT_TRUE(moduleNames[0] == "builtin.module:top_module");
}

TEST(MLIR_PassManagerTest, ExplicitBehaviourTests) {
    constexpr StringLiteral inputIR = R"(
        module @top_module {
          func.func @top_func() {
            func.func @top_func_nested() {

                return
            }
            return
          }
          module @inner_module {
            func.func @inner_func() {
              return
            }
          }
        }
    )";

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    SmallVector<std::string> funcNames;

    mlir::PassManager pm(&ctx);
    pm.addPass(std::make_unique<CollectFuncNamesPass>(funcNames));

    // error: 'builtin.module' op trying to schedule a pass on an unsupported operation
    EXPECT_FALSE(mlir::succeeded(pm.run(module.get())));

    mlir::PassManager properPmSetup(module.get()->getName(), mlir::OpPassManager::Nesting::Explicit);
    // explicitly nest for func::FuncOp
    properPmSetup.addNestedPass<mlir::func::FuncOp>(std::make_unique<CollectFuncNamesPass>(funcNames));

    EXPECT_TRUE(mlir::succeeded(properPmSetup.run(module.get())));

    // In other words implicit nesting is not "recursive"
    EXPECT_TRUE(funcNames.size() == 1);
    EXPECT_TRUE(funcNames[0] == "func.func:top_func");
}
