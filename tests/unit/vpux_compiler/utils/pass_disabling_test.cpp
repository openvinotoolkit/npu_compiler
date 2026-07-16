//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "vpux/compiler/utils/pass_disabling_callback.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <cstdlib>

using namespace vpux;

namespace {
class CounterPass : public mlir::PassWrapper<CounterPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterPass)

    CounterPass(int& counter, int n = 1)
            : _counter(counter),
              _n(n),
              _passId("counter-pass" + std::to_string(n)),
              _passName("CounterPass" + std::to_string(n)) {
    }

    llvm::StringRef getName() const override {
        return _passName;
    }

    llvm::StringRef getArgument() const override {
        return _passId;
    }

    void runOnOperation() override {
        _counter += _n;
    }

private:
    int& _counter;
    int _n;
    std::string _passId, _passName;
};

void runTest(mlir::MLIRContext& ctx, mlir::PassManager& pm) {
    mlir::OwningOpRef<mlir::ModuleOp> mod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    ASSERT_TRUE(mlir::succeeded(pm.run(*mod)));
}

NpuActionHandler makeActionHandler(StringRef disabledPasses) {
    NpuActionHandler handler;
    handler.setCallback(PassDisablingCallback(disabledPasses));
    handler.addBreakpointManager(PassDisablingCallback::createBreakpointManager());
    return handler;
}
}  // namespace

using PassDisablingTest = MLIR_UnitBase;

TEST_F(PassDisablingTest, PassIncrementsWhenEnabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(makeActionHandler(""));

    int counter = 0;

    pm.addPass(std::make_unique<CounterPass>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 1);
}

TEST_F(PassDisablingTest, PassDoesNotIncrementWhenDisabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(makeActionHandler("CounterPass1"));

    int counter = 0;

    pm.addPass(std::make_unique<CounterPass>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
}

TEST_F(PassDisablingTest, MultiplePassesIncrementWhenEnabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(makeActionHandler(""));

    int counter = 0;
    int counter2 = 0;

    pm.addPass(std::make_unique<CounterPass>(counter));
    pm.addPass(std::make_unique<CounterPass>(counter2));
    pm.addPass(std::make_unique<CounterPass>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 2);
    EXPECT_EQ(counter2, 1);
}

TEST_F(PassDisablingTest, MultipleDoNotIncrementWhenDisabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(makeActionHandler("counter-pass1"));

    int counter = 0;
    int counter2 = 0;

    pm.addPass(std::make_unique<CounterPass>(counter));
    pm.addPass(std::make_unique<CounterPass>(counter2));
    pm.addPass(std::make_unique<CounterPass>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
    EXPECT_EQ(counter2, 0);
}

TEST_F(PassDisablingTest, RegexAlternationMatchesTwoPasses) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(makeActionHandler("counter-pass1|counter-pass2"));

    int counter = 0;
    int counter2 = 0;
    int counter3 = 0;

    pm.addPass(std::make_unique<CounterPass>(counter));
    pm.addPass(std::make_unique<CounterPass>(counter2, 2));
    pm.addPass(std::make_unique<CounterPass>(counter3, 3));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
    EXPECT_EQ(counter2, 0);
    EXPECT_EQ(counter3, 3);
}

TEST_F(PassDisablingTest, RegexDotMatchesAll) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(makeActionHandler("counter-pass."));

    int counter = 0;
    int counter2 = 0;
    int counter3 = 0;

    pm.addPass(std::make_unique<CounterPass>(counter));
    pm.addPass(std::make_unique<CounterPass>(counter2, 2));
    pm.addPass(std::make_unique<CounterPass>(counter3, 3));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
    EXPECT_EQ(counter2, 0);
    EXPECT_EQ(counter3, 0);
}
