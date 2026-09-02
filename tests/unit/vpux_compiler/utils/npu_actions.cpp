//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/npu_actions.hpp"
#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {
struct TestActionHandler {
    size_t counter = 0;

    void operator()(mlir::function_ref<void()> transform, const mlir::tracing::Action& action) {
        ASSERT_TRUE(mlir::isa<NpuCompilerAction>(action));
        transform();
        counter++;
    }
};

}  // namespace

TEST(NpuCompilerActionTest, SkipExecuteAction) {
    mlir::MLIRContext ctx;
    ctx.registerActionHandler(TestActionHandler());
    const auto* handler = ctx.getActionHandler().target<TestActionHandler>();
    ASSERT_NE(handler, nullptr);

    bool lambdaCalled = false;
    details::ActionCreatorBase(&ctx, {}, "test") << [&] {
        lambdaCalled = true;
    };

    ASSERT_TRUE(lambdaCalled);
    ASSERT_EQ(handler->counter, 0);
}

TEST(NpuCompilerActionTest, ExecuteAction) {
    mlir::MLIRContext ctx;
    ctx.registerActionHandler(TestActionHandler());
    const auto* handler = ctx.getActionHandler().target<TestActionHandler>();
    ASSERT_NE(handler, nullptr);

    bool lambdaCalled = false;
    details::ActionCreatorWithContextDispatch(&ctx, {}, "test") << [&] {
        lambdaCalled = true;
    };

    ASSERT_TRUE(lambdaCalled);
    ASSERT_EQ(handler->counter, 1);
}

TEST(NpuCompilerActionTest, ExecuteActionViaMacro) {
    mlir::MLIRContext ctx;
    ctx.registerActionHandler(TestActionHandler());
    const auto* handler = ctx.getActionHandler().target<TestActionHandler>();
    ASSERT_NE(handler, nullptr);

    bool lambdaCalled = false;
    NPU_EXECUTE_ACTION(&ctx, "test", {}) {
        lambdaCalled = true;
    };

    ASSERT_TRUE(lambdaCalled);
    constexpr size_t count = isDeveloperBuild() ? 1 : 0;
    ASSERT_EQ(handler->counter, count);
}
