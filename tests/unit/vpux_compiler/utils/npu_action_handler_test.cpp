//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "common/utils.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <cstdlib>
#include <string>
#include <vector>

using namespace vpux;

namespace {

constexpr llvm::StringLiteral SKIP_PASS_ARGUMENT = "skip-pass";
constexpr llvm::StringLiteral APPLY_PASS_ARGUMENT = "apply-pass";

enum class ObserverEventKind { Before, After };

struct ObserverEvent {
    ObserverEventKind kind;
    std::string passArgument;
    bool hasBreakpoint = false;
    bool willExecute = false;
};

class TestPass final : public mlir::PassWrapper<TestPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPass)

    TestPass(int& counter, llvm::StringRef argument): _counter(counter), _argument(argument.str()) {
    }

    llvm::StringRef getName() const override {
        return _argument;
    }

    llvm::StringRef getArgument() const override {
        return _argument;
    }

    void runOnOperation() override {
        ++_counter;
    }

private:
    int& _counter;
    std::string _argument;
};

class TestBreakpoint final : public mlir::tracing::BreakpointBase<TestBreakpoint> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBreakpoint)

    void print(llvm::raw_ostream& os) const override {
        os << "test breakpoint";
    }
};

class PassExecutionBreakpointManager final :
        public mlir::tracing::BreakpointManagerBase<PassExecutionBreakpointManager> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PassExecutionBreakpointManager)

    mlir::tracing::Breakpoint* match(const mlir::tracing::Action& action) const override {
        if (llvm::isa<mlir::PassExecutionAction>(&action) && _breakpoint.isEnabled()) {
            return &_breakpoint;
        }

        return nullptr;
    }

private:
    mutable TestBreakpoint _breakpoint;
};

class RecordingObserver final : public NpuActionHandler::Observer {
public:
    explicit RecordingObserver(std::vector<ObserverEvent>& events): _events(events) {
    }

    void beforeExecute(const mlir::tracing::ActionActiveStack* actionStack, mlir::tracing::Breakpoint* breakpoint,
                       bool willExecute) override {
        record(ObserverEventKind::Before, actionStack, breakpoint != nullptr, willExecute);
    }

    void afterExecute(const mlir::tracing::ActionActiveStack* actionStack) override {
        record(ObserverEventKind::After, actionStack, false, true);
    }

private:
    void record(ObserverEventKind kind, const mlir::tracing::ActionActiveStack* actionStack, bool hasBreakpoint,
                bool willExecute) {
        const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
        if (passAction == nullptr) {
            return;
        }

        _events.push_back({kind, passAction->getPass().getArgument().str(), hasBreakpoint, willExecute});
    }

private:
    std::vector<ObserverEvent>& _events;
};

static mlir::LogicalResult runPasses(mlir::MLIRContext& ctx, mlir::PassManager& pm) {
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    return pm.run(module.get());
}

static void addSkipAndApplyPasses(mlir::PassManager& pm, int& skipCounter, int& applyCounter) {
    pm.addPass(std::make_unique<TestPass>(skipCounter, SKIP_PASS_ARGUMENT));
    pm.addPass(std::make_unique<TestPass>(applyCounter, APPLY_PASS_ARGUMENT));
}

static NpuActionHandler::Control skipOnlySkipPass(const mlir::tracing::ActionActiveStack* actionStack) {
    const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
    if (passAction != nullptr && passAction->getPass().getArgument() == SKIP_PASS_ARGUMENT) {
        return NpuActionHandler::Control::Skip;
    }

    return NpuActionHandler::Control::Apply;
}

}  // namespace

using NpuActionHandlerTest = MLIR_UnitBase;

TEST_F(NpuActionHandlerTest, ObserverRunsBeforeAndAfterEachAppliedPass) {
    mlir::MLIRContext ctx(registry);
    ctx.disableMultithreading();
    mlir::PassManager pm(&ctx);

    std::vector<ObserverEvent> events;
    NpuActionHandler handler;
    handler.registerObserver(std::make_unique<RecordingObserver>(events));
    ctx.registerActionHandler(handler);

    int skipCounter = 0;
    int applyCounter = 0;
    addSkipAndApplyPasses(pm, skipCounter, applyCounter);

    ASSERT_TRUE(mlir::succeeded(runPasses(ctx, pm)));

    EXPECT_EQ(skipCounter, 1);
    EXPECT_EQ(applyCounter, 1);

    ASSERT_EQ(events.size(), 4);
    EXPECT_EQ(events[0].kind, ObserverEventKind::Before);
    EXPECT_EQ(events[0].passArgument, SKIP_PASS_ARGUMENT.str());
    EXPECT_FALSE(events[0].hasBreakpoint);
    EXPECT_TRUE(events[0].willExecute);

    EXPECT_EQ(events[1].kind, ObserverEventKind::After);
    EXPECT_EQ(events[1].passArgument, SKIP_PASS_ARGUMENT.str());

    EXPECT_EQ(events[2].kind, ObserverEventKind::Before);
    EXPECT_EQ(events[2].passArgument, APPLY_PASS_ARGUMENT.str());
    EXPECT_FALSE(events[2].hasBreakpoint);
    EXPECT_TRUE(events[2].willExecute);

    EXPECT_EQ(events[3].kind, ObserverEventKind::After);
    EXPECT_EQ(events[3].passArgument, APPLY_PASS_ARGUMENT.str());
}

TEST_F(NpuActionHandlerTest, BreakpointCallbackSkipsOrAppliesPasses) {
    mlir::MLIRContext ctx(registry);
    ctx.disableMultithreading();
    mlir::PassManager pm(&ctx);

    auto callback = skipOnlySkipPass;
    NpuActionHandler handler;
    handler.setCallback(callback);
    handler.addBreakpointManager(std::make_unique<PassExecutionBreakpointManager>());
    ctx.registerActionHandler(handler);

    int skipCounter = 0;
    int applyCounter = 0;
    addSkipAndApplyPasses(pm, skipCounter, applyCounter);

    ASSERT_TRUE(mlir::succeeded(runPasses(ctx, pm)));

    EXPECT_EQ(skipCounter, 0);
    EXPECT_EQ(applyCounter, 1);
}

TEST_F(NpuActionHandlerTest, ObserverBeforeRunsForSkippedPassAndAfterOnlyForAppliedPass) {
    mlir::MLIRContext ctx(registry);
    ctx.disableMultithreading();
    mlir::PassManager pm(&ctx);

    std::vector<ObserverEvent> events;
    auto callback = skipOnlySkipPass;
    NpuActionHandler handler;
    handler.setCallback(callback);
    handler.registerObserver(std::make_unique<RecordingObserver>(events));
    handler.addBreakpointManager(std::make_unique<PassExecutionBreakpointManager>());
    ctx.registerActionHandler(handler);

    int skipCounter = 0;
    int applyCounter = 0;
    addSkipAndApplyPasses(pm, skipCounter, applyCounter);

    ASSERT_TRUE(mlir::succeeded(runPasses(ctx, pm)));

    EXPECT_EQ(skipCounter, 0);
    EXPECT_EQ(applyCounter, 1);

    ASSERT_EQ(events.size(), 3);
    EXPECT_EQ(events[0].kind, ObserverEventKind::Before);
    EXPECT_EQ(events[0].passArgument, SKIP_PASS_ARGUMENT.str());
    EXPECT_TRUE(events[0].hasBreakpoint);
    EXPECT_FALSE(events[0].willExecute);

    EXPECT_EQ(events[1].kind, ObserverEventKind::Before);
    EXPECT_EQ(events[1].passArgument, APPLY_PASS_ARGUMENT.str());
    EXPECT_TRUE(events[1].hasBreakpoint);
    EXPECT_TRUE(events[1].willExecute);

    EXPECT_EQ(events[2].kind, ObserverEventKind::After);
    EXPECT_EQ(events[2].passArgument, APPLY_PASS_ARGUMENT.str());
}
