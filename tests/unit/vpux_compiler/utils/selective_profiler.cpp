//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compiler_profiling/selective_profiler.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "vpux/compiler/utils/npu_actions.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"

#include <gtest/gtest.h>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <string>

using namespace vpux;

namespace {

struct SimplePatternMatcher : mlir::OpRewritePattern<Const::DeclareOp> {
    SimplePatternMatcher(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
        this->setDebugName("SimplePatternMatcher");
    }

private:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final {
        auto oldContentAttr = origOp.getContentAttr();
        if (llvm::any_of(oldContentAttr.getTransformations(), [](const auto& t) {
                return mlir::isa<Const::RescaleAttr>(t);
            })) {
            return mlir::failure();
        }

        auto newContentAttr = oldContentAttr.transform().rescale(2.0).get();
        rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), std::move(newContentAttr));
        return mlir::success();
    }
};

class CounterPass : public mlir::PassWrapper<CounterPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterPass)

    CounterPass(int n = 1): _passId("counter-pass" + std::to_string(n)), _passName("CounterPass" + std::to_string(n)) {
    }

    llvm::StringRef getName() const override {
        return _passName;
    }

    llvm::StringRef getArgument() const override {
        return _passId;
    }

    void runOnOperation() override {
        auto moduleOp = getOperation();
        auto ctx = moduleOp.getContext();

        mlir::RewritePatternSet patterns(ctx);
        patterns.add<SimplePatternMatcher>(ctx);
        if (mlir::failed(mlir::applyPatternsGreedily(moduleOp, std::move(patterns),
                                                     vpux::getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
        }
    }

private:
    std::string _passId, _passName;
};

struct PseudoProfiler : vpux::compiler_profiling::SelectiveProfiler {
    std::string& records;
    PseudoProfiler(llvm::StringRef selection, std::string& records): SelectiveProfiler(selection), records(records) {
        records += '\n';
    }

private:
    void profileBeforeExecute(const mlir::tracing::Action* action, size_t depth) override {
        records += vpux::formatv("BEFORE: Action {0} with depth {1}", getActionName(action), depth).str() + '\n';
    }

    void profileAfterExecute(const mlir::tracing::Action* action, size_t depth) override {
        records += vpux::formatv("AFTER: Action {0} with depth {1}", getActionName(action), depth).str() + '\n';
    }

    std::string getActionName(const mlir::tracing::Action* action) const {
        if (const auto* npuAction = mlir::dyn_cast<vpux::NpuCompilerAction>(action)) {
            return npuAction->getName();
        }
        if (const auto* passAction = mlir::dyn_cast<mlir::PassExecutionAction>(action)) {
            return passAction->getPass().getName().str();
        }
        return action->getTag().str();
    }
};

}  // namespace

struct SelectiveProfilerTest : public MLIR_UnitBase {
    mlir::MLIRContext ctx;

    static constexpr llvm::StringLiteral SIMPLE_IR = R"(
    module @SimpleIr {
        func.func @main(%arg: tensor<42xf32>) -> tensor<42xf32> {
            %c = const.Declare tensor<1xf32> = dense<17.0> : tensor<1xf32>
            %res = IE.Add(%arg, %c) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                : tensor<42xf32>, tensor<1xf32> -> tensor<42xf32>
            return %res : tensor<42xf32>
        }
    }
    )";

    SelectiveProfilerTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<config::ConfigDialect>();
        ctx.loadDialect<Const::ConstDialect>();
        ctx.loadDialect<IE::IEDialect>();

        ctx.disableMultithreading();

        // make sure our action handler is always present regardless of
        // developer or non-developer build. this is good enough for the test.
        ctx.registerActionHandler(NpuActionHandler());
    }

    void runTest() {
        mlir::OwningOpRef<mlir::ModuleOp> m;
        ctx.executeAction<NpuCompilerAction>(
                [&]() {
                    m = mlir::parseSourceString<mlir::ModuleOp>(SIMPLE_IR, &ctx);
                },
                {}, "my-custom-test-action");
        ASSERT_NE(m.get(), nullptr);

        mlir::PassManager pm(&ctx);
        pm.addPass(std::make_unique<CounterPass>(0));
        pm.addPass(std::make_unique<CounterPass>(1));
        ASSERT_TRUE(mlir::succeeded(pm.run(*m)));
    }
};

TEST_F(SelectiveProfilerTest, Noop) {
    auto& npuActionHandler = getActionHandler(ctx);
    std::string records;
    npuActionHandler.registerObserver(std::make_unique<PseudoProfiler>("", records));

    runTest();

    ASSERT_EQ(records, std::string("\n"));
}

TEST_F(SelectiveProfilerTest, NpuAction) {
    auto& npuActionHandler = getActionHandler(ctx);
    std::string records;
    npuActionHandler.registerObserver(std::make_unique<PseudoProfiler>("my-custom-test-action", records));

    runTest();

    constexpr llvm::StringLiteral expectedString = R"(
BEFORE: Action my-custom-test-action with depth 0
AFTER: Action my-custom-test-action with depth 0
)";
    ASSERT_EQ(records, expectedString.str());
}

TEST_F(SelectiveProfilerTest, PassExecutionAction) {
    auto& npuActionHandler = getActionHandler(ctx);
    std::string records;
    npuActionHandler.registerObserver(std::make_unique<PseudoProfiler>("counter-pass0", records));

    runTest();

    // two greedy iterations since CounterPass0 makes 1 modification to IR (so
    // algorithm carries forward trying to converge); first iteration also tries
    // pattern application twice.
    constexpr llvm::StringLiteral expectedString = R"(
BEFORE: Action CounterPass0 with depth 0
BEFORE: Action GreedyPatternRewriteIteration with depth 1
BEFORE: Action apply-pattern with depth 2
AFTER: Action apply-pattern with depth 2
BEFORE: Action apply-pattern with depth 2
AFTER: Action apply-pattern with depth 2
AFTER: Action GreedyPatternRewriteIteration with depth 1
BEFORE: Action GreedyPatternRewriteIteration with depth 1
BEFORE: Action apply-pattern with depth 2
AFTER: Action apply-pattern with depth 2
AFTER: Action GreedyPatternRewriteIteration with depth 1
AFTER: Action CounterPass0 with depth 0
)";
    ASSERT_EQ(records, expectedString.str());
}

TEST_F(SelectiveProfilerTest, BothKindsOfActions) {
    auto& npuActionHandler = getActionHandler(ctx);
    std::string records;
    npuActionHandler.registerObserver(std::make_unique<PseudoProfiler>("my-custom.*|CounterPass1", records));

    runTest();

    // one greedy iteration since CounterPass1 makes no modifications to IR
    // (CounterPass0 already did).
    constexpr llvm::StringLiteral expectedString = R"(
BEFORE: Action my-custom-test-action with depth 0
AFTER: Action my-custom-test-action with depth 0
BEFORE: Action CounterPass1 with depth 0
BEFORE: Action GreedyPatternRewriteIteration with depth 1
BEFORE: Action apply-pattern with depth 2
AFTER: Action apply-pattern with depth 2
AFTER: Action GreedyPatternRewriteIteration with depth 1
AFTER: Action CounterPass1 with depth 0
)";
    ASSERT_EQ(records, expectedString.str());
}
