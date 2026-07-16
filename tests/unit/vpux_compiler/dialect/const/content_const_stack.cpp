//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/constant_call_stack.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <gtest/gtest.h>

#include <mlir/Debug/ExecutionContext.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace vpux;

namespace {

constexpr const char* FUNC_IR = R"(
            module @test {
                func.func @main() -> tensor<2x3x1x1xf32> {
                    %cst = const.Declare tensor<2x3x1x1xf32> = dense_resource<cstin> : tensor<2x3x1x1xf32>
                    return %cst : tensor<2x3x1x1xf32>
                }
            }

            {-#
            dialect_resources: {
                builtin: {
                cstin: "0x0400000001000000"
                }
            }
            #-}
        )";

class MultithreadVerificationState final {
public:
    explicit MultithreadVerificationState(size_t expectedParticipants): _expectedParticipants(expectedParticipants) {
    }

    void recordThread(std::thread::id threadId) {
        std::unique_lock<std::mutex> lock(_mutex);
        _threadIds.insert(threadId);

        ++_arrived;
        if (_arrived == _expectedParticipants) {
            _released = true;
            _cv.notify_all();
            return;
        }

        _cv.wait(lock, [&]() {
            return _released;
        });
    }

    bool wasMultithreaded() const {
        std::unique_lock<std::mutex> lock(_mutex);
        return _threadIds.size() == _expectedParticipants;
    }

private:
    size_t _expectedParticipants = 0;
    mutable std::mutex _mutex;
    std::condition_variable_any _cv;
    std::unordered_set<std::thread::id> _threadIds;
    size_t _arrived = 0;
    bool _released = false;
};

}  // namespace

class AddTransformationsPass :
        public mlir::PassWrapper<AddTransformationsPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    AddTransformationsPass() = default;
    explicit AddTransformationsPass(MultithreadVerificationState* mtVerificationState)
            : _mtVerificationState(mtVerificationState) {
    }

    void runOnOperation() override {
        auto funcOp = getOperation();

        auto currentThreadId = std::this_thread::get_id();
        if (_mtVerificationState != nullptr) {
            _mtVerificationState->recordThread(currentThreadId);
        }
        funcOp.walk([&](vpux::Const::DeclareOp declareOp) {
            const auto& baseContent = declareOp.getContentAttr().getBaseContent();
            auto newContentAttrSetup = Const::ContentSetup(baseContent, baseContent.getType()).add(1);
            auto newContentAttr = Const::ContentAttr::get(baseContent, std::move(newContentAttrSetup));
            declareOp.setContentAttr(std::move(newContentAttr));
        });
    }

private:
    MultithreadVerificationState* _mtVerificationState = nullptr;
};

namespace {
class MLIR_ContentSetupCallStackTest : public MLIR_UnitBase {
public:
    MLIR_ContentSetupCallStackTest(): MLIR_UnitBase() {
        configureContext();
    }

    void configureContext() {
        targetCtx.appendDialectRegistry(registry);
        targetCtx.loadDialect<Const::ConstDialect, IE::IEDialect, mlir::func::FuncDialect>();
        mlir::tracing::ExecutionContext debugExectx;
        debugObserver = std::make_shared<vpux::Const::CallStackObserver>();
        debugExectx.registerObserver(debugObserver.get());
        targetCtx.registerActionHandler(debugExectx);
    }

    mlir::MLIRContext targetCtx;
    std::shared_ptr<vpux::Const::CallStackObserver> debugObserver;
};

class MLIR_ContentSetupCallStackNoObserverTest : public MLIR_UnitBase {
public:
    MLIR_ContentSetupCallStackNoObserverTest(): MLIR_UnitBase() {
        configureContext();
    }

    void configureContext() {
        targetCtx.appendDialectRegistry(registry);
        targetCtx.loadDialect<Const::ConstDialect, IE::IEDialect, mlir::func::FuncDialect>();
        mlir::tracing::ExecutionContext debugExectx;
        targetCtx.registerActionHandler(debugExectx);
    }

    mlir::MLIRContext targetCtx;
};
}  // namespace

void runMT(mlir::OwningOpRef<mlir::ModuleOp>& module, size_t hwThreads) {
    using namespace vpux::Const;

    mlir::func::FuncOp originalFunc;
    module->walk([&](mlir::func::FuncOp f) {
        originalFunc = f;
    });
    ASSERT_TRUE(originalFunc);

    auto* ctx = module->getContext();
    originalFunc.setName("main0");
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module->getBody());
    for (unsigned i = 1; i < hwThreads; ++i) {
        auto clone = originalFunc.clone();
        clone.setName(("main" + std::to_string(i)).c_str());
        builder.insert(clone);
    }
    const auto funcs = module->getOps<mlir::func::FuncOp>();
    ASSERT_EQ(std::distance(funcs.begin(), funcs.end()), hwThreads);

    ctx->enableMultithreading(true);
    MultithreadVerificationState verificationState(hwThreads);

    mlir::PassManager pm(ctx);
    pm.nest<mlir::func::FuncOp>().addPass(std::make_unique<AddTransformationsPass>(&verificationState));
    ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    EXPECT_TRUE(verificationState.wasMultithreaded());
}

TEST_F(MLIR_ContentSetupCallStackTest, MultiThreadedContentSetupTracingBase) {
    const unsigned hwThreads = std::thread::hardware_concurrency();
    if (hwThreads < 2) {
        GTEST_SKIP() << "Single-core machine; cannot verify multithreaded execution";
        return;
    }

    auto module = mlir::parseSourceString<mlir::ModuleOp>(FUNC_IR, &targetCtx);
    ASSERT_TRUE(module);
    runMT(module, hwThreads);
    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(&targetCtx);

    module->walk([&](vpux::Const::DeclareOp declareOp) {
        auto contentAttr = declareOp.getContentAttr();
        auto baseContent = contentAttr.getBaseContent();
        vpux::Const::AddAttr addAttr = nullptr;

        for (auto attr : contentAttr.getTransformations()) {
            if (auto add = mlir::dyn_cast<vpux::Const::AddAttr>(attr)) {
                addAttr = add;
            }
        }

        EXPECT_TRUE(addAttr != nullptr);
        EXPECT_TRUE(csCache.getSpecificCallStack(baseContent, addAttr).find("AddTransformationsPass") !=
                    std::string::npos);
    });
}

TEST_F(MLIR_ContentSetupCallStackNoObserverTest, NoObserver) {
    const unsigned hwThreads = std::thread::hardware_concurrency();
    if (hwThreads < 2) {
        GTEST_SKIP() << "Single-core machine; cannot verify multithreaded execution";
        return;
    }

    auto module = mlir::parseSourceString<mlir::ModuleOp>(FUNC_IR, &targetCtx);
    ASSERT_TRUE(module);
    runMT(module, hwThreads);
    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(&targetCtx);

    module->walk([&](vpux::Const::DeclareOp declareOp) {
        auto contentAttr = declareOp.getContentAttr();
        auto baseContent = contentAttr.getBaseContent();
        vpux::Const::AddAttr addAttr = nullptr;

        for (auto attr : contentAttr.getTransformations()) {
            if (auto add = mlir::dyn_cast<vpux::Const::AddAttr>(attr)) {
                addAttr = add;
            }
        }

        EXPECT_TRUE(addAttr != nullptr);
        EXPECT_TRUE(csCache.getSpecificCallStack(baseContent, addAttr).empty());
    });
}
