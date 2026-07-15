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
#include <mlir/IR/Action.h>
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
struct DummyAction final : mlir::tracing::ActionImpl<DummyAction> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyAction)
    using ActionImpl::ActionImpl;
    static constexpr llvm::StringLiteral tag = "dummy-action";
};
}  // namespace

namespace {
class MLIR_ContentSetupCallStackActionTest : public MLIR_UnitBase {
public:
    MLIR_ContentSetupCallStackActionTest(): MLIR_UnitBase() {
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

class MLIR_ContentSetupCallStackNoObserverActionTest : public MLIR_UnitBase {
public:
    MLIR_ContentSetupCallStackNoObserverActionTest(): MLIR_UnitBase() {
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

Const::ContentSetup runAction(vpux::Const::TraceId id, mlir::RankedTensorType baseType, mlir::MLIRContext& ctx) {
    using namespace vpux::Const;
    auto baseContentAttrSetup = Const::ContentSetup(id, baseType);

    ctx.executeAction<DummyAction>(
            [&]() {
                baseContentAttrSetup = baseContentAttrSetup.add(1).add(2);

                auto actualTransformations = baseContentAttrSetup.getTransformations();
                EXPECT_EQ(actualTransformations.size(), 1);
            },
            {});
    return baseContentAttrSetup;
}

TEST_F(MLIR_ContentSetupCallStackActionTest, AddAndFuseTransformation) {
    using namespace vpux::Const;

    SmallVector<uint8_t> data = {1, 1, 1, 1};
    auto baseType = mlir::RankedTensorType::get(ArrayRef<int64_t>{4}, getInt8Type(&targetCtx));
    auto baseAttr = Const::createConstContent(baseType, ArrayRef(data));

    auto baseContentAttrSetup = runAction(baseAttr, baseType, targetCtx);
    auto base = Const::ContentAttr::get(baseAttr, baseContentAttrSetup);
    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(&targetCtx);
    AddAttr addAttr = nullptr;
    for (auto attr : base.getTransformations()) {
        if (auto add = mlir::dyn_cast<AddAttr>(attr)) {
            addAttr = add;
        }
    }
    EXPECT_TRUE(addAttr != nullptr);
    EXPECT_TRUE(csCache.getSpecificCallStack(baseAttr, addAttr).find("dummy-action") != std::string::npos);
    EXPECT_EQ(csCache.getSpecificCallStack(baseAttr, nullptr), "UNKNOWN_CALL_STACK");
}

TEST_F(MLIR_ContentSetupCallStackActionTest, NullptrTest) {
    using namespace vpux::Const;

    auto baseType = mlir::RankedTensorType::get(ArrayRef<int64_t>{4}, getInt8Type(&targetCtx));
    auto baseContentAttrSetup = runAction(nullptr, baseType, targetCtx);

    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(&targetCtx);
    AddAttr addAttr = nullptr;
    for (auto attr : baseContentAttrSetup.getTransformations()) {
        if (auto add = mlir::dyn_cast<AddAttr>(attr)) {
            addAttr = add;
        }
    }
    EXPECT_TRUE(addAttr != nullptr);
    EXPECT_EQ(csCache.getSpecificCallStack(nullptr, addAttr), "UNKNOWN_CALL_STACK");
    EXPECT_EQ(csCache.getSpecificCallStack(nullptr, nullptr), "UNKNOWN_CALL_STACK");
}

TEST_F(MLIR_ContentSetupCallStackNoObserverActionTest, NoObserverAction) {
    using namespace vpux::Const;

    SmallVector<uint8_t> data = {1, 1, 1, 1};
    auto baseType = mlir::RankedTensorType::get(ArrayRef<int64_t>{4}, getInt8Type(&targetCtx));
    auto baseAttr = Const::createConstContent(baseType, ArrayRef(data));
    auto baseContentAttrSetup = runAction(baseAttr, baseType, targetCtx);

    auto base = Const::ContentAttr::get(baseAttr, baseContentAttrSetup);
    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(&targetCtx);
    AddAttr addAttr = nullptr;
    for (auto attr : base.getTransformations()) {
        if (auto add = mlir::dyn_cast<AddAttr>(attr)) {
            addAttr = add;
        }
    }
    EXPECT_TRUE(addAttr != nullptr);
    EXPECT_TRUE(csCache.getSpecificCallStack(baseAttr, addAttr).empty());
}
