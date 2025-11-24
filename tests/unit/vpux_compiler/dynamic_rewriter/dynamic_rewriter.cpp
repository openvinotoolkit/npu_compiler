//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/dynamic_rewriter/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;
namespace {

// Test dynamic rewriter registration and execution
// Run cmd: npuUnitTests --gtest_filter="DynamicRewriterTest.*"

//
// DummyRewriterN for testing
//

#define DEFINE_CUSTOM_DUMMY_REWRITE(IDX)                                                                               \
    class DummyRewriter##IDX final : public mlir::RewritePattern {                                                     \
    public:                                                                                                            \
        DummyRewriter##IDX(mlir::MLIRContext* ctx, Logger log)                                                         \
                : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, ctx), _log(std::move(log)) { \
            setDebugName("dummy-rewriter-" #IDX);                                                                      \
            _log.trace("DummyRewriter" #IDX " created");                                                               \
        }                                                                                                              \
        mlir::LogicalResult matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const override {                 \
            _log.trace("DummyRewriter" #IDX " executed");                                                              \
            return mlir::failure();                                                                                    \
        }                                                                                                              \
                                                                                                                       \
    private:                                                                                                           \
        Logger _log;                                                                                                   \
    }

DEFINE_CUSTOM_DUMMY_REWRITE(0);
DEFINE_CUSTOM_DUMMY_REWRITE(1);
DEFINE_CUSTOM_DUMMY_REWRITE(2);
DEFINE_CUSTOM_DUMMY_REWRITE(3);
DEFINE_CUSTOM_DUMMY_REWRITE(4);
DEFINE_CUSTOM_DUMMY_REWRITE(5);
DEFINE_CUSTOM_DUMMY_REWRITE(6);
DEFINE_CUSTOM_DUMMY_REWRITE(7);
DEFINE_CUSTOM_DUMMY_REWRITE(8);
DEFINE_CUSTOM_DUMMY_REWRITE(9);
DEFINE_CUSTOM_DUMMY_REWRITE(10);
DEFINE_CUSTOM_DUMMY_REWRITE(11);
DEFINE_CUSTOM_DUMMY_REWRITE(12);
DEFINE_CUSTOM_DUMMY_REWRITE(13);
DEFINE_CUSTOM_DUMMY_REWRITE(14);
DEFINE_CUSTOM_DUMMY_REWRITE(15);
DEFINE_CUSTOM_DUMMY_REWRITE(16);

void registerAllRewriters(vpux::RewriterRegistry& registry, Logger log) {
    registry.registerRewriter<DummyRewriter0>("dummy-rewriter-0", log);
    registry.registerRewriter<DummyRewriter1>("dummy-rewriter-1", log);
    registry.registerRewriter<DummyRewriter2>("dummy-rewriter-2", log);
    registry.registerRewriter<DummyRewriter3>("dummy-rewriter-3", log);
    registry.registerRewriter<DummyRewriter4>("dummy-rewriter-4", log);
    registry.registerRewriter<DummyRewriter5>("dummy-rewriter-5", log);
    registry.registerRewriter<DummyRewriter6>("dummy-rewriter-6", log);
    registry.registerRewriter<DummyRewriter7>("dummy-rewriter-7", log);
    registry.registerRewriter<DummyRewriter8>("dummy-rewriter-8", log);
    registry.registerRewriter<DummyRewriter9>("dummy-rewriter-9", log);
    registry.registerRewriter<DummyRewriter10>("dummy-rewriter-10", log);
    registry.registerRewriter<DummyRewriter11>("dummy-rewriter-11", log);
    registry.registerRewriter<DummyRewriter12>("dummy-rewriter-12", log);
    registry.registerRewriter<DummyRewriter13>("dummy-rewriter-13", log);
    registry.registerRewriter<DummyRewriter14>("dummy-rewriter-14", log);
    registry.registerRewriter<DummyRewriter15>("dummy-rewriter-15", log);
    registry.registerRewriter<DummyRewriter16>("dummy-rewriter-16", log);
}

void registerOptimizeSet1(vpux::RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet(
            "optimize-set-1",
            [&](Logger log) {
                registry.registerRewriter<DummyRewriter0>("dummy-rewriter-0", log);
                registry.registerRewriter<DummyRewriter1>("dummy-rewriter-1", log);
                registry.registerRewriter<DummyRewriter2>("dummy-rewriter-2", log);
            },
            log);
}

void registerOptimizeSet2(vpux::RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet(
            "optimize-set-2",
            [&](Logger log) {
                registry.registerRewriter<DummyRewriter3>("dummy-rewriter-3", log);
                registry.registerRewriter<DummyRewriter4>("dummy-rewriter-4", log);
                registry.registerRewriter<DummyRewriter5>("dummy-rewriter-5", log);
            },
            log);
}

void registerOptimizeSet3(vpux::RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet(
            "optimize-set-3",
            [&](Logger log) {
                registry.registerRewriter<DummyRewriter6>("dummy-rewriter-6", log);
                registry.registerRewriter<DummyRewriter7>("dummy-rewriter-7", log);
                registry.registerRewriter<DummyRewriter8>("dummy-rewriter-8", log);
            },
            log);
}

void registerOptimizeSet4(vpux::RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet(
            "optimize-set-4",
            [&](Logger log) {
                registry.registerRewriter<DummyRewriter9>("dummy-rewriter-9", log);
                registry.registerRewriter<DummyRewriter10>("dummy-rewriter-10", log);
                registry.registerRewriter<DummyRewriter11>("dummy-rewriter-11", log);
            },
            log);
}

void registerOverlappingGroup(vpux::RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet(
            "overlapping-group",
            [&](Logger log) {
                // This group has heavy overlap with other sections
                registry.registerRewriter<DummyRewriter0>("dummy-rewriter-0", log);
                registry.registerRewriter<DummyRewriter3>("dummy-rewriter-3", log);
                registry.registerRewriter<DummyRewriter6>("dummy-rewriter-6", log);
                registry.registerRewriter<DummyRewriter9>("dummy-rewriter-9", log);
                registry.registerRewriter<DummyRewriter12>("dummy-rewriter-12", log);
                registry.registerRewriter<DummyRewriter13>("dummy-rewriter-13", log);
            },
            log);
}

void setupTestRegistry(vpux::RewriterRegistry& registry, Logger log) {
    registerAllRewriters(registry, log);
    registerOptimizeSet1(registry, log);
    registerOptimizeSet2(registry, log);
    registerOptimizeSet3(registry, log);
    registerOptimizeSet4(registry, log);
    registerOverlappingGroup(registry, log);
}

class DynamicRewriterTest : public testing::Test {
public:
    void SetUp() override {
        _registry = RegistryManager::createCustomRegistry();
    }

    void TearDown() override {
        _registry->clear();
    }

protected:
    mlir::MLIRContext _context;
    std::unique_ptr<RewriterRegistry> _registry;
    vpux::Logger _log = vpux::Logger("DynamicRewriterTest", vpux::LogLevel::Trace);
};

TEST_F(DynamicRewriterTest, AddAllRewriters) {
    registerAllRewriters(*_registry, _log);
    auto registeredRewriters = _registry->getRegisteredRewriters();
    EXPECT_GE(registeredRewriters.size(), 17);

    // Verify all rewriters are added to pattern set
    mlir::RewritePatternSet patterns(&_context);
    _registry->addAllRewriters(&_context, _log, patterns);
    EXPECT_EQ(patterns.getNativePatterns().size(), registeredRewriters.size());
}

TEST_F(DynamicRewriterTest, AddSingleRewriter) {
    _registry->registerRewriter<DummyRewriter5>("dummy-rewriter-5", _log);
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-5"));

    // Verify the rewriter is added to pattern set
    mlir::RewritePatternSet patterns(&_context);
    bool success = _registry->addRewriter(&_context, _log, "dummy-rewriter-5", patterns);
    EXPECT_TRUE(success);
    EXPECT_EQ(patterns.getNativePatterns().size(), 1);
    EXPECT_EQ(patterns.getNativePatterns()[0]->getDebugName(), "dummy-rewriter-5");
}

TEST_F(DynamicRewriterTest, AddRewriterSet) {
    registerOptimizeSet1(*_registry, _log);
    auto registeredRewriters = _registry->getRegisteredRewriters();
    EXPECT_GE(registeredRewriters.size(), 3);
    EXPECT_TRUE(_registry->hasRewriterSet("optimize-set-1"));
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-0"));
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-1"));
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-2"));
    // Check section content
    auto sections = _registry->getRegisteredRewriterSets();
    auto it = sections.find("optimize-set-1");
    EXPECT_NE(it, sections.end());
    auto sectionRewriters = it->second;
    EXPECT_EQ(sectionRewriters.size(), 3);
    EXPECT_NE(std::find(sectionRewriters.begin(), sectionRewriters.end(), "dummy-rewriter-0"), sectionRewriters.end());
    EXPECT_NE(std::find(sectionRewriters.begin(), sectionRewriters.end(), "dummy-rewriter-1"), sectionRewriters.end());
    EXPECT_NE(std::find(sectionRewriters.begin(), sectionRewriters.end(), "dummy-rewriter-2"), sectionRewriters.end());

    mlir::RewritePatternSet patterns(&_context);
    bool success = _registry->addRewriterSet(&_context, _log, "optimize-set-1", patterns);
    EXPECT_TRUE(success);
    // Verify the rewriter is added to pattern set
    EXPECT_EQ(patterns.getNativePatterns().size(), 3);
    EXPECT_EQ(patterns.getNativePatterns()[0]->getDebugName(), "dummy-rewriter-0");
    EXPECT_EQ(patterns.getNativePatterns()[1]->getDebugName(), "dummy-rewriter-1");
    EXPECT_EQ(patterns.getNativePatterns()[2]->getDebugName(), "dummy-rewriter-2");
}

TEST_F(DynamicRewriterTest, AddMultipleRewritersFromString) {
    _registry->registerRewriter<DummyRewriter1>("dummy-rewriter-1", _log);
    _registry->registerRewriter<DummyRewriter3>("dummy-rewriter-3", _log);
    _registry->registerRewriter<DummyRewriter7>("dummy-rewriter-7", _log);

    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-1"));
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-3"));
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-7"));

    mlir::RewritePatternSet patterns(&_context);
    std::string rewriterList = "dummy-rewriter-1,dummy-rewriter-3,dummy-rewriter-7";
    bool success = _registry->addRewritersFromString(&_context, _log, rewriterList, patterns);
    EXPECT_TRUE(success);
    // Verify the rewriters are added to pattern set
    EXPECT_EQ(patterns.getNativePatterns().size(), 3);
    EXPECT_EQ(patterns.getNativePatterns()[0]->getDebugName(), "dummy-rewriter-1");
    EXPECT_EQ(patterns.getNativePatterns()[1]->getDebugName(), "dummy-rewriter-3");
    EXPECT_EQ(patterns.getNativePatterns()[2]->getDebugName(), "dummy-rewriter-7");
}

TEST_F(DynamicRewriterTest, MixSetsAndRewriters) {
    registerOptimizeSet1(*_registry, _log);
    registerOptimizeSet3(*_registry, _log);
    _registry->registerRewriter<DummyRewriter10>("dummy-rewriter-10", _log);
    _registry->registerRewriter<DummyRewriter15>("dummy-rewriter-15", _log);

    // Check rewriters from sections and individual entries
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-0"));   // from optimize-set-1
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-1"));   // from optimize-set-1
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-2"));   // from optimize-set-1
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-10"));  // individual
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-6"));   // from optimize-set-3
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-7"));   // from optimize-set-3
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-8"));   // from optimize-set-3
    EXPECT_TRUE(_registry->hasRewriter("dummy-rewriter-15"));  // individual
    // Check section content
    auto sections = _registry->getRegisteredRewriterSets();
    auto it1 = sections.find("optimize-set-1");
    EXPECT_NE(it1, sections.end());
    auto sectionRewriters1 = it1->second;
    EXPECT_EQ(sectionRewriters1.size(), 3);
    EXPECT_NE(std::find(sectionRewriters1.begin(), sectionRewriters1.end(), "dummy-rewriter-0"),
              sectionRewriters1.end());
    EXPECT_NE(std::find(sectionRewriters1.begin(), sectionRewriters1.end(), "dummy-rewriter-1"),
              sectionRewriters1.end());
    EXPECT_NE(std::find(sectionRewriters1.begin(), sectionRewriters1.end(), "dummy-rewriter-2"),
              sectionRewriters1.end());

    auto it3 = sections.find("optimize-set-3");
    EXPECT_NE(it3, sections.end());
    auto sectionRewriters3 = it3->second;
    EXPECT_EQ(sectionRewriters3.size(), 3);
    EXPECT_NE(std::find(sectionRewriters3.begin(), sectionRewriters3.end(), "dummy-rewriter-6"),
              sectionRewriters3.end());
    EXPECT_NE(std::find(sectionRewriters3.begin(), sectionRewriters3.end(), "dummy-rewriter-7"),
              sectionRewriters3.end());
    EXPECT_NE(std::find(sectionRewriters3.begin(), sectionRewriters3.end(), "dummy-rewriter-8"),
              sectionRewriters3.end());
    mlir::RewritePatternSet patterns(&_context);

    std::string mixedList = "optimize-set-1,dummy-rewriter-10,optimize-set-3,dummy-rewriter-15";
    bool success = _registry->addRewritersFromString(&_context, _log, mixedList, patterns);
    EXPECT_TRUE(success);
    // Verify the rewriters are added to pattern set
    EXPECT_EQ(patterns.getNativePatterns().size(), 8);
    EXPECT_EQ(patterns.getNativePatterns()[0]->getDebugName(), "dummy-rewriter-0");
    EXPECT_EQ(patterns.getNativePatterns()[1]->getDebugName(), "dummy-rewriter-1");
    EXPECT_EQ(patterns.getNativePatterns()[2]->getDebugName(), "dummy-rewriter-2");
    EXPECT_EQ(patterns.getNativePatterns()[3]->getDebugName(), "dummy-rewriter-10");
    EXPECT_EQ(patterns.getNativePatterns()[4]->getDebugName(), "dummy-rewriter-6");
    EXPECT_EQ(patterns.getNativePatterns()[5]->getDebugName(), "dummy-rewriter-7");
    EXPECT_EQ(patterns.getNativePatterns()[6]->getDebugName(), "dummy-rewriter-8");
    EXPECT_EQ(patterns.getNativePatterns()[7]->getDebugName(), "dummy-rewriter-15");
}

TEST_F(DynamicRewriterTest, SetContentWithOverlaps) {
    registerOptimizeSet1(*_registry, _log);
    registerOptimizeSet2(*_registry, _log);
    registerOptimizeSet3(*_registry, _log);
    registerOptimizeSet4(*_registry, _log);
    registerOverlappingGroup(*_registry, _log);

    auto sections = _registry->getRegisteredRewriterSets();
    EXPECT_TRUE(sections.find("optimize-set-1") != sections.end());
    EXPECT_TRUE(sections.find("optimize-set-2") != sections.end());
    EXPECT_TRUE(sections.find("optimize-set-3") != sections.end());
    EXPECT_TRUE(sections.find("optimize-set-4") != sections.end());
    EXPECT_TRUE(sections.find("overlapping-group") != sections.end());

    auto it = sections.find("overlapping-group");
    auto overlappingRewriters = it->second;
    EXPECT_EQ(overlappingRewriters.size(), 6);
    EXPECT_NE(std::find(overlappingRewriters.begin(), overlappingRewriters.end(), "dummy-rewriter-0"),
              overlappingRewriters.end());
    EXPECT_NE(std::find(overlappingRewriters.begin(), overlappingRewriters.end(), "dummy-rewriter-3"),
              overlappingRewriters.end());
    EXPECT_NE(std::find(overlappingRewriters.begin(), overlappingRewriters.end(), "dummy-rewriter-6"),
              overlappingRewriters.end());
    EXPECT_NE(std::find(overlappingRewriters.begin(), overlappingRewriters.end(), "dummy-rewriter-9"),
              overlappingRewriters.end());
    EXPECT_NE(std::find(overlappingRewriters.begin(), overlappingRewriters.end(), "dummy-rewriter-12"),
              overlappingRewriters.end());
    EXPECT_NE(std::find(overlappingRewriters.begin(), overlappingRewriters.end(), "dummy-rewriter-13"),
              overlappingRewriters.end());
}

TEST_F(DynamicRewriterTest, InvalidRewriterName) {
    setupTestRegistry(*_registry, _log);
    mlir::RewritePatternSet patterns(&_context);
    bool success = _registry->addRewriter(&_context, _log, "non-existent-rewriter", patterns);

    EXPECT_FALSE(success);
}

TEST_F(DynamicRewriterTest, InvalidSetName) {
    setupTestRegistry(*_registry, _log);
    mlir::RewritePatternSet patterns(&_context);

    bool success = _registry->addRewriterSet(&_context, _log, "non-existent-section", patterns);

    EXPECT_FALSE(success);
}

TEST_F(DynamicRewriterTest, MixValidAndInvalidNamesWithOverlaps) {
    setupTestRegistry(*_registry, _log);
    mlir::RewritePatternSet patterns(&_context);

    std::string mixedList = "optimize-set-1,invalid-rewriter,dummy-rewriter-5,overlapping-group,invalid-section";
    bool success = _registry->addRewritersFromString(&_context, _log, mixedList, patterns);

    EXPECT_FALSE(success);
}

TEST_F(DynamicRewriterTest, EmptyStringInput) {
    setupTestRegistry(*_registry, _log);
    mlir::RewritePatternSet patterns(&_context);

    bool success = _registry->addRewritersFromString(&_context, _log, "", patterns);

    EXPECT_TRUE(success);
}

// Test rewriters that require initialization with function context
// e.g. the number of operations in function

// Run cmd: npuUnitTests --gtest_filter="RewriterInitializeTest.*"

class RewriterRequireInitialize final : public mlir::RewritePattern, public IInitializableRewriter {
public:
    RewriterRequireInitialize(mlir::MLIRContext* ctx, std::shared_ptr<int64_t> valueTracker)
            : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
              _valueTracker(valueTracker) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const override {
        return mlir::failure();
    }

    void initialize(mlir::func::FuncOp funcOp) override {
        _numOfOps = std::distance(funcOp.getBody().front().begin(), funcOp.getBody().front().end());
        *_valueTracker = _numOfOps;
    }

private:
    int64_t _numOfOps = 0;
    std::shared_ptr<int64_t> _valueTracker;
};

class RewriterInitializeTest : public testing::Test {
public:
    void SetUp() override {
        auto registry = vpux::createDialectRegistry();
        vpux::registerDynamicRewriterExecutorPass();
        auto interfacesRegistry = vpux::createInterfacesRegistry(config::ArchKind::NPU40XX);
        interfacesRegistry->registerInterfaces(registry);

        _context.loadDialect<Const::ConstDialect>();
        _context.loadDialect<mlir::func::FuncDialect>();
        _context.appendDialectRegistry(registry);
        _valueTracker = std::make_shared<int64_t>(-1);
    }

    void TearDown() override {
        _valueTracker.reset();
    }

protected:
    mlir::MLIRContext _context;
    std::shared_ptr<int64_t> _valueTracker;
};

TEST_F(RewriterInitializeTest, InitializeRewriter) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() -> tensor<1xf32> {
                %cst0 = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
                return %cst0 : tensor<1xf32>
            }
        }
    )";

    auto log = vpux::Logger("RewriterInitializeTest", vpux::LogLevel::Trace);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &_context);
    ASSERT_TRUE(module.get() != nullptr);

    // Register rewriter
    auto& rewriterRegistry = RegistryManager::getGlobalRegistry();
    rewriterRegistry.registerRewriter<RewriterRequireInitialize>("rewriter-require-initialize", _valueTracker);

    // Before running DynamicRewriterExecutorPass
    EXPECT_EQ(*_valueTracker, -1);

    // Run DynamicRewriterExecutorPass
    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::createDynamicRewriterExecutorPass("rewriter-require-initialize", log));
    EXPECT_TRUE(mlir::succeeded(pm.run(module.get())));

    // After running DynamicRewriterExecutorPass
    // Ops in IR: const.Declare / func.return
    EXPECT_EQ(*_valueTracker, 2);
}

}  // namespace
