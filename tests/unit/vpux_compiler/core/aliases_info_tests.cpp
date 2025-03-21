//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace AlisesInfoTest {

class TestMultiViewOp : public mlir::Op<TestMultiViewOp, vpux::MultiViewOpInterface::Trait> {
public:
    using Op::Op;

    static constexpr StringLiteral getOperationName() {
        return StringLiteral("test.multiview");
    }

    static ArrayRef<StringRef> getAttributeNames() {
        return {};
    }

    mlir::Value getViewSource(ptrdiff_t resIndex) {
        return (*this)->getOperand((*this)->getNumOperands() - (*this)->getNumResults() + resIndex);
    }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::TypeRange resultTypes,
                      mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes = {}) {
        state.addTypes(resultTypes);
        state.addOperands(operands);
        state.addAttributes(attributes);
    }
};

class TestGroupedViewOp : public mlir::Op<TestGroupedViewOp, vpux::GroupedViewOpInterface::Trait> {
public:
    using Op::Op;

    static constexpr StringLiteral getOperationName() {
        return StringLiteral("test.groupedview");
    }

    static ArrayRef<StringRef> getAttributeNames() {
        return {};
    }

    mlir::ValueRange getViewSources() {
        return (*this)->getOperands();
    }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::TypeRange resultTypes,
                      mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes = {}) {
        state.addTypes(resultTypes);
        state.addOperands(operands);
        state.addAttributes(attributes);
    }
};

class TestDialect final : public mlir::Dialect {
public:
    explicit TestDialect(mlir::MLIRContext* ctx)
            : mlir::Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<TestDialect>()) {
        addOperations<TestMultiViewOp>();
        addOperations<TestGroupedViewOp>();
    }

    static constexpr StringLiteral getDialectNamespace() {
        return StringLiteral("test");
    }
};

}  // namespace AlisesInfoTest

TEST(MLIR_AliasesInfo, OpWithNoUsers) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %nouse = memref.alloc(): memref<512xf32>
                return %arg : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    func.walk([&](mlir::memref::AllocOp allocOp) {
        const auto allocRes = allocOp.getResult();

        const auto allocSource = info.getSource(allocRes);
        EXPECT_TRUE(allocSource == nullptr);

        const auto allocRoot = info.getRoot(allocRes);
        EXPECT_TRUE(allocRoot == allocRes);

        const auto& allocAliases = info.getAllAliases(allocRes);
        EXPECT_EQ(allocAliases.size(), 1);
        for (const auto alias : allocAliases) {
            auto* producerOp = alias.getDefiningOp();
            ASSERT_TRUE(producerOp != nullptr);
            EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp));
            EXPECT_TRUE(producerOp == allocOp);
        }
    });
}

TEST(MLIR_AliasesInfo, TestMultiViewOp) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %0 = memref.alloc(): memref<100xf32>
                %1 = memref.subview %arg[0][50][1] : memref<100xf32> to memref<50xf32>
                %2:2 = "test.multiview"(%0, %1) : (memref<100xf32>, memref<50xf32>) -> (memref<100xf32>, memref<50xf32>)
                return %2#0 : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto funcArgSource = info.getSource(funcArg);
    EXPECT_TRUE(funcArgSource == nullptr);

    const auto funcArgRoot = info.getRoot(funcArg);
    EXPECT_TRUE(funcArgRoot == funcArg);

    const auto& funcArgAliases = info.getAllAliases(funcArg);
    EXPECT_EQ(funcArgAliases.size(), 3) << "%arg aliases: %arg, %1, %2#1";

    for (const auto alias : funcArgAliases) {
        if (auto* producerOp = alias.getDefiningOp()) {
            EXPECT_TRUE(mlir::isa<mlir::memref::SubViewOp>(producerOp) ||
                        mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp));

            if (mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp)) {
                EXPECT_EQ(alias.cast<mlir::OpResult>().getResultNumber(), 1);
            }
        } else {
            EXPECT_TRUE(alias == funcArg);
        }
    }

    func.walk([&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto allocRes = allocOp.getResult();

            const auto allocSource = info.getSource(allocRes);
            EXPECT_TRUE(allocSource == nullptr);

            const auto allocRoot = info.getRoot(allocRes);
            EXPECT_TRUE(allocRoot == allocRes);

            const auto& allocAliases = info.getAllAliases(allocRes);
            EXPECT_EQ(allocAliases.size(), 2) << "%0 aliases: %0, %2#0";
            for (const auto alias : allocAliases) {
                auto* producerOp = alias.getDefiningOp();
                ASSERT_TRUE(producerOp != nullptr);

                EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp) ||
                            mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp))
                        << "producerOp = " << producerOp->getName().getStringRef().data();

                if (mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp)) {
                    EXPECT_EQ(alias.cast<mlir::OpResult>().getResultNumber(), 0);
                } else {
                    EXPECT_TRUE(producerOp == allocOp);
                }
            }
        } else if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
            const auto viewRes = viewOp.getResult();

            const auto viewSource = info.getSource(viewRes);
            EXPECT_TRUE(viewSource == viewOp.getViewSource());

            const auto viewRoot = info.getRoot(viewRes);
            EXPECT_TRUE((viewRoot).isa<mlir::BlockArgument>());
        } else if (auto viewOp = mlir::dyn_cast<AlisesInfoTest::TestMultiViewOp>(op)) {
            const auto viewRes0 = viewOp->getResult(0);

            const auto viewSource0 = info.getSource(viewRes0);
            EXPECT_TRUE(viewSource0 == viewOp.getViewSource(0));

            const auto viewRoot0 = info.getRoot(viewRes0);
            EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>((viewRoot0).getDefiningOp()));

            const auto viewRes1 = viewOp->getResult(1);

            const auto viewSource1 = info.getSource(viewRes1);
            EXPECT_TRUE(viewSource1 == viewOp.getViewSource(1));

            const auto viewRoot1 = info.getRoot(viewRes1);
            EXPECT_TRUE((viewRoot1).isa<mlir::BlockArgument>());
        }
    });
}

TEST(MLIR_AliasesInfo, TestGroupedViewOp) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %0 = memref.alloc(): memref<50xf32>
                %1 = "test.groupedview"(%arg, %0) : (memref<100xf32>, memref<50xf32>) -> memref<100xf32>
                return %1 : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto funcArgSource = info.getSource(funcArg);
    EXPECT_TRUE(funcArgSource == nullptr);

    const auto funcArgRoot = info.getRoot(funcArg);
    EXPECT_TRUE(funcArgRoot == funcArg);

    const auto& funcArgAliases = info.getAllAliases(funcArg);
    EXPECT_EQ(funcArgAliases.size(), 2) << "%arg aliases: %arg, %1";

    for (const auto alias : funcArgAliases) {
        if (auto* producerOp = alias.getDefiningOp()) {
            EXPECT_TRUE(mlir::isa<AlisesInfoTest::TestGroupedViewOp>(producerOp));
        } else {
            EXPECT_TRUE(alias == funcArg);
        }
    }

    func.walk([&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto allocRes = allocOp.getResult();

            const auto allocSource = info.getSource(allocRes);
            EXPECT_TRUE(allocSource == nullptr);

            const auto allocRoot = info.getRoot(allocRes);
            EXPECT_TRUE(allocRoot == allocRes);

            const auto& allocAliases = info.getAllAliases(allocRes);
            EXPECT_EQ(allocAliases.size(), 2) << "%0 aliases: %0, %1";
            for (const auto alias : allocAliases) {
                auto* producerOp = alias.getDefiningOp();
                ASSERT_TRUE(producerOp != nullptr);

                EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp) ||
                            mlir::isa<AlisesInfoTest::TestGroupedViewOp>(producerOp))
                        << "producerOp = " << producerOp->getName().getStringRef().data();
            }
        } else if (auto viewOp = mlir::dyn_cast<AlisesInfoTest::TestGroupedViewOp>(op)) {
            const auto viewRes = viewOp->getResult(0);

            const auto viewSources = info.getSources(viewRes);
            EXPECT_EQ(viewSources.size(), 2) << "test.groupedview sources: %arg, %0";
            EXPECT_TRUE(viewSources.size() == viewOp.getViewSources().size());
            for (const auto& source : viewOp.getViewSources()) {
                EXPECT_TRUE(std::find(viewSources.begin(), viewSources.end(), source) != viewSources.end());
            }

            const auto viewRoots = info.getRoots(viewRes);
            EXPECT_EQ(viewRoots.size(), 2) << "test.groupedview roots: %arg, %0";
            for (const auto& root : viewRoots) {
                EXPECT_TRUE(root.isa<mlir::BlockArgument>() || mlir::isa<mlir::memref::AllocOp>(root.getDefiningOp()));
            }
        }
    });
}

TEST(MLIR_AliasesInfo, AsyncRegions) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::async::AsyncDialect>();
    registry.insert<mlir::func::FuncDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<70xf32> {
                %0 = memref.subview %arg[0][90][1] : memref<100xf32> to memref<90xf32>

                %t1, %f1 =
                    async.execute () -> !async.value<memref<80xf32>>
                    {
                        %1 = memref.subview %0[0][80][1] : memref<90xf32> to memref<80xf32>
                        async.yield %1 : memref<80xf32>
                    }

                %t2, %f2 =
                    async.execute [%t1](%f1 as %1 : !async.value<memref<80xf32>>) -> !async.value<memref<70xf32>>
                    {
                        %2 = memref.subview %1[0][70][1] : memref<80xf32> to memref<70xf32>
                        async.yield %2 : memref<70xf32>
                    }

                %2 = async.await %f2 : !async.value<memref<70xf32>>

                return %2 : memref<70xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto& aliases = info.getAllAliases(funcArg);
    EXPECT_EQ(aliases.size(), 8) << "%arg aliases: %arg, %0, %1+%f1, %1+%2+%f2, %2";
}

TEST(MLIR_AliasesInfo, RemoveAlias) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<50xf32> {
                %0 = memref.alloc(): memref<100xf32>
                %1 = memref.subview %0[0][50][1] : memref<100xf32> to memref<50xf32>
                return %1 : memref<50xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    func.walk([&](mlir::Operation* op) {
        if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
            const auto viewRes = viewOp.getResult();

            const auto viewSources = info.getSources(viewRes);
            EXPECT_EQ(viewSources.size(), 1) << "memref.subview sources: %0";
            const auto viewSource = *viewSources.begin();

            const auto viewRoot = info.getRoot(viewRes);
            EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(viewRoot.getDefiningOp()));
            EXPECT_EQ(info.getAllAliases(viewRoot).size(), 2);

            info.removeAlias(viewRes);

            EXPECT_ANY_THROW(info.getRoots(viewRes));
            EXPECT_ANY_THROW(info.getSources(viewRes));
            EXPECT_EQ(info.getAllAliases(viewRoot).size(), 1);

            info.addAlias(viewSource, viewRes);

            EXPECT_EQ(info.getSources(viewRes).size(), 1);
            EXPECT_EQ(*info.getSources(viewRes).begin(), viewSource);

            EXPECT_EQ(info.getRoots(viewRes).size(), 1);

            EXPECT_EQ(info.getAllAliases(viewRoot).size(), 2);
        }
    });
}

TEST(MLIR_AliasesInfo, RemoveAllAlias) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %0 = memref.alloc(): memref<50xf32>
                %1 = "test.groupedview"(%arg, %0) : (memref<100xf32>, memref<50xf32>) -> memref<100xf32>
                return %1 : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    func.walk([&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto allocRes = allocOp.getResult();

            const auto allocSource = info.getSource(allocRes);
            EXPECT_TRUE(allocSource == nullptr);

            const auto allocRoot = info.getRoot(allocRes);
            EXPECT_TRUE(allocRoot == allocRes);

            const auto& allocAliases = info.getAllAliases(allocRes);
            EXPECT_EQ(allocAliases.size(), 2) << "%0 aliases: %0, %1";
            for (const auto alias : allocAliases) {
                auto* producerOp = alias.getDefiningOp();
                ASSERT_TRUE(producerOp != nullptr);

                EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp) ||
                            mlir::isa<AlisesInfoTest::TestGroupedViewOp>(producerOp))
                        << "producerOp = " << producerOp->getName().getStringRef().data();
            }

            info.remove(allocRes);
            info.addAlias(allocRes, allocRes);

            const auto newAllocSource = info.getSource(allocRes);
            EXPECT_TRUE(newAllocSource == nullptr);

            const auto newAllocRoot = info.getRoot(allocRes);
            EXPECT_TRUE(newAllocRoot == allocRes);

            const auto& newAllocAliases = info.getAllAliases(allocRes);
            EXPECT_EQ(newAllocAliases.size(), 1) << "new %0 aliases: %0";
            EXPECT_TRUE(*newAllocAliases.begin() == allocRes);

            auto alias = *newAllocAliases.begin();
            auto* newProducerOp = alias.getDefiningOp();
            ASSERT_TRUE(newProducerOp != nullptr);
            EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(newProducerOp))
                    << "producerOp = " << newProducerOp->getName().getStringRef().data();
        } else if (auto viewOp = mlir::dyn_cast<AlisesInfoTest::TestGroupedViewOp>(op)) {
            const auto viewRes = viewOp->getResult(0);

            const auto viewSources = info.getSources(viewRes);
            EXPECT_EQ(viewSources.size(), 2) << "test.groupedview sources: %arg, %0";
            EXPECT_TRUE(viewSources.size() == viewOp.getViewSources().size());
            for (const auto& source : viewOp.getViewSources()) {
                EXPECT_TRUE(std::find(viewSources.begin(), viewSources.end(), source) != viewSources.end());
            }

            const auto viewRoots = info.getRoots(viewRes);
            EXPECT_EQ(viewRoots.size(), 2) << "test.groupedview roots: %arg, %0";
            for (const auto& root : viewRoots) {
                EXPECT_TRUE(root.isa<mlir::BlockArgument>() || mlir::isa<mlir::memref::AllocOp>(root.getDefiningOp()));
            }

            info.remove(viewRes);
            info.addAlias(viewRes, viewRes);

            const auto newViewSources = info.getSource(viewRes);
            EXPECT_TRUE(newViewSources == nullptr);

            const auto newViewRoot = info.getRoot(viewRes);
            EXPECT_TRUE(newViewRoot == viewRes);

            const auto& newViewAliases = info.getAllAliases(viewRes);
            EXPECT_EQ(newViewAliases.size(), 1) << "new %1 aliases: %1";
            EXPECT_TRUE(*newViewAliases.begin() == viewRes);

            auto alias = *newViewAliases.begin();
            auto* newProducerOp = alias.getDefiningOp();
            ASSERT_TRUE(newProducerOp != nullptr);
            EXPECT_TRUE(mlir::isa<AlisesInfoTest::TestGroupedViewOp>(newProducerOp))
                    << "producerOp = " << newProducerOp->getName().getStringRef().data();
        }
    });
}

TEST(MLIR_AliasesInfo, CallOp) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<vpux::VPUIP::VPUIPDialect>();

    // Here MultiViewOpInterface is registered for func::CallOp.
    // The rationale behind this approach is that:
    // - AliasesInfo is supposed to be used for 'bufferized' dialect
    // - In real pipeline first bufferized dialect is VPUIP
    // - This allows the code from VPUIP to be reused
    // - This allows the func::CallOp to be processed in a unified way and does not change the analysis code
    VPUIP::VPUIPDialect::setupExtraInterfaces(registry);

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @TwoFunctions {
            func.func @foo1(%arg0: memref<1x8x60x60xf16>, %arg1: memref<1x4x60x60xf16>, %arg2: memref<1x2x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
                %0 = memref.subview %arg0[0, 0, 0, 0][1, 4, 60, 60][1, 1, 1, 1] : memref<1x8x60x60xf16> to memref<1x4x60x60xf16>
                memref.copy %0, %arg1 : memref<1x4x60x60xf16> to  memref<1x4x60x60xf16>
                %1 = memref.subview %arg0[0, 0, 0, 0][1, 2, 60, 60][1, 1, 1, 1] : memref<1x8x60x60xf16> to memref<1x2x60x60xf16>
                memref.copy %1, %arg2 : memref<1x2x60x60xf16> to  memref<1x2x60x60xf16>
                return %arg1, %arg2 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
            }

            func.func @foo2(%arg0: memref<1x4x60x60xf16>, %arg1: memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> {
                memref.copy %arg0, %arg1 : memref<1x4x60x60xf16> to  memref<1x4x60x60xf16>
                return %arg1 : memref<1x4x60x60xf16>
            }

            func.func @main(%arg0: memref<1x8x60x60xf16>, %arg1: memref<1x4x60x60xf16>, %arg2: memref<1x2x60x60xf16>,
                            %argExtra: memref<1x4x60x60xf16>)
                    -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
                %alloc = memref.alloc() : memref<1x4x60x60xf16>
                %0:2 = call @foo1(%arg0, %alloc, %arg2)
                    : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
                    -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
                %1 = call @foo2(%0#0, %arg1) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>

                // repeat again with diff. arguments
                %alloc2 = memref.alloc() : memref<1x4x60x60xf16>
                %2:2 = call @foo1(%arg0, %alloc2, %arg2)
                    : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
                    -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
                %3 = call @foo2(%2#0, %argExtra) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>)
                    -> memref<1x4x60x60xf16>

                return %1, %0#1, %3, %2#1 :
                    memref<1x4x60x60xf16>, memref<1x2x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);
    func.walk([&](mlir::func::ReturnOp op) {
        for (const auto& resultValue : op.getOperands()) {
            auto source = info.getRoot(resultValue);

            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(source));
        }
    });
}
