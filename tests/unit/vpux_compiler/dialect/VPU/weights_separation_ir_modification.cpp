//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/weights_separation_ir_modification.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"

#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

struct MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater : MLIR_UnitBase {
public:
    MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
        ctx.loadDialect<VPU::VPUDialect>();
    }

    mlir::func::FuncOp findSpecificFunction(mlir::StringRef funcName, mlir::ModuleOp moduleOp) {
        mlir::func::FuncOp result;
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            if (funcOp.getSymName() == funcName) {
                result = funcOp;
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
        });
        return result;
    }

    size_t countNumberOfOps(mlir::ModuleOp moduleOp, FuncRef<bool(mlir::Operation*)> pred) {
        size_t count = 0;
        moduleOp.walk([&](mlir::Operation* op) {
            if (pred(op)) {
                count++;
            }
        });
        return count;
    }

    mlir::MLIRContext ctx;
    Logger log = Logger::global();
};

// Note: this is some representative example of the complex IR that can be
// produced by compiler at the time of weights separation.
constexpr llvm::StringLiteral NORMAL_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @main {
    func.func nested @bar(%arg0: tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16> {
        %ov1_0 = const.Declare tensor<1x1x5x2xf16> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf16>,
            [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 3, 0]>]
        %ov1_1 = const.Declare tensor<1x1x2x5xf16> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf16>,
            [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>]

        %transpose = VPU.MemPermute(%ov1_1) {dst_order = #NCHW, mem_perm = #NCWH}
        : tensor<1x1x2x5xf16> -> tensor<1x1x5x2xf16>
        %add1 = VPU.Add(%ov1_0, %transpose) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x5x2xf16>, tensor<1x1x5x2xf16> -> tensor<1x1x5x2xf16>
        %subview = VPU.Slice %add1 [0, 0, 0, 0] [1, 1, 2, 2] : tensor<1x1x5x2xf16> to tensor<1x1x2x2xf16>
        %add2 = VPU.Add(%arg0, %subview) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>

        return %add2 : tensor<1x1x2x2xf16>
    }

    func.func nested @foo(%arg0: tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16> {
        %ov1_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf16>,
            [#const.Add<1.0>]

        %call = func.call @bar(%arg0) : (tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>
        %add = VPU.Add(%call, %ov1_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>

        return %add : tensor<1x1x2x2xf16>
    }

    func.func @main() -> tensor<1x1x2x2xf16> {
        %ov1_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf16>,
            [#const.Add<1.0>]

        %call1 = func.call @foo(%ov1_0) : (tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>
        %call2 = func.call @foo(%call1) : (tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>

        return %call2 : tensor<1x1x2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater, NormalConstants) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(NORMAL_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBeforeOfMain = mainFunc.getNumArguments();

    auto fooFunc = findSpecificFunction("foo", module.get());
    ASSERT_NE(fooFunc, nullptr);
    const auto argCountBeforeOfFoo = fooFunc.getNumArguments();

    auto barFunc = findSpecificFunction("bar", module.get());
    ASSERT_NE(barFunc, nullptr);
    const auto argCountBeforeOfBar = barFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/true);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";
    const auto constCount = countNumberOfOps(*module, [](mlir::Operation* op) {
        return mlir::isa<Const::DeclareOp>(op);
    });
    ASSERT_EQ(constCount, 0) << "All constants must have been removed from the module";

    const mlir::Type expectedTypesOfMain[] = {
            mlir::RankedTensorType::get({1, 1, 2, 2}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({1, 1, 5, 2}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({1, 1, 2, 5}, mlir::Float16Type::get(&ctx)),
    };
    ASSERT_EQ(mainFunc.getNumArguments(), argCountBeforeOfMain + 3);
    for (size_t i = 0; i < 3; i++) {
        auto argType = mainFunc.getArgumentTypes()[argCountBeforeOfMain + i];
        ASSERT_EQ(argType, expectedTypesOfMain[i]) << "Argument #" << i << " should match the expected type";
    }

    const auto expectedTypesOfFoo = ArrayRef(expectedTypesOfMain);
    ASSERT_EQ(fooFunc.getNumArguments(), argCountBeforeOfFoo + 3);
    for (size_t i = 0; i < 3; i++) {
        auto argType = fooFunc.getArgumentTypes()[argCountBeforeOfFoo + i];
        ASSERT_EQ(argType, expectedTypesOfFoo[i]) << "Argument #" << i << " should match the expected type";
    }

    const mlir::Type expectedTypesOfBar[] = {
            expectedTypesOfMain[1],
            expectedTypesOfMain[2],
    };
    ASSERT_EQ(barFunc.getNumArguments(), argCountBeforeOfBar + 2);
    for (size_t i = 0; i < 2; i++) {
        auto argType = barFunc.getArgumentTypes()[argCountBeforeOfBar + i];
        ASSERT_EQ(argType, expectedTypesOfBar[i]) << "Argument #" << i << " should match the expected type";
    }
}

// Note: this is a special case of IR with view-like-only transformations for
// both normal weights and duplicates
constexpr llvm::StringLiteral VIEW_LIKES_NORMAL_AND_DUPLICATES_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD00112233"
        }
    }
#-}

#NC = affine_map<(d0, d1) -> (d0, d1)>

module @main {
    func.func @main() -> tensor<1x1x1x1xf16> {
        %normal_ov1_0 = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Reshape<[1, 1, 2, 2]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
        %normal_ov1_1 = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Reshape<[2, 2, 1, 1]>, #const.SubView<[1, 1, 0, 0], [1, 1, 1, 1]>]
        %add1 = VPU.Add(%normal_ov1_0, %normal_ov1_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

        %duplicate_ov2_0 = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_2> : tensor<2x3xf16>,
            [#const.Reshape<[1, 1, 2, 3]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
        %duplicate_ov2_1 = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_2> : tensor<3x2xf16>,
            [#const.Reshape<[1, 1, 3, 2]>, #const.SubView<[0, 0, 1, 1], [1, 1, 1, 1]>]
        %add2 = VPU.Add(%duplicate_ov2_0, %duplicate_ov2_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

        %out = VPU.Add(%add1, %add2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
        return %out : tensor<1x1x1x1xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater, ViewLikeConstants_SkipViewLike) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKES_NORMAL_AND_DUPLICATES_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBefore = mainFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/true);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";

    const auto argCountAfter = mainFunc.getNumArguments();
    ASSERT_EQ(argCountAfter, argCountBefore) << "No new arguments due to view-like being skipped";
}

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater, ViewLikeConstants_NoSkipViewLike) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKES_NORMAL_AND_DUPLICATES_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBefore = mainFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/false);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";
    const auto constCount = countNumberOfOps(*module, [](mlir::Operation* op) {
        return mlir::isa<Const::DeclareOp>(op);
    });
    ASSERT_EQ(constCount, 0) << "All constants must have been removed from the module";

    const auto argCountAfter = mainFunc.getNumArguments();
    ASSERT_EQ(argCountAfter, argCountBefore + 3)
            << "View-like constants are not skipped, so must be converted to arguments";

    // Note: from the tensor type, it is visible that these weights are
    // *original* (coming from original ov::Model) as they have no
    // transformations on them, yet are passed to main().
    const mlir::Type expectedTypes[] = {
            mlir::RankedTensorType::get({2, 2}, mlir::Float16Type::get(&ctx)),
            // Note: duplicates count as separate weights
            mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({3, 2}, mlir::Float16Type::get(&ctx)),
    };
    for (size_t i = 0; i < 3; i++) {
        auto argType = mainFunc.getArgumentTypes()[argCountBefore + i];
        ASSERT_EQ(argType, expectedTypes[i]) << "Argument #" << i << " should match the expected type";
    }
}

constexpr llvm::StringLiteral VIEW_LIKE_AND_NON_VIEW_LIKE_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD00112233"
        }
    }
#-}

#NC = affine_map<(d0, d1) -> (d0, d1)>

module @main {
    func.func nested @foo() -> tensor<1x2x3x1xf16> {
        %compute_ov1 = const.Declare tensor<1x2x3x1xf16> = dense_resource<vpux_ow_1> : tensor<2x3xf16>,
            [#const.Add<42.0>, #const.Reshape<[1, 2, 3, 1]>]
        return %compute_ov1 : tensor<1x2x3x1xf16>
    }

    func.func @main() -> tensor<1x1x1x1xf16> {
        %view_like_ov1 = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_1> : tensor<2x3xf16>,
            [#const.Reshape<[1, 2, 3, 1]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]

        %call = func.call @foo() : () -> tensor<1x2x3x1xf16>
        %slice = VPU.Slice %call [0, 1, 2, 0] [1, 1, 1, 1] : tensor<1x2x3x1xf16> to tensor<1x1x1x1xf16>

        %out = VPU.Add(%slice, %view_like_ov1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
        return %out : tensor<1x1x1x1xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater, ViewLikeAndNonViewLikeConstant_SkipViewLike) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_AND_NON_VIEW_LIKE_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBefore = mainFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/true);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";
    const auto constCount = countNumberOfOps(*module, [](mlir::Operation* op) {
        return mlir::isa<Const::DeclareOp>(op);
    });
    ASSERT_EQ(constCount, 1) << "Constant with view-like transformations stays in module";

    const auto argCountAfter = mainFunc.getNumArguments();
    ASSERT_EQ(argCountAfter, argCountBefore + 1) << "Only non-view-like transformations are extracted";

    const mlir::Type expectedTypes[] = {
            mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx)),
    };
    ASSERT_EQ(mainFunc.getArgumentTypes()[argCountBefore], expectedTypes[0])
            << "The only new argument must match the expected type";
}

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater, ViewLikeAndNonViewLikeConstant_NoSkipViewLike) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_AND_NON_VIEW_LIKE_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBefore = mainFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/false);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";
    const auto constCount = countNumberOfOps(*module, [](mlir::Operation* op) {
        return mlir::isa<Const::DeclareOp>(op);
    });
    ASSERT_EQ(constCount, 0) << "All constants must have been removed from the module";

    const auto argCountAfter = mainFunc.getNumArguments();
    ASSERT_EQ(argCountAfter, argCountBefore + 2);

    // Note: 2 separate entries because 1 weight is being transformed
    // (#const.Add<42>) but the other is not.
    const mlir::Type expectedTypes[] = {
            mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx)),
    };
    for (size_t i = 0; i < 2; i++) {
        auto argType = mainFunc.getArgumentTypes()[argCountBefore + i];
        ASSERT_EQ(argType, expectedTypes[i]) << "Argument #" << i << " should match the expected type";
    }
}

// Note: this is a special case of IR with the same weight having only view-like
// transformations in one case, and compute transformations in another case, on
// top of this, weight duplicate is used
constexpr llvm::StringLiteral VIEW_LIKE_AND_NON_VIEW_LIKE_DUPLICATES_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD00112233"
        }
    }
#-}

#NC = affine_map<(d0, d1) -> (d0, d1)>

module @main {
    func.func nested @foo() -> tensor<1x2x3x1xf16> {
        %compute_ov1 = const.Declare tensor<1x2x3x1xf16> = dense_resource<vpux_ow_1> : tensor<2x3xf16>,
            [#const.Add<42.0>, #const.Reshape<[1, 2, 3, 1]>]
        return %compute_ov1 : tensor<1x2x3x1xf16>
    }

    func.func @main() -> tensor<1x1x1x1xf16> {
        %view_like_ov1 = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_1> : tensor<3x2xf16>,
            [#const.Reshape<[1, 3, 2, 1]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]

        %call = func.call @foo() : () -> tensor<1x2x3x1xf16>
        %slice = VPU.Slice %call [0, 1, 2, 0] [1, 1, 1, 1] : tensor<1x2x3x1xf16> to tensor<1x1x1x1xf16>

        %out = VPU.Add(%slice, %view_like_ov1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
        return %out : tensor<1x1x1x1xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater,
       ViewLikeAndNonViewLikeConstant_Duplicates_SkipViewLike) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_AND_NON_VIEW_LIKE_DUPLICATES_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBefore = mainFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/true);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";
    const auto constCount = countNumberOfOps(*module, [](mlir::Operation* op) {
        return mlir::isa<Const::DeclareOp>(op);
    });
    ASSERT_EQ(constCount, 1) << "Constant with view-like transformations stays in module";

    const auto argCountAfter = mainFunc.getNumArguments();
    ASSERT_EQ(argCountAfter, argCountBefore + 1) << "Only non-view-like transformations are extracted";

    const mlir::Type expectedTypes[] = {
            mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx)),
    };
    ASSERT_EQ(mainFunc.getArgumentTypes()[argCountBefore], expectedTypes[0])
            << "The only new argument must match the expected type";
}

TEST_F(MLIR_VPU_WeightsSeparationIrModification_MainFunctionUpdater,
       ViewLikeAndNonViewLikeConstant_Duplicates_NoSkipViewLike) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_AND_NON_VIEW_LIKE_DUPLICATES_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto mainFunc = findSpecificFunction("main", module.get());
    ASSERT_NE(mainFunc, nullptr);
    const auto argCountBefore = mainFunc.getNumArguments();

    auto tree = VPU::getOutliningRepresentation(mainFunc);
    VPU::MainFunctionUpdater updater(log, module.get(), [](Const::DeclareOp op) {
        return VPU::isSuitableForWeightlessCompilation(op, /*skipViewLikeOnly=*/false);
    });
    tree.apply(updater);

    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module))) << "IR must be valid";
    const auto constCount = countNumberOfOps(*module, [](mlir::Operation* op) {
        return mlir::isa<Const::DeclareOp>(op);
    });
    ASSERT_EQ(constCount, 0) << "All constants must have been removed from the module";

    const auto argCountAfter = mainFunc.getNumArguments();
    ASSERT_EQ(argCountAfter, argCountBefore + 2);

    const mlir::Type expectedTypes[] = {
            mlir::RankedTensorType::get({3, 2}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({2, 3}, mlir::Float16Type::get(&ctx)),
    };
    for (size_t i = 0; i < 2; i++) {
        auto argType = mainFunc.getArgumentTypes()[argCountBefore + i];
        ASSERT_EQ(argType, expectedTypes[i]) << "Argument #" << i << " should match the expected type";
    }
}
