//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace vpux::VPU {
bool operator==(const TransformationsSplit& x, const TransformationsSplit& y) {
    return x.getContentAttr() == y.getContentAttr();
}
}  // namespace vpux::VPU

using namespace vpux;

struct MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo : MLIR_UnitBase {
public:
    MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    template <typename Pred>
    std::vector<VPU::TransformationsSplit> extractSpecificSplits(mlir::ModuleOp moduleOp, Pred pred) {
        std::vector<VPU::TransformationsSplit> splits;
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            auto moveWorthy = VPU::collectMoveWorthyConstants(log, funcOp, [](Const::DeclareOp constOp) {
                return VPU::isSuitableForWeightlessCompilation(constOp, /*skipViewLikeOnly=*/true);
            });
            std::copy_if(moveWorthy.begin(), moveWorthy.end(), std::back_inserter(splits), pred);
        });

        llvm::sort(splits);  // for VPU::sliceAccordingToMemoryLimit()
        return splits;
    }

    std::vector<VPU::TransformationsSplit> extractSplits(mlir::ModuleOp moduleOp) {
        return extractSpecificSplits(moduleOp, [](const VPU::TransformationsSplit&) {
            return true;
        });
    }

    mlir::MLIRContext ctx;
    Logger log = Logger::global();
};

// Note: comparing transformations splits is complicated, so sort both ranges to
// simplify it
#define ASSERT_EQ_SPLITS(x, y)     \
    std::sort(x.begin(), x.end()); \
    std::sort(y.begin(), y.end()); \
    ASSERT_EQ(x, y)

constexpr llvm::StringLiteral INPUT_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_3: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_4: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<102x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.PadWithZero<[0, 0], [100, 0]>]
        // ov1 = 2 * 2 * f16 + 102 * 2 * f16

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        %ov3_1 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_3> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        %ov3_2 = const.Declare tensor<52x2xf16> = dense_resource<vpux_ow_3> : tensor<2x2xf16>,
            [#const.PadWithZero<[0, 0], [50, 0]>]
        // ov3 = 2 * 2 * f16 + 52 * 2 * f16

        %ov4 = const.Declare tensor<2x2xf64> = dense_resource<vpux_ow_4> : tensor<2x2xf16>,
            [#const.CastElemType<f64>]
        // ov4 = 2 * 2 * f64

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, vpux::Byte(std::numeric_limits<int64_t>::max()));
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, CompletelySlicedInit) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    // Note: Memory limit = 0 means that every constant would be in its own
    // init. The nature of the procedure is such that even though the limit is
    // exceeded, *all* constants that share *the same dense_resource<>* are
    // guaranteed to appear in *the same* init. This is the intended behavior of
    // the algorithm.
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, vpux::Byte(0));
    ASSERT_EQ(allSlices.size(), 4);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, PartialSlicing) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    // Note: allow ov2 and ov4 to be stored together
    const auto justEnoughMemory =
            /* ov2 + ov4 inputs */ vpux::Byte(2 * 2 * sizeof(vpux::type::float16)) +
            vpux::Byte(2 * 2 * sizeof(vpux::type::float16)) +
            /* transformed(ov2) + transformed(ov4) outputs  */ vpux::Byte(2 * 2 * sizeof(vpux::type::float16)) +
            vpux::Byte(2 * 2 * sizeof(double));
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, vpux::Byte(justEnoughMemory));
    ASSERT_EQ(allSlices.size(), 3);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        std::vector<VPU::TransformationsSplit> expected;
        // Note: ov2 and ov4 are assumed to be stored together
        if (resourceName == "vpux_ow_2" || resourceName == "vpux_ow_4") {
            expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
                const auto xName = getResourceName(x.getContentAttr().getBaseContent()).str();
                return xName == "vpux_ow_2" || xName == "vpux_ow_4";
            });
        } else {
            expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
                return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
            });
        }

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_SUBVIEWS = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<1x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 0], [1, 2]>]
        %ov1_1 = const.Declare tensor<1x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[1, 0], [1, 2]>]
        %ov1_2 = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 0], [2, 1]>]
        %ov1_3 = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 1], [2, 1]>]
        // ov1 = 2 * 2 * f16 (subviews do not count)

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit_SubView) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_SUBVIEWS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 output (single) */ tensorSize2x2xf16 +
            /* ov2 output */ tensorSize2x2xf16;
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, justEnoughMemory);
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SlicedInit_SubView) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_SUBVIEWS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 output (single) */ tensorSize2x2xf16 +
            /* ov2 output */ tensorSize2x2xf16;
    // subtract 1 to disallow single init
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, justEnoughMemory - vpux::Byte(1));
    ASSERT_EQ(allSlices.size(), 2);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_REORDERS = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<1x2xf16, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[0, 0], [1, 2]>]
        %ov1_1 = const.Declare tensor<1x2xf16, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[1, 0], [1, 2]>]
        %ov1_2 = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.PadWithZero<[0, 0], [0, 1]>, #const.SubView<[0, 0], [2, 1]>]
        // ov1 with reorder = 2 * 2 * f16 + 2 * 3 * f16 (subviews do not count)

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit_ReorderAndPad) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_REORDERS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 outputs */ tensorSize2x2xf16 + vpux::Byte(2 * 3 * sizeof(vpux::type::float16)) +
            /* ov2 output */ tensorSize2x2xf16;
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, justEnoughMemory);
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SlicedInit_ReorderAndPad) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_REORDERS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 outputs */ tensorSize2x2xf16 + vpux::Byte(2 * 3 * sizeof(vpux::type::float16)) +
            /* ov2 output */ tensorSize2x2xf16;
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, justEnoughMemory - vpux::Byte(1));
    ASSERT_EQ(allSlices.size(), 2);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_SAME_BLOB = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCDEFEFEFEF",
            vpux_ow_2: "0x10000000ABABABABCDCDCDEF"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<2x3xf16> = dense_resource<vpux_ow_1> : tensor<2x3xf16>,
            [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<102x3xf16> = dense_resource<vpux_ow_1> : tensor<2x3xf16>,
            [#const.PadWithZero<[0, 0], [100, 0]>]

        // Note: shape of dense_resource<> differs (same blob, not same base content)
        %ov1_2 = const.Declare tensor<6xf16> = dense_resource<vpux_ow_1> : tensor<6xf16>,
            [#const.Add<2.0>]
        // Note: element type of dense_resource<> differs (same blob, not same base content)
        %ov1_3 = const.Declare tensor<2x3xf16> = dense_resource<vpux_ow_1> : tensor<2x3xi16>,
            [#const.CastElemType<f16>, #const.Add<2.0>]
        // Note: whole tensor type of dense_resource<> differs
        %ov1_4 = const.Declare tensor<6xf32> = dense_resource<vpux_ow_1> : tensor<3xf32>,
            [#const.PadWithZero<[0], [3]>]

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit_SameBlobDifferedResources) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_SAME_BLOB, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    // Note: this test is to see how slicing works for a very special case of
    // the same AsmResourceBlob being used by multiple dense_resource<>
    // attributes, thus the actual slicing threshold is less important here and
    // we can use "big value" and "small value" respectively.
    const auto wellEnoughMemory = vpux::Byte(1000);
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, wellEnoughMemory);
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SlicedInit_SameBlobDifferedResources) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_SAME_BLOB, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    // Note: this test is to see how slicing works for a very special case of
    // the same AsmResourceBlob being used by multiple dense_resource<>
    // attributes, thus the actual slicing threshold is less important here and
    // we can use "big value" and "small value" respectively.
    const auto notEnoughMemory = vpux::Byte(2);
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, notEnoughMemory);
    ASSERT_EQ(allSlices.size(), 2);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

struct MLIR_VPU_WeightsSeparationUtils_Obfuscation : MLIR_UnitBase {
    MLIR_VPU_WeightsSeparationUtils_Obfuscation(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Core::CoreDialect>();
        ctx.loadDialect<Const::ConstDialect>();
        ctx.loadDialect<IE::IEDialect>();
    }

    mlir::MLIRContext ctx;
    Logger log = Logger::global();

    static mlir::Operation* createSlice(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                        ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
        return builder.create<IE::SliceOp>(loc, input, offsets, sizes);
    }

    static mlir::Operation* createConcat(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<mlir::Value> inputs,
                                         int64_t axis) {
        return builder.create<IE::ConcatOp>(loc, inputs, axis);
    }
};

constexpr llvm::StringLiteral INPUT_IR_OBFUSCATION = R"(
module @main {
    func.func private @dummy(%a0: tensor<2xf32>) -> tensor<2xf32> {
        return %a0 : tensor<2xf32>
    }

    func.func @foo(%a0: tensor<1x1x1xf32>, %a1: tensor<1x1x2xf16>, %a2: tensor<4x2x1xsi8>,
                   %extra: tensor<2x3x4xf16>)
                   -> (tensor<1x1x1xsi8>, tensor<1x1x2xf32>, tensor<4x2x1xf16>, tensor<2x3x4xf16>) {
        %0 = IE.Convert(%a0) {dstElemType = si8} : tensor<1x1x1xf32> -> tensor<1x1x1xsi8>
        %1 = IE.Convert(%a1) {dstElemType = f32} : tensor<1x1x2xf16> -> tensor<1x1x2xf32>
        %2 = IE.Convert(%a2) {dstElemType = f16} : tensor<4x2x1xsi8> -> tensor<4x2x1xf16>
        return %0, %1, %2, %extra : tensor<1x1x1xsi8>, tensor<1x1x2xf32>, tensor<4x2x1xf16>, tensor<2x3x4xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Noop) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {}, createSlice);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Noop_SingleIndex) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {0}, createSlice);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Partial) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {1, 2}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 1}, mlir::Float32Type::get(&ctx)),
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            // Note: new input is added as the last argument
            mlir::RankedTensorType::get({12}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Full) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {0, 1, 2, 3}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({64}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Noop) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op, {}, createSlice);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Noop_SingleIndex) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op, {std::vector<size_t>({0})}, createSlice);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Partial_FirstArgUntouched) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op,
                              {std::vector<size_t>({1, 2}), std::vector<size_t>({3})}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 1}, mlir::Float32Type::get(&ctx)),
            // Note: new inputs are appended
            mlir::RankedTensorType::get({12}, getInt8Type(&ctx)),
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Partial_MiddleArgUntouched) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op,
                              {std::vector<size_t>({0, 3}), std::vector<size_t>({2})}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 2}, mlir::Float16Type::get(&ctx)),
            // Note: new inputs are appended
            mlir::RankedTensorType::get({52}, getInt8Type(&ctx)),
            mlir::RankedTensorType::get({4, 2, 1}, getSInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Partial_LastArgUntouched) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op,
                              {std::vector<size_t>({1}), std::vector<size_t>({0, 2})}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            // Note: new inputs are appended
            mlir::RankedTensorType::get({1, 1, 2}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({12}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Partial_TwoArgsUntouched) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op,
                              {std::vector<size_t>({3}), std::vector<size_t>({1})}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 1}, mlir::Float32Type::get(&ctx)),
            mlir::RankedTensorType::get({4, 2, 1}, getSInt8Type(&ctx)),
            // Note: new inputs are appended
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({1, 1, 2}, mlir::Float16Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Full_OneGroup) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op, {std::vector<size_t>({0, 1, 2, 3})},
                              createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({64}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Full_TwoGroups) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op,
                              {std::vector<size_t>({1, 3}), std::vector<size_t>({0, 2})}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({52}, getInt8Type(&ctx)),
            mlir::RankedTensorType::get({12}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputGroups_Full_ThreeGroups) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputGroups(log, appendLoc(op.getLoc(), "test"), op,
                              {std::vector<size_t>({3}), std::vector<size_t>({0, 2}), std::vector<size_t>({1})},
                              createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            mlir::RankedTensorType::get({12}, getInt8Type(&ctx)),
            mlir::RankedTensorType::get({1, 1, 2}, mlir::Float16Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateOutputs_Noop) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateOutputs(log, appendLoc(op.getLoc(), "test"), op, {0}, createConcat);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getResults(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateOutputs_Partial) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateOutputs(log, appendLoc(op.getLoc(), "test"), op, {1, 2}, createConcat);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 1}, getSInt8Type(&ctx)),
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            // Note: new output is added as the last argument
            mlir::RankedTensorType::get({24}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getResults(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateOutputs_Full) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateOutputs(log, appendLoc(op.getLoc(), "test"), op, {0, 1, 2, 3}, createConcat);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({73}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getResults(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

namespace {
// An utility pass to call an analysis and forward its result outside
class GetWsAnalysisResult : public mlir::PassWrapper<GetWsAnalysisResult, vpux::ModulePass> {
    VPU::WeightsSeparationInfo::Options _options;
    std::vector<VPU::TransformationsSplit>& _splits;

public:
    GetWsAnalysisResult(VPU::WeightsSeparationInfo::Options options, std::vector<VPU::TransformationsSplit>& splits)
            : _options(options), _splits(splits) {
    }

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GetWsAnalysisResult);

    ::llvm::StringRef getName() const override {
        return "GetWsAnalysisResult";
    }

    void safeRunOnModule() final {
        auto moduleOp = getOperation();
        VPU::WeightsSeparationInfo::setOptions(moduleOp, _options);
        VPUX_SCOPE_EXIT {
            VPU::WeightsSeparationInfo::removeOptions(moduleOp);
        };

        auto& analysis = getAnalysis<VPU::WeightsSeparationInfo>();
        _splits = analysis.getCollectedSplits();

        // Ensure new analysis object is created every time
        analysis.invalidate();
    }
};
}  // namespace

// Tests VPU::WeightsSeparationInfo analysis behaviour
struct MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo : MLIR_UnitBase {
    MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    std::vector<VPU::TransformationsSplit> collectSplitsFromModule(mlir::ModuleOp moduleOp,
                                                                   VPU::WeightsSeparationInfo::Options options) {
        std::vector<VPU::TransformationsSplit> splits;
        mlir::PassManager pm(moduleOp->getName());
        pm.addPass(std::make_unique<GetWsAnalysisResult>(options, splits));
        VPUX_THROW_UNLESS(mlir::succeeded(pm.run(moduleOp)), "Pass must succeed");
        return splits;
    }

    std::vector<VPU::TransformationsSplit> filterSplitsByName(const std::vector<VPU::TransformationsSplit>& splits,
                                                              llvm::StringRef name) {
        std::vector<VPU::TransformationsSplit> result;
        std::copy_if(splits.begin(), splits.end(), std::back_inserter(result),
                     [&](const VPU::TransformationsSplit& split) {
                         return getResourceName(split.getContentAttr().getBaseContent()) == name;
                     });
        return result;
    }

    mlir::MLIRContext ctx;
    Logger log = Logger::global();
};

constexpr llvm::StringLiteral VIEW_LIKE_AND_NORMAL_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCDEFEFEFEF",
            vpux_ow_3: "0x10000000ABABABABCDCDCDCD2222222233333333",
            vpux_ow_4: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @MainModule {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    func.func private @extra_call(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %wai_ow_1 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>

        %normal_ow_2 = const.Declare tensor<3xf32> = dense_resource<vpux_ow_2> : tensor<3xf32>,
            [#const.Rescale<5.0>]

        %some_ow_3 = const.Declare tensor<8xf16> = dense_resource<vpux_ow_3> : tensor<8xf16>,
            [#const.Add<4.0>]

        %wai_ow_4 = const.Declare tensor<4x2xui8> = dense_resource<vpux_ow_4> : tensor<4x2xui8>

        return %arg0 : tensor<2x2xf16>
    }

    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %normal_ow_1 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>]

        %view_like_ow_2 = const.Declare tensor<1x3xf32> = dense_resource<vpux_ow_2> : tensor<3xf32>,
            [#const.Reshape<[1, 3]>]

        %some_ow_3 = const.Declare tensor<10xf16> = dense_resource<vpux_ow_3> : tensor<8xf16>,
            [#const.PadWithZero<[0], [2]>]

        %view_like_ow_4 = const.Declare tensor<1x4x2xui8> = dense_resource<vpux_ow_4> : tensor<4x2xui8>,
            [#const.Reshape<[1, 4, 2]>]

        %call = func.call @extra_call(%arg0) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %call : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo, DefaultWeightless) {
    auto moduleOp = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_AND_NORMAL_IR, &ctx);
    ASSERT_TRUE(moduleOp.get() != nullptr);

    const auto splits = collectSplitsFromModule(moduleOp.get(), /*options=*/{});
    ASSERT_FALSE(splits.empty());

    // in the current implementation, view-like transformations are ignored, so:
    // 1. there's 1 vpux_ow_1 split
    // 2. there's 1 vpux_ow_2 split
    // 3. there's 2 vpux_ow_3 splits
    // 4. there's 0 vpux_ow_4 splits
    const auto ow1Splits = filterSplitsByName(splits, "vpux_ow_1");
    ASSERT_EQ(ow1Splits.size(), 1);
    ASSERT_EQ(ow1Splits.front().getContentAttr().getTransformations().size(), 1);
    ASSERT_TRUE(mlir::isa<Const::AddAttr>(ow1Splits.front().getContentAttr().getTransformations().front()));

    const auto ow2Splits = filterSplitsByName(splits, "vpux_ow_2");
    ASSERT_EQ(ow2Splits.size(), 1);
    ASSERT_EQ(ow2Splits.front().getContentAttr().getTransformations().size(), 1);
    ASSERT_TRUE(mlir::isa<Const::RescaleAttr>(ow2Splits.front().getContentAttr().getTransformations().front()));

    const auto ow3Splits = filterSplitsByName(splits, "vpux_ow_3");
    ASSERT_EQ(ow3Splits.size(), 2);
    for (const auto& split : ow3Splits) {
        ASSERT_EQ(split.getContentAttr().getTransformations().size(), 1);
        ASSERT_TRUE((mlir::isa<Const::AddAttr, Const::PadWithZeroAttr>(
                split.getContentAttr().getTransformations().front())));
    }

    const auto ow4Splits = filterSplitsByName(splits, "vpux_ow_4");
    ASSERT_EQ(ow4Splits.size(), 0);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo, WeightlessWithViewLikeIncluded) {
    auto moduleOp = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_AND_NORMAL_IR, &ctx);
    ASSERT_TRUE(moduleOp.get() != nullptr);

    VPU::WeightsSeparationInfo::Options options;
    options.weightlessSkipViewLikeOnly = false;
    const auto splits = collectSplitsFromModule(moduleOp.get(), options);
    ASSERT_FALSE(splits.empty());

    // when view-like transformations are not skipped, they are also collected
    // as splits, except when all of the transformations of a weight are
    // view-like. Thus:
    // 1. there's 2 vpux_ow_1 splits
    // 2. there's 2 vpux_ow_2 splits
    // 3. there's 2 vpux_ow_3 splits
    // 4. there's 0 vpux_ow_4 splits (all are view-like)
    const auto ow1Splits = filterSplitsByName(splits, "vpux_ow_1");
    ASSERT_EQ(ow1Splits.size(), 2);
    for (const auto& split : ow1Splits) {
        ASSERT_TRUE(split.getContentAttr().getTransformations().size() == 0 ||
                    split.getContentAttr().getTransformations().size() == 1);
        if (!split.getContentAttr().getTransformations().empty()) {
            ASSERT_TRUE(mlir::isa<Const::AddAttr>(split.getContentAttr().getTransformations().front()));
        }
    }

    const auto ow2Splits = filterSplitsByName(splits, "vpux_ow_2");
    ASSERT_EQ(ow2Splits.size(), 2);
    for (const auto& split : ow2Splits) {
        ASSERT_EQ(split.getContentAttr().getTransformations().size(), 1);
        ASSERT_TRUE((mlir::isa<Const::RescaleAttr, Const::ReshapeAttr>(
                split.getContentAttr().getTransformations().front())));
    }

    const auto ow3Splits = filterSplitsByName(splits, "vpux_ow_3");
    ASSERT_EQ(ow3Splits.size(), 2);
    for (const auto& split : ow3Splits) {
        ASSERT_EQ(split.getContentAttr().getTransformations().size(), 1);
        ASSERT_TRUE((mlir::isa<Const::AddAttr, Const::PadWithZeroAttr>(
                split.getContentAttr().getTransformations().front())));
    }

    const auto ow4Splits = filterSplitsByName(splits, "vpux_ow_4");
    ASSERT_EQ(ow4Splits.size(), 0) << "WeightsSeparationInfo models the logic of init schedule. Thus, if all of "
                                      "weight's transformations are view-like, there's nothing to do in init.";
}

constexpr llvm::StringLiteral VIEW_LIKE_WITH_DUPLICATES_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD00112233"
        }
    }
#-}

module @MainModule {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %view_like_ow_1 = const.Declare tensor<1x2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Reshape<[1, 2, 2]>]
        %wai_ow_1_another_type = const.Declare tensor<2x2xi16> = dense_resource<vpux_ow_1> : tensor<2x2xi16>

        %wai_ow_2 = const.Declare tensor<2x3xf16> = dense_resource<vpux_ow_2> : tensor<2x3xf16>
        %norma_ow_2_another_shape = const.Declare tensor<3x2xf16> = dense_resource<vpux_ow_2> : tensor<3x2xf16>,
            [#const.Add<1.0>]

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo, WeightlessWithViewLikeIncluded_Duplicates) {
    auto moduleOp = mlir::parseSourceString<mlir::ModuleOp>(VIEW_LIKE_WITH_DUPLICATES_IR, &ctx);
    ASSERT_TRUE(moduleOp.get() != nullptr);

    VPU::WeightsSeparationInfo::Options options;
    options.weightlessSkipViewLikeOnly = false;
    const auto splits = collectSplitsFromModule(moduleOp.get(), options);
    ASSERT_FALSE(splits.empty());

    const auto ow1Splits = filterSplitsByName(splits, "vpux_ow_1");
    ASSERT_EQ(ow1Splits.size(), 0) << "All constants are view-like and are thus skipped";

    const auto ow2Splits = filterSplitsByName(splits, "vpux_ow_2");
    ASSERT_EQ(ow2Splits.size(), 2)
            << "At least one constant has non-trivial transformations, so this weight is used by init";
}

constexpr llvm::StringLiteral HOST_INPUT_IR_0 = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_0: "0x1000000001020304",
            vpux_ow_1: "0x100000000506",
            vpux_ow_2: "0x1000000001020607",
            vpux_ow_3: "0x1000000002030405"
        }
    }
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Module0 {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<2xf16>
    }

    func.func private @kernel_func0() -> (tensor<4xf16>, tensor<2xf16>, tensor<4xf16>, tensor<4xf16>) {
        %cst0 = const.Declare tensor<4xf16> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.CastElemType<f16>]
        %cst1 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.CastElemType<f16>]
        %cst2 = const.Declare tensor<4xf16> = dense_resource<vpux_ow_2> : tensor<4xui8>, [#const.CastElemType<f16>]
        %cst3 = const.Declare tensor<4xf16> = dense_resource<vpux_ow_3> : tensor<4xui8>, [#const.CastElemType<f16>, #const.Rescale<5.0>]

        return %cst0, %cst1, %cst2, %cst3: tensor<4xf16>, tensor<2xf16>, tensor<4xf16>, tensor<4xf16>
    }
    func.func private @kernel_func1() -> (tensor<2xf16>, tensor<4xf16>, tensor<4xf16>) {
        %cst0 = const.Declare tensor<4xf16> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.CastElemType<f16>]
        %cst1 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.CastElemType<f16>]
        %cst2 = const.Declare tensor<4xf16> = dense_resource<vpux_ow_3> : tensor<4xui8>, [#const.CastElemType<f16>, #const.Add<1.0>]

        return %cst1, %cst0, %cst2: tensor<2xf16>, tensor<4xf16>, tensor<4xf16>
    }
    func.func private @kernel_func2() -> (tensor<2xf16>) {
        %cst0 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.CastElemType<f16>]

        return %cst0: tensor<2xf16>
    }
    func.func @main() -> tensor<2xf16> {
        %idx = arith.constant 0 : index // hardcoded index to simplify the test
        %0 = scf.index_switch %idx -> tensor<2xf16>
        case 0 {
            %call0:4 = func.call @kernel_func0() : () -> (tensor<4xf16>, tensor<2xf16>, tensor<4xf16>, tensor<4xf16>)
            scf.yield %call0#1 : tensor<2xf16>
        }
        case 1 {
            %call1:3 = func.call @kernel_func1() : () -> (tensor<2xf16>, tensor<4xf16>, tensor<4xf16>)
            scf.yield %call1#0 : tensor<2xf16>
        }
        default {
            %call2 = func.call @kernel_func2() : () -> (tensor<2xf16>)
            scf.yield %call2 : tensor<2xf16>
        }
        return %0: tensor<2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo, HostPipeline) {
    auto moduleOp = mlir::parseSourceString<mlir::ModuleOp>(HOST_INPUT_IR_0, &ctx);
    ASSERT_TRUE(moduleOp.get() != nullptr);

    VPU::WeightsSeparationInfo::Options options;
    options.weightsAnalysisMode = VPU::WeightsSeparationInfo::Options::WeightsAnalysisMode::HostCompile;
    const auto splits = collectSplitsFromModule(moduleOp.get(), options);
    ASSERT_FALSE(splits.empty());

    // In this test we want to check that when the same constant is used in multiple entry points, it is collected only
    // once. Thus, we expect to have 2 splits (1 for vpux_ow_0 and 1 for vpux_ow_1), and not 4.
    const auto ow0Splits = filterSplitsByName(splits, "vpux_ow_0");
    ASSERT_EQ(ow0Splits.size(), 1);
    ASSERT_EQ(ow0Splits.front().getContentAttr().getTransformations().size(), 1);
    ASSERT_TRUE(mlir::isa<Const::CastElemTypeAttr>(ow0Splits.front().getContentAttr().getTransformations().front()));

    const auto ow1Splits = filterSplitsByName(splits, "vpux_ow_1");
    ASSERT_EQ(ow1Splits.size(), 1);
    ASSERT_EQ(ow1Splits.front().getContentAttr().getTransformations().size(), 1);
    ASSERT_TRUE(mlir::isa<Const::CastElemTypeAttr>(ow1Splits.front().getContentAttr().getTransformations().front()));

    // Should be excluded from splits as it is not repeated (used only in kernel_func0).
    const auto ow2Splits = filterSplitsByName(splits, "vpux_ow_2");
    ASSERT_TRUE(ow2Splits.empty());

    // Should be excluded from splits as constants have different transformations across kernel functions.
    const auto ow3Splits = filterSplitsByName(splits, "vpux_ow_3");
    ASSERT_TRUE(ow3Splits.empty());
}

struct MLIR_VPU_HostWeightsExtraction : MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo {
    MLIR_VPU_HostWeightsExtraction(): MLIR_VPU_WeightsSeparationUtils_WeightsSeparationInfo() {
        ctx.loadDialect<mlir::arith::ArithDialect>();
        ctx.loadDialect<mlir::memref::MemRefDialect>();
    }
};

constexpr llvm::StringLiteral HOST_INPUT_IR_1 = R"(
{-#
dialect_resources: {
    builtin: {
        vpux_ow_0: "0x1000000001020304", // [1, 2, 3, 4]
        vpux_ow_1: "0x100000000506"  // [5, 6]
    }
}
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Module0 {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
        DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    func.func private @kernel_func0() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]

        return %cst0, %cst1: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }
    func.func private @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>) {
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]

        return %cst1, %cst0: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
    }
    func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        %call0:2 = func.call @kernel_func0() : () -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
        %call1:2 = func.call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
        return %call0#0, %call1#0: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }
})";

TEST_F(MLIR_VPU_HostWeightsExtraction, ExtractConcatWeightsIR) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(HOST_INPUT_IR_1, &ctx);
    ASSERT_FALSE(module.get() == nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions =
            VPU::InitCompilerOptions(config::Platform::NPU4000, config::CompilationMode::HostCompile);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    VPU::WeightsSeparationInfo::Options options;
    options.weightsAnalysisMode = VPU::WeightsSeparationInfo::Options::WeightsAnalysisMode::HostCompile;
    const auto extractedSplits = collectSplitsFromModule(module.get(), options);
    ASSERT_FALSE(extractedSplits.empty());
    ASSERT_EQ(extractedSplits.size(), 2);

    std::vector<std::vector<char>> extractedConstContents;
    for (auto constOp : extractedSplits) {
        const auto content = constOp.getContentAttr().fold();
        const auto contentType = content.getType();
        const auto byteSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
        std::vector<char> bytes(byteSize);
        content.copyTo(MutableArrayRef<char>(bytes));
        extractedConstContents.push_back(std::move(bytes));
    }
    ASSERT_FALSE(extractedConstContents.empty());

    {
        pm.addPass(VPU::createExtractWeightsPass(log));
        pm.addPass(VPU::createConcatHostConstsPass(log));
        ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));
    }

    auto mainFunc = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_FALSE(mainFunc == nullptr);

    // Check arguments of slice ops in kernel_func0 function, they should be the same as the content of extracted
    // constants which are concatenated together. No need to check it for kernel_func1 since they are the same
    // constants.
    auto kernelFunc0 = module.get().lookupSymbol<mlir::func::FuncOp>("kernel_func0");
    ASSERT_FALSE(kernelFunc0 == nullptr);

    std::vector<std::pair<int64_t, int64_t>> sliceRanges;
    for (auto sliceOp : kernelFunc0.getOps<VPU::SliceOp>()) {
        const auto offsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        const auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
        ASSERT_FALSE(offsets.empty());
        ASSERT_FALSE(sizes.empty());
        sliceRanges.emplace_back(offsets[0], sizes[0]);
    }
    ASSERT_FALSE(sliceRanges.empty());

    auto arithConsts = to_std_vector(mainFunc.getOps<mlir::arith::ConstantOp>());
    ASSERT_FALSE(arithConsts.empty());
    ASSERT_EQ(arithConsts.size(), 1);

    mlir::arith::ConstantOp concatenatedConstOp = nullptr;
    size_t expectedConcatSize = 0;
    for (const auto& bytes : extractedConstContents) {
        expectedConcatSize += bytes.size();
    }

    auto arithConstOp = arithConsts[0];
    auto denseAttr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(arithConstOp.getValue());
    ASSERT_FALSE(denseAttr == nullptr);

    auto type = mlir::dyn_cast<mlir::RankedTensorType>(denseAttr.getType());
    ASSERT_FALSE(type == nullptr || !type.getElementType().isInteger(8));

    const auto numElements = checked_cast<size_t>(denseAttr.getNumElements());
    ASSERT_EQ(numElements, expectedConcatSize);
    concatenatedConstOp = arithConstOp;

    ASSERT_FALSE(concatenatedConstOp == nullptr);

    auto concatenatedDense = mlir::cast<mlir::DenseIntElementsAttr>(concatenatedConstOp.getValue());
    std::vector<char> concatenatedBytes;
    concatenatedBytes.reserve(concatenatedDense.getNumElements());
    for (auto v : concatenatedDense.getValues<mlir::APInt>()) {
        concatenatedBytes.push_back(checked_cast<char>(v.getSExtValue()));
    }

    std::vector<std::vector<char>> slicedContents;
    slicedContents.reserve(sliceRanges.size());
    for (const auto& [offset, size] : sliceRanges) {
        ASSERT_GE(offset, 0);
        ASSERT_GE(size, 0);
        const auto begin = checked_cast<size_t>(offset);
        const auto length = checked_cast<size_t>(size);
        ASSERT_LE(begin + length, concatenatedBytes.size());

        std::vector<char> sliceBytes;
        auto sliceBegin = std::next(concatenatedBytes.begin(), checked_cast<std::ptrdiff_t>(begin));
        auto sliceEnd = std::next(sliceBegin, checked_cast<std::ptrdiff_t>(length));
        sliceBytes.insert(sliceBytes.end(), sliceBegin, sliceEnd);
        slicedContents.push_back(std::move(sliceBytes));
    }

    auto bytesLess = [](const std::vector<char>& lhs, const std::vector<char>& rhs) {
        if (lhs.size() != rhs.size()) {
            return lhs.size() < rhs.size();
        }
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    };

    std::sort(extractedConstContents.begin(), extractedConstContents.end(), bytesLess);
    std::sort(slicedContents.begin(), slicedContents.end(), bytesLess);

    ASSERT_EQ(extractedConstContents.size(), slicedContents.size());
    EXPECT_EQ(extractedConstContents, slicedContents);
}
