//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Run cmd: npuUnitTests --gtest_filter="InferReduceExtraNCETypes.*"

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

class InferReduceExtraNCETypes : public MLIR_UnitBase {
public:
    InferReduceExtraNCETypes(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<VPU::VPUDialect, mlir::tensor::TensorDialect>();
    }

    // Builds an F16 ranked tensor with the given logical NCHW shape and memory layout.
    mlir::Type makeF16Tensor(SmallVector<int64_t> shape, DimsOrder order) {
        return getTensorType(ShapeRef(shape), mlir::Float16Type::get(&ctx), order, /*memSpace=*/nullptr);
    }

    mlir::MLIRContext ctx;
};

}  // namespace

// ---------------------------------------------------------------------------
// No reduce outputs active
// ---------------------------------------------------------------------------

// resultSegmentSizes has only the main output entry — axes are irrelevant.
TEST_F(InferReduceExtraNCETypes, NoReduceOutputs_NoAxes) {
    const auto mainType = makeF16Tensor({1, 64, 8, 8}, DimsOrder::NHWC);
    const SmallVector<int32_t> segSizes = {1};
    SmallVector<mlir::Type> results;

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              /*axes=*/std::nullopt, segSizes, results)));
    EXPECT_TRUE(results.empty());
}

// Axes present but no reduce segment flag set — nothing should be appended.
TEST_F(InferReduceExtraNCETypes, NoReduceOutputs_WithAxes) {
    const auto mainType = makeF16Tensor({1, 64, 8, 8}, DimsOrder::NHWC);
    const SmallVector<int32_t> segSizes = {1};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

// A reduce segment is active but no axes attribute is provided — must fail.
TEST_F(InferReduceExtraNCETypes, ReduceOutputPresent_MissingAxes) {
    const auto mainType = makeF16Tensor({1, 64, 8, 8}, DimsOrder::NHWC);
    // segSizes: {main=1, reduce_xy_max=1, reduce_xy_min=0, tensor_minmax=0}
    const SmallVector<int32_t> segSizes = {1, 1, 0, 0};
    SmallVector<mlir::Type> results;

    EXPECT_TRUE(mlir::failed(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                           /*axes=*/std::nullopt, segSizes, results)));
}

// axes.size() == 2 with rank 4: neither 1 nor equal to rank — must fail.
TEST_F(InferReduceExtraNCETypes, InvalidAxesSize) {
    const auto mainType = makeF16Tensor({1, 64, 8, 8}, DimsOrder::NHWC);
    const SmallVector<int32_t> segSizes = {1, 1, 0, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    EXPECT_TRUE(mlir::failed(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                           b.getI64ArrayAttr({0, 1}), segSizes, results)));
}

// ---------------------------------------------------------------------------
// Single reduce_xy_max — verify shape with tiled main outputs
// ---------------------------------------------------------------------------

// H-tiled partial output [1,64,4,8]: reduce_xy_max must have shape [1,1,4,8].
// axes_value=[1] reduces logical dim 1 (C in the NCHW logical ordering).
TEST_F(InferReduceExtraNCETypes, ReduceMaxXY_HTile) {
    const auto mainType = makeF16Tensor({1, 64, 4, 8}, DimsOrder::NHWC);
    // segSizes: {main=1, reduce_xy_max=1, reduce_xy_min=0, tensor_minmax=0}
    const SmallVector<int32_t> segSizes = {1, 1, 0, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    ASSERT_EQ(results.size(), 1u);
    const auto outType = mlir::cast<NDTypeInterface>(results[0]);
    EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 4, 8}));
}

// C-tiled partial output [1,32,8,8]: reduce_xy_max must have shape [1,1,8,8]
// (the tile only covers half the channels, but the channel dim is reduced to 1).
TEST_F(InferReduceExtraNCETypes, ReduceMaxXY_CTile) {
    const auto mainType = makeF16Tensor({1, 32, 8, 8}, DimsOrder::NHWC);
    const SmallVector<int32_t> segSizes = {1, 1, 0, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    ASSERT_EQ(results.size(), 1u);
    const auto outType = mlir::cast<NDTypeInterface>(results[0]);
    EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 8, 8}));
}

// W-tiled partial output [1,64,8,4]: reduce_xy_max must have shape [1,1,8,4].
TEST_F(InferReduceExtraNCETypes, ReduceMaxXY_WTile) {
    const auto mainType = makeF16Tensor({1, 64, 8, 4}, DimsOrder::NHWC);
    const SmallVector<int32_t> segSizes = {1, 1, 0, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    ASSERT_EQ(results.size(), 1u);
    const auto outType = mlir::cast<NDTypeInterface>(results[0]);
    EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 8, 4}));
}

// ---------------------------------------------------------------------------
// Both reduce_xy_max and reduce_xy_min active simultaneously
// ---------------------------------------------------------------------------

// H-tiled [1,64,4,8] with both xy outputs: two extras, both shaped [1,1,4,8].
TEST_F(InferReduceExtraNCETypes, ReduceMaxAndMinXY_HTile) {
    const auto mainType = makeF16Tensor({1, 64, 4, 8}, DimsOrder::NHWC);
    // segSizes: {main=1, reduce_xy_max=1, reduce_xy_min=1, tensor_minmax=0}
    const SmallVector<int32_t> segSizes = {1, 1, 1, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    ASSERT_EQ(results.size(), 2u);
    for (const auto& t : results) {
        const auto outType = mlir::cast<NDTypeInterface>(t);
        EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 4, 8}));
    }
}

// C-tiled [1,32,8,8] with both xy outputs: two extras, both shaped [1,1,8,8].
TEST_F(InferReduceExtraNCETypes, ReduceMaxAndMinXY_CTile) {
    const auto mainType = makeF16Tensor({1, 32, 8, 8}, DimsOrder::NHWC);
    const SmallVector<int32_t> segSizes = {1, 1, 1, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    ASSERT_EQ(results.size(), 2u);
    for (const auto& t : results) {
        const auto outType = mlir::cast<NDTypeInterface>(t);
        EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 8, 8}));
    }
}

// ---------------------------------------------------------------------------
// Per-tensor reduction (reduce_tensor_min_max, all axes)
// ---------------------------------------------------------------------------

// All axes=[0,1,2,3] reduce the entire shape to [1,1,1,1].
TEST_F(InferReduceExtraNCETypes, TensorMinMax_AllAxes) {
    const auto mainType = makeF16Tensor({1, 64, 8, 8}, DimsOrder::NHWC);
    // segSizes: {main=1, reduce_xy_max=0, reduce_xy_min=0, tensor_minmax=1}
    const SmallVector<int32_t> segSizes = {1, 0, 0, 1};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({0, 1, 2, 3}), segSizes, results)));
    ASSERT_EQ(results.size(), 1u);
    const auto outType = mlir::cast<NDTypeInterface>(results[0]);
    EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 1, 1}));
}

// Element type is preserved in the inferred reduce output.
TEST_F(InferReduceExtraNCETypes, ElementTypePreserved) {
    const auto mainType = getTensorType(ShapeRef({1, 64, 8, 8}), mlir::Float32Type::get(&ctx), DimsOrder::NHWC,
                                        /*memSpace=*/nullptr);
    const SmallVector<int32_t> segSizes = {1, 1, 0, 0};
    SmallVector<mlir::Type> results;
    mlir::Builder b(&ctx);

    ASSERT_TRUE(mlir::succeeded(VPU::inferReduceExtraNCETypes(mlir::UnknownLoc::get(&ctx), mainType,
                                                              b.getI64ArrayAttr({1}), segSizes, results)));
    ASSERT_EQ(results.size(), 1u);
    const auto outType = mlir::cast<NDTypeInterface>(results[0]);
    EXPECT_EQ(outType.getElementType(), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(outType.getShape(), ShapeRef({1, 1, 8, 8}));
}
