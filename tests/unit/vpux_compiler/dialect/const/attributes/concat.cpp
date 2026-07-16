//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_ConcatAttrTest : public MLIR_UnitBase {
public:
    mlir::MLIRContext ctx;

public:
    MLIR_ConcatAttrTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }
};

namespace {
template <typename T>
ArrayRef<char> convertArrayRef(ArrayRef<T> typed) {
    return ArrayRef<char>(reinterpret_cast<const char*>(typed.data()), typed.size() * sizeof(T));
}
}  // namespace

// Regression test for smoke_GroupConvolution3D_ExplicitPadding.
// Weights [4,2,3,3] f16 dense<1.0>, groups=2 → SubView+PadWithZero per group → Concat along OC.
// Verifies block-diagonal structure: group 0 at IC 0-1, group 1 at IC 2-3.
TEST_F(MLIR_ConcatAttrTest, ExactFunctest_GroupConv3D_WeightPipeline_F16) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto sliceWeightType = mlir::RankedTensorType::get({4, 2, 3, 3}, f16Type);

    std::vector<vpux::type::float16> sliceWeightVals(72, vpux::type::float16(1.0f));
    const auto base = Const::createExternalConstContent(sliceWeightType, convertArrayRef(ArrayRef(sliceWeightVals)),
                                                        "functest_w");

    const auto svG0Offset = Shape{0, 0, 0, 0};
    const auto svG0Shape = Shape{2, 2, 3, 3};
    const auto padG0Before = Shape{0, 0, 0, 0};
    const auto padG0After = Shape{0, 2, 0, 0};
    auto setupG0 = Const::ContentSetup(base, sliceWeightType);
    setupG0 = setupG0.subview(svG0Offset, svG0Shape);
    setupG0 = setupG0.padWithZero(padG0Before, padG0After);
    auto contentAttrG0 = Const::ContentAttr::get(base, std::move(setupG0));

    const auto svG1Offset = Shape{2, 0, 0, 0};
    const auto svG1Shape = Shape{2, 2, 3, 3};
    const auto padG1Before = Shape{0, 2, 0, 0};
    const auto padG1After = Shape{0, 0, 0, 0};
    auto setupG1 = Const::ContentSetup(base, sliceWeightType);
    setupG1 = setupG1.subview(svG1Offset, svG1Shape);
    setupG1 = setupG1.padWithZero(padG1Before, padG1After);
    auto contentAttrG1 = Const::ContentAttr::get(base, std::move(setupG1));

    std::vector<Const::ContentAttr> inputContents{contentAttrG0, contentAttrG1};

    SmallVector<SmallVector<int64_t>> offsets = {{0, 0, 0, 0}, {2, 0, 0, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    auto contentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);
    const auto content = contentAttr.fold();

    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());

    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 144);

    int nonZeroCount = 0;
    int zeroCount = 0;
    for (int oc = 0; oc < 4; ++oc) {
        for (int ic = 0; ic < 4; ++ic) {
            for (int hw = 0; hw < 9; ++hw) {
                int idx = oc * 36 + ic * 9 + hw;
                float expected;
                if (oc < 2 && ic < 2) {
                    expected = 1.0f;
                    nonZeroCount++;
                } else if (oc >= 2 && ic >= 2) {
                    expected = 1.0f;
                    nonZeroCount++;
                } else {
                    expected = 0.0f;
                    zeroCount++;
                }
                EXPECT_FLOAT_EQ(resultVals[idx], expected) << "Mismatch at OC=" << oc << " IC=" << ic << " hw=" << hw;
            }
        }
    }

    EXPECT_EQ(nonZeroCount, 72);
    EXPECT_EQ(zeroCount, 72);
}

// Full NCE weight pipeline: Concat → PadWithZero OC alignment → SubView back to original OC.
TEST_F(MLIR_ConcatAttrTest, FullNCEWeightPipeline_ConcatPadSubView) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto baseWeightType = mlir::RankedTensorType::get({4, 2, 3, 3}, f16Type);
    std::vector<vpux::type::float16> baseVals(72, vpux::type::float16(1.0f));
    const auto base =
            Const::createExternalConstContent(baseWeightType, convertArrayRef(ArrayRef(baseVals)), "nce_test_w");

    const auto svG0Offset = Shape{0, 0, 0, 0};
    const auto svG0Shape = Shape{2, 2, 3, 3};
    const auto padG0Before = Shape{0, 0, 0, 0};
    const auto padG0After = Shape{0, 2, 0, 0};
    auto setupG0 = Const::ContentSetup(base, baseWeightType);
    setupG0 = setupG0.subview(svG0Offset, svG0Shape);
    setupG0 = setupG0.padWithZero(padG0Before, padG0After);
    auto contentAttrG0 = Const::ContentAttr::get(base, std::move(setupG0));

    const auto svG1Offset = Shape{2, 0, 0, 0};
    const auto svG1Shape = Shape{2, 2, 3, 3};
    const auto padG1Before = Shape{0, 2, 0, 0};
    const auto padG1After = Shape{0, 0, 0, 0};
    auto setupG1 = Const::ContentSetup(base, baseWeightType);
    setupG1 = setupG1.subview(svG1Offset, svG1Shape);
    setupG1 = setupG1.padWithZero(padG1Before, padG1After);
    auto contentAttrG1 = Const::ContentAttr::get(base, std::move(setupG1));

    std::vector<Const::ContentAttr> inputContents{contentAttrG0, contentAttrG1};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0, 0, 0}, {2, 0, 0, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    const auto padFinalBefore = Shape{0, 0, 0, 0};
    const auto padFinalAfter = Shape{12, 0, 0, 0};
    const auto svFinalOffset = Shape{0, 0, 0, 0};
    const auto svFinalShape = Shape{4, 4, 3, 3};
    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    setup = setup.padWithZero(padFinalBefore, padFinalAfter);
    setup = setup.subview(svFinalOffset, svFinalShape);
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));
    const auto content = contentAttr.fold();

    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 144);

    int nonZeroCount = 0;
    int zeroCount = 0;
    for (int oc = 0; oc < 4; ++oc) {
        for (int ic = 0; ic < 4; ++ic) {
            for (int hw = 0; hw < 9; ++hw) {
                int idx = oc * 36 + ic * 9 + hw;
                float expected;
                if (oc < 2 && ic < 2) {
                    expected = 1.0f;
                    nonZeroCount++;
                } else if (oc >= 2 && ic >= 2) {
                    expected = 1.0f;
                    nonZeroCount++;
                } else {
                    expected = 0.0f;
                    zeroCount++;
                }
                EXPECT_FLOAT_EQ(resultVals[idx], expected) << "Mismatch at OC=" << oc << " IC=" << ic << " hw=" << hw;
            }
        }
    }
    EXPECT_EQ(nonZeroCount, 72);
    EXPECT_EQ(zeroCount, 72);
}

// SubView along concat axis selects only the first input.
// Verifies moveSubViewIntoConcat produces a ConcatAttr with one input.
TEST_F(MLIR_ConcatAttrTest, SubViewIntoConcatSameAxisSingleInput) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto inputType = mlir::RankedTensorType::get({2, 4}, f16Type);

    // Two 2x4 inputs, each filled with distinct values
    std::vector<vpux::type::float16> valsA(8, vpux::type::float16(1.0f));
    std::vector<vpux::type::float16> valsB(8, vpux::type::float16(2.0f));
    const auto baseA = Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsA)), "sv_concat_a");
    const auto baseB = Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsB)), "sv_concat_b");

    auto contentAttrA = Const::ContentAttr::get(baseA, Const::ContentSetup(baseA, inputType));
    auto contentAttrB = Const::ContentAttr::get(baseB, Const::ContentSetup(baseB, inputType));

    std::vector<Const::ContentAttr> inputContents{contentAttrA, contentAttrB};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {2, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    // Concat along axis 0 → 4x4 output
    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);

    // SubView [0,0] shape [2,4] → selects only first input
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    SmallVector<int64_t> svOffset1 = {0, 0};
    SmallVector<int64_t> svShape1 = {2, 4};
    setup = setup.subview(ShapeRef(svOffset1), ShapeRef(svShape1));
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));

    // Structural check: optimization should produce a single ConcatAttr (with one input)
    auto transformations = contentAttr.getTransformations();
    ASSERT_EQ(transformations.size(), 1u);
    EXPECT_TRUE(mlir::isa<Const::ConcatAttr>(transformations[0]));

    auto newConcat = mlir::cast<Const::ConcatAttr>(transformations[0]);
    EXPECT_EQ(newConcat.getConstants().size(), 1u);

    // Data correctness: all values should be 1.0 (from first input)
    const auto content = contentAttr.fold();
    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 8u);
    for (size_t i = 0; i < resultVals.size(); ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 1.0f) << "Mismatch at index " << i;
    }
}

// SubView on a non-concat axis (axis 1) slices all inputs equally.
// Verifies moveSubViewIntoConcat pushes SubView into each input.
TEST_F(MLIR_ConcatAttrTest, SubViewIntoConcatDifferentAxis) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto inputType = mlir::RankedTensorType::get({2, 4}, f16Type);

    std::vector<vpux::type::float16> valsA(8, vpux::type::float16(1.0f));
    std::vector<vpux::type::float16> valsB(8, vpux::type::float16(2.0f));
    const auto baseA =
            Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsA)), "sv_concat_diff_a");
    const auto baseB =
            Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsB)), "sv_concat_diff_b");

    auto contentAttrA = Const::ContentAttr::get(baseA, Const::ContentSetup(baseA, inputType));
    auto contentAttrB = Const::ContentAttr::get(baseB, Const::ContentSetup(baseB, inputType));

    std::vector<Const::ContentAttr> inputContents{contentAttrA, contentAttrB};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {2, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    // Concat along axis 0 → 4x4 output
    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);

    // SubView [0,1] shape [4,2] → slices axis 1, keeps full axis 0
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    SmallVector<int64_t> svOffset2 = {0, 1};
    SmallVector<int64_t> svShape2 = {4, 2};
    setup = setup.subview(ShapeRef(svOffset2), ShapeRef(svShape2));
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));

    // Structural check: single ConcatAttr with both inputs preserved
    auto transformations = contentAttr.getTransformations();
    ASSERT_EQ(transformations.size(), 1u);
    EXPECT_TRUE(mlir::isa<Const::ConcatAttr>(transformations[0]));

    auto newConcat = mlir::cast<Const::ConcatAttr>(transformations[0]);
    EXPECT_EQ(newConcat.getConstants().size(), 2u);

    // Data correctness: first 2 rows = 1.0, last 2 rows = 2.0, each row has 2 elements
    const auto content = contentAttr.fold();
    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 8u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 1.0f) << "Mismatch at index " << i;
    }
    for (size_t i = 4; i < 8; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 2.0f) << "Mismatch at index " << i;
    }
}

// SubView along concat axis spans two inputs partially.
// Verifies moveSubViewIntoConcat handles multi-input overlap.
TEST_F(MLIR_ConcatAttrTest, SubViewIntoConcatSpansMultipleInputs) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto inputType = mlir::RankedTensorType::get({2, 4}, f16Type);

    std::vector<vpux::type::float16> valsA(8, vpux::type::float16(1.0f));
    std::vector<vpux::type::float16> valsB(8, vpux::type::float16(2.0f));
    const auto baseA =
            Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsA)), "sv_concat_multi_a");
    const auto baseB =
            Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsB)), "sv_concat_multi_b");

    auto contentAttrA = Const::ContentAttr::get(baseA, Const::ContentSetup(baseA, inputType));
    auto contentAttrB = Const::ContentAttr::get(baseB, Const::ContentSetup(baseB, inputType));

    std::vector<Const::ContentAttr> inputContents{contentAttrA, contentAttrB};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {2, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    // Concat along axis 0 → 4x4 output
    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);

    // SubView [1,0] shape [2,4] → rows 1-2, spanning both inputs
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    SmallVector<int64_t> svOffset3 = {1, 0};
    SmallVector<int64_t> svShape3 = {2, 4};
    setup = setup.subview(ShapeRef(svOffset3), ShapeRef(svShape3));
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));

    // Structural check: ConcatAttr with 2 inputs (partial from each)
    auto transformations = contentAttr.getTransformations();
    ASSERT_EQ(transformations.size(), 1u);
    EXPECT_TRUE(mlir::isa<Const::ConcatAttr>(transformations[0]));

    auto newConcat = mlir::cast<Const::ConcatAttr>(transformations[0]);
    EXPECT_EQ(newConcat.getConstants().size(), 2u);

    // Data correctness: row 1 from input A (1.0), row 0 from input B (2.0)
    const auto content = contentAttr.fold();
    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 8u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 1.0f) << "Mismatch at index " << i;
    }
    for (size_t i = 4; i < 8; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 2.0f) << "Mismatch at index " << i;
    }
}

// 3D concat of 3 constants along axis 0, SubView selects N values from first, all of second, M values from third.
// Verifies moveSubViewIntoConcat with higher-rank tensors and 3 inputs.
TEST_F(MLIR_ConcatAttrTest, SubViewIntoConcat3DThreeInputs) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    // Three inputs: [2,3,4], [3,3,4], [2,3,4] concatenated along axis 0 → [7,3,4]
    const auto typeA = mlir::RankedTensorType::get({2, 3, 4}, f16Type);
    const auto typeB = mlir::RankedTensorType::get({3, 3, 4}, f16Type);
    const auto typeC = mlir::RankedTensorType::get({2, 3, 4}, f16Type);

    std::vector<vpux::type::float16> valsA(24, vpux::type::float16(1.0f));
    std::vector<vpux::type::float16> valsB(36, vpux::type::float16(2.0f));
    std::vector<vpux::type::float16> valsC(24, vpux::type::float16(3.0f));

    const auto baseA = Const::createExternalConstContent(typeA, convertArrayRef(ArrayRef(valsA)), "sv_3d_a");
    const auto baseB = Const::createExternalConstContent(typeB, convertArrayRef(ArrayRef(valsB)), "sv_3d_b");
    const auto baseC = Const::createExternalConstContent(typeC, convertArrayRef(ArrayRef(valsC)), "sv_3d_c");

    auto contentAttrA = Const::ContentAttr::get(baseA, Const::ContentSetup(baseA, typeA));
    auto contentAttrB = Const::ContentAttr::get(baseB, Const::ContentSetup(baseB, typeB));
    auto contentAttrC = Const::ContentAttr::get(baseC, Const::ContentSetup(baseC, typeC));

    std::vector<Const::ContentAttr> inputContents{contentAttrA, contentAttrB, contentAttrC};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0, 0}, {2, 0, 0}, {5, 0, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);

    // SubView [1,0,0] shape [5,3,4]: takes 1 row from A, all 3 from B, 1 row from C
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    SmallVector<int64_t> svOffset = {1, 0, 0};
    SmallVector<int64_t> svShape = {5, 3, 4};
    setup = setup.subview(ShapeRef(svOffset), ShapeRef(svShape));
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));

    // Structural check: ConcatAttr with 3 inputs (partial A, full B, partial C)
    auto transformations = contentAttr.getTransformations();
    ASSERT_EQ(transformations.size(), 1u);
    EXPECT_TRUE(mlir::isa<Const::ConcatAttr>(transformations[0]));
    auto newConcat = mlir::cast<Const::ConcatAttr>(transformations[0]);
    EXPECT_EQ(newConcat.getConstants().size(), 3u);

    // Data correctness: 1*12 values of 1.0, 3*12 values of 2.0, 1*12 values of 3.0
    const auto content = contentAttr.fold();
    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 60u);  // 5*3*4 = 60
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 1.0f) << "Mismatch at index " << i;
    }
    for (size_t i = 12; i < 48; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 2.0f) << "Mismatch at index " << i;
    }
    for (size_t i = 48; i < 60; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 3.0f) << "Mismatch at index " << i;
    }
}

// SubView with shape=1 along a non-concat dimension (rank-reducing slice).
// Verifies the intersection correctly excludes inputs with no overlap in ALL dimensions.
TEST_F(MLIR_ConcatAttrTest, SubViewIntoConcatRankReducingSlice) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    // Two inputs [2,4] concatenated along axis 0 → [4,4]
    const auto inputType = mlir::RankedTensorType::get({2, 4}, f16Type);

    // Input A: values 1..8, Input B: values 11..18
    std::vector<vpux::type::float16> valsA;
    for (int i = 1; i <= 8; ++i) {
        valsA.push_back(vpux::type::float16(static_cast<float>(i)));
    }
    std::vector<vpux::type::float16> valsB;
    for (int i = 11; i <= 18; ++i) {
        valsB.push_back(vpux::type::float16(static_cast<float>(i)));
    }

    const auto baseA = Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsA)), "sv_rank_red_a");
    const auto baseB = Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsB)), "sv_rank_red_b");

    auto contentAttrA = Const::ContentAttr::get(baseA, Const::ContentSetup(baseA, inputType));
    auto contentAttrB = Const::ContentAttr::get(baseB, Const::ContentSetup(baseB, inputType));

    std::vector<Const::ContentAttr> inputContents{contentAttrA, contentAttrB};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {2, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);

    // SubView [1,1] shape [1,2]: a thin slice from row 1, columns 1-2 (entirely within input A)
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    SmallVector<int64_t> svOffset = {1, 1};
    SmallVector<int64_t> svShape = {1, 2};
    setup = setup.subview(ShapeRef(svOffset), ShapeRef(svShape));
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));

    // Structural check: ConcatAttr with 1 input (only A overlaps at row 1)
    auto transformations = contentAttr.getTransformations();
    ASSERT_EQ(transformations.size(), 1u);
    EXPECT_TRUE(mlir::isa<Const::ConcatAttr>(transformations[0]));
    auto newConcat = mlir::cast<Const::ConcatAttr>(transformations[0]);
    EXPECT_EQ(newConcat.getConstants().size(), 1u);

    // Data correctness: row 1 of A is [5,6,7,8], columns 1-2 → [6,7]
    const auto content = contentAttr.fold();
    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 2u);
    EXPECT_FLOAT_EQ(resultVals[0], 6.0f);
    EXPECT_FLOAT_EQ(resultVals[1], 7.0f);
}

// SubView with shape=1 along concat axis at the exact boundary between two inputs.
// Verifies only the input at that position is selected.
TEST_F(MLIR_ConcatAttrTest, SubViewIntoConcatSingleRowAtBoundary) {
    const auto f16Type = mlir::Float16Type::get(&ctx);
    const auto inputType = mlir::RankedTensorType::get({2, 3}, f16Type);

    std::vector<vpux::type::float16> valsA(6, vpux::type::float16(1.0f));
    std::vector<vpux::type::float16> valsB(6, vpux::type::float16(2.0f));

    const auto baseA = Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsA)), "sv_boundary_a");
    const auto baseB = Const::createExternalConstContent(inputType, convertArrayRef(ArrayRef(valsB)), "sv_boundary_b");

    auto contentAttrA = Const::ContentAttr::get(baseA, Const::ContentSetup(baseA, inputType));
    auto contentAttrB = Const::ContentAttr::get(baseB, Const::ContentSetup(baseB, inputType));

    std::vector<Const::ContentAttr> inputContents{contentAttrA, contentAttrB};
    SmallVector<SmallVector<int64_t>> offsets = {{0, 0}, {2, 0}};
    auto staticOffsets = getIntArrayOfArray(&ctx, ArrayRef(offsets));

    // Concat along axis 0 → [4,3]
    auto concatContentAttr = Const::createConcatContentAttr(&ctx, staticOffsets, /*axis=*/0, inputContents);

    // SubView [2,0] shape [1,3]: first row of input B (exactly at boundary)
    auto setup = Const::ContentSetup(concatContentAttr.getBaseContent(), concatContentAttr.getBaseContent().getType(),
                                     concatContentAttr.getTransformations());
    SmallVector<int64_t> svOffset = {2, 0};
    SmallVector<int64_t> svShape = {1, 3};
    setup = setup.subview(ShapeRef(svOffset), ShapeRef(svShape));
    auto contentAttr = Const::ContentAttr::get(concatContentAttr.getBaseContent(), std::move(setup));

    // Structural check: only input B selected
    auto transformations = contentAttr.getTransformations();
    ASSERT_EQ(transformations.size(), 1u);
    auto newConcat = mlir::cast<Const::ConcatAttr>(transformations[0]);
    EXPECT_EQ(newConcat.getConstants().size(), 1u);

    // Data correctness: all 2.0
    const auto content = contentAttr.fold();
    const auto resultVals = content.getValues<float>();
    ASSERT_EQ(resultVals.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(resultVals[i], 2.0f) << "Mismatch at index " << i;
    }
}
