//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include <gtest/gtest.h>

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>

#include "common/utils.hpp"

using namespace vpux;

namespace {

ShapeInfo makeShapeInfo(std::initializer_list<int64_t> shapeVals, std::initializer_list<int64_t> boundVals = {}) {
    return ShapeInfo{SmallVector<int64_t>(shapeVals), SmallVector<int64_t>(boundVals)};
}

using MLIR_InferEltwiseOutputShapeInfoTest = MLIR_UnitBase;

TEST(MLIR_InferEltwiseOutputShapeInfoTest, IdentityNoBroadcast) {
    mlir::MLIRContext ctx;
    const auto loc = mlir::UnknownLoc::get(&ctx);

    const auto lhs = makeShapeInfo({1, 2, 3});
    const auto rhs = makeShapeInfo({1, 2, 3});

    const auto result = inferEltwiseOutputShapeInfo(lhs, rhs, IE::AutoBroadcastType::NONE_OR_EXPLICIT, loc);

    EXPECT_EQ(result.shape, (SmallVector<int64_t>{1, 2, 3}));
    EXPECT_TRUE(result.bounds.empty());
}

TEST(MLIR_InferEltwiseOutputShapeInfoTest, NumpyBroadcastStaticShapes) {
    mlir::MLIRContext ctx;
    const auto loc = mlir::UnknownLoc::get(&ctx);

    const auto lhs = makeShapeInfo({1, 3, 1});
    const auto rhs = makeShapeInfo({4, 1, 5});

    const auto result = inferEltwiseOutputShapeInfo(lhs, rhs, IE::AutoBroadcastType::NUMPY, loc);

    EXPECT_EQ(result.shape, (SmallVector<int64_t>{4, 3, 5}));
    EXPECT_TRUE(result.bounds.empty());
}

TEST(MLIR_InferEltwiseOutputShapeInfoTest, BroadcastFailureReturnsEmptyShapeInfo) {
    mlir::MLIRContext ctx;
    const auto loc = mlir::UnknownLoc::get(&ctx);

    const auto lhs = makeShapeInfo({2, 3});
    const auto rhs = makeShapeInfo({4, 3});

    const auto result = inferEltwiseOutputShapeInfo(lhs, rhs, IE::AutoBroadcastType::NONE_OR_EXPLICIT, loc);

    EXPECT_TRUE(result.shape.empty());
    EXPECT_TRUE(result.bounds.empty());
}

TEST(MLIR_InferEltwiseOutputShapeInfoTest, NumpyBroadcastLargeSpatialTensor) {
    mlir::MLIRContext ctx;
    const auto loc = mlir::UnknownLoc::get(&ctx);

    const auto lhs = makeShapeInfo({1, 1, 1024, 1024});
    const auto rhs = makeShapeInfo({1, 1, 1, 1});

    const auto result = inferEltwiseOutputShapeInfo(lhs, rhs, IE::AutoBroadcastType::NUMPY, loc);

    EXPECT_EQ(result.shape, (SmallVector<int64_t>{1, 1, 1024, 1024}));
    EXPECT_TRUE(result.bounds.empty());
}

}  // namespace
