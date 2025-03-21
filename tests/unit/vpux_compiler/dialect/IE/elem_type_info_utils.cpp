//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_IE_ElemTypeInfoUtils = testing::Test;

namespace {
std::vector<int64_t> transposeShape(const std::vector<int64_t>& shape, const std::vector<uint32_t>& order) {
    std::vector<int64_t> outShape;
    const auto reorder = [&](const int64_t orderDim) -> int64_t {
        return shape.at(orderDim);
    };
    std::transform(order.cbegin(), order.cend(), std::back_inserter(outShape), reorder);
    return outShape;
}

mlir::quant::UniformQuantizedPerAxisType composeElementType(const std::vector<int64_t>& shape, const int32_t axis,
                                                            mlir::MLIRContext* ctx) {
    const auto dimSize = shape.at(axis);
    const SmallVector<double> scales(dimSize, 1.0);
    const SmallVector<int64_t> zeroPoints(dimSize, 8);
    return mlir::quant::UniformQuantizedPerAxisType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(ctx),
                                                         mlir::Float32Type::get(ctx), scales, zeroPoints, axis, -128,
                                                         127);
}

std::vector<std::vector<uint32_t>> insertDim(const std::vector<uint32_t>& order, const uint32_t rank) {
    std::vector<std::vector<uint32_t>> result;
    // Value is equal to rank - 1 because, for example, for rank 2 permutations are
    // [{0, 1}, {1, 0}] not [{1, 2}, {2, 1}]
    const auto dimVal = rank - 1;
    for (size_t dimPos = 0; dimPos < rank; dimPos++) {
        // Reset the order on each iteration.
        std::vector<uint32_t> newPermute = order;
        const auto insertPos = newPermute.begin() + dimPos;
        newPermute.insert(insertPos, dimVal);
        result.push_back(newPermute);
    }
    return result;
}

std::vector<std::vector<uint32_t>> generateOrders(const size_t rank) {
    std::vector<std::vector<uint32_t>> result;
    if (rank <= 1) {
        result.push_back({0});
        return result;
    }
    // For each permutation ab of {1,2,...,n-1} form n others by inserting the number n in all possible places:
    // Permutation 1. ab: nab, anb, abn
    // Permutation 2, ba: nba, bna, ban

    // 1. Recursively generate permutations for the lesser rank.
    // For rank = 1, permutations = {0}.
    // 2. Iterate over all these permutations and insert rank - 1 there.
    // permutations = {0}, result = [{0, 1}, {1, 0}]
    // 3. Append insertions and return.
    const auto permutations = generateOrders(rank - 1);
    for (const auto& perm : permutations) {
        const auto insertions = insertDim(perm, rank);
        result.insert(result.end(), insertions.begin(), insertions.end());
    }
    return result;
}
};  // namespace

TEST_F(MLIR_IE_ElemTypeInfoUtils, inferElemTypeTranspose) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const std::vector<int64_t> inShapeVec = {2, 16, 4, 8};
    const auto orders = generateOrders(inShapeVec.size());
    for (const auto& order : orders) {
        for (int32_t quantAxis = 0; quantAxis < static_cast<int32_t>(order.size()); quantAxis++) {
            const auto inputElemType = composeElementType(inShapeVec, quantAxis, &ctx);
            const auto map = mlir::AffineMap::getPermutationMap(order, &ctx);
            const auto outType = vpux::IE::inferElemTypeTranspose(map, inputElemType);
            ASSERT_NE(outType, nullptr);
            const auto perAxisQType = outType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
            ASSERT_NE(perAxisQType, nullptr);

            const auto outShapeVec = transposeShape(inShapeVec, order);
            const auto tensorType = mlir::RankedTensorType::get(outShapeVec, perAxisQType);
            const auto ndType = tensorType.dyn_cast_or_null<vpux::NDTypeInterface>();
            ASSERT_NE(ndType, nullptr);
            EXPECT_TRUE(vpux::validateQuantElemType(mlir::UnknownLoc::get(&ctx), ndType).succeeded());
        }
    }
}
