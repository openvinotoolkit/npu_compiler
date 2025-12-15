//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"

#include <gtest/gtest.h>
#include <mlir/Dialect/Quant/IR/Quant.h>

using namespace vpux;

using PermuteInferParams =
        std::tuple<vpux::DimsOrder::StorageType, vpux::DimsOrder::StorageType, vpux::DimsOrder::StorageType, int32_t>;
using MLIR_IE_PermuteInfer = testing::TestWithParam<PermuteInferParams>;

Const::DeclareOp buildConstant(mlir::MLIRContext& ctx, const ShapeRef shape, const int32_t axis,
                               const DimsOrder inputOrder) {
    std::vector<uint8_t> content(shape.totalSize(), 1);
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), getUInt8Type(&ctx));
    const auto dataAttr = Const::createConstContent(dataType, mlir::ArrayRef<uint8_t>(content));
    const auto baseContentAttr = Const::ContentAttr::get(dataAttr);
    const int64_t storageTypeMin = std::numeric_limits<uint8_t>::min();
    const int64_t storageTypeMax = std::numeric_limits<uint8_t>::max();
    const auto origAxisDimSize = shape[Dim(axis)];
    const SmallVector<double> scales(origAxisDimSize, 1.0);
    const SmallVector<int64_t> zeroPoints(origAxisDimSize, 8);
    const auto quantType =
            mlir::quant::UniformQuantizedPerAxisType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales,
                                                          zeroPoints, axis, storageTypeMin, storageTypeMax);
    auto contentAttr = baseContentAttr.transform().reorder(inputOrder).castElemType(quantType).get();
    const auto quantDataType = contentAttr.getType();
    mlir::OpBuilder builder(&ctx);
    return builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(&ctx), quantDataType, std::move(contentAttr));
}

TEST_P(MLIR_IE_PermuteInfer, inferPermuteReturnTypeComponents) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::quant::QuantDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    const auto params = GetParam();
    const auto inputOrder = vpux::DimsOrder::fromCode(std::get<0>(params));
    const auto memPerm = vpux::DimsOrder::fromCode(std::get<1>(params)).toAffineMap(&ctx);
    const auto dstOrder = vpux::DimsOrder::fromCode(std::get<2>(params)).toAffineMap(&ctx);
    const int32_t axis = std::get<3>(params);

    const Shape shape = {2, 4, 6, 8};
    auto declareOp = buildConstant(ctx, shape, axis, inputOrder);
    const auto inputVal = declareOp.getOutput();
    const auto origAxisDimSize = shape[Dim(axis)];

    SmallVector<mlir::ShapedTypeComponents> inferredReturnShapes;
    const bool inferMemSpace = false;
    inferPermuteReturnTypeComponents(inputVal, memPerm, dstOrder, inferredReturnShapes, inferMemSpace);
    ASSERT_EQ(inferredReturnShapes.size(), 1);

    const auto outDims = inferredReturnShapes[0].getDims();
    const mlir::Type elemType = inferredReturnShapes[0].getElementType();
    ASSERT_NE(elemType, nullptr);

    const auto perAxisQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    ASSERT_NE(perAxisQuantType, nullptr);

    const auto newAxis = perAxisQuantType.getQuantizedDimension();
    const auto newAxisDimSize = outDims[newAxis];
    ASSERT_EQ(newAxisDimSize, origAxisDimSize);

    declareOp->erase();
}

const std::vector<int32_t> axes = {0, 1, 2, 3};

const std::vector<vpux::DimsOrder::StorageType> layouts = {
        0x1234, 0x1243, 0x1324, 0x1342, 0x1423, 0x1432,
};

INSTANTIATE_TEST_SUITE_P(PermuteCasts, MLIR_IE_PermuteInfer,
                         testing::Combine(testing::ValuesIn(layouts), testing::ValuesIn(layouts),
                                          testing::ValuesIn(layouts), testing::ValuesIn(axes)));
