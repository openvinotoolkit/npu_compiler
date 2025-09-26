//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <algorithm>

using namespace vpux;

template <typename T>
std::vector<T> generateValues(size_t n) {
    std::vector<T> vals(n);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<T>(i);
    }

    return vals;
}

struct ConvertMemPermuteToTransposeAndReorderData {
    std::vector<int64_t> inShape;
    vpux::DimsOrder::StorageType baseLayout;
    vpux::DimsOrder::StorageType dstOrder;
    vpux::DimsOrder::StorageType memPerm;
};

class MLIR_VPU_WeightsSeparation : public testing::TestWithParam<ConvertMemPermuteToTransposeAndReorderData> {};

TEST_P(MLIR_VPU_WeightsSeparation, ConvertMemPermuteToTransposeAndReorder) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    const auto params = GetParam();

    const auto inShape = params.inShape;
    const auto baseLayout = vpux::DimsOrder::fromCode(params.baseLayout);
    const auto dstOrder = vpux::DimsOrder::fromCode(params.dstOrder);
    const auto memPerm = vpux::DimsOrder::fromCode(params.memPerm);

    const auto baseType = getTensorType(ShapeRef(inShape), mlir::Float32Type::get(&ctx), baseLayout, nullptr);
    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetupOriginal(baseType);
    Const::ContentSetup baseContentAttrSetupTransformed(baseType);

    // Expected values
    baseContentAttrSetupOriginal = baseContentAttrSetupOriginal.memPermute(dstOrder, memPerm);
    const auto contentOriginal = Const::ContentAttr::get(baseAttr, std::move(baseContentAttrSetupOriginal)).fold();

    // Result values
    {
        const auto memPermuteAttr = Const::MemPermuteAttr::get(mlir::AffineMapAttr::get(dstOrder.toAffineMap(&ctx)),
                                                               mlir::AffineMapAttr::get(memPerm.toAffineMap(&ctx)));

        const auto inType = mlir::cast<NDTypeInterface>(baseType);
        const auto [identityLayout, inMemShape, memPermuteMap, dstOrderMap, outShape] =
                VPU::extractMemPermuteConversionAttributes(inType, memPermuteAttr);

        baseContentAttrSetupTransformed = baseContentAttrSetupTransformed.reshape(ShapeRef(inMemShape.raw()))
                                                  .layoutCast(DimsOrder::fromAffineMap(identityLayout));
        baseContentAttrSetupTransformed =
                baseContentAttrSetupTransformed.transpose(DimsOrder::fromAffineMap(memPermuteMap));

        baseContentAttrSetupTransformed =
                baseContentAttrSetupTransformed.reshape(outShape).layoutCast(DimsOrder::fromAffineMap(dstOrderMap));
    }

    const auto contentTransformed =
            Const::ContentAttr::get(baseAttr, std::move(baseContentAttrSetupTransformed)).fold();

    const auto contentValsOriginal = contentOriginal.getValues<float>();
    const auto contentValsTransformed = contentTransformed.getValues<float>();
    EXPECT_EQ(contentValsOriginal.size(), contentValsTransformed.size());

    for (size_t index = 0; index < vals.size(); index++) {
        EXPECT_EQ(contentValsOriginal[index], contentValsTransformed[index]);
    }
}

std::vector<ConvertMemPermuteToTransposeAndReorderData> parametersVector = {
        {{1, 2, 3, 4}, 0x1234, 0x1234, 0x1234}, {{1, 4, 2, 3}, 0x1234, 0x1234, 0x1234},
        {{1, 3, 4, 2}, 0x1432, 0x1234, 0x1234}, {{1, 4, 2, 3}, 0x1432, 0x1234, 0x1234},
        {{1, 3, 4, 2}, 0x1234, 0x1234, 0x1423}, {{1, 2, 2, 2}, 0x1234, 0x1234, 0x1432},
        {{1, 3, 4, 2}, 0x1234, 0x1234, 0x1243}, {{1, 2, 3, 4}, 0x1234, 0x1234, 0x1432},
        {{1, 2, 2, 2}, 0x1234, 0x1234, 0x1234}, {{1, 2, 2, 2}, 0x1234, 0x1234, 0x1423},
        {{1, 2, 2, 2}, 0x1432, 0x1234, 0x1423}, {{1, 2, 2, 2}, 0x1234, 0x1234, 0x1243},
        {{1, 2, 3, 4}, 0x1234, 0x1432, 0x1234}, {{1, 4, 2, 3}, 0x1234, 0x1432, 0x1234},
        {{1, 3, 4, 2}, 0x1432, 0x1432, 0x1234}, {{1, 3, 4, 2}, 0x1432, 0x1342, 0x1234},
        {{1, 4, 2, 3}, 0x1432, 0x1432, 0x1234}, {{1, 3, 4, 2}, 0x1234, 0x1432, 0x1423},
        {{1, 3, 4, 2}, 0x1234, 0x1432, 0x1243}, {{1, 2, 3, 4}, 0x1234, 0x1432, 0x1432},
        {{1, 2, 2, 2}, 0x1234, 0x1432, 0x1234}, {{1, 2, 2, 2}, 0x1234, 0x1432, 0x1423},
        {{1, 2, 2, 2}, 0x1432, 0x1432, 0x1423}, {{1, 2, 2, 2}, 0x1234, 0x1432, 0x1243},
        {{1, 2, 2, 2}, 0x1234, 0x1432, 0x1432}, {{1, 2, 2, 2}, 0x1342, 0x1342, 0x1342},
        {{1, 2, 3, 4}, 0x1234, 0x1432, 0x1234}, {{1, 4, 2, 3}, 0x1234, 0x1432, 0x1234},
        {{1, 4, 2, 3}, 0x1432, 0x1342, 0x1234}, {{1, 2, 3, 4}, 0x1234, 0x1342, 0x1432},
        {{1, 2, 2, 2}, 0x1234, 0x1342, 0x1234}, {{1, 2, 2, 2}, 0x1234, 0x1342, 0x1423},
        {{1, 2, 2, 2}, 0x1432, 0x1342, 0x1423}, {{1, 2, 2, 2}, 0x1234, 0x1342, 0x1243},
        {{1, 2, 2, 2}, 0x1234, 0x1342, 0x1432}, {{1, 3, 4, 2}, 0x1432, 0x1423, 0x1234},
        {{1, 4, 2, 3}, 0x1432, 0x1423, 0x1234}, {{1, 2, 3, 4}, 0x1234, 0x1423, 0x1432},
        {{1, 2, 2, 2}, 0x1234, 0x1423, 0x1234}, {{1, 2, 2, 2}, 0x1234, 0x1423, 0x1423},
        {{1, 2, 2, 2}, 0x1432, 0x1423, 0x1423}, {{1, 2, 2, 2}, 0x1234, 0x1423, 0x1243},
        {{1, 2, 2, 2}, 0x1234, 0x1423, 0x1432}, {{1, 2, 2, 2}, 0x1234, 0x1342, 0x1342},
        {{1, 2, 3, 4}, 0x1234, 0x1342, 0x1342}, {{1, 2, 2, 2}, 0x1342, 0x1234, 0x1234}};
//       inShape     baseLayout dstOrder memPerm

INSTANTIATE_TEST_SUITE_P(smoke_WeightsSeparation, MLIR_VPU_WeightsSeparation, testing::ValuesIn(parametersVector));
