//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

#include <functional>

namespace {
auto dummyMemSpace(mlir::MLIRContext* ctx) {
    const auto nameAttr = mlir::FlatSymbolRefAttr::get(ctx, "@DUMMY");
    return vpux::IndexedSymbolAttr::get(ctx, {nameAttr, vpux::getIntAttr(ctx, 0)});
}

auto dummyDistributionInfoAttr(mlir::MLIRContext* ctx) {
    const auto distributionModeAttr = vpux::VPU::DistributionModeAttr::get(ctx, vpux::VPU::DistributionMode::NONE);
    const auto numClustersAttr = vpux::getIntAttr(ctx, 1);
    return vpux::VPU::DistributionInfoAttr::get(ctx, distributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                                numClustersAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr);
}
}  // namespace

using CreateTensorType = std::function<mlir::Type(mlir::MLIRContext*, mlir::AffineMapAttr, vpux::IndexedSymbolAttr)>;

struct MLIR_BufferizeTest : ::testing::TestWithParam<CreateTensorType> {
    mlir::DialectRegistry registry;
    mlir::MLIRContext ctx;
    mlir::Block block;  // used for creation of temporary values

    mlir::Value getTensorValue() {
        const auto orderAttr = mlir::AffineMapAttr::get(vpux::DimsOrder::NHWC.toAffineMap(&ctx));
        const auto memSpace = dummyMemSpace(&ctx);

        const auto tensorType = GetParam()(&ctx, orderAttr, memSpace);
        return block.addArgument(tensorType, mlir::UnknownLoc::get(&ctx));
    }

    MLIR_BufferizeTest() {
        registry = vpux::createDialectRegistry();
        ctx.appendDialectRegistry(registry);  // required for NDTypeInterface attachments
        ctx.loadDialect<vpux::Const::ConstDialect>();
        ctx.loadDialect<mlir::bufferization::BufferizationDialect>();

        ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();
    }
};

TEST_P(MLIR_BufferizeTest, getBufferType) {
    const auto options = vpux::getOneShotBufferizationOptions();
    const auto tensorValue = getTensorValue();
    const auto tensorNdType = mlir::cast<vpux::NDTypeInterface>(tensorValue.getType());

    const auto memref = vpux::getBufferType(tensorValue);

    ASSERT_EQ(memref.getShape(), tensorNdType.getShape());
    ASSERT_EQ(memref.getElementType(), tensorNdType.getElementType());
    ASSERT_EQ(memref.hasRank(), tensorNdType.hasRank());
    const bool isRankedTensor = mlir::isa<mlir::RankedTensorType>(tensorNdType);
    if (isRankedTensor) {
        ASSERT_EQ(memref.getRank(), tensorNdType.getRank());
        ASSERT_EQ(memref.getMemSpace(), tensorNdType.getMemSpace());
        ASSERT_EQ(memref.getDimsOrder(), tensorNdType.getDimsOrder());
    }
}

TEST_P(MLIR_BufferizeTest, getBuffer) {
    const auto options = vpux::getOneShotBufferizationOptions();
    const auto tensorValue = getTensorValue();
    const auto tensorNdType = mlir::cast<vpux::NDTypeInterface>(tensorValue.getType());

    mlir::IRRewriter rewriter(&ctx);
    const auto memrefValue = vpux::getBuffer(rewriter, tensorValue);
    const auto memref = mlir::cast<vpux::NDTypeInterface>(memrefValue.getType());

    ASSERT_EQ(memref.getShape(), tensorNdType.getShape());
    ASSERT_EQ(memref.getElementType(), tensorNdType.getElementType());
    ASSERT_EQ(memref.hasRank(), tensorNdType.hasRank());
    const bool isRankedTensor = mlir::isa<mlir::RankedTensorType>(tensorNdType);
    if (isRankedTensor) {
        ASSERT_EQ(memref.getRank(), tensorNdType.getRank());
        ASSERT_EQ(memref.getMemSpace(), tensorNdType.getMemSpace());
        ASSERT_EQ(memref.getDimsOrder(), tensorNdType.getDimsOrder());
    }
}

INSTANTIATE_TEST_SUITE_P(AllTensorTypes, MLIR_BufferizeTest,
                         ::testing::Values(
                                 // RankedTensor
                                 [](mlir::MLIRContext* ctx, mlir::AffineMapAttr orderAttr,
                                    vpux::IndexedSymbolAttr memSpace) -> mlir::Type {
                                     const auto elemType = mlir::Float32Type::get(ctx);
                                     return mlir::RankedTensorType::get(
                                             {1, 2, 3, 4}, elemType, vpux::TensorAttr::get(ctx, orderAttr, memSpace));
                                 },
                                 // UnrankedTensor
                                 [](mlir::MLIRContext* ctx, mlir::AffineMapAttr orderAttr,
                                    vpux::IndexedSymbolAttr memSpace) -> mlir::Type {
                                     const auto elemType = mlir::Float32Type::get(ctx);
                                     // Note: unranked tensor does not allow
                                     // setting a vpux::TensorAttr, so one
                                     // cannot reasonably set memSpace / order
                                     // for it!
                                     std::ignore = orderAttr;
                                     std::ignore = memSpace;
                                     return mlir::UnrankedTensorType::get(elemType);
                                 },
                                 // DistributedTensor
                                 [](mlir::MLIRContext* ctx, mlir::AffineMapAttr orderAttr,
                                    vpux::IndexedSymbolAttr memSpace) -> mlir::Type {
                                     const auto elemType = mlir::Float32Type::get(ctx);
                                     return vpux::VPU::DistributedTensorType::get(ctx, {1, 2, 3, 4}, elemType,
                                                                                  orderAttr, memSpace,
                                                                                  dummyDistributionInfoAttr(ctx));
                                 },
                                 // SparseTensor (with RankedTensor inside)
                                 [](mlir::MLIRContext* ctx, mlir::AffineMapAttr orderAttr,
                                    vpux::IndexedSymbolAttr memSpace) -> mlir::Type {
                                     const auto elemType = mlir::Float32Type::get(ctx);
                                     const auto dataType = mlir::RankedTensorType::get(
                                             {1, 2, 3, 4}, elemType, vpux::TensorAttr::get(ctx, orderAttr, memSpace));

                                     return vpux::VPU::SparseTensorType::get(dataType);
                                 }));
