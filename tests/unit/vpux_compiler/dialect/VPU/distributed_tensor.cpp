//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/utils.hpp"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

using MLIR_NDTypeInterface = MLIR_UnitBase;
using MLIR_ClusterShapeUtils = MLIR_NDTypeInterface;
using MLIR_ClusterShapeUtilsDeathTest = MLIR_NDTypeInterface;

TEST_F(MLIR_NDTypeInterface, SegmentedDistributedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 13 * 16);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({212992_Bit, 16384_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 4 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 4 * 16);

    const SmallVector<int64_t> newShape({1, 32, 52, 8});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    const SmallVector<int64_t> newShape2({1, 32, 26, 16});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    const auto newDimsOrder = DimsOrder::NCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape),
                                            vpux::ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

// 5D case where split dimension is 0 that is Group
TEST_F(MLIR_NDTypeInterface, Segmented5DDistributedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({64, 1, 64, 32, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::GNHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({64, 1, 64, 32, 1}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({64, 1, 32, 1, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 5);
    EXPECT_EQ(ndType.getNumElements(), 64 * 64 * 32 * 1);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::GNHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({32768_Bit, 32768_Bit, 16_Bit, 1024_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({32768_Bit, 32768_Bit, 1024_Bit, 1024_Bit, 16_Bit});

    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    // [type size in bytes] * [G/4] * N * C * H * W
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 16 * 1 * 64 * 32 * 1);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 16 * 1 * 64 * 32 * 1);

    // Change shape
    const SmallVector<int64_t> newShape({32, 1, 64, 32, 1});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), vpux::ShapeRef(newShape));

    // Change element type
    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    // Change shape and element type
    const SmallVector<int64_t> newShape2({32, 1, 64, 32, 1});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    // Change dims order
    const auto newDimsOrder = DimsOrder::GNCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    // Change mem space
    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    // Change strides, expect exception
    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit, 128_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    // Extarct dense tile, slice over H
    const SmallVector<int64_t> tileOffset({0, 0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({64, 1, 32, 32, 1});
    const SmallVector<Bit> tileStrides({16384_Bit, 16384_Bit, 16_Bit, 512_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    // Extarct view tile, expect exception
    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape),
                                            vpux::ShapeRef(tileElemStrides)));

    // Erase tiled info
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2, 1});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST_F(MLIR_NDTypeInterface, Segmented5DDistributedTensorTypeNonDivisible) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({6, 1, 1, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 6);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({64, 1, 64, 32, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::GNHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({64, 1, 64, 32, 1}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({64, 1, 32, 1, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 5);
    EXPECT_EQ(ndType.getNumElements(), 64 * 64 * 32 * 1);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::GNHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({32768_Bit, 32768_Bit, 16_Bit, 1024_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({32768_Bit, 32768_Bit, 1024_Bit, 1024_Bit, 16_Bit});

    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    // [type size in bytes] * [G/6] * N * C * H * W
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 11 * 1 * 64 * 32 * 1);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 11 * 1 * 64 * 32 * 1);

    // Change shape
    const SmallVector<int64_t> newShape({32, 1, 64, 32, 1});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), vpux::ShapeRef(newShape));

    // Change element type
    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    // Change shape and element type
    const SmallVector<int64_t> newShape2({32, 1, 64, 32, 1});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    // Change dims order
    const auto newDimsOrder = DimsOrder::GNCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    // Change mem space
    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    // Change strides, expect exception
    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit, 128_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    // Extarct dense tile, slice over G
    const SmallVector<int64_t> tileOffset({0, 0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 1, 64, 32, 1});
    const SmallVector<Bit> tileStrides({32768_Bit, 32768_Bit, 16_Bit, 1024_Bit, 1024_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    // Extarct view tile, expect exception
    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape),
                                            vpux::ShapeRef(tileElemStrides)));

    // Erase tiled info
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2, 1});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

// Test out combination of SEGMENTED | DUPLICATED mode

TEST_F(MLIR_NDTypeInterface, SegmentedDuplicatedDistributedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 13 * 16);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({212992_Bit, 16384_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 13 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 13 * 16);

    const SmallVector<int64_t> newShape({1, 32, 52, 8});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), vpux::ShapeRef(newShape));

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));
    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    const SmallVector<int64_t> newShape2({1, 32, 32, 32});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 8));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    const auto newDimsOrder = DimsOrder::NCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape),
                                            vpux::ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedUniformDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedDuplicatedDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const Strides expectedStrides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedDuplicatedUniformDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const Strides expectedStrides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedDistribution1x1KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                            getIntAttr(&ctx, 0));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedUniformDistribution1x1KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                            getIntAttr(&ctx, 0));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedDistribution3x3KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 5, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedDistribution3x3KernelStride1EqualMemoryCompute) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

    const auto equalMemoryAndComputeView = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, equalMemoryAndComputeView);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 5, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    for (const auto shapePair : zip(perClusterComputeShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 6, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedMemoryShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedUniformDistribution3x3KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 4, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 6, 0}), Shape({0, 0, 9, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedDistribution3x3KernelStride2) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedUniformDistribution3x3KernelStride2) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 26, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 7, 16}), Shape({1, 64, 7, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 14, 0}), Shape({0, 0, 20, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 8, 16}), Shape({1, 64, 7, 16}), Shape({1, 64, 7, 16}), Shape({1, 64, 7, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 13, 0}), Shape({0, 0, 19, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 7, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {114688_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {114688_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {114688_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedDistributionWithComputeShapesAndOffsets) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 12, 16});
    SmallVector<SmallVector<int64_t>> computeShapes;
    computeShapes.push_back(SmallVector<int64_t>({1, 64, 3, 16}));
    computeShapes.push_back(SmallVector<int64_t>({1, 64, 4, 16}));
    computeShapes.push_back(SmallVector<int64_t>({1, 64, 4, 16}));
    computeShapes.push_back(SmallVector<int64_t>({1, 64, 3, 16}));
    const auto computeShapesAttr = vpux::getIntArrayOfArray(&ctx, computeShapes);

    SmallVector<SmallVector<int64_t>> computeOffsets;
    computeOffsets.push_back(SmallVector<int64_t>({0, 0, 0, 0}));
    computeOffsets.push_back(SmallVector<int64_t>({0, 0, 2, 0}));
    computeOffsets.push_back(SmallVector<int64_t>({0, 0, 5, 0}));
    computeOffsets.push_back(SmallVector<int64_t>({0, 0, 8, 0}));
    const auto computeOffsetsAttr = vpux::getIntArrayOfArray(&ctx, computeOffsets);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            computeShapesAttr, computeOffsetsAttr, computeShapesAttr, computeOffsetsAttr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    SmallVector<Shape> expectedShapes;
    for (auto computeShape : computeShapes) {
        expectedShapes.push_back(Shape(computeShape));
    }
    for (const auto shapePair : zip(perClusterComputeShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    SmallVector<Shape> expectedOffsets;
    for (auto computeOffset : computeOffsets) {
        expectedShapes.push_back(Shape(computeOffset));
    }
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }
    const SmallVector<Strides> expectedStrides({{49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit}});

    const auto perClusterStridedShapes = distributedType.getPerClusterMemoryStridedShapes();

    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();

        EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }
}

// Single axis H alignment, H SEGMENTED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisSegmentedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 9, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 60, 18, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 18 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 18 * 16);
}

// Multiple axis H and K alignment, H SEGMENTED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedMultiAxisSegmentedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 9, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 18, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 18 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 18 * 16);
}

// Single axis H alignment, DUPLICATED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisDuplicatedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 60, 63, 16}), Shape({1, 60, 63, 16}), Shape({1, 60, 63, 16}), Shape({1, 60, 63, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 0, 0}), Shape({0, 0, 0, 0}), Shape({0, 0, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 63, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 60, 63, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
}

// Single axis H alignment, SEGMENTED|DUPLICATED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisSegmentedDuplicatedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 18, 16}), Shape({1, 60, 9, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 60, 63, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 60, 18, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
}

// Multiple axis H and K alignment, SEGMENTED|DUPLICATED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedMultiAxisSegmentedDuplicatedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 18, 16}), Shape({1, 64, 9, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 18, 0}), Shape({0, 0, 36, 0}), Shape({0, 0, 54, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 63, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 18, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 63 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 63 * 16);
}

// Single axis K alignment, SEGMENTED mode, K tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisSegmentedModeKTiling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 110, 59, 16});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 16, 59, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0}), Shape({0, 96, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 32, 59, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
}

// Single axis K alignment, SEGMENTED mode, K tiling, invalid 4 cluster tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisSegmentedModeKTilingInvalid4Clusters) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    EXPECT_ANY_THROW(distributedType.getPerClusterComputeShapes());
    EXPECT_ANY_THROW(distributedType.getPerClusterComputeShapeOffsets());
    EXPECT_ANY_THROW(distributedType.getPerClusterMemoryShapes());
    EXPECT_ANY_THROW(distributedType.getPerClusterMemoryShapeOffsets());
    EXPECT_ANY_THROW(distributedType.getLargestCompactShape());
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_ANY_THROW(distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_ANY_THROW(ndType.getTotalAllocSize().count());
    EXPECT_ANY_THROW(ndType.getCompactAllocSize().count());
}

// Single axis K alignment, SEGMENTED mode, K tiling, valid 3 cluster tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisSegmentedModeKTilingValid3Clusters) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto numClustersAttr = getIntAttr(&ctx, 3);
    const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets({Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 32, 59, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
}

// Single axis K alignment, SEGMENTED|DUPLICATED mode, K tiling, valid 3 cluster tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisSegmentedDuplicatedModeKTilingValid3Clusters) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto numClustersAttr = getIntAttr(&ctx, 3);
    const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16}), Shape({1, 32, 59, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets({Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 96, 59, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 32, 59, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 96 * 59 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 96 * 59 * 16);
}

// Single axis K alignment, OVERLAPPED mode, H tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedSingleAxisOverlappedModeHTiling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto shape = SmallVector<int64_t>({1, 60, 13, 15});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 15}), Shape({1, 64, 4, 15}), Shape({1, 64, 4, 15}), Shape({1, 64, 1, 15})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 15}), Shape({1, 64, 5, 15}), Shape({1, 64, 5, 15}), Shape({1, 64, 2, 15})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 15}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 5 * 15);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 5 * 15);
}

// Single axis W alignment, OVERLAPPED mode, H tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedTensorDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto shape = SmallVector<int64_t>({1, 60, 13, 15});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 1, 16}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 60, 4, 16}), Shape({1, 60, 4, 16}), Shape({1, 60, 4, 16}), Shape({1, 60, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 60, 4, 16}), Shape({1, 60, 5, 16}), Shape({1, 60, 5, 16}), Shape({1, 60, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 60, 4, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 5 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 5 * 16);
}

TEST_F(MLIR_ClusterShapeUtilsDeathTest, AlignedTensorDistribution) {
    testing::GTEST_FLAG(death_test_style) = "threadsafe";
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const mlir::Type elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // Single axis H alignment, OVERLAPPED mode, H tiling, invalid alignment axis same as tiling axis
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto shape = SmallVector<int64_t>({1, 60, 59, 15});
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);
    const Const::OpaqueI64ElementsAttr dynamicDimsMask = nullptr;

    EXPECT_EQ(VPU::DistributedTensorType::getChecked(mlir::UnknownLoc::get(&ctx), &ctx, ArrayRef(shape), elemType,
                                                     dimsOrder, dimsSpace, distributedAttr, dynamicDimsMask),
              nullptr);
}

TEST_F(MLIR_NDTypeInterface, SubByteSegmentedDistributedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    // SI4 quantized type
    const auto elemType = mlir::quant::UniformQuantizedType::getChecked(
            mlir::UnknownLoc::get(&ctx), mlir::quant::QuantizationFlags::Signed, vpux::getSInt4Type(&ctx),
            mlir::Float16Type::get(&ctx), 1.0, 0, -7, 7);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Tensor is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 13 * 16);

    EXPECT_TRUE(mlir::isa<mlir::quant::UniformQuantizedType>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({53248_Bit, 4_Bit, 4096_Bit, 256_Bit});
    const SmallVector<vpux::Bit> memStrides({53248_Bit, 4096_Bit, 256_Bit, 4_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 4);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 64 * 4 * 16 / 2);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 64 * 4 * 16 / 2);

    const SmallVector<int64_t> newShape({1, 32, 52, 8});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElemType.getElementType()));

    const SmallVector<int64_t> newShape2({1, 32, 26, 16});
    const auto changedShapeElemType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape2), mlir::IntegerType::get(&ctx, 4));
    EXPECT_EQ(changedShapeElemType.getShape(), vpux::ShapeRef(newShape2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeElemType.getElementType()));

    const auto newDimsOrder = DimsOrder::NCHW;
    const auto changedDimsOrder = ndType.changeDimsOrder(newDimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), newDimsOrder);

    const auto newMemSpace = vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto changedMemSpace = ndType.changeMemSpace(newMemSpace);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    const SmallVector<Bit> newStrides({106496_Bit, 4_Bit, 4096_Bit, 256_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({20480_Bit, 4_Bit, 1024_Bit, 128_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(vpux::ShapeRef(tileOffset), vpux::ShapeRef(tileShape),
                                            vpux::ShapeRef(tileElemStrides)));
    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}
