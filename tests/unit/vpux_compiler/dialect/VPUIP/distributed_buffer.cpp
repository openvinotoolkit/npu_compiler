//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
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

TEST_F(MLIR_NDTypeInterface, SegmentedDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 13);

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
    const auto chnagedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(chnagedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElementType.getElementType()));

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedShapeAndElementType.getElementType()));

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);
    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> newStrides2({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({638976_Bit, 16_Bit, 32768_Bit, 1024_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST_F(MLIR_NDTypeInterface, Segmented5DDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({64, 1, 64, 32, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::GNHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({2048, 2048, 1, 64, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

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
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 16 * 1 * 64 * 32 * 1);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 16 * 1 * 64 * 32 * 1);

    const SmallVector<int64_t> newShape({32, 1, 64, 32, 1});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto chnagedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(chnagedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElementType.getElementType()));

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedShapeAndElementType.getElementType()));

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::GNCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::GNCHW);
    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit, 128_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({64, 1, 32, 32, 1});
    const SmallVector<Bit> tileStrides({16384_Bit, 16384_Bit, 16_Bit, 512_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1, 1});
    const SmallVector<Bit> newStrides2({65536_Bit, 32768_Bit, 16_Bit, 1024_Bit, 1024_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({1, 3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({32768_Bit, 98304_Bit, 16_Bit, 2048_Bit, 1024_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2, 1});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST_F(MLIR_NDTypeInterface, SegmentedDuplicatedDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 13);

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
    const auto chnagedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(chnagedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElementType.getElementType()));

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedShapeAndElementType.getElementType()));

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);

    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> newStrides2({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({638976_Bit, 16_Bit, 32768_Bit, 1024_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST_F(MLIR_NDTypeInterface, SegmentedOverlappedDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 3);
    const auto memNumTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 3, 1}));
    const auto computeShapes = SmallVector<SmallVector<int64_t>>{
            {1, 32, 48, 48},  // Cluster 0: channels 0-31
            {1, 32, 48, 48},  // Cluster 1: channels 32-63
            {1, 32, 48, 48}   // Cluster 2: channels 64-95
    };
    const auto computeOffsets = SmallVector<SmallVector<int64_t>>{
            {0, 0, 0, 0},   // Cluster 0
            {0, 32, 0, 0},  // Cluster 1
            {0, 64, 0, 0}   // Cluster 2
    };
    const auto memoryShapes = SmallVector<SmallVector<int64_t>>{
            {1, 96, 16, 48},  // Cluster 0: lines 0-15
            {1, 96, 16, 48},  // Cluster 1: lines 16-31
            {1, 96, 16, 48}   // Cluster 2: lines 32-47
    };
    const auto memoryOffsets = SmallVector<SmallVector<int64_t>>{
            {0, 0, 0, 0},   // Cluster 0
            {0, 0, 16, 0},  // Cluster 1
            {0, 0, 32, 0}   // Cluster 2
    };
    const auto computeShapesAttr = getIntArrayOfArray(&ctx, computeShapes);
    const auto computeOffsetsAttr = getIntArrayOfArray(&ctx, computeOffsets);
    const auto memoryShapesAttr = getIntArrayOfArray(&ctx, memoryShapes);
    const auto memoryOffsetsAttr = getIntArrayOfArray(&ctx, memoryOffsets);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            computeShapesAttr, computeOffsetsAttr, memoryShapesAttr, memoryOffsetsAttr, nullptr, memNumTilesAttr);

    const auto shape = SmallVector<int64_t>({1, 96, 48, 48});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    // Element strides for NHWC with shape [1, 96, 48, 48]: [C*H*W, 1, C*W, C]
    const auto elemStrides = SmallVector<int64_t>({96 * 48 * 48, 1, 96 * 48, 96});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 96, 48, 48}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 48, 48, 96}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 96 * 48 * 48);

    EXPECT_TRUE(mlir::isa<mlir::Float16Type>(ndType.getElementType()));

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    // Strides for NHWC layout with full shape [1, 96, 48, 48]
    // [C*H*W*16, 16, C*W*16, C*16] = [96*48*48*16, 16, 96*48*16, 96*16]
    const SmallVector<vpux::Bit> strides({3538944_Bit, 16_Bit, 73728_Bit, 1536_Bit});
    const SmallVector<vpux::Bit> memStrides({3538944_Bit, 73728_Bit, 1536_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    // Total alloc size for memory shape per cluster: 96*16*48 elements * 2 bytes
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 96 * 16 * 48);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 96 * 16 * 48);

    const SmallVector<int64_t> newShape({1, 32, 24, 96});
    EXPECT_ANY_THROW(ndType.changeShape(vpux::ShapeRef(newShape)));
    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElementType.getElementType()));

    EXPECT_ANY_THROW(ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx)));

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);

    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({7077888_Bit, 16_Bit, 73728_Bit, 1536_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    EXPECT_ANY_THROW(ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape)));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

TEST_F(MLIR_NDTypeInterface, CompressedSegmentedDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({64, 16, 1, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::OYXI.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({16, 1, 16, 16});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const SmallVector<int64_t> numElems(64, 15);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));

    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto sparsityCompression = VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                         numElemsAttr, getIntAttr(&ctx, alignment));

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(VPUIP::DistributedBufferType::get(
            &ctx, shape, elemType, layout, dimsSpace, distributedAttr, sparsityCompression));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));
    EXPECT_EQ(ndType.getTotalAllocSize().count(),
              (64 / 4) * (15 * sizeof(vpux::type::float16) + 2));  // weight-set size aligned to 16 bytes
    EXPECT_EQ(ndType.getCompactAllocSize().count(),
              (64 / 4) * (15 * sizeof(vpux::type::float16) + 2));  // weight-set size aligned to 16 bytes

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 16, 1, 1});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto distTiledType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(tiledType);
    ASSERT_TRUE(distTiledType != nullptr);
    auto tiledNumElems = distTiledType.getSparsityCompression().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);
}

TEST_F(MLIR_NDTypeInterface, CompressedDuplicatedDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({64, 80, 1, 1});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::OYXI.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({80, 1, 80, 80});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto sparsityCompression = VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                         numElemsAttr, getIntAttr(&ctx, alignment));

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(VPUIP::DistributedBufferType::get(
            &ctx, shape, elemType, layout, dimsSpace, distributedAttr, sparsityCompression));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));

    int64_t totalByteSize = 0;
    for (auto elems : numElems) {
        int64_t weightSetByteSize = elems * sizeof(vpux::type::float16);
        totalByteSize += vpux::alignValUp(weightSetByteSize, alignment);
    }
    EXPECT_EQ(ndType.getTotalAllocSize().count(), totalByteSize);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), totalByteSize);

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({32, 80, 1, 1});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto distTiledType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(tiledType);
    ASSERT_TRUE(distTiledType != nullptr);
    auto tiledNumElems = distTiledType.getSparsityCompression().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedBufferDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    // SOH
    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedBufferUniformDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    // SOH
    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedBufferDistributionStrideOnW) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    // SOH, W strided
    const auto shape = SmallVector<int64_t>({1, 64, 13, 4});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 4 * 2 * 13, 1, 64 * 4 * 2, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 4}), Shape({1, 64, 4, 4}), Shape({1, 64, 4, 4}), Shape({1, 64, 1, 4})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 4}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{32768_Bit, 16_Bit, 8192_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 8192_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 8192_Bit, 1024_Bit},
                                                {8192_Bit, 16_Bit, 8192_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 8 * 4 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 4 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedBufferUniformDistributionStrideOnW) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    // SOH, W strided
    const auto shape = SmallVector<int64_t>({1, 64, 13, 4});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 4 * 2 * 13, 1, 64 * 4 * 2, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 4}), Shape({1, 64, 3, 4}), Shape({1, 64, 3, 4}), Shape({1, 64, 3, 4})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 4}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{32768_Bit, 16_Bit, 8192_Bit, 1024_Bit},
                                                {24576_Bit, 16_Bit, 8192_Bit, 1024_Bit},
                                                {24576_Bit, 16_Bit, 8192_Bit, 1024_Bit},
                                                {24576_Bit, 16_Bit, 8192_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 8 * 4 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 4 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedDuplicatedBufferDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    // SOH duplicated
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const Strides expectedStrides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 13 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedOverlappedBufferDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 3);
    const auto memNumTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 3, 1}));
    const auto computeShapes = SmallVector<SmallVector<int64_t>>{
            {1, 32, 48, 48},  // Cluster 0: channels 0-31
            {1, 32, 48, 48},  // Cluster 1: channels 32-63
            {1, 32, 48, 48}   // Cluster 2: channels 64-95
    };
    const auto computeOffsets = SmallVector<SmallVector<int64_t>>{
            {0, 0, 0, 0},   // Cluster 0
            {0, 32, 0, 0},  // Cluster 1
            {0, 64, 0, 0}   // Cluster 2
    };
    const auto memoryShapes = SmallVector<SmallVector<int64_t>>{
            {1, 96, 16, 48},  // Cluster 0: lines 0-15
            {1, 96, 16, 48},  // Cluster 1: lines 16-31
            {1, 96, 16, 48}   // Cluster 2: lines 32-47
    };
    const auto memoryOffsets = SmallVector<SmallVector<int64_t>>{
            {0, 0, 0, 0},   // Cluster 0
            {0, 0, 16, 0},  // Cluster 1
            {0, 0, 32, 0}   // Cluster 2
    };
    const auto computeShapesAttr = getIntArrayOfArray(&ctx, computeShapes);
    const auto computeOffsetsAttr = getIntArrayOfArray(&ctx, computeOffsets);
    const auto memoryShapesAttr = getIntArrayOfArray(&ctx, memoryShapes);
    const auto memoryOffsetsAttr = getIntArrayOfArray(&ctx, memoryOffsets);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            computeShapesAttr, computeOffsetsAttr, memoryShapesAttr, memoryOffsetsAttr, nullptr, memNumTilesAttr);

    const auto shape = SmallVector<int64_t>({1, 96, 48, 48});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    // Element strides for NHWC with shape [1, 96, 48, 48]: [C*H*W, 1, C*W, C]
    const auto elemStrides = SmallVector<int64_t>({96 * 48 * 48, 1, 96 * 48, 96});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    // Compute shapes: each cluster gets [1, 32, 48, 48]
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 32, 48, 48}), Shape({1, 32, 48, 48}), Shape({1, 32, 48, 48})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    // Compute offsets: segmented on C axis at [0, 32, 64]
    const SmallVector<Shape> expectedComputeOffsets({Shape({0, 0, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 64, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    // Memory shapes: each cluster gets [1, 96, 16, 48]
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 96, 16, 48}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    // Memory offsets: segmented on H axis at [0, 16, 32]
    const SmallVector<Shape> expectedMemoryOffsets({Shape({0, 0, 0, 0}), Shape({0, 0, 16, 0}), Shape({0, 0, 32, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 32, 48, 48}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    // Strides for memory shape [1, 96, 16, 48] in NHWC: [C*H*W*16, 16, C*W*16, C*16]
    // Using per-cluster memory shape: [96*16*48*16, 16, 96*48*16, 96*16]
    const Strides expectedStrides({1179648_Bit, 16_Bit, 73728_Bit, 1536_Bit});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }

    // Total alloc size: 96*16*48 elements * 2 bytes
    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 96 * 16 * 48 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 96 * 16 * 48 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, SegmentedDuplicatedBufferUniformDistribution) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    // SOH duplicated
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const Strides expectedStrides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 13 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferDistribution1x1KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // SOH overlapped, 1x1s1p0
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                            getIntAttr(&ctx, 0));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {16384_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferDistribution3x3KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    // SOH overlapped, 3x3s1p1
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 5, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 6 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 6 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferDistribution3x3KernelStride1EqualMemoryCompute) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    // SOH overlapped, 3x3s1p1
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto equalMemoryAndComputeView = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, equalMemoryAndComputeView, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 5, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    for (const auto shapePair : zip(perClusterComputeShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 6, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedMemoryShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {98304_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 6 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 6 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferDistribution3x3KernelStride2) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    // SOH overlapped, 3x3s2p1
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 5 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 5 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferUniformDistribution1x1KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // SOH overlapped, 1x1s1p0
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 0), getIntAttr(&ctx, 0), getIntAttr(&ctx, 0),
                                            getIntAttr(&ctx, 0));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferUniformDistribution3x3KernelStride1) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    // SOH overlapped, 3x3s1p1
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16}), Shape({1, 64, 3, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 10, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 4, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 6, 0}), Shape({0, 0, 9, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {81920_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 5 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 5 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferUniformDistribution3x3KernelStride2) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 26, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 26, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    // SOH overlapped, 3x3s2p1
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernel, pads, strides, numClustersAttr, nullptr,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 7, 16}), Shape({1, 64, 7, 16}), Shape({1, 64, 6, 16}), Shape({1, 64, 6, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 14, 0}), Shape({0, 0, 20, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 8, 16}), Shape({1, 64, 7, 16}), Shape({1, 64, 7, 16}), Shape({1, 64, 7, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 13, 0}), Shape({0, 0, 19, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 7, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {114688_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {114688_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {114688_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[0]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[0]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 8 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 8 * 16 * 2);
}

TEST_F(MLIR_ClusterShapeUtils, OverlappedBufferWithComputeShapesAndOffsets) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 12, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 12, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

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

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            computeShapesAttr, computeOffsetsAttr, computeShapesAttr, computeOffsetsAttr, nullptr, nullptr);

    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    SmallVector<Shape> expectedShapes;
    for (auto computeShape : computeShapes) {
        expectedShapes.push_back(Shape(computeShape));
    }
    for (const auto shapePair : zip(perClusterComputeShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    SmallVector<Shape> expectedOffsets;
    for (auto computeOffset : computeOffsets) {
        expectedShapes.push_back(Shape(computeOffset));
    }
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }
    const SmallVector<Strides> expectedStrides({{49152_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {65536_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {49152_Bit, 16_Bit, 16384_Bit, 1024_Bit}});

    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();

    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();

        EXPECT_EQ(stridedShape.shape, expectedShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }

    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * 4 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

// Single axis H alignment, H SEGMENTED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisSegmentedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 18 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 18 * 16);
}

// Multiple axis H and K alignment, H SEGMENTED mode
// TODO: why disabled?
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferMultiAxisSegmentedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 18 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 18 * 16);
}

// Single axis H alignment, DUPLICATED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisDuplicatedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    EXPECT_EQ(largestComputeShape, Shape({1, 60, 63, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
}

// Single axis H alignment, SEGMENTED|DUPLICATED mode

TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisSegmentedDuplicatedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 63 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 63 * 16);
}

// Multiple axis H and K alignment, SEGMENTED|DUPLICATED mode
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferMultiAxisSegmentedDuplicatedMode) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 60, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({60 * 16 * 59, 1, 60 * 16, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 63 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 63 * 16);
}

// Single axis K alignment, SEGMENTED mode, K tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisSegmentedModeKTiling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 110, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({110 * 16 * 59, 1, 110 * 16, 110});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
}

// Single axis K alignment, SEGMENTED mode, K tiling, invalid 4 cluster tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisSegmentedModeKTilingInvalid4Clusters) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({96 * 16 * 59, 1, 96 * 16, 96});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_ANY_THROW(ndType.getTotalAllocSize().count());
    EXPECT_ANY_THROW(ndType.getCompactAllocSize().count());
}

// Single axis K alignment, SEGMENTED mode, K tiling, valid 3 cluster tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisSegmentedModeKTilingValid3Clusters) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto numClustersAttr = getIntAttr(&ctx, 3);
    const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({96 * 16 * 59, 1, 96 * 16, 96});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 32 * 59 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 32 * 59 * 16);
}

// Single axis K alignment, SEGMENTED|DUPLICATED mode, K tiling, valid 3 cluster tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisSegmentedDuplicatedModeKTilingValid3Clusters) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto numClustersAttr = getIntAttr(&ctx, 3);
    const auto shape = SmallVector<int64_t>({1, 96, 59, 16});
    const auto elemStrides = SmallVector<int64_t>({96 * 16 * 59, 1, 96 * 16, 96});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, alignment, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 96 * 59 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 96 * 59 * 16);
}

// Single axis K alignment, OVERLAPPED mode, H tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_AlignedBufferSingleAxisOverlappedModeHTiling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto shape = SmallVector<int64_t>({1, 60, 13, 15});
    const auto elemStrides = SmallVector<int64_t>({60 * 15 * 13, 1, 60 * 15, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 5, 15}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 5 * 15);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 5 * 15);
}

// Single axis W alignment, OVERLAPPED mode, H tiling
TEST_F(MLIR_ClusterShapeUtils, DISABLED_WidthAlignedBufferSingleAxisOverlappedModeHTiling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));
    const auto shape = SmallVector<int64_t>({1, 60, 13, 15});
    const auto elemStrides = SmallVector<int64_t>({60 * 15 * 13, 1, 60 * 15, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 1, 16}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

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
    EXPECT_EQ(largestComputeShape, Shape({1, 60, 5, 16}));
    const auto numClusters = distributedType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedType.getCompactShape(clusterIdx));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(distributedType);
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 60 * 5 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 60 * 5 * 16);
}

TEST_F(MLIR_ClusterShapeUtilsDeathTest, AlignedBufferDistribution) {
    testing::GTEST_FLAG(death_test_style) = "threadsafe";
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

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
    const auto elemStrides = SmallVector<int64_t>({60 * 15 * 59, 1, 60 * 15, 60});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const mlir::MemRefLayoutAttrInterface layout = vpux::MemRefAttr::get(dimsOrder, stridesAttr,
                                                                         /*allocSize=*/nullptr, &ctx);
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 9, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const VPUIP::SparsityCompressionAttr sparsityCompressionAttr = nullptr;

    EXPECT_EQ(VPUIP::DistributedBufferType::getChecked(mlir::UnknownLoc::get(&ctx), &ctx, ArrayRef(shape), elemType,
                                                       layout, dimsSpace, distributedAttr, sparsityCompressionAttr),
              nullptr);
}

// SOH, K striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferSegmentedDistributionKStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13, 1, 64 * 2 * 16, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {32768_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * 4 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

// Overlapped, K striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferOverlappedDistributionKStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 2}));

    const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13, 1, 64 * 2 * 16, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                strides, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 5, 16}), Shape({1, 64, 2, 16})});
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 3, 0}), Shape({0, 0, 7, 0}), Shape({0, 0, 11, 0})});
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));

    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {163840_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {163840_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {65536_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * 5 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 5 * 16 * 2);
}

// SOK, H striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferKSegmentedDuplicatedDistributionHStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{425984_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {425984_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {425984_Bit, 16_Bit, 16384_Bit, 1024_Bit},
                                                {425984_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 64 * (13 * 2) * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
}

// SOK, no duplication, H striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferKSegmentedDistributionHStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{106496_Bit, 16_Bit, 4096_Bit, 256_Bit},
                                                {106496_Bit, 16_Bit, 4096_Bit, 256_Bit},
                                                {106496_Bit, 16_Bit, 4096_Bit, 256_Bit},
                                                {106496_Bit, 16_Bit, 4096_Bit, 256_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 16 * (13 * 2) * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 16 * 13 * 16 * 2);
}

// SOK, weights set (IC*Kw*Kh) striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferKSegmentedDistributionWeightSetStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    // Original size is {64, 3, 3, 3} but all values are packed and
    // aligned to 16 bytes so their memory access is aligned to 16 bytes.
    const auto shape = SmallVector<int64_t>({64, 1, 1, 27});
    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::OYXI.toAffineMap(&ctx));

    const auto elemStrides = SmallVector<int64_t>({32, 1, 32, 32});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({4, 1, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({16, 1, 1, 27}), Shape({16, 1, 1, 27}), Shape({16, 1, 1, 27}), Shape({16, 1, 1, 27})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({16, 0, 0, 0}), Shape({32, 0, 0, 0}), Shape({48, 0, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({16, 1, 1, 27}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{512_Bit, 16_Bit, 512_Bit, 512_Bit},
                                                {512_Bit, 16_Bit, 512_Bit, 512_Bit},
                                                {512_Bit, 16_Bit, 512_Bit, 512_Bit},
                                                {512_Bit, 16_Bit, 512_Bit, 512_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), 16 * 32 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 16 * 27 * 2);
}

// SOK, K striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferKSegmentedDuplicatedDistributionKStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 2 * 16, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{425984_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {425984_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {425984_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {425984_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * 13 * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
}

// SOK, H + K striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferKSegmentedDuplicatedDistributionHAndKStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13 * 2, 1, 64 * 2 * 16, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape> expectedMemoryShapes(numClustersAttr.getInt(), Shape({1, 64, 13, 16}));
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape> expectedMemoryOffsets(numClustersAttr.getInt(), Shape({0, 0, 0, 0}));
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{851968_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {851968_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {851968_Bit, 16_Bit, 32768_Bit, 2048_Bit},
                                                {851968_Bit, 16_Bit, 32768_Bit, 2048_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * (13 * 2) * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 13 * 16 * 2);
}

// SOH, W + K striding
TEST_F(MLIR_ClusterShapeUtils, StridedBufferHSegmentedDistributionWAndKStride) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 2 * 13, 1, 64 * 2 * 16 * 2, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    const SmallVector<Strides> expectedStrides({{262144_Bit, 16_Bit, 65536_Bit, 2048_Bit},
                                                {262144_Bit, 16_Bit, 65536_Bit, 2048_Bit},
                                                {262144_Bit, 16_Bit, 65536_Bit, 2048_Bit},
                                                {65536_Bit, 16_Bit, 65536_Bit, 2048_Bit}});
    const auto perClusterStridedShapes = distributedBufferType.getPerClusterMemoryStridedShapes();
    for (const auto& p : perClusterStridedShapes | indexed) {
        const auto cluster = p.index();
        const auto stridedShape = p.value();
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[cluster]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[cluster]);
    }
    const auto largestStridedShape = distributedBufferType.getLargestStridedShape();
    EXPECT_EQ(largestStridedShape.shape, expectedMemoryShapes[1]);
    EXPECT_EQ(largestStridedShape.strides, expectedStrides[1]);
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const auto stridedShape = distributedBufferType.getStridedShape(clusterIdx);
        EXPECT_EQ(stridedShape.shape, expectedMemoryShapes[clusterIdx]);
        EXPECT_EQ(stridedShape.strides, expectedStrides[clusterIdx]);
    }

    EXPECT_EQ(distributedBufferType.getTotalAllocSize().count(), (64 * 2) * (4 * 2) * 16 * 2);
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 4 * 16 * 2);
}

// SOH, H striding - unsupported
TEST_F(MLIR_ClusterShapeUtils, StridedBufferHSegmentedDistributionHStrideUnsupported) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13 * 2, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 4, 16}), Shape({1, 64, 1, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 0, 4, 0}), Shape({0, 0, 8, 0}), Shape({0, 0, 12, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 64, 4, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    // const SmallVector<Strides> expectedStrides({{131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
    //                                             {131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
    //                                             {131072_Bit, 16_Bit, 16384_Bit, 1024_Bit},
    //                                             { 32768_Bit, 16_Bit, 16384_Bit, 1024_Bit}});
    EXPECT_ANY_THROW(distributedBufferType.getPerClusterMemoryStridedShapes());
    // 64 * 16 * (4 * 2) * 2
    EXPECT_ANY_THROW(distributedBufferType.getTotalAllocSize().count());
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 64 * 16 * 4 * 2);
}

// SOK, no duplication, K striding - unsupported
TEST_F(MLIR_ClusterShapeUtils, StridedBufferKSegmentedDistributionKStrideUnsupported) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto numClustersAttr = getIntAttr(&ctx, 4);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto elemStrides = SmallVector<int64_t>({64 * 2 * 16 * 13, 1, 64 * 2 * 16, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);

    const auto perClusterComputeShapes = distributedBufferType.getPerClusterComputeShapes();
    const SmallVector<Shape> expectedComputeShapes(
            {Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16}), Shape({1, 16, 13, 16})});
    for (const auto shapePair : zip(perClusterComputeShapes, expectedComputeShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterComputeOffsets = distributedBufferType.getPerClusterComputeShapeOffsets();
    const SmallVector<Shape> expectedComputeOffsets(
            {Shape({0, 0, 0, 0}), Shape({0, 16, 0, 0}), Shape({0, 32, 0, 0}), Shape({0, 48, 0, 0})});
    for (const auto shapePair : zip(perClusterComputeOffsets, expectedComputeOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto perClusterMemoryShapes = distributedBufferType.getPerClusterMemoryShapes();
    const SmallVector<Shape>& expectedMemoryShapes = expectedComputeShapes;
    for (const auto shapePair : zip(perClusterMemoryShapes, expectedMemoryShapes)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }
    const auto perClusterMemoryOffsets = distributedBufferType.getPerClusterMemoryShapeOffsets();
    const SmallVector<Shape>& expectedMemoryOffsets = expectedComputeOffsets;
    for (const auto shapePair : zip(perClusterMemoryOffsets, expectedMemoryOffsets)) {
        EXPECT_EQ(std::get<0>(shapePair), std::get<1>(shapePair));
    }

    const auto largestComputeShape = distributedBufferType.getLargestCompactShape();
    EXPECT_EQ(largestComputeShape, Shape({1, 16, 13, 16}));
    const auto numClusters = distributedBufferType.getDistribution().getNumClusters().getInt();
    for (auto clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        EXPECT_EQ(expectedComputeShapes[clusterIdx], distributedBufferType.getCompactShape(clusterIdx));
    }

    // const SmallVector<Strides> expectedStrides({{106496_Bit, 16_Bit, 8192_Bit, 512_Bit},
    //                                             {106496_Bit, 16_Bit, 8192_Bit, 512_Bit},
    //                                             {106496_Bit, 16_Bit, 8192_Bit, 512_Bit},
    //                                             {106496_Bit, 16_Bit, 8192_Bit, 512_Bit}});
    EXPECT_ANY_THROW(distributedBufferType.getPerClusterMemoryStridedShapes());
    // (16 * 2) * 16 * 13 * 2
    EXPECT_ANY_THROW(distributedBufferType.getTotalAllocSize().count());
    EXPECT_EQ(distributedBufferType.getCompactAllocSize().count(), 16 * 13 * 16 * 2);
}

TEST_F(MLIR_NDTypeInterface, SubByteSegmentedDistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, 4);
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    // SI4 quantized type
    const auto elemType = mlir::quant::UniformQuantizedType::getChecked(
            mlir::UnknownLoc::get(&ctx), mlir::quant::QuantizationFlags::Signed, vpux::getSInt4Type(&ctx),
            mlir::Float16Type::get(&ctx), 1.0, 0, -7, 7);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr));
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 13);

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
    const auto chnagedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(chnagedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElementType.getElementType()));

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::IntegerType::get(&ctx, 4));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(changedShapeAndElementType.getElementType()));

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);
    EXPECT_ANY_THROW(ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME)));

    const SmallVector<Bit> newStrides({106496_Bit, 4_Bit, 4096_Bit, 256_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({20480_Bit, 4_Bit, 1024_Bit, 128_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> newStrides2({106496_Bit, 4_Bit, 4096_Bit, 256_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({159744_Bit, 4_Bit, 8192_Bit, 256_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    EXPECT_ANY_THROW(ndType.eraseTiledInfo());
    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));
}

// Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
// when all per-cluster memory offsets are byte-aligned.
//
// Shape [1, 4, 1, 1] in NHWC, si4 (4 bits). SEGMENTED over C with 2 tiles.
// Cluster 1 offset = [0, 2, 0, 0]: bit offset = 2 * 4 = 8 bits. 8 % 8 == 0, byte-aligned.
TEST_F(MLIR_NDTypeInterface, SubByteBufferVerifyByteAligned) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    auto makeLayout = [&](DimsOrder order, ArrayRef<int64_t> elemStrides) {
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        return vpux::MemRefAttr::get(orderAttr, stridesAttr, /*allocSize=*/nullptr, &ctx);
    };

    const auto si4Type = mlir::quant::UniformQuantizedType::getChecked(
            mlir::UnknownLoc::get(&ctx), mlir::quant::QuantizationFlags::Signed, vpux::getSInt4Type(&ctx),
            mlir::Float16Type::get(&ctx), 1.0, 0, -7, 7);
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto loc = mlir::UnknownLoc::get(&ctx);

    {
        // Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
        // when all per-cluster memory offsets are byte-aligned.
        //
        // Shape [1, 4, 1, 1] in NHWC, si4 (4 bits). SEGMENTED over C with 2 tiles.
        // Cluster 1 offset = [0, 2, 0, 0]: bit offset = 2 * 4 = 8 bits. 8 % 8 == 0, byte-aligned.
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 4, 1, 1});
        // NHWC element strides for shape [1, 4, 1, 1]: [C*H*W, 1, C*W, C] = [4, 1, 4, 4]
        const auto layout = makeLayout(DimsOrder::NHWC, {4, 1, 4, 4});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::succeeded(result));
    }

    // Verify that DistributedBufferType::verify fails for a subbyte (si4) SEGMENTED type
    // when a per-cluster memory offset is not byte-aligned.
    //
    // Shape [2, 6, 1, 1] in NHWC, si4 (4 bits). SEGMENTED over C with 2 tiles.
    // Cluster 1 offset = [0, 3, 0, 0]: bit offset = 3 * 4 = 12 bits. 12 % 8 == 4, not byte-aligned.
    {
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({2, 6, 1, 1});
        // NHWC element strides for shape [2, 6, 1, 1]: [C*H*W, 1, C*W, C] = [6, 1, 6, 6]
        const auto layout = makeLayout(DimsOrder::NHWC, {6, 1, 6, 6});

        // Suppress diagnostic output emitted by verify on failure.
        mlir::ScopedDiagnosticHandler diagHandler(&ctx, [](mlir::Diagnostic&) {
            return mlir::success();
        });

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::failed(result));
    }

    {
        // Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
        // when all per-cluster memory offsets are byte-aligned.
        //
        // Shape [1, 2, 6, 1] in NHWC, si4 (4 bits). SEGMENTED over H with 2 tiles.
        // Cluster 1 offset = [0, 0, 3, 0]: bit offset = 3 * 2 * 4bits = 24 bits. 24 % 8 == 0, byte-aligned.
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 2, 6, 1});
        // NHWC element strides for shape [1, 2, 6, 1]: [C*H*W, 1, C*W, C] = [12, 1, 2, 2]
        const auto layout = makeLayout(DimsOrder::NHWC, {12, 1, 2, 2});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::succeeded(result));
    }

    {
        // Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
        // when all per-cluster memory offsets are byte-aligned.
        //
        // Shape [1, 1, 5, 1] in NHWC, si4 (4 bits). SEGMENTED over H with 2 tiles, with alignment [1, 1, 2, 1].
        // Cluster 1 offset = [0, 0, 4, 0]: bit offset = 4 * 4bits = 16 bits. 16 % 8 == 0, byte-aligned.
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr,
                                                                    nullptr, numClusters, alignment, nullptr, nullptr,
                                                                    nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 1, 5, 1});
        // NHWC element strides for shape [1, 1, 5, 1]: [C*H*W, 1, C*W, C] = [5, 1, 1, 1]
        const auto layout = makeLayout(DimsOrder::NHWC, {5, 1, 1, 1});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::succeeded(result));
    }

    {
        // Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
        // when all per-cluster memory offsets are byte-aligned.
        //
        // Shape [1, 1, 7, 1] in NHWC, si4 (4 bits). SEGMENTED over H with 2 tiles.
        // Cluster 1 offset = [0, 0, 4, 0]: bit offset = 4 * 4bits = 16 bits. 16 % 8 == 0, byte-aligned.
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 1, 7, 1});
        // NHWC element strides for shape [1, 1, 7, 1]: [C*H*W, 1, C*W, C] = [7, 1, 1, 1]
        const auto layout = makeLayout(DimsOrder::NHWC, {7, 1, 1, 1});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::succeeded(result));
    }
}

// Verify that DistributedBufferType::verify uses the actual layout strides rather than
// deriving compact strides from the shape when checking per-cluster byte alignment.
//
// Shape [1, 3, 2, 1] NHWC si4, SEGMENTED H÷2.
// Compact H element stride = 3 → bit stride = 12 bits. Cluster-1 H=1 offset = 12 bits (12%8≠0).
//   With compact strides this offset would not be byte-aligned.
//   Non-compact H stride = 6 → bit stride = 24 bits. Cluster-1 H=1 offset = 24 bits (24%8=0). PASS
//   Non-compact H stride = 9 → bit stride = 36 bits. Cluster-1 H=1 offset = 36 bits (36%8≠0). FAIL
//
// Shape [1, 2, 2, 2] NHWC si4, SEGMENTED C÷2.
// Compact C element stride = 1 → bit stride = 4 bits. Cluster-1 C=1 offset = 4 bits (4%8≠0).
//   Non-compact C stride = 2 → bit stride = 8 bits. Cluster-1 C=1 offset = 8 bits (8%8=0). PASS
TEST_F(MLIR_NDTypeInterface, NonCompactStridesBufferVerify) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    auto makeLayout = [&](DimsOrder order, ArrayRef<int64_t> elemStrides) {
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
        const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
        return vpux::MemRefAttr::get(orderAttr, stridesAttr, /*allocSize=*/nullptr, &ctx);
    };

    const auto si4Type = mlir::quant::UniformQuantizedType::getChecked(
            mlir::UnknownLoc::get(&ctx), mlir::quant::QuantizationFlags::Signed, vpux::getSInt4Type(&ctx),
            mlir::Float16Type::get(&ctx), 1.0, 0, -7, 7);
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto loc = mlir::UnknownLoc::get(&ctx);

    {
        // Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
        // when a non-compact H stride makes a cluster-1 offset byte-aligned.
        //
        // Shape [1, 3, 2, 1] NHWC, SEGMENTED H÷2.
        // Compact H elem stride = W*C = 1*3 = 3. Non-compact H elem stride = 6 (padded row).
        // Cluster-1 H=1 bit offset = 1 * 6 * 4bits = 24 bits. 24 % 8 == 0, byte-aligned.
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 3, 2, 1});
        // Non-compact logical element strides [N=12, C=1, H=6, W=3]: H stride padded to 6 (from compact 3).
        const auto layout = makeLayout(DimsOrder::NHWC, {12, 1, 6, 3});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::succeeded(result));
    }

    {
        // Verify that DistributedBufferType::verify fails for a subbyte (si4) SEGMENTED type
        // when a non-compact H stride still leaves a cluster-1 offset not byte-aligned.
        //
        // Shape [1, 3, 2, 1] NHWC, SEGMENTED H÷2.
        // Non-compact H elem stride = 9 (odd multiple of 4 bits).
        // Cluster-1 H=1 bit offset = 1 * 9 * 4bits = 36 bits. 36 % 8 != 0, not byte-aligned.
        mlir::ScopedDiagnosticHandler diagHandler(&ctx, [](mlir::Diagnostic&) {
            return mlir::success();
        });

        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 3, 2, 1});
        // Non-compact logical element strides [N=18, C=1, H=9, W=3]: H stride = 9 (still odd in bits).
        const auto layout = makeLayout(DimsOrder::NHWC, {18, 1, 9, 3});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::failed(result));
    }

    {
        // Verify that DistributedBufferType::verify succeeds for a subbyte (si4) SEGMENTED type
        // when a non-compact C stride makes a cluster-1 C offset byte-aligned.
        //
        // Shape [1, 2, 2, 2] NHWC, SEGMENTED C÷2.
        // Compact C elem stride = 1 (4 bits). Non-compact C elem stride = 2 (8 bits per channel).
        // Cluster-1 C=1 bit offset = 1 * 2 * 4bits = 8 bits. 8 % 8 == 0, byte-aligned.
        const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
        const auto numClusters = getIntAttr(&ctx, 2);
        const auto distributedAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto shape = SmallVector<int64_t>({1, 2, 2, 2});
        // Non-compact logical element strides [N=16, C=2, H=8, W=4]: C stride padded to 2 (from compact 1).
        const auto layout = makeLayout(DimsOrder::NHWC, {16, 2, 8, 4});

        const auto result = VPUIP::DistributedBufferType::verify(
                [loc]() -> mlir::InFlightDiagnostic {
                    return mlir::emitError(loc);
                },
                shape, si4Type, layout, dimsSpace, distributedAttr, nullptr);
        EXPECT_TRUE(mlir::succeeded(result));
    }
}
