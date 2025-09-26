//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;
using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;

using MLIR_ExplicitDistributionAndSparseTypesUtils = MLIR_UnitBase;

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSparseData) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 18, 18});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 12, 18}, SmallVector<int64_t>{1, 64, 11, 18}});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, perClusterMemoryShapes);
    const auto perClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, perClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 9, 18}, SmallVector<int64_t>{1, 64, 9, 18}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 9, 0}});
    const auto perClusterComputeShapesAttr = getIntArrayOfArray(&ctx, perClusterComputeShapes);
    const auto perClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, perClusterComputeOffsets);

    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterComputeShapesAttr, perClusterComputeOffsetsAttr,
                                           perClusterMemoryShapesAttr, perClusterMemoryOffsetsAttr, nullptr);

    // Activation Sparsity, no Storage Element Table
    auto dataExplicitDistributedAttr = VPU::getExplicitDistrAttrForSparseData(distributedAttr, shape, nullptr, &ctx);
    EXPECT_EQ(distributedAttr, dataExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSparseData_InterpNearest) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 18, 18});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 12, 18}, SmallVector<int64_t>{1, 64, 11, 18}});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, perClusterMemoryShapes);
    const auto perClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, perClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 9, 18}, SmallVector<int64_t>{1, 64, 9, 18}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 9, 0}});
    const auto perClusterComputeShapesAttr = getIntArrayOfArray(&ctx, perClusterComputeShapes);
    const auto perClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, perClusterComputeOffsets);

    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterComputeShapesAttr, perClusterComputeOffsetsAttr,
                                           perClusterMemoryShapesAttr, perClusterMemoryOffsetsAttr, nullptr);

    // Activation Sparsity + SETable - Interp NEAREST
    const SmallVector<float> scale({1, 1, 3, 3});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);
    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);
    const SmallVector<int64_t> sizes({1, 64, 18, 18});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);

    const auto inferredShape = SEInterpolateAttr.backInferInputShape(shape);
    auto dataExplicitDistributedAttr =
            VPU::getExplicitDistrAttrForSparseData(distributedAttr, inferredShape, SEInterpolateAttr, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 4, 6}, SmallVector<int64_t>{1, 64, 4, 6}});
    const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 2, 0}});
    const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
    const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 3, 6}, SmallVector<int64_t>{1, 64, 3, 6}});
    const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 3, 0}});
    const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
    const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
            expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, dataExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSparseData_InterpBilinear) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 19, 19});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 13, 19}, SmallVector<int64_t>{1, 64, 10, 19}});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 9, 0}});
    const auto perClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, perClusterMemoryShapes);
    const auto perClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, perClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 10, 19}, SmallVector<int64_t>{1, 64, 9, 19}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 10, 0}});
    const auto perClusterComputeShapesAttr = getIntArrayOfArray(&ctx, perClusterComputeShapes);
    const auto perClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, perClusterComputeOffsets);

    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterComputeShapesAttr, perClusterComputeOffsetsAttr,
                                           perClusterMemoryShapesAttr, perClusterMemoryOffsetsAttr, nullptr);

    // Activation Sparsity + SETable - Interp BILINEAR
    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);
    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);
    const SmallVector<int64_t> sizes({1, 64, 19, 19});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::BILINEAR);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);

    const auto inferredShape = SEInterpolateAttr.backInferInputShape(shape);
    auto dataExplicitDistributedAttr =
            VPU::getExplicitDistrAttrForSparseData(distributedAttr, inferredShape, SEInterpolateAttr, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 7, 9}, SmallVector<int64_t>{1, 64, 5, 9}});
    const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 4, 0}});
    const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
    const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 5, 9}, SmallVector<int64_t>{1, 64, 4, 9}});
    const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 5, 0}});
    const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
    const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
            expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, dataExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSparsityMap_Activation) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 18, 18});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 12, 18}, SmallVector<int64_t>{1, 64, 11, 18}});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, perClusterMemoryShapes);
    const auto perClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, perClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 9, 18}, SmallVector<int64_t>{1, 64, 9, 18}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 9, 0}});
    const auto perClusterComputeShapesAttr = getIntArrayOfArray(&ctx, perClusterComputeShapes);
    const auto perClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, perClusterComputeOffsets);

    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterComputeShapesAttr, perClusterComputeOffsetsAttr,
                                           perClusterMemoryShapesAttr, perClusterMemoryOffsetsAttr, nullptr);

    auto sparsityMapExplicitDistributedAttr =
            VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, shape, nullptr, &ctx);
    EXPECT_EQ(distributedAttr, sparsityMapExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSparsityMap_WeightsDuplicated) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto sparsityMapShape = Shape({64, 1, 1, 256});
    const auto isWeights = mlir::UnitAttr::get(&ctx);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const int64_t numClusters = 2;
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({numClusters, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    const PerClusterShapesOffsetsVec perClusterShapes(numClusters, SmallVector<int64_t>{64, 16, 2, 2});
    const PerClusterShapesOffsetsVec perClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    auto sparsityMapExplicitDistributedAttr =
            VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, sparsityMapShape, isWeights, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters, SmallVector<int64_t>{64, 1, 1, 256});
    const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});
    const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
    const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
            expectedPerClusterOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, sparsityMapExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSparsityMap_WeightsSegmented) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto sparsityMapShape = Shape({64, 1, 1, 256});
    const auto isWeights = mlir::UnitAttr::get(&ctx);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const int64_t numClusters = 2;
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({numClusters, 1, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{32, 16, 2, 2}, SmallVector<int64_t>{32, 16, 2, 2}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{32, 0, 0, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    auto sparsityMapExplicitDistributedAttr =
            VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, sparsityMapShape, isWeights, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterShapes(
            {SmallVector<int64_t>{32, 1, 1, 256}, SmallVector<int64_t>{32, 1, 1, 256}});
    const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{32, 0, 0, 0}});
    const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
    const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
            expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
            expectedPerClusterOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, sparsityMapExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSETable_SegmentedSeTableChannels) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 18, 18});
    const auto seTableShape = Shape({1, 2, 18, 18});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 32, 18, 18}, SmallVector<int64_t>{1, 32, 18, 18}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto seSize = shape[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];
    auto seTableExplicitDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, seSize, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterShapes(
            {SmallVector<int64_t>{1, 1, 18, 18}, SmallVector<int64_t>{1, 1, 18, 18}});
    const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 1, 0, 0}});
    const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
    const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
            expectedPerClusterOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, seTableExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSETable_NotSegmentedSeTableChannels) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 18, 18});
    const auto seTableShape = Shape({1, 1, 18, 18});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 32, 18, 18}, SmallVector<int64_t>{1, 32, 18, 18}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto seSize = shape[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];
    auto seTableExplicitDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, seSize, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterShapes(
            {SmallVector<int64_t>{1, 1, 18, 18}, SmallVector<int64_t>{1, 1, 18, 18}});
    const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 0}});
    const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
    const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
            expectedPerClusterOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, seTableExplicitDistributedAttr);
}

TEST_F(MLIR_ExplicitDistributionAndSparseTypesUtils, getExplicitDistrAttrForSETable_OverlappedSeTable) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = Shape({1, 64, 18, 18});
    const auto seTableShape = Shape({1, 2, 18, 18});

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterMemoryShapes(
            {SmallVector<int64_t>{1, 64, 12, 18}, SmallVector<int64_t>{1, 64, 11, 18}});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, perClusterMemoryShapes);
    const auto perClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, perClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{1, 64, 9, 18}, SmallVector<int64_t>{1, 64, 9, 18}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 9, 0}});
    const auto perClusterComputeShapesAttr = getIntArrayOfArray(&ctx, perClusterComputeShapes);
    const auto perClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, perClusterComputeOffsets);

    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterComputeShapesAttr, perClusterComputeOffsetsAttr,
                                           perClusterMemoryShapesAttr, perClusterMemoryOffsetsAttr, nullptr);

    const auto seSize = shape[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];
    auto seTableExplicitDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, seSize, &ctx);

    const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(
            {SmallVector<int64_t>{1, 2, 12, 18}, SmallVector<int64_t>{1, 2, 11, 18}});
    const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
    const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);

    const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(
            {SmallVector<int64_t>{1, 2, 9, 18}, SmallVector<int64_t>{1, 2, 9, 18}});
    const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 9, 0}});
    const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
    const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);

    const auto expectedDistributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
            expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

    EXPECT_EQ(expectedDistributedAttr, seTableExplicitDistributedAttr);
}
