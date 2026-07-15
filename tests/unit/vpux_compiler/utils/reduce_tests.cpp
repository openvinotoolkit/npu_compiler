//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>
#include <mlir/Parser/Parser.h>

using namespace vpux;

TEST(MLIR_ReduceTest, calculateReducedOutputLayout) {
    // No alignment single axis tiling
    {
        mlir::SmallVector<std::tuple<DimsOrder, mlir::SmallVector<int64_t>, DimsOrder>> dimOrderVec = {
                {/*inputDimOrder*/ DimsOrder::fromCode(0x1234), /*axes*/ {1, 2},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x12)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x1432), /*axes*/ {1, 2},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x21)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x1324), /*axes*/ {2},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x123)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x13452), /*axes*/ {2, 3},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x123)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x13452), /*axes*/ {1, 5},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x231)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x12345), /*axes*/ {1, 2, 3, 4},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1)}};

        for (auto it : dimOrderVec) {
            auto inputDimOrder = std::get<0>(it);
            auto axes = std::get<1>(it);
            auto actualOutputDimOrder = vpux::IE::calculateReducedOutputLayout(inputDimOrder, axes);
            EXPECT_EQ(actualOutputDimOrder, std::get<2>(it));
        }
    }
}

// =============================================================================
// Tests for VPU::getReduceOutputType and VPU::getReduceOutputBuffers
// =============================================================================

using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class MLIR_ReduceTileUtils : public vpux::VPU::arch37xx::UnitTest {};

// ---------------------------------------------------------------------------
// Helper: parse a module (verification disabled) and return the first
// VPU.NCE.MaxPool operation found inside a function.
// The returned OwningOpRef keeps the module (and its ops) alive.
// ---------------------------------------------------------------------------

struct ParsedModule {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::Operation* op = nullptr;
};

static ParsedModule parseNCEOp(mlir::MLIRContext* ctx, llvm::StringRef ir) {
    // Disable post-parse verification so that NCE-specific invariants (e.g.
    // CMX memory-space requirements, DPU workload constraints) do not prevent
    // the tests from constructing minimal ops.
    mlir::ParserConfig parseConfig(ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, parseConfig);
    if (!module) {
        return {};
    }

    mlir::Operation* found = nullptr;
    module.get()->walk([&](vpux::VPU::NCEOpInterface nceOp) {
        found = nceOp.getOperation();
        return mlir::WalkResult::interrupt();
    });

    return {std::move(module), found};
}

// ---------------------------------------------------------------------------
// Shared MLIR snippets
// ---------------------------------------------------------------------------

// 2-result pool: axes=[1] (C)
static constexpr llvm::StringLiteral kPoolReduceC = R"mlir(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  func.func @pool_reduce_c(
      %arg0: tensor<1x32x16x16xf16, {order = #NHWC}>)
      -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>) {
    %out, %red = VPU.NCE.MaxPool(%arg0) {
        axes_value          = [1],
        kernel_size         = [1, 1],
        pad                 = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        ppe                 = #VPU.PPEStub<>,
        strides             = [1, 1],
        resultSegmentSizes  = array<i32: 1, 1, 0, 0>
    } -> tensor<1x32x16x16xf16, {order = #NHWC}>,
        tensor<1x1x16x16xf16,  {order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16]
                       pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %out, %red
        : tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
  }
}
)mlir";

// 1-result pool: no reduce outputs, no axes_value
static constexpr llvm::StringLiteral kPoolSingleResult = R"mlir(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  func.func @pool_single(
      %arg0: tensor<1x32x16x16xf16, {order = #NHWC}>)
      -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %out = VPU.NCE.MaxPool(%arg0) {
        kernel_size         = [1, 1],
        pad                 = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        ppe                 = #VPU.PPEStub<>,
        strides             = [1, 1],
        resultSegmentSizes  = array<i32: 1, 0, 0, 0>
    } -> tensor<1x32x16x16xf16, {order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16]
                       pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %out : tensor<1x32x16x16xf16, {order = #NHWC}>
  }
}
)mlir";

// 3-result pool: output + reduce_xy_max + reduce_xy_min, axes=[1]
static constexpr llvm::StringLiteral kPoolThreeResults = R"mlir(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  func.func @pool_three_results(
      %arg0: tensor<1x32x16x16xf16, {order = #NHWC}>)
      -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>) {
    %out, %rmax, %rmin = VPU.NCE.MaxPool(%arg0) {
        axes_value          = [1],
        kernel_size         = [1, 1],
        pad                 = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        ppe                 = #VPU.PPEStub<>,
        strides             = [1, 1],
        resultSegmentSizes  = array<i32: 1, 1, 1, 0>
    } -> tensor<1x32x16x16xf16, {order = #NHWC}>,
        tensor<1x1x16x16xf16,  {order = #NHWC}>,
        tensor<1x1x16x16xf16,  {order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16]
                       pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %out, %rmax, %rmin
        : tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
  }
}
)mlir";

// 3-result NCEMatMul: output [1x1x32x16x16] + reduce_xy_max [1x1x1x16x16] + reduce_xy_min [1x1x1x16x16]
// GNHWC layout, axes_value=[2] reduces the C dimension (logical dim 2 in GNHWC).
static constexpr llvm::StringLiteral kMatMulReduceC = R"mlir(
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
module {
  func.func @matmul_reduce_c(
      %arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
      %arg1: tensor<3x32x32x1x1xf16,  {order = #GNHWC}>)
      -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>) {
    %out, %rmax, %rmin = VPU.NCE.MatMul(%arg0, %arg1) rawFilterShape [3, 32, 32, 1, 1] {
        axes_value          = [2],
        pad                 = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        ppe                 = #VPU.PPEStub<>,
        strides             = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
        resultSegmentSizes  = array<i32: 1, 1, 1, 0>
    } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
        tensor<3x1x1x16x16xf16,  {order = #GNHWC}>,
        tensor<3x1x1x16x16xf16,  {order = #GNHWC}> {
      VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [3, 1, 32, 16, 16]
                       outOffsets [0, 0, 0, 0, 0] outSizes [3, 1, 32, 16, 16]
                       pad [0, 0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
    }
    return %out, %rmax, %rmin
        : tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>
  }
}
)mlir";

// ---------------------------------------------------------------------------
// Helper: build a OVERLAPPED-over-H VPU::DistributionInfoAttr for 2 clusters.
// Input/output shape: [1, 32, 16, 16] f16 NHWC.
// ---------------------------------------------------------------------------
static VPU::DistributionInfoAttr makeOverlappedOverHAttr(mlir::MLIRContext* ctx, ShapeRef shape) {
    const auto distributionMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, 2, 1});
    const auto numClustersAttr = getIntAttr(ctx, 2);

    const auto hSizeCl0 = vpux::divUp(shape[Dims4D::Act::H], (int64_t)2);
    const auto hSizeCl1 = shape[Dims4D::Act::H] - hSizeCl0;

    const auto indW = Dims4D::Act::W;
    const auto indC = Dims4D::Act::C;
    const auto indN = Dims4D::Act::N;

    // Per-cluster memory/compute shapes and offsets for [1,32,16,16] split over H
    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{shape[indN], shape[indC], hSizeCl0, shape[indW]},
             SmallVector<int64_t>{shape[indN], shape[indC], hSizeCl1, shape[indW]}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, hSizeCl0, 0}});
    const PerClusterShapesOffsetsVec perClusterSMemoryShapes(
            {SmallVector<int64_t>{shape[indN], shape[indC], hSizeCl0 + 1, shape[indW]},
             SmallVector<int64_t>{shape[indN], shape[indC], hSizeCl1 + 1, shape[indW]}});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, hSizeCl0 - 1, 0}});

    return VPU::DistributionInfoAttr::get(
            ctx, distributionMode, numTilesAttr,
            /*kernel=*/nullptr, /*pad=*/nullptr, /*stride=*/nullptr, numClustersAttr, /*alignment=*/nullptr,
            /*uniformDistributedSegments=*/nullptr, getIntArrayOfArray(ctx, perClusterComputeShapes),
            getIntArrayOfArray(ctx, perClusterComputeOffsets), getIntArrayOfArray(ctx, perClusterSMemoryShapes),
            getIntArrayOfArray(ctx, perClusterMemoryOffsets),
            /*equalMemoryAndComputeView=*/nullptr, /*memoryNumTiles=*/nullptr);
}

// ---------------------------------------------------------------------------
// Helper: build a SEGMENTED-over-G VPU::DistributionInfoAttr for 2 clusters.
// ---------------------------------------------------------------------------
static VPU::DistributionInfoAttr makeSegmentedOverGAttr(mlir::MLIRContext* ctx, ShapeRef shape) {
    const auto distributionMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{2, 1, 1, 1, 1});
    const auto numClustersAttr = getIntAttr(ctx, 2);

    const auto gSizeCl0 = vpux::divUp(shape[DimsGroups5D::Act::G], (int64_t)2);
    const auto gSizeCl1 = shape[DimsGroups5D::Act::G] - gSizeCl0;

    // Per-cluster compute/memory shapes and offsets split evenly over G (dim 0)
    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{gSizeCl0, shape[DimsGroups5D::Act::N], shape[DimsGroups5D::Act::C],
                                  shape[DimsGroups5D::Act::H], shape[DimsGroups5D::Act::W]},
             SmallVector<int64_t>{gSizeCl1, shape[DimsGroups5D::Act::N], shape[DimsGroups5D::Act::C],
                                  shape[DimsGroups5D::Act::H], shape[DimsGroups5D::Act::W]}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0, 0}, SmallVector<int64_t>{gSizeCl0, 0, 0, 0, 0}});

    return VPU::DistributionInfoAttr::get(
            ctx, distributionMode, numTilesAttr,
            /*kernel=*/nullptr, /*pad=*/nullptr, /*stride=*/nullptr, numClustersAttr, /*alignment=*/nullptr,
            /*uniformDistributedSegments=*/nullptr, getIntArrayOfArray(ctx, perClusterShapes),
            getIntArrayOfArray(ctx, perClusterOffsets), getIntArrayOfArray(ctx, perClusterShapes),
            getIntArrayOfArray(ctx, perClusterOffsets),
            /*equalMemoryAndComputeView=*/nullptr, /*memoryNumTiles=*/nullptr);
}

// ---------------------------------------------------------------------------
// Helper: build a SEGMENTED|MULTICASTED VPU::DistributionInfoAttr for 2 clusters.
// Compute shapes are split over H; memory shapes are full (multicasted).
// ---------------------------------------------------------------------------
static VPU::DistributionInfoAttr makeSegMulticastedAttr(mlir::MLIRContext* ctx, ShapeRef shape) {
    const auto distributionMode =
            VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
    const auto numTilesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, 2, 1});
    const auto numClustersAttr = getIntAttr(ctx, 2);

    const auto indN = Dims4D::Act::N;
    const auto indC = Dims4D::Act::C;
    const auto indH = Dims4D::Act::H;
    const auto indW = Dims4D::Act::W;

    const auto hSizeCl0 = vpux::divUp(shape[indH], (int64_t)2);
    const auto hSizeCl1 = shape[indH] - hSizeCl0;

    // Compute shapes split over H; memory shapes replicated across all clusters.
    const auto zeroOffset = SmallVector<int64_t>{0, 0, 0, 0};
    const PerClusterShapesOffsetsVec perClusterComputeShapes(
            {SmallVector<int64_t>{shape[indN], shape[indC], hSizeCl0, shape[indW]},
             SmallVector<int64_t>{shape[indN], shape[indC], hSizeCl1, shape[indW]}});
    const PerClusterShapesOffsetsVec perClusterComputeOffsets({zeroOffset, SmallVector<int64_t>{0, 0, hSizeCl0, 0}});

    const auto memoryShape = SmallVector<int64_t>(shape.begin(), shape.end());
    const PerClusterShapesOffsetsVec perClusterMemoryShapes({memoryShape, memoryShape});
    const PerClusterShapesOffsetsVec perClusterMemoryOffsets({zeroOffset, zeroOffset});

    return VPU::DistributionInfoAttr::get(
            ctx, distributionMode, numTilesAttr,
            /*kernel=*/nullptr, /*pad=*/nullptr, /*stride=*/nullptr, numClustersAttr, /*alignment=*/nullptr,
            /*uniformDistributedSegments=*/nullptr, getIntArrayOfArray(ctx, perClusterComputeShapes),
            getIntArrayOfArray(ctx, perClusterComputeOffsets), getIntArrayOfArray(ctx, perClusterMemoryShapes),
            getIntArrayOfArray(ctx, perClusterMemoryOffsets),
            /*equalMemoryAndComputeView=*/nullptr, /*memoryNumTiles=*/nullptr);
}

// ---------------------------------------------------------------------------
// getReduceOutputType — single-cluster (NDTypeInterface overload)
// ---------------------------------------------------------------------------

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_SingleCluster_EmptyForSingleResult) {
    auto parsed = parseNCEOp(&ctx, kPoolSingleResult);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto result = VPU::getReduceOutputType(parsed.op, outputType);

    EXPECT_TRUE(result.empty());
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_SingleCluster_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    // Output shape: [1, 32, 16, 16]; axes_value=[1] → reduce shape [1, 1, 16, 16]
    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    auto result = VPU::getReduceOutputType(parsed.op, outputType);

    ASSERT_EQ(result.size(), 1u);
    const auto reduceShape = result[0].getShape();
    EXPECT_EQ(reduceShape, Shape({1, 1, 16, 16}));

    // Tiled output shape: [1, 32, 8, 7]; axes_value=[1] → reduce shape [1, 1, 8, 7]
    outputType = outputType.changeShape({1, 32, 8, 7});
    result = VPU::getReduceOutputType(parsed.op, outputType);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].getShape(), Shape({1, 1, 8, 7}));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_SingleCluster_ThreeResults) {
    auto parsed = parseNCEOp(&ctx, kPoolThreeResults);
    ASSERT_NE(parsed.op, nullptr);

    // Output shape: [1, 32, 16, 16]; axes_value=[1] → reduce shape [1, 1, 16, 16]
    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    auto result = VPU::getReduceOutputType(parsed.op, outputType);

    ASSERT_EQ(result.size(), 2u);
    const auto reduceShape = result[0].getShape();
    EXPECT_EQ(reduceShape, Shape({1, 1, 16, 16}));
    EXPECT_EQ(result[1].getShape(), reduceShape);

    // Tiled output shape: [1, 32, 5, 4]; axes_value=[1] → reduce shape [1, 1, 5, 4]
    outputType = outputType.changeShape({1, 32, 5, 4});
    result = VPU::getReduceOutputType(parsed.op, outputType);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].getShape(), Shape({1, 1, 5, 4}));
    EXPECT_EQ(result[1].getShape(), Shape({1, 1, 5, 4}));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_SingleCluster_NCEMatMul) {
    auto parsed = parseNCEOp(&ctx, kMatMulReduceC);
    ASSERT_NE(parsed.op, nullptr);

    // Output shape: [3, 1, 32, 16, 16]; axes_value=[2] → reduce shape [3, 1, 1, 16, 16]
    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    auto result = VPU::getReduceOutputType(parsed.op, outputType);

    ASSERT_EQ(result.size(), 2u);
    const auto reduceShape = result[0].getShape();
    EXPECT_EQ(reduceShape, Shape({3, 1, 1, 16, 16}));
    EXPECT_EQ(result[1].getShape(), reduceShape);

    // Tiled output shape: [2, 1, 32, 5, 4]; axes_value=[2] → reduce shape [2, 1, 1, 5, 4]
    outputType = outputType.changeShape({2, 1, 32, 5, 4});
    result = VPU::getReduceOutputType(parsed.op, outputType);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].getShape(), Shape({2, 1, 1, 5, 4}));
    EXPECT_EQ(result[1].getShape(), result[0].getShape());
}

// ---------------------------------------------------------------------------
// getReduceOutputType — multi-cluster: NDTypeInterface / DistributedTensorType
// ---------------------------------------------------------------------------

void compareDistributions(const VPU::DistributionInfo& reducedDistr, const VPU::DistributionInfo& expectedDistr) {
    EXPECT_EQ(reducedDistr.getDistributionMode(), expectedDistr.getDistributionMode());
    EXPECT_EQ(reducedDistr.getNumTiles(), expectedDistr.getNumTiles());
    EXPECT_EQ(reducedDistr.getNumClusters(), expectedDistr.getNumClusters());

    const auto& reducedMemShapes = reducedDistr.getMemoryShapes();
    const auto& expectedMemShapes = expectedDistr.getMemoryShapes();
    ASSERT_EQ(reducedMemShapes.size(), expectedMemShapes.size());
    const auto& reducedComputeShapes = reducedDistr.getComputeShapes();
    const auto& expectedComputeShapes = expectedDistr.getComputeShapes();
    ASSERT_EQ(reducedComputeShapes.size(), expectedComputeShapes.size());

    const auto cDim = reducedMemShapes.front().size() == 4 ? Dims4D::Act::C : DimsGroups5D::Act::C;

    for (size_t cl = 0; cl < reducedMemShapes.size(); ++cl) {
        for (size_t dim = 0; dim < reducedMemShapes[cl].size(); ++dim) {
            if (dim != static_cast<size_t>(cDim.ind())) {
                EXPECT_EQ(reducedMemShapes[cl][dim], expectedMemShapes[cl][dim]);
                EXPECT_EQ(reducedComputeShapes[cl][dim], expectedComputeShapes[cl][dim]);
            } else {
                EXPECT_EQ(reducedMemShapes[cl][dim], 1);      // C dimension must be reduced to 1
                EXPECT_EQ(reducedComputeShapes[cl][dim], 1);  // C dimension must be reduced to 1
            }
        }
    }
}

void compareDistributions(const VPU::DistributionInfoAttr reducedDistrAttr,
                          const VPU::DistributionInfoAttr expectedDistrAttr) {
    const auto reducedDistr = VPU::DistributionInfo::getClassFromAttr(reducedDistrAttr);
    const auto expectedDistr = VPU::DistributionInfo::getClassFromAttr(expectedDistrAttr);
    compareDistributions(reducedDistr, expectedDistr);
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_DistributedTensorType_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // Build a DistributedTensorType for the primary output shape [1,32,16,16]
    // OVERLAPPED over H (dim 2), 2 clusters.
    const auto distributedAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(distributedType);
    auto result = VPU::getReduceOutputType(parsed.op, ndType);
    ASSERT_EQ(result.size(), 1u);

    // The result type must be a DistributedTensorType with reduced C dimension.
    auto reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(result[0]);
    ASSERT_NE(reducedDistType, nullptr);

    EXPECT_EQ(result[0].getShape(), Shape({1, 1, 16, 16}));

    // Per-cluster memory shapes on axis C must also be 1 after the reduction.
    compareDistributions(reducedDistType.getDistribution(), distributedAttr);

    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistributedAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    result = VPU::getReduceOutputType(parsed.op, mlir::cast<vpux::NDTypeInterface>(distributedType));
    ASSERT_EQ(result.size(), 1u);

    // The result type must be a DistributedTensorType with reduced W dimension.
    reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(result[0]);
    ASSERT_NE(reducedDistType, nullptr);

    EXPECT_EQ(result[0].getShape(), Shape({1, 1, 7, 5}));

    // Per-cluster memory shapes on axis W must also be 1 after the reduction.
    compareDistributions(reducedDistType.getDistribution(), tiledDistributedAttr);
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_DistributedTensorType_SegMulticasted_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // Build a DistributedTensorType for the primary output shape [1,32,16,16]
    // SEGMENTED|MULTICASTED 2 clusters.
    const auto distributedAttr = makeSegMulticastedAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(distributedType);
    auto result = VPU::getReduceOutputType(parsed.op, ndType);
    ASSERT_EQ(result.size(), 1u);

    // The result type must be a DistributedTensorType with reduced C dimension.
    auto reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(result[0]);
    ASSERT_NE(reducedDistType, nullptr);

    EXPECT_EQ(result[0].getShape(), Shape({1, 1, 16, 16}));

    // Per-cluster memory shapes on axis C must also be 1 after the reduction.
    compareDistributions(reducedDistType.getDistribution(), distributedAttr);

    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistributedAttr = makeSegMulticastedAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    result = VPU::getReduceOutputType(parsed.op, mlir::cast<vpux::NDTypeInterface>(distributedType));
    ASSERT_EQ(result.size(), 1u);

    // The result type must be a DistributedTensorType with reduced C dimension.
    reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(result[0]);
    ASSERT_NE(reducedDistType, nullptr);

    EXPECT_EQ(result[0].getShape(), Shape({1, 1, 7, 5}));

    // Per-cluster memory shapes on axis C must also be 1 after the reduction.
    compareDistributions(reducedDistType.getDistribution(), tiledDistributedAttr);
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_DistributedTensorType_ThreeResults) {
    auto parsed = parseNCEOp(&ctx, kPoolThreeResults);
    ASSERT_NE(parsed.op, nullptr);

    // Output shape: [1, 32, 16, 16]; axes_value=[1] → 2 reduce outputs each with shape [1, 1, 16, 16].
    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    auto result = VPU::getReduceOutputType(parsed.op, mlir::cast<vpux::NDTypeInterface>(distributedType));
    ASSERT_EQ(result.size(), 2u);

    for (const auto& reduceType : result) {
        auto reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(reduceType);
        ASSERT_NE(reducedDistType, nullptr);
        EXPECT_EQ(reduceType.getShape(), Shape({1, 1, 16, 16}));
        compareDistributions(reducedDistType.getDistribution(), distributedAttr);
    }

    // Tiled variant: output [1, 32, 7, 5] → reduce shape [1, 1, 7, 5].
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistributedAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    result = VPU::getReduceOutputType(parsed.op, mlir::cast<vpux::NDTypeInterface>(distributedType));
    ASSERT_EQ(result.size(), 2u);

    for (const auto& reduceType : result) {
        auto reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(reduceType);
        ASSERT_NE(reducedDistType, nullptr);
        EXPECT_EQ(reduceType.getShape(), Shape({1, 1, 7, 5}));
        compareDistributions(reducedDistType.getDistribution(), tiledDistributedAttr);
    }
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_DistributedTensorType_NCEMatMulReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kMatMulReduceC);
    ASSERT_NE(parsed.op, nullptr);

    // Output shape: [3, 1, 32, 16, 16] GNHWC; axes_value=[2] → 2 reduce outputs with shape [3, 1, 1, 16, 16].
    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeSegmentedOverGAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{3, 1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    auto result = VPU::getReduceOutputType(parsed.op, mlir::cast<vpux::NDTypeInterface>(distributedType));
    ASSERT_EQ(result.size(), 2u);

    for (const auto& reduceType : result) {
        auto reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(reduceType);
        ASSERT_NE(reducedDistType, nullptr);
        EXPECT_EQ(reduceType.getShape(), Shape({3, 1, 1, 16, 16}));
        compareDistributions(reducedDistType.getDistribution(), distributedAttr);
    }

    // Tiled variant: output [2, 1, 32, 16, 16] → reduce shape [2, 1, 1, 16, 16].
    const auto tiledShape = Shape({2, 1, 32, 16, 16});
    const auto tiledDistributedAttr = makeSegmentedOverGAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    result = VPU::getReduceOutputType(parsed.op, mlir::cast<vpux::NDTypeInterface>(distributedType));
    ASSERT_EQ(result.size(), 2u);

    for (const auto& reduceType : result) {
        auto reducedDistType = mlir::dyn_cast<VPU::DistributedTensorType>(reduceType);
        ASSERT_NE(reducedDistType, nullptr);
        EXPECT_EQ(reduceType.getShape(), Shape({2, 1, 1, 16, 16}));
        compareDistributions(reducedDistType.getDistribution(), tiledDistributedAttr);
    }
}

// ---------------------------------------------------------------------------
// getReduceOutputType — multi-cluster: TypeAndDistributionPair overload
// ---------------------------------------------------------------------------

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_TypeAndDistributionPair_OverlappedOverH_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // OVERLAPPED over H, 2 clusters. Output shape: [1,32,16,16]; axes_value=[1] → reduce shape [1,1,16,16].
    const auto distrAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    const auto result = VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{outputType, distributionMap});

    ASSERT_EQ(result.size(), 1u);
    const auto& [reduceType, reduceDistMap] = result[0];
    EXPECT_EQ(reduceType.getShape(), Shape({1, 1, 16, 16}));
    EXPECT_TRUE(reduceDistMap.contains(reduceType));

    const auto& reduceDistInfo = reduceDistMap.at(reduceType);
    compareDistributions(reduceDistInfo, distInfo);

    // Tiled variant: output [1, 32, 7, 5] → reduce shape [1, 1, 7, 5].
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistrAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    const auto tiledResult =
            VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});
    ASSERT_EQ(tiledResult.size(), 1u);
    const auto& [tiledReduceType, tiledReduceDistMap] = tiledResult[0];
    EXPECT_EQ(tiledReduceType.getShape(), Shape({1, 1, 7, 5}));
    EXPECT_TRUE(tiledReduceDistMap.contains(tiledReduceType));
    compareDistributions(tiledReduceDistMap.at(tiledReduceType), tiledDistInfo);
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_TypeAndDistributionPair_SegMulticasted_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // SEGMENTED|MULTICASTED over H, 2 clusters. Output shape: [1,32,16,16]; axes_value=[1] → reduce [1,1,16,16].
    const auto distrAttr = makeSegMulticastedAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    const auto result = VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{outputType, distributionMap});

    ASSERT_EQ(result.size(), 1u);
    const auto& [reduceType, reduceDistMap] = result[0];
    EXPECT_EQ(reduceType.getShape(), Shape({1, 1, 16, 16}));
    EXPECT_TRUE(reduceDistMap.contains(reduceType));

    compareDistributions(reduceDistMap.at(reduceType), distInfo);

    // Tiled variant: output [1, 32, 7, 5] → reduce shape [1, 1, 7, 5].
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistrAttr = makeSegMulticastedAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    const auto tiledResult =
            VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});
    ASSERT_EQ(tiledResult.size(), 1u);
    const auto& [tiledReduceType, tiledReduceDistMap] = tiledResult[0];
    EXPECT_EQ(tiledReduceType.getShape(), Shape({1, 1, 7, 5}));
    EXPECT_TRUE(tiledReduceDistMap.contains(tiledReduceType));
    compareDistributions(tiledReduceDistMap.at(tiledReduceType), tiledDistInfo);
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_TypeAndDistributionPair_ThreeResults) {
    auto parsed = parseNCEOp(&ctx, kPoolThreeResults);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // OVERLAPPED over H, 2 clusters. Output shape: [1,32,16,16]; axes_value=[1] → 2 reduce outputs [1,1,16,16].
    const auto distrAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    const auto result = VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{outputType, distributionMap});

    ASSERT_EQ(result.size(), 2u);
    for (const auto& [reduceType, reduceDistMap] : result) {
        EXPECT_EQ(reduceType.getShape(), Shape({1, 1, 16, 16}));
        EXPECT_TRUE(reduceDistMap.contains(reduceType));
        compareDistributions(reduceDistMap.at(reduceType), distInfo);
    }

    // Tiled variant: output [1, 32, 7, 5] → 2 reduce outputs with shape [1, 1, 7, 5].
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistrAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    const auto tiledResult =
            VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});
    ASSERT_EQ(tiledResult.size(), 2u);
    for (const auto& [tiledReduceType, tiledReduceDistMap] : tiledResult) {
        EXPECT_EQ(tiledReduceType.getShape(), Shape({1, 1, 7, 5}));
        EXPECT_TRUE(tiledReduceDistMap.contains(tiledReduceType));
        compareDistributions(tiledReduceDistMap.at(tiledReduceType), tiledDistInfo);
    }
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputType_MultiCluster_TypeAndDistributionPair_NCEMatMulReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kMatMulReduceC);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // SEGMENTED over G, 2 clusters. Output shape: [3,1,32,16,16]; axes_value=[2] → 2 reduce outputs [3,1,1,16,16].
    const auto distrAttr = makeSegmentedOverGAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    const auto result = VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{outputType, distributionMap});

    ASSERT_EQ(result.size(), 2u);
    for (const auto& [reduceType, reduceDistMap] : result) {
        EXPECT_EQ(reduceType.getShape(), Shape({3, 1, 1, 16, 16}));
        EXPECT_TRUE(reduceDistMap.contains(reduceType));
        compareDistributions(reduceDistMap.at(reduceType), distInfo);
    }

    // Tiled variant: output [2, 1, 32, 16, 16] → 2 reduce outputs with shape [2, 1, 1, 16, 16].
    const auto tiledShape = Shape({2, 1, 32, 16, 16});
    const auto tiledDistrAttr = makeSegmentedOverGAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    const auto tiledResult =
            VPU::getReduceOutputType(parsed.op, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});
    ASSERT_EQ(tiledResult.size(), 2u);
    for (const auto& [tiledReduceType, tiledReduceDistMap] : tiledResult) {
        EXPECT_EQ(tiledReduceType.getShape(), Shape({2, 1, 1, 16, 16}));
        EXPECT_TRUE(tiledReduceDistMap.contains(tiledReduceType));
        compareDistributions(tiledReduceDistMap.at(tiledReduceType), tiledDistInfo);
    }
}

// ---------------------------------------------------------------------------
// getReduceOutputBuffers — single-cluster (NDTypeInterface overload)
// ---------------------------------------------------------------------------

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_SingleCluster_SuccessForSingleResult) {
    auto parsed = parseNCEOp(&ctx, kPoolSingleResult);
    ASSERT_NE(parsed.op, nullptr);

    SmallVector<Byte> buffers;
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto status = VPU::getReduceOutputBuffers(parsed.op, buffers, outputType);

    // Single-result op → success, no buffers appended.
    EXPECT_TRUE(mlir::succeeded(status));
    EXPECT_TRUE(buffers.empty());
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_SingleCluster_ReduceC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    SmallVector<Byte> buffers;
    const auto status = VPU::getReduceOutputBuffers(parsed.op, buffers, outputType);

    EXPECT_TRUE(mlir::succeeded(status));
    // One secondary result → one buffer entry.
    ASSERT_EQ(buffers.size(), 1u);
    // Reduce shape [1, 1, 16, 16] f16 → 1*1*16*16 * 2 bytes = 512 bytes.
    EXPECT_EQ(buffers[0], Byte(512));

    // Tiled output: [1, 32, 8, 7] → reduce shape [1, 1, 8, 7] → 1*1*8*7 * 2 bytes = 112 bytes.
    const auto tiledShape = Shape({1, 32, 8, 7});
    outputType = outputType.changeShape(tiledShape);
    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus = VPU::getReduceOutputBuffers(parsed.op, tiledBuffers, outputType);

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 1u);
    EXPECT_EQ(tiledBuffers[0], Byte(112));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_SingleCluster_TwoBuffersForThreeResults) {
    auto parsed = parseNCEOp(&ctx, kPoolThreeResults);
    ASSERT_NE(parsed.op, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    SmallVector<Byte> buffers;
    const auto status = VPU::getReduceOutputBuffers(parsed.op, buffers, outputType);

    EXPECT_TRUE(mlir::succeeded(status));
    // Two secondary results → two buffer entries each of size 512 bytes.
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0], Byte(512));
    EXPECT_EQ(buffers[1], Byte(512));

    // Tiled output: [1, 32, 5, 4] → reduce shape [1, 1, 5, 4] → 1*1*5*4 * 2 bytes = 40 bytes each.
    const auto tiledShape = Shape({1, 32, 5, 4});
    outputType = outputType.changeShape(tiledShape);
    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus = VPU::getReduceOutputBuffers(parsed.op, tiledBuffers, outputType);

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 2u);
    EXPECT_EQ(tiledBuffers[0], Byte(40));
    EXPECT_EQ(tiledBuffers[1], Byte(40));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_SingleCluster_NCEMatMul_TwoBuffers_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kMatMulReduceC);
    ASSERT_NE(parsed.op, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    SmallVector<Byte> buffers;
    const auto status = VPU::getReduceOutputBuffers(parsed.op, buffers, outputType);

    EXPECT_TRUE(mlir::succeeded(status));
    // Two secondary results → two buffer entries.
    ASSERT_EQ(buffers.size(), 2u);
    // Reduce shape [3,1,1,16,16] f16 → 3*1*1*16*16 * 2 bytes = 1536 bytes each.
    EXPECT_EQ(buffers[0], Byte(1536));
    EXPECT_EQ(buffers[1], Byte(1536));

    // Tiled output: [2, 1, 32, 16, 16] → reduce shape [2, 1, 1, 16, 16] → 2*1*1*16*16 * 2 bytes = 1024 bytes each.
    const auto tiledShape = Shape({2, 1, 32, 16, 16});
    outputType = outputType.changeShape(tiledShape);
    SmallVector<Byte> tiledBuffers2;
    const auto tiledStatus = VPU::getReduceOutputBuffers(parsed.op, tiledBuffers2, outputType);

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers2.size(), 2u);
    EXPECT_EQ(tiledBuffers2[0], Byte(1024));
    EXPECT_EQ(tiledBuffers2[1], Byte(1024));
}

// ---------------------------------------------------------------------------
// getReduceOutputBuffers — multi-cluster: TypeAndDistributionPair overload
// ---------------------------------------------------------------------------

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_TypeAndDistributionPair_OverlappedOverH_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // OVERLAPPED over H, 2 clusters. Output shape: [1,32,16,16]; axes_value=[1] → reduce shape [1,1,16,16].
    // Memory shapes: {1,32,9,16} per cluster. After C reduce: {1,1,9,16} → max = 144 * 2 = 288 bytes.
    const auto distrAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, TypeAndDistributionPair{outputType, distributionMap});

    EXPECT_TRUE(mlir::succeeded(status));
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers[0], Byte(288));

    // Tiled output: [1, 32, 7, 5] → memory shapes {1,32,5,5}/{1,32,4,5} → after C reduce: max {1,1,5,5} = 50 bytes.
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistrAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    const auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus = VPU::getReduceOutputBuffers(
            parsed.op, tiledBuffers, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 1u);
    EXPECT_EQ(tiledBuffers[0], Byte(50));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_TypeAndDistributionPair_SegMulticasted_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // SEGMENTED|MULTICASTED over H, 2 clusters. Memory shapes: both {1,32,16,16} (replicated).
    // After C reduce: {1,1,16,16} → max = 256 * 2 = 512 bytes.
    const auto distrAttr = makeSegMulticastedAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, TypeAndDistributionPair{outputType, distributionMap});

    EXPECT_TRUE(mlir::succeeded(status));
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers[0], Byte(512));

    // Tiled output: [1, 32, 7, 5] → memory shapes both {1,32,7,5} → after C reduce: {1,1,7,5} = 70 bytes.
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistrAttr = makeSegMulticastedAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    const auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus = VPU::getReduceOutputBuffers(
            parsed.op, tiledBuffers, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 1u);
    EXPECT_EQ(tiledBuffers[0], Byte(70));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_TypeAndDistributionPair_ThreeResults) {
    auto parsed = parseNCEOp(&ctx, kPoolThreeResults);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // OVERLAPPED over H, 2 clusters. Output [1,32,16,16]; axes_value=[1] → 2 reduce outputs.
    // Memory shapes: {1,32,9,16} per cluster → after C reduce: {1,1,9,16} → 144 * 2 = 288 bytes each.
    const auto distrAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, TypeAndDistributionPair{outputType, distributionMap});

    EXPECT_TRUE(mlir::succeeded(status));
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0], Byte(288));
    EXPECT_EQ(buffers[1], Byte(288));

    // Tiled output: [1, 32, 7, 5] → memory shapes {1,32,5,5}/{1,32,4,5} → max {1,1,5,5} = 50 bytes each.
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistrAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    const auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus = VPU::getReduceOutputBuffers(
            parsed.op, tiledBuffers, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 2u);
    EXPECT_EQ(tiledBuffers[0], Byte(50));
    EXPECT_EQ(tiledBuffers[1], Byte(50));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_TypeAndDistributionPair_NCEMatMulReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kMatMulReduceC);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    // SEGMENTED over G, 2 clusters. Output [3,1,32,16,16]; axes_value=[2] → 2 reduce outputs.
    // Memory shapes: {2,1,32,16,16}/{1,1,32,16,16} → after C reduce: max {2,1,1,16,16} = 512 * 2 = 1024 bytes each.
    const auto distrAttr = makeSegmentedOverGAttr(&ctx, outputType.getShape());
    const VPU::DistributionInfo distInfo = VPU::DistributionInfo::getClassFromAttr(distrAttr);

    VPU::TensorDistributionMap distributionMap;
    distributionMap[outputType] = distInfo;

    using TypeAndDistributionPair = std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>;
    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, TypeAndDistributionPair{outputType, distributionMap});

    EXPECT_TRUE(mlir::succeeded(status));
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0], Byte(1024));
    EXPECT_EQ(buffers[1], Byte(1024));

    // Tiled output: [2, 1, 32, 16, 16] → memory shapes both {1,1,32,16,16} → after C reduce: 256 * 2 = 512 bytes each.
    const auto tiledShape = Shape({2, 1, 32, 16, 16});
    const auto tiledDistrAttr = makeSegmentedOverGAttr(&ctx, tiledShape);
    const VPU::DistributionInfo tiledDistInfo = VPU::DistributionInfo::getClassFromAttr(tiledDistrAttr);
    const auto tiledOutputType = outputType.changeShape(tiledShape);

    VPU::TensorDistributionMap tiledDistributionMap;
    tiledDistributionMap[tiledOutputType] = tiledDistInfo;

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus = VPU::getReduceOutputBuffers(
            parsed.op, tiledBuffers, TypeAndDistributionPair{tiledOutputType, tiledDistributionMap});

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 2u);
    EXPECT_EQ(tiledBuffers[0], Byte(512));
    EXPECT_EQ(tiledBuffers[1], Byte(512));
}

// ---------------------------------------------------------------------------
// getReduceOutputBuffers — multi-cluster: DistributedTensorType overload
// ---------------------------------------------------------------------------

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_DistributedTensorType_ReduceC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    // Construct DistributedTensorType for [1,32,16,16] OVERLAPPED over H, 2 clusters.
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(status));
    ASSERT_EQ(buffers.size(), 1u);
    // After reducing C axis, per-cluster memory shape is [1,1,9,16] → 144 elements * 2 bytes = 288 bytes.
    EXPECT_EQ(buffers[0], Byte(288));

    // Tiled [1,32,7,5]: memory shapes {1,32,5,5}/{1,32,4,5} → after C reduce: max {1,1,5,5} = 50 bytes.
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistributedAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus =
            VPU::getReduceOutputBuffers(parsed.op, tiledBuffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 1u);
    EXPECT_EQ(tiledBuffers[0], Byte(50));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_DistributedTensorType_SegMulticasted_ReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kPoolReduceC);
    ASSERT_NE(parsed.op, nullptr);

    // Construct DistributedTensorType for [1,32,16,16] SEGMENTED|MULTICASTED, 2 clusters.
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeSegMulticastedAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(status));
    ASSERT_EQ(buffers.size(), 1u);
    // Memory shapes both {1,32,16,16} (replicated) → after C reduce: {1,1,16,16} → 256 * 2 = 512 bytes.
    EXPECT_EQ(buffers[0], Byte(512));

    // Tiled [1,32,7,5]: memory shapes both {1,32,7,5} → after C reduce: {1,1,7,5} = 70 bytes.
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistributedAttr = makeSegMulticastedAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus =
            VPU::getReduceOutputBuffers(parsed.op, tiledBuffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 1u);
    EXPECT_EQ(tiledBuffers[0], Byte(70));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_DistributedTensorType_ThreeResults) {
    auto parsed = parseNCEOp(&ctx, kPoolThreeResults);
    ASSERT_NE(parsed.op, nullptr);

    // Construct DistributedTensorType for [1,32,16,16] OVERLAPPED over H, 2 clusters.
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(status));
    // Two secondary results → two buffer entries.
    ASSERT_EQ(buffers.size(), 2u);
    // Memory shapes {1,32,9,16} per cluster → after C reduce: {1,1,9,16} → 144 * 2 = 288 bytes each.
    EXPECT_EQ(buffers[0], Byte(288));
    EXPECT_EQ(buffers[1], Byte(288));

    // Tiled [1,32,7,5]: memory shapes {1,32,5,5}/{1,32,4,5} → after C reduce: max {1,1,5,5} = 50 bytes each.
    const auto tiledShape = Shape({1, 32, 7, 5});
    const auto tiledDistributedAttr = makeOverlappedOverHAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus =
            VPU::getReduceOutputBuffers(parsed.op, tiledBuffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 2u);
    EXPECT_EQ(tiledBuffers[0], Byte(50));
    EXPECT_EQ(tiledBuffers[1], Byte(50));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_DistributedTensorType_NCEMatMulReduceAxisC) {
    auto parsed = parseNCEOp(&ctx, kMatMulReduceC);
    ASSERT_NE(parsed.op, nullptr);

    // Construct DistributedTensorType for [3,1,32,16,16] SEGMENTED over G, 2 clusters.
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeSegmentedOverGAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{3, 1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(status));
    // Two secondary results → two buffer entries.
    ASSERT_EQ(buffers.size(), 2u);
    // Memory shapes {2,1,32,16,16}/{1,1,32,16,16} → after C reduce: max {2,1,1,16,16} → 512 * 2 = 1024 bytes each.
    EXPECT_EQ(buffers[0], Byte(1024));
    EXPECT_EQ(buffers[1], Byte(1024));

    // Tiled [2,1,32,16,16]: memory shapes both {1,1,32,16,16} → after C reduce: {1,1,1,16,16} → 512 bytes each.
    const auto tiledShape = Shape({2, 1, 32, 16, 16});
    const auto tiledDistributedAttr = makeSegmentedOverGAttr(&ctx, tiledShape);
    distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(
            distributedType.changeShapeForExplicitDistribution(tiledShape, tiledDistributedAttr));
    ASSERT_NE(distributedType, nullptr);

    SmallVector<Byte> tiledBuffers;
    const auto tiledStatus =
            VPU::getReduceOutputBuffers(parsed.op, tiledBuffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    EXPECT_TRUE(mlir::succeeded(tiledStatus));
    ASSERT_EQ(tiledBuffers.size(), 2u);
    EXPECT_EQ(tiledBuffers[0], Byte(512));
    EXPECT_EQ(tiledBuffers[1], Byte(512));
}

TEST_F(MLIR_ReduceTileUtils, GetReduceOutputBuffers_MultiCluster_DistributedTensorType_SingleResult) {
    auto parsed = parseNCEOp(&ctx, kPoolSingleResult);
    ASSERT_NE(parsed.op, nullptr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(parsed.op->getResult(0).getType());
    const auto distributedAttr = makeOverlappedOverHAttr(&ctx, outputType.getShape());
    const auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    auto distributedType =
            VPU::DistributedTensorType::get(&ctx, SmallVector<int64_t>{1, 32, 16, 16}, outputType.getElementType(),
                                            order, outputType.getMemSpace(), distributedAttr);

    SmallVector<Byte> buffers;
    const auto status =
            VPU::getReduceOutputBuffers(parsed.op, buffers, mlir::cast<vpux::NDTypeInterface>(distributedType));

    // Single-result op → success with no buffers.
    EXPECT_TRUE(mlir::succeeded(status));
    EXPECT_TRUE(buffers.empty());
}
