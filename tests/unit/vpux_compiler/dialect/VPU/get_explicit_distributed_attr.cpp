//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;

void testExplicitDistributedAttr(llvm::StringLiteral inputIR, vpux::VPU::DistributionInfoAttr expectedDistributedAttr,
                                 vpux::ShapeRef shape, mlir::MLIRContext* ctx) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto mode = expectedDistributedAttr.getMode().getValue();
    const auto numTiles = expectedDistributedAttr.getNumTiles();
    const auto numClusters = expectedDistributedAttr.getNumClusters();
    const auto alignment = expectedDistributedAttr.getAlignment();
    const auto uniformDistributedSeg = expectedDistributedAttr.getUniformDistributedSegments();

    for (auto& op : func.getOps()) {
        if (auto clusterOp = mlir::dyn_cast<vpux::VPU::ClusteredOpInterface>(op)) {
            auto overlapParams = vpux::VPU::OverlapDistributionParams();

            // HW ops & Concat will receive their overlapped params explicitly.
            if (mode == VPU::DistributionMode::OVERLAPPED &&
                (mlir::isa<vpux::VPU::NCEOpInterface>(op) || mlir::isa<VPU::ConcatOp>(op))) {
                const auto tempDistribution = VPU::DistributionInfoAttr::get(
                        ctx, expectedDistributedAttr.getMode(), numTiles, expectedDistributedAttr.getKernel(),
                        expectedDistributedAttr.getPads(), expectedDistributedAttr.getStrides(), numClusters, alignment,
                        uniformDistributedSeg, nullptr, nullptr, nullptr, nullptr, nullptr);

                const auto perClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, tempDistribution).value();
                const auto perClusterMemoryOffsets = VPU::getPerClusterMemoryShapeOffsets(shape, tempDistribution);
                const auto perClusterComputeShapes = VPU::getPerClusterComputeShapes(shape, tempDistribution);
                const auto perClusterComputeOffsets = VPU::getPerClusterComputeShapeOffsets(shape, tempDistribution);

                overlapParams.setMemoryShapes(VPU::arrayOfArrayFromShape(perClusterMemoryShapes));
                overlapParams.setMemoryOffsets(VPU::arrayOfArrayFromShape(perClusterMemoryOffsets));
                overlapParams.setComputeShapes(VPU::arrayOfArrayFromShape(perClusterComputeShapes));
                overlapParams.setComputeOffsets(VPU::arrayOfArrayFromShape(perClusterComputeOffsets));
            }

            auto numTilesArr = numTiles ? parseIntArrayAttr<int64_t>(numTiles) : SmallVector<int64_t>{};
            auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};

            auto distributedAttr = VPU::DistributionInfo::getAttrFromClass(
                    ctx, clusterOp.getExplicitDistributionInfoAttr(shape, mode, numTilesArr, numClusters.getInt(),
                                                                   alignmentArr, uniformDistributedSeg ? true : false,
                                                                   overlapParams));

            ASSERT_EQ(distributedAttr.getMemoryShapes(), expectedDistributedAttr.getMemoryShapes());
            ASSERT_EQ(distributedAttr.getMemoryOffsets(), expectedDistributedAttr.getMemoryOffsets());
            ASSERT_EQ(distributedAttr.getComputeShapes(), expectedDistributedAttr.getComputeShapes());
            ASSERT_EQ(distributedAttr.getComputeOffsets(), expectedDistributedAttr.getComputeOffsets());
        }
    }
}

using MLIR_GetExplicitDistributionInfoAttrTest = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_GetExplicitDistributionInfoAttrTest, SWOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
                %0 = VPU.Interpolate(%arg0) {
                    attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>,
                    cube_coeff = -7.500000e-01 : f64,
                    mode = <LINEAR_ONNX>,
                    nearest_mode = <ROUND_PREFER_FLOOR>,
                    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                    shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
                    initial_input_dims_attr = [1, 1, 96, 160],
                    initial_output_dims_attr = [1, 1, 192, 320],
                    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    scales_attr = [2.000000e+00, 2.000000e+00],
                    sizes_attr = [192, 320],
                    tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
                        : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
                return %0 : tensor<1x1x192x320xf16>
            }
        }
    )";
    vpux::Shape outputShape = {1, 1, 192, 320};
    vpux::Shape inputShape = {1, 1, 96, 160};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 1, 49, 160});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 47, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto overlappedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, overlappedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, inputShape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 1, 96, 320});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 96, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, outputShape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 1, 192, 320});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, outputShape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributionInfoAttrTest, HWOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
                %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
                %0 = VPU.NCE.MaxPool(%arg0, %cst_0) {
                        ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        strides = [1, 1],
                        kernel_size = [1, 1]
                    } -> tensor<1x32x112x112xf16, {order = #NHWC}>
                return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
            }
        }
    )";

    vpux::Shape shape = {1, 32, 112, 112};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1));
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 32, 56, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 56, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 32, 57, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 55, 0}});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);

        const auto overlappedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, overlappedMode, numTiles, kernel, pads, strides, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, shape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 32, 56, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 56, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 32, 112, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 32, 56, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 56, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 32, 112, 112});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(numClusters.getInt(),
                                                                         SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segDupMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributionInfoAttrTest, SparseHWSOKOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !SparseOutputType = !VPU.SparseTensor<
            data=tensor<1x256x28x28x!quant.uniform<u8:f16, 0.0085653950186336744>, {order = #NHWC}>,
            sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>>
        module @test {
            func.func @main(%arg0: tensor<1x512x28x28xf16, {order = #NHWC}>) -> !SparseOutputType {
                %weights = const.Declare tensor<256x512x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<256x512x1x1xf16, {order = #NHWC}>
                %wtable = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %weights, %wtable) {
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<
                            mode = <NOOP>,
                            clamp_low = 0 : i64, clamp_high = 255 : i64,
                            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                            fp_prelu_alpha = 1.000000e+00 : f64>,
                            rawFilterShape = [256, 512, 1, 1], strides = [1, 1]
                    } : tensor<1x512x28x28xf16, {order = #NHWC}>, tensor<256x512x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> !SparseOutputType
                return %0 : !SparseOutputType
            }
        }
    )";
    vpux::Shape shape = {1, 256, 28, 28};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 6, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 6);
    const auto alignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 48, 28, 28}, SmallVector<int64_t>{1, 48, 28, 28},
                 SmallVector<int64_t>{1, 48, 28, 28}, SmallVector<int64_t>{1, 48, 28, 28},
                 SmallVector<int64_t>{1, 48, 28, 28}, SmallVector<int64_t>{1, 16, 28, 28}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 48, 0, 0}, SmallVector<int64_t>{0, 96, 0, 0},
                 SmallVector<int64_t>{0, 144, 0, 0}, SmallVector<int64_t>{0, 192, 0, 0},
                 SmallVector<int64_t>{0, 240, 0, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, alignment, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(
                {SmallVector<int64_t>{1, 48, 28, 28}, SmallVector<int64_t>{1, 48, 28, 28},
                 SmallVector<int64_t>{1, 48, 28, 28}, SmallVector<int64_t>{1, 48, 28, 28},
                 SmallVector<int64_t>{1, 48, 28, 28}, SmallVector<int64_t>{1, 16, 28, 28}});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 48, 0, 0}, SmallVector<int64_t>{0, 96, 0, 0},
                 SmallVector<int64_t>{0, 144, 0, 0}, SmallVector<int64_t>{0, 192, 0, 0},
                 SmallVector<int64_t>{0, 240, 0, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 256, 28, 28});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(numClusters.getInt(),
                                                                         SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segDupMode, numTiles, nullptr, nullptr, nullptr, numClusters, alignment, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributionInfoAttrTest, NCEPermuteOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
        module @test attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode =
        #config.compilation_mode<DefaultHW>} {
            func.func @main(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order
            = #NHWC}> {

                %0 = VPU.NCE.Permute(%arg0) {
                    dstElemType = !qElemType,
                    dstOrder = #NHWC,
                    expandedChannels = 4 : i64,
                    ppe = #VPU.PPEInt<
                        mode = <NOOP>,
                        clamp_low = -2147483648 : i64,
                        clamp_high = 2147483647 : i64,
                        lrelu_mult = 1 : i64,
                        lrelu_shift = 0 : i64,
                        fp_prelu_alpha = 1.000000e+00 : f64
                    >
                } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

                return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
            }
        }
    )";
    vpux::Shape shape = {1, 4, 224, 224};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1));
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 4, 112, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 112, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 4, 113, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 111, 0}});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);

        const auto overlappedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, overlappedMode, numTiles, kernel, pads, strides, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, shape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 4, 112, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 112, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 4, 224, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
        const PerClusterShapesOffsetsVec expectedPerClusterComputeShapes(numClusters.getInt(),
                                                                         SmallVector<int64_t>{1, 4, 112, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 112, 0}});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryShapes(numClusters.getInt(),
                                                                        SmallVector<int64_t>{1, 4, 224, 224});
        const PerClusterShapesOffsetsVec expectedPerClusterMemoryOffsets(numClusters.getInt(),
                                                                         SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeShapes);
        const auto expectedPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterComputeOffsets);
        const auto expectedPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryShapes);
        const auto expectedPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterMemoryOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segDupMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterComputeShapesAttr, expectedPerClusterComputeOffsetsAttr,
                expectedPerClusterMemoryShapesAttr, expectedPerClusterMemoryOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}

TEST_F(MLIR_GetExplicitDistributionInfoAttrTest, ConcatOp) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x48x32x32xf16, {order = #NHWC}>,
                            %arg1: tensor<1x48x32x32xf16, {order = #NHWC}>)
                    -> tensor<1x96x32x32xf16, {order = #NHWC}> {
                %0 = VPU.Concat(%arg0, %arg1) {
                    static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]
                } : tensor<1x48x32x32xf16, {order = #NHWC}>,
                    tensor<1x48x32x32xf16, {order = #NHWC}>
                        -> tensor<1x96x32x32xf16, {order = #NHWC}>

                return %0 : tensor<1x96x32x32xf16, {order = #NHWC}>
            }
        }
    )";
    vpux::Shape shape = {1, 96, 32, 32};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
        const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                                getIntAttr(&ctx, 1));
        const auto strides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));

        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 96, 17, 32});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto overlappedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, overlappedMode, numTiles, kernel, pads, strides, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        testExplicitDistributedAttr(inputIR, overlappedDistributedAtr, shape, &ctx);
    }

    {
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 96, 16, 32});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 16, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto segmentedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, segmentedDistributedAtr, shape, &ctx);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 96, 32, 32});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributionInfoAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);
        testExplicitDistributedAttr(inputIR, duplicatedDistributedAtr, shape, &ctx);
    }
}
