//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/init/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;
constexpr StringRef CMX_NAME = "CMX_NN";

void testDistributedAttr(llvm::StringLiteral inputIR, vpux::NDTypeInterface inputType,
                         VPU::DistributionInfo& inputDistribution, vpux::NDTypeInterface expectedType,
                         VPU::DistributionInfo& expectedDistribution, mlir::MLIRContext* ctx) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto checkElementType = [&](mlir::Type res, mlir::Type exp) {
        auto resQuant = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(res);
        auto expQuant = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(exp);
        if (resQuant != nullptr && expQuant != nullptr) {
            EXPECT_TRUE(resQuant.getExpressedType() == expQuant.getExpressedType());
            EXPECT_EQ(resQuant.getZeroPoint(), expQuant.getZeroPoint());
            EXPECT_EQ(resQuant.getScale(), expQuant.getScale());
            return;
        }

        ASSERT_EQ(res, exp);
    };
    for (auto& op : func.getOps()) {
        if (auto distributedCastOp = mlir::dyn_cast<vpux::VPU::DistributedCastOpInterface>(op)) {
            auto resDistributedTypeWithDistribution =
                    distributedCastOp.inferCastedTypeAndDistribution(inputType, inputDistribution);

            // Could not infer output distribution
            if (expectedType == nullptr) {
                ASSERT_EQ(mlir::failed(resDistributedTypeWithDistribution), true);
            } else {
                ASSERT_EQ(mlir::succeeded(resDistributedTypeWithDistribution), true);

                auto resultType = mlir::cast<vpux::NDTypeInterface>(resDistributedTypeWithDistribution.value().first);
                auto resultDistribution = resDistributedTypeWithDistribution.value().second;
                EXPECT_EQ(resultType.getShape(), expectedType.getShape());
                EXPECT_EQ(resultType.getDimsOrder(), expectedType.getDimsOrder());
                checkElementType(resultType.getElementType(), expectedType.getElementType());
                EXPECT_EQ(resultType.getMemSpace(), expectedType.getMemSpace());
                EXPECT_EQ(resultDistribution, expectedDistribution);
            }
        }
    }
}

using MLIR_DistributedCastOpInterfaceTest = vpux::VPU::arch37xx::UnitTest;
using MLIR_DistributedCastOpInterfaceArch50XXTest = vpux::VPU::arch50xx::UnitTest;

TEST_F(MLIR_DistributedCastOpInterfaceTest, QuantizeCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        !qElemType = !quant.uniform<u8:f16, 0.01:32>
        !qElemType1 = !quant.uniform<u8:f16, 0.02:64>
        module @test {
            func.func @main(%arg0: tensor<1x128x16x16x!qElemType>) -> tensor<1x128x16x16x!qElemType1> {
                %0 = VPU.QuantizeCast(%arg0) {dstElemType = !qElemType1}
                        : tensor<1x128x16x16x!qElemType> -> tensor<1x128x16x16x!qElemType1>
                return %0 : tensor<1x128x16x16x!qElemType1>
            }
        }
    )";
    const vpux::Shape shape = {1, 128, 16, 16};

    const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
    const auto numClusters = 2;
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto dimsOrder = DimsOrder::NCHW;
    const auto inQuantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                                                                    0.01, 32, 0, 255);
    const auto outQuantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx),
                                                                     mlir::Float16Type::get(&ctx), 0.02, 64, 0, 255);

    // Distribution does not change between input and output for QuantizeCast
    auto getInputTypeAndTest = [&](VPU::DistributionInfo& distribution) {
        const auto inputType = vpux::getTensorType(shape, inQuantType, dimsOrder, memSpace);
        const auto outputType = mlir::cast<NDTypeInterface>(inputType).changeElemType(outQuantType);

        testDistributedAttr(inputIR, inputType, distribution, outputType, distribution, &ctx);
    };

    {
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto overlappedDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {}, numClusters, {}, {}, expectedPerClusterShapes,
                expectedPerClusterOffsets, expectedPerClusterShapes, expectedPerClusterOffsets, {}, std::nullopt);

        getInputTypeAndTest(overlappedDistribution);
    }

    {
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto duplicatedDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {}, {}, expectedPerClusterShapes,
                expectedPerClusterOffsets, expectedPerClusterShapes, expectedPerClusterOffsets, {}, std::nullopt);

        getInputTypeAndTest(duplicatedDistribution);
    }
}

TEST_F(MLIR_DistributedCastOpInterfaceArch50XXTest, AffineReshape) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test attributes {config.platform = #config.platform<NPU5010>} {
            func.func @main(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x128x1x256xf16, {order = #NWCH}> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 256]}
                        : tensor<1x128x16x16xf16, {order = #NHWC}> -> tensor<1x128x1x256xf16, {order = #NWCH}>
                return %0 : tensor<1x128x1x256xf16, {order = #NWCH}>
            }
        }
    )";
    const vpux::Shape shape = {1, 128, 16, 16};
    const vpux::Shape newShape = {1, 128, 1, 256};

    const auto numClusters = 2;
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto inDimsOrder = DimsOrder::NHWC;
    const auto outDimsOrder = DimsOrder::NWCH;
    auto fp16Type = mlir::Float16Type::get(&ctx);

    auto getInputTypeAndTest = [&](VPU::DistributionInfo& inDistribution, VPU::DistributionInfo& outDistribution) {
        const auto inputType = vpux::getTensorType(shape, fp16Type, inDimsOrder, memSpace);

        if (outDistribution.getDistributionMode() == VPU::DistributionMode::NONE) {
            testDistributedAttr(inputIR, inputType, inDistribution, nullptr, outDistribution, &ctx);
            return;
        }

        const auto outputType = vpux::getTensorType(newShape, fp16Type, outDimsOrder, memSpace);
        testDistributedAttr(inputIR, inputType, inDistribution, outputType, outDistribution, &ctx);
    };

    // SEGMENTED on H (dim 2): input H=16 maps to merged output W=256 (16*16).
    // Per-cluster input H=8 produces per-cluster output W = 8*16 = 128.
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto segDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {},
                                                     numClusters, {}, {}, inPerClusterShapes, inPerClusterOffsets,
                                                     inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        const auto outNumTiles = SmallVector<int64_t>({1, 1, 1, 2});
        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 1, 128});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 128}});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, outNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, outPerClusterShapes, outPerClusterOffsets,
                                                     outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getInputTypeAndTest(segDistribution, outDistribution);
    }

    {
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {},
                                                    {}, inPerClusterShapes, inPerClusterOffsets, inPerClusterShapes,
                                                    inPerClusterOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto duplicatedDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {}, {}, expectedPerClusterShapes,
                expectedPerClusterOffsets, expectedPerClusterShapes, expectedPerClusterOffsets, {}, std::nullopt);

        getInputTypeAndTest(inDistribution, duplicatedDistribution);
    }

    // SEGMENTED on C (dim 1): input C=128 maps via split to output C=128 (split outer) and H=1 (split inner).
    // Per-cluster input C=64 produces per-cluster output C=64 (ratio=1 since input/output C both 128).
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 16, 16}, SmallVector<int64_t>{1, 64, 16, 16}});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});

        auto segDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {},
                                                     numClusters, {}, {}, inPerClusterShapes, inPerClusterOffsets,
                                                     inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        const auto outNumTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec outPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 1, 256}, SmallVector<int64_t>{1, 64, 1, 256}});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, outNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, outPerClusterShapes, outPerClusterOffsets,
                                                     outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getInputTypeAndTest(segDistribution, outDistribution);
    }

    // OVERLAPPED on H (dim 2) with halo: input H split with 1-element halo => memory H=9, compute H=8.
    // Through merge dim_mapping [[0],[1,2],[3],[3]]: output memory W = 9*16 = 144, compute W = 8*16 = 128.
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});
        const PerClusterShapesOffsetsVec inPerClusterMemoryShapes(numClusters, SmallVector<int64_t>{1, 128, 9, 16});
        const PerClusterShapesOffsetsVec inPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        auto ovrDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {}, numClusters, {}, {}, inPerClusterComputeShapes,
                inPerClusterComputeOffsets, inPerClusterMemoryShapes, inPerClusterMemoryOffsets, {}, std::nullopt);

        const auto outNumTiles = SmallVector<int64_t>({1, 1, 1, 2});
        const PerClusterShapesOffsetsVec outPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 128, 1, 128});
        const PerClusterShapesOffsetsVec outPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 128}});
        const PerClusterShapesOffsetsVec outPerClusterMemoryShapes(numClusters, SmallVector<int64_t>{1, 128, 1, 144});
        const PerClusterShapesOffsetsVec outPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 112}});

        auto outDistribution =
                VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, outNumTiles, {}, {}, {}, numClusters, {}, {},
                                      outPerClusterComputeShapes, outPerClusterComputeOffsets,
                                      outPerClusterMemoryShapes, outPerClusterMemoryOffsets, {}, std::nullopt);

        getInputTypeAndTest(ovrDistribution, outDistribution);
    }

    // SEGMENTED|DUPLICATED (compound mode) should fail — not applicable to view-like AffineReshape
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});

        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inPerMemoryClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED, numTiles, {}, {}, {}, numClusters,
                {}, {}, inPerClusterComputeShapes, inPerClusterComputeOffsets, inPerMemoryClusterShapes,
                inPerMemoryClusterOffsets, {}, std::nullopt);
        VPU::DistributionInfo emptyDistribution{};
        getInputTypeAndTest(inDistribution, emptyDistribution);
    }

    // SEGMENTED|OVERLAPPED (compound mode) should fail — not applicable to view-like AffineReshape
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const auto memNumTiles = SmallVector<int64_t>({1, 1, 2, 1});

        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerMemoryClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {}, numClusters,
                {}, {}, inPerClusterComputeShapes, inPerClusterComputeOffsets, inPerMemoryClusterShapes,
                inPerMemoryClusterOffsets, {}, memNumTiles);
        VPU::DistributionInfo emptyDistribution{};
        getInputTypeAndTest(inDistribution, emptyDistribution);
    }
}

// Reproducer for rank-changing AffineReshape (4D→5D) with SEGMENTED distribution.
// dim_mapping [[0],[0],[1],[2,3,4]]: input dims 0,1 merge → output dim 0; input dim 3 splits → output dims 2,3,4.
// SEGMENTED on C (dim 1) with 3 clusters. Currently fails due to shape.size() != outShape.size() guard.
TEST_F(MLIR_DistributedCastOpInterfaceArch50XXTest, AffineReshape5DRankChange) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test attributes {config.platform = #config.platform<NPU5010>} {
            func.func @main(%arg0: tensor<1x256x2048x16xf16>) -> tensor<256x2048x16x1x1xf16> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [256, 2048, 16, 1, 1]}
                        : tensor<1x256x2048x16xf16> -> tensor<256x2048x16x1x1xf16>
                return %0 : tensor<256x2048x16x1x1xf16>
            }
        }
    )";
    const vpux::Shape shape = {1, 256, 2048, 16};
    const vpux::Shape newShape = {256, 2048, 16, 1, 1};

    const auto numClusters = 3;
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto inDimsOrder = DimsOrder::NCHW;
    const auto outDimsOrder = DimsOrder::GNCHW;
    auto fp16Type = mlir::Float16Type::get(&ctx);

    // SEGMENTED on C (dim 1) with 3 clusters: per-cluster C = 86, 85, 85
    // After merge (N*C → output dim 0): output per-cluster dim 0 = 1*86=86, 1*85=85, 1*85=85
    {
        const auto numTiles = SmallVector<int64_t>({1, 3, 1, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes({SmallVector<int64_t>{1, 86, 2048, 16},
                                                             SmallVector<int64_t>{1, 85, 2048, 16},
                                                             SmallVector<int64_t>{1, 85, 2048, 16}});
        const PerClusterShapesOffsetsVec inPerClusterOffsets({SmallVector<int64_t>{0, 0, 0, 0},
                                                              SmallVector<int64_t>{0, 86, 0, 0},
                                                              SmallVector<int64_t>{0, 171, 0, 0}});

        auto segDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {},
                                                     numClusters, {}, {}, inPerClusterShapes, inPerClusterOffsets,
                                                     inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        const auto outNumTiles = SmallVector<int64_t>({3, 1, 1, 1, 1});
        const PerClusterShapesOffsetsVec outPerClusterShapes({SmallVector<int64_t>{86, 2048, 16, 1, 1},
                                                              SmallVector<int64_t>{85, 2048, 16, 1, 1},
                                                              SmallVector<int64_t>{85, 2048, 16, 1, 1}});
        const PerClusterShapesOffsetsVec outPerClusterOffsets({SmallVector<int64_t>{0, 0, 0, 0, 0},
                                                               SmallVector<int64_t>{86, 0, 0, 0, 0},
                                                               SmallVector<int64_t>{171, 0, 0, 0, 0}});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, outNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, outPerClusterShapes, outPerClusterOffsets,
                                                     outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        const auto inputType = vpux::getTensorType(shape, fp16Type, inDimsOrder, memSpace);
        const auto outputType = vpux::getTensorType(newShape, fp16Type, outDimsOrder, memSpace);
        testDistributedAttr(inputIR, inputType, segDistribution, outputType, outDistribution, &ctx);
    }
}

TEST_F(MLIR_DistributedCastOpInterfaceTest, PermuteCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
        module @test {
            func.func @main(%arg0: tensor<1x256x40x1xf16, {order = #NHWC}>) -> tensor<1x40x1x256xf16, {order = #NCHW}> {
                %0 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW}
                        : tensor<1x256x40x1xf16, {order = #NHWC}> -> tensor<1x40x1x256xf16, {order = #NCHW}>
                return %0 : tensor<1x40x1x256xf16, {order = #NCHW}>
            }
        }
    )";
    const vpux::Shape shape = {1, 256, 40, 1};
    const vpux::Shape newShape = {1, 40, 1, 256};

    const auto numClusters = 2;
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto inDimsOrder = DimsOrder::NHWC;
    const auto outDimsOrder = DimsOrder::NCHW;
    auto fp16Type = mlir::Float16Type::get(&ctx);

    auto getTypesAndTest = [&](VPU::DistributionInfo& inDistribution, VPU::DistributionInfo& outDistribution) {
        const auto inputType = vpux::getTensorType(shape, fp16Type, inDimsOrder, memSpace);

        if (outDistribution.getDistributionMode() == VPU::DistributionMode::NONE) {
            testDistributedAttr(inputIR, inputType, inDistribution, nullptr, outDistribution, &ctx);
            return;
        }

        const auto outputType = vpux::getTensorType(newShape, fp16Type, outDimsOrder, memSpace);

        testDistributedAttr(inputIR, inputType, inDistribution, outputType, outDistribution, &ctx);
    };

    {
        const auto inNumTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 256, 20, 1});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 20, 0}});

        auto ovrDistribution = VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, inNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, inPerClusterShapes, inPerClusterOffsets,
                                                     inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        const auto outNumTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 20, 1, 256});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 20, 0, 0}});
        auto segDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, outNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, outPerClusterShapes, outPerClusterOffsets,
                                                     outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getTypesAndTest(ovrDistribution, segDistribution);
    }

    {
        const auto inNumTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 256, 20, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 20, 0}});

        const PerClusterShapesOffsetsVec inPerClusterMemoryShapes(
                {SmallVector<int64_t>{1, 256, 22, 1}, SmallVector<int64_t>{1, 256, 21, 1}});
        const PerClusterShapesOffsetsVec inPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 19, 0}});

        auto ovrDistribution =
                VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, inNumTiles, {}, {}, {}, numClusters, {}, {},
                                      inPerClusterComputeShapes, inPerClusterComputeOffsets, inPerClusterMemoryShapes,
                                      inPerClusterMemoryOffsets, {}, std::nullopt);
        VPU::DistributionInfo emptyDistribution{};
        // output axis is C and Overlapped is not equivalent to Segmented => cannot infer distribution
        getTypesAndTest(ovrDistribution, emptyDistribution);
    }

    {
        const auto inNumTiles = SmallVector<int64_t>({1, 2, 1, 1});

        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 64, 40, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 40, 1});
        const PerClusterShapesOffsetsVec inPerMemoryClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED, inNumTiles, {}, {}, {},
                numClusters, {}, {}, inPerClusterComputeShapes, inPerClusterComputeOffsets, inPerMemoryClusterShapes,
                inPerMemoryClusterOffsets, {}, std::nullopt);

        const auto outNumTiles = SmallVector<int64_t>({1, 1, 1, 2});

        const PerClusterShapesOffsetsVec outPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 40, 1, 64});
        const PerClusterShapesOffsetsVec outPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 64}});
        const PerClusterShapesOffsetsVec outPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 40, 1, 128});
        const PerClusterShapesOffsetsVec outPerMemoryClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto outDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED, outNumTiles, {}, {}, {},
                numClusters, {}, {}, outPerClusterComputeShapes, outPerClusterComputeOffsets, outPerMemoryClusterShapes,
                outPerMemoryClusterOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    {
        const auto inNumTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const auto inMemoryNumTiles = SmallVector<int64_t>({1, 1, 2, 1});

        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 128, 40, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 128, 0, 0}});
        const PerClusterShapesOffsetsVec inPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 256, 20, 1});
        const PerClusterShapesOffsetsVec inPerMemoryClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 20, 0}});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, inNumTiles, {}, {}, {},
                numClusters, {}, {}, inPerClusterComputeShapes, inPerClusterComputeOffsets, inPerMemoryClusterShapes,
                inPerMemoryClusterOffsets, {}, inMemoryNumTiles);

        const auto outNumTiles = SmallVector<int64_t>({1, 1, 1, 2});
        const auto outMemoryNumTiles = SmallVector<int64_t>({1, 2, 1, 1});

        const PerClusterShapesOffsetsVec outPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 40, 1, 128});
        const PerClusterShapesOffsetsVec outPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 128}});
        const PerClusterShapesOffsetsVec outPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 20, 1, 256});
        const PerClusterShapesOffsetsVec outPerMemoryClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 20, 0, 0}});

        auto outDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, outNumTiles, {}, {}, {},
                numClusters, {}, {}, outPerClusterComputeShapes, outPerClusterComputeOffsets, outPerMemoryClusterShapes,
                outPerMemoryClusterOffsets, {}, outMemoryNumTiles);

        getTypesAndTest(inDistribution, outDistribution);
    }
}

TEST_F(MLIR_DistributedCastOpInterfaceTest, Slice) {
    // Input: [1, 128, 16, 16] -> Slice C dim 128->64 -> [1, 64, 16, 16]
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x128x16x16xf16>) -> tensor<1x64x16x16xf16> {
                %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 64, 16, 16]
                        : tensor<1x128x16x16xf16> to tensor<1x64x16x16xf16>
                return %0 : tensor<1x64x16x16xf16>
            }
        }
    )";
    const vpux::Shape inShape = {1, 128, 16, 16};
    const vpux::Shape outShape = {1, 64, 16, 16};

    const auto numClusters = 2;
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto dimsOrder = DimsOrder::NCHW;
    auto fp16Type = mlir::Float16Type::get(&ctx);

    auto getTypesAndTest = [&](VPU::DistributionInfo& inDistribution, VPU::DistributionInfo& outDistribution) {
        const auto inputType = vpux::getTensorType(inShape, fp16Type, dimsOrder, memSpace);
        if (outDistribution.getDistributionMode() == VPU::DistributionMode::NONE) {
            testDistributedAttr(inputIR, inputType, inDistribution, nullptr, outDistribution, &ctx);
            return;
        }
        const auto outputType = vpux::getTensorType(outShape, fp16Type, dimsOrder, memSpace);
        testDistributedAttr(inputIR, inputType, inDistribution, outputType, outDistribution, &ctx);
    };

    // Case 1: DUPLICATED - distribution passes through, C dim in per-cluster shapes adjusted
    {
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {},
                                                    {}, inPerClusterShapes, inPerClusterOffsets, inPerClusterShapes,
                                                    inPerClusterOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {},
                                                     {}, outPerClusterShapes, outPerClusterOffsets, outPerClusterShapes,
                                                     outPerClusterOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 2: SEGMENTED on H (not the sliced dim) - can propagate, C dim adjusted
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                                    {}, {}, inPerClusterShapes, inPerClusterOffsets, inPerClusterShapes,
                                                    inPerClusterOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 8, 16});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {},
                                                     numClusters, {}, {}, outPerClusterShapes, outPerClusterOffsets,
                                                     outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 3: SEGMENTED on C (the sliced dim) - cannot propagate, return failure
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                                    {}, {}, inPerClusterShapes, inPerClusterOffsets, inPerClusterShapes,
                                                    inPerClusterOffsets, {}, std::nullopt);
        VPU::DistributionInfo emptyDistribution{};

        getTypesAndTest(inDistribution, emptyDistribution);
    }

    // Case 4: OVERLAPPED on H (not the sliced dim) - can propagate, C dim adjusted in compute and memory shapes
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});

        const PerClusterShapesOffsetsVec inComputeShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});
        const PerClusterShapesOffsetsVec inMemoryShapes(
                {SmallVector<int64_t>{1, 128, 9, 16}, SmallVector<int64_t>{1, 128, 9, 16}});
        const PerClusterShapesOffsetsVec inMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {},
                                                    numClusters, {}, {}, inComputeShapes, inComputeOffsets,
                                                    inMemoryShapes, inMemoryOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec outComputeShapes(numClusters, SmallVector<int64_t>{1, 64, 8, 16});
        const PerClusterShapesOffsetsVec outComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});
        const PerClusterShapesOffsetsVec outMemoryShapes(
                {SmallVector<int64_t>{1, 64, 9, 16}, SmallVector<int64_t>{1, 64, 9, 16}});
        const PerClusterShapesOffsetsVec outMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {},
                                                     numClusters, {}, {}, outComputeShapes, outComputeOffsets,
                                                     outMemoryShapes, outMemoryOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 5: OVERLAPPED on C (the sliced dim) - cannot propagate, return failure
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec inComputeShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec inComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {},
                                                    numClusters, {}, {}, inComputeShapes, inComputeOffsets,
                                                    inComputeShapes, inComputeOffsets, {}, std::nullopt);
        VPU::DistributionInfo emptyDistribution{};

        getTypesAndTest(inDistribution, emptyDistribution);
    }

    // Case 6: SEGMENTED|DUPLICATED, SEGMENTED on C (the sliced dim) → DUPLICATED
    // The SEGMENTED axis is being sliced, but memory is DUPLICATED (each cluster holds the full
    // tensor), so propagation succeeds and the SEGMENTED aspect is dropped.
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec inComputeShapes(
                {SmallVector<int64_t>{1, 64, 16, 16}, SmallVector<int64_t>{1, 64, 16, 16}});
        const PerClusterShapesOffsetsVec inComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inMemoryShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inMemoryOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED, numTiles, {}, {}, {}, numClusters,
                {}, {}, inComputeShapes, inComputeOffsets, inMemoryShapes, inMemoryOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec outZeroOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {},
                                                     {}, outPerClusterShapes, outZeroOffsets, outPerClusterShapes,
                                                     outZeroOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 7: SEGMENTED|MULTICASTED, SEGMENTED on C (the sliced dim) → DUPLICATED
    // MULTICASTED is treated like DUPLICATED: memory is broadcast, so output is plain DUPLICATED.
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec inComputeShapes(
                {SmallVector<int64_t>{1, 64, 16, 16}, SmallVector<int64_t>{1, 64, 16, 16}});
        const PerClusterShapesOffsetsVec inComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inMemoryShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inMemoryOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution =
                VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED, numTiles,
                                      {}, {}, {}, numClusters, {}, {}, inComputeShapes, inComputeOffsets,
                                      inMemoryShapes, inMemoryOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec outZeroOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {},
                                                     {}, outPerClusterShapes, outZeroOffsets, outPerClusterShapes,
                                                     outZeroOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 8: SEGMENTED|OVERLAPPED, SEGMENTED on C, OVERLAPPED on H; slice reduces C (not the
    // memory-visible axis) → pure OVERLAPPED with memory_num_tiles promoted to num_tiles.
    // H halos in memory shapes are preserved; only the C dim is narrowed.
    {
        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const auto memNumTiles = SmallVector<int64_t>({1, 1, 2, 1});

        const PerClusterShapesOffsetsVec inComputeShapes(
                {SmallVector<int64_t>{1, 64, 16, 16}, SmallVector<int64_t>{1, 64, 16, 16}});
        const PerClusterShapesOffsetsVec inComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        // Memory is OVERLAPPED on H: each cluster stores 8 rows plus 1 halo row
        const PerClusterShapesOffsetsVec inMemoryShapes(numClusters, SmallVector<int64_t>{1, 128, 9, 16});
        const PerClusterShapesOffsetsVec inMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {}, numClusters,
                {}, {}, inComputeShapes, inComputeOffsets, inMemoryShapes, inMemoryOffsets, {}, memNumTiles);

        // Expected: OVERLAPPED with num_tiles = former memory_num_tiles; C dim sliced 128→64,
        // H halo structure (9 rows, offset 7 for cluster 1) is preserved unchanged.
        const PerClusterShapesOffsetsVec outMemoryShapes(numClusters, SmallVector<int64_t>{1, 64, 9, 16});
        const PerClusterShapesOffsetsVec outMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        auto outDistribution = VPU::DistributionInfo(VPU::DistributionMode::OVERLAPPED, memNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, outMemoryShapes, outMemoryOffsets,
                                                     outMemoryShapes, outMemoryOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 9: SEGMENTED|OVERLAPPED, but slice reduces H (the OVERLAPPED memory axis) → failure
    {
        // Different IR: slice H dim 16→8 instead of C
        constexpr llvm::StringLiteral inputIRSliceH = R"(
            module @test {
                func.func @main(%arg0: tensor<1x128x16x16xf16>) -> tensor<1x128x8x16xf16> {
                    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 8, 16]
                            : tensor<1x128x16x16xf16> to tensor<1x128x8x16xf16>
                    return %0 : tensor<1x128x8x16xf16>
                }
            }
        )";

        const auto numTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const auto memNumTiles = SmallVector<int64_t>({1, 1, 2, 1});

        const PerClusterShapesOffsetsVec inComputeShapes(
                {SmallVector<int64_t>{1, 64, 16, 16}, SmallVector<int64_t>{1, 64, 16, 16}});
        const PerClusterShapesOffsetsVec inComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inMemoryShapes(numClusters, SmallVector<int64_t>{1, 128, 9, 16});
        const PerClusterShapesOffsetsVec inMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        auto inDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, numTiles, {}, {}, {}, numClusters,
                {}, {}, inComputeShapes, inComputeOffsets, inMemoryShapes, inMemoryOffsets, {}, memNumTiles);

        const auto hSliceInputType = vpux::getTensorType(inShape, fp16Type, dimsOrder, memSpace);
        VPU::DistributionInfo emptyDistribution{};
        // sliceDims = {H}, memTilingDim = H → cannot propagate
        testDistributedAttr(inputIRSliceH, hSliceInputType, inDistribution, nullptr, emptyDistribution, &ctx);
    }

    // Case 10: explicit SEGMENTED on H, alignment on C (sliced dim) — alignment scales down.
    // Input C=128, slice to C=64, alignment=[1,16,1,1]: scaleFactor=2, 16/2=8 → [1,8,1,1].
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const auto alignment = SmallVector<int64_t>({1, 16, 1, 1});

        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                                    alignment, {}, inPerClusterShapes, inPerClusterOffsets,
                                                    inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        const auto expectedAlignment = SmallVector<int64_t>({1, 8, 1, 1});
        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 8, 16});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto outDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters, expectedAlignment, {},
                outPerClusterShapes, outPerClusterOffsets, outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 11: explicit SEGMENTED on H, alignment on C — alignment dropped (not evenly divisible).
    // Input C=128, slice to C=64, alignment=[1,9,1,1]: scaleFactor=2, 9%2≠0 → alignment dropped.
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const auto alignment = SmallVector<int64_t>({1, 9, 1, 1});

        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                                    alignment, {}, inPerClusterShapes, inPerClusterOffsets,
                                                    inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        // Alignment dropped: empty.
        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 8, 16});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto outDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters, /*alignment=*/{}, {},
                outPerClusterShapes, outPerClusterOffsets, outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 12: explicit SEGMENTED on H, alignment on H (non-sliced dim) — alignment unchanged.
    // Input C=128, slice to C=64, alignment=[1,1,4,1]: H not sliced → alignment unchanged.
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const auto alignment = SmallVector<int64_t>({1, 1, 4, 1});

        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                                    alignment, {}, inPerClusterShapes, inPerClusterOffsets,
                                                    inPerClusterShapes, inPerClusterOffsets, {}, std::nullopt);

        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 64, 8, 16});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto outDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters, alignment, {}, outPerClusterShapes,
                outPerClusterOffsets, outPerClusterShapes, outPerClusterOffsets, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }

    // Case 13: non-explicit SEGMENTED on H, alignment on C (sliced dim) — alignment scales down.
    // Exercises the !hasExplicit code path in inferCastedTypeAndDistribution.
    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const auto alignment = SmallVector<int64_t>({1, 16, 1, 1});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                                    alignment, {}, /*computeShapes=*/{}, /*computeOffsets=*/{},
                                                    /*memoryShapes=*/{}, /*memoryOffsets=*/{}, {}, std::nullopt);

        const auto expectedAlignment = SmallVector<int64_t>({1, 8, 1, 1});
        auto outDistribution =
                VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {}, numClusters,
                                      expectedAlignment, {}, /*computeShapes=*/{}, /*computeOffsets=*/{},
                                      /*memoryShapes=*/{}, /*memoryOffsets=*/{}, {}, std::nullopt);

        getTypesAndTest(inDistribution, outDistribution);
    }
}
