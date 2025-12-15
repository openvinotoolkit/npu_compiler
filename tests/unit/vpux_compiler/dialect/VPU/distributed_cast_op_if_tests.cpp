//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
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
                expectedPerClusterOffsets, expectedPerClusterShapes, expectedPerClusterOffsets, {});

        getInputTypeAndTest(overlappedDistribution);
    }

    {
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto duplicatedDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {}, {}, expectedPerClusterShapes,
                expectedPerClusterOffsets, expectedPerClusterShapes, expectedPerClusterOffsets, {});

        getInputTypeAndTest(duplicatedDistribution);
    }
}

TEST_F(MLIR_DistributedCastOpInterfaceTest, AffineReshape) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
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

    // Only DUPLICATED distribution is legal for AffineReshape
    auto getInputTypeAndTest = [&](VPU::DistributionInfo& inDistribution, VPU::DistributionInfo& outDistribution) {
        const auto inputType = vpux::getTensorType(shape, fp16Type, inDimsOrder, memSpace);

        if (outDistribution.getDistributionMode() == VPU::DistributionMode::NONE) {
            testDistributedAttr(inputIR, inputType, inDistribution, nullptr, outDistribution, &ctx);
            return;
        }

        const auto outputType = vpux::getTensorType(newShape, fp16Type, outDimsOrder, memSpace);
        testDistributedAttr(inputIR, inputType, inDistribution, outputType, outDistribution, &ctx);
    };

    {
        const auto numTiles = SmallVector<int64_t>({1, 1, 2, 1});
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        auto segDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, numTiles, {}, {}, {},
                                                     numClusters, {}, {}, inPerClusterShapes, inPerClusterOffsets,
                                                     inPerClusterShapes, inPerClusterOffsets, {});
        VPU::DistributionInfo emptyDistribution{};

        getInputTypeAndTest(segDistribution, emptyDistribution);
    }

    {
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto inDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {},
                                                    {}, inPerClusterShapes, inPerClusterOffsets, inPerClusterShapes,
                                                    inPerClusterOffsets, {});

        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters, SmallVector<int64_t>{1, 128, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto duplicatedDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::DUPLICATED, {}, {}, {}, {}, numClusters, {}, {}, expectedPerClusterShapes,
                expectedPerClusterOffsets, expectedPerClusterShapes, expectedPerClusterOffsets, {});

        getInputTypeAndTest(inDistribution, duplicatedDistribution);
    }

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
                inPerMemoryClusterOffsets, {});
        VPU::DistributionInfo emptyDistribution{};
        getInputTypeAndTest(inDistribution, emptyDistribution);
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
                                                     inPerClusterShapes, inPerClusterOffsets, {});

        const auto outNumTiles = SmallVector<int64_t>({1, 2, 1, 1});
        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters, SmallVector<int64_t>{1, 20, 1, 256});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 20, 0, 0}});
        auto segDistribution = VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, outNumTiles, {}, {}, {},
                                                     numClusters, {}, {}, outPerClusterShapes, outPerClusterOffsets,
                                                     outPerClusterShapes, outPerClusterOffsets, {});

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
                                      inPerClusterMemoryOffsets, {});
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
                inPerMemoryClusterOffsets, {});

        const auto outNumTiles = SmallVector<int64_t>({1, 1, 1, 2});

        const PerClusterShapesOffsetsVec outPerClusterComputeShapes(numClusters, SmallVector<int64_t>{1, 40, 1, 64});
        const PerClusterShapesOffsetsVec outPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 64}});
        const PerClusterShapesOffsetsVec outPerMemoryClusterShapes(numClusters, SmallVector<int64_t>{1, 40, 1, 128});
        const PerClusterShapesOffsetsVec outPerMemoryClusterOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        auto outDistribution = VPU::DistributionInfo(
                VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED, outNumTiles, {}, {}, {},
                numClusters, {}, {}, outPerClusterComputeShapes, outPerClusterComputeOffsets, outPerMemoryClusterShapes,
                outPerMemoryClusterOffsets, {});

        getTypesAndTest(inDistribution, outDistribution);
    }
}
