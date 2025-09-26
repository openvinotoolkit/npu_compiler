//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

struct OverlappedDistributionTestParams {
    llvm::StringLiteral inputIR;
    SmallVector<int64_t> numTiles;
    SmallVector<SmallVector<int64_t>> memoryShapes;
    SmallVector<SmallVector<int64_t>> memoryOffsets;
};

class GetOverlapDistributionParamsTests : public testing::TestWithParam<OverlappedDistributionTestParams> {};

TEST_P(GetOverlapDistributionParamsTests, GetMemoryViewFromProducerConsumers) {
    const auto params = GetParam();
    const llvm::StringLiteral inputIR = params.inputIR;
    const auto numTiles = params.numTiles;
    const auto expectedMemoryShapes = params.memoryShapes;
    const auto expectedMemoryOffsets = params.memoryOffsets;

    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::config::ArchKind::NPU37XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numClusters = numTiles[VPU::getDistributedTilingAxis(numTiles)];

    VPU::ClusteredOpInterface producer = nullptr;
    SmallVector<VPU::ClusteredOpInterface> consumers = {};

    func.walk([&](VPU::ClusteredOpInterface op) {
        // producer
        if (mlir::isa<VPU::NCEAveragePoolOp>(op)) {
            producer = op;
            return;
        }

        consumers.push_back(op);
    });

    const auto resOverlapParams = VPU::getOverlappedDistributionParameters(
            mlir::cast<vpux::NDTypeInterface>(producer->getResult(0).getType()), consumers, numClusters, numTiles,
            true);

    EXPECT_EQ(SmallVector<SmallVector<int64_t>>(resOverlapParams.getMemoryShapes()), expectedMemoryShapes);
    EXPECT_EQ(SmallVector<SmallVector<int64_t>>(resOverlapParams.getMemoryOffsets()), expectedMemoryOffsets);
}

// clang-format off

llvm::StringLiteral poolwith2ConvConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x28x27xf16, {order = #NHWC}>)
          -> (tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<144x144x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x144x28x27xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    rawFilterShape = [144, 144, 3, 3],
                    strides = [1, 1]}
                        : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x28x27xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                    rawFilterShape = [144, 144, 5, 5],
                    strides = [1, 1]}
                        : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<144x144x5x5xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x28x27xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>
        }
})";

llvm::StringLiteral poolwith3ConvConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x28x27xf16, {order = #NHWC}>)
          -> (tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<144x144x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x5x5xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<144x144x7x9xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x7x9xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x144x28x27xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    rawFilterShape = [144, 144, 3, 3],
                    strides = [1, 1]}
                        : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x28x27xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                    rawFilterShape = [144, 144, 5, 5],
                    strides = [1, 1]}
                        : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<144x144x5x5xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x28x27xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%0, %w2, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 4 : i64, right = 4 : i64, top = 3 : i64, bottom = 3 : i64>,
                    rawFilterShape = [144, 144, 7, 9],
                    strides = [1, 1]}
                        : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<144x144x7x9xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x28x27xf16, {order = #NHWC}>

            return %1, %2, %3 : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>
        }
})";

std::vector<OverlappedDistributionTestParams> consumerUnionIncludesProducerParams = {
    {
        poolwith2ConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 9, 27}, {1, 144, 11, 27}, {1, 144, 11, 27}, {1, 144, 9, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 5, 0}, {0, 0, 12, 0}, {0, 0, 19, 0}}
    },
    {
        poolwith2ConvConsumers, /*numTiles=*/{1, 1, 1, 4},
        /*memoryShapes=*/{{1, 144, 28, 9}, {1, 144, 28, 11}, {1, 144, 28, 11}, {1, 144, 28, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 5}, {0, 0, 0, 12}, {0, 0, 0, 19}}
    },
    {
        poolwith3ConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 10, 27}, {1, 144, 13, 27}, {1, 144, 13, 27}, {1, 144, 10, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 4, 0}, {0, 0, 11, 0}, {0, 0, 18, 0}}
    },
    {
        poolwith3ConvConsumers, /*numTiles=*/{1, 1, 1, 4},
        /*memoryShapes=*/{{1, 144, 28, 11}, {1, 144, 28, 15}, {1, 144, 28, 15}, {1, 144, 28, 10}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 3}, {0, 0, 0, 10}, {0, 0, 0, 17}}
    }
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(ConsumerUnionIncludesProducer, GetOverlapDistributionParamsTests,
                         testing::ValuesIn(consumerUnionIncludesProducerParams));

// clang-format off

llvm::StringLiteral poolwith2NCEInterpConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x96x20x20xf16, {order = #NHWC}>)
          -> (tensor<1x96x40x40xf16, {order = #NHWC}>, tensor<1x96x60x60xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<96x96x2x2xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<96x96x2x2xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<96x96x3x3xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

            %sparseMap0 = const.Declare tensor<1x96x41x41xi1> = dense<1> : tensor<1x96x41x41xi1>
            %sparseMap1 = const.Declare tensor<1x96x62x62xi1> = dense<1> : tensor<1x96x62x62xi1>

            %storageElement0 = VPU.StorageElementTable {
                dataElemType = i32, seDepth = 1, seSize = [96], dataShape = [1, 96, 20, 20],
                seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0]>}
                    -> tensor<1x1x41x41xi32, {order = #NHWC}>
            %storageElement1 = VPU.StorageElementTable {
                dataElemType = i32, seDepth = 1, seSize = [96], dataShape = [1, 96, 20, 20],
                seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 3.0, 3.0]>}
                    -> tensor<1x1x62x62xi32, {order = #NHWC}>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x96x20x20xf16, {order = #NHWC}>

            %input0 = VPU.GroupSparseTensor(%0, %sparseMap0, %storageElement0) {
                seAttr = #VPU.SEInterpolate<
                    mode = <BILINEAR>,
                    coordinate_transformation_mode = <ASYMMETRIC>,
                    scale = [1.0, 1.0, 2.0, 2.0]>
            } -> !VPU.SparseTensor<data=tensor<1x96x20x20xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x41x41xi1>,
                           storage_element_table=tensor<1x1x41x41xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0]>>

            %1 = VPU.NCE.Interpolate(%input0, %w0, %weightsTable) {
                rawFilterShape = [96, 96, 2, 2],
                strides = [1, 1],
                mode = #VPU.nce_interpolate_mode<BILINEAR>,
                scales_attr = [2, 2],
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x96x40x40xf16, {order = #NHWC}>

            %input1 = VPU.GroupSparseTensor(%0, %sparseMap1, %storageElement1) {
                seAttr = #VPU.SEInterpolate<
                    mode = <BILINEAR>,
                    coordinate_transformation_mode = <ASYMMETRIC>,
                    scale = [1.0, 1.0, 3.0, 3.0]>
            } -> !VPU.SparseTensor<data=tensor<1x96x20x20xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x62x62xi1>,
                           storage_element_table=tensor<1x1x62x62xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 3.0, 3.0]>>

            %2 = VPU.NCE.Interpolate(%input1, %w1, %weightsTable) {
                rawFilterShape = [96, 96, 3, 3],
                strides = [1, 1],
                mode = #VPU.nce_interpolate_mode<BILINEAR>,
                scales_attr = [3, 3],
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x96x60x60xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x96x40x40xf16, {order = #NHWC}>, tensor<1x96x60x60xf16, {order = #NHWC}>
        }
})";

llvm::StringLiteral poolWithNCEInterpAndConvConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x96x20x20xf16, {order = #NHWC}>)
          -> (tensor<1x96x40x40xf16, {order = #NHWC}>, tensor<1x96x20x20xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<96x96x2x2xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<96x96x2x2xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<96x96x3x3xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

            %sparseMap0 = const.Declare tensor<1x96x41x41xi1> = dense<1> : tensor<1x96x41x41xi1>
            %storageElement0 = VPU.StorageElementTable {
                dataElemType = i32, seDepth = 1, seSize = [96], dataShape = [1, 96, 20, 20],
                seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0]>}
                    -> tensor<1x1x41x41xi32, {order = #NHWC}>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x96x20x20xf16, {order = #NHWC}>

            %input0 = VPU.GroupSparseTensor(%0, %sparseMap0, %storageElement0) {
                seAttr = #VPU.SEInterpolate<
                    mode = <BILINEAR>,
                    coordinate_transformation_mode = <ASYMMETRIC>,
                    scale = [1.0, 1.0, 2.0, 2.0]>
            } -> !VPU.SparseTensor<data=tensor<1x96x20x20xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x41x41xi1>,
                           storage_element_table=tensor<1x1x41x41xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0]>>

            %1 = VPU.NCE.Interpolate(%input0, %w0, %weightsTable) {
                rawFilterShape = [96, 96, 2, 2],
                strides = [1, 1],
                mode = #VPU.nce_interpolate_mode<BILINEAR>,
                scales_attr = [2, 2],
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x96x40x40xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [96, 96, 3, 3],
                strides = [1, 1]}
                    : tensor<1x96x20x20xf16, {order = #NHWC}>, tensor<1x96x40x40xf16, {order = #NHWC}>, tensor<96x1x1x4xsi32> -> tensor<1x96x20x20xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x96x40x40xf16, {order = #NHWC}>, tensor<1x96x20x20xf16, {order = #NHWC}>
        }
})";

std::vector<OverlappedDistributionTestParams> distributedSeTableTensorsParams = {
    {
        poolwith2NCEInterpConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 96, 6, 20}, {1, 96, 6, 20}, {1, 96, 6, 20}, {1, 96, 5, 20}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 5, 0}, {0, 0, 10, 0}, {0, 0, 15, 0}}
    },
    {
        poolwith2NCEInterpConsumers, /*numTiles=*/{1, 1, 1, 4},
        /*memoryShapes=*/{{1, 96, 20, 6}, {1, 96, 20, 6}, {1, 96, 20, 6}, {1, 96, 20, 5}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 5}, {0, 0, 0, 10}, {0, 0, 0, 15}}
    },
    {
        poolWithNCEInterpAndConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 96, 6, 20}, {1, 96, 7, 20}, {1, 96, 7, 20}, {1, 96, 6, 20}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 4, 0}, {0, 0, 9, 0}, {0, 0, 14, 0}}
    },
    {
        poolWithNCEInterpAndConvConsumers, /*numTiles=*/{1, 1, 1, 4},
        /*memoryShapes=*/{{1, 96, 20, 6}, {1, 96, 20, 7}, {1, 96, 20, 7}, {1, 96, 20, 6}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 4}, {0, 0, 0, 9}, {0, 0, 0, 14}}
    }
};

// clang-format on

// TODO: Enable again after E#112803 is implemented
INSTANTIATE_TEST_SUITE_P(DISABLED_DistributedTensorWithSETable, GetOverlapDistributionParamsTests,
                         testing::ValuesIn(distributedSeTableTensorsParams));

// clang-format off

llvm::StringLiteral convConsumersSameKernelDiffStrides = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 2 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x32x28x27xf16, {order = #NHWC}>)
          -> (tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x28x27xf16, {order = #NHWC}>) {
            %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}>
                      = dense<1.0> : tensor<32x32x1x1xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x32x28x27xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %weights, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [32, 32, 1, 1],
                    strides = [2, 2]}
                        : tensor<1x32x28x27xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x14x14xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %weights, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [32, 32, 1, 1],
                    strides = [1, 1]}
                        : tensor<1x32x28x27xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x28x27xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x28x27xf16, {order = #NHWC}>
        }
})";

llvm::StringLiteral consumersWithMismatchedMemoryView = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 6 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x112x111xf16, {order = #NHWC}>)
          -> (tensor<1x16x112x111xf16, {order = #NHWC}>, tensor<1x16x56x56xf16, {order = #NHWC}>, tensor<1x16x56x55xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x112x111xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 1, 1],
                    strides = [1, 1]}
                        : tensor<1x16x112x111xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x112x111xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 1, 1],
                    strides = [2, 2]}
                        : tensor<1x16x112x111xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x56x56xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%0, %w2, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 3, 3],
                    strides = [2, 2]}
                        : tensor<1x16x112x111xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x56x55xf16, {order = #NHWC}>

            return %1, %2, %3 : tensor<1x16x112x111xf16, {order = #NHWC}>, tensor<1x16x56x56xf16, {order = #NHWC}>, tensor<1x16x56x55xf16, {order = #NHWC}>
        }
})";

std::vector<OverlappedDistributionTestParams> consumerUnionDoesNotIncludeProducerParams  = {
    {
        convConsumersSameKernelDiffStrides, /*numTiles=*/{1, 1, 2, 1},
        /*memoryShapes=*/{{1, 32, 14, 27}, {1, 32, 14, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 14, 0}}
    },
    {
        convConsumersSameKernelDiffStrides, /*numTiles=*/{1, 1, 1, 2},
        /*memoryShapes=*/{{1, 32, 28, 14}, {1, 32, 28, 13}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 14}}
    },
    {
        consumersWithMismatchedMemoryView, /*numTiles=*/{1, 1, 6, 1},
        /*memoryShapes=*/{{1, 16, 20, 111}, {1, 16, 21, 111}, {1, 16, 20, 111}, {1, 16, 19, 111}, {1, 16, 19, 111}, {1, 16, 19, 111}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 19, 0}, {0, 0, 38, 0}, {0, 0, 57, 0}, {0, 0, 75, 0}, {0, 0, 93, 0}}
    },
    {
        consumersWithMismatchedMemoryView, /*numTiles=*/{1, 1, 1, 6},
        /*memoryShapes=*/{{1, 16, 112, 20}, {1, 16, 112, 20}, {1, 16, 112, 20}, {1, 16, 112, 20}, {1, 16, 112, 20}, {1, 16, 112, 20}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 19}, {0, 0, 0, 37}, {0, 0, 0, 55}, {0, 0, 0, 73}, {0, 0, 0, 91}}
    }
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(ConsumerUnionDoesNotIncludeProducer, GetOverlapDistributionParamsTests,
                         testing::ValuesIn(consumerUnionDoesNotIncludeProducerParams));

// clang-format off

llvm::StringLiteral consumerNotSOHOrWCompatible = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 3 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x2x2xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<16x16x7x7xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x7x7xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    rawFilterShape = [16, 16, 3, 3],
                    strides = [1, 1]}
                        : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 1, 1],
                    strides = [2, 2]}
                        : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x4xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%0, %w2, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 7, 7],
                    strides = [1, 1]}
                        : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x7x7xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x2x2xf16, {order = #NHWC}>

            return %1, %2, %3 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x2x2xf16, {order = #NHWC}>
        }
})";

llvm::StringLiteral noCompatibleConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>)
          -> (tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<1x144x16x16xf16, {order = #NHWC}>) {
            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            %1 = VPU.MVN(%0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                normalize_variance = true}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            %2 = VPU.HSwish(%0)
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<1x144x16x16xf16, {order = #NHWC}>
        }
})";

llvm::StringLiteral noConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>)
          -> tensor<1x144x16x16xf16, {order = #NHWC}> {
            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            return %0 : tensor<1x144x16x16xf16, {order = #NHWC}>
        }
})";

llvm::StringLiteral mixedConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 2 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x96x160xf16, {order = #NHWC}>)
          -> (tensor<1x16x192x320xf16, {order = #NHWC}>, tensor<1x16x96x160xf16, {order = #NHWC}>) {
            %weights = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x96x160xf16, {order = #NHWC}>
            %1 = VPU.Interpolate(%0) {
                    attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
                    shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>,
                    nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false,
                    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                    cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
                    initial_input_dims_attr = [1, 16, 96, 160],
                    initial_output_dims_attr = [1, 16, 192, 320],
                    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    scales_attr = [2.000000e+00, 2.000000e+00],
                    sizes_attr = [192, 320],
                    tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]}
                  : tensor<1x16x96x160xf16, {order = #NHWC}>
                  -> tensor<1x16x192x320xf16, {order = #NHWC}>
            %2 = VPU.NCE.Convolution(%0, %weights, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                    rawFilterShape = [16, 16, 5, 5],
                    strides = [1, 1]}
                        : tensor<1x16x96x160xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x96x160xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x16x192x320xf16, {order = #NHWC}>, tensor<1x16x96x160xf16, {order = #NHWC}>
        }
})";

std::vector<OverlappedDistributionTestParams> incompatibleConsumerParams  = {
    {
        consumerNotSOHOrWCompatible, /*numTiles=*/{1, 1, 3, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 5, 8}, {1, 16, 3, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 5, 0}}
    },
    {
        consumerNotSOHOrWCompatible, /*numTiles=*/{1, 1, 1, 3},
        /*memoryShapes=*/{{1, 16, 8, 4}, {1, 16, 8, 5}, {1, 16, 8, 3}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 5}}
    },
    {
        noCompatibleConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 4, 16}, {1, 144, 4, 16}, {1, 144, 4, 16}, {1, 144, 4, 16}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 4, 0}, {0, 0, 8, 0}, {0, 0, 12, 0}}
    },
    {
        noConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 4, 16}, {1, 144, 4, 16}, {1, 144, 4, 16}, {1, 144, 4, 16}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 4, 0}, {0, 0, 8, 0}, {0, 0, 12, 0}}
    },
    {
        mixedConsumers, /*numTiles=*/{1, 1, 2, 1},
        /*memoryShapes=*/{{1, 16, 50, 160}, {1, 16, 50, 160}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 46, 0}}
    }
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(IncompatibleConsumers, GetOverlapDistributionParamsTests,
                         testing::ValuesIn(incompatibleConsumerParams));
