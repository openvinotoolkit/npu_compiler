//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

struct OverlappedTestParams {
    llvm::StringLiteral inputIR;
    SmallVector<int64_t> numTiles;
    SmallVector<SmallVector<int64_t>> memoryShapes;
    SmallVector<SmallVector<int64_t>> memoryOffsets;
    SmallVector<int64_t> tileShape;
};

class GetOverlapSiblingsTests : public testing::TestWithParam<llvm::StringLiteral> {};

class GetActivationOverlapTests : public testing::TestWithParam<OverlappedTestParams> {};
class GetOutputOverlapTests : public testing::TestWithParam<OverlappedTestParams> {};

namespace {
struct ProducersConsumers {
    std::set<VPU::ClusteredOpInterface> producers = {};
    std::set<VPU::ClusteredOpInterface> consumers = {};

    ProducersConsumers(const std::set<VPU::ClusteredOpInterface>& prod, const std::set<VPU::ClusteredOpInterface>& cons)
            : producers(prod), consumers(cons) {};
};

bool isConcatOverH(VPU::ConcatOp concat) {
    auto isOffsetOnH = [](mlir::ArrayAttr offset) {
        auto offsetVector = Shape(parseIntArrayAttr<int64_t>(offset));
        return offsetVector[Dims4D::Act::H] != 0;
    };

    if (concat.getStaticOffsets().has_value()) {
        const auto concatDims = concat.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();
        return llvm::any_of(concatDims, isOffsetOnH);
    } else if (concat.getPerAxis().has_value()) {
        const auto concatAxis = concat.getPerAxis().value().getAxis().getValue().getSExtValue();
        return concatAxis == Dims4D::Act::H.ind();
    }

    return false;
}

ProducersConsumers getProducerConsumersSets(mlir::func::FuncOp func, bool ignoreSEPOps = true,
                                            bool isMaxPoolProducer = false) {
    std::set<VPU::ClusteredOpInterface> consumers = {};
    std::set<VPU::ClusteredOpInterface> producers = {};
    func.walk([&](VPU::ClusteredOpInterface op) {
        // producer
        if (mlir::isa<VPU::NCEAveragePoolOp>(op)) {
            producers.emplace(op);
            return;
        }

        // not a sibling
        if (mlir::isa<VPU::NCEMaxPoolOp>(op)) {
            if (isMaxPoolProducer) {
                producers.emplace(op);
            }
            return;
        }

        // TODO: NCE.Interpolate is a sibling, but will not be considered in memory view computation unless it is the
        // only consumer; that should change after E#112803 is implemented and test can be simplified
        if (ignoreSEPOps && mlir::isa<VPU::NCEInterpolateOp>(op)) {
            return;
        }

        // invalid Concat over H with SOH parent/consumer strategy - not a sibling
        if (auto concat = mlir::dyn_cast<VPU::ConcatOp>(op.getOperation())) {
            if (isConcatOverH(concat)) {
                return;
            }
        }

        consumers.emplace(op);
    });

    return ProducersConsumers(producers, consumers);
}

}  // namespace

TEST_P(GetOverlapSiblingsTests, GetOps) {
    const auto inputIR = GetParam();

    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::config::ArchKind::NPU40XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto producersConsumers = getProducerConsumersSets(func, false);
    auto expectedSiblings = producersConsumers.consumers;
    auto siblingAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    for (auto op : expectedSiblings) {
        const auto actualSiblings = siblingAnalysis.getSiblings(op);

        EXPECT_EQ(actualSiblings, expectedSiblings);
    }
}

TEST_P(GetActivationOverlapTests, GetParams) {
    const auto params = GetParam();
    const llvm::StringLiteral inputIR = params.inputIR;
    const auto numTiles = params.numTiles;
    const auto expectedMemoryShapes = params.memoryShapes;
    const auto expectedMemoryOffsets = params.memoryOffsets;
    const auto inputTileShape = params.tileShape;

    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::config::ArchKind::NPU40XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);

    const auto producersConsumers = getProducerConsumersSets(func);

    SmallVector<int64_t> clusteringDims = {};
    for (size_t clusterIdx = 0; clusterIdx < numTiles.size(); clusterIdx++) {
        if (numTiles[clusterIdx] > 1) {
            clusteringDims.push_back(clusterIdx);
        }
    }

    vpux::NDTypeInterface inputType = nullptr;
    const auto& expectedSiblings = producersConsumers.consumers;
    auto siblingAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    for (auto op : expectedSiblings) {
        if (!inputTileShape.empty()) {
            inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType())
                                .extractDenseTile(ShapeRef{0, 0, 0, 0}, ShapeRef(inputTileShape));
        }

        auto actualOverlappedParams = VPU::getActivationOverlappedParams(
                op, numTiles, uniformDistributedSegments ? true : false, siblingAnalysis, inputType);

        const auto memShapes = actualOverlappedParams.getMemoryShapes();
        const auto memOffsets = actualOverlappedParams.getMemoryOffsets();

        for (size_t idx = 0; idx < memShapes.size(); idx++) {
            const auto& clusterMemShapes = memShapes[idx];
            const auto& clusterMemOffsets = memOffsets[idx];
            const auto& expectedClusterMemShapes = expectedMemoryShapes[idx];
            const auto& expectedClusterMemOffsets = expectedMemoryOffsets[idx];

            // Check only overlapped clustering dim, as siblings may have different channel num, for example
            for (const auto& dim : clusteringDims) {
                EXPECT_EQ(clusterMemShapes[dim], expectedClusterMemShapes[dim]);
                EXPECT_EQ(clusterMemOffsets[dim], expectedClusterMemOffsets[dim]);
            }
        }
    }
}

TEST_P(GetOutputOverlapTests, GetParams) {
    const auto params = GetParam();
    const llvm::StringLiteral inputIR = params.inputIR;
    const auto numTiles = params.numTiles;
    const auto expectedMemoryShapes = params.memoryShapes;
    const auto expectedMemoryOffsets = params.memoryOffsets;
    const auto outputTileShape = params.tileShape;

    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::config::ArchKind::NPU40XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto uniformDistributedSegments = mlir::UnitAttr::get(&ctx);

    const auto producersConsumers =
            getProducerConsumersSets(func, /*ignore sep ops*/ false, /*max pool producer*/ true);

    NDTypeInterface outputType = nullptr;
    auto siblingAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    for (auto op : producersConsumers.producers) {
        if (!outputTileShape.empty()) {
            outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType())
                                 .extractDenseTile(ShapeRef{0, 0, 0, 0}, ShapeRef(outputTileShape));
        }

        auto producerOverlappedParams =
                VPU::getOutputOverlappedParams(op, numTiles, uniformDistributedSegments ? true : false, outputType,
                                               TileInfo(ShapeRef()), siblingAnalysis);

        if (mlir::isa<VPU::NCEMaxPoolOp>(op.getOperation())) {
            // MaxPool producer is used when not all produces should have the same set of consumers.
            // We check the memory view of the AvgPool against the expected one.
            // As a result MaxPool will not have the same memory view as the expected one.
            EXPECT_NE(SmallVector<SmallVector<int64_t>>(producerOverlappedParams.getMemoryShapes()),
                      expectedMemoryShapes);
            EXPECT_NE(SmallVector<SmallVector<int64_t>>(producerOverlappedParams.getMemoryOffsets()),
                      expectedMemoryOffsets);
        } else {
            EXPECT_EQ(SmallVector<SmallVector<int64_t>>(producerOverlappedParams.getMemoryShapes()),
                      expectedMemoryShapes);
            EXPECT_EQ(SmallVector<SmallVector<int64_t>>(producerOverlappedParams.getMemoryOffsets()),
                      expectedMemoryOffsets);
        }
    }
}

// clang-format off

//
// AveragePool ----> Conv %1
//              \--> Conv %2
//
// Siblings: Conv %1 & Conv %2
//

llvm::StringLiteral twoConvConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
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

//
// AveragePool ----> Conv %1
//              \--> NCE.Interp %2
//
// Siblings: Conv %1 && NCE.Interp %2
// (Caveat: NCE.Interp is not taken into consideration when computing memory view for Conv input
//  or AvgPool output until E#112803 is solved)
//

llvm::StringLiteral nceInterpAndConvConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x28x27xf16, {order = #NHWC}>)
          -> (tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x56x54xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<144x144x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x5x5xf16, {order = #NHWC}>
            %w3 = const.Declare tensor<144x144x2x2xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x2x2xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>

            %sparseMap = const.Declare tensor<1x96x57x55xi1> = dense<1> : tensor<1x96x57x55xi1>
            %storageElement = VPU.StorageElementTable {
                dataElemType = i32, seDepth = 1, seSize = [144], dataShape = [1, 144, 28, 27],
                seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0]>}
                    -> tensor<1x1x57x55xi32, {order = #NHWC}>

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

            %interpIn = VPU.GroupSparseTensor(%0, %sparseMap, %storageElement) {
                seAttr = #VPU.SEInterpolate<
                    mode = <BILINEAR>,
                    coordinate_transformation_mode = <ASYMMETRIC>,
                    scale = [1.0, 1.0, 2.0, 2.0]>
            } -> !VPU.SparseTensor<data=tensor<1x144x28x27xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x57x55xi1>,
                           storage_element_table=tensor<1x1x57x55xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0]>>

            %2 = VPU.NCE.Interpolate(%interpIn, %w3, %weightsTable) {
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [144, 144, 2, 2],
                strides = [1, 1],
                mode = #VPU.nce_interpolate_mode<BILINEAR>,
                scales_attr = [2, 2]}
                        -> tensor<1x144x56x54xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x56x54xf16, {order = #NHWC}>
        }
})";

//
//              /--> HSwish %3
// AveragePool ----> Conv %1
//              \--> Conv %2
//
// Siblings: Conv %1, Conv %2, HSwish %3
//

llvm::StringLiteral threeClusteredConsumers = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x28x27xf16, {order = #NHWC}>)
          -> (tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>) {
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

            %3 = VPU.HSwish(%0)
                    : tensor<1x144x28x27xf16, {order = #NHWC}>
                        -> tensor<1x144x28x27xf16, {order = #NHWC}>

            return %1, %2, %3 : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>
        }
})";

//
//              /--> Negative %2 (currently not a clustered op)
// AveragePool ----> Conv %1
//
// Siblings: Conv %1
//

llvm::StringLiteral oneConsumerNotClustered = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x28x27xf16, {order = #NHWC}>)
          -> (tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x3x3xf16, {order = #NHWC}>
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

            %2 = VPU.Negative(%0)
                : tensor<1x144x28x27xf16, {order = #NHWC}>
                    -> tensor<1x144x28x27xf16, {order = #NHWC}>

            return %1, %2 : tensor<1x144x28x27xf16, {order = #NHWC}>, tensor<1x144x28x27xf16, {order = #NHWC}>
        }
})";

// clang-format on

std::vector<llvm::StringLiteral> directConsumersWithOneInput = {twoConvConsumers, nceInterpAndConvConsumers,
                                                                threeClusteredConsumers, oneConsumerNotClustered};

INSTANTIATE_TEST_SUITE_P(DirectConsumersWithOneActInput, GetOverlapSiblingsTests,
                         testing::ValuesIn(directConsumersWithOneInput));

// clang-format off

std::vector<OverlappedTestParams> directConsumersWithOneInputMemoryView = {
    {
        twoConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 9, 27}, {1, 144, 11, 27}, {1, 144, 11, 27}, {1, 144, 9, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 5, 0}, {0, 0, 12, 0}, {0, 0, 19, 0}},
        /*tileShape=*/{}
    },
    {
        nceInterpAndConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 8, 27}, {1, 144, 9, 27}, {1, 144, 9, 27}, {1, 144, 8, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 6, 0}, {0, 0, 13, 0}, {0, 0, 20, 0}},
        /*tileShape=*/{}
    }
};

INSTANTIATE_TEST_SUITE_P(DirectConsumersWithOneActInput, GetActivationOverlapTests,
                         testing::ValuesIn(directConsumersWithOneInputMemoryView));

INSTANTIATE_TEST_SUITE_P(DirectConsumersWithOneActInput, GetOutputOverlapTests,
                         testing::ValuesIn(directConsumersWithOneInputMemoryView));

std::vector<OverlappedTestParams> directConsumersWithOneInputMemoryViewTile = {
    {
        twoConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 6, 14}, {1, 144, 8, 14}, {1, 144, 7, 14}, {1, 144, 5, 14}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 6, 0}, {0, 0, 9, 0}},
        /*tileShape=*/{1, 144, 14, 14}
    },
    {
        nceInterpAndConvConsumers, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 144, 5, 13}, {1, 144, 6, 13}, {1, 144, 5, 13}, {1, 144, 4, 13}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 3, 0}, {0, 0, 7, 0}, {0, 0, 10, 0}},
        /*tileShape=*/{1, 144, 14, 13}
    }
};

INSTANTIATE_TEST_SUITE_P(DirectConsumersWithOneActInputOutputTile, GetOutputOverlapTests,
                         testing::ValuesIn(directConsumersWithOneInputMemoryViewTile));
INSTANTIATE_TEST_SUITE_P(DirectConsumersWithOneActInputInputTile, GetActivationOverlapTests,
                         testing::ValuesIn(directConsumersWithOneInputMemoryViewTile));


//
// AveragePool ----> QuantizeCast ---> Conv %2
//              \--> QuantizeCast ---> Conv %4
//
// Siblings: Conv %2, Conv %4
//

llvm::StringLiteral quantizeCastDirectConsumer = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    !qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
    !qElemType1 = !quant.uniform<u8:f16, 0.0038832720588235295:128>
    !qElemType2 = !quant.uniform<u8:f16, 0.013744638480392158:128>
    !qElemType3 = !quant.uniform<u8:f16, 0.047862344452225551:128>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 2 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x32x28x27x!qElemType, {order = #NHWC}>)
          -> (tensor<1x32x14x14x!qElemType2, {order = #NHWC}>, tensor<1x32x28x27x!qElemType3, {order = #NHWC}>) {
            %weights0 = const.Declare tensor<32x32x1x1x!qElemType1, {order = #NHWC}>
                = dense<1.0> : tensor<32x32x1x1xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
            %weights1 = const.Declare tensor<32x32x5x5x!qElemType1, {order = #NHWC}>
                = dense<1.0> : tensor<32x32x5x5xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
            %weightsTable = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x32x28x27x!qElemType, {order = #NHWC}>

            %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType2}
                : tensor<1x32x28x27x!qElemType, {order = #NHWC}> -> tensor<1x32x28x27x!qElemType2, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%1, %weights0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [32, 32, 1, 1],
                    strides = [2, 2]}
                        : tensor<1x32x28x27x!qElemType2, {order = #NHWC}>, tensor<32x32x1x1x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x14x14x!qElemType2, {order = #NHWC}>

            %3 = VPU.QuantizeCast(%0) {dstElemType = !qElemType3}
                : tensor<1x32x28x27x!qElemType, {order = #NHWC}> -> tensor<1x32x28x27x!qElemType3, {order = #NHWC}>

            %4 = VPU.NCE.Convolution(%3, %weights1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                    rawFilterShape = [32, 32, 5, 5],
                    strides = [1, 1]}
                        : tensor<1x32x28x27x!qElemType3, {order = #NHWC}>, tensor<32x32x5x5x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x28x27x!qElemType3, {order = #NHWC}>

            return %2, %4 : tensor<1x32x14x14x!qElemType2, {order = #NHWC}>, tensor<1x32x28x27x!qElemType3, {order = #NHWC}>
        }
})";

//
// AveragePool ----> Conv %5
//              \--> QuantizeCast ---> Conv %2
//                        \----------> Conv %4
//
// Siblings: Conv %2, Conv %4, Conv %5
//

llvm::StringLiteral multipleConsumersOfQuantizeCast = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    !qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
    !qElemType1 = !quant.uniform<u8:f16, 0.0038832720588235295:128>
    !qElemType2 = !quant.uniform<u8:f16, 0.013744638480392158:128>
    !qElemType3 = !quant.uniform<u8:f16, 0.047862344452225551:128>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 2 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x32x28x27x!qElemType, {order = #NHWC}>)
          -> (tensor<1x32x14x14x!qElemType2, {order = #NHWC}>, tensor<1x32x28x27x!qElemType3, {order = #NHWC}>, tensor<1x32x28x27x!qElemType3, {order = #NHWC}>) {
            %weights0 = const.Declare tensor<32x32x1x1x!qElemType1, {order = #NHWC}>
                = dense<1.0> : tensor<32x32x1x1xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
            %weights1 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}>
                = dense<1.0> : tensor<32x32x3x3xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
            %weightsTable = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x32x28x27x!qElemType, {order = #NHWC}>

            %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType2}
                : tensor<1x32x28x27x!qElemType, {order = #NHWC}> -> tensor<1x32x28x27x!qElemType2, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%1, %weights0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [32, 32, 1, 1],
                    strides = [2, 2]}
                        : tensor<1x32x28x27x!qElemType2, {order = #NHWC}>, tensor<32x32x1x1x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x14x14x!qElemType2, {order = #NHWC}>

            %4 = VPU.NCE.Convolution(%1, %weights1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    rawFilterShape = [32, 32, 3, 3],
                    strides = [1, 1]}
                        : tensor<1x32x28x27x!qElemType2, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x28x27x!qElemType3, {order = #NHWC}>

            %5 = VPU.NCE.Convolution(%0, %weights0, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [32, 32, 1, 1],
                    strides = [1, 1]}
                        : tensor<1x32x28x27x!qElemType, {order = #NHWC}>, tensor<32x32x1x1x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x28x27x!qElemType3, {order = #NHWC}>

            return %2, %4, %5 : tensor<1x32x14x14x!qElemType2, {order = #NHWC}>, tensor<1x32x28x27x!qElemType3, {order = #NHWC}>, tensor<1x32x28x27x!qElemType3, {order = #NHWC}>
        }
})";

// clang-format on

std::vector<llvm::StringLiteral> withQuantizeCastConsumers = {quantizeCastDirectConsumer,
                                                              multipleConsumersOfQuantizeCast};

INSTANTIATE_TEST_SUITE_P(QuantizeCastDirectConsumers, GetOverlapSiblingsTests,
                         testing::ValuesIn(withQuantizeCastConsumers));

// clang-format off

std::vector<OverlappedTestParams> quantizeCastMemoryView = {
    {
        quantizeCastDirectConsumer, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 32, 9, 27}, {1, 32, 11, 27}, {1, 32, 11, 27}, {1, 32, 9, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 5, 0}, {0, 0, 12, 0}, {0, 0, 19, 0}},
        /*tileShape=*/{}
    },
    {
        multipleConsumersOfQuantizeCast, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 32, 8, 27}, {1, 32, 9, 27}, {1, 32, 9, 27}, {1, 32, 8, 27}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 6, 0}, {0, 0, 13, 0}, {0, 0, 20, 0}},
        /*tileShape=*/{}
    }
};

INSTANTIATE_TEST_SUITE_P(QuantizeCastDirectConsumers, GetActivationOverlapTests,
                         testing::ValuesIn(quantizeCastMemoryView));

INSTANTIATE_TEST_SUITE_P(QuantizeCastDirectConsumers, GetOutputOverlapTests,
                         testing::ValuesIn(quantizeCastMemoryView));

std::vector<OverlappedTestParams> quantizeCastMemoryViewOutputTile = {
    {
        quantizeCastDirectConsumer, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 32, 6, 14}, {1, 32, 7, 14}, {1, 32, 7, 14}, {1, 32, 5, 14}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 5, 0}, {0, 0, 8, 0}},
        /*tileShape=*/{1, 32, 13, 14}
    },
    {
        multipleConsumersOfQuantizeCast, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 32, 5, 14}, {1, 32, 5, 14}, {1, 32, 5, 14}, {1, 32, 4, 14}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 3, 0}, {0, 0, 6, 0}, {0, 0, 9, 0}},
        /*tileShape=*/{1, 32, 13, 14}
    }
};

INSTANTIATE_TEST_SUITE_P(QuantizeCastDirectConsumersTiled, GetOutputOverlapTests,
                         testing::ValuesIn(quantizeCastMemoryViewOutputTile));

//
//              /-----> Conv %2
// AveragePool ------> Conv %1 ---> Eltwise %3
//              \---------------------^
//
// Siblings: Conv %1, Conv %2, Eltwise %3
//

llvm::StringLiteral eltwiseResidualBlock = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
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

            %3 = VPU.NCE.Eltwise(%0, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>}
                        -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %2, %3 : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

//
//                 /-----> Conv %1
// AveragePool %0 --------> Eltwise %4
//                             ^
// AveragePool %2 -------------|
//                \-----> Conv %3
//
// Siblings: Conv %1, Conv %3, Eltwise %4
//

llvm::StringLiteral eltwiseWithParentsInDiffSubgraphs = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
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

            %2 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [2, 2],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%2, %w1, %weightsTable) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [16, 16, 1, 1],
                    strides = [1, 1]}
                        : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>,  tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %4 = VPU.NCE.Eltwise(%0, %2) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>}
                        -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %1, %3, %4 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

// clang-format on

std::vector<llvm::StringLiteral> withEltwiseConsumers = {eltwiseResidualBlock, eltwiseWithParentsInDiffSubgraphs};

INSTANTIATE_TEST_SUITE_P(EltwiseConsumers, GetOverlapSiblingsTests, testing::ValuesIn(withEltwiseConsumers));

// clang-format off

std::vector<OverlappedTestParams> eltwiseMemoryView = {
    {
        eltwiseWithParentsInDiffSubgraphs, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 3, 8}, {1, 16, 4, 8}, {1, 16, 4, 8}, {1, 16, 3, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 3, 0}, {0, 0, 5, 0}},
        /*tileShape=*/{}
    }
};

INSTANTIATE_TEST_SUITE_P(EltwiseConsumers, GetActivationOverlapTests, testing::ValuesIn(eltwiseMemoryView));

INSTANTIATE_TEST_SUITE_P(EltwiseConsumers, GetOutputOverlapTests, testing::ValuesIn(eltwiseMemoryView));

std::vector<OverlappedTestParams> eltwiseMemoryViewOutputTile = {
    {
        eltwiseWithParentsInDiffSubgraphs, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 3, 8}, {1, 16, 4, 8}, {1, 16, 4, 8}, {1, 16, 2, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 3, 0}, {0, 0, 5, 0}},
        /*tileShape=*/{1, 16, 7, 8}
    }
};

INSTANTIATE_TEST_SUITE_P(EltwiseConsumersTiled, GetOutputOverlapTests, testing::ValuesIn(eltwiseMemoryViewOutputTile));

//
//              /-----> Conv %2
// AveragePool ------> Conv %1 ---> Eltwise {inplace} %3 --> Conv %4
//              \---------------------^
//
// Siblings: Conv %1, Conv %2, Eltwise %3, Conv %4
//

llvm::StringLiteral eltwiseInPlaceSubgraph = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [2, 2]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x4xf16, {order = #NHWC}>

            %3 = VPU.NCE.Eltwise(%0, %1) {
                is_inplace = true,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %4 = VPU.NCE.Convolution(%3, %w2, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                rawFilterShape = [16, 16, 5, 5],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %2, %4 : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

//
//                 /-----> Conv %1
// AveragePool %0 --------> Eltwise {inplace} %4 ---> Conv %5
//                             ^
// AveragePool %2 -------------|
//                \-----> Conv %3
//
// Siblings: Conv %1, Conv %3, Eltwise %4, Conv %5
//

llvm::StringLiteral eltwiseInPlaceWithParentsInDiffSubgraphs = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [2, 2],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%2, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %4 = VPU.NCE.Eltwise(%0, %2) {
                is_inplace = true,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %5 = VPU.NCE.Convolution(%4, %w2, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                rawFilterShape = [16, 16, 5, 5],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %1, %3, %5 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

// clang-format on

std::vector<llvm::StringLiteral> withInPlaceEltwiseConsumers = {eltwiseInPlaceSubgraph,
                                                                eltwiseInPlaceWithParentsInDiffSubgraphs};

INSTANTIATE_TEST_SUITE_P(InPlaceEltwiseConsumers, GetOverlapSiblingsTests,
                         testing::ValuesIn(withInPlaceEltwiseConsumers));

// clang-format off

std::vector<OverlappedTestParams> inPlaceEltwiseMemoryView = {
    {
        eltwiseInPlaceWithParentsInDiffSubgraphs, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 6, 8}, {1, 16, 6, 8}, {1, 16, 4, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 4, 0}},
        /*tileShape=*/{}
    }
};

INSTANTIATE_TEST_SUITE_P(InPlaceEltwiseConsumers, GetActivationOverlapTests,
                         testing::ValuesIn(inPlaceEltwiseMemoryView));

INSTANTIATE_TEST_SUITE_P(InPlaceEltwiseConsumers, GetOutputOverlapTests,
                         testing::ValuesIn(inPlaceEltwiseMemoryView));

std::vector<OverlappedTestParams> inPlaceEltwiseMemoryViewOutputTile = {
    {
        eltwiseInPlaceWithParentsInDiffSubgraphs, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 6, 8}, {1, 16, 5, 8}, {1, 16, 3, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 4, 0}},
        /*tileShape=*/{1, 16, 7, 8}
    }
};

INSTANTIATE_TEST_SUITE_P(InPlaceEltwiseConsumersTiled, GetOutputOverlapTests,
                         testing::ValuesIn(inPlaceEltwiseMemoryViewOutputTile));


//
//              /-----> Conv %2
// AveragePool ------> Conv %1 ---> Concat %3 --> Conv %4
//              \---------------------^
//
// Siblings: Conv %1, Conv %2, Concat %3, Conv %4
//

llvm::StringLiteral concatSubgraph = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        }
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<16x32x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x32x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [2, 2]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x4xf16, {order = #NHWC}>

            %3 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}:
                    tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
                        -> tensor<1x32x8x8xf16, {order = #NHWC}>

            %4 = VPU.NCE.Convolution(%3, %w2, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                rawFilterShape = [16, 32, 5, 5],
                strides = [1, 1]}
                    : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<16x32x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %2, %4 : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

//
//                 /----> Conv %3
// AveragePool %0 ------> Conv %2 ---> Concat %4 --> MaxPool %5
// AveragePool %1 -----------------------^
//
// Siblings: Conv %2, Conv %3
//

llvm::StringLiteral notSOHCompatibleConcatSubgraph = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        }
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.MaxPool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%0, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [2, 2]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x4xf16, {order = #NHWC}>

            %4 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}:
                    tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
                        -> tensor<1x16x16x8xf16, {order = #NHWC}>

            %5 = VPU.NCE.MaxPool(%4) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [5, 5],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x16x8xf16, {order = #NHWC}>

            return %3, %5 : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}>
        }
})";

//
//                 /-----> Conv %1
// AveragePool %0 --------> Concat %4 ---> Conv %5
//                             ^
// AveragePool %2 -------------|
//                \-----> Conv %3
//
// Siblings: Conv %1, Conv %3, Concat %4, Conv %5
//

llvm::StringLiteral concatWithParentsInDiffSubgraphs = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        }
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x1x1xf16, {order = #NHWC}>
            %w2 = const.Declare tensor<16x32x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x32x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [2, 2],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %3 = VPU.NCE.Convolution(%2, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %4 = VPU.Concat(%0, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}:
                    tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
                        -> tensor<1x32x8x8xf16, {order = #NHWC}>

            %5 = VPU.NCE.Convolution(%4, %w2, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                rawFilterShape = [16, 32, 5, 5],
                strides = [1, 1]}
                    : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<16x32x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            return %1, %3, %5 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
        }
})";

// clang-format on

std::vector<llvm::StringLiteral> withConcatConsumers = {concatSubgraph, notSOHCompatibleConcatSubgraph,
                                                        concatWithParentsInDiffSubgraphs};

INSTANTIATE_TEST_SUITE_P(ConcatConsumers, GetOverlapSiblingsTests, testing::ValuesIn(withConcatConsumers));

// clang-format off

std::vector<OverlappedTestParams> concatConsumersMemoryView = {
    {   // for kernel 3x3, s1x1, pad {1,1,1,1}
        notSOHCompatibleConcatSubgraph, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 3, 8}, {1, 16, 4, 8}, {1, 16, 4, 8}, {1, 16, 3, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 3, 0}, {0, 0, 5, 0}},
        /*tileShape=*/{}
    },
    {   // for kernel 5x5, s1x1, pad {2,2,2,2}
        concatWithParentsInDiffSubgraphs, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 6, 8}, {1, 16, 6, 8}, {1, 16, 4, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 4, 0}},
        /*tileShape=*/{}
    }
};

INSTANTIATE_TEST_SUITE_P(ConcatConsumers, GetActivationOverlapTests,
                         testing::ValuesIn(concatConsumersMemoryView));

INSTANTIATE_TEST_SUITE_P(ConcatConsumers, GetOutputOverlapTests,
                         testing::ValuesIn(concatConsumersMemoryView));

std::vector<OverlappedTestParams> concatConsumersMemoryViewOutputTile = {
    {   // for kernel 3x3, s1x1, pad {1,1,1,1}
        notSOHCompatibleConcatSubgraph, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 3, 8}, {1, 16, 4, 8}, {1, 16, 4, 8}, {1, 16, 2, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 3, 0}, {0, 0, 5, 0}},
        /*tileShape=*/{1, 16, 7, 8}
    },
    {   // for kernel 5x5, s1x1, pad {2,2,2,2}
        concatWithParentsInDiffSubgraphs, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 6, 8}, {1, 16, 5, 8}, {1, 16, 3, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 4, 0}},
        /*tileShape=*/{1, 16, 7, 8}
    }
};

INSTANTIATE_TEST_SUITE_P(ConcatConsumersTiled, GetOutputOverlapTests,
                         testing::ValuesIn(concatConsumersMemoryViewOutputTile));

//
//                 /-----> Conv %2
// AveragePool %1 --------> Concat %3 ---> Eltwise %6
// AveragePool %0 ------------/               ^
//                                            |
// AveragePool %4 ----------------------------|
//                \-----> Conv %5
//
// Siblings: Conv %2, Conv %5, Concat %3, Eltwise %6
//

llvm::StringLiteral mixedSubgraph0 = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        }
        func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>, %arg1: tensor<1x32x8x8xf16, {order = #NHWC}>)
          -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
            %w1 = const.Declare tensor<16x32x5x5xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<16x32x5x5xf16, {order = #NHWC}>
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %1 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [2, 2],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%1, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %3 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}:
                    tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
                        -> tensor<1x32x8x8xf16, {order = #NHWC}>

            %4 = VPU.NCE.AveragePool(%arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x32x8x8xf16, {order = #NHWC}>

            %5 = VPU.NCE.Convolution(%4, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                rawFilterShape = [16, 32, 5, 5],
                strides = [1, 1]}
                    : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<16x32x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8xf16, {order = #NHWC}>

            %6 = VPU.NCE.Eltwise(%3, %4) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x32x8x8xf16, {order = #NHWC}>

            return %2, %5, %6 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}>
        }
})";

//
// Summary:
//
// %0 = VPU.NCE.AveragePool(%arg0)
// %1 = VPU.NCE.AveragePool(%arg1)
// %2 = VPU.NCE.Convolution(%0, ...)
// %3 = VPU.QuantizeCast(%0)
// %4 = VPU.NCE.Convolution(%3, ...)
// %5 = VPU.NCE.Eltwise(%1, %3)
// %6 = VPU.Concat(%1, %2)
// %7 = VPU.NCE.Convolution(%6, ...)
//
// Siblings: Conv %2, Conv %4, Eltwise %5, Concat %6, Conv %7
//

llvm::StringLiteral mixedSubgraph1 = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    !qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
    !qElemType1 = !quant.uniform<u8:f16, 0.0038832720588235295:128>
    !qElemType2 = !quant.uniform<u8:f16, 0.013744638480392158:128>
    !qElemType3 = !quant.uniform<u8:f16, 0.047862344452225551:128>
    module @test attributes {config.arch = #config.arch_kind<NPU40XX>} {
        config.Resources 4 of @NCE at 6.000000e+02 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        }
        func.func @main(%arg0: tensor<1x16x8x8x!qElemType, {order = #NHWC}>, %arg1: tensor<1x16x8x8x!qElemType, {order = #NHWC}>)
          -> (tensor<1x16x8x8x!qElemType3, {order = #NHWC}>, tensor<1x16x8x8x!qElemType2, {order = #NHWC}>, tensor<1x16x8x8x!qElemType3, {order = #NHWC}>) {
            %w0 = const.Declare tensor<16x16x3x3x!qElemType1, {order = #NHWC}>
                    = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
            %w1 = const.Declare tensor<16x32x5x5x!qElemType1, {order = #NHWC}>
                    = dense<1.0> : tensor<16x32x5x5xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
            %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

            %0 = VPU.NCE.AveragePool(%arg0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8x!qElemType, {order = #NHWC}>

            %1 = VPU.NCE.AveragePool(%arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x16x8x8x!qElemType2, {order = #NHWC}>

            %2 = VPU.NCE.Convolution(%0, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8x!qElemType, {order = #NHWC}>, tensor<16x16x3x3x!qElemType1, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8x!qElemType2, {order = #NHWC}>

            %3 = VPU.QuantizeCast(%0) {dstElemType = !qElemType2}
                : tensor<1x16x8x8x!qElemType, {order = #NHWC}> -> tensor<1x16x8x8x!qElemType2, {order = #NHWC}>

            %4 = VPU.NCE.Convolution(%3, %w0, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]}
                    : tensor<1x16x8x8x!qElemType2, {order = #NHWC}>, tensor<16x16x3x3x!qElemType1, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8x!qElemType3, {order = #NHWC}>

            %5 = VPU.NCE.Eltwise(%1, %3) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>}
                    -> tensor<1x16x8x8x!qElemType2, {order = #NHWC}>

            %6 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}:
                    tensor<1x16x8x8x!qElemType2, {order = #NHWC}>, tensor<1x16x8x8x!qElemType2, {order = #NHWC}>
                        -> tensor<1x32x8x8x!qElemType2, {order = #NHWC}>

            %7 = VPU.NCE.Convolution(%6, %w1, %weightsTable) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
                rawFilterShape = [16, 32, 5, 5],
                strides = [1, 1]}
                    : tensor<1x32x8x8x!qElemType2, {order = #NHWC}>, tensor<16x32x5x5x!qElemType1, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x8x8x!qElemType3, {order = #NHWC}>

            return %4, %5, %7 : tensor<1x16x8x8x!qElemType3, {order = #NHWC}>, tensor<1x16x8x8x!qElemType2, {order = #NHWC}>, tensor<1x16x8x8x!qElemType3, {order = #NHWC}>
        }
})";

// clang-format on

// Due stopping the graph traversal early to avoid increase in compilation time for
// real models, some of the above tests will no longer find all the possible siblings.
// Disabling all the MixedConsumers tests for now, until better handling of "levels"
// can be implemented.
// TODO: E#115755
std::vector<llvm::StringLiteral> mixedConsumersSubgraph = {mixedSubgraph0, mixedSubgraph1};

INSTANTIATE_TEST_SUITE_P(DISABLED_MixedConsumers, GetOverlapSiblingsTests, testing::ValuesIn(mixedConsumersSubgraph));

// clang-format off

std::vector<OverlappedTestParams> mixedConsumersMemoryView = {
    {   // for kernel 5x5, s1x1, pad {2,2,2,2}
        mixedSubgraph1, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 6, 8}, {1, 16, 6, 8}, {1, 16, 4, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 4, 0}},
        /*tileShape=*/{}
    }
};

INSTANTIATE_TEST_SUITE_P(DISABLED_MixedConsumers, GetActivationOverlapTests,
                         testing::ValuesIn(mixedConsumersMemoryView));

INSTANTIATE_TEST_SUITE_P(DISABLED_MixedConsumers, GetOutputOverlapTests,
                         testing::ValuesIn(mixedConsumersMemoryView));

std::vector<OverlappedTestParams> mixedConsumersMemoryViewOutputTile = {
    {   // for kernel 5x5, s1x1, pad {2,2,2,2}
        mixedSubgraph1, /*numTiles=*/{1, 1, 4, 1},
        /*memoryShapes=*/{{1, 16, 4, 8}, {1, 16, 6, 8}, {1, 16, 5, 8}, {1, 16, 3, 8}},
        /*memoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 4, 0}},
        /*tileShape=*/{1, 16, 7, 8}
    }
};

INSTANTIATE_TEST_SUITE_P(DISABLED_MixedConsumersTiled, GetOutputOverlapTests,
                         testing::ValuesIn(mixedConsumersMemoryViewOutputTile));
