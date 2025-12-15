//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include <vector>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_distributed_ops_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_distributed_ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

struct SETablePatchParams {
    bool resetBasePtrs;
    std::vector<int32_t> inputSETableValues;
    std::vector<int64_t> seTableShape;
    std::vector<int64_t> numTiles;
    int64_t numClusters;
    // NCE Input distribution parameters
    std::vector<int64_t> dataShape;
    std::vector<std::vector<int64_t>> nceInputComputeShapes;
    std::vector<std::vector<int64_t>> nceInputComputeOffsets;
    std::vector<std::vector<int64_t>> nceInputMemoryShapes;
    std::vector<std::vector<int64_t>> nceInputMemoryOffsets;
    // SE Table subview parameters
    std::vector<std::vector<int64_t>> seTableMemoryShapes;
    std::vector<std::vector<int64_t>> seTableMemoryOffsets;
    std::vector<std::vector<int32_t>> expectedPatchedValues;
};

class SETablePatchTests : public testing::TestWithParam<SETablePatchParams> {};

TEST_P(SETablePatchTests, patchSETableValue) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    const auto params = GetParam();

    mlir::OpBuilder builder(&ctx);
    auto loc = mlir::UnknownLoc::get(&ctx);

    // Create proper MLIR module structure
    auto moduleOp = mlir::ModuleOp::create(loc);
    auto funcOp = mlir::func::FuncOp::create(loc, "SETablePatchTestFunc", builder.getFunctionType({}, {}));
    moduleOp.push_back(funcOp);

    auto entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create SE table constant with NHWC layout
    const auto seTableTensorType = mlir::RankedTensorType::get(params.seTableShape, builder.getIntegerType(32));
    const auto seTableContent = mlir::DenseElementsAttr::get(seTableTensorType, ArrayRef(params.inputSETableValues));
    const auto seTableConstant = Const::ContentAttr::get(seTableContent);
    auto ddrMemSpaceAttr = vpux::IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::DDR), 0);

    // Create SE table type with NHWC layout
    const auto seTableType =
            mlir::MemRefType::get(params.seTableShape, builder.getIntegerType(32),
                                  mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx)), ddrMemSpaceAttr);

    // Create a constant operation
    auto constOp = builder.create<Const::DeclareOp>(loc, seTableType, seTableConstant);
    VPUX_SCOPE_EXIT {
        constOp->erase();
    };

    // Create NCE Input distribution attribute
    auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    auto numTilesAttr = getIntArrayAttr(&ctx, params.numTiles);
    auto nceInputComputeShapesAttr = getIntArrayOfArray(&ctx, params.nceInputComputeShapes);
    auto nceInputComputeOffsetsAttr = getIntArrayOfArray(&ctx, params.nceInputComputeOffsets);
    auto nceInputMemoryShapesAttr = getIntArrayOfArray(&ctx, params.nceInputMemoryShapes);
    auto nceInputMemoryOffsetsAttr = getIntArrayOfArray(&ctx, params.nceInputMemoryOffsets);

    auto nceInputDistributionAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, /*kernelSize=*/nullptr,
            /*pads=*/nullptr, /*kernelStrides=*/nullptr,
            /*numClusters=*/getIntAttr(&ctx, params.numClusters),
            /*alignment=*/nullptr,
            /*uniformDistributedSegments=*/nullptr, nceInputComputeShapesAttr, nceInputComputeOffsetsAttr,
            nceInputMemoryShapesAttr, nceInputMemoryOffsetsAttr,
            /*equalMemoryAndComputeView=*/nullptr);

    // Create distributed buffer type for NCE Input
    auto memSpaceAttr = vpux::IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    auto distributedType = VPUIP::DistributedBufferType::get(
            &ctx, params.dataShape, builder.getF16Type(), mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx)),
            memSpaceAttr, nceInputDistributionAttr);

    // Loop through all clusters to test patchSETableValue
    for (int64_t clusterId = 0; clusterId < params.numClusters; ++clusterId) {
        auto targetMemoryShape = params.seTableMemoryShapes[clusterId];
        auto targetMemoryOffset = params.seTableMemoryOffsets[clusterId];
        builder.setInsertionPointAfter(constOp);
        auto subviewOp = builder.createOrFold<VPUIP::SubViewOp>(loc, constOp, targetMemoryOffset, targetMemoryShape);
        VPUX_SCOPE_EXIT {
            subviewOp.getDefiningOp()->erase();
        };
        auto subviewConstOp = subviewOp.getDefiningOp<Const::DeclareOp>();

        // Call patchSETableValue function with the subview result
        auto patchedValue = VPUIP::patchSETableValue(loc, subviewConstOp, distributedType, clusterId, builder,
                                                     params.resetBasePtrs);

        // Extract the patched SE table values
        auto patchedConstOp = mlir::cast<Const::DeclareOp>(patchedValue.getDefiningOp());
        VPUX_SCOPE_EXIT {
            patchedConstOp->erase();
        };
        auto patchedContent = patchedConstOp.getContent();
        auto patchedValues = to_small_vector(patchedContent.getValues<int32_t>());

        // Verify the patched values match the expected values for this cluster
        EXPECT_EQ(patchedValues.size(), params.expectedPatchedValues[clusterId].size())
                << "Cluster " << clusterId << " has incorrect number of patched values";
        for (size_t i = 0; i < patchedValues.size(); ++i) {
            EXPECT_EQ(patchedValues[i], params.expectedPatchedValues[clusterId][i])
                    << "Cluster " << clusterId << " has incorrect patched value at index " << i;
        }
    }
}

// clang-format off

// Test parameters for patchSETableValue
std::vector<SETablePatchParams> seTablePatchParams = {
    // Test case 1: H-tiling case
    {
        /*resetBasePtrs*/false,
        /*inputSETableValues=*/{
            0x0000, 0x0000, 0x0400, 0x0800, 0x0C00, 0x0C00,
            0x0000, 0x0000, 0x0400, 0x0800, 0x0C00, 0x0C00,
            0x1000, 0x1000, 0x1400, 0x1800, 0x1C00, 0x1C00,
            0x2000, 0x2000, 0x2400, 0x2800, 0x2C00, 0x2C00,
            0x2001, 0x2001, 0x2401, 0x2801, 0x2C01, 0x2C01,
            0x2001, 0x2001, 0x2401, 0x2801, 0x2C01, 0x2C01
        },
        /*seTableShape=*/{1, 1, 6, 6},
        /*numTiles=*/{1, 1, 3, 1},
        /*numClusters=*/3,
        /*dataShape=*/{1, 16, 4, 4},
        /*nceInputComputeShapes=*/{{1, 16, 1, 4}, {1, 16, 2, 4}, {1, 16, 1, 4}},
        /*nceInputComputeOffsets=*/{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 3, 0}},
        /*nceInputMemoryShapes=*/{{1, 16, 3, 4}, {1, 16, 3, 4}, {1, 16, 2, 4}},
        /*nceInputMemoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 2, 0}},
        /*seTableMemoryShapes=*/{{1, 1, 4, 6}, {1, 1, 3, 6}, {1, 1, 3, 6}},
        /*seTableMemoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 3, 0}},
        /*expectedPatchedValues=*/{
            // Expected patched values for cluster 0
            {0x0000, 0x0000, 0x0400, 0x0800, 0x0C00, 0x0C00,
             0x0000, 0x0000, 0x0400, 0x0800, 0x0C00, 0x0C00,
             0x1000, 0x1000, 0x1400, 0x1800, 0x1C00, 0x1C00,
             0x2000, 0x2000, 0x2400, 0x2800, 0x2C00, 0x2C00},
            // Expected patched values for cluster 1
            {0x0001, 0x0001, 0x0401, 0x0801, 0x0C01, 0x0C01,
             0x1001, 0x1001, 0x1401, 0x1801, 0x1C01, 0x1C01,
             0x2001, 0x2001, 0x2401, 0x2801, 0x2C01, 0x2C01},
            // Expected patched values for cluster 2
            {0x0002, 0x0002, 0x0402, 0x0802, 0x0C02, 0x0C02,
             0x1002, 0x1002, 0x1402, 0x1802, 0x1C02, 0x1C02,
             0x1002, 0x1002, 0x1402, 0x1802, 0x1C02, 0x1C02}
        }
    },

    // Test case 2: W-tiling case
    {
        /*resetBasePtrs*/false,
        /*inputSETableValues=*/{
            0x0000, 0x0000, 0x0400, 0x0800, 0x0801, 0x0801,
            0x0000, 0x0000, 0x0400, 0x0800, 0x0801, 0x0801,
            0x1000, 0x1000, 0x1400, 0x1800, 0x1801, 0x1801,
            0x2000, 0x2000, 0x2400, 0x2800, 0x2801, 0x2801,
            0x3000, 0x3000, 0x3400, 0x3800, 0x3801, 0x3801,
            0x3000, 0x3000, 0x3400, 0x3800, 0x3801, 0x3801
        },
        /*seTableShape=*/{1, 1, 6, 6},
        /*numTiles=*/{1, 1, 1, 3},
        /*numClusters=*/3,
        /*dataShape=*/{1, 16, 4, 4},
        /*nceInputComputeShapes=*/{{1, 16, 4, 1}, {1, 16, 4, 2}, {1, 16, 4, 1}},
        /*nceInputComputeOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 3}},
        /*nceInputMemoryShapes=*/{{1, 16, 4, 3}, {1, 16, 4, 3}, {1, 16, 4, 2}},
        /*nceInputMemoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}},
        /*seTableMemoryShapes=*/{{1, 1, 6, 4}, {1, 1, 6, 3}, {1, 1, 6, 3}},
        /*seTableMemoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}},
        /*expectedPatchedValues=*/{
            // Expected patched values for cluster 0
            {0x0000, 0x0000, 0x0400, 0x0800,
             0x0000, 0x0000, 0x0400, 0x0800,
             0x1000, 0x1000, 0x1400, 0x1800,
             0x2000, 0x2000, 0x2400, 0x2800,
             0x3000, 0x3000, 0x3400, 0x3800,
             0x3000, 0x3000, 0x3400, 0x3800},
            // Expected patched values for cluster 1
            {0x0001, 0x0401, 0x0801,
             0x0001, 0x0401, 0x0801,
             0x1001, 0x1401, 0x1801,
             0x2001, 0x2401, 0x2801,
             0x3001, 0x3401, 0x3801,
             0x3001, 0x3401, 0x3801},
            // Expected patched values for cluster 2
            {0x0002, 0x0402, 0x0402,
             0x0002, 0x0402, 0x0402,
             0x1002, 0x1402, 0x1402,
             0x2002, 0x2402, 0x2402,
             0x3002, 0x3402, 0x3402,
             0x3002, 0x3402, 0x3402},
        }
    },

    // Test case 3: W-tiling case on other arch
    {
        /*resetBasePtrs*/true,
        /*inputSETableValues=*/{
            0x0000, 0x0000, 0x0400, 0x0800, 0x0801, 0x0801,
            0x0000, 0x0000, 0x0400, 0x0800, 0x0801, 0x0801,
            0x1000, 0x1000, 0x1400, 0x1800, 0x1801, 0x1801,
            0x2000, 0x2000, 0x2400, 0x2800, 0x2801, 0x2801,
            0x3000, 0x3000, 0x3400, 0x3800, 0x3801, 0x3801,
            0x3000, 0x3000, 0x3400, 0x3800, 0x3801, 0x3801
        },
        /*seTableShape=*/{1, 1, 6, 6},
        /*numTiles=*/{1, 1, 1, 3},
        /*numClusters=*/3,
        /*dataShape=*/{1, 16, 4, 4},
        /*nceInputComputeShapes=*/{{1, 16, 4, 1}, {1, 16, 4, 2}, {1, 16, 4, 1}},
        /*nceInputComputeOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 3}},
        /*nceInputMemoryShapes=*/{{1, 16, 4, 3}, {1, 16, 4, 3}, {1, 16, 4, 2}},
        /*nceInputMemoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}},
        /*seTableMemoryShapes=*/{{1, 1, 6, 4}, {1, 1, 6, 3}, {1, 1, 6, 3}},
        /*seTableMemoryOffsets=*/{{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}},
        /*expectedPatchedValues=*/{
            // Expected patched values for cluster 0
            {0x0000, 0x0000, 0x0400, 0x0800,
             0x0000, 0x0000, 0x0400, 0x0800,
             0x1000, 0x1000, 0x1400, 0x1800,
             0x2000, 0x2000, 0x2400, 0x2800,
             0x3000, 0x3000, 0x3400, 0x3800,
             0x3000, 0x3000, 0x3400, 0x3800},
            // Expected patched values for cluster 1
            {0x0000, 0x0400, 0x0800,
             0x0000, 0x0400, 0x0800,
             0x1000, 0x1400, 0x1800,
             0x2000, 0x2400, 0x2800,
             0x3000, 0x3400, 0x3800,
             0x3000, 0x3400, 0x3800},
            // Expected patched values for cluster 2
            {0x0000, 0x0400, 0x0400,
             0x0000, 0x0400, 0x0400,
             0x1000, 0x1400, 0x1400,
             0x2000, 0x2400, 0x2400,
             0x3000, 0x3400, 0x3400,
             0x3000, 0x3400, 0x3400},
        }
    }
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(unit, SETablePatchTests, testing::ValuesIn(seTablePatchParams));
