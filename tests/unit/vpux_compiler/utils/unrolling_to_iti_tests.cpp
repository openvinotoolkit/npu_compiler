//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";

using MLIR_ITIUnrollTest = MLIR_UnitBase;

int64_t getInwardHaloConsumerMatches(VPUIP::HaloRegionAttr inwardHalo, ArrayRef<mlir::Value> outputIti) {
    SmallVector<VPUIP::HaloRegionAttr> consumerInwardHalos;
    int64_t count = 0;
    for (auto output : outputIti) {
        auto targetIti = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(output.getType());
        EXPECT_TRUE(targetIti != nullptr);

        count += llvm::count(targetIti.getInwardHaloRegions(), inwardHalo);
    }

    return count;
}

void checkOutwardHalosConsumersMatchOutputIti(ArrayRef<VPUIP::OutwardHaloRegionAttr> outwardHaloRegions,
                                              ArrayRef<mlir::Value> outputIti) {
    for (const auto outwardHalo : outwardHaloRegions) {
        for (const auto inwardHaloAttr : outwardHalo.getInwardHaloRegions()) {
            auto inwardHalo = mlir::cast<vpux::VPUIP::HaloRegionAttr>(inwardHaloAttr);
            int64_t count = getInwardHaloConsumerMatches(inwardHalo, outputIti);
            EXPECT_EQ(count, 1);
        }
    }
}

VPUIP::HaloRegionAttr getInwardHalo(mlir::MLIRContext* ctx, mlir::ArrayAttr shapeAttr, ArrayRef<int64_t> offset,
                                    const int64_t clusterId) {
    const auto offsetAttr = getIntArrayAttr(ctx, offset);
    return VPUIP::HaloRegionAttr::get(ctx, shapeAttr, offsetAttr, getIntAttr(ctx, clusterId));
}

VPUIP::OutwardHaloRegionAttr getOutwardHalo(mlir::MLIRContext* ctx, mlir::ArrayAttr haloShapeAttr,
                                            const int64_t clusterId, ArrayRef<int64_t> outwardOffset,
                                            ArrayRef<SmallVector<int64_t>> inwardOffsets,
                                            SmallVector<int64_t> inwardClusters) {
    const auto outwardHaloOffsetAttr = getIntArrayAttr(ctx, outwardOffset);

    SmallVector<mlir::Attribute> inwardHalos;
    for (size_t idx = 0; idx < inwardClusters.size(); idx++) {
        inwardHalos.push_back(getInwardHalo(ctx, haloShapeAttr, inwardOffsets[idx], inwardClusters[idx]));
    }

    const auto inwardHalosArrayAttr = mlir::ArrayAttr::get(ctx, inwardHalos);
    return VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, outwardHaloOffsetAttr, getIntAttr(ctx, clusterId),
                                             inwardHalosArrayAttr);
}

}  // namespace

/*
SOH Overlapped on 3 tiles -> ITIBuffer - Non compact strides

!VPUIP.DistributedBuffer<
    1x64x48x16xi1, {order = #NHWC, strides = [98304, 1, 2048, 128]}, @CMX_NN, {
        mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1],
        num_clusters = 3, kernel = [3, 3], strides = [1, 1],
        pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
}>

to

!OutputITI0 = !VPUIP.ITIBuffer<
    1x64x17x16xi1, {order = #NHWC, strides = [98304, 1, 2048, 128]}, [@CMX_NN, 0],
    inwardHaloRegions = [
        {shape = [1, 64, 1, 16], offset = [0, 0, 16, 0], cluster_id = 0}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 64, 1, 16], offset = [0, 0, 15, 0], cluster_id = 0,
                inwardHaloRegions = [
                    {shape = [1, 64, 1, 16], offset = [0, 0, 0, 0], cluster_id = 1}
                ]
        }
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x64x18x16xi1, {order = #NHWC, strides = [98304, 1, 2048, 128]}, [@CMX_NN, 1],
    inwardHaloRegions = [
        {shape = [1, 64, 1, 16], offset = [0, 0, 0, 0], cluster_id = 1},
        {shape = [1, 64, 1, 16], offset = [0, 0, 17, 0], cluster_id = 1}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 64, 1, 16], offset = [0, 0, 1, 0], cluster_id = 1,
                inwardHaloRegions = [
                    {shape = [1, 64, 1, 16], offset = [0, 0, 16, 0], cluster_id = 0}
                ]
        },
        {
            shape = [1, 64, 1, 16], offset = [0, 0, 16, 0], cluster_id = 1,
                inwardHaloRegions = [
                    {shape = [1, 64, 1, 16], offset = [0, 0, 0, 0], cluster_id = 2}
                ]
        },
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x64x17x16xi1, {order = #NHWC, strides = [98304, 1, 2048, 128]}, [@CMX_NN, 2],
    inwardHaloRegions = [
        {shape = [1, 64, 1, 16], offset = [0, 0, 0, 0], cluster_id = 2}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 64, 1, 16], offset = [0, 0, 1, 0], cluster_id = 2,
                inwardHaloRegions = [
                    {shape = [1, 64, 1, 16], offset = [0, 0, 17, 0], cluster_id = 1}
                ]
        }
]>

*/
TEST_F(MLIR_ITIUnrollTest, getPerClusterOutputHaloBuffers_SOH) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();
    ctx.loadDialect<VPURT::VPURTDialect>();

    const int64_t numClusters = 3;

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, numClusters, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    const auto shape = SmallVector<int64_t>({1, 64, 48, 16});
    const auto elemType = mlir::IntegerType::get(&ctx, 1);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));

    // strides are not compact
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 48 * 2, 1, 64 * 16 * 2, 64 * 2});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto kernel = getIntArrayAttr(&ctx, SmallVector<int64_t>({3, 3}));
    const auto pads = VPU::PaddingAttr::get(&ctx, getIntAttr(&ctx, 1), getIntAttr(&ctx, 1), getIntAttr(&ctx, 1),
                                            getIntAttr(&ctx, 1));
    const auto kernelStrides = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1}));
    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, kernel, pads,
                                                                kernelStrides, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);
    auto ndType = mlir::cast<vpux::NDTypeInterface>(distributedBufferType);

    const auto expectedStrides = ndType.getStrides();

    mlir::OpBuilder builder(&ctx);
    auto clusterOperand = builder.create<VPURT::DeclareBufferOp>(mlir::UnknownLoc::get(&ctx), distributedBufferType,
                                                                 VPURT::BufferSection::CMX_NN, /*byte_offset=*/0);
    VPUX_SCOPE_EXIT {
        clusterOperand->erase();
    };

    SmallVector<mlir::Value> outPerCluster;
    SmallVector<SmallVector<mlir::Value>> outItiPerCluster;
    VPUX_SCOPE_EXIT {
        for (auto value : outPerCluster) {
            value.getDefiningOp()->erase();
        }
    };

    std::tie(outPerCluster, outItiPerCluster) = VPUIP::getPerClusterOutputHaloBuffers(
            &ctx, mlir::UnknownLoc::get(&ctx), "output", clusterOperand, numClusters);

    EXPECT_EQ(outPerCluster.size(), numClusters);
    EXPECT_EQ(outItiPerCluster.size(), numClusters);
    auto haloShapeAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 64, 1, 16});

    for (int64_t clusterId = 0; clusterId < numClusters; clusterId++) {
        auto itiType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(outPerCluster[clusterId].getType());
        EXPECT_TRUE(itiType != nullptr);

        const auto shape = mlir::cast<vpux::NDTypeInterface>(itiType).getShape();
        const auto expectedShape = (clusterId == 0 || clusterId == numClusters - 1) ? vpux::Shape({1, 64, 17, 16})
                                                                                    : vpux::Shape({1, 64, 18, 16});
        EXPECT_EQ(shape, expectedShape);

        const auto strides = mlir::cast<vpux::NDTypeInterface>(itiType).getStrides();
        EXPECT_EQ(strides, expectedStrides);

        // Check inward halos are of expected shape and are placed at the right offset
        auto expectedInwardHalos = SmallVector<VPUIP::HaloRegionAttr>();
        if (clusterId == 0) {
            expectedInwardHalos.push_back(
                    getInwardHalo(&ctx, haloShapeAttr, SmallVector<int64_t>{0, 0, 16, 0}, clusterId));
        } else if (clusterId == numClusters - 1) {
            expectedInwardHalos.push_back(
                    getInwardHalo(&ctx, haloShapeAttr, SmallVector<int64_t>{0, 0, 0, 0}, clusterId));
        } else {
            expectedInwardHalos.push_back(
                    getInwardHalo(&ctx, haloShapeAttr, SmallVector<int64_t>{0, 0, 0, 0}, clusterId));
            expectedInwardHalos.push_back(
                    getInwardHalo(&ctx, haloShapeAttr, SmallVector<int64_t>{0, 0, 17, 0}, clusterId));
        }

        EXPECT_EQ(expectedInwardHalos.size(), itiType.getInwardHaloRegions().size());
        for (const auto expectedHalo : expectedInwardHalos) {
            const auto count = llvm::count(itiType.getInwardHaloRegions(), expectedHalo);
            EXPECT_EQ(count, 1);
        }

        // Check outward halos are of expected shape and start at the right offset. Additionally, ensure
        // the corresponding inward halos have correct offsets.
        auto expectedOutwardHalos = SmallVector<VPUIP::OutwardHaloRegionAttr>();
        if (clusterId == 0) {
            const SmallVector<SmallVector<int64_t>> inwardOffsets = {{0, 0, 0, 0}};
            expectedOutwardHalos.push_back(getOutwardHalo(
                    &ctx, haloShapeAttr, clusterId, SmallVector<int64_t>{0, 0, 15, 0}, inwardOffsets, {clusterId + 1}));
        } else if (clusterId == numClusters - 1) {
            const auto targetOffset = mlir::cast<vpux::NDTypeInterface>(outPerCluster[clusterId - 1].getType())
                                              .getShape()[Dims4D::Act::H] -
                                      1;
            const SmallVector<SmallVector<int64_t>> inwardOffsets = {{0, 0, targetOffset, 0}};
            expectedOutwardHalos.push_back(getOutwardHalo(
                    &ctx, haloShapeAttr, clusterId, SmallVector<int64_t>{0, 0, 1, 0}, inwardOffsets, {clusterId - 1}));
        } else {
            const auto targetOffset = mlir::cast<vpux::NDTypeInterface>(outPerCluster[clusterId - 1].getType())
                                              .getShape()[Dims4D::Act::H] -
                                      1;
            const SmallVector<SmallVector<int64_t>> inwardOffsetsTop = {{0, 0, targetOffset, 0}};
            expectedOutwardHalos.push_back(getOutwardHalo(&ctx, haloShapeAttr, clusterId,
                                                          SmallVector<int64_t>{0, 0, 1, 0}, inwardOffsetsTop,
                                                          {clusterId - 1}));

            const SmallVector<SmallVector<int64_t>> inwardOffsetsBottom = {{0, 0, 0, 0}};
            expectedOutwardHalos.push_back(getOutwardHalo(&ctx, haloShapeAttr, clusterId,
                                                          SmallVector<int64_t>{0, 0, 16, 0}, inwardOffsetsBottom,
                                                          {clusterId + 1}));
        }

        EXPECT_EQ(expectedOutwardHalos.size(), itiType.getOutwardHaloRegions().size());
        for (const auto expectedHalo : expectedOutwardHalos) {
            const auto count = llvm::count(itiType.getOutwardHaloRegions(), expectedHalo);
            EXPECT_EQ(count, 1);
        }

        // Ensure there is a 1:1 match between inward halo in outward halo and one of the inward halos of the
        // output_iti_buffs
        checkOutwardHalosConsumersMatchOutputIti(itiType.getOutwardHaloRegions(), outItiPerCluster[clusterId]);
    }
}

/*
SOK DUPLICATED|SEGMENTED on 2 tiles -> ITIBuffer

!VPUIP.DistributedBuffer<
    1x64x32x16xf16, {order = #NHWC}, @CMX_NN, {
        mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2},
}>

to

!OutputITI0 = !VPUIP.ITIBuffer<
    1x64x32x16xf16, #NHWC, [@CMX_NN, 0],
    inwardHaloRegions = [
        {shape = [1, 32, 32, 16], offset = [0, 32, 0, 0], cluster_id = 0}
    ],
    outwardHaloRegions = [
            shape = [1, 32, 32, 16], offset = [0, 0, 0, 0], cluster_id = 0,
                inwardHaloRegions = [
                    {shape = [1, 32, 32, 16], offset = [0, 0, 0, 0], cluster_id = 1}
                ]
        }
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x64x32x16xf16, #NHWC, [@CMX_NN, 1],
    inwardHaloRegions = [
        {shape = [1, 32, 32, 16], offset = [0, 0, 0, 0], cluster_id = 1}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 32, 32, 16], offset = [0, 32, 0, 0], cluster_id = 1,
                inwardHaloRegions = [
                    {shape = [1, 32, 32, 16], offset = [0, 32, 0, 0], cluster_id = 0}
                ]
        }
]>

*/

TEST_F(MLIR_ITIUnrollTest, getPerClusterOutputHaloBuffers_SOK) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();
    ctx.loadDialect<VPURT::VPURTDialect>();

    const int64_t numClusters = 2;

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, numClusters, 1, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    const auto shape = SmallVector<int64_t>({1, 64, 32, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 32, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);
    auto ndType = mlir::cast<vpux::NDTypeInterface>(distributedBufferType);

    const auto expectedStrides = ndType.getStrides();

    mlir::OpBuilder builder(&ctx);
    auto clusterOperand = builder.create<VPURT::DeclareBufferOp>(mlir::UnknownLoc::get(&ctx), distributedBufferType,
                                                                 VPURT::BufferSection::CMX_NN, /*byte_offset=*/0);
    VPUX_SCOPE_EXIT {
        clusterOperand->erase();
    };

    SmallVector<mlir::Value> outPerCluster;
    SmallVector<SmallVector<mlir::Value>> outItiPerCluster;
    VPUX_SCOPE_EXIT {
        for (auto value : outPerCluster) {
            value.getDefiningOp()->erase();
        }
    };

    std::tie(outPerCluster, outItiPerCluster) = VPUIP::getPerClusterOutputHaloBuffers(
            &ctx, mlir::UnknownLoc::get(&ctx), "output", clusterOperand, numClusters);

    EXPECT_EQ(outPerCluster.size(), numClusters);
    EXPECT_EQ(outItiPerCluster.size(), numClusters);

    const int64_t channelsPerCluster = shape[Dims4D::Act::C.ind()] / numClusters;
    auto haloShapeAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, channelsPerCluster, 32, 16});

    for (int64_t clusterId = 0; clusterId < numClusters; clusterId++) {
        auto itiType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(outPerCluster[clusterId].getType());
        EXPECT_TRUE(itiType != nullptr);

        const auto shape = mlir::cast<vpux::NDTypeInterface>(itiType).getShape();
        EXPECT_EQ(shape, vpux::ShapeRef({1, 64, 32, 16}));

        const auto strides = mlir::cast<vpux::NDTypeInterface>(itiType).getStrides();
        EXPECT_EQ(strides, expectedStrides);

        // Check inward halos are of expected shape and are placed at the right offset
        auto expectedInwardHalos = SmallVector<VPUIP::HaloRegionAttr>();

        for (int64_t prodCluster = 0; prodCluster < numClusters; prodCluster++) {
            if (clusterId == prodCluster) {
                continue;
            }

            expectedInwardHalos.push_back(getInwardHalo(
                    &ctx, haloShapeAttr, SmallVector<int64_t>{0, prodCluster * channelsPerCluster, 0, 0}, clusterId));
        }

        EXPECT_EQ(expectedInwardHalos.size(), itiType.getInwardHaloRegions().size());
        for (const auto expectedHalo : expectedInwardHalos) {
            const auto count = llvm::count(itiType.getInwardHaloRegions(), expectedHalo);
            EXPECT_EQ(count, 1);
        }

        // Check outward halos are of expected shape and start at the right offset. Additionally, ensure
        // the corresponding inward halos have correct offsets.
        auto expectedOutwardHalos = SmallVector<VPUIP::OutwardHaloRegionAttr>();
        SmallVector<int64_t> clustersToBroadcastTo;
        SmallVector<SmallVector<int64_t>> inwardOffsets;
        for (int64_t targetCluster = 0; targetCluster < numClusters; targetCluster++) {
            if (clusterId == targetCluster) {
                continue;
            }

            clustersToBroadcastTo.push_back(targetCluster);
            inwardOffsets.push_back({0, clusterId * channelsPerCluster, 0, 0});
        }

        expectedOutwardHalos.push_back(getOutwardHalo(&ctx, haloShapeAttr, clusterId,
                                                      SmallVector<int64_t>{0, clusterId * channelsPerCluster, 0, 0},
                                                      inwardOffsets, clustersToBroadcastTo));

        EXPECT_EQ(expectedOutwardHalos.size(), itiType.getOutwardHaloRegions().size());
        for (const auto expectedHalo : expectedOutwardHalos) {
            const auto count = llvm::count(itiType.getOutwardHaloRegions(), expectedHalo);
            EXPECT_EQ(count, 1);
        }

        // Ensure there is a 1:1 match between inward halo in outward halo and one of the inward halos of the
        // output_iti_buffs
        checkOutwardHalosConsumersMatchOutputIti(itiType.getOutwardHaloRegions(), outItiPerCluster[clusterId]);
    }
}

/*
SOK SEGMENTED|MULTICASTED on 4 tiles -> ITIBuffer

!VPUIP.DistributedBuffer<
    1x64x61x16xf16, {order = #NHWC}, @CMX_NN, {
        mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 4, 1], num_clusters = 4},
}>

to

!OutputITI0 = !VPUIP.ITIBuffer<
    1x64x61x16xf16, #NHWC, [@CMX_NN, 0],
    inwardHaloRegions = [
        {shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 0},
        {shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 0},
        {shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 0}
    ],
    outwardHaloRegions = [
            shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 0,
                inwardHaloRegions = [
                    {shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1},
                    {shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 2},
                    {shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 3}
                ]
        }
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x64x61x16xf16, #NHWC, [@CMX_NN, 1],
    inwardHaloRegions = [
        {shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1},
        {shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 1},
        {shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 1}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 1,
                inwardHaloRegions = [
                    {shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 0},
                    {shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 2},
                    {shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 3}
                ]
        }
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x64x61x16xf16, #NHWC, [@CMX_NN, 2],
    inwardHaloRegions = [
        {shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 2},
        {shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 2},
        {shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 2}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 2,
                inwardHaloRegions = [
                    {shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 0},
                    {shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 1},
                    {shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 3}
                ]
        }
]>

!OutputITI3 = !VPUIP.ITIBuffer<
    1x64x61x16xf16, #NHWC, [@CMX_NN, 3],
    inwardHaloRegions = [
        {shape = [1, 64, 16, 16], offset = [0, 0, 0, 0], cluster_id = 3},
        {shape = [1, 64, 16, 16], offset = [0, 0, 16, 0], cluster_id = 3},
        {shape = [1, 64, 16, 16], offset = [0, 0, 32, 0], cluster_id = 3}
    ],
    outwardHaloRegions = [
        {
            shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 1,
                inwardHaloRegions = [
                    {shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 0},
                    {shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 1},
                    {shape = [1, 64, 13, 16], offset = [0, 0, 48, 0], cluster_id = 2}
                ]
        }
]>

*/

TEST_F(MLIR_ITIUnrollTest, getPerClusterOutputHaloBuffers_HKSwitch) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();
    ctx.loadDialect<VPURT::VPURTDialect>();

    const int64_t numClusters = 4;

    const auto distributionModeAttr =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
    const auto numTilesAttr = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, numClusters, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    const auto shape = SmallVector<int64_t>({1, 64, 61, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 61, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(&ctx, distributionModeAttr, numTilesAttr, nullptr,
                                                                nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);
    const auto distributedBufferType =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, layout, dimsSpace, distributedAttr);
    auto ndType = mlir::cast<vpux::NDTypeInterface>(distributedBufferType);

    const auto expectedStrides = ndType.getStrides();

    mlir::OpBuilder builder(&ctx);
    auto clusterOperand = builder.create<VPURT::DeclareBufferOp>(mlir::UnknownLoc::get(&ctx), distributedBufferType,
                                                                 VPURT::BufferSection::CMX_NN, /*byte_offset=*/0);
    VPUX_SCOPE_EXIT {
        clusterOperand->erase();
    };

    SmallVector<mlir::Value> outPerCluster;
    SmallVector<SmallVector<mlir::Value>> outItiPerCluster;
    VPUX_SCOPE_EXIT {
        for (auto value : outPerCluster) {
            value.getDefiningOp()->erase();
        }
    };

    std::tie(outPerCluster, outItiPerCluster) = VPUIP::getPerClusterOutputHaloBuffers(
            &ctx, mlir::UnknownLoc::get(&ctx), "output", clusterOperand, numClusters);

    EXPECT_EQ(outPerCluster.size(), numClusters);
    EXPECT_EQ(outItiPerCluster.size(), numClusters);

    const SmallVector<int64_t> linesPerCluster = {16, 16, 16, 13};

    for (int64_t clusterId = 0; clusterId < numClusters; clusterId++) {
        auto itiType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(outPerCluster[clusterId].getType());
        EXPECT_TRUE(itiType != nullptr);

        const auto shape = mlir::cast<vpux::NDTypeInterface>(itiType).getShape();
        EXPECT_EQ(shape, vpux::ShapeRef({1, 64, 61, 16}));

        const auto strides = mlir::cast<vpux::NDTypeInterface>(itiType).getStrides();
        EXPECT_EQ(strides, expectedStrides);

        // Check inward halos are of expected shape and are placed at the right offset
        auto expectedInwardHalos = SmallVector<VPUIP::HaloRegionAttr>();
        for (int64_t prodCluster = 0; prodCluster < numClusters; prodCluster++) {
            if (clusterId == prodCluster) {
                continue;
            }

            auto inwardHaloShapeAttr =
                    getIntArrayAttr(&ctx, SmallVector<int64_t>{shape[Dims4D::Act::N], shape[Dims4D::Act::C],
                                                               linesPerCluster[prodCluster], shape[Dims4D::Act::W]});

            expectedInwardHalos.push_back(getInwardHalo(&ctx, inwardHaloShapeAttr,
                                                        SmallVector<int64_t>{0, 0, prodCluster * linesPerCluster[0], 0},
                                                        clusterId));
        }

        EXPECT_EQ(expectedInwardHalos.size(), itiType.getInwardHaloRegions().size());
        for (const auto expectedHalo : expectedInwardHalos) {
            const auto count = llvm::count(itiType.getInwardHaloRegions(), expectedHalo);
            EXPECT_EQ(count, 1);
        }

        // Check outward halos are of expected shape and start at the right offset. Additionally, ensure
        // the corresponding inward halos have correct offsets.
        auto expectedOutwardHalos = SmallVector<VPUIP::OutwardHaloRegionAttr>();
        SmallVector<int64_t> clustersToBroadcastTo;
        SmallVector<SmallVector<int64_t>> inwardOffsets;

        auto haloShapeAttr =
                getIntArrayAttr(&ctx, SmallVector<int64_t>{shape[Dims4D::Act::N], shape[Dims4D::Act::C],
                                                           linesPerCluster[clusterId], shape[Dims4D::Act::W]});
        const auto sliceOffset = SmallVector<int64_t>{0, 0, clusterId * linesPerCluster[0], 0};
        for (int64_t targetCluster = 0; targetCluster < numClusters; targetCluster++) {
            if (clusterId == targetCluster) {
                continue;
            }

            clustersToBroadcastTo.push_back(targetCluster);
            inwardOffsets.push_back(sliceOffset);
        }

        expectedOutwardHalos.push_back(
                getOutwardHalo(&ctx, haloShapeAttr, clusterId, sliceOffset, inwardOffsets, clustersToBroadcastTo));

        EXPECT_EQ(expectedOutwardHalos.size(), itiType.getOutwardHaloRegions().size());
        for (const auto expectedHalo : expectedOutwardHalos) {
            const auto count = llvm::count(itiType.getOutwardHaloRegions(), expectedHalo);
            EXPECT_EQ(count, 1);
        }

        // Ensure there is a 1:1 match between inward halo in outward halo and one of the inward halos of the
        // output_iti_buffs
        checkOutwardHalosConsumersMatchOutputIti(itiType.getOutwardHaloRegions(), outItiPerCluster[clusterId]);
    }
}
