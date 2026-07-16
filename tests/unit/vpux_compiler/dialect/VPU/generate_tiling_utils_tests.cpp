//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"

using namespace vpux;

using MLIR_VPU_Generate_Tiling = MLIR_UnitBase;

TEST_F(MLIR_VPU_Generate_Tiling, Calculate_Workload_Number_NPU37XX) {
    VPU::registerStrategies(registry, config::Platform::NPU3720);
    mlir::MLIRContext ctx(registry);

    SmallVector<Shape> shapesOf2Clusters{{1, 2048, 16, 16}, {1, 2032, 16, 16}};
    SmallVector<int64_t> supportedChannels{64, 32, 16};

    mlir::Location loc = mlir::UnknownLoc::get(&ctx);
    // Create a dummy op for testing
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    mlir::Operation* op = module.get();
    auto workloadInformationFor2Clusters =
            VPU::getWorkLoadInformationForNCEWithSparseOutput(op, shapesOf2Clusters, supportedChannels);
    auto [maxNumPerClusterFor2Clusters, wlNumInTotalFor2Clusters] = workloadInformationFor2Clusters.value();
    // cluster-0: 32, 32,..., 32, 32
    // cluster-1: 32, 32,..., 32, 16
    EXPECT_EQ(maxNumPerClusterFor2Clusters, 64);
    EXPECT_EQ(wlNumInTotalFor2Clusters, 128);

    SmallVector<Shape> shapesOf2ClustersWorseCase1{{1, 112, 16, 16}, {1, 112, 16, 16}};
    auto workloadInformationFor2ClustersWorseCase1 =
            VPU::getWorkLoadInformationForNCEWithSparseOutput(op, shapesOf2ClustersWorseCase1, supportedChannels);
    auto [maxNumPerClusterFor2ClustersWorseCase1, wlNumInTotalFor2ClustersWorseCase1] =
            workloadInformationFor2ClustersWorseCase1.value();
    // cluster-0: 16, 16, 16, 16, 16, 16, 16
    // cluster-1: 16, 16, 16, 16, 16, 16, 16
    EXPECT_EQ(maxNumPerClusterFor2ClustersWorseCase1, 7);
    EXPECT_EQ(wlNumInTotalFor2ClustersWorseCase1, 14);

    SmallVector<Shape> shapesOf2ClustersWorseCase2{{1, 80, 16, 16}, {1, 80, 16, 16}};
    auto workloadInformationFor2ClustersWorseCase2 =
            VPU::getWorkLoadInformationForNCEWithSparseOutput(op, shapesOf2ClustersWorseCase2, supportedChannels);
    auto [maxNumPerClusterFor2ClustersWorseCase2, wlNumInTotalFor2ClustersWorseCase2] =
            workloadInformationFor2ClustersWorseCase2.value();
    // cluster-0: 16, 16, 16, 16, 16
    // cluster-1: 16, 16, 16, 16, 16
    EXPECT_EQ(maxNumPerClusterFor2ClustersWorseCase2, 5);
    EXPECT_EQ(wlNumInTotalFor2ClustersWorseCase2, 10);
}

TEST_F(MLIR_VPU_Generate_Tiling, Calculate_Workload_Number_NPU40XX) {
    VPU::registerStrategies(registry, config::Platform::NPU4000);
    mlir::MLIRContext ctx(registry);

    SmallVector<Shape> shapesOf6Clusters{{1, 128, 16, 16}, {1, 128, 16, 16}, {1, 128, 16, 16},
                                         {1, 128, 16, 16}, {1, 128, 16, 16}, {1, 96, 16, 16}};
    SmallVector<int64_t> supportedChannels{64, 32, 16};

    mlir::Location loc = mlir::UnknownLoc::get(&ctx);
    // Create a dummy op for testing
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    mlir::Operation* op = module.get();
    auto workloadInformationFor6Clusters =
            VPU::getWorkLoadInformationForNCEWithSparseOutput(op, shapesOf6Clusters, supportedChannels);
    auto [maxNumPerClusterFor6Clusters, wlNumInTotalFor6Clusters] = workloadInformationFor6Clusters.value();
    // cluster-0: 64, 64   cluster-1: 64, 64 cluster-2: 64, 64
    // cluster-3: 64, 64   cluster-4: 64, 64 cluster-5: 64, 32
    EXPECT_EQ(maxNumPerClusterFor6Clusters, 2);
    EXPECT_EQ(wlNumInTotalFor6Clusters, 12);

    SmallVector<Shape> shapesOf6ClustersCase1{{1, 128, 16, 16}, {1, 128, 16, 16}, {1, 128, 16, 16},
                                              {1, 128, 16, 16}, {1, 128, 16, 16}, {1, 32, 16, 16}};
    auto workloadInformationFor6ClustersCase1 =
            VPU::getWorkLoadInformationForNCEWithSparseOutput(op, shapesOf6ClustersCase1, supportedChannels);
    auto [maxNumPerClusterFor6ClustersCase1, wlNumInTotalFor6ClustersCase1] =
            workloadInformationFor6ClustersCase1.value();
    // cluster-0: 64, 64   cluster-1: 64, 64 cluster-2: 64, 64
    // cluster-3: 64, 64   cluster-4: 64, 64 cluster-5: 32
    EXPECT_EQ(maxNumPerClusterFor6ClustersCase1, 2);
    EXPECT_EQ(wlNumInTotalFor6ClustersCase1, 11);
}
