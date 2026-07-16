//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include "common/utils.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_MCStrategy_Getter = MLIR_UnitBase;

TEST_F(MLIR_MCStrategy_Getter, MCGetterListNPU37XX) {
    VPU::registerStrategies(registry, config::Platform::NPU3720);
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClusters = 2;

    SmallVector<VPU::MultiClusterStrategy> strategyNPU37XXSet;
    const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
    auto mcGetter = strategyFactory->getMultiClusterStrategy(numClusters);

    mcGetter->getMCStrategies(strategyNPU37XXSet);
    EXPECT_EQ(strategyNPU37XXSet.size(), 5);

    SmallVector<VPU::MultiClusterStrategy> strategyNPU37XX1TileSet;
    mcGetter = strategyFactory->getMultiClusterStrategy(1);

    mcGetter->getMCStrategies(strategyNPU37XX1TileSet);
    EXPECT_EQ(strategyNPU37XX1TileSet.size(), 1);
}

TEST_F(MLIR_MCStrategy_Getter, MCGetterListNPU40XX) {
    VPU::registerStrategies(registry, config::Platform::NPU4000);
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClusters = 2;

    SmallVector<VPU::MultiClusterStrategy> strategyVPU40XX2TilesSet;
    const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
    auto mcGetter = strategyFactory->getMultiClusterStrategy(numClusters);

    mcGetter->getMCStrategies(strategyVPU40XX2TilesSet);
    EXPECT_EQ(strategyVPU40XX2TilesSet.size(), 6);

    SmallVector<VPU::MultiClusterStrategy> strategyVPU40XX6TilesSet;
    mcGetter = strategyFactory->getMultiClusterStrategy(6);

    mcGetter->getMCStrategies(strategyVPU40XX6TilesSet);
    EXPECT_EQ(strategyVPU40XX6TilesSet.size(), 8);
}
