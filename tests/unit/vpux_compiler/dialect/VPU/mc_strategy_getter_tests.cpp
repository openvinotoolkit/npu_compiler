//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/mc_strategy_getter.hpp"

#include "common/utils.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_MCStrategy_Getter = MLIR_UnitBase;

TEST_F(MLIR_MCStrategy_Getter, MCGetterList) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClusters = 2;

    SmallVector<VPU::MultiClusterStrategy> strategyNPU37XXSet;
    auto mcGetter = VPU::createMCStrategyGetter(config::ArchKind::NPU37XX, numClusters);

    mcGetter->getMCStrategies(strategyNPU37XXSet);
    EXPECT_EQ(strategyNPU37XXSet.size(), 5);

    SmallVector<VPU::MultiClusterStrategy> strategyNPU37XX1TileSet;
    mcGetter = VPU::createMCStrategyGetter(config::ArchKind::NPU37XX, 1);

    mcGetter->getMCStrategies(strategyNPU37XX1TileSet);
    EXPECT_EQ(strategyNPU37XX1TileSet.size(), 1);

    SmallVector<VPU::MultiClusterStrategy> strategyVPU40XX2TilesSet;
    mcGetter = VPU::createMCStrategyGetter(config::ArchKind::NPU40XX, numClusters);

    mcGetter->getMCStrategies(strategyVPU40XX2TilesSet);
    EXPECT_EQ(strategyVPU40XX2TilesSet.size(), 6);

    SmallVector<VPU::MultiClusterStrategy> strategyVPU40XX6TilesSet;
    mcGetter = VPU::createMCStrategyGetter(config::ArchKind::NPU40XX, 6);

    mcGetter->getMCStrategies(strategyVPU40XX6TilesSet);
    EXPECT_EQ(strategyVPU40XX6TilesSet.size(), 8);
}
