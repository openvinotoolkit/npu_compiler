//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

using CostModelShaveUtil = MLIR_UnitBase;

TEST_F(CostModelShaveUtil, testCreateShaveCostModelUtilsAndCheckOpNPU37) {
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU37XX);
    VPU::CostModelConfig::setCMShaveUtils(config::ArchKind::NPU37XX);

    const auto& shaveUtils = VPU::CostModelConfig::getShaveCostModelUtilsInterface(config::ArchKind::NPU37XX);

    EXPECT_TRUE(shaveUtils.isSwKernelOpSupported("SoftMax"));
    EXPECT_FALSE(shaveUtils.isSwKernelOpSupported("NonExistOp"));
    EXPECT_EQ(shaveUtils.getSwKernelContainer().size(), 73);  // Total size of Shave1 API
}

TEST_F(CostModelShaveUtil, testCreateShaveCostModelUtilsAndCheckOpNPU40) {
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU40XX);
    VPU::CostModelConfig::setCMShaveUtils(config::ArchKind::NPU40XX);

    const auto& shaveUtils = VPU::CostModelConfig::getShaveCostModelUtilsInterface(config::ArchKind::NPU40XX);

    EXPECT_TRUE(shaveUtils.isSwKernelOpSupported("SoftMax"));
    EXPECT_FALSE(shaveUtils.isSwKernelOpSupported("NonExistOp"));
    EXPECT_EQ(shaveUtils.getSwKernelContainer().size(), 73);  // Total size of Shave1 API
}

TEST_F(CostModelShaveUtil, testCreateShaveCostModelUtilsAndCheckOpNPU50) {
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU50XX);
    VPU::CostModelConfig::setCMShaveUtils(config::ArchKind::NPU50XX);

    const auto& shaveUtils = VPU::CostModelConfig::getShaveCostModelUtilsInterface(config::ArchKind::NPU50XX);

    EXPECT_TRUE(shaveUtils.isSwKernelOpSupported("SoftMax"));
    EXPECT_FALSE(shaveUtils.isSwKernelOpSupported("NonExistOp"));
    EXPECT_EQ(shaveUtils.getSwKernelContainer().size(), 73);  // Total size of Shave1 API
}
