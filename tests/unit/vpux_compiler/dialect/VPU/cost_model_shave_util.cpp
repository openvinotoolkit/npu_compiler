//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

class CostModelShaveUtil : public MLIR_UnitBase {
public:
    CostModelShaveUtil(): MLIR_UnitBase() {
    }

    void setFactory(config::Platform platform) {
        ctx.loadDialect<vpux::VPU::VPUDialect>();
        VPU::initializeSingletons(registry, platform);

        ctx.appendDialectRegistry(registry);
    }

    mlir::MLIRContext ctx;
};

TEST_F(CostModelShaveUtil, testCreateShaveCostModelUtilsAndCheckOpNPU37) {
    setFactory(config::Platform::NPU3720);

    const auto& shaveUtils = VPU::getShaveCostModelUtils(&ctx);

    EXPECT_TRUE(shaveUtils.isSwKernelOpSupported("SoftMax"));
    EXPECT_FALSE(shaveUtils.isSwKernelOpSupported("NonExistOp"));
}

TEST_F(CostModelShaveUtil, testCreateShaveCostModelUtilsAndCheckOpNPU40) {
    setFactory(config::Platform::NPU4000);

    const auto& shaveUtils = VPU::getShaveCostModelUtils(&ctx);

    EXPECT_TRUE(shaveUtils.isSwKernelOpSupported("SoftMax"));
    EXPECT_FALSE(shaveUtils.isSwKernelOpSupported("NonExistOp"));
}

TEST_F(CostModelShaveUtil, testCreateShaveCostModelUtilsAndCheckOpNPU50) {
    setFactory(config::Platform::NPU5010);

    const auto& shaveUtils = VPU::getShaveCostModelUtils(&ctx);

    EXPECT_TRUE(shaveUtils.isSwKernelOpSupported("SoftMax"));
    EXPECT_FALSE(shaveUtils.isSwKernelOpSupported("NonExistOp"));
}
