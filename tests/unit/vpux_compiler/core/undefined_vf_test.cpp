//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_vf.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_UndefinedVF.*"

using namespace vpux;
using namespace vpux::VPU;

class MLIR_UndefinedVF : public testing::Test {
protected:
    mlir::MLIRContext _ctx;
    mlir::OpBuilder _builder{&_ctx};
    std::unique_ptr<mlir::Block> _block;

    MLIR_UndefinedVF() {
        auto registry = vpux::createDialectRegistry();
        _ctx.appendDialectRegistry(registry);
        _ctx.loadAllAvailableDialects();

        _block = std::make_unique<mlir::Block>();
        _builder.setInsertionPointToStart(_block.get());
    }

    mlir::Value createBuffer(vpux::AddressType rawSize, vpux::AddressType rawAlign) {
        auto shape = SmallVector<int64_t>{static_cast<int64_t>(rawSize)};
        auto type = mlir::MemRefType::get(shape, _builder.getIntegerType(8));
        auto alloc = _builder.create<mlir::memref::AllocOp>(_builder.getUnknownLoc(), type);
        if (rawAlign != 0) {
            alloc.setAlignment(rawAlign);
        }
        return alloc;
    }

    OpAllocationInfo createComputeOp(size_t opIdx, const SmallVector<mlir::Value>& inBuffers,
                                     const SmallVector<mlir::Value>& outBuffers) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DPU, 0};
        return OpAllocationInfo(opIdx, qt, inBuffers, outBuffers, AllocationType::COMPUTE);
    }

    ComputeRegion makeComputeRegion(LoopType type) {
        auto schedulingLoop = std::make_unique<SchedulingLoop>();
        schedulingLoop->type = type;

        LoopBody iteration0;
        {
            auto inBuf = createBuffer(1024, 64);
            auto outBuf = createBuffer(1024, 64);
            iteration0.push_back(createComputeOp(0, {inBuf}, {outBuf}));
        }
        LoopBody iteration1;
        {
            auto inBuf = createBuffer(1024, 64);
            auto outBuf = createBuffer(1024, 64);
            iteration1.push_back(createComputeOp(1, {inBuf}, {outBuf}));
        }

        schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
        return ComputeRegion(std::move(schedulingLoop));
    }
};

TEST_F(MLIR_UndefinedVF, NameAndConstruction) {
    UndefinedVF scenario;
    EXPECT_EQ(scenario.getName(), "UndefinedVF");
}

TEST_F(MLIR_UndefinedVF, EmptyResultOnVfRegion) {
    auto region = makeComputeRegion(LoopType::VF);
    UndefinedVF scenario;
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/8192);

    EXPECT_TRUE(result.empty());
    EXPECT_EQ(result.reservedSize, 0u);
    EXPECT_TRUE(result.sharedExternalBuffers.empty());
}

TEST_F(MLIR_UndefinedVF, ThrowsOnNonVfRegion) {
    UndefinedVF scenario;
    {
        auto region = makeComputeRegion(LoopType::Tiling);
        EXPECT_ANY_THROW(scenario.getScheduleStrategy(region, /*memorySize=*/8192));
    }
    {
        auto region = makeComputeRegion(LoopType::None);
        EXPECT_ANY_THROW(scenario.getScheduleStrategy(region, /*memorySize=*/8192));
    }
}
