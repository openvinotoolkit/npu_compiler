//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_tiling.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

#include <map>

// Run cmd: npuUnitTests --gtest_filter="MLIR_UndefinedTiling.*"

using namespace vpux;

class MLIR_UndefinedTiling : public testing::Test {
protected:
    struct ComputeAllocDeallocByOp {
        std::map<size_t, SmallVector<vpux::AddressType>> allocations;
        std::map<size_t, SmallVector<mlir::Value>> deallocations;
    };

    mlir::MLIRContext _ctx;
    mlir::OpBuilder _builder{&_ctx};
    std::unique_ptr<mlir::Block> _block;

    MLIR_UndefinedTiling() {
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

    ComputeAllocDeallocByOp collectComputeAllocationsByOp(const PredefinedSchedule& schedule) {
        ComputeAllocDeallocByOp byOpIdx;

        for (const auto& iterationSchedule : schedule) {
            for (const auto& explicitStep : iterationSchedule) {
                if (explicitStep.allocInfo.allocationType != AllocationType::COMPUTE) {
                    continue;
                }

                SmallVector<vpux::AddressType> addresses;
                for (const auto& [_, address] : explicitStep.allocations) {
                    addresses.push_back(address);
                }

                byOpIdx.allocations[explicitStep.allocInfo.opIdx] = std::move(addresses);
                byOpIdx.deallocations[explicitStep.allocInfo.opIdx] = explicitStep.deallocations;
            }
        }

        return byOpIdx;
    }
};

TEST_F(MLIR_UndefinedTiling, BasicTilingUniformSizes) {
    // Case 1: Basic uniform tiling (full two-set allocation)
    //
    // ┌───────────────────────────┐
    // │ set 1: out (1024)         │ 3072
    // │ set 1: in  (1024)         │ 2048
    // ├───────────────────────────┤
    // │ set 0: out (1024)         │ 1024
    // │ set 0: in  (1024)         │ 0
    // └───────────────────────────┘
    //
    // - 2 iterations, each with [in=1024, out=1024]
    // Expected:
    //   iter 0 -> [0, 1024]
    //   iter 1 -> second set (>= 2048)
    //   total local size = 4096

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

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

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/8192);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 2);

    ASSERT_EQ(allocations.at(0).size(), 2);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);

    ASSERT_EQ(allocations.at(1).size(), 2);
    EXPECT_GE(allocations.at(1)[0], 2048);

    EXPECT_EQ(scheduleResult.reservedSize, 4096);
}

TEST_F(MLIR_UndefinedTiling, TilingUnevenSizes) {
    // Case 2: Uneven tile sizes (max-slot normalization)
    //
    // Iterations:
    //   iter 0: [in=1024, out=1024]
    //   iter 1: [in=1024, out=1024]
    //   iter 2: [in= 512, out= 512]
    //
    // Slot size is chosen by per-operand maximum, so both slots remain 1024.
    //
    // ┌───────────────────────────┐
    // │ set 1: out slot (1024)    │ 3072
    // │ set 1: in  slot (1024)    │ 2048
    // ├───────────────────────────┤
    // │ set 0: out slot (1024)    │ 1024
    // │ set 0: in  slot (1024)    │ 0
    // └───────────────────────────┘
    //
    // Expected: iter 2 reuses iter 0 addresses, total local size = 4096.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

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

    LoopBody iteration2;
    {
        auto inBuf = createBuffer(512, 64);
        auto outBuf = createBuffer(512, 64);
        iteration2.push_back(createComputeOp(2, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 3);

    ASSERT_EQ(allocations.at(0).size(), 2);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);

    ASSERT_EQ(allocations.at(2).size(), 2);
    EXPECT_EQ(allocations.at(2)[0], 0);
    EXPECT_EQ(allocations.at(2)[1], 1024);

    EXPECT_EQ(scheduleResult.reservedSize, 4096);
}

TEST_F(MLIR_UndefinedTiling, TilingNoPrefetch) {
    // Case 3: No prefetching (single set only)
    //
    // ┌───────────────────────────┐ ← memorySize=2048
    // │ out (1024)                │ 1024
    // │ in  (1024)                │ 0
    // └───────────────────────────┘
    //
    // - 4 iterations, each [in=1024, out=1024]
    // - No room for set 1, so all iterations reuse set 0.
    //
    // Expected:
    //   iter 0/1/2/3 -> [0, 1024]

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

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

    LoopBody iteration2;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {inBuf}, {outBuf}));
    }

    LoopBody iteration3;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/2048);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    for (size_t opIdx = 0; opIdx < 4; ++opIdx) {
        ASSERT_EQ(allocations.at(opIdx).size(), 2);
        EXPECT_EQ(allocations.at(opIdx)[0], 0);
        EXPECT_EQ(allocations.at(opIdx)[1], 1024);
    }

    EXPECT_EQ(scheduleResult.reservedSize, 2048);
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.empty());
}

TEST_F(MLIR_UndefinedTiling, TilingSharedBufferWithPartialPrefetch) {
    // Case 4: Shared buffer + partial prefetch
    //
    // Shared region:
    //   weight = 2048 (reserved once for all iterations)
    //
    // Local region (available = 7168 - 2048 = 5120):
    // ┌───────────────────────────┐
    // │ set 1: actIn1 (1024)      │ 4096
    // │ set 1: actIn0 (1024)      │ 3072
    // ├───────────────────────────┤
    // │ set 0: out   (1024)       │ 2048
    // │ set 0: actIn1(1024)       │ 1024
    // │ set 0: actIn0(1024)       │ 0
    // └───────────────────────────┘
    //
    // set 1 cannot fit all 3 locals, so output reuses set 0 slot.
    // Expected pattern:
    //   even iters  -> [0, 1024, 2048]
    //   odd  iters  -> [3072, 4096, 2048]

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto weightBuf = createBuffer(2048, 64);

    LoopBody iteration0;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    LoopBody iteration2;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    LoopBody iteration3;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/7168);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    ASSERT_EQ(allocations.at(0).size(), 3);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);
    EXPECT_EQ(allocations.at(0)[2], 2048);

    ASSERT_EQ(allocations.at(1).size(), 3);
    EXPECT_EQ(allocations.at(1)[0], 3072);
    EXPECT_EQ(allocations.at(1)[1], 4096);
    EXPECT_EQ(allocations.at(1)[2], 2048);

    ASSERT_EQ(allocations.at(2).size(), 3);
    EXPECT_EQ(allocations.at(2)[0], 0);
    EXPECT_EQ(allocations.at(2)[1], 1024);
    EXPECT_EQ(allocations.at(2)[2], 2048);

    ASSERT_EQ(allocations.at(3).size(), 3);
    EXPECT_EQ(allocations.at(3)[0], 3072);
    EXPECT_EQ(allocations.at(3)[1], 4096);
    EXPECT_EQ(allocations.at(3)[2], 2048);

    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    EXPECT_EQ(scheduleResult.reservedSize, 5120);
}

TEST_F(MLIR_UndefinedTiling, TilingFullPipelining) {
    // Case 5: Full pipelining with shared weight
    //
    // Shared region:
    //   weight = 2048 (shared across all iterations)
    //
    // Local ping-pong region:
    // ┌───────────────────────────┐
    // │ set 1: out (1024)         │ 3072
    // │ set 1: in  (1024)         │ 2048
    // ├───────────────────────────┤
    // │ set 0: out (1024)         │ 1024
    // │ set 0: in  (1024)         │ 0
    // └───────────────────────────┘
    //
    // Expected A/B/A/B pattern:
    //   iter 0 -> [0, 1024]
    //   iter 1 -> [2048, 3072]
    //   iter 2 -> [0, 1024]
    //   iter 3 -> [2048, 3072]

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto weightBuf = createBuffer(2048, 64);

    LoopBody iteration0;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {weightBuf, actInBuf}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {weightBuf, actInBuf}, {outBuf}));
    }

    LoopBody iteration2;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {weightBuf, actInBuf}, {outBuf}));
    }

    LoopBody iteration3;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {weightBuf, actInBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    ASSERT_EQ(allocations.at(0).size(), 2);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);

    ASSERT_EQ(allocations.at(1).size(), 2);
    EXPECT_EQ(allocations.at(1)[0], 2048);
    EXPECT_EQ(allocations.at(1)[1], 3072);

    ASSERT_EQ(allocations.at(2).size(), 2);
    EXPECT_EQ(allocations.at(2)[0], 0);
    EXPECT_EQ(allocations.at(2)[1], 1024);

    ASSERT_EQ(allocations.at(3).size(), 2);
    EXPECT_EQ(allocations.at(3)[0], 2048);
    EXPECT_EQ(allocations.at(3)[1], 3072);

    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    EXPECT_EQ(scheduleResult.reservedSize, 4096);
}

TEST_F(MLIR_UndefinedTiling, SharedDedupAcrossOperands) {
    // Case 6: Shared dedup across operand positions
    //
    //   op inputs: [sharedBuf, sharedBuf]
    //
    // Shared reservation:
    // ┌───────────────────────────┐
    // │ sharedBuf (1536)          │ reserved once
    // └───────────────────────────┘
    //
    // Local region only needs output slot:
    // ┌───────────────────────────┐
    // │ out slot (512)            │
    // └───────────────────────────┘
    //
    // Expected: one local allocation per compute op, sharedExternalBuffers=1.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto sharedBuf = createBuffer(1536, 64);

    LoopBody iteration0;
    {
        auto outBuf = createBuffer(512, 64);
        iteration0.push_back(createComputeOp(0, {sharedBuf, sharedBuf}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto outBuf = createBuffer(512, 64);
        iteration1.push_back(createComputeOp(1, {sharedBuf, sharedBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/2560);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 2);
    ASSERT_EQ(allocations.at(0).size(), 1);
    ASSERT_EQ(allocations.at(1).size(), 1);

    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    EXPECT_EQ(scheduleResult.reservedSize, 1024);
}

TEST_F(MLIR_UndefinedTiling, PerOperandGlobalMaxSlot) {
    // Per-operand global max slot (different max size per operand index)
    //
    // Iteration buffers:
    //   iter 0: in=4096, out=512
    //   iter 1: in=1024, out=2048
    //
    // Slot size is computed per operand index:
    //   operand 0 (input slot):  max(4096, 1024) = 4096
    //   operand 1 (output slot): max( 512, 2048) = 2048
    //
    // So one local set size is:
    //   setSize = 4096 + 2048 = 6144
    //
    // Address layout (local region):
    // ┌───────────────────────────┐
    // │ set 1: output slot (2048) │ starts at 10240
    // │ set 1: input  slot (4096) │ starts at 6144
    // ├───────────────────────────┤
    // │ set 0: output slot (2048) │ starts at 4096
    // │ set 0: input  slot (4096) │ starts at 0
    // └───────────────────────────┘
    //
    // Expected addresses:
    //   set 0 => [input=0,    output=4096]
    //   set 1 => [input=6144, output=10240]

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    LoopBody iteration0;
    {
        auto inBuf = createBuffer(4096, 64);
        auto outBuf = createBuffer(512, 64);
        iteration0.push_back(createComputeOp(0, {inBuf}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(2048, 64);
        iteration1.push_back(createComputeOp(1, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/12288);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 2);

    ASSERT_EQ(allocations.at(0).size(), 2);
    ASSERT_EQ(allocations.at(1).size(), 2);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 4096);
    EXPECT_EQ(allocations.at(1)[0], 6144);
    EXPECT_EQ(allocations.at(1)[1], 10240);
    EXPECT_EQ(scheduleResult.reservedSize, 12288);
}

TEST_F(MLIR_UndefinedTiling, OOMThrowsFailFast) {
    // Case 8: OOM during shared reservation
    //
    // Required shared region:
    // ┌───────────────────────────┐
    // │ sharedBuf (2048)          │
    // └───────────────────────────┘
    // But memorySize=1024.
    //
    // Expected: fail fast with vpux::Exception before local allocation.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto sharedBuf = createBuffer(2048, 64);

    LoopBody iteration0;
    {
        auto outBuf = createBuffer(512, 64);
        iteration0.push_back(createComputeOp(0, {sharedBuf}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto outBuf = createBuffer(512, 64);
        iteration1.push_back(createComputeOp(1, {sharedBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;

    EXPECT_THROW(static_cast<void>(scenario.getScheduleStrategy(region, /*memorySize=*/1024)), vpux::Exception);
}

TEST_F(MLIR_UndefinedTiling, FlashSDPALayoutFullPipelining) {
    // Case 9: FlashSDPA full pipelining
    //
    // Shared buffers:
    //   K, V, const1, const2, const3, stateAliasA, stateAliasB, stateAliasC (8 total)
    //
    // Local per-tile buffers: q=1024, mask=1024, outA=1024
    //
    // Local layout (two full sets):
    // ┌───────────────────────────┐
    // │ set 1: outA (1024)        │ 5120
    // │ set 1: mask (1024)        │ 4096
    // │ set 1: q    (1024)        │ 3072
    // ├───────────────────────────┤
    // │ set 0: outA (1024)        │ 2048
    // │ set 0: mask (1024)        │ 1024
    // │ set 0: q    (1024)        │ 0
    // └───────────────────────────┘
    //
    // Expected A/B/A/B for local addresses and sharedExternalBuffers=8.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto sharedK = createBuffer(512, 64);
    auto sharedV = createBuffer(512, 64);
    auto sharedConst1 = createBuffer(256, 64);
    auto sharedConst2 = createBuffer(256, 64);
    auto sharedConst3 = createBuffer(256, 64);

    auto stateAliasA = createBuffer(1024, 64);
    auto stateAliasB = createBuffer(1024, 64);
    auto stateAliasC = createBuffer(1024, 64);

    std::vector<LoopBody> loopBodies;
    loopBodies.reserve(4);

    for (size_t tileIdx = 0; tileIdx < 4; ++tileIdx) {
        auto q = createBuffer(1024, 64);
        auto mask = createBuffer(1024, 64);
        auto outA = createBuffer(1024, 64);

        LoopBody body;
        body.push_back(createComputeOp(tileIdx,
                                       {sharedK, sharedV, sharedConst1, sharedConst2, sharedConst3, q, mask,
                                        stateAliasA, stateAliasB, stateAliasC},
                                       {outA, stateAliasA, stateAliasB, stateAliasC}));
        loopBodies.push_back(std::move(body));
    }

    schedulingLoop->loopBodies = std::move(loopBodies);

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    for (size_t opIdx = 0; opIdx < 4; ++opIdx) {
        ASSERT_EQ(allocations.at(opIdx).size(), 3);
    }

    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);
    EXPECT_EQ(allocations.at(0)[2], 2048);

    EXPECT_EQ(allocations.at(1)[0], 3072);
    EXPECT_EQ(allocations.at(1)[1], 4096);
    EXPECT_EQ(allocations.at(1)[2], 5120);

    EXPECT_EQ(allocations.at(2)[0], 0);
    EXPECT_EQ(allocations.at(2)[1], 1024);
    EXPECT_EQ(allocations.at(2)[2], 2048);

    EXPECT_EQ(allocations.at(3)[0], 3072);
    EXPECT_EQ(allocations.at(3)[1], 4096);
    EXPECT_EQ(allocations.at(3)[2], 5120);

    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 8);
    EXPECT_EQ(scheduleResult.reservedSize, 6144);
}

TEST_F(MLIR_UndefinedTiling, FlashSDPAPrefetching) {
    // Case 10: FlashSDPA prefetching-like fallback
    //
    // Shared topology is same as Case 9 (8 shared buffers).
    // Local per-tile buffers: q=1024, mask=1024, outA=2048 (larger than Case 9)
    //
    // Under memorySize=11008, local layout becomes mixed:
    // ┌───────────────────────────┐
    // │ set 1: mask (1024)        │ 5120
    // │ set 1: q    (1024)        │ 4096
    // ├───────────────────────────┤
    // │ shared/reused outA slot   │ 2048
    // ├───────────────────────────┤
    // │ set 0: mask (1024)        │ 1024
    // │ set 0: q    (1024)        │ 0
    // └───────────────────────────┘
    //
    // Expected:
    //   q/mask alternate A/B/A/B
    //   outA reuses the same slot (2048)

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto sharedK = createBuffer(512, 64);
    auto sharedV = createBuffer(512, 64);
    auto sharedConst1 = createBuffer(256, 64);
    auto sharedConst2 = createBuffer(256, 64);
    auto sharedConst3 = createBuffer(256, 64);

    auto stateAliasA = createBuffer(1024, 64);
    auto stateAliasB = createBuffer(1024, 64);
    auto stateAliasC = createBuffer(1024, 64);

    std::vector<LoopBody> loopBodies;
    loopBodies.reserve(4);

    for (size_t tileIdx = 0; tileIdx < 4; ++tileIdx) {
        auto q = createBuffer(1024, 64);
        auto mask = createBuffer(1024, 64);
        auto outA = createBuffer(2048, 64);

        LoopBody body;
        body.push_back(createComputeOp(tileIdx,
                                       {sharedK, sharedV, sharedConst1, sharedConst2, sharedConst3, q, mask,
                                        stateAliasA, stateAliasB, stateAliasC},
                                       {outA, stateAliasA, stateAliasB, stateAliasC}));
        loopBodies.push_back(std::move(body));
    }

    schedulingLoop->loopBodies = std::move(loopBodies);

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/11008);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    for (size_t opIdx = 0; opIdx < 4; ++opIdx) {
        ASSERT_EQ(allocations.at(opIdx).size(), 3);
    }

    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);
    EXPECT_EQ(allocations.at(0)[2], 2048);

    EXPECT_EQ(allocations.at(1)[0], 4096);
    EXPECT_EQ(allocations.at(1)[1], 5120);
    EXPECT_EQ(allocations.at(1)[2], 2048);

    EXPECT_EQ(allocations.at(2)[0], 0);
    EXPECT_EQ(allocations.at(2)[1], 1024);
    EXPECT_EQ(allocations.at(2)[2], 2048);

    EXPECT_EQ(allocations.at(3)[0], 4096);
    EXPECT_EQ(allocations.at(3)[1], 5120);
    EXPECT_EQ(allocations.at(3)[2], 2048);

    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 8);
    EXPECT_EQ(scheduleResult.reservedSize, 6144);
}

TEST_F(MLIR_UndefinedTiling, 2D_Tiling_FullPipelining) {
    // Case 11: 2D tiling model
    // SOK, tilingStrategy=[1, 2, 3, 1]
    //   - 2 groups, 3 tiles per group.
    //   - weights are split into 2 groups, each shared within a group.
    //   - activation is shared across all tiles.
    //
    //   ------------------------------- Group1 -------------------------------
    //     Act_global + W1_g1 + WT_g1_0 -> DPU_1_0 -> O_1_0
    //     Act_global + W1_g1 + WT_g1_1 -> DPU_1_1 -> O_1_1
    //     Act_global + W1_g1 + WT_g1_2 -> DPU_1_2 -> O_1_2
    //   ----------------------------------------------------------------------
    //
    //   ------------------------------- Group2 -------------------------------
    //     Act_global + W1_g2 + WT_g2_0 -> DPU_1_0 -> O_1_0
    //     Act_global + W1_g2 + WT_g2_1 -> DPU_1_1 -> O_1_1
    //     Act_global + W1_g2 + WT_g2_2 -> DPU_1_2 -> O_1_2
    //   ----------------------------------------------------------------------
    //
    // Buffer semantics used here:
    //   - Act_global = global external shared buffer
    //   - W1_g1 / W1_g2 = group-local shared buffer
    //   - WT_g1_n / WT_g2_n = individual per-compute buffers
    //   - O_g1_n / O_g2_n = per-compute outputs
    //
    // Memory model (current UndefinedTiling implementation):
    //
    // ┌────────────────────────────────────────────────────────────┐ ← CMX memorySize
    // │ Shared external region (recognized by frequency criterion) │
    // │   - Act_global                                             │
    // ├────────────────────────────────────────────────────────────┤
    // │ Local working region                                       │
    // │   - W1_g1, W1_g2 (group-local shared buffers)              │
    // │   - WT_g1_n, WT_g2_n (individual per-compute buffers)      │
    // │   - O_g1_n, O_g2_n (per-compute outputs)                   │
    // │   - ping-pong/slot reuse across iterations                 │
    // └────────────────────────────────────────────────────────────┘

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto actGlobal = createBuffer(1024, 64);
    auto w1g1 = createBuffer(1024, 64);
    auto w1g2 = createBuffer(1024, 64);

    SmallVector<mlir::Value> weightTableG1ByTile;
    SmallVector<mlir::Value> outG1ByTile;
    SmallVector<mlir::Value> weightTableG2ByTile;
    SmallVector<mlir::Value> outG2ByTile;
    weightTableG1ByTile.reserve(4);
    outG1ByTile.reserve(4);
    weightTableG2ByTile.reserve(4);
    outG2ByTile.reserve(4);

    std::vector<LoopBody> loopBodies;
    loopBodies.reserve(4);

    size_t opIdx = 0;
    for (size_t tileIdx = 0; tileIdx < 4; ++tileIdx) {
        auto weightTableG1 = createBuffer(1024, 64);
        auto outG1 = createBuffer(1024, 64);

        auto weightTableG2 = createBuffer(1024, 64);
        auto outG2 = createBuffer(1024, 64);

        weightTableG1ByTile.push_back(weightTableG1);
        outG1ByTile.push_back(outG1);
        weightTableG2ByTile.push_back(weightTableG2);
        outG2ByTile.push_back(outG2);

        LoopBody body;
        body.push_back(createComputeOp(opIdx++, {w1g1, actGlobal, weightTableG1}, {outG1}));
        body.push_back(createComputeOp(opIdx++, {w1g2, actGlobal, weightTableG2}, {outG2}));
        loopBodies.push_back(std::move(body));
    }

    schedulingLoop->loopBodies = std::move(loopBodies);

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    const auto& deallocations = allocDealloc.deallocations;
    ASSERT_EQ(allocations.size(), 8);
    ASSERT_EQ(deallocations.size(), 8);

    const auto containsBuffer = [](const SmallVector<mlir::Value>& values, mlir::Value target) {
        return llvm::find(values, target) != values.end();
    };

    // Without sorting, ops are processed in iteration order.
    // Allocation order per operand: [1(shared/skip), 0(W1), 2(WT), 3(O)]
    //
    // Memory layout (local region):
    // ┌───────────────────────────┐
    // │ O_g2 slot (1024)          │ 5120
    // │ WT_g2 slot (1024)         │ 4096
    // │ W1_g2 slot (1024)         │ 3072
    // ├───────────────────────────┤
    // │ O_g1 slot (1024)          │ 2048
    // │ WT_g1 slot (1024)         │ 1024
    // │ W1_g1 slot (1024)         │ 0
    // └───────────────────────────┘
    //
    // op0: W1_g1(0) + WT_g1_0(1024) + O_g1_0(2048)
    ASSERT_EQ(allocations.at(0).size(), 3);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);
    EXPECT_EQ(allocations.at(0)[2], 2048);
    EXPECT_TRUE(deallocations.at(0).empty());

    // op1: W1_g2(3072) + WT_g2_0(4096) + O_g2_0(5120), no deallocations
    ASSERT_EQ(allocations.at(1).size(), 3);
    EXPECT_EQ(allocations.at(1)[0], 3072);
    EXPECT_EQ(allocations.at(1)[1], 4096);
    EXPECT_EQ(allocations.at(1)[2], 5120);
    EXPECT_TRUE(deallocations.at(1).empty());

    // op2: W1_g1 alive, WT_g1_1 replaces WT_g1_0, O_g1_1 replaces O_g1_0
    ASSERT_EQ(allocations.at(2).size(), 2);
    EXPECT_EQ(allocations.at(2)[0], 1024);
    EXPECT_EQ(allocations.at(2)[1], 2048);
    EXPECT_EQ(deallocations.at(2).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(2), weightTableG1ByTile[0]));
    EXPECT_TRUE(containsBuffer(deallocations.at(2), outG1ByTile[0]));

    // op3: W1_g2 alive, WT_g2_1 replaces WT_g2_0, O_g2_1 replaces O_g2_0
    ASSERT_EQ(allocations.at(3).size(), 2);
    EXPECT_EQ(allocations.at(3)[0], 4096);
    EXPECT_EQ(allocations.at(3)[1], 5120);
    EXPECT_EQ(deallocations.at(3).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(3), weightTableG2ByTile[0]));
    EXPECT_TRUE(containsBuffer(deallocations.at(3), outG2ByTile[0]));

    // op4: same slots as op2
    ASSERT_EQ(allocations.at(4).size(), 2);
    EXPECT_EQ(allocations.at(4)[0], 1024);
    EXPECT_EQ(allocations.at(4)[1], 2048);
    EXPECT_EQ(deallocations.at(4).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(4), weightTableG1ByTile[1]));
    EXPECT_TRUE(containsBuffer(deallocations.at(4), outG1ByTile[1]));

    // op5: same slots as op3
    ASSERT_EQ(allocations.at(5).size(), 2);
    EXPECT_EQ(allocations.at(5)[0], 4096);
    EXPECT_EQ(allocations.at(5)[1], 5120);
    EXPECT_EQ(deallocations.at(5).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(5), weightTableG2ByTile[1]));
    EXPECT_TRUE(containsBuffer(deallocations.at(5), outG2ByTile[1]));

    // op6: same slots as op2/op4
    ASSERT_EQ(allocations.at(6).size(), 2);
    EXPECT_EQ(allocations.at(6)[0], 1024);
    EXPECT_EQ(allocations.at(6)[1], 2048);
    EXPECT_EQ(deallocations.at(6).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(6), weightTableG1ByTile[2]));
    EXPECT_TRUE(containsBuffer(deallocations.at(6), outG1ByTile[2]));

    // op7: same slots as op3/op5
    ASSERT_EQ(allocations.at(7).size(), 2);
    EXPECT_EQ(allocations.at(7)[0], 4096);
    EXPECT_EQ(allocations.at(7)[1], 5120);
    EXPECT_EQ(deallocations.at(7).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(7), weightTableG2ByTile[2]));
    EXPECT_TRUE(containsBuffer(deallocations.at(7), outG2ByTile[2]));

    // Shared external behavior:
    // - Only Act_global is recognized as sharedExternalBuffer by current global-frequency criterion.
    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    // scheduleResult.reservedSize is local working memory only:
    //   W1 slots: 2 * 1024 = 2048
    //   WT slots: 2 * 1024 = 2048
    //   O  slots: 2 * 1024 = 2048
    //   total = 6144
    EXPECT_EQ(scheduleResult.reservedSize, 6144);
}

TEST_F(MLIR_UndefinedTiling, 2D_Tiling_Prefetching) {
    // Case 12: 2D tiling prefetching fallback (tighter memory)
    //
    // Same topology as Case 11, but memory is reduced so allocator cannot keep
    // all full-pipelining slots simultaneously.
    //
    // Memory condition in current algorithm:
    //   availableMemory = memorySize - sharedExternalReserved
    //   sharedExternalReserved = Act_global = 1024
    //   with memorySize=6144 => availableMemory=5120
    //
    // Under this limit:
    //   - allocator still keeps two WT/O slot pairs (1024/2048 and 3072/4096)
    //   - W1_g2 no longer gets a dedicated extra slot at 5120 (prefetch fallback)

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto actGlobal = createBuffer(1024, 64);
    auto w1g1 = createBuffer(1024, 64);
    auto w1g2 = createBuffer(1024, 64);

    SmallVector<mlir::Value> weightTableG1ByTile;
    SmallVector<mlir::Value> outG1ByTile;
    SmallVector<mlir::Value> weightTableG2ByTile;
    SmallVector<mlir::Value> outG2ByTile;
    weightTableG1ByTile.reserve(4);
    outG1ByTile.reserve(4);
    weightTableG2ByTile.reserve(4);
    outG2ByTile.reserve(4);

    std::vector<LoopBody> loopBodies;
    loopBodies.reserve(4);

    size_t opIdx = 0;
    for (size_t tileIdx = 0; tileIdx < 4; ++tileIdx) {
        auto weightTableG1 = createBuffer(1024, 64);
        auto outG1 = createBuffer(1024, 64);

        auto weightTableG2 = createBuffer(1024, 64);
        auto outG2 = createBuffer(1024, 64);

        weightTableG1ByTile.push_back(weightTableG1);
        outG1ByTile.push_back(outG1);
        weightTableG2ByTile.push_back(weightTableG2);
        outG2ByTile.push_back(outG2);

        LoopBody body;
        body.push_back(createComputeOp(opIdx++, {w1g1, actGlobal, weightTableG1}, {outG1}));
        body.push_back(createComputeOp(opIdx++, {w1g2, actGlobal, weightTableG2}, {outG2}));
        loopBodies.push_back(std::move(body));
    }

    schedulingLoop->loopBodies = std::move(loopBodies);

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/6144);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    const auto& deallocations = allocDealloc.deallocations;
    ASSERT_EQ(allocations.size(), 8);
    ASSERT_EQ(deallocations.size(), 8);

    const auto containsBuffer = [](const SmallVector<mlir::Value>& values, mlir::Value target) {
        return llvm::find(values, target) != values.end();
    };

    // Without sorting, ops are processed in iteration order.
    // availableMemory = 6144 - 1024 (actGlobal) = 5120.
    // W1_g1/W1_g2 each get a slot, WT/O share a single slot that overflows.
    //
    // Memory layout (local region, availableMemory=5120):
    // ┌───────────────────────────┐
    // │ WT_g2 slot (1024)         │ 4096
    // │ W1_g2 slot (1024)         │ 3072
    // ├───────────────────────────┤
    // │ O shared slot (1024)      │ 2048  ← single slot, reused by all
    // │ WT_g1 slot (1024)         │ 1024
    // │ W1_g1 slot (1024)         │ 0
    // └───────────────────────────┘
    //
    // op0: W1_g1(0) + WT_g1_0(1024) + O_g1_0(2048)
    ASSERT_EQ(allocations.at(0).size(), 3);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);
    EXPECT_EQ(allocations.at(0)[2], 2048);
    EXPECT_TRUE(deallocations.at(0).empty());

    // op1: W1_g2(3072) + WT_g2_0(4096) + O_g2_0 reuses O slot at 2048
    //   O_g1_0 at (2048,3072) conflicts -> deallocate
    ASSERT_EQ(allocations.at(1).size(), 3);
    EXPECT_EQ(allocations.at(1)[0], 3072);
    EXPECT_EQ(allocations.at(1)[1], 4096);
    EXPECT_EQ(allocations.at(1)[2], 2048);
    EXPECT_EQ(deallocations.at(1).size(), 1);
    EXPECT_TRUE(containsBuffer(deallocations.at(1), outG1ByTile[0]));

    // op2: W1_g1 alive, WT_g1_1 replaces WT_g1_0, O_g1_1 replaces O_g2_0
    ASSERT_EQ(allocations.at(2).size(), 2);
    EXPECT_EQ(allocations.at(2)[0], 1024);
    EXPECT_EQ(allocations.at(2)[1], 2048);
    EXPECT_EQ(deallocations.at(2).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(2), weightTableG1ByTile[0]));
    EXPECT_TRUE(containsBuffer(deallocations.at(2), outG2ByTile[0]));

    // op3: W1_g2 alive, WT_g2_1 replaces WT_g2_0, O_g2_1 replaces O_g1_1
    ASSERT_EQ(allocations.at(3).size(), 2);
    EXPECT_EQ(allocations.at(3)[0], 4096);
    EXPECT_EQ(allocations.at(3)[1], 2048);
    EXPECT_EQ(deallocations.at(3).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(3), weightTableG2ByTile[0]));
    EXPECT_TRUE(containsBuffer(deallocations.at(3), outG1ByTile[1]));

    // op4: same pattern as op2
    ASSERT_EQ(allocations.at(4).size(), 2);
    EXPECT_EQ(allocations.at(4)[0], 1024);
    EXPECT_EQ(allocations.at(4)[1], 2048);
    EXPECT_EQ(deallocations.at(4).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(4), weightTableG1ByTile[1]));
    EXPECT_TRUE(containsBuffer(deallocations.at(4), outG2ByTile[1]));

    // op5: same pattern as op3
    ASSERT_EQ(allocations.at(5).size(), 2);
    EXPECT_EQ(allocations.at(5)[0], 4096);
    EXPECT_EQ(allocations.at(5)[1], 2048);
    EXPECT_EQ(deallocations.at(5).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(5), weightTableG2ByTile[1]));
    EXPECT_TRUE(containsBuffer(deallocations.at(5), outG1ByTile[2]));

    // op6: same pattern as op2/4
    ASSERT_EQ(allocations.at(6).size(), 2);
    EXPECT_EQ(allocations.at(6)[0], 1024);
    EXPECT_EQ(allocations.at(6)[1], 2048);
    EXPECT_EQ(deallocations.at(6).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(6), weightTableG1ByTile[2]));
    EXPECT_TRUE(containsBuffer(deallocations.at(6), outG2ByTile[2]));

    // op7: same pattern as op3/5
    ASSERT_EQ(allocations.at(7).size(), 2);
    EXPECT_EQ(allocations.at(7)[0], 4096);
    EXPECT_EQ(allocations.at(7)[1], 2048);
    EXPECT_EQ(deallocations.at(7).size(), 2);
    EXPECT_TRUE(containsBuffer(deallocations.at(7), weightTableG2ByTile[2]));
    EXPECT_TRUE(containsBuffer(deallocations.at(7), outG1ByTile[3]));

    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    EXPECT_EQ(scheduleResult.reservedSize, 5120);
}

// ---------------------------------------------------------------------------
// Tests targeting collectComputeOpBuffers behavior
// ---------------------------------------------------------------------------

TEST_F(MLIR_UndefinedTiling, CollectComputeOpBuffers_IterationOrderPreserved) {
    // Verify that compute ops are processed in original iteration order.
    //
    // Setup:
    //   iter 0: [uniqueIn0, uniqueOut0]          (all unique buffers)
    //   iter 1: [sharedIn,  uniqueOut1]           (sharedIn used in 1 & 2)
    //   iter 2: [sharedIn,  uniqueOut2]           (sharedIn used in 1 & 2)
    //
    // iter 0 always first, then iter 1, then iter 2.
    // Observable effect: iter 0 gets slot A at offset 0, iter 1 gets slot B,
    // iter 2 reuses slot A. If sorting moved iter 0 last, its address would
    // differ.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto sharedIn = createBuffer(1024, 64);

    LoopBody iteration0;
    {
        auto uniqueIn0 = createBuffer(1024, 64);
        auto uniqueOut0 = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {uniqueIn0}, {uniqueOut0}));
    }

    LoopBody iteration1;
    {
        auto uniqueOut1 = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {sharedIn}, {uniqueOut1}));
    }

    LoopBody iteration2;
    {
        auto uniqueOut2 = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {sharedIn}, {uniqueOut2}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 3);

    // iter 0 gets the first slot pair (offset 0).
    ASSERT_EQ(allocations.at(0).size(), 2);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);

    // iter 1 gets the second slot pair.
    ASSERT_EQ(allocations.at(1).size(), 2);
    EXPECT_GE(allocations.at(1)[0], 2048);

    // iter 2: sharedIn is still alive from iter 1 (no re-allocation needed),
    // only the output buffer is allocated, reusing iter 0's output slot.
    ASSERT_EQ(allocations.at(2).size(), 1);
    EXPECT_EQ(allocations.at(2)[0], 1024);
}

TEST_F(MLIR_UndefinedTiling, CollectComputeOpBuffers_InputOutputDedup) {
    // Verify that a buffer appearing as both input and output is counted once
    // per compute op. This means the buffer id appears only once in the
    // compute op's buffer list, and its frequency is computed correctly.
    //
    // Setup:
    //   iter 0: in=[sharedBuf], out=[sharedBuf, uniqueOut0]
    //   iter 1: in=[sharedBuf], out=[sharedBuf, uniqueOut1]
    //
    // sharedBuf appears as both input and output. The dedup logic should
    // count it once per op. Since it appears in all 2 compute ops, it
    // should be identified as globally shared.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto sharedBuf = createBuffer(1024, 64);

    LoopBody iteration0;
    {
        auto uniqueOut0 = createBuffer(512, 64);
        iteration0.push_back(createComputeOp(0, {sharedBuf}, {sharedBuf, uniqueOut0}));
    }

    LoopBody iteration1;
    {
        auto uniqueOut1 = createBuffer(512, 64);
        iteration1.push_back(createComputeOp(1, {sharedBuf}, {sharedBuf, uniqueOut1}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/4096);

    // sharedBuf appears in every compute op (input + output deduped), so it is shared.
    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.count(sharedBuf) > 0);

    // Only the unique output operand is locally allocated per iteration.
    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 2);
    ASSERT_EQ(allocations.at(0).size(), 1);
    ASSERT_EQ(allocations.at(1).size(), 1);
}

TEST_F(MLIR_UndefinedTiling, CollectComputeOpBuffers_ProducerConsumerChain) {
    // Verify correct handling of producer-consumer chains where the output
    // of one compute op is the input of the next.
    // Previously, such dependencies were detected and sorting was skipped.
    // Now sorting is always skipped, so iteration order is preserved regardless.
    //
    // Setup (2D tiling with dependency chain per iteration):
    //   iter 0:
    //     compute op 0: in=[actIn0], out=[intermediate0]
    //     compute op 1: in=[intermediate0], out=[finalOut0]
    //   iter 1:
    //     compute op 2: in=[actIn1], out=[intermediate1]
    //     compute op 3: in=[intermediate1], out=[finalOut1]

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    LoopBody iteration0;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto intermediate0 = createBuffer(1024, 64);
        auto finalOut0 = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {actIn0}, {intermediate0}));
        iteration0.push_back(createComputeOp(1, {intermediate0}, {finalOut0}));
    }

    LoopBody iteration1;
    {
        auto actIn1 = createBuffer(1024, 64);
        auto intermediate1 = createBuffer(1024, 64);
        auto finalOut1 = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(2, {actIn1}, {intermediate1}));
        iteration1.push_back(createComputeOp(3, {intermediate1}, {finalOut1}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    // All 4 compute ops should produce valid allocations.
    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    // Op 0: allocates actIn0 + intermediate0 -> 2 allocations.
    ASSERT_EQ(allocations.at(0).size(), 2);
    EXPECT_EQ(allocations.at(0)[0], 0);
    EXPECT_EQ(allocations.at(0)[1], 1024);

    // Op 1: intermediate0 is still alive from op 0 -> only finalOut0 allocated.
    ASSERT_EQ(allocations.at(1).size(), 1);

    // Op 2: allocates actIn1 + intermediate1 -> 2 allocations.
    ASSERT_EQ(allocations.at(2).size(), 2);

    // Op 3: intermediate1 is still alive from op 2 -> only finalOut1 allocated.
    ASSERT_EQ(allocations.at(3).size(), 1);

    // No globally shared buffers (no buffer appears in every compute op).
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.empty());
}

TEST_F(MLIR_UndefinedTiling, CollectComputeOpBuffers_NoComputeOpsReturnsEmpty) {
    // Verify that when no COMPUTE ops exist (only DATA_IN), the collect phase
    // returns empty and getScheduleStrategy produces an empty schedule.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    VPURT::TaskQueueType dmaQt{config::ExecutorKind::DMA_NN, 0};

    LoopBody iteration0;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(OpAllocationInfo(0, dmaQt, {inBuf}, {outBuf}, AllocationType::DATA_IN));
    }

    schedulingLoop->loopBodies = {std::move(iteration0)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/8192);

    EXPECT_TRUE(scheduleResult.schedule.empty());
    EXPECT_EQ(scheduleResult.reservedSize, 0);
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.empty());
}

TEST_F(MLIR_UndefinedTiling, CollectComputeOpBuffers_FrequencyTableDrivesSharedDetection) {
    // Verify that the frequency table computed by collectComputeOpBuffers
    // correctly identifies globally shared buffers (those used by ALL compute ops).
    //
    // Setup:
    //   weightA: used in iter 0, 1, 2 (all 3 ops) -> globally shared
    //   weightB: used in iter 0, 1 only (2 of 3 ops) -> NOT shared
    //
    // Expected: only weightA is in sharedExternalBuffers.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto weightA = createBuffer(1024, 64);
    auto weightB = createBuffer(512, 64);

    LoopBody iteration0;
    {
        auto actIn = createBuffer(512, 64);
        auto outBuf = createBuffer(512, 64);
        iteration0.push_back(createComputeOp(0, {weightA, weightB, actIn}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto actIn = createBuffer(512, 64);
        auto outBuf = createBuffer(512, 64);
        iteration1.push_back(createComputeOp(1, {weightA, weightB, actIn}, {outBuf}));
    }

    LoopBody iteration2;
    {
        auto actIn = createBuffer(512, 64);
        auto outBuf = createBuffer(512, 64);
        // weightB is NOT used here, replaced by a unique buffer.
        auto uniqueWeight = createBuffer(512, 64);
        iteration2.push_back(createComputeOp(2, {weightA, uniqueWeight, actIn}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    // Only weightA appears in all 3 compute ops -> shared.
    // weightB appears in only 2 of 3 -> not shared.
    EXPECT_EQ(scheduleResult.sharedExternalBuffers.size(), 1);
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.count(weightA) > 0);
}

TEST_F(MLIR_UndefinedTiling, PartialDoubleBufferingSuppressed) {
    // Case: Partial double-buffering suppression
    //
    // Each iteration has a large IN buffer (3072) and a small OUT buffer (1024).
    // One set = 3072 + 1024 = 4096.
    // Two full sets would need 8192, but available memory is only 6144.
    //
    // IN cannot be double-buffered (4096 + 3072 = 7168 > 6144) -> reuses set 0.
    // OUT *could* fit a 2nd slot (4096 + 1024 = 5120 <= 6144) -> gets double-buffered.
    // This "partial double-buffering" wastes 1024 bytes without enabling pipelining,
    // because IN still blocks the pipeline on every iteration.
    // Once IN hits OOM, OUT is also forced to reuse set 0 -> all iterations share the same addresses.
    //
    // ┌───────────────────────────┐ <- availableMemory = 6144
    // │ (unused 2048)             │
    // ├───────────────────────────┤
    // │ set 0: out (1024)         │ 3072
    // │ set 0: in  (3072)         │ 0
    // └───────────────────────────┘
    //
    // Expected:
    //   All iterations -> [0, 3072]
    //   reservedSize = 4096

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    LoopBody iteration0;
    {
        auto inBuf = createBuffer(3072, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {inBuf}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto inBuf = createBuffer(3072, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {inBuf}, {outBuf}));
    }

    LoopBody iteration2;
    {
        auto inBuf = createBuffer(3072, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {inBuf}, {outBuf}));
    }

    LoopBody iteration3;
    {
        auto inBuf = createBuffer(3072, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    // memorySize=6144: enough for one full set (4096) but not two (8192).
    // Crucially, OUT alone could fit a 2nd slot (4096+1024=5120 <= 6144),
    // but the reachedMemoryLimit flag prevents this partial double-buffering.
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/6144);

    const auto allocDealloc = collectComputeAllocationsByOp(scheduleResult.schedule);
    const auto& allocations = allocDealloc.allocations;
    ASSERT_EQ(allocations.size(), 4);

    // All iterations must reuse the same set 0 addresses: IN=[0,3072), OUT=[3072,4096).
    for (size_t opIdx = 0; opIdx < 4; ++opIdx) {
        ASSERT_EQ(allocations.at(opIdx).size(), 2);
        EXPECT_EQ(allocations.at(opIdx)[0], 0);
        EXPECT_EQ(allocations.at(opIdx)[1], 3072);
    }

    // Only one set consumed: reservedSize = 4096, not 5120.
    EXPECT_EQ(scheduleResult.reservedSize, 4096);
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.empty());
}

TEST_F(MLIR_UndefinedTiling, InsufficientLocalMemoryReturnsEmptySchedule) {
    // Case 13: After reserving shared buffers, remaining memory is too small
    // for even a single set of operand slots. verifyOperandSlotRequirements
    // catches this and returns an empty schedule instead of crashing.
    //
    // Shared buffer reservation:
    //   weight = 2048 (shared across all 2 iterations)
    //
    // Available CMX after shared reservation:
    //   memorySize - weight = 2560 - 2048 = 512
    //
    // Operand slot requirements (per-operand max):
    //   operand 0 (input):  1024
    //   operand 1 (output): 1024
    //   minimum local = 2048 > 512 available
    //
    // Expected: empty schedule, reservedSize=0, no crash.

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    auto weightBuf = createBuffer(2048, 64);

    LoopBody iteration0;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {weightBuf, inBuf}, {outBuf}));
    }

    LoopBody iteration1;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {weightBuf, inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    vpux::VPU::UndefinedTiling scenario;
    const auto scheduleResult = scenario.getScheduleStrategy(region, /*memorySize=*/2560);

    // verifyOperandSlotRequirements should detect insufficient memory and bail out gracefully.
    EXPECT_TRUE(scheduleResult.schedule.empty());
    EXPECT_EQ(scheduleResult.reservedSize, 0);
    EXPECT_TRUE(scheduleResult.sharedExternalBuffers.empty());
    EXPECT_EQ(scheduleResult.baseAlignment, vpux::DEFAULT_CMX_ALIGNMENT);
}
