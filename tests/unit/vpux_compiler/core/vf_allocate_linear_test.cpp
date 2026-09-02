//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/vf_allocate_linear.hpp"

#include "vpux/compiler/core/scheduling/classifier_vf.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// Run cmd: npuUnitTests --gtest_filter="MLIR_VFAllocateLinear.*"

using namespace vpux;

// Helper class for building VF compute regions with buffers and operations in MLIR context
class MLIR_VFAllocateLinear : public testing::Test {
protected:
    mlir::MLIRContext _ctx;
    mlir::OpBuilder _builder{&_ctx};
    std::unique_ptr<mlir::Block> _block;

    MLIR_VFAllocateLinear() {
        auto registry = vpux::createDialectRegistry();
        _ctx.appendDialectRegistry(registry);
        _ctx.loadAllAvailableDialects();
        _block = std::make_unique<mlir::Block>();
        _builder.setInsertionPointToStart(_block.get());
    }

    mlir::Value createBuffer(vpux::AddressType rawSize, vpux::AddressType rawAlign = 64) {
        auto shape = SmallVector<int64_t>{static_cast<int64_t>(rawSize)};
        auto type = mlir::MemRefType::get(shape, _builder.getIntegerType(8));
        auto alloc = _builder.create<mlir::memref::AllocOp>(_builder.getUnknownLoc(), type);
        if (rawAlign != 0) {
            alloc.setAlignment(rawAlign);
        }
        return alloc;
    }

    OpAllocationInfo computeOp(size_t opIdx, const SmallVector<mlir::Value>& inBufs,
                               const SmallVector<mlir::Value>& outBufs) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DPU, 0};
        return OpAllocationInfo(opIdx, qt, inBufs, outBufs, AllocationType::COMPUTE);
    }

    OpAllocationInfo dataIn(size_t opIdx, const SmallVector<mlir::Value>& inBufs,
                            const SmallVector<mlir::Value>& outBufs) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DMA_NN, 0};
        return OpAllocationInfo(opIdx, qt, inBufs, outBufs, AllocationType::DATA_IN);
    }

    OpAllocationInfo dataOut(size_t opIdx, const SmallVector<mlir::Value>& inBufs,
                             const SmallVector<mlir::Value>& outBufs) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DMA_NN, 0};
        return OpAllocationInfo(opIdx, qt, inBufs, outBufs, AllocationType::DATA_OUT);
    }

    ComputeRegion makeVfRegion(std::vector<LoopBody> bodies) {
        auto loop = std::make_unique<SchedulingLoop>();
        loop->type = LoopType::VF;
        loop->loopBodies = std::move(bodies);
        return ComputeRegion(std::move(loop));
    }
};

// ---------------------------------------------------------------------------
// Check basic MINIMAL strategy allocation for trivial 2-op region.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, MinimalRegionFeasible) {
    auto in0 = createBuffer(1024);
    auto mid0 = createBuffer(1024);
    auto out0 = createBuffer(1024);
    auto mid1 = createBuffer(1024);
    LoopBody body0{computeOp(0, {in0}, {mid0}), computeOp(1, {mid0}, {out0})};
    LoopBody body1{computeOp(2, {in0}, {mid1}), computeOp(3, {mid1}, {out0})};
    auto region = makeVfRegion({body0, body1});

    const size_t memoryLimit = 16384;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;  // FetchScope=NONE, OutputResidency=DROP

    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    EXPECT_TRUE(result.feasible);
    EXPECT_EQ(result.iterationSchedule.size(), classifier.opsPerIteration);
    EXPECT_EQ(result.persistentReservedBytes, 2048u);
    EXPECT_TRUE(result.spilledBuffers.empty());
    EXPECT_LE(result.peakUsedBytes, memoryLimit);
}

// ---------------------------------------------------------------------------
// Check if weight buffer classified as persistent across iterations would remain
// persistent after allocation and be part of sharedExternalBuffers.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, AcceptWeightAsPersistent) {
    auto weight = createBuffer(2048);  // loop-invariant => PERSISTENT_CANDIDATE
    auto in = createBuffer(1024);
    auto out = createBuffer(1024);

    const size_t memoryLimit = 16384;
    // Share `in` and `out` across both iterations so the classifier does not reject the
    // region as partial-external. `in` becomes PERSISTENT_CANDIDATE (external, consumed by
    // every iter) and `out` becomes PERSISTENT_CANDIDATE (produced by every iter, never
    // consumed inside the loop).
    LoopBody body{computeOp(0, {weight, in}, {out})};
    auto region = makeVfRegion({body, body});
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    ASSERT_GE(classifier.persistentFitOrder.size(), 1u);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    EXPECT_TRUE(result.feasible);
    EXPECT_EQ(result.persistentReservedBytes, classifier.persistentFitReservedBytes);
    EXPECT_EQ(result.sharedExternalBuffers.size(), classifier.persistentFitOrder.size());
    EXPECT_EQ(result.sharedExternalBuffers.count(weight), 1u);
}

// ---------------------------------------------------------------------------
// Check both peakUsedBytes and persistentReservedBytes are within the memory limit.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, UsedBytesWithinLimit) {
    auto weight = createBuffer(2048);
    auto in = createBuffer(1024);
    auto out = createBuffer(1024);
    LoopBody body{computeOp(0, {weight, in}, {out})};
    auto region = makeVfRegion({body, body});
    const size_t memoryLimit = 16384;
    const auto classifier = classifyVFRegion(region, memoryLimit);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    EXPECT_TRUE(result.feasible);
    EXPECT_LE(result.peakUsedBytes, memoryLimit);
    EXPECT_LE(result.persistentReservedBytes, memoryLimit);
    EXPECT_LE(result.peakUsedBytes + result.persistentReservedBytes, memoryLimit);
}

// ---------------------------------------------------------------------------
// Check iterationSchedule has one entry per template op based on first iteration
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, IterationScheduleSizeEqualsOpsPerIteration) {
    auto in0 = createBuffer(512);
    auto t00 = createBuffer(512);
    auto t01 = createBuffer(512);
    auto out0 = createBuffer(512);
    auto t10 = createBuffer(512);
    auto t11 = createBuffer(512);
    LoopBody body0{computeOp(0, {in0}, {t00}), computeOp(1, {t00}, {t01}), computeOp(2, {t01}, {out0})};
    LoopBody body1{computeOp(3, {in0}, {t10}), computeOp(4, {t10}, {t11}), computeOp(5, {t11}, {out0})};
    auto region = makeVfRegion({body0, body1});
    const auto classifier = classifyVFRegion(region, /*memoryLimit=*/16384);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, /*memoryLimit=*/16384, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 3u);
    EXPECT_EQ(result.iterationSchedule[0].allocInfo.opIdx, 0u);
    EXPECT_EQ(result.iterationSchedule[1].allocInfo.opIdx, 1u);
    EXPECT_EQ(result.iterationSchedule[2].allocInfo.opIdx, 2u);
}

// ---------------------------------------------------------------------------
// Check sharedExternalBuffers are deterministic across allocations for same compute region
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, SharedExternalBuffersDeterministic) {
    auto w0 = createBuffer(1024);
    auto w1 = createBuffer(1024);
    auto in = createBuffer(512);
    auto out = createBuffer(512);
    const size_t memoryLimit = 16384;
    // Share `in` and `out` across both iterations to keep the classifier happy.
    LoopBody body{computeOp(0, {w0, w1, in}, {out})};
    auto region = makeVfRegion({body, body});
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    ASSERT_GE(classifier.persistentFitOrder.size(), 2u);

    VfSchedStrategyDescriptor params;
    const auto r1 = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();
    const auto r2 = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(r1.feasible);
    ASSERT_TRUE(r2.feasible);
    SmallVector<mlir::Value> v1(r1.sharedExternalBuffers.begin(), r1.sharedExternalBuffers.end());
    SmallVector<mlir::Value> v2(r2.sharedExternalBuffers.begin(), r2.sharedExternalBuffers.end());
    EXPECT_EQ(v1, v2);
}

// ---------------------------------------------------------------------------
// Two-op chain per iteration where each op has its own loop-invariant weight.
// Both weights are shared externals classified as PERSISTENT_CANDIDATE by the
// classifier. PERSISTENT_CANDIDATE is yet non-downgradable: if any PC cannot be
// admitted (because it would crowd out the volatile working set), the
// allocation is infeasible instead of silently downgrading to TEMPORARY.
//
//   in -> op1(weightA) -> mid -> op2(weightB) -> out
//
// Sizes are tuned so memoryLimit only admits weightA as persistent:
//   * op1: weightA(4096) + in(2560) + mid(2048) = 8704
//   * op2: weightA(4096) + mid(2048) + weightB(2048) + out(512) = 8704
//   * Admitting both weights: 4096 + 2048 + tempPeak(4608) = 10752 > 8704.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, PersistentThatCrowdsVolatilePeakIsInfeasible) {
    auto weightA = createBuffer(4096);  // Loop-invariant weight of op1 (PE, admits first).
    auto weightB = createBuffer(2048);  // Loop-invariant weight of op2 (PE, cannot fit alongside).

    // Share region I/O and intermediate mid across both iterations so the classifier can
    // model each SSA as either PERSISTENT_CANDIDATE (in/out) or TEMPORARY (mid) instead of
    // rejecting the region as partial-external.
    auto in = createBuffer(2560);
    auto mid = createBuffer(2048);
    auto out = createBuffer(512);

    LoopBody body{computeOp(0, {weightA, in}, {mid}), computeOp(1, {weightB, mid}, {out})};
    auto region = makeVfRegion({body, body});

    const AddressType memoryLimit = 8704;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);
    // There are initially 4 persistent buffers: CMX input, CMX output, weightA, weightB, but not all of them
    // can be admitted based on memory limit
    EXPECT_FALSE(classifier.classifierSupportedRegion);
    EXPECT_EQ(classifier.persistentFitOrder.size(), 4u);
    EXPECT_LT(classifier.persistentFitInitial.size(), 4u);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    // PERSISTENT_CANDIDATE downgrade to TEMPORARY is not yet supported, so the allocator
    // must refuse the region instead of admitting only weightA.
    EXPECT_FALSE(result.feasible);
    EXPECT_TRUE(result.sharedExternalBuffers.empty());
    EXPECT_TRUE(result.spilledBuffers.empty());
}

// ---------------------------------------------------------------------------
// Spill trigger in a branched template body:
//
//     in0 -> OP0 -> OP3->|
//                        OP4 -> out0
//     in1 -> OP1 -> OP2->|
//
//     LoopBody order: OP0, OP1, OP2, OP3, OP4.
//
//     With a tight CMX budget, OP1 output must be spilled before OP2->OP3 can
//     continue, then later consumed by OP4.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, TemporaryBufferSpillWithParallelBranches) {
    // Iteration 0:
    auto in00 = createBuffer(512);
    auto in01 = createBuffer(512);
    // Keep OP0 output large enough so OP1 output cannot be allocated while OP0
    // output remains live in CMX.
    auto op00Out = createBuffer(6144);
    auto op01Out = createBuffer(3072);
    auto op02Out = createBuffer(2048);
    auto op03Out = createBuffer(2048);
    auto out0 = createBuffer(2048);

    LoopBody body0{
            dataIn(0, {}, {in00}),                     // OP0
            dataIn(1, {}, {in01}),                     // OP1
            computeOp(2, {in00}, {op00Out}),           // OP2
            computeOp(3, {in01}, {op01Out}),           // OP3
            computeOp(4, {op01Out}, {op02Out}),        // OP4
            computeOp(5, {op00Out}, {op03Out}),        // OP5
            computeOp(6, {op02Out, op03Out}, {out0}),  // OP6
            dataOut(7, {out0}, {})                     // OP7
    };

    // Iteration 1
    auto in10 = createBuffer(512);
    auto in11 = createBuffer(512);
    // Keep OP0 output large enough so OP1 output cannot be allocated while OP0
    // output remains live in CMX.
    auto op10Out = createBuffer(6144);
    auto op11Out = createBuffer(3072);
    auto op12Out = createBuffer(2048);
    auto op13Out = createBuffer(2048);
    auto out1 = createBuffer(2048);

    LoopBody body1{
            dataIn(8, {}, {in10}),                      // OP0
            dataIn(9, {}, {in11}),                      // OP1
            computeOp(10, {in10}, {op10Out}),           // OP2
            computeOp(11, {in11}, {op11Out}),           // OP3
            computeOp(12, {op11Out}, {op12Out}),        // OP4
            computeOp(13, {op10Out}, {op13Out}),        // OP5
            computeOp(14, {op12Out, op13Out}, {out1}),  // OP6
            dataOut(15, {out1}, {})                     // OP7
    };

    auto region = makeVfRegion({body0, body1});

    // 6144 + 3072 > 8192, so OP2 allocation cannot proceed unless OP1 output
    // is spilled first.
    constexpr AddressType memoryLimit = 8192;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;  // FetchScope=NONE, OutputResidency=DROP
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 8u);

    // During scheduling OP0 and OP2 outputs had to be spilled
    ASSERT_EQ(result.spilledBuffers.size(), 2);

    // OP0 output spilling
    EXPECT_NE(std::find(result.spilledBuffers.begin(), result.spilledBuffers.end(), op00Out),
              result.spilledBuffers.end());

    // Spill is materialized as a deallocation at OP3 entry in local schedule.
    const auto& op3Entry = result.iterationSchedule[3];
    ASSERT_EQ(op3Entry.allocInfo.opIdx, 3u);
    EXPECT_NE(std::find(op3Entry.deallocations.begin(), op3Entry.deallocations.end(), op00Out),
              op3Entry.deallocations.end());

    // OP4 output spilling
    EXPECT_NE(std::find(result.spilledBuffers.begin(), result.spilledBuffers.end(), op02Out),
              result.spilledBuffers.end());

    // Spill is materialized as a deallocation at OP5 entry in local schedule.
    const auto& op5Entry = result.iterationSchedule[5];
    ASSERT_EQ(op5Entry.allocInfo.opIdx, 5u);
    EXPECT_NE(std::find(op5Entry.deallocations.begin(), op5Entry.deallocations.end(), op02Out),
              op5Entry.deallocations.end());
}

// ---------------------------------------------------------------------------
// Spill victim selection must exclude the current op's inputs even when those
// inputs were never previously spilled. The largest live buffer at the moment
// OP4 is scheduled is OP2's output (op0Out), which is also OP4's input. A naive
// "largest first" rule would pick it as the spill victim, but doing so would
// leave OP4 without its input in CMX. The allocator must skip op0Out and pick
// the smaller non-input live buffer (op1Out) instead.
//
//     OP0 -> in0a -> OP2 -> op0Out --------> OP4 -> op2Out ---> OP5 -> out -> OP6
//     OP1 -> in0b -> OP3 -> op1Out --------------------------->|
//
//     LoopBody order: OP0 (DATA_IN), OP1 (DATA_IN), OP2, OP3, OP4, OP5, OP6 (DATA_OUT)
//
//     Sizes are tuned so:
//       * At OP4 scheduling, op0Out(3072) + op1Out(1024) live in CMX. Free space
//         is fragmented by earlier DATA_IN buffers, so the 4096-byte op2Out
//         cannot fit -> spill is required.
//       * Only op1Out is a legitimate victim; spilling op0Out would evict
//         OP4's input.
//       * After spilling op1Out, the freed 1024-byte slot is adjacent to the
//         top gap and forms a contiguous 4096-byte region that fits op2Out.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, SpillVictimSelectionExcludesCurrentOpInput) {
    // Iteration 0. Allocation order matters: both DATA_IN buffers are produced
    // first (they die early and free their slots at the bottom), then op0Out
    // (large, OP4 input) is placed above them, and op1Out (small, spillable) is
    // placed above op0Out. Freeing op1Out later merges with the top gap into a
    // contiguous hole large enough for op2Out.
    auto in0a = createBuffer(512);
    auto in0b = createBuffer(512);
    auto op0Out = createBuffer(3072);  // OP4 input — must NOT be spilled
    auto op1Out = createBuffer(1024);  // Non-input to OP4 — the only legal spill victim
    auto op2Out = createBuffer(4096);  // OP4 output — triggers the spill
    auto out0 = createBuffer(512);

    LoopBody body0{
            dataIn(0, {}, {in0a}),                   // OP0
            dataIn(1, {}, {in0b}),                   // OP1
            computeOp(2, {in0a}, {op0Out}),          // OP2
            computeOp(3, {in0b}, {op1Out}),          // OP3
            computeOp(4, {op0Out}, {op2Out}),        // OP4 — spill trigger
            computeOp(5, {op1Out, op2Out}, {out0}),  // OP5 — consumes spilled op1Out
            dataOut(6, {out0}, {})                   // OP6
    };

    // Iteration 1
    auto in1a = createBuffer(512);
    auto in1b = createBuffer(512);
    auto op10Out = createBuffer(3072);
    auto op11Out = createBuffer(1024);
    auto op12Out = createBuffer(4096);
    auto out1 = createBuffer(512);

    LoopBody body1{
            dataIn(7, {}, {in1a}),                      // OP0
            dataIn(8, {}, {in1b}),                      // OP1
            computeOp(9, {in1a}, {op10Out}),            // OP2
            computeOp(10, {in1b}, {op11Out}),           // OP3
            computeOp(11, {op10Out}, {op12Out}),        // OP4
            computeOp(12, {op11Out, op12Out}, {out1}),  // OP5
            dataOut(13, {out1}, {})                     // OP6
    };

    auto region = makeVfRegion({body0, body1});

    // memoryLimit is set so that at OP4 scheduling the largest contiguous free
    // region is smaller than op2Out(4096) — the 512-byte holes left by dead
    // DATA_IN buffers cannot be reused for op2Out. A spill of op1Out is
    // required to merge the top free gap into a 4096-byte contiguous region.
    constexpr AddressType memoryLimit = 8192;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;  // FetchScope=NONE, OutputResidency=DROP
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 7u);

    // Exactly one spill happens during OP4 scheduling: op1Out, the non-input
    // live buffer. op0Out (OP4's input) must be preserved.
    ASSERT_EQ(result.spilledBuffers.size(), 1u);
    EXPECT_NE(std::find(result.spilledBuffers.begin(), result.spilledBuffers.end(), op1Out),
              result.spilledBuffers.end());
    EXPECT_EQ(std::find(result.spilledBuffers.begin(), result.spilledBuffers.end(), op0Out),
              result.spilledBuffers.end());

    // Spill is materialized as a deallocation at OP4 entry in local schedule.
    // OP4's input (op0Out) must not appear in OP4's deallocation list.
    const auto& op4Entry = result.iterationSchedule[4];
    ASSERT_EQ(op4Entry.allocInfo.opIdx, 4u);
    EXPECT_NE(std::find(op4Entry.deallocations.begin(), op4Entry.deallocations.end(), op1Out),
              op4Entry.deallocations.end());
    EXPECT_EQ(std::find(op4Entry.deallocations.begin(), op4Entry.deallocations.end(), op0Out),
              op4Entry.deallocations.end());

    EXPECT_LE(result.peakUsedBytes, memoryLimit);
}
