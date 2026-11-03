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
        auto alloc = mlir::memref::AllocOp::create(_builder, _builder.getUnknownLoc(), type);
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

    // Assigned CMX byte range [start, end) of a buffer within the iteration schedule.
    static std::optional<std::pair<AddressType, AddressType>> getBufferAssignedRange(
            const VfAllocateResult& result, const ClassifierVFResult& classifier, mlir::Value buf) {
        for (const auto& entry : result.iterationSchedule) {
            for (const auto& [v, offset] : entry.allocations) {
                if (v == buf) {
                    const auto& rec = classifier.buffers[classifier.valueToRecord.at(buf)];
                    return std::make_pair(offset, offset + rec.size);
                }
            }
        }
        return std::nullopt;
    }

    static bool rangesOverlap(std::pair<AddressType, AddressType> lhs, std::pair<AddressType, AddressType> rhs) {
        return lhs.first < rhs.second && rhs.first < lhs.second;
    }

    // Template position of the schedule entry that allocated `buf`, if any.
    static std::optional<size_t> getBufferAllocEntryPos(const VfAllocateResult& result, mlir::Value buf) {
        for (const auto& entry : result.iterationSchedule) {
            for (const auto& [v, offset] : entry.allocations) {
                std::ignore = offset;
                if (v == buf) {
                    return entry.templatePos;
                }
            }
        }
        return std::nullopt;
    }
};

// ---------------------------------------------------------------------------
// Reclaim speculative last-compute reservations under memory pressure.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, ReclaimEarlyReservationWithoutSpillingUnproducedBuffer) {
    auto makeBody = [&](size_t firstOp) {
        auto input = createBuffer(64);
        auto mid = createBuffer(64);
        auto scratch = createBuffer(256);
        auto futureInput = createBuffer(512);
        auto output = createBuffer(64);
        return LoopBody{dataIn(firstOp, {}, {input}),           computeOp(firstOp + 1, {input}, {mid}),
                        dataIn(firstOp + 2, {}, {scratch}),     dataOut(firstOp + 3, {scratch}, {}),
                        dataIn(firstOp + 4, {}, {futureInput}), computeOp(firstOp + 5, {mid, futureInput}, {output}),
                        dataOut(firstOp + 6, {output}, {})};
    };
    auto region = makeVfRegion({makeBody(0), makeBody(7)});
    constexpr AddressType memoryLimit = 768;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    VfSchedStrategyDescriptor params;
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;

    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();
    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 7u);
    EXPECT_TRUE(result.spilledBuffers.empty());

    // Producing mid initially reserves futureInput for the last compute. Loading
    // scratch must reclaim that reservation, then futureInput is allocated at its
    // own DMA. It must never appear as a spill of data that does not exist yet.
    const auto futureInput = region.schedulingLoop->loopBodies.front()[4].outBuffers.front();
    size_t allocationCount = 0;
    for (const auto& entry : result.iterationSchedule) {
        if (entry.allocInfo.opIdx < 4) {
            EXPECT_FALSE(llvm::is_contained(entry.deallocations, futureInput));
        }
        for (const auto& allocation : entry.allocations) {
            if (allocation.first == futureInput) {
                ++allocationCount;
                EXPECT_EQ(entry.allocInfo.opIdx, 4u);
            }
        }
    }
    EXPECT_EQ(allocationCount, 1u);
}

TEST_F(MLIR_VFAllocateLinear, PreserveProducedTemporaryReloadedByOverlap) {
    auto makeBody = [&](size_t firstOp) {
        auto seed = createBuffer(64);
        auto produced = createBuffer(512);
        auto scratchA = createBuffer(512);
        auto mid = createBuffer(64);
        auto scratchB = createBuffer(512);
        auto output = createBuffer(64);
        return LoopBody{dataIn(firstOp, {}, {seed}),           computeOp(firstOp + 1, {seed}, {produced}),
                        dataIn(firstOp + 2, {}, {scratchA}),   dataOut(firstOp + 3, {scratchA}, {}),
                        computeOp(firstOp + 4, {seed}, {mid}), dataIn(firstOp + 5, {}, {scratchB}),
                        dataOut(firstOp + 6, {scratchB}, {}),  computeOp(firstOp + 7, {produced, mid}, {output}),
                        dataOut(firstOp + 8, {output}, {})};
    };
    auto region = makeVfRegion({makeBody(0), makeBody(9)});
    const auto produced = region.schedulingLoop->loopBodies.front()[1].outBuffers.front();
    constexpr AddressType memoryLimit = 768;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    ASSERT_EQ(classifier.buffers[classifier.valueToRecord.at(produced)].category, BufferCategory::TEMPORARY);
    VfSchedStrategyDescriptor params;
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();
    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 9u);
    EXPECT_TRUE(llvm::any_of(result.iterationSchedule[1].allocations, [&](const auto& a) {
        return a.first == produced;
    }));
    EXPECT_TRUE(llvm::is_contained(result.iterationSchedule[2].deallocations, produced));
    EXPECT_TRUE(llvm::any_of(result.iterationSchedule[4].allocations, [&](const auto& a) {
        return a.first == produced;
    }));
    EXPECT_TRUE(llvm::is_contained(result.iterationSchedule[5].deallocations, produced));
}

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

    VfSchedStrategyDescriptor params;  // IntraIterFetch=DISABLED, LastComputeOverlapWithNextIterInputs=DISABLED

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
// Fully-covered explicit external output across distinct per-iteration SSA values
// must be forwarded to sharedExternalBuffers as one group.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, DistinctExplicitExternalGroupFlowsToSharedExternalBuffers) {
    auto weight = createBuffer(256, /*rawAlign=*/64);
    auto intermediate0 = createBuffer(128, /*rawAlign=*/64);
    auto externalBuf0 = createBuffer(64, /*rawAlign=*/64);
    auto intermediate1 = createBuffer(128, /*rawAlign=*/64);
    auto externalBuf1 = createBuffer(64, /*rawAlign=*/64);

    LoopBody iteration0;
    iteration0.push_back(dataIn(/*opIdx=*/0, {weight}, {intermediate0}));
    iteration0.push_back(computeOp(/*opIdx=*/1, {intermediate0}, {externalBuf0}));
    iteration0.push_back(dataOut(/*opIdx=*/2, {externalBuf0}, {}));

    LoopBody iteration1;
    iteration1.push_back(dataIn(/*opIdx=*/3, {weight}, {intermediate1}));
    iteration1.push_back(computeOp(/*opIdx=*/4, {intermediate1}, {externalBuf1}));
    iteration1.push_back(dataOut(/*opIdx=*/5, {externalBuf1}, {}));

    auto region = makeVfRegion({std::move(iteration0), std::move(iteration1)});
    region.externalBuffers.insert(externalBuf0);
    region.externalBuffers.insert(externalBuf1);

    const size_t memoryLimit = 4096;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    ASSERT_EQ(classifier.explicitExternalGroups.size(), 1u);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    EXPECT_EQ(result.sharedExternalBuffers.count(externalBuf0), 1u);
    EXPECT_EQ(result.sharedExternalBuffers.count(externalBuf1), 1u);
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
// When a PERSISTENT_CANDIDATE cannot be admitted (because it would crowd out
// the volatile working set) the classifier keeps the region supported and the
// allocator downgrades the unadmitted PC to TEMPORARY. During scheduling the
// downgraded `out` collides with the still-live `mid` in the volatile zone at
// op1, so the allocator evicts the cheapest admitted persistent (`in`) — the
// eviction shrinks the persistent zone, retroactively registers `in` as a
// TEMPORARY at its original persistent slot, and records a spill dealloc at
// the current op.
//
//   in -> op1(weightA) -> mid -> op2(weightB) -> out
//
// Sizes are tuned so classifier admits weightA + in but rejects weightB + out:
//   * op1: weightA(4096) + in(2560) + mid(2048) = 8704
//   * op2: weightA(4096) + mid(2048) + weightB(2048) + out(512) = 8704
//   * Admitting all persistents: 4096 + 2560 + 2048 + 512 + tempPeak(2048)
//     = 11264 > 8704.
//
// Admission order (DESC by cost): weightA(4096) admitted, in(2560) admitted,
// weightB and out rejected. persistentFloor starts at 8704 - 6656 = 2048.
// At op1 the volatile zone [0, 2048) is occupied by mid; alloc of `out` fails
// -> `in` is evicted: floor moves to 2048 + 2560 = 4608, and `in` is
// retroactively allocated at addr 2048 in iter[0] and deallocated in iter[1].
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, PersistentThatCrowdsVolatilePeakDowngradesToTemporary) {
    auto weightA = createBuffer(4096);
    auto weightB = createBuffer(2048);

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
    // 4 persistent candidates: CMX input, CMX output, weightA, weightB. Not all fit under
    // the volatile budget, but the region stays classifier-supported: unadmitted PCs are
    // downgraded to TEMPORARY at scheduling time.
    EXPECT_TRUE(classifier.classifierSupportedRegion);
    EXPECT_EQ(classifier.persistentFitOrder.size(), 4u);
    EXPECT_LT(classifier.persistentFitInitial.size(), 4u);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    // Schedule is feasible thanks to persistent-buffer eviction. `in` was the
    // cheapest admitted persistent and gets evicted at op1 so `out` can fit.
    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 2u);

    // `in` is the only spilled buffer; weightA stays admitted and reaches
    // sharedExternalBuffers.
    ASSERT_EQ(result.spilledBuffers.size(), 1u);
    EXPECT_EQ(result.spilledBuffers.front(), in);
    EXPECT_EQ(result.sharedExternalBuffers.count(weightA), 1u);
    EXPECT_EQ(result.sharedExternalBuffers.count(in), 0u);

    // Retroactive alloc of `in` at its original persistent slot (persistentFloor
    // before eviction = 8704 - 6656 = 2048) is spliced into op0's schedule
    // entry alongside the normal `mid` allocation.
    const auto& op0Entry = result.iterationSchedule[0];
    ASSERT_EQ(op0Entry.allocInfo.opIdx, 0u);
    const auto findAlloc = [](const ComputeExplicitSchedule& e, mlir::Value v) {
        return std::find_if(e.allocations.begin(), e.allocations.end(), [&](const auto& p) {
            return p.first == v;
        });
    };
    const auto inAllocIt = findAlloc(op0Entry, in);
    ASSERT_NE(inAllocIt, op0Entry.allocations.end());
    EXPECT_EQ(inAllocIt->second, 2048u);
    EXPECT_NE(findAlloc(op0Entry, mid), op0Entry.allocations.end());

    // Spill dealloc of `in` lands on op1's schedule entry.
    const auto& op1Entry = result.iterationSchedule[1];
    ASSERT_EQ(op1Entry.allocInfo.opIdx, 1u);
    EXPECT_NE(std::find(op1Entry.deallocations.begin(), op1Entry.deallocations.end(), in),
              op1Entry.deallocations.end());

    // Persistent zone shrinks by the evicted buffer's aligned size (2560),
    // freeing space that the retry allocation of `out` reuses at addr 2048.
    EXPECT_EQ(result.persistentReservedBytes, 4096u);
    EXPECT_LE(result.peakUsedBytes, memoryLimit);
}

TEST_F(MLIR_VFAllocateLinear, PersistentEvictionWithMisalignedMemoryLimitKeepsReservationAligned) {
    auto weightA = createBuffer(4096);
    auto weightB = createBuffer(2048);
    auto in = createBuffer(2560);
    auto mid = createBuffer(2048);
    auto out = createBuffer(512);

    LoopBody body{computeOp(0, {weightA, in}, {mid}), computeOp(1, {weightB, mid}, {out})};
    auto region = makeVfRegion({body, body});

    // Deliberately 32 bytes off the buffers' 64-byte alignment. Previously the
    // first persistent eviction consumed those 32 bytes without admission
    // having reserved them.
    constexpr AddressType memoryLimit = 8736;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    EXPECT_EQ((memoryLimit - result.persistentReservedBytes) % 64, 0u);
    EXPECT_GE(result.persistentReservedBytes, 4096u);
    EXPECT_LE(result.peakUsedBytes, memoryLimit);
}

TEST_F(MLIR_VFAllocateLinear, PersistentAdmissionUsesEvictionOrderForMixedAlignments) {
    auto persistentA = createBuffer(65, /*rawAlign=*/64);
    auto persistentB = createBuffer(63, /*rawAlign=*/32);
    auto persistentC = createBuffer(63, /*rawAlign=*/32);
    auto persistentD = createBuffer(63, /*rawAlign=*/32);
    auto temporary0 = createBuffer(64, /*rawAlign=*/1);
    auto temporary1 = createBuffer(64, /*rawAlign=*/1);

    LoopBody body0{computeOp(0, {persistentA, persistentB, persistentC, persistentD}, {temporary0}),
                   dataOut(1, {temporary0}, {})};
    LoopBody body1{computeOp(2, {persistentA, persistentB, persistentC, persistentD}, {temporary1}),
                   dataOut(3, {temporary1}, {})};
    auto region = makeVfRegion({body0, body1});

    // Four candidates bypass the classifier's small mixed-alignment order
    // search. The allocator must therefore evaluate its own back-to-front
    // eviction order instead of assuming forward admission order is equivalent.
    constexpr AddressType memoryLimit = 321;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    ASSERT_EQ(classifier.persistentFitOrder.size(), 4u);

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    EXPECT_EQ(result.sharedExternalBuffers.size(), 4u);
    EXPECT_LE(result.peakUsedBytes, memoryLimit);
}

// Verify that admission packs persistent buffers in the same back-to-front
// order used by eviction. Otherwise admission may underestimate the required
// space, causing an out-of-bounds allocation after a persistent spill.

// The four persistent candidates exceed the classifier's exhaustive
// mixed-alignment search limit. Verify that allocator admission instead packs
// them in the normal back-to-front eviction order.
//
// With memoryLimit=321, eviction order D, C, B, A consumes 257 bytes,
// leaving [0, 64) for the temporary buffer:
//
//   temporary: [0, 64)
//   D:         [64, 127)
//   C:         [128, 191)
//   B:         [192, 255)
//   A:         [256, 321)
//
// All four persistents must be admitted and the complete schedule must fit.
TEST_F(MLIR_VFAllocateLinear, PersistentEvictionSkipRejectsMixedAlignmentRepackOverflow) {
    auto persistentA = createBuffer(65, /*rawAlign=*/64);
    auto persistentB = createBuffer(63, /*rawAlign=*/32);
    auto persistentC = createBuffer(63, /*rawAlign=*/32);
    auto persistentD = createBuffer(63, /*rawAlign=*/32);
    auto mid0 = createBuffer(64, /*rawAlign=*/1);
    auto out0 = createBuffer(64, /*rawAlign=*/1);
    auto mid1 = createBuffer(64, /*rawAlign=*/1);
    auto out1 = createBuffer(64, /*rawAlign=*/1);

    // At op1, B/C/D are protected as current inputs. Memory pressure therefore
    // considers A even though it is not the back of the modeled eviction stack.
    LoopBody body0{computeOp(0, {persistentA}, {mid0}),
                   computeOp(1, {persistentB, persistentC, persistentD, mid0}, {out0}), dataOut(2, {out0}, {})};
    LoopBody body1{computeOp(3, {persistentA}, {mid1}),
                   computeOp(4, {persistentB, persistentC, persistentD, mid1}, {out1}), dataOut(5, {out1}, {})};
    auto region = makeVfRegion({body0, body1});

    constexpr AddressType memoryLimit = 321;
    auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.classifierSupportedRegion);
    ASSERT_EQ(classifier.persistentFitOrder.size(), 4u);

    // Exercise the allocator's spill-time safety check directly. The normal
    // classifier accounts for both live temporaries and may admit fewer
    // persistents before this path is reached.
    classifier.temporaryPeakBytes = 64;

    VfSchedStrategyDescriptor params;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    // Repacking A before the protected B/C/D buffers exceeds the reserved zone.
    // The allocator must reject that persistent spill instead of placing A past
    // memoryLimit; with no other legal victim, this schedule is infeasible.
    EXPECT_FALSE(result.feasible);
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

    VfSchedStrategyDescriptor params;  // IntraIterFetch=DISABLED, LastComputeOverlapWithNextIterInputs=DISABLED
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

    VfSchedStrategyDescriptor params;  // IntraIterFetch=DISABLED, LastComputeOverlapWithNextIterInputs=DISABLED
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

// ---------------------------------------------------------------------------
// Verify that last compute op output in a loop body that is later copied
// out by a DATA_OUT op does not overlap in memory with any of first compute op inputs
// which are loaded by DATA_IN ops
//
//        DATA_IN -->|     DATA_IN ---->|
//     DATA_IN -->COMPUTE_OP ----> COMPUTE_OP -> DATA_OUT
//                                         |------> DATA_OUT
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, LastComputeOpOutputsDoesNotOverlapFirstComputeOpInputs) {
    // Iteration 0
    //   OP0 = DATA_IN
    //   OP1 = DATA_IN
    //   OP2 = COMPUTE_OP
    //   OP3 = DATA_IN
    //   OP4 = COMPUTE_OP
    //   OP5 = DATA_OUT
    //   OP6 = DATA_OUT

    auto in0a = createBuffer(512);
    auto in0b = createBuffer(512);
    auto mid0 = createBuffer(512);
    auto in0c = createBuffer(512);
    auto out0a = createBuffer(256);
    auto out0b = createBuffer(256);

    LoopBody body0{dataIn(0, {}, {in0a}),
                   dataIn(1, {}, {in0b}),
                   computeOp(2, {in0a, in0b}, {mid0}),
                   dataIn(3, {}, {in0c}),
                   computeOp(4, {mid0, in0c}, {out0a, out0b}),
                   dataOut(5, {out0a}, {}),
                   dataOut(6, {out0b}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(512);
    auto in1b = createBuffer(512);
    auto mid1 = createBuffer(512);
    auto in1c = createBuffer(512);
    auto out1a = createBuffer(256);
    auto out1b = createBuffer(256);

    LoopBody body1{dataIn(6, {}, {in1a}),
                   dataIn(7, {}, {in1b}),
                   computeOp(8, {in1a, in1b}, {mid1}),
                   dataIn(9, {}, {in1c}),
                   computeOp(10, {mid1, in1c}, {out1a, out1b}),
                   dataOut(11, {out1a}, {}),
                   dataOut(12, {out1b}, {})};

    auto region = makeVfRegion({body0, body1});

    constexpr AddressType memoryLimit = 2048;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;  // IntraIterFetch=DISABLED, LastComputeOverlapWithNextIterInputs=DISABLED
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 7u);

    auto in0aRange = getBufferAssignedRange(result, classifier, in0a);
    auto in0bRange = getBufferAssignedRange(result, classifier, in0b);
    auto out0aRange = getBufferAssignedRange(result, classifier, out0a);
    auto out0bRange = getBufferAssignedRange(result, classifier, out0b);
    ASSERT_TRUE(in0aRange.has_value());
    ASSERT_TRUE(in0bRange.has_value());
    ASSERT_TRUE(out0aRange.has_value());
    ASSERT_TRUE(out0bRange.has_value());

    // Make sure out0a and out0b do not overlap with in0a and in0b
    EXPECT_FALSE(rangesOverlap(*out0aRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*out0bRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*out0aRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*out0bRange, *in0aRange));
}

// ---------------------------------------------------------------------------
// Variant of LastComputeOpOutputsDoesNotOverlapFirstComputeOpInputs where
// the first compute op has two inputs fed by two DATA_IN DMAs, and the last
// compute op produces a single output drained by one DATA_OUT DMA.
//
// Buffer sizes are tuned so `out0` cannot fit while BOTH first-input ranges
// are excluded, but does fit once the second (later-loaded) input range is
// dropped from the exclusion set. This exercises the incremental-shrink retry
// logic in tryAllocTemporary: try N extras, N-1, ..., 1, then no extras. The
// resulting placement overlaps in0b but preserves the pipeline-critical
// non-overlap with in0a.
//
//     DATA_IN -->|
//     DATA_IN -->COMPUTE_OP ----> COMPUTE_OP -> DATA_OUT
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, LastComputeOpSingleOutputDoesNotOverlapFirstComputeOpFirstInput) {
    // Iteration 0
    //   OP0 = DATA_IN
    //   OP1 = DATA_IN
    //   OP2 = COMPUTE_OP
    //   OP3 = COMPUTE_OP
    //   OP4 = DATA_OUT

    auto in0a = createBuffer(1024);
    auto in0b = createBuffer(1024);
    auto mid0 = createBuffer(512);
    auto out0 = createBuffer(1024);

    LoopBody body0{dataIn(0, {}, {in0a}), dataIn(1, {}, {in0b}), computeOp(2, {in0a, in0b}, {mid0}),
                   computeOp(3, {mid0}, {out0}), dataOut(4, {out0}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(1024);
    auto in1b = createBuffer(1024);
    auto mid1 = createBuffer(512);
    auto out1 = createBuffer(1024);

    LoopBody body1{dataIn(5, {}, {in1a}), dataIn(6, {}, {in1b}), computeOp(7, {in1a, in1b}, {mid1}),
                   computeOp(8, {mid1}, {out1}), dataOut(9, {out1}, {})};

    auto region = makeVfRegion({body0, body1});

    // Layout at OP3 (out0 allocation): in0a and in0b freed and merged into a
    // single [0, 2048) gap; mid0 occupies [2048, 2560); top free gap is
    // [2560, memoryLimit). memoryLimit is chosen so:
    //   * Excluding both first-input ranges leaves only [2560, memoryLimit) = 896B
    //     — too small for out0(1024).
    //   * Excluding only in0a's range leaves [1024, 2048) = 1024B — fits out0
    //     exactly at addr 1024, overlapping in0b's old range.
    constexpr AddressType memoryLimit = 3456;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;  // IntraIterFetch=DISABLED, LastComputeOverlapWithNextIterInputs=DISABLED
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 5u);

    auto in0aRange = getBufferAssignedRange(result, classifier, in0a);
    auto in0bRange = getBufferAssignedRange(result, classifier, in0b);
    auto out0Range = getBufferAssignedRange(result, classifier, out0);
    ASSERT_TRUE(in0aRange.has_value());
    ASSERT_TRUE(in0bRange.has_value());
    ASSERT_TRUE(out0Range.has_value());

    // Output range does not overlap with first input range (in0a) but due to memory constraints
    // it does overlap with second input range (in0b). This way DATA_OUT DMA can be pipelines together with
    // first DATA_IN DMA of next iteration but not with second DATA_IN DMA of next iteration.
    EXPECT_FALSE(rangesOverlap(*out0Range, *in0aRange));
    EXPECT_TRUE(rangesOverlap(*out0Range, *in0bRange));
}

// ---------------------------------------------------------------------------
// Variant of LastComputeOpOutputsDoesNotOverlapFirstComputeOpInputs where the
// last compute op emits two outputs but there is not enough free memory to
// keep both of them off ALL first-compute inputs. Sizes are tuned so:
//   * The longer-lived output (out0b, drained by the later DATA_OUT) is
//     allocated first (buffers are sorted by lastUseOp DESC before alloc).
//     Its extras=[in0a, in0b] attempt fails; the tryAllocTemporary
//     shrink loop drops in0b (tail-most) and succeeds — placing out0b in
//     in0b's old slot, so out0b overlaps in0b but avoids in0a.
//   * The shorter-lived output (out0a) is allocated next; in0b's slot is
//     now occupied so its exclusion is trimmed away by gap intersection,
//     leaving only in0a excluded. out0a fits into the top free gap and
//     avoids in0a.
//
//        DATA_IN(in0b) -->|     DATA_IN ---->|
//     DATA_IN(in0a) -->COMPUTE_OP ----> COMPUTE_OP -> DATA_OUT (out0a)
//                                         |------------> DATA_OUT (out0b)
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, LastComputeOpSecondOutputAvoidsFirstInputButOverlapsSecondInput) {
    // Iteration 0
    //   OP0 = DATA_IN         -> in0a
    //   OP1 = DATA_IN         -> in0b
    //   OP2 = COMPUTE_OP      -> mid0
    //   OP3 = COMPUTE_OP      -> out0a, out0b
    //   OP4 = DATA_OUT(out0a)   // out0a drained first  -> shorter life
    //   OP5 = DATA_OUT(out0b)   // out0b drained last   -> longer life

    auto in0a = createBuffer(1024);
    auto in0b = createBuffer(1024);
    auto mid0 = createBuffer(512);
    auto out0a = createBuffer(512);
    auto out0b = createBuffer(1024);

    LoopBody body0{dataIn(0, {}, {in0a}),
                   dataIn(1, {}, {in0b}),
                   computeOp(2, {in0a, in0b}, {mid0}),
                   computeOp(3, {mid0}, {out0b, out0a}),
                   dataOut(4, {out0a}, {}),
                   dataOut(5, {out0b}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(1024);
    auto in1b = createBuffer(1024);
    auto mid1 = createBuffer(512);
    auto out1a = createBuffer(512);
    auto out1b = createBuffer(1024);

    LoopBody body1{dataIn(6, {}, {in1a}),
                   dataIn(7, {}, {in1b}),
                   computeOp(8, {in1a, in1b}, {mid1}),
                   computeOp(9, {mid1}, {out1b, out1a}),
                   dataOut(10, {out1a}, {}),
                   dataOut(11, {out1b}, {})};

    auto region = makeVfRegion({body0, body1});

    // Layout at OP3 (last compute) after in0a/in0b are freed:
    //   free gap [0, 2048) (in0a+in0b slot merged) and top free gap
    //   [2560, memoryLimit); mid0 occupies [2048, 2560).
    // memoryLimit=3072 -> top gap = 512B.
    //   * out_0 (1024B) with extras=[in0a, in0b]: excluded [0,2048), remaining
    //     top gap 512B — fails. Retry drops in0b: excluded [0,1024), remaining
    //     [1024, 2048) fits out_0 at addr 1024 (overlaps in0b, avoids in0a).
    //   * out_1 (512B) fits into the top gap [2560, 3072) — avoids in0a.
    constexpr AddressType memoryLimit = 3072;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 6u);

    auto in0aRange = getBufferAssignedRange(result, classifier, in0a);
    auto in0bRange = getBufferAssignedRange(result, classifier, in0b);
    auto out0aRange = getBufferAssignedRange(result, classifier, out0a);
    auto out0bRange = getBufferAssignedRange(result, classifier, out0b);
    ASSERT_TRUE(in0aRange.has_value());
    ASSERT_TRUE(in0bRange.has_value());
    ASSERT_TRUE(out0aRange.has_value());
    ASSERT_TRUE(out0bRange.has_value());

    // out0b fits only after dropping in0b from the extras — it lands in in0b's
    // old slot.
    EXPECT_FALSE(rangesOverlap(*out0bRange, *in0aRange));
    EXPECT_TRUE(rangesOverlap(*out0bRange, *in0bRange));
}

// ---------------------------------------------------------------------------
// Verify that last compute op inputs and outputs in a loop body that is later copied
// out by a DATA_OUT op does not overlap in memory with any of first compute op inputs
// which are loaded by DATA_IN ops
//
//        DATA_IN -->|    DATA_IN ---->|    DATA_IN ---->|
//     DATA_IN -->COMPUTE_OP ----> COMPUTE_OP ----> COMPUTE_OP -> DATA_OUT
//                                                          |------> DATA_OUT
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, LastComputeOpInputsAndOutputsDoNotOverlapFirstComputeOpInputs) {
    // Iteration 0
    //   OP0 = DATA_IN
    //   OP1 = DATA_IN
    //   OP2 = COMPUTE_OP
    //   OP3 = DATA_IN
    //   OP4 = COMPUTE_OP
    //   OP5 = DATA_IN
    //   OP6 = COMPUTE_OP
    //   OP7 = DATA_OUT
    //   OP8 = DATA_OUT

    auto in0a = createBuffer(512);
    auto in0b = createBuffer(512);
    auto mid0a = createBuffer(512);
    auto in0c = createBuffer(512);
    auto mid0b = createBuffer(512);
    auto in0d = createBuffer(512);
    auto out0a = createBuffer(256);
    auto out0b = createBuffer(256);

    LoopBody body0{dataIn(0, {}, {in0a}),
                   dataIn(1, {}, {in0b}),
                   computeOp(2, {in0a, in0b}, {mid0a}),
                   dataIn(3, {}, {in0c}),
                   computeOp(4, {mid0a, in0c}, {mid0b}),
                   dataIn(5, {}, {in0d}),
                   computeOp(6, {mid0b, in0d}, {out0a, out0b}),
                   dataOut(7, {out0a}, {}),
                   dataOut(8, {out0b}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(512);
    auto in1b = createBuffer(512);
    auto mid1a = createBuffer(512);
    auto in1c = createBuffer(512);
    auto mid1b = createBuffer(512);
    auto in1d = createBuffer(512);
    auto out1a = createBuffer(256);
    auto out1b = createBuffer(256);

    LoopBody body1{dataIn(9, {}, {in1a}),
                   dataIn(10, {}, {in1b}),
                   computeOp(11, {in1a, in1b}, {mid1a}),
                   dataIn(12, {}, {in1c}),
                   computeOp(13, {mid1a, in1c}, {mid1b}),
                   dataIn(14, {}, {in1d}),
                   computeOp(15, {mid1b, in1d}, {out1a, out1b}),
                   dataOut(16, {out1a}, {}),
                   dataOut(17, {out1b}, {})};

    auto region = makeVfRegion({body0, body1});

    // memoryLimit is sized so the next iteration inputs DMAs co-allocation heuristic can place
    // last-compute in/out buffers (mid0b, in0d, out0a, out0b) alongside
    // op4's own output without spilling into the first-compute input
    // ranges (in0a, in0b). At op4:
    //   * in0c occupies [0, 512), mid0a occupies [1024, 1536), and the
    //     [512, 1024) slot (in0b's old range) is excluded from placement.
    //   * The extended attempt fills [1536, 3072) with out0b, out0a,
    //     mid0b, in0d (256 + 256 + 512 + 512 = 1536 bytes).
    constexpr AddressType memoryLimit = 3072;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;  // IntraIterFetch=DISABLED
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 9u);

    auto in0aRange = getBufferAssignedRange(result, classifier, in0a);
    auto in0bRange = getBufferAssignedRange(result, classifier, in0b);
    auto mid0bRange = getBufferAssignedRange(result, classifier, mid0b);
    auto in0dRange = getBufferAssignedRange(result, classifier, in0d);
    auto out0aRange = getBufferAssignedRange(result, classifier, out0a);
    auto out0bRange = getBufferAssignedRange(result, classifier, out0b);
    ASSERT_TRUE(in0aRange.has_value());
    ASSERT_TRUE(in0bRange.has_value());
    ASSERT_TRUE(mid0bRange.has_value());
    ASSERT_TRUE(in0dRange.has_value());
    ASSERT_TRUE(out0aRange.has_value());
    ASSERT_TRUE(out0bRange.has_value());

    // Neither the last compute op inputs (mid0b, in0d) nor its outputs (out0a, out0b)
    // may overlap with the first compute op inputs (in0a, in0b).
    EXPECT_FALSE(rangesOverlap(*mid0bRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*mid0bRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*in0dRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*in0dRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*out0aRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*out0aRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*out0bRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*out0bRange, *in0bRange));
}

// ---------------------------------------------------------------------------
// Variant of LastComputeOpInputsAndOutputsDoNotOverlapFirstComputeOpInputs with
// four compute ops that all run in place: each DPU writes its result back into
// its own first input buffer, so a single activation buffer (act0) is threaded
// through the whole chain as both input and output of every compute op.
//
//     DATA_IN(act0) --> DPU0 --> DPU1 --> DPU2 --> DPU3 --> DATA_OUT(act0)
//                        ^        ^        ^        ^
//                        |        |        |        |
//                    DATA_IN  DATA_IN  DATA_IN  DATA_IN
//                     (w0a)    (w0b)    (w0c)    (w0d)
//
// In-place chaining means the activation cannot be steered away from the first
// compute op input ranges — it *is* one of them. What the overlap heuristic can
// still do is keep the last compute op's other input (the w0d weight tile) off
// the ranges of the first compute op inputs, so the next iteration's activation
// and first weight DMAs can be issued while DPU3 and the DATA_OUT of this
// iteration are still running.
//
// The overlap-aware allocation keeps w0d separate from the first-compute inputs
// by excluding w0a's range when w0d is allocated at OP7; the shared act0 buffer
// remains live throughout the chain and is intentionally not an exclusion candidate.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, InPlaceComputeChainLastComputeWeightsNotOverlapFirstComputeOpInputs) {
    // Iteration 0
    //   OP0 = DATA_IN                          -> act0
    //   OP1 = DATA_IN                          -> w0a
    //   OP2 = COMPUTE_OP(act0, w0a) -> act0      // in place
    //   OP3 = DATA_IN                          -> w0b
    //   OP4 = COMPUTE_OP(act0, w0b) -> act0      // in place
    //   OP5 = DATA_IN                          -> w0c
    //   OP6 = COMPUTE_OP(act0, w0c) -> act0      // in place
    //   OP7 = DATA_IN                          -> w0d
    //   OP8 = COMPUTE_OP(act0, w0d) -> act0      // in place
    //   OP9 = DATA_OUT(act0)

    auto act0 = createBuffer(1024);
    auto w0a = createBuffer(512);
    auto w0b = createBuffer(512);
    auto w0c = createBuffer(512);
    auto w0d = createBuffer(512);

    LoopBody body0{dataIn(0, {}, {act0}),
                   dataIn(1, {}, {w0a}),
                   computeOp(2, {act0, w0a}, {act0}),
                   dataIn(3, {}, {w0b}),
                   computeOp(4, {act0, w0b}, {act0}),
                   dataIn(5, {}, {w0c}),
                   computeOp(6, {act0, w0c}, {act0}),
                   dataIn(7, {}, {w0d}),
                   computeOp(8, {act0, w0d}, {act0}),
                   dataOut(9, {act0}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto act1 = createBuffer(1024);
    auto w1a = createBuffer(512);
    auto w1b = createBuffer(512);
    auto w1c = createBuffer(512);
    auto w1d = createBuffer(512);

    LoopBody body1{dataIn(10, {}, {act1}),
                   dataIn(11, {}, {w1a}),
                   computeOp(12, {act1, w1a}, {act1}),
                   dataIn(13, {}, {w1b}),
                   computeOp(14, {act1, w1b}, {act1}),
                   dataIn(15, {}, {w1c}),
                   computeOp(16, {act1, w1c}, {act1}),
                   dataIn(17, {}, {w1d}),
                   computeOp(18, {act1, w1d}, {act1}),
                   dataOut(19, {act1}, {})};

    auto region = makeVfRegion({body0, body1});

    // memoryLimit leaves room for the in-place activation plus two concurrent
    // weight slots and nothing more:
    //   * OP0 co-allocates act0 at [0, 1024) and w0d at [1024, 1536).
    //   * OP1 places w0a in the only gap left, [1536, 2048).
    //   * OP3 and OP5 reuse that same freed slot for w0b and w0c, again
    //     [1536, 2048) each — the middle weights rotate through one slot.
    //   * w0d keeps its dedicated slot for the whole iteration, so it never
    //     lands on w0a's range.
    constexpr AddressType memoryLimit = 2048;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);
    ASSERT_TRUE(classifier.classifierSupportedRegion);

    // The activation is recognized as structurally in place: same SSA value on
    // the input and the output side of the same op.
    const auto act0RecIt = classifier.valueToRecord.find(act0);
    ASSERT_NE(act0RecIt, classifier.valueToRecord.end());
    EXPECT_TRUE(classifier.buffers[act0RecIt->second].isInPlaceAlias);

    VfSchedStrategyDescriptor params;  // FetchScope=NONE
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 10u);
    EXPECT_TRUE(result.spilledBuffers.empty());
    EXPECT_LE(result.peakUsedBytes, memoryLimit);

    // An in-place op must not re-allocate its output: act0 is allocated once by
    // the DATA_IN that loads it and stays at that address for the whole chain.
    size_t act0AllocCount = 0;
    for (const auto& entry : result.iterationSchedule) {
        act0AllocCount += std::count_if(entry.allocations.begin(), entry.allocations.end(), [&](const auto& p) {
            return p.first == act0;
        });
    }
    EXPECT_EQ(act0AllocCount, 1u);

    // All four compute ops reuse already-live buffers only, so none of them
    // emits an allocation.
    EXPECT_TRUE(result.iterationSchedule[2].allocations.empty());
    EXPECT_TRUE(result.iterationSchedule[4].allocations.empty());
    EXPECT_TRUE(result.iterationSchedule[6].allocations.empty());
    EXPECT_TRUE(result.iterationSchedule[8].allocations.empty());

    auto act0Range = getBufferAssignedRange(result, classifier, act0);
    auto w0aRange = getBufferAssignedRange(result, classifier, w0a);
    auto w0bRange = getBufferAssignedRange(result, classifier, w0b);
    auto w0cRange = getBufferAssignedRange(result, classifier, w0c);
    auto w0dRange = getBufferAssignedRange(result, classifier, w0d);
    ASSERT_TRUE(act0Range.has_value());
    ASSERT_TRUE(w0aRange.has_value());
    ASSERT_TRUE(w0bRange.has_value());
    ASSERT_TRUE(w0cRange.has_value());
    ASSERT_TRUE(w0dRange.has_value());

    // The last compute op input (w0d) avoids the ranges of both first compute op
    // inputs (act0 and w0a), so the next iteration's DATA_IN DMAs can overlap
    // with this iteration's last compute op and its DATA_OUT.
    EXPECT_FALSE(rangesOverlap(*w0dRange, *act0Range));
    EXPECT_FALSE(rangesOverlap(*w0dRange, *w0aRange));
}

// ---------------------------------------------------------------------------
// IntraIterFetch=ENABLED: the DATA_IN DMAs that load the next COMPUTE op's inputs
// are allocated while the current COMPUTE op still owns its buffers, so they
// get ranges disjoint from all of its inputs and outputs and can execute in
// parallel with it.
//
//   OP0_DATA_IN OP1_DATA_IN ->|            OP3_DATA_IN ->| OP4_DATA_IN ->|
//        |---------------------> OP2_COMPUTE -----------|---------------|-> OP5_COMPUTE -> OP6_DATA_OUT
//
// Layout with prefetching at OP2 (limit 4096, everything 512B):
//   in0a [0, 512), in0b [512, 1024), mid0 [1024, 1536) are live, so the
//   prefetched in0c lands at [1536, 2048) and in0d at [2048, 2560).
// Without prefetching OP3 first releases in0a/in0b (dead after OP2) and in0c
// reuses in0a's range, which is exactly what serializes the DMA behind OP2.
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, PrefetchOfNextComputeInputsAvoidsCurrentComputeRanges) {
    // Iteration 0
    auto in0a = createBuffer(512);
    auto in0b = createBuffer(512);
    auto mid0 = createBuffer(512);
    auto in0c = createBuffer(512);
    auto in0d = createBuffer(512);
    auto out0 = createBuffer(512);

    LoopBody body0{dataIn(0, {}, {in0a}), dataIn(1, {}, {in0b}), computeOp(2, {in0a, in0b}, {mid0}),
                   dataIn(3, {}, {in0c}), dataIn(4, {}, {in0d}), computeOp(5, {mid0, in0c, in0d}, {out0}),
                   dataOut(6, {out0}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(512);
    auto in1b = createBuffer(512);
    auto mid1 = createBuffer(512);
    auto in1c = createBuffer(512);
    auto in1d = createBuffer(512);
    auto out1 = createBuffer(512);

    LoopBody body1{dataIn(7, {}, {in1a}),  dataIn(8, {}, {in1b}),  computeOp(9, {in1a, in1b}, {mid1}),
                   dataIn(10, {}, {in1c}), dataIn(11, {}, {in1d}), computeOp(12, {mid1, in1c, in1d}, {out1}),
                   dataOut(13, {out1}, {})};

    auto region = makeVfRegion({body0, body1});

    constexpr AddressType memoryLimit = 4096;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;
    params.intraIterFetch = VfSchedStrategyDescriptor::IntraIterFetch::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 7u);
    EXPECT_TRUE(result.spilledBuffers.empty());

    const auto in0aRange = getBufferAssignedRange(result, classifier, in0a);
    const auto in0bRange = getBufferAssignedRange(result, classifier, in0b);
    const auto mid0Range = getBufferAssignedRange(result, classifier, mid0);
    const auto in0cRange = getBufferAssignedRange(result, classifier, in0c);
    const auto in0dRange = getBufferAssignedRange(result, classifier, in0d);
    ASSERT_TRUE(in0aRange.has_value());
    ASSERT_TRUE(in0bRange.has_value());
    ASSERT_TRUE(mid0Range.has_value());
    ASSERT_TRUE(in0cRange.has_value());
    ASSERT_TRUE(in0dRange.has_value());

    // Next COMPUTE op inputs must not share memory with any input or output of
    // the COMPUTE op they are prefetched over.
    EXPECT_FALSE(rangesOverlap(*in0cRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*in0cRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*in0cRange, *mid0Range));
    EXPECT_FALSE(rangesOverlap(*in0dRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*in0dRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*in0dRange, *mid0Range));
    // The two prefetched buffers must not share memory with each other either.
    EXPECT_FALSE(rangesOverlap(*in0cRange, *in0dRange));

    // A prefetch only decides the address early - the allocation entry stays at
    // the DATA_IN op that produces the buffer. Downstream consumers rely on it:
    // verifyLocalSchedule() requires every entry to allocate its own outputs and
    // the pipelined pairwise rewrite moves compute stages together with their
    // input DMA entries across iterations.
    EXPECT_EQ(getBufferAllocEntryPos(result, in0c), std::optional<size_t>(3u));
    EXPECT_EQ(getBufferAllocEntryPos(result, in0d), std::optional<size_t>(4u));

    // Reference point: with IntraIterFetch=DISABLED the same region reuses the COMPUTE
    // op's released input ranges for the next DATA_IN buffers.
    VfSchedStrategyDescriptor noPrefetchParams;
    const auto noPrefetch = VfAllocateLinear(region, classifier, memoryLimit, noPrefetchParams).performAllocation();
    ASSERT_TRUE(noPrefetch.feasible);
    const auto refIn0aRange = getBufferAssignedRange(noPrefetch, classifier, in0a);
    const auto refIn0cRange = getBufferAssignedRange(noPrefetch, classifier, in0c);
    ASSERT_TRUE(refIn0aRange.has_value());
    ASSERT_TRUE(refIn0cRange.has_value());
    EXPECT_TRUE(rangesOverlap(*refIn0cRange, *refIn0aRange));
}

TEST_F(MLIR_VFAllocateLinear, PrefetchRespectsLastComputeInputExclusions) {
    auto makeBody = [&](size_t firstOp) {
        auto firstInput = createBuffer(1024);
        auto mid = createBuffer(512);
        auto scratch = createBuffer(512);
        auto nextInput = createBuffer(1536);
        auto lastInput = createBuffer(1024);
        auto otherInput = createBuffer(512);
        auto output = createBuffer(512);
        return LoopBody{dataIn(firstOp, {}, {firstInput}),
                        computeOp(firstOp + 1, {firstInput}, {mid, scratch}),
                        computeOp(firstOp + 2, {mid, scratch}, {nextInput}),
                        dataIn(firstOp + 3, {}, {lastInput}),
                        dataIn(firstOp + 4, {}, {otherInput}),
                        dataOut(firstOp + 5, {otherInput}, {}),
                        computeOp(firstOp + 6, {nextInput, lastInput}, {output}),
                        dataOut(firstOp + 7, {output}, {})};
    };
    auto region = makeVfRegion({makeBody(0), makeBody(8)});
    const auto& body = region.schedulingLoop->loopBodies.front();
    const auto firstInput = body[0].outBuffers.front();
    const auto otherInput = body[4].outBuffers.front();
    constexpr AddressType memoryLimit = 3584;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;
    params.intraIterFetch = VfSchedStrategyDescriptor::IntraIterFetch::ENABLED;
    params.lastCompOverlapNextIterIn = VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();
    ASSERT_TRUE(result.feasible);
    EXPECT_TRUE(result.spilledBuffers.empty());

    const auto firstRange = getBufferAssignedRange(result, classifier, firstInput);
    const auto otherRange = getBufferAssignedRange(result, classifier, otherInput);
    ASSERT_TRUE(firstRange.has_value());
    ASSERT_TRUE(otherRange.has_value());
    // The last-compute input cannot prefetch into the first-input range, so
    // the following DATA_IN candidate reserves that gap instead.
    EXPECT_TRUE(rangesOverlap(*otherRange, *firstRange));
}

// ---------------------------------------------------------------------------
// Prefetching is best effort: a candidate that does not fit next to the current
// COMPUTE op's buffers is left to the regular path at its own position, and the
// remaining candidates are still attempted.
//
// Layout at OP2 (limit 3072): in0a [0, 512), in0b [512, 1024) and mid0
// [1024, 1536) are live, leaving 1536B free.
//   * in0c (1536B) fits exactly -> prefetched at [1536, 3072).
//   * in0d (512B) has no free range left -> not prefetched. At OP4 in0a/in0b are
//     already released, so it reuses in0a's range at [0, 512).
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, PartialPrefetchWhenNotAllNextComputeInputsFit) {
    // Iteration 0
    auto in0a = createBuffer(512);
    auto in0b = createBuffer(512);
    auto mid0 = createBuffer(512);
    auto in0c = createBuffer(1536);
    auto in0d = createBuffer(512);
    auto out0 = createBuffer(512);

    LoopBody body0{dataIn(0, {}, {in0a}), dataIn(1, {}, {in0b}), computeOp(2, {in0a, in0b}, {mid0}),
                   dataIn(3, {}, {in0c}), dataIn(4, {}, {in0d}), computeOp(5, {mid0, in0c, in0d}, {out0}),
                   dataOut(6, {out0}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(512);
    auto in1b = createBuffer(512);
    auto mid1 = createBuffer(512);
    auto in1c = createBuffer(1536);
    auto in1d = createBuffer(512);
    auto out1 = createBuffer(512);

    LoopBody body1{dataIn(7, {}, {in1a}),  dataIn(8, {}, {in1b}),  computeOp(9, {in1a, in1b}, {mid1}),
                   dataIn(10, {}, {in1c}), dataIn(11, {}, {in1d}), computeOp(12, {mid1, in1c, in1d}, {out1}),
                   dataOut(13, {out1}, {})};

    auto region = makeVfRegion({body0, body1});

    constexpr AddressType memoryLimit = 3072;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;
    params.intraIterFetch = VfSchedStrategyDescriptor::IntraIterFetch::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 7u);
    EXPECT_TRUE(result.spilledBuffers.empty());

    const auto in0aRange = getBufferAssignedRange(result, classifier, in0a);
    const auto in0bRange = getBufferAssignedRange(result, classifier, in0b);
    const auto mid0Range = getBufferAssignedRange(result, classifier, mid0);
    const auto in0cRange = getBufferAssignedRange(result, classifier, in0c);
    const auto in0dRange = getBufferAssignedRange(result, classifier, in0d);
    ASSERT_TRUE(in0aRange.has_value());
    ASSERT_TRUE(in0bRange.has_value());
    ASSERT_TRUE(mid0Range.has_value());
    ASSERT_TRUE(in0cRange.has_value());
    ASSERT_TRUE(in0dRange.has_value());

    // in0c was prefetched over OP2 - no conflict with OP2's buffers.
    EXPECT_FALSE(rangesOverlap(*in0cRange, *in0aRange));
    EXPECT_FALSE(rangesOverlap(*in0cRange, *in0bRange));
    EXPECT_FALSE(rangesOverlap(*in0cRange, *mid0Range));

    // in0d did not fit next to OP2's buffers, so it is placed only at its own
    // position and reuses the range of an OP2 input.
    EXPECT_TRUE(rangesOverlap(*in0dRange, *in0aRange));

    // Both allocations are recorded at the DATA_IN ops producing them - a
    // prefetch moves the address decision earlier, not the schedule entry.
    EXPECT_EQ(getBufferAllocEntryPos(result, in0c), std::optional<size_t>(3u));
    EXPECT_EQ(getBufferAllocEntryPos(result, in0d), std::optional<size_t>(4u));
}

// ---------------------------------------------------------------------------
// Memory reserved by a prefetch is handed back when a later allocation cannot
// be satisfied. The buffer holds no data yet (its DATA_IN DMA is not scheduled),
// so cancelling costs nothing and runs before persistent eviction and temporary
// spilling. Without it the allocator would have to spill a buffer that has not
// been produced yet - which the downstream scheduler cannot express - or declare
// the region infeasible.
//
// Layout (limit 3072): at OP2 in0a [0, 512), in0b [512, 1024), mid0
// [1024, 1536) are live, leaving 1536B.
//   * in0c (2048B) does not fit -> not prefetched.
//   * in0d (256B) fits -> prefetched at [1536, 1792).
// At OP3 in0a/in0b are released but the largest free range is still too small
// for in0c, so the in0d prefetch is cancelled first and mid0 is then spilled,
// which merges the whole space and lets in0c land at [0, 2048).
// ---------------------------------------------------------------------------
TEST_F(MLIR_VFAllocateLinear, PendingPrefetchIsCancelledBeforeSpillingLiveBuffers) {
    // Iteration 0
    auto in0a = createBuffer(512);
    auto in0b = createBuffer(512);
    auto mid0 = createBuffer(512);
    auto in0c = createBuffer(2048);
    auto in0d = createBuffer(256);
    auto out0 = createBuffer(256);

    LoopBody body0{dataIn(0, {}, {in0a}), dataIn(1, {}, {in0b}), computeOp(2, {in0a, in0b}, {mid0}),
                   dataIn(3, {}, {in0c}), dataIn(4, {}, {in0d}), computeOp(5, {mid0, in0c, in0d}, {out0}),
                   dataOut(6, {out0}, {})};

    // Iteration 1 - mirror structure with fresh SSA values.
    auto in1a = createBuffer(512);
    auto in1b = createBuffer(512);
    auto mid1 = createBuffer(512);
    auto in1c = createBuffer(2048);
    auto in1d = createBuffer(256);
    auto out1 = createBuffer(256);

    LoopBody body1{dataIn(7, {}, {in1a}),  dataIn(8, {}, {in1b}),  computeOp(9, {in1a, in1b}, {mid1}),
                   dataIn(10, {}, {in1c}), dataIn(11, {}, {in1d}), computeOp(12, {mid1, in1c, in1d}, {out1}),
                   dataOut(13, {out1}, {})};

    auto region = makeVfRegion({body0, body1});

    constexpr AddressType memoryLimit = 3072;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    VfSchedStrategyDescriptor params;
    params.intraIterFetch = VfSchedStrategyDescriptor::IntraIterFetch::ENABLED;
    const auto result = VfAllocateLinear(region, classifier, memoryLimit, params).performAllocation();

    ASSERT_TRUE(result.feasible);
    ASSERT_EQ(result.iterationSchedule.size(), 7u);

    // mid0 is the only spill victim: in0d, still unproduced at OP3, must not be
    // picked as one.
    ASSERT_EQ(result.spilledBuffers.size(), 1u);
    EXPECT_EQ(result.spilledBuffers.front(), mid0);

    // Both buffers end up allocated at their own DATA_IN positions: in0c because
    // it never fit next to OP2's buffers, in0d because its prefetch was cancelled.
    EXPECT_EQ(getBufferAllocEntryPos(result, in0c), std::optional<size_t>(3u));
    EXPECT_EQ(getBufferAllocEntryPos(result, in0d), std::optional<size_t>(4u));

    EXPECT_LE(result.peakUsedBytes, memoryLimit);
}
