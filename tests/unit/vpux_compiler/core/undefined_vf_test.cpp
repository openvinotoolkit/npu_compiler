//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_vf.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <functional>
#include <tuple>

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

    mlir::Value createBuffer(vpux::AddressType rawSize, vpux::AddressType rawAlign = 64) {
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

    OpAllocationInfo createDataInOp(size_t opIdx, const SmallVector<mlir::Value>& inBuffers,
                                    const SmallVector<mlir::Value>& outBuffers) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DMA_NN, 0};
        return OpAllocationInfo(opIdx, qt, inBuffers, outBuffers, AllocationType::DATA_IN);
    }

    ComputeRegion makeVfRegion(std::vector<LoopBody> bodies, LoopType type = LoopType::VF) {
        auto loop = std::make_unique<SchedulingLoop>();
        loop->type = type;
        loop->loopBodies = std::move(bodies);
        return ComputeRegion(std::move(loop));
    }

    // Build simple chained iteration bodies with defined
    // number of iterations, ops per iteration and buffer size passed between ops.
    // When `weightsSize > 0`, prepend each compute op with a DATA_IN DMA that fetches a
    // per-iteration weight tile from a single shared external buffer. The shared
    // buffer is created once (outside the per-iteration loop) and referenced by
    // every iteration, so the classifier promotes it to PERSISTENT_CANDIDATE.
    std::vector<LoopBody> makeChainedBodies(size_t numIterations, size_t opsPerIter, vpux::AddressType bufSize,
                                            vpux::AddressType weightsSize = 0) {
        std::vector<LoopBody> bodies;
        bodies.reserve(numIterations);
        size_t opIdx = 0;
        const bool withWeights = weightsSize > 0;
        mlir::Value sharedWeights = withWeights ? createBuffer(weightsSize) : mlir::Value{};
        // Single escaping-output SSA shared across every iteration.
        auto sharedFinalOutput = createBuffer(bufSize);
        for (size_t k = 0; k < numIterations; ++k) {
            LoopBody body;
            body.reserve(withWeights ? opsPerIter * 2 : opsPerIter);
            SmallVector<mlir::Value> prevBuf;
            for (size_t p = 0; p < opsPerIter; ++p) {
                SmallVector<mlir::Value> computeInputs = prevBuf;
                if (withWeights) {
                    SmallVector<mlir::Value> weightTile{createBuffer(weightsSize)};
                    body.push_back(createDataInOp(opIdx++, {sharedWeights}, weightTile));
                    computeInputs.push_back(weightTile.front());
                }
                const bool isLastOp = (p + 1 == opsPerIter);
                SmallVector<mlir::Value> outBuf = isLastOp ? SmallVector<mlir::Value>{sharedFinalOutput}
                                                           : SmallVector<mlir::Value>{createBuffer(bufSize)};
                body.push_back(createComputeOp(opIdx++, computeInputs, outBuf));
                prevBuf = outBuf;
            }
            bodies.push_back(std::move(body));
        }
        return bodies;
    }
};

TEST_F(MLIR_UndefinedVF, NameAndConstruction) {
    UndefinedVF scenario;
    EXPECT_EQ(scenario.getName(), "UndefinedVF");
}

//
// runStrategySearch tests
//

// runStrategySearch driven directly with stubbed callbacks.
// Bypasses the classifier path of getScheduleStrategy so the search
// can be exercised in isolation. A trivial always-feasible allocator
TEST_F(MLIR_UndefinedVF, RunStrategySearchWithStubbedAllocator) {
    // Always-feasible allocator (recipe is ignored in this stub)
    UndefinedVF::AllocatorFn allocator = [](const VfStrategyRecipe& recipe) {
        std::ignore = recipe;  // unused in this stub
        VfAllocateResult r;
        r.feasible = true;
        r.peakUsedBytes = 1024;
        r.persistentReservedBytes = 0;
        return r;
    };

    UndefinedVF scenario;
    const auto outcome = scenario.runStrategySearch(allocator);

    EXPECT_TRUE(outcome.result.feasible);
    ASSERT_TRUE(outcome.selectedVFStrategy.has_value());
}

// runStrategySearch driven by the real VfAllocateLinear allocator
// Mirrors the wiring done in getScheduleStrategy
TEST_F(MLIR_UndefinedVF, RunStrategySearchWithRealLinearAllocatorForFeasibleSchedule) {
    constexpr AddressType memoryLimit = 65536;
    auto region = makeVfRegion(makeChainedBodies(/*iters=*/3, /*ops=*/3, /*size=*/1024));
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);
    ASSERT_GT(classifier.opsPerIteration, 0u);

    // Real allocator — same recipe → params translation as the production
    //`tryStrategy helper
    UndefinedVF::AllocatorFn allocator = [&](const VfStrategyRecipe& recipe) {
        VfSchedStrategyDescriptor params{recipe.fetchScp, recipe.outputRes};
        return VfAllocateLinear(region, classifier, memoryLimit, params, Logger::global()).performAllocation();
    };

    UndefinedVF scenario;
    const auto outcome = scenario.runStrategySearch(allocator);

    EXPECT_TRUE(outcome.result.feasible);
    EXPECT_LE(outcome.result.peakUsedBytes, memoryLimit);
    EXPECT_EQ(outcome.result.iterationSchedule.size(), classifier.opsPerIteration);
}

//
// getScheduleStrategy tests
//

TEST_F(MLIR_UndefinedVF, ThrowsOnNonVfRegion) {
    UndefinedVF scenario;
    {
        auto region = makeVfRegion(makeChainedBodies(/*iters=*/2, /*ops=*/1, /*size=*/1024), LoopType::Tiling);
        EXPECT_ANY_THROW(scenario.getScheduleStrategy(region, /*memorySize=*/8192));
    }
    {
        auto region = makeVfRegion(makeChainedBodies(/*iters=*/1, /*ops=*/1, /*size=*/1024), LoopType::None);
        EXPECT_ANY_THROW(scenario.getScheduleStrategy(region, /*memorySize=*/8192));
    }
}

// MINIMAL on a simple region populates schedule; the shared escaping output emitted by
// makeChainedBodies is the sole PERSISTENT_CANDIDATE.
TEST_F(MLIR_UndefinedVF, MinimalRegionFeasibleAndPopulated) {
    UndefinedVF scenario;
    auto region = makeVfRegion(makeChainedBodies(/*iters=*/2, /*ops=*/2, /*size=*/1024));
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/16384);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.schedule.size(), 2u);
    // reservedSize is the TEMPORARY working-area size carved out for the loop body
    // by the outer scheduler. Must be > 0 for any non-trivial region.
    EXPECT_GT(result.reservedSize, 0u);
    EXPECT_LE(result.reservedSize, 16384u);
    // Every iteration writes into the same shared final-output SSA => one PE entry.
    EXPECT_EQ(result.sharedExternalBuffers.size(), 1u);
}

// Iteration count is replicated across the predefined schedule.
TEST_F(MLIR_UndefinedVF, IterationCountReplicated) {
    UndefinedVF scenario;
    constexpr size_t kIters = 5;
    constexpr size_t kOps = 3;
    auto region = makeVfRegion(makeChainedBodies(kIters, kOps, /*size=*/512));
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/32768);
    ASSERT_EQ(result.schedule.size(), kIters);
    for (const auto& iter : result.schedule) {
        EXPECT_EQ(iter.size(), kOps);
    }
}

// Per-iteration buffer identity: each replicated entry references the
// iteration-specific outBuffers (not the iteration-0 template).
TEST_F(MLIR_UndefinedVF, PerIterationBufferIdentity) {
    UndefinedVF scenario;
    constexpr size_t kIters = 3;
    constexpr size_t kOps = 2;
    auto bodies = makeChainedBodies(kIters, kOps, /*size=*/256);
    // Snapshot iteration-k outBuffers per op for cross-checking.
    std::vector<std::vector<mlir::Value>> expectedOutsByIter(kIters);
    for (size_t k = 0; k < kIters; ++k) {
        for (size_t p = 0; p < kOps; ++p) {
            for (auto v : bodies[k][p].outBuffers) {
                expectedOutsByIter[k].push_back(v);
            }
        }
    }
    auto region = makeVfRegion(std::move(bodies));
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/16384);
    ASSERT_EQ(result.schedule.size(), kIters);
    for (size_t k = 0; k < kIters; ++k) {
        for (size_t p = 0; p < kOps; ++p) {
            const auto& entry = result.schedule[k][p];
            // allocInfo must come from iteration k.
            for (auto v : entry.allocInfo.outBuffers) {
                EXPECT_NE(std::find(expectedOutsByIter[k].begin(), expectedOutsByIter[k].end(), v),
                          expectedOutsByIter[k].end())
                        << "iter=" << k << " op=" << p << " allocInfo references foreign value";
            }
            // Allocations entries must reference iteration-k SSA values.
            for (const auto& [v, _] : entry.allocations) {
                EXPECT_NE(std::find(expectedOutsByIter[k].begin(), expectedOutsByIter[k].end(), v),
                          expectedOutsByIter[k].end())
                        << "iter=" << k << " op=" << p << " allocation references foreign value";
            }
        }
    }
}

// Schedule addresses are replicated identically across all iterations.
TEST_F(MLIR_UndefinedVF, ScheduleAddressesAreReplicated) {
    UndefinedVF scenario;
    constexpr size_t kIters = 4;
    constexpr size_t kOps = 2;
    auto region = makeVfRegion(makeChainedBodies(kIters, kOps, /*size=*/512));
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/16384);
    ASSERT_EQ(result.schedule.size(), kIters);

    for (size_t p = 0; p < kOps; ++p) {
        const auto& iter0Allocs = result.schedule[0][p].allocations;
        for (size_t k = 1; k < kIters; ++k) {
            const auto& iterKAllocs = result.schedule[k][p].allocations;
            ASSERT_EQ(iter0Allocs.size(), iterKAllocs.size()) << "iter=" << k << " op=" << p;
            for (size_t i = 0; i < iter0Allocs.size(); ++i) {
                EXPECT_EQ(iter0Allocs[i].second, iterKAllocs[i].second)
                        << "iter=" << k << " op=" << p << " allocSlot=" << i;
            }
        }
    }
}

// Infeasibility due to memory limit check
TEST_F(MLIR_UndefinedVF, FeasibilityAssertionDueToMemLimit) {
    UndefinedVF scenario;
    auto in = createBuffer(/*size=*/64);
    auto out = createBuffer(/*size=*/8192);
    LoopBody body0{createComputeOp(0, {in}, {out})};
    LoopBody body1{createComputeOp(1, {in}, {out})};
    auto region = makeVfRegion({std::move(body0), std::move(body1)});
    auto result = scenario.getScheduleStrategy(region, /*memorySize=*/256);
    ASSERT_TRUE(result.empty());
}

// Persistent buffers are excluded from `reservedSize`. Shared weights consumed by every
// iteration become PERSISTENT_CANDIDATE and are accounted separately via
// `sharedExternalBuffers`, so `reservedSize` covers only the temporary working area.
TEST_F(MLIR_UndefinedVF, ReservedSizeExcludesPersistentSharedWeights) {
    UndefinedVF scenario;
    constexpr size_t kIters = 3;
    constexpr size_t kOps = 2;
    constexpr vpux::AddressType kBufSize = 512;
    constexpr vpux::AddressType kWeightsSize = 1024;
    constexpr vpux::AddressType kMem = 32768;

    // Baseline: no shared weights. The escaping output emitted by makeChainedBodies is a
    // single shared PERSISTENT_CANDIDATE (produced-never-consumed) => 1 shared external.
    auto baselineRegion = makeVfRegion(makeChainedBodies(kIters, kOps, kBufSize));
    const auto baseline = scenario.getScheduleStrategy(baselineRegion, kMem);
    ASSERT_FALSE(baseline.empty());
    EXPECT_EQ(baseline.sharedExternalBuffers.size(), 1u);

    // With shared weights: DATA_IN in each iteration fetches from a single external
    // weight buffer -> classified as PERSISTENT_CANDIDATE and moved to sharedExternalBuffers.
    // Combined with the shared escaping output that yields 2 entries.
    auto region = makeVfRegion(makeChainedBodies(kIters, kOps, kBufSize, kWeightsSize));
    const auto result = scenario.getScheduleStrategy(region, kMem);
    ASSERT_FALSE(result.empty());

    // Shared weights + shared escaping output => 2 externals.
    ASSERT_EQ(result.sharedExternalBuffers.size(), 2u);

    // reservedSize is the temporary working area only. The persistent zone
    // (aligned weight size) must NOT be included.
    const vpux::AddressType alignedWeights = vpux::alignValUp<vpux::AddressType>(kWeightsSize, /*alignment=*/64);
    EXPECT_LE(result.reservedSize, kMem - alignedWeights);
    EXPECT_GT(result.reservedSize, 0u);

    // Adding a DATA_IN op per compute step increases the temporary footprint
    // (per-iteration weight tile), so the temporary reservedSize must not drop
    // below the baseline that had no DATA_IN op at all.
    EXPECT_GE(result.reservedSize, baseline.reservedSize);
}

// When the classifier flags a region as unsupported (partial-external buffer pattern),
// UndefinedVF must return an empty schedule instead of attempting to allocate.
TEST_F(MLIR_UndefinedVF, UnsupportedRegionReturnsEmptySchedule) {
    // Per-iteration distinct region outputs with no in-loop consumer: iteration 0's
    // COMPUTE produces `out0`, iteration 1's COMPUTE produces `out1`. Both leave the loop
    // without a DATA_OUT sink, so the template buffer `out0` has producerIterCount=1 and
    // consumerIterCount=0 -> partial-external-output -> classifierSupportedRegion=false.
    auto in = createBuffer(/*size=*/64);  // shared loop-invariant external input
    auto out0 = createBuffer(/*size=*/64);
    auto out1 = createBuffer(/*size=*/64);
    LoopBody body0{createComputeOp(0, {in}, {out0})};
    LoopBody body1{createComputeOp(1, {in}, {out1})};
    auto region = makeVfRegion({std::move(body0), std::move(body1)});

    UndefinedVF scenario;
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/8192);

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(result.schedule.empty());
    EXPECT_TRUE(result.sharedExternalBuffers.empty());
    EXPECT_EQ(result.reservedSize, 0u);
}
