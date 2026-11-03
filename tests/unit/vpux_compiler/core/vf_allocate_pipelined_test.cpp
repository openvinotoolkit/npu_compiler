//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/vf_allocate_linear.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/undefined_vf.hpp"

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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Run cmd: npuUnitTests --gtest_filter="MLIR_VFAllocatePipelined.*"

using namespace vpux;
using namespace vpux::VPU;

class MLIR_VFAllocatePipelined : public testing::Test {
protected:
    mlir::MLIRContext _ctx;
    mlir::OpBuilder _builder{&_ctx};
    std::unique_ptr<mlir::Block> _block;

    MLIR_VFAllocatePipelined() {
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
                               const SmallVector<mlir::Value>& outBufs,
                               config::ExecutorKind execKind = config::ExecutorKind::DPU) {
        VPURT::TaskQueueType qt{execKind, 0};
        return OpAllocationInfo(opIdx, qt, inBufs, outBufs, AllocationType::COMPUTE);
    }

    OpAllocationInfo dataInOp(size_t opIdx, const SmallVector<mlir::Value>& inBufs,
                              const SmallVector<mlir::Value>& outBufs, uint32_t dmaQueueId) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DMA_NN, dmaQueueId};
        return OpAllocationInfo(opIdx, qt, inBufs, outBufs, AllocationType::DATA_IN);
    }

    OpAllocationInfo dataOutOp(size_t opIdx, const SmallVector<mlir::Value>& inBufs,
                               const SmallVector<mlir::Value>& outBufs, uint32_t dmaQueueId) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DMA_NN, dmaQueueId};
        return OpAllocationInfo(opIdx, qt, inBufs, outBufs, AllocationType::DATA_OUT);
    }

    ComputeRegion makeVfRegion(std::vector<LoopBody> bodies) {
        auto loop = std::make_unique<SchedulingLoop>();
        loop->type = LoopType::VF;
        loop->loopBodies = std::move(bodies);
        return ComputeRegion(std::move(loop));
    }

    // Builds a 2-iteration PredefinedSchedule: iteration 0 (curr) holds `anchorCount`
    // independent computes of `anchorKind`, iteration 1 (next) holds `leadCount`
    // independent computes of `leadKind` (no inter-dependencies, so every one of them
    // is eligible as its own hoist segment). Op indices are assigned in creation
    // order: anchors first [0, anchorCount), then lead ops [anchorCount, anchorCount +
    // leadCount), which keeps id arithmetic simple for callers that compute the
    // expected post-hoist order (see expectedOverflowHoist below).
    PredefinedSchedule makeOverflowSchedule(config::ExecutorKind anchorKind, size_t anchorCount,
                                            config::ExecutorKind leadKind, size_t leadCount) {
        auto makeEntry = [](OpAllocationInfo allocInfo) {
            return ComputeExplicitSchedule(allocInfo, /*deallocations=*/{}, /*allocations=*/{});
        };

        size_t opIdx = 0;
        IterationSchedule currStep;
        for (size_t i = 0; i < anchorCount; ++i) {
            currStep.push_back(makeEntry(computeOp(opIdx++, {createBuffer(1024)}, {createBuffer(1024)}, anchorKind)));
        }

        IterationSchedule nextStep;
        for (size_t i = 0; i < leadCount; ++i) {
            nextStep.push_back(makeEntry(computeOp(opIdx++, {createBuffer(1024)}, {createBuffer(1024)}, leadKind)));
        }

        return PredefinedSchedule{std::move(currStep), std::move(nextStep)};
    }

    static SmallVector<size_t> opIds(const IterationSchedule& iteration) {
        SmallVector<size_t> ids;
        ids.reserve(iteration.size());
        for (const auto& entry : iteration) {
            ids.push_back(entry.allocInfo.opIdx);
        }
        return ids;
    }

    struct ExpectedOverflowHoist {
        bool hoisted = false;
        SmallVector<size_t> schedule0Ids;
        SmallVector<size_t> schedule1Ids;
    };

    // Re-derives groupSegmentsByAnchor's expected outcome for a schedule built by
    // makeOverflowSchedule: with anchorCount == 0 nothing can be hoisted; otherwise
    // every lead segment is assigned 1:1 to an anchor while segments outnumber
    // anchors, and once anchors run out the remaining (overflow) segments are all
    // merged into the last anchor's group.
    static ExpectedOverflowHoist expectedOverflowHoist(size_t anchorCount, size_t leadCount) {
        ExpectedOverflowHoist expected;
        if (anchorCount == 0 || leadCount == 0) {
            for (size_t i = 0; i < anchorCount; ++i) {
                expected.schedule0Ids.push_back(i);
            }
            for (size_t i = 0; i < leadCount; ++i) {
                expected.schedule1Ids.push_back(anchorCount + i);
            }
            return expected;
        }

        const size_t numGroups = std::min(leadCount, anchorCount);
        std::vector<SmallVector<size_t>> groups(numGroups);
        if (leadCount <= anchorCount) {
            for (size_t i = 0; i < leadCount; ++i) {
                groups[i] = {i};
            }
        } else {
            const size_t keepIndividual = anchorCount - 1;
            for (size_t i = 0; i < keepIndividual; ++i) {
                groups[i] = {i};
            }
            SmallVector<size_t> merged;
            for (size_t i = keepIndividual; i < leadCount; ++i) {
                merged.push_back(i);
            }
            groups[anchorCount - 1] = std::move(merged);
        }

        for (size_t anchorIdx = 0; anchorIdx < anchorCount; ++anchorIdx) {
            if (anchorIdx < groups.size()) {
                for (const auto segIdx : groups[anchorIdx]) {
                    expected.schedule0Ids.push_back(anchorCount + segIdx);
                }
            }
            expected.schedule0Ids.push_back(anchorIdx);
        }

        expected.hoisted = true;
        return expected;
    }
};

// Parameterized over (anchorCount, leadCount, kind direction): overflow segments
// beyond the anchor count are always merged into the last anchor's group,
// regardless of how many extra lead-kind ops exist or which kind leads.
TEST_F(MLIR_VFAllocatePipelined, PipeliningHoistMergesFullOverflowChain) {
    struct Case {
        size_t anchorCount;
        size_t leadCount;
    };
    const std::vector<Case> cases = {
            {0, 1},  // no opposite-kind anchor: nothing can be hoisted
            {1, 1},  // 1:1, no overflow, 2 Op case
            {1, 3},  // classic case: 3 independent leads, only 1 anchor
            {1, 4},  // 1 anchor, 4 independent leads (e.g. 1 SHV anchor + 4 DPU leads)
            {2, 2},  // 1:1, no overflow
            {2, 5},  // overflow: first anchor keeps its own, rest merge into the last
            {3, 3},  // 1:1, no overflow
    };

    for (const auto& [anchorCount, leadCount] : cases) {
        for (const auto& [anchorKind, leadKind] :
             {std::pair{config::ExecutorKind::SHAVE_ACT, config::ExecutorKind::DPU},
              std::pair{config::ExecutorKind::DPU, config::ExecutorKind::SHAVE_ACT}}) {
            auto schedule = makeOverflowSchedule(anchorKind, anchorCount, leadKind, leadCount);

            const auto expected = expectedOverflowHoist(anchorCount, leadCount);
            // TO visualize schedules use dumpScheduleASCII as shown below
            // dumpScheduleASCII(schedule, Logger::global(), "before pairwise rewrite", regionIdx);

            const bool hoisted = rewriteVFPipelinedScheduleIntoPairwiseSteps(schedule);

            EXPECT_EQ(hoisted, expected.hoisted) << "anchorCount=" << anchorCount << " leadCount=" << leadCount;
            ASSERT_EQ(schedule.size(), 2u);
            EXPECT_EQ(opIds(schedule[0]), expected.schedule0Ids)
                    << "anchorCount=" << anchorCount << " leadCount=" << leadCount;
            EXPECT_EQ(opIds(schedule[1]), expected.schedule1Ids)
                    << "anchorCount=" << anchorCount << " leadCount=" << leadCount;

            // TO visualize schedules use dumpScheduleASCII as shown below
            // dumpScheduleASCII(schedule, Logger::global(), "after pairwise rewrite", regionIdx);
        }
    }
}

TEST_F(MLIR_VFAllocatePipelined, MatchesUndefinedVFPipeliningToLinearHalfMemoryModel) {
    auto sharedIn = createBuffer(1024);
    auto sharedOut = createBuffer(1024);
    auto t00 = createBuffer(1024);
    auto t10 = createBuffer(1024);

    LoopBody body0{computeOp(0, {sharedIn}, {t00}, config::ExecutorKind::DPU),
                   computeOp(1, {t00}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};
    LoopBody body1{computeOp(2, {sharedIn}, {t10}, config::ExecutorKind::DPU),
                   computeOp(3, {t10}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};
    auto region = makeVfRegion({body0, body1});

    constexpr AddressType memoryLimit = 16384;

    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);
    const AddressType oneBankBudget = (memoryLimit + classifier.persistentCandidateTotalBytes) / 2;

    const VfSchedStrategyDescriptor params{VfSchedStrategyDescriptor::IntraIterFetch::ENABLED,
                                           VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::DISABLED};

    const auto linearOneBank = VfAllocateLinear(region, classifier, oneBankBudget, params).performAllocation();
    ASSERT_TRUE(linearOneBank.feasible);

    UndefinedVF scenario;
    const auto scheduled = scenario.getScheduleStrategy(region, memoryLimit);
    ASSERT_FALSE(scheduled.empty());
    ASSERT_EQ(scheduled.schedule.size(), 2u);

    ASSERT_GE(linearOneBank.peakUsedBytes, linearOneBank.persistentReservedBytes);
    const AddressType linearVolatile = linearOneBank.peakUsedBytes - linearOneBank.persistentReservedBytes;
    const AddressType expectedReserved = 2 * linearVolatile;

    EXPECT_EQ(scheduled.reservedSize, expectedReserved);
    EXPECT_EQ(scheduled.sharedExternalBuffers, linearOneBank.sharedExternalBuffers);
}

// If VF_PIPELINING is infeasible for a recipe, strategy search should continue
// and pick the first feasible fallback.
TEST_F(MLIR_VFAllocatePipelined, RunStrategySearchFallsBackWhenPipeliningIsInfeasible) {
    bool seenPipelining = false;
    bool seenFallback = false;

    UndefinedVF::AllocatorFn allocator = [&](const VfStrategyRecipe& recipe) {
        VfAllocateResult r;
        if (recipe.id == VPU::VFScenario::VF_PIPELINING) {
            seenPipelining = true;
            r.feasible = false;
            r.peakUsedBytes = 9999;
            return r;
        }

        seenFallback = true;
        r.feasible = true;
        r.peakUsedBytes = 1024;
        return r;
    };

    UndefinedVF scenario;
    const auto outcome = scenario.runStrategySearch(allocator);

    EXPECT_TRUE(seenPipelining);
    EXPECT_TRUE(seenFallback);
    ASSERT_TRUE(outcome.selectedVFStrategy.has_value());
    EXPECT_NE(outcome.selectedVFStrategy.value(), VPU::VFScenario::VF_PIPELINING);
    EXPECT_TRUE(outcome.result.feasible);
}

// When VF_PIPELINING is selected, odd iterations use upper-bank offsets while
// even iterations keep base template addresses.
TEST_F(MLIR_VFAllocatePipelined, PipeliningAlternatesEvenOddBanks) {
    UndefinedVF scenario;

    auto sharedIn = createBuffer(1024);
    auto sharedOut = createBuffer(1024);

    auto mid0 = createBuffer(2048);
    auto mid1 = createBuffer(2048);
    auto mid2 = createBuffer(2048);

    LoopBody body0{computeOp(0, {sharedIn}, {mid0}, config::ExecutorKind::DPU),
                   computeOp(1, {mid0}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};
    LoopBody body1{computeOp(2, {sharedIn}, {mid1}, config::ExecutorKind::DPU),
                   computeOp(3, {mid1}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};
    LoopBody body2{computeOp(4, {sharedIn}, {mid2}, config::ExecutorKind::DPU),
                   computeOp(5, {mid2}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};

    auto region = makeVfRegion({std::move(body0), std::move(body1), std::move(body2)});
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/65536);

    ASSERT_EQ(result.schedule.size(), 3u);
    ASSERT_GT(result.reservedSize, 0u);
    ASSERT_EQ(result.reservedSize % 2, 0u);

    const AddressType bankSize = result.reservedSize / 2;

    auto findAllocAddrForBuffer = [](const auto& schedule, mlir::Value target) -> std::optional<AddressType> {
        for (const auto& iteration : schedule) {
            for (const auto& entry : iteration) {
                for (const auto& [buf, addr] : entry.allocations) {
                    if (buf == target) {
                        return addr;
                    }
                }
            }
        }
        return std::nullopt;
    };

    const auto evenAddr = findAllocAddrForBuffer(result.schedule, mid0);
    const auto oddAddr = findAllocAddrForBuffer(result.schedule, mid1);
    const auto even2Addr = findAllocAddrForBuffer(result.schedule, mid2);
    ASSERT_TRUE(evenAddr.has_value());
    ASSERT_TRUE(oddAddr.has_value());
    ASSERT_TRUE(even2Addr.has_value());

    EXPECT_EQ(oddAddr.value(), evenAddr.value() + bankSize);
    EXPECT_EQ(even2Addr.value(), evenAddr.value());
}

// Explicit DPU->SHV VF graph example to document the pipelined memory model:
// one-bank allocation must fit in CMX/2, and full pipelined footprint is
// persistent + 2 * bankSize.
TEST_F(MLIR_VFAllocatePipelined, DpuShvGraphDoubleBufferingBudgetIsExplicit) {
    auto sharedIn = createBuffer(1024);
    auto sharedOut = createBuffer(1024);

    // Iteration 0
    auto dpuOut0 = createBuffer(3072);

    // Iteration 1
    auto dpuOut1 = createBuffer(3072);

    LoopBody body0{computeOp(0, {sharedIn}, {dpuOut0}, config::ExecutorKind::DPU),
                   computeOp(1, {dpuOut0}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};
    LoopBody body1{computeOp(2, {sharedIn}, {dpuOut1}, config::ExecutorKind::DPU),
                   computeOp(3, {dpuOut1}, {sharedOut}, config::ExecutorKind::SHAVE_ACT)};
    auto region = makeVfRegion({std::move(body0), std::move(body1)});

    constexpr AddressType memoryLimit = 16384;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);
    const AddressType oneBankBudget = (memoryLimit + classifier.persistentCandidateTotalBytes) / 2;

    const VfSchedStrategyDescriptor params{VfSchedStrategyDescriptor::IntraIterFetch::ENABLED,
                                           VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::DISABLED};

    const auto linearOneBank = VfAllocateLinear(region, classifier, oneBankBudget, params).performAllocation();
    ASSERT_TRUE(linearOneBank.feasible);

    UndefinedVF scenario;
    const auto scheduled = scenario.getScheduleStrategy(region, memoryLimit);
    ASSERT_FALSE(scheduled.empty());
    ASSERT_EQ(scheduled.schedule.size(), 2u);

    ASSERT_GE(linearOneBank.peakUsedBytes, linearOneBank.persistentReservedBytes);
    const AddressType oneBankVolatile = linearOneBank.peakUsedBytes - linearOneBank.persistentReservedBytes;
    const AddressType bankSize = oneBankVolatile;

    EXPECT_EQ(scheduled.reservedSize, 2 * bankSize);
    EXPECT_EQ(scheduled.sharedExternalBuffers, linearOneBank.sharedExternalBuffers);
    EXPECT_LE(scheduled.reservedSize + linearOneBank.persistentReservedBytes, memoryLimit);

    auto findAllocAddrForBuffer = [](const auto& schedule, mlir::Value target) -> std::optional<AddressType> {
        for (const auto& iteration : schedule) {
            for (const auto& entry : iteration) {
                for (const auto& [buf, addr] : entry.allocations) {
                    if (buf == target) {
                        return addr;
                    }
                }
            }
        }
        return std::nullopt;
    };

    const auto evenAddr = findAllocAddrForBuffer(scheduled.schedule, dpuOut0);
    const auto oddAddr = findAllocAddrForBuffer(scheduled.schedule, dpuOut1);
    ASSERT_TRUE(evenAddr.has_value());
    ASSERT_TRUE(oddAddr.has_value());
    const auto size = checked_cast<AddressType>(getTotalSize(dpuOut0).count());

    EXPECT_EQ(oddAddr.value(), evenAddr.value() + bankSize);
    EXPECT_LT(evenAddr.value(), oddAddr.value());

    EXPECT_LE(evenAddr.value() + size, bankSize);
    EXPECT_GE(oddAddr.value(), bankSize);
    EXPECT_LE(oddAddr.value() + size, scheduled.reservedSize);
}

// Documents the VF_PIPELINING memory budget formula:
//   persistentCandidateBytes = persistentCandidateTotalBytes (single copy across banks)
//   largestOpLocalBytes      = peakBaselineBytes - persistentCandidateBytes
//   oneBankBudget            = (memoryLimit + persistentCandidateBytes) / 2
// Feasibility requires 2*largestOpLocalBytes + persistentCandidateBytes <= memoryLimit.
TEST_F(MLIR_VFAllocatePipelined, PipeliningMemoryBudgetFormulaMatchesWorkedExample) {
    auto sharedIn = createBuffer(1024);        // persistent: reused by every iteration, feeds DPU directly
    auto dpuWeightSource = createBuffer(256);  // persistent: external source, fetched by DMA_IN for DPU
    auto dmaSource = createBuffer(256);        // persistent: external source, fetched by DMA_IN for SHV
    auto tapSink = createBuffer(512);          // persistent: external sink, drained by DMA_OUT after DPU
    auto sharedOut = createBuffer(1024);       // persistent: external sink, drained by DMA_OUT after SHV

    auto dpuOut0 = createBuffer(3072);  // temporary: fresh per-iteration DPU output
    auto dpuOut1 = createBuffer(3072);
    auto dpuWeightTile0 = createBuffer(256);  // temporary: fresh per-iteration DMA-fetched tile for DPU
    auto dpuWeightTile1 = createBuffer(256);
    auto extraWeight0 = createBuffer(256);  // temporary: fresh per-iteration DMA-fetched tile for SHV
    auto extraWeight1 = createBuffer(256);
    auto shvOut0 = createBuffer(1024);  // temporary: fresh per-iteration SHV output, drained via DMA_OUT
    auto shvOut1 = createBuffer(1024);

    LoopBody body0{dataInOp(0, {dpuWeightSource}, {dpuWeightTile0}, 0),
                   computeOp(1, {sharedIn, dpuWeightTile0}, {dpuOut0}, config::ExecutorKind::DPU),
                   dataOutOp(2, {dpuOut0}, {tapSink}, 0),
                   dataInOp(3, {dmaSource}, {extraWeight0}, 0),
                   computeOp(4, {dpuOut0, extraWeight0}, {shvOut0}, config::ExecutorKind::SHAVE_ACT),
                   dataOutOp(5, {shvOut0}, {sharedOut}, 0)};
    LoopBody body1{dataInOp(6, {dpuWeightSource}, {dpuWeightTile1}, 0),
                   computeOp(7, {sharedIn, dpuWeightTile1}, {dpuOut1}, config::ExecutorKind::DPU),
                   dataOutOp(8, {dpuOut1}, {tapSink}, 0),
                   dataInOp(9, {dmaSource}, {extraWeight1}, 0),
                   computeOp(10, {dpuOut1, extraWeight1}, {shvOut1}, config::ExecutorKind::SHAVE_ACT),
                   dataOutOp(11, {shvOut1}, {sharedOut}, 0)};
    auto region = makeVfRegion({std::move(body0), std::move(body1)});

    constexpr AddressType memoryLimit = 16384;
    const auto classifier = classifyVFRegion(region, memoryLimit);
    ASSERT_TRUE(classifier.iterationIdentityHolds);

    // persistentCandidateBytes: sharedIn(1024) + dpuWeightSource(256) + dmaSource(256) + tapSink(512)
    // + sharedOut(1024) = 3072. Every one of these has no in-loop producer (externally fetched) or
    // no in-loop consumer (an escaping sink) - the same "persistent, single copy" pattern,
    // regardless of whether the op touching it is COMPUTE, DMA_IN, or DMA_OUT.
    const AddressType persistentCandidateBytes = classifier.persistentCandidateTotalBytes;
    EXPECT_EQ(persistentCandidateBytes, 3072u);

    // loopBodyNoSpillPeakBytes = max COMPUTE op footprint:
    //   DPU: sharedIn(1024) + dpuWeightTile(256) + dpuOut(3072) = 4352
    //   SHV: dpuOut(3072) + extraWeight(256) + shvOut(1024)     = 4352  (tie, DPU found first wins)
    // The winning op (DPU) contributes its own PERSISTENT_CANDIDATE input (sharedIn=1024) as
    // largestComputeOpWeights, so peakBaselineBytes = 4352 + (3072 - 1024) = 6400.
    EXPECT_EQ(classifier.peakBaselineBytes, 6400u);

    // largestOpLocalBytes = peakBaselineBytes - persistentCandidateBytes = 6400 - 3072 = 3328
    // (DPU's footprint minus its own weight)
    const AddressType largestOpLocalBytes = saturatingSub(classifier.peakBaselineBytes, persistentCandidateBytes);
    EXPECT_EQ(largestOpLocalBytes, 3328u);

    // Feasibility gate: 2*largestOpLocalBytes + persistentCandidateBytes <= memoryLimit
    EXPECT_LE(2 * largestOpLocalBytes + persistentCandidateBytes, memoryLimit);

    // oneBankBudget = (memoryLimit + persistentCandidateBytes) / 2 = (16384 + 3072) / 2 = 9728
    const AddressType oneBankBudget = (memoryLimit + persistentCandidateBytes) / 2;
    EXPECT_EQ(oneBankBudget, 9728u);

    // The real allocator, run with oneBankBudget (not a blind memoryLimit/2 split), must produce
    // the same one-bank schedule UndefinedVF selects for VF_PIPELINING.
    const VfSchedStrategyDescriptor params{VfSchedStrategyDescriptor::IntraIterFetch::ENABLED,
                                           VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::DISABLED};
    const auto oneBankResult = VfAllocateLinear(region, classifier, oneBankBudget, params).performAllocation();
    ASSERT_TRUE(oneBankResult.feasible);
    ASSERT_GE(oneBankResult.peakUsedBytes, oneBankResult.persistentReservedBytes);
    const AddressType bankSize = oneBankResult.peakUsedBytes - oneBankResult.persistentReservedBytes;

    UndefinedVF scenario;
    const auto scheduled = scenario.getScheduleStrategy(region, memoryLimit);
    ASSERT_FALSE(scheduled.empty());
    EXPECT_EQ(scheduled.reservedSize, 2 * bankSize);
    EXPECT_LE(scheduled.reservedSize + oneBankResult.persistentReservedBytes, memoryLimit);
}

TEST_F(MLIR_VFAllocatePipelined, PipeliningSimpleAlternatingBodiesKeepPerIterationReplay) {
    UndefinedVF scenario;

    auto verifySimpleCase = [&](config::ExecutorKind firstKind, config::ExecutorKind secondKind) {
        auto sharedIn = createBuffer(1024);
        auto sharedOut = createBuffer(1024);
        auto dmaSource0 = createBuffer(1024);
        auto dmaSource1 = createBuffer(1024);
        auto stage0Sink = createBuffer(1024);
        auto stage1Sink = createBuffer(1024);
        auto makeBody = [&](size_t baseOpIdx, mlir::Value midBuf) {
            auto firstDma0 = createBuffer(256);
            auto firstDma1 = createBuffer(256);
            auto secondDma0 = createBuffer(256);
            auto secondDma1 = createBuffer(256);
            return LoopBody{dataInOp(baseOpIdx + 0, {dmaSource0}, {firstDma0}, 0),
                            dataInOp(baseOpIdx + 1, {dmaSource1}, {firstDma1}, 1),
                            computeOp(baseOpIdx + 2, {sharedIn, firstDma0, firstDma1}, {midBuf}, firstKind),
                            dataOutOp(baseOpIdx + 3, {midBuf}, {stage0Sink}, 0),
                            dataInOp(baseOpIdx + 4, {dmaSource0}, {secondDma0}, 0),
                            dataInOp(baseOpIdx + 5, {dmaSource1}, {secondDma1}, 1),
                            computeOp(baseOpIdx + 6, {midBuf, secondDma0, secondDma1}, {sharedOut}, secondKind),
                            dataOutOp(baseOpIdx + 7, {sharedOut}, {stage1Sink}, 1)};
        };

        auto mid0 = createBuffer(1024);
        auto mid1 = createBuffer(1024);
        auto mid2 = createBuffer(1024);
        auto mid3 = createBuffer(1024);

        auto region = makeVfRegion({makeBody(0, mid0), makeBody(8, mid1), makeBody(16, mid2), makeBody(24, mid3)});

        const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/65536);
        ASSERT_EQ(result.schedule.size(), 4u);
        EXPECT_TRUE(result.enableIterationOverlap);

        auto opIds = [](const IterationSchedule& iteration) {
            SmallVector<size_t> ids;
            ids.reserve(iteration.size());
            for (const auto& entry : iteration) {
                ids.push_back(entry.allocInfo.opIdx);
            }
            return ids;
        };

        EXPECT_EQ(opIds(result.schedule[0]), (SmallVector<size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
        EXPECT_EQ(opIds(result.schedule[1]), (SmallVector<size_t>{8, 9, 10, 11, 12, 13, 14, 15}));
        EXPECT_EQ(opIds(result.schedule[2]), (SmallVector<size_t>{16, 17, 18, 19, 20, 21, 22, 23}));
        EXPECT_EQ(opIds(result.schedule[3]), (SmallVector<size_t>{24, 25, 26, 27, 28, 29, 30, 31}));
    };

    verifySimpleCase(config::ExecutorKind::DPU, config::ExecutorKind::SHAVE_ACT);
    verifySimpleCase(config::ExecutorKind::SHAVE_ACT, config::ExecutorKind::DPU);
}

TEST_F(MLIR_VFAllocatePipelined, PipeliningComplexAlternatingBodiesHoistWavefrontStages) {
    UndefinedVF scenario;

    auto verifyComplexCase = [&](config::ExecutorKind firstKind, config::ExecutorKind secondKind) {
        auto sharedIn = createBuffer(1024);
        auto sharedOut = createBuffer(1024);
        auto dmaSource0 = createBuffer(1024);
        auto dmaSource1 = createBuffer(1024);
        auto stage0Sink = createBuffer(1024);
        auto stage1Sink = createBuffer(1024);
        auto stage2Sink = createBuffer(1024);
        auto makeBody = [&](size_t baseOpIdx, mlir::Value stage0Buf, mlir::Value stage1Buf) {
            auto stage0Dma0 = createBuffer(256);
            auto stage0Dma1 = createBuffer(256);
            auto stage1Dma0 = createBuffer(256);
            auto stage1Dma1 = createBuffer(256);
            auto stage2Dma0 = createBuffer(256);
            auto stage2Dma1 = createBuffer(256);
            return LoopBody{dataInOp(baseOpIdx + 0, {dmaSource0}, {stage0Dma0}, 0),
                            dataInOp(baseOpIdx + 1, {dmaSource1}, {stage0Dma1}, 1),
                            computeOp(baseOpIdx + 2, {sharedIn, stage0Dma0, stage0Dma1}, {stage0Buf}, firstKind),
                            dataOutOp(baseOpIdx + 3, {stage0Buf}, {stage0Sink}, 0),
                            dataInOp(baseOpIdx + 4, {dmaSource0}, {stage1Dma0}, 0),
                            dataInOp(baseOpIdx + 5, {dmaSource1}, {stage1Dma1}, 1),
                            computeOp(baseOpIdx + 6, {stage0Buf, stage1Dma0, stage1Dma1}, {stage1Buf}, secondKind),
                            dataOutOp(baseOpIdx + 7, {stage1Buf}, {stage1Sink}, 1),
                            dataInOp(baseOpIdx + 8, {dmaSource0}, {stage2Dma0}, 0),
                            dataInOp(baseOpIdx + 9, {dmaSource1}, {stage2Dma1}, 1),
                            computeOp(baseOpIdx + 10, {stage1Buf, stage2Dma0, stage2Dma1}, {sharedOut}, firstKind),
                            dataOutOp(baseOpIdx + 11, {sharedOut}, {stage2Sink}, 0)};
        };

        auto buf0 = createBuffer(1024);
        auto buf1 = createBuffer(1024);
        auto buf2 = createBuffer(1024);
        auto buf3 = createBuffer(1024);
        auto buf4 = createBuffer(1024);
        auto buf5 = createBuffer(1024);
        auto buf6 = createBuffer(1024);
        auto buf7 = createBuffer(1024);

        auto region = makeVfRegion({makeBody(0, buf0, buf1), makeBody(12, buf2, buf3), makeBody(24, buf4, buf5),
                                    makeBody(36, buf6, buf7)});

        const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/65536);
        ASSERT_EQ(result.schedule.size(), 4u);
        EXPECT_FALSE(result.enableIterationOverlap);

        auto opIds = [](const IterationSchedule& iteration) {
            SmallVector<size_t> ids;
            ids.reserve(iteration.size());
            for (const auto& entry : iteration) {
                ids.push_back(entry.allocInfo.opIdx);
            }
            return ids;
        };

        EXPECT_EQ(opIds(result.schedule[0]), (SmallVector<size_t>{0, 1, 2, 3, 12, 13, 14, 4, 5, 6, 7, 8, 9, 10, 11}));
        EXPECT_EQ(opIds(result.schedule[1]), (SmallVector<size_t>{15, 24, 25, 26, 16, 17, 18, 19, 20, 21, 22, 23}));
        EXPECT_EQ(opIds(result.schedule[2]), (SmallVector<size_t>{27, 36, 37, 38, 28, 29, 30, 31, 32, 33, 34, 35}));
        EXPECT_EQ(opIds(result.schedule[3]), (SmallVector<size_t>{39, 40, 41, 42, 43, 44, 45, 46, 47}));
    };

    verifyComplexCase(config::ExecutorKind::DPU, config::ExecutorKind::SHAVE_ACT);
    verifyComplexCase(config::ExecutorKind::SHAVE_ACT, config::ExecutorKind::DPU);
}

// Pure DPU VF graph must skip VF_PIPELINING and keep identical local template
// addresses across iterations.
TEST_F(MLIR_VFAllocatePipelined, DpuOnlyGraphDoesNotAlternateBanks) {
    UndefinedVF scenario;
    auto sharedIn = createBuffer(1024);
    auto sharedOut = createBuffer(1024);

    auto mid0 = createBuffer(1024);
    auto mid1 = createBuffer(1024);
    auto mid2 = createBuffer(1024);

    LoopBody body0{computeOp(0, {sharedIn}, {mid0}, config::ExecutorKind::DPU),
                   computeOp(1, {mid0}, {sharedOut}, config::ExecutorKind::DPU)};
    LoopBody body1{computeOp(2, {sharedIn}, {mid1}, config::ExecutorKind::DPU),
                   computeOp(3, {mid1}, {sharedOut}, config::ExecutorKind::DPU)};
    LoopBody body2{computeOp(4, {sharedIn}, {mid2}, config::ExecutorKind::DPU),
                   computeOp(5, {mid2}, {sharedOut}, config::ExecutorKind::DPU)};

    auto region = makeVfRegion({std::move(body0), std::move(body1), std::move(body2)});
    const auto result = scenario.getScheduleStrategy(region, /*memorySize=*/16384);

    ASSERT_EQ(result.schedule.size(), 3u);
    for (size_t opPos = 0; opPos < result.schedule[0].size(); ++opPos) {
        const auto& iter0 = result.schedule[0][opPos].allocations;
        const auto& iter1 = result.schedule[1][opPos].allocations;
        const auto& iter2 = result.schedule[2][opPos].allocations;
        ASSERT_EQ(iter0.size(), iter1.size());
        ASSERT_EQ(iter0.size(), iter2.size());
        for (size_t i = 0; i < iter0.size(); ++i) {
            EXPECT_EQ(iter0[i].second, iter1[i].second);
            EXPECT_EQ(iter0[i].second, iter2[i].second);
        }
    }
}
