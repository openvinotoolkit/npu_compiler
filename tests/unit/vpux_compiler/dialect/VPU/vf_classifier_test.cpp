//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/classifier_vf.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_ClassifierVF.*"

using namespace vpux;

class MLIR_ClassifierVF : public testing::Test {
protected:
    mlir::MLIRContext _ctx;
    mlir::OpBuilder _builder{&_ctx};
    std::unique_ptr<mlir::Block> _block;

    MLIR_ClassifierVF() {
        auto registry = vpux::createDialectRegistry();
        _ctx.appendDialectRegistry(registry);
        _ctx.loadAllAvailableDialects();

        _block = std::make_unique<mlir::Block>();
        _builder.setInsertionPointToStart(_block.get());
    }

    mlir::Value createBuffer(vpux::AddressType rawSize, vpux::AddressType rawAlign = 64) {
        const auto shape = SmallVector<int64_t>{static_cast<int64_t>(rawSize)};
        auto type = mlir::MemRefType::get(shape, _builder.getIntegerType(8));
        auto alloc = _builder.create<mlir::memref::AllocOp>(_builder.getUnknownLoc(), type);
        if (rawAlign != 0) {
            alloc.setAlignment(rawAlign);
        }
        return alloc;
    }

    OpAllocationInfo createOp(size_t opIdx, AllocationType allocationType, const SmallVector<mlir::Value>& inBuffers,
                              const SmallVector<mlir::Value>& outBuffers) {
        const auto executor = (allocationType == AllocationType::DATA_IN || allocationType == AllocationType::DATA_OUT)
                                      ? config::ExecutorKind::DMA_NN
                                      : config::ExecutorKind::DPU;
        VPURT::TaskQueueType queueType{executor, 0};
        return OpAllocationInfo(opIdx, queueType, inBuffers, outBuffers, allocationType);
    }

    const BufferRecord& getRecord(const ClassifierVFResult& result, mlir::Value value) const {
        const auto it = result.valueToRecord.find(value);
        if (it == result.valueToRecord.end()) {
            ADD_FAILURE() << "Missing BufferRecord for queried value";
            static const BufferRecord kMissingRecord{};
            return kMissingRecord;
        }
        return result.buffers[it->second];
    }
};

// Buffer is not produced inside the loop, but consumed in every iteration.
TEST_F(MLIR_ClassifierVF, ClassifiesLoopInvariantAsPersistentExternal) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto loopInvariantInput = createBuffer(1024);
    auto prefetchableTmp0 = createBuffer(256);
    auto mid0 = createBuffer(128);
    auto out0 = createBuffer(64);
    auto prefetchableTmp1 = createBuffer(256);
    auto mid1 = createBuffer(128);
    auto out1 = createBuffer(64);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {loopInvariantInput}, {prefetchableTmp0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {prefetchableTmp0}, {mid0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {mid0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {loopInvariantInput}, {prefetchableTmp1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {prefetchableTmp1}, {mid1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {mid1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/8192, Logger::global());

    ASSERT_TRUE(result.iterationIdentityHolds);

    const auto& rec = getRecord(result, loopInvariantInput);
    EXPECT_EQ(rec.category, BufferCategory::PERSISTENT_CANDIDATE);
    EXPECT_TRUE(rec.isPrefetchableWeight);
}
// Buffer produced by COMPUTE is not treated as prefetchable weight
TEST_F(MLIR_ClassifierVF, ComputeProducedTemporaryIsNotPrefetchable) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto loopInvariantInput = createBuffer(1024);
    auto prefetchableTmp0 = createBuffer(256);
    auto temporary0 = createBuffer(256);
    auto sink0 = createBuffer(256);
    auto prefetchableTmp1 = createBuffer(256);
    auto temporary1 = createBuffer(256);
    auto sink1 = createBuffer(256);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {loopInvariantInput}, {prefetchableTmp0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {prefetchableTmp0}, {temporary0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {temporary0}, {sink0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {sink0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {loopInvariantInput}, {prefetchableTmp1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {prefetchableTmp1}, {temporary1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {temporary1}, {sink1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {sink1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/8192, Logger::global());

    const auto& rec = getRecord(result, temporary0);
    EXPECT_EQ(rec.category, BufferCategory::TEMPORARY);
    EXPECT_FALSE(rec.isPrefetchableWeight);
}

TEST_F(MLIR_ClassifierVF, IterationIdentityFalseWhenDifferentBufferCountAcrossIterations) {
    for (bool mismatchInBuffers : {true, false}) {
        auto schedulingLoop = std::make_unique<SchedulingLoop>();
        schedulingLoop->type = LoopType::VF;

        auto in0a = createBuffer(256);
        auto in0b = createBuffer(128);
        auto out0a = createBuffer(256);
        auto out0b = createBuffer(128);
        auto tail0 = createBuffer(64);

        auto regionIn0 = createBuffer(64);
        auto regionIn1 = createBuffer(64);
        auto dmaIn0 = createBuffer(64);
        auto dmaIn1 = createBuffer(64);
        auto in1 = createBuffer(256);
        auto out1 = createBuffer(256);
        auto tail1 = createBuffer(64);

        LoopBody iteration0;
        iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {regionIn0}, {dmaIn0}));
        if (mismatchInBuffers) {
            iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {in0a, in0b}, {out0a}));
        } else {
            iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {in0a}, {out0a, out0b}));
        }
        iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {out0a}, {tail0}));
        iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {tail0}, {}));

        LoopBody iteration1;
        iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {regionIn1}, {dmaIn1}));
        iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {in1}, {out1}));
        iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {out1}, {tail1}));
        iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {tail1}, {}));

        schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
        ComputeRegion region(std::move(schedulingLoop));

        const auto result = classifyVFRegion(region, /*memoryLimit=*/8192, Logger::global());
        EXPECT_FALSE(result.iterationIdentityHolds);
    }
}

TEST_F(MLIR_ClassifierVF, IterationIdentityFalseWhenEqualityTopologyDiffersAcrossIterations) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto regionIn0 = createBuffer(64);
    auto dmaIn0 = createBuffer(64);
    auto otherIn0 = createBuffer(64);
    auto out0 = createBuffer(64);
    auto tail0 = createBuffer(64);

    auto regionIn1 = createBuffer(64);
    auto dmaIn1 = createBuffer(64);
    auto otherIn1 = createBuffer(64);
    auto tail1 = createBuffer(64);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {regionIn0}, {dmaIn0}));
    // Non in-place shape: in=[A,B], out=[C] -> in=[0,1], out=[2]
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaIn0, otherIn0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {out0}, {tail0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {tail0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {regionIn1}, {dmaIn1}));
    // In-place shape: in=[X,Y], out=[X] -> in=[0,1], out=[0]
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {dmaIn1, otherIn1}, {dmaIn1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {dmaIn1}, {tail1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {tail1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/8192, Logger::global());
    EXPECT_FALSE(result.iterationIdentityHolds);
}

TEST_F(MLIR_ClassifierVF, BumpRecordSizeAndAlignment) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto in0 = createBuffer(64, /*rawAlign=*/32);
    auto mid0 = createBuffer(64, /*rawAlign=*/32);
    auto out0 = createBuffer(64, /*rawAlign=*/64);

    auto in1 = createBuffer(64, /*rawAlign=*/32);
    auto mid1 = createBuffer(64, /*rawAlign=*/32);
    auto out1 = createBuffer(256, /*rawAlign=*/128);
    auto regionIn0 = createBuffer(32, /*rawAlign=*/32);
    auto regionIn1 = createBuffer(32, /*rawAlign=*/32);
    auto dmaIn0 = createBuffer(32, /*rawAlign=*/32);
    auto dmaIn1 = createBuffer(32, /*rawAlign=*/32);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {regionIn0}, {dmaIn0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaIn0, in0}, {mid0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {mid0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {regionIn1}, {dmaIn1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {dmaIn1, in1}, {mid1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {mid1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/8192, Logger::global());
    const auto& rec = getRecord(result, out0);

    EXPECT_TRUE(result.iterationIdentityHolds);
    EXPECT_EQ(rec.size, 256);
    EXPECT_EQ(rec.alignment, 128);
}

TEST_F(MLIR_ClassifierVF, BasicClassification) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    // Shared region input: same SSA is consumed by DATA_IN in every iteration so the
    // classifier can promote it to PERSISTENT_CANDIDATE. Per-iteration distinct region
    // inputs are now rejected as partial-external and yield an unsupported region.
    auto regionIn = createBuffer(40, /*rawAlign=*/1);
    auto w0 = createBuffer(32, /*rawAlign=*/1);
    auto w1 = createBuffer(48, /*rawAlign=*/1);
    auto w2 = createBuffer(64, /*rawAlign=*/1);

    auto dmaIn0 = createBuffer(80, /*rawAlign=*/1);
    auto outA0 = createBuffer(16, /*rawAlign=*/1);
    auto outB0 = createBuffer(96, /*rawAlign=*/1);
    auto regionOut0 = createBuffer(128, /*rawAlign=*/1);

    auto dmaIn1 = createBuffer(80, /*rawAlign=*/1);
    auto outA1 = createBuffer(16, /*rawAlign=*/1);
    auto outB1 = createBuffer(96, /*rawAlign=*/1);
    auto regionOut1 = createBuffer(128, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {regionIn}, {dmaIn0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {w0, dmaIn0}, {outA0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {w1, outA0}, {outB0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::COMPUTE, {w2, outB0}, {regionOut0}));
    iteration0.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_OUT, {regionOut0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::DATA_IN, {regionIn}, {dmaIn1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {w0, dmaIn1}, {outA1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::COMPUTE, {w1, outA1}, {outB1}));
    iteration1.push_back(createOp(/*opIdx=*/8, AllocationType::COMPUTE, {w2, outB1}, {regionOut1}));
    iteration1.push_back(createOp(/*opIdx=*/9, AllocationType::DATA_OUT, {regionOut1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/2048, Logger::global());

    EXPECT_TRUE(result.iterationIdentityHolds);
    EXPECT_TRUE(result.classifierSupportedRegion);
    // Persistent externals: shared region input consumed by DATA_IN in every iter and
    // weights w0/w1/w2 consumed by COMPUTE in every iter.
    EXPECT_EQ(result.persistentCandidateTotalBytes, 40 + 32 + 48 + 64);
    // Largest COMPUTE op is op3: w2 + outB + regionOut = 64 + 96 + 128 = 288.
    EXPECT_EQ(result.loopBodyNoSpillPeakBytes, 288);
    // Subtract only persistent weights used by largest op (w2 = 64).
    EXPECT_EQ(result.peakBaselineBytes, 288 + (40 + 32 + 48 + 64) - 64);
    // Region input/output are taken from DATA_IN/DATA_OUT boundaries only.
    EXPECT_EQ(result.inputBytes, 80);
    EXPECT_EQ(result.outputBytes, 128);
}

TEST_F(MLIR_ClassifierVF, SmallMixedAlignmentPersistentFitUsesWorstPackedOrder) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    // DATA_IN removed: the previous per-iteration distinct region-input SSAs are now
    // rejected by the classifier as partial-external. The persistent-fit ordering under
    // test only exercises PERSISTENT_CANDIDATE weights, so we drop the DATA_IN boundary.
    auto persistentA = createBuffer(8, /*rawAlign=*/1);
    auto persistentB = createBuffer(17, /*rawAlign=*/16);
    auto persistentC = createBuffer(3, /*rawAlign=*/8);
    auto temporary0 = createBuffer(60, /*rawAlign=*/1);
    auto final0 = createBuffer(1, /*rawAlign=*/1);
    auto temporary1 = createBuffer(60, /*rawAlign=*/1);
    auto final1 = createBuffer(1, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(
            createOp(/*opIdx=*/0, AllocationType::COMPUTE, {persistentA, persistentB, persistentC}, {temporary0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {temporary0}, {final0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::DATA_OUT, {final0}, {}));

    LoopBody iteration1;
    iteration1.push_back(
            createOp(/*opIdx=*/3, AllocationType::COMPUTE, {persistentA, persistentB, persistentC}, {temporary1}));
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::COMPUTE, {temporary1}, {final1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::DATA_OUT, {final1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/100, Logger::global());

    const auto persistentAIdx = result.valueToRecord.lookup(persistentA);
    const auto persistentBIdx = result.valueToRecord.lookup(persistentB);
    const auto persistentCIdx = result.valueToRecord.lookup(persistentC);
    const auto temporaryIdx = result.valueToRecord.lookup(temporary0);

    // Since persistentFitInitial is expected to be smaller than the original number
    // of persistent classified buffers (persistentFitOrder size) at least buffer got downgraded
    // to TEMPORARY, which is not supported until E#222690 is done
    EXPECT_FALSE(result.classifierSupportedRegion);

    EXPECT_EQ(result.buffers[temporaryIdx].category, BufferCategory::TEMPORARY);
    EXPECT_EQ(result.temporaryPeakBytes, 61);
    EXPECT_EQ(result.persistentCandidateTotalBytes, 48);
    // loopBodyNoSpillPeakBytes = max COMPUTE op peak (A + B + C + temp0 = 8+32+8+60 = 108)
    EXPECT_EQ(result.loopBodyNoSpillPeakBytes, 108);
    // peakBaselineBytes = loopBodyNoSpillPeakBytes + (persistentCandidateTotalBytes - weights in largest op)
    // = 108 + (48 - 48) = 108
    EXPECT_EQ(result.peakBaselineBytes, 108);

    ASSERT_EQ(result.persistentFitOrder.size(), 3);
    EXPECT_EQ(result.persistentFitOrder[0], persistentAIdx);
    EXPECT_EQ(result.persistentFitOrder[1], persistentBIdx);
    EXPECT_EQ(result.persistentFitOrder[2], persistentCIdx);

    // We can only fit 2 at the best due to the large alignment of persistentB
    ASSERT_EQ(result.persistentFitInitial.size(), 2);
    EXPECT_EQ(result.persistentFitInitial[0], persistentAIdx);
    EXPECT_EQ(result.persistentFitInitial[1], persistentBIdx);
    EXPECT_EQ(result.persistentFitReservedBytes, 33);
    EXPECT_EQ(result.persistentFitBudgetBytes, 39);
}

TEST_F(MLIR_ClassifierVF, PersistentBuffersAllFitInSortedOrder) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    // DATA_IN removed: the previous per-iteration distinct region-input SSAs are now
    // rejected by the classifier as partial-external. The persistent-fit ordering under
    // test only exercises PERSISTENT_CANDIDATE weights, so we drop the DATA_IN boundary.
    auto persistentA = createBuffer(7, /*rawAlign=*/1);
    auto persistentB = createBuffer(17, /*rawAlign=*/16);
    auto persistentC = createBuffer(3, /*rawAlign=*/8);
    auto persistentD = createBuffer(9, /*rawAlign=*/4);
    auto temporary0 = createBuffer(20, /*rawAlign=*/1);
    auto final0 = createBuffer(1, /*rawAlign=*/1);
    auto temporary1 = createBuffer(20, /*rawAlign=*/1);
    auto final1 = createBuffer(1, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::COMPUTE,
                                  {persistentA, persistentB, persistentC, persistentD}, {temporary0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {temporary0}, {final0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::DATA_OUT, {final0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/3, AllocationType::COMPUTE,
                                  {persistentA, persistentB, persistentC, persistentD}, {temporary1}));
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::COMPUTE, {temporary1}, {final1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::DATA_OUT, {final1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/100, Logger::global());

    const auto persistentAIdx = result.valueToRecord.lookup(persistentA);
    const auto persistentBIdx = result.valueToRecord.lookup(persistentB);
    const auto persistentCIdx = result.valueToRecord.lookup(persistentC);
    const auto persistentDIdx = result.valueToRecord.lookup(persistentD);
    const auto temporaryIdx = result.valueToRecord.lookup(temporary0);

    EXPECT_TRUE(result.classifierSupportedRegion);
    EXPECT_EQ(result.buffers[temporaryIdx].category, BufferCategory::TEMPORARY);
    EXPECT_EQ(result.temporaryPeakBytes, 21);
    EXPECT_EQ(result.persistentCandidateTotalBytes, 59);
    // loopBodyNoSpillPeakBytes = max COMPUTE op peak (A + B + C + D + temp = 7+32+8+12+20 = 79)
    EXPECT_EQ(result.loopBodyNoSpillPeakBytes, 79);
    // peakBaselineBytes = loopBodyNoSpillPeakBytes + (persistentCandidateTotalBytes - weights)
    // = 79 + (59 - 59) = 79
    EXPECT_EQ(result.peakBaselineBytes, 79);

    ASSERT_EQ(result.persistentFitOrder.size(), 4);
    EXPECT_EQ(result.persistentFitOrder[0], persistentBIdx);
    EXPECT_EQ(result.persistentFitOrder[1], persistentDIdx);
    EXPECT_EQ(result.persistentFitOrder[2], persistentCIdx);
    EXPECT_EQ(result.persistentFitOrder[3], persistentAIdx);

    ASSERT_EQ(result.persistentFitInitial.size(), 4);
    EXPECT_EQ(result.persistentFitInitial[0], persistentBIdx);
    EXPECT_EQ(result.persistentFitInitial[1], persistentDIdx);
    EXPECT_EQ(result.persistentFitInitial[2], persistentCIdx);
    EXPECT_EQ(result.persistentFitInitial[3], persistentAIdx);
    EXPECT_EQ(result.persistentFitReservedBytes, 59);
    EXPECT_EQ(result.persistentFitBudgetBytes, 79);
}

TEST_F(MLIR_ClassifierVF, HintIsFullPrefetching) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto sharedInput = createBuffer(128, /*rawAlign=*/1);
    auto dmaTmp0 = createBuffer(128, /*rawAlign=*/1);
    auto mid0 = createBuffer(64, /*rawAlign=*/1);
    auto out0 = createBuffer(64, /*rawAlign=*/1);
    auto dmaTmp1 = createBuffer(128, /*rawAlign=*/1);
    auto mid1 = createBuffer(64, /*rawAlign=*/1);
    auto out1 = createBuffer(64, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {sharedInput}, {dmaTmp0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaTmp0}, {mid0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {mid0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {sharedInput}, {dmaTmp1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {dmaTmp1}, {mid1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {mid1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/1024, Logger::global());

    EXPECT_EQ(result.prefetchHint, VFPrefetchHint::FULL_PREFETCHING);
}

TEST_F(MLIR_ClassifierVF, HintIsLastOpPrefetching) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto sharedInput = createBuffer(100, /*rawAlign=*/1);
    auto dmaTmp0 = createBuffer(400, /*rawAlign=*/1);
    auto mid0 = createBuffer(200, /*rawAlign=*/1);
    auto out0 = createBuffer(200, /*rawAlign=*/1);
    auto dmaTmp1 = createBuffer(400, /*rawAlign=*/1);
    auto mid1 = createBuffer(200, /*rawAlign=*/1);
    auto out1 = createBuffer(200, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {sharedInput}, {dmaTmp0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaTmp0}, {mid0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {mid0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {sharedInput}, {dmaTmp1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {dmaTmp1}, {mid1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {mid1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/1024, Logger::global());

    EXPECT_EQ(result.prefetchHint, VFPrefetchHint::LASTOP_PREFETCHING);
}

TEST_F(MLIR_ClassifierVF, HintIsWeightsPrefetching) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto sharedInput = createBuffer(100, /*rawAlign=*/1);
    auto dmaTmp0 = createBuffer(500, /*rawAlign=*/1);
    auto mid0 = createBuffer(300, /*rawAlign=*/1);
    auto out0 = createBuffer(300, /*rawAlign=*/1);
    auto dmaTmp1 = createBuffer(500, /*rawAlign=*/1);
    auto mid1 = createBuffer(300, /*rawAlign=*/1);
    auto out1 = createBuffer(300, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {sharedInput}, {dmaTmp0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaTmp0}, {mid0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {mid0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {sharedInput}, {dmaTmp1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {dmaTmp1}, {mid1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {mid1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/1024, Logger::global());

    EXPECT_EQ(result.prefetchHint, VFPrefetchHint::WEIGHTS_PREFETCHING);
}

TEST_F(MLIR_ClassifierVF, HintIsMinimal) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    auto sharedInput = createBuffer(200, /*rawAlign=*/1);
    auto dmaTmp0 = createBuffer(600, /*rawAlign=*/1);
    auto mid0 = createBuffer(300, /*rawAlign=*/1);
    auto out0 = createBuffer(300, /*rawAlign=*/1);
    auto dmaTmp1 = createBuffer(600, /*rawAlign=*/1);
    auto mid1 = createBuffer(300, /*rawAlign=*/1);
    auto out1 = createBuffer(300, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {sharedInput}, {dmaTmp0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaTmp0}, {mid0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::COMPUTE, {mid0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::DATA_IN, {sharedInput}, {dmaTmp1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::COMPUTE, {dmaTmp1}, {mid1}));
    iteration1.push_back(createOp(/*opIdx=*/6, AllocationType::COMPUTE, {mid1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/7, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/1024, Logger::global());

    EXPECT_EQ(result.prefetchHint, VFPrefetchHint::MINIMAL);
}

// A per-iteration distinct region input (external buffer only consumed by ONE iteration)
// is a partial-external pattern the classifier cannot model soundly. The classifier must
// mark the region as unsupported and return early (no persistent-fit ordering, no memory
// stats), so downstream stages can fall back to a safe path.
TEST_F(MLIR_ClassifierVF, PartialExternalInputMarksUnsupported) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    // regionIn0 is consumed only by iteration 0's DATA_IN. regionIn1 is a separate SSA
    // (not tracked in the template's buffer records), so from the classifier's point of
    // view the template buffer `regionIn0` has consumerIterCount=1 < numIterations=2.
    auto regionIn0 = createBuffer(128, /*rawAlign=*/1);
    auto regionIn1 = createBuffer(128, /*rawAlign=*/1);
    auto dmaIn0 = createBuffer(128, /*rawAlign=*/1);
    auto out0 = createBuffer(64, /*rawAlign=*/1);
    auto dmaIn1 = createBuffer(128, /*rawAlign=*/1);
    auto out1 = createBuffer(64, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::DATA_IN, {regionIn0}, {dmaIn0}));
    iteration0.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {dmaIn0}, {out0}));
    iteration0.push_back(createOp(/*opIdx=*/2, AllocationType::DATA_OUT, {out0}, {}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/3, AllocationType::DATA_IN, {regionIn1}, {dmaIn1}));
    iteration1.push_back(createOp(/*opIdx=*/4, AllocationType::COMPUTE, {dmaIn1}, {out1}));
    iteration1.push_back(createOp(/*opIdx=*/5, AllocationType::DATA_OUT, {out1}, {}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/4096, Logger::global());

    EXPECT_TRUE(result.iterationIdentityHolds);
    EXPECT_FALSE(result.classifierSupportedRegion);
    // Early-return path skips persistent-fit ordering and memory-stats computation.
    EXPECT_TRUE(result.persistentFitOrder.empty());
    EXPECT_TRUE(result.persistentFitInitial.empty());
    EXPECT_EQ(result.persistentCandidateTotalBytes, 0u);
    EXPECT_EQ(result.peakBaselineBytes, 0u);
    EXPECT_EQ(result.loopBodyNoSpillPeakBytes, 0u);
}

// A per-iteration distinct region output (external buffer only produced by ONE iteration,
// with no in-loop consumer) is the mirror of the partial-external-input pattern and must
// likewise be flagged as unsupported.
TEST_F(MLIR_ClassifierVF, PartialExternalOutputMarksUnsupported) {
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::VF;

    // regionOut0 is produced by iteration 0's last COMPUTE and never consumed inside the
    // loop (no DATA_OUT). regionOut1 is a separate SSA (not tracked), so template buffer
    // `regionOut0` has producerIterCount=1 < numIterations=2 and consumerIterCount=0.
    auto in = createBuffer(64, /*rawAlign=*/1);  // shared external input (loop-invariant)
    auto regionOut0 = createBuffer(64, /*rawAlign=*/1);
    auto regionOut1 = createBuffer(64, /*rawAlign=*/1);

    LoopBody iteration0;
    iteration0.push_back(createOp(/*opIdx=*/0, AllocationType::COMPUTE, {in}, {regionOut0}));

    LoopBody iteration1;
    iteration1.push_back(createOp(/*opIdx=*/1, AllocationType::COMPUTE, {in}, {regionOut1}));

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};
    ComputeRegion region(std::move(schedulingLoop));

    const auto result = classifyVFRegion(region, /*memoryLimit=*/4096, Logger::global());

    EXPECT_TRUE(result.iterationIdentityHolds);
    EXPECT_FALSE(result.classifierSupportedRegion);
    EXPECT_TRUE(result.persistentFitOrder.empty());
    EXPECT_EQ(result.persistentCandidateTotalBytes, 0u);
    EXPECT_EQ(result.peakBaselineBytes, 0u);
}
