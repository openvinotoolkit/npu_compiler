//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <deque>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_DMAOUTOFORDEROPTIMIZATION
#define GEN_PASS_DEF_DMAOUTOFORDEROPTIMIZATION
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

//
//  DMAOutOfOrderOptimizationPass
//

class DMAOutOfOrderOptimizationPass final :
        public VPUIP::arch40xx::impl::DMAOutOfOrderOptimizationBase<DMAOutOfOrderOptimizationPass> {
public:
    explicit DMAOutOfOrderOptimizationPass(std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool isMemoryOverlap(VPURT::DeclareBufferOp bufOp1, VPURT::DeclareBufferOp bufOp2);
    std::optional<WorkloadManagementMode> _workloadManagementMode;
};

bool DMAOutOfOrderOptimizationPass::isMemoryOverlap(VPURT::DeclareBufferOp bufOp1, VPURT::DeclareBufferOp bufOp2) {
    // When buffers are located in different memory types, the two buffers are definitely non-overlapping
    if (bufOp1.getSection() != bufOp2.getSection()) {
        return false;
    }

    // When buffers are located on different section indexes (e.g. CMX tiles), the two buffers are definitely
    // non-overlapping
    if (bufOp1.getSectionIndex().has_value() && bufOp2.getSectionIndex().has_value()) {
        auto bufOp1Sections = parseIntArrayAttr<int64_t>(bufOp1.getSectionIndex().value());
        auto bufOp2Sections = parseIntArrayAttr<int64_t>(bufOp2.getSectionIndex().value());
        // For each section in bufOp2 check if is present in bufOp1 section indexes
        // Example to handle:
        //  bufOp1 CMX[0, 1]
        //  bufOp2 CMX[0]

        bool commonSectionIndex = false;
        for (auto bufOp2Section : bufOp2Sections) {
            if (std::find(bufOp1Sections.begin(), bufOp1Sections.end(), bufOp2Section) != bufOp1Sections.end()) {
                // Common element located
                commonSectionIndex = true;
                break;
            }
        }

        if (!commonSectionIndex) {
            return false;
        }
    }

    auto type1 = mlir::cast<vpux::NDTypeInterface>(bufOp1.getBuffer().getType());
    auto type2 = mlir::cast<vpux::NDTypeInterface>(bufOp2.getBuffer().getType());

    // Need to check the offsets of the buffers to determine whether there is an overlap for other possible cases:
    // 1. when both buffers are on DDR
    // 2. when two buffers are both on CMX but one is DUPLICATED on all CMX tiles
    auto start1 = bufOp1.getByteOffset();
    auto end1 = start1 + type1.getTotalAllocSize().count();
    auto start2 = bufOp2.getByteOffset();
    auto end2 = start2 + type2.getTotalAllocSize().count();

    if (end1 <= start2 || end2 <= start1) {
        return false;
    }

    return true;
}

void DMAOutOfOrderOptimizationPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    // Algorithm will check each independent DMA queues identified by port and channel
    // and will store vector of previous DMA tasks which are to be checked for memory overlap
    // against the next task. Based on HW details to make decision for task[N] to clear ORD bit
    // Task[N-1] and Task[N-2] need to be checked for overlap. Task[N-2] only if Task[N-1] had
    // ORD bit also set. Checking Task[N-3] is not needed
    vpux::DenseMap<int64_t, std::deque<VPUIP::DMATypeOpInterface>> dmaPreviousTasksQueueMap;

    // Traverse IR and check all dmaOps
    funcOp->walk([&](VPUIP::DMATypeOpInterface dmaOp) {
        VPUX_THROW_WHEN(dmaOp.getPortAttribute() == nullptr, "DMA op has no port attribute, op - '{0}'",
                        dmaOp->getLoc());

        const auto port = dmaOp.getPortVal();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
        const auto portValue = port.value();
        auto channelType = dmaOp.getChannelType();
        auto dmaQueueTaskId = getDMAQueueIdEncoding(portValue, channelType);
        _log.trace("Identified DMA operation on port - {0} and channel - {1}:'{2}', ", portValue, channelType,
                   dmaOp->getLoc());

        auto taskOp = dmaOp->getParentOfType<VPURT::TaskOp>();
        VPUX_THROW_WHEN(taskOp == nullptr, "Parent must be VPURT::TaskOp");

        // Skip DMAs which are used for profiling buffer management because
        // applying OOO optimization to them can lead to concurrency issues,
        // specifically in DMA HWP. Clean data for this queue so that next DMA
        // is also not conidered for ORD clear
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(dmaOp.getOperation())) {
            if (nndmaOp.getProfilingBufferMgmt()) {
                dmaPreviousTasksQueueMap[dmaQueueTaskId].clear();
                dmaPreviousTasksQueueMap.erase(dmaQueueTaskId);
                return;
            }
        }

        if (dmaPreviousTasksQueueMap.find(dmaQueueTaskId) == dmaPreviousTasksQueueMap.end()) {
            // First task on this queue. Just store information about it
            dmaPreviousTasksQueueMap[dmaQueueTaskId].push_back(dmaOp);
            return;
        }

        bool memOverlap = false;
        for (auto inputOperand : VPUIP::getLayerInputs(dmaOp.getOperation())) {
            auto dmaInputDeclBuff = inputOperand.getDefiningOp<VPURT::DeclareBufferOp>();
            VPUX_THROW_UNLESS(dmaInputDeclBuff || inputOperand.getDefiningOp<Const::DeclareOp>(),
                              "DMA op does not have supported input - '{0}'", dmaOp->getLoc());
            if (dmaInputDeclBuff) {
                for (auto& prevDmaOp : dmaPreviousTasksQueueMap[dmaQueueTaskId]) {
                    auto prevDmaOutputDeclBuff = prevDmaOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

                    VPUX_THROW_UNLESS(prevDmaOutputDeclBuff,
                                      "DMA op does not have DeclareBufferOp on its output - '{0}'",
                                      prevDmaOp->getLoc());

                    // Perform a check on dma inputs and outputs to understand if there is
                    // no overlap
                    if ((memOverlap = isMemoryOverlap(prevDmaOutputDeclBuff, dmaInputDeclBuff))) {
                        break;
                    }
                }
            }

            if (memOverlap) {
                break;
            }
        }

        if (memOverlap) {
            _log.nest().trace("No optimization possible due to overlap in memory ranges");
            dmaPreviousTasksQueueMap[dmaQueueTaskId].clear();
        } else {
            // No overlap. OOO bit can be set
            _log.nest().trace("Out of order execution possible, disable ORD bit");
            dmaOp.setOutOfOrder();
        }
        dmaPreviousTasksQueueMap[dmaQueueTaskId].push_back(dmaOp);
        // Only store parent and grand parent tasks as only they are relevant for
        // mem overlap check
        if (dmaPreviousTasksQueueMap[dmaQueueTaskId].size() > 2) {
            dmaPreviousTasksQueueMap[dmaQueueTaskId].pop_front();
        }
    });

    if (!_workloadManagementMode.has_value() || _workloadManagementMode <= WorkloadManagementMode::PWLM_V2_PAGES) {
        VPURT::verifyBarrierSlots(funcOp, _log);
    }
}

}  // namespace

//
// createDMAOutOfOrderOptimizationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createDMAOutOfOrderOptimizationPass(
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<DMAOutOfOrderOptimizationPass>(workloadManagementMode, log);
}
