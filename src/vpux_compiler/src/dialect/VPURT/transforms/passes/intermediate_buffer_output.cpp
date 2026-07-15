//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/execution_group_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/compiler/utils/strings.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/utils/core/developer_build_utils.hpp"
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include <limits>

namespace vpux::VPURT {
#define GEN_PASS_DECL_INTERMEDIATEBUFFEROUTPUT
#define GEN_PASS_DEF_INTERMEDIATEBUFFEROUTPUT
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

bool validSizeT(int number) {
    return number >= static_cast<int>(std::numeric_limits<size_t>::min());
}

SmallVector<mlir::Value> getUniqueVals(mlir::Operation* op, Logger log) {
    SmallVector<mlir::Value> inputs;
    ValueOrderedSet uniqueVals;
    log.trace("Operation buffers with indices:");
    for (size_t index = 0; index < op->getNumOperands(); ++index) {
        // Note: outputs of operation also part of operands
        auto val = op->getOperand(index);
        if (val == nullptr || uniqueVals.find(val) != uniqueVals.end()) {
            continue;
        }

        log.nest().trace("Index={0}, buffer {1}", uniqueVals.size(), val);
        inputs.push_back(val);
        uniqueVals.insert(val);
    }
    return inputs;
}

void logTaskInfo(size_t opIndex, VPURT::TaskOp taskOp, Logger log) {
    log.trace("opIndex {0}", opIndex);
    log.trace("taskLoc {0}", taskOp.getLoc());
    log.trace("task {0}", taskOp);
}

std::set<VPURT::ConfigureBarrierOp> getUsedBarriers(size_t insertionIndex, size_t printIndex, BarrierInfo& barrierInfo,
                                                    Logger log) {
    std::set<VPURT::ConfigureBarrierOp> usedWaitBarriers;
    // store all wait barriers - they will definitely be used
    for (size_t opIndex = 0; opIndex <= insertionIndex; ++opIndex) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(opIndex);
        for (auto bar : taskOp.getWaitBarriers()) {
            auto barrierOp = bar.getDefiningOp<VPURT::ConfigureBarrierOp>();
            usedWaitBarriers.insert(barrierOp);
        }
        if (opIndex > printIndex) {
            logTaskInfo(opIndex, taskOp, log);
        }
    }
    // for insertion point also store update barriers - only one, no more needed
    // as only first will be used as wait barrier for copy out
    auto insertionTaskOp = barrierInfo.getTaskOpAtIndex(insertionIndex);
    if (!insertionTaskOp.getUpdateBarriers().empty()) {
        auto barrierOp = (*insertionTaskOp.getUpdateBarriers().begin()).getDefiningOp<VPURT::ConfigureBarrierOp>();
        usedWaitBarriers.insert(barrierOp);
    }

    return usedWaitBarriers;
}

// Retrieve barriers for new copy out DMA
// Wait barrier is needed to guarantee dependency to copy out buffer producer
// Update barrier (final barrier) is needed to signal schedule completion
std::pair<mlir::Value, mlir::Value> getCopyOutWaitAndUpdateBars(mlir::OpBuilder& builder,
                                                                std::set<VPURT::ConfigureBarrierOp>& usedBarriers,
                                                                VPURT::TaskOp insertionTaskOp,
                                                                BarrierInfo& barrierInfo) {
    // find index of latest barrier that will not be removed
    auto latestBarOp = *std::max_element(usedBarriers.begin(), usedBarriers.end(),
                                         [&](VPURT::ConfigureBarrierOp a, VPURT::ConfigureBarrierOp b) {
                                             return barrierInfo.getIndex(a) < barrierInfo.getIndex(b);
                                         });
    builder.setInsertionPoint(latestBarOp->getNextNode());

    auto latestBarInd = barrierInfo.getIndex(latestBarOp);
    VPUX_THROW_UNLESS(latestBarInd < barrierInfo.getNumOfBarrierOps() - 1, "No barrier will be removed.");

    auto nextFreeBar = latestBarInd + 1;

    // Identify copyOut wait barrier
    mlir::Value copyOutWaitBar;
    if (!insertionTaskOp.getUpdateBarriers().empty()) {
        // reuse existing barrier
        copyOutWaitBar = *insertionTaskOp.getUpdateBarriers().begin();
    } else {
        // create new barrier
        auto nextBar = mlir::cast<VPURT::ConfigureBarrierOp>(barrierInfo.getBarrierOpAtIndex(nextFreeBar));
        auto newWaitBarrierPid = nextBar.getId();
        auto newWaitBarrierWlmPageAttr = nextBar.getWlmPageAttr();
        auto newWaitBarrierOp = builder.create<VPURT::ConfigureBarrierOp>(nextBar.getLoc(), newWaitBarrierPid, false,
                                                                          false, newWaitBarrierWlmPageAttr);
        copyOutWaitBar = newWaitBarrierOp.getBarrier();
        usedBarriers.insert(newWaitBarrierOp);
        nextFreeBar++;
    }

    // Create update barrier for copyOut DMA - final barrier
    auto nextBar = mlir::cast<VPURT::ConfigureBarrierOp>(barrierInfo.getBarrierOpAtIndex(nextFreeBar));
    auto newUpdateBarrierPid = nextBar.getId();
    auto newUpdateBarrierWlmPageAttr = nextBar.getWlmPageAttr();

    auto newUpdateBarrierOp = builder.create<VPURT::ConfigureBarrierOp>(nextBar.getLoc(), newUpdateBarrierPid, true,
                                                                        false, newUpdateBarrierWlmPageAttr);
    auto copyOutUpdateBar = newUpdateBarrierOp.getBarrier();
    usedBarriers.insert(newUpdateBarrierOp);

    return {copyOutWaitBar, copyOutUpdateBar};
}

mlir::Type insertNewCopyOut(mlir::Value targetBuffer, size_t insertionTaskIndex, mlir::Value waitBar,
                            mlir::Value updateBar, BarrierInfo& barrierInfo) {
    auto insertionTaskOp = barrierInfo.getTaskOpAtIndex(insertionTaskIndex);
    mlir::OpBuilder builder(insertionTaskOp.getOperation());
    // create new output buffer
    builder.setInsertionPoint(targetBuffer.getDefiningOp());
    auto memAttr = IndexedSymbolAttr::get(insertionTaskOp.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    const auto bufferType = mlir::cast<vpux::NDTypeInterface>(targetBuffer.getType());
    auto newType =
            vpux::getMemRefType(bufferType.getShape(), bufferType.getElementType(), bufferType.getDimsOrder(), memAttr);
    // create new output buffer
    auto newBuffer = builder.create<VPURT::DeclareBufferOp>(insertionTaskOp.getLoc(), newType,
                                                            VPURT::BufferSection::NetworkOutput, 0);

    // create new final DMA after insertion point
    builder.setInsertionPointAfter(insertionTaskOp);
    auto newDma = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange(waitBar), mlir::ValueRange(updateBar),
                                                        insertionTaskOp.getLoc(), targetBuffer, newBuffer, 0, false,
                                                        false, nullptr, false);

    auto newDmaTaskOp = newDma->getParentOfType<VPURT::TaskOp>();

    auto waitBarrierOp = waitBar.getDefiningOp<VPURT::ConfigureBarrierOp>();
    if (waitBarrierOp.getWlmPage().has_value()) {
        auto waitBarrierPage = waitBarrierOp.getWlmPage().value();
        newDmaTaskOp.setWlmPage(waitBarrierPage);
    }

    return newDma.getType();
}

void updateOutputType(mlir::Type newOutType, mlir::func::FuncOp funcOp) {
    // update function return ops
    auto functionResults = to_small_vector(funcOp.getOps<mlir::func::ReturnOp>());
    for (size_t index = 1; index < functionResults.size(); ++index) {
        // remove other return ops
        functionResults[index].erase();
    }

    // update function result operands
    const auto numInputs = funcOp.getNumArguments() - funcOp.getNumResults();
    while (numInputs + 1 < funcOp.getNumArguments()) {
        funcOp.getArgument(numInputs + 1).dropAllUses();
        VPUX_THROW_WHEN(failed(funcOp.eraseArgument(numInputs + 1)), "Failed to erase argument {0}", numInputs + 1);
    }

    mlir::OpBuilder builder(functionResults[0]);
    SmallVector<mlir::Value> funcOutputs;
    for (auto operand : functionResults[0].getOperands()) {
        if (operand != nullptr) {
            funcOutputs.push_back(operand);
        }
    }

    // create new return with new type
    VPUX_THROW_WHEN(funcOutputs.size() != 1, "Unsupported number of outputs");
    funcOutputs[funcOutputs.size() - 1].setType(newOutType);
    builder.create<mlir::func::ReturnOp>(functionResults[0].getLoc(), funcOutputs);
    functionResults[0].erase();

    SmallVector<mlir::Type> newInputsTypes;
    for (auto blockArg : funcOp.getArguments()) {
        bool returnOpUser = false;
        for (auto userOp : blockArg.getUsers()) {
            if (mlir::isa<mlir::func::ReturnOp>(userOp)) {
                returnOpUser = true;
                break;
            }
        }
        if (returnOpUser) {
            continue;
        }
        newInputsTypes.push_back(blockArg.getType());
    }

    // update function types
    SmallVector<mlir::Type> newResultTypes = {newOutType};
    newInputsTypes.push_back(newOutType);

    auto newFunctionType = mlir::FunctionType::get(funcOp.getContext(), newInputsTypes, newResultTypes);
    funcOp.setType(newFunctionType);

    // update module output
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
    if (netInfoOps.empty()) {
        return;
    }

    auto newOutTypeND = mlir::cast<vpux::NDTypeInterface>(newOutType);
    // precision must be float or integer
    mlir::Type elementType = mlir::Float32Type::get(funcOp.getContext());
    if (newOutTypeND.getElementType().isF32()) {
        elementType = mlir::Float16Type::get(funcOp.getContext());
    } else if (newOutTypeND.getElementType().isF16()) {
        elementType = mlir::Float16Type::get(funcOp.getContext());
    } else if (auto integerInput = mlir::dyn_cast<mlir::IntegerType>(newOutTypeND.getElementType())) {
        elementType =
                mlir::IntegerType::get(funcOp.getContext(), integerInput.getWidth(), integerInput.getSignedness());
    } else if (mlir::isa<mlir::quant::QuantizedType>(newOutTypeND.getElementType())) {
        elementType = mlir::IntegerType::get(funcOp.getContext(), 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    } else {
        VPUX_THROW("Unsupported element type {0}, please add case", elementType);
    }
    const auto newOutTensorType = mlir::RankedTensorType::get(newOutTypeND.getShape(), elementType);

    auto netInfo = net::getNetworkInfo(moduleOp);
    auto outputsInfo = to_small_vector(netInfo.getOutputsInfo().getOps<net::DataInfoOp>());
    for (auto p : outputsInfo | indexed) {
        auto outputIdx = p.index();
        auto outputInfo = p.value();

        if (outputIdx == 0) {
            outputInfo.setUserType(mlir::cast<mlir::TensorType>(newOutTensorType));
        } else {
            outputInfo.erase();
        }
    }
}

void filterUsedBarriers(std::set<VPURT::ConfigureBarrierOp>& usedBarriers, mlir::func::FuncOp funcOp) {
    auto taskOps = to_small_vector(funcOp.getOps<VPURT::TaskOp>());
    mlir::DenseMap<VPURT::TaskOp, mlir::DenseSet<VPURT::ConfigureBarrierOp>> filteredTaskBarriers;

    for (auto& taskOp : taskOps) {
        auto updateBarriers = taskOp.getUpdateBarriers();
        for (auto bar : updateBarriers) {
            auto childBarrierOp = bar.getDefiningOp<VPURT::ConfigureBarrierOp>();
            if (usedBarriers.find(childBarrierOp) != usedBarriers.end()) {
                filteredTaskBarriers[taskOp].insert(childBarrierOp);
            }
        }

        taskOp.getUpdateBarriersMutable().clear();
        for (auto bar : filteredTaskBarriers[taskOp]) {
            taskOp.getUpdateBarriersMutable().append(bar.getBarrier());
        }
    }
}

void filterUsedTasks(size_t insertionIndex, BarrierInfo& barrierInfo) {
    // remove all operation after insertion index
    for (auto opIndex = insertionIndex + 1; opIndex < barrierInfo.getNumOfTasks(); ++opIndex) {
        barrierInfo.getTaskOpAtIndex(opIndex).erase();
    }
}

void removeUnusedBarriers(std::set<VPURT::ConfigureBarrierOp>& toNotRemove, mlir::func::FuncOp funcOp) {
    auto barrierOps = to_small_vector(funcOp.getOps<VPURT::ConfigureBarrierOp>());

    for (auto& barrierOp : barrierOps) {
        // remove barriers with no use or which were not identified as barriers that should not be removed
        if (!barrierOp.getBarrier().use_empty() && toNotRemove.find(barrierOp) != toNotRemove.end()) {
            continue;
        }
        barrierOp->erase();
    }
}

void removeUnusedDeclareOps(mlir::func::FuncOp funcOp) {
    // Removing redundant constants of declare buffer ops is functionally not needed
    // but it cleans IR and makes reading it easier
    funcOp->walk([&](Const::DeclareOp constOp) {
        if (constOp.getOutput().use_empty()) {
            constOp.erase();
        }
    });
    funcOp->walk([&](VPURT::DeclareBufferOp bufOp) {
        if (bufOp.getBuffer().use_empty()) {
            bufOp.erase();
        }
    });
}

void addLastTaskPerQueueDepToFinalBarrier(mlir::func::FuncOp funcOp, mlir::Value newCopyOutWaitBar) {
    // Make sure each FIFO has dependency to final barrier
    for (auto taskQueueFirstAndLastOp : vpux::VPURT::getTaskQueuesFirstAndLastOp(funcOp)) {
        auto& lastOpInQueue = taskQueueFirstAndLastOp.second.second;
        if (!lastOpInQueue.getUpdateBarriers().empty()) {
            continue;
        }
        // Add dependency to final barrier
        lastOpInQueue.getUpdateBarriersMutable().assign(newCopyOutWaitBar);
    }
}

class IntermediateBufferOutputPass final :
        public VPURT::impl::IntermediateBufferOutputBase<IntermediateBufferOutputPass> {
public:
    explicit IntermediateBufferOutputPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

////
// DEBUG TOOL which enables to dump buffer of operation at any moment
////
void IntermediateBufferOutputPass::safeRunOnFunc() {
    int _opIndex = -1;
    int _bufferIndex = -1;
    int _insertionIndex = -1;

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    barrierInfo.buildTaskQueueTypeMap();

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    // define strings for env variables
    std::string opIndexStr = "-1";
    std::string bufferIndexStr = "-1";
    std::string insertionIndexStr = "-1";
    // get value from env
    parseEnv("IE_NPU_DEBUG_OP_INDEX", opIndexStr);
    parseEnv("IE_NPU_DEBUG_BUFFER_INDEX", bufferIndexStr);
    parseEnv("IE_NPU_DEBUG_INSERTION_INDEX", insertionIndexStr);
    // cast values
    _opIndex = std::stoi(opIndexStr);
    _bufferIndex = std::stoi(bufferIndexStr);
    _insertionIndex = std::stoi(insertionIndexStr);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    _opIndex = opIndexVal.hasValue() ? checked_cast<size_t>(opIndexVal.getValue()) : _opIndex;
    _bufferIndex = bufferIndexVal.hasValue() ? checked_cast<size_t>(bufferIndexVal.getValue()) : _bufferIndex;
    _insertionIndex =
            insertionIndexVal.hasValue() ? checked_cast<size_t>(insertionIndexVal.getValue()) : _insertionIndex;

    auto funcOp = getOperation();
    mlir::OpBuilder builder(funcOp);

    _log.trace("Selected _opIndex {0}", _opIndex);
    _log.trace("Selected _bufferIndex {0}", _bufferIndex);
    _log.trace("Selected _insertionIndex {0}", _insertionIndex);

    // TODO: E#92445 unique and simple identification of the same ops with changed IR order

    // ensure all values in size_t range
    if (!validSizeT(_opIndex) || !validSizeT(_insertionIndex)) {
        _log.warning("Selected indices for ops are not in size_t range");
        return;
    }

    // index of operation of which buffer will be output
    size_t opIndex = checked_cast<size_t>(_opIndex);
    // index of operation after which buffer should output
    size_t insertionIndex = checked_cast<size_t>(_insertionIndex);
    // index of operation to log 10 previous operations
    size_t numOpsToPrint = 10;
    size_t printIndex = std::max(numOpsToPrint, opIndex) - numOpsToPrint;

    auto numOfTaskOps = barrierInfo.getNumOfTasks();
    // ensure opIndex and insertionIndex exist
    if (numOfTaskOps < opIndex) {
        _log.warning("Selected opIndex is not in IR {0}, max index {1}", opIndex, numOfTaskOps);
        return;
    }
    if (numOfTaskOps < insertionIndex) {
        _log.warning("Selected insertionIndex is not in IR {0}, max index {1}", insertionIndex, numOfTaskOps);
        return;
    }

    const auto targetTaskOp = barrierInfo.getTaskOpAtIndex(opIndex).getInnerTaskOp();
    _log.trace("targetTaskOp {0}", *targetTaskOp);
    auto uniqueVals = getUniqueVals(targetTaskOp, _log);

    // ensure valid buffer index
    if (!validSizeT(_bufferIndex)) {
        _log.warning("Selected index for buffer is not in size_t range");
        return;
    }
    // index of buffer which to output
    size_t bufferIndex = checked_cast<size_t>(_bufferIndex);

    // ensure opIndex has bufferIndex
    if (bufferIndex > uniqueVals.size() - 1) {
        _log.warning("Selected bufferIndex {0} is not valid, max index for targetTaskOp is {1}", bufferIndex,
                     uniqueVals.size() - 1);
        return;
    }

    auto insertionTaskOp = barrierInfo.getTaskOpAtIndex(insertionIndex);

    // retrieve used barriers by Tasks to insertion point
    auto usedBarriers = getUsedBarriers(insertionIndex, printIndex, barrierInfo, _log);

    _log.trace("Get copy out wait and update barriers");
    auto [copyOutWaitBar, copyOutUpdateBar] =
            getCopyOutWaitAndUpdateBars(builder, usedBarriers, insertionTaskOp, barrierInfo);

    // insert new copy out for target buffer
    _log.trace("Create new copy out DMA");
    auto newOutType =
            insertNewCopyOut(uniqueVals[bufferIndex], insertionIndex, copyOutWaitBar, copyOutUpdateBar, barrierInfo);

    // update all output types
    _log.trace("Update function output types");
    updateOutputType(newOutType, funcOp);

    // remove tasks and unused barriers after insertion point
    filterUsedBarriers(usedBarriers, funcOp);
    filterUsedTasks(insertionIndex, barrierInfo);
    _log.trace("Remove unused barriers and declare ops");
    removeUnusedBarriers(usedBarriers, funcOp);
    removeUnusedDeclareOps(funcOp);
    _log.trace("Make sure copy out DMA depends on last task per queue");
    addLastTaskPerQueueDepToFinalBarrier(funcOp, copyOutWaitBar);

    auto newFinalBarrierOp = copyOutUpdateBar.getDefiningOp<VPURT::ConfigureBarrierOp>();

    if (!newFinalBarrierOp.getWlmPage().has_value()) {
        return;
    }

    _log.trace("Patch WLM management ops");
    // Patch IR for WLM management ops
    // For Enqueue DMAs:
    //   Identify last DPU and SHV per queue that is present in IR
    //  Any Enqueue DMAs present in IR scheduling DPUs/SHVs no longer present in IR need to be patched or
    //  converted into sync DMAs
    // For Fetch DMAs:
    //   Identify DPU and SHV execution groups and replace Fetch DMAs with SyncDMAs
    //   for no longer present in IR groups
    // For Barrier Programming DMAs:
    //   DMAs which configure barriers for pages no longer present in IR need to be
    //   replaced with SyncDMAs
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto numClusters = config::getTileExecutor(module).getCount();
    auto& ctx = getContext();
    auto wlmPageLast = newFinalBarrierOp.getWlmPage().value();

    barrierInfo = BarrierInfo(funcOp);
    barrierInfo.buildTaskQueueTypeMap();

    mlir::DenseMap<std::tuple<config::ExecutorKind, size_t, size_t>, std::pair<size_t, size_t>>
            lastTaskWorkloadRangePerQueueType;
    // Identify last workload index per queue type present in IR. This is needed to later
    // identify enqueue DMAs needed to be patched or removed
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        auto taskInd = barrierInfo.getIndex(taskOp);
        auto queueType = barrierInfo.getTaskQueueType(taskInd);

        if (queueType.type != config::ExecutorKind::DPU && queueType.type != config::ExecutorKind::SHAVE_ACT) {
            return;
        }

        auto [tileIdx, listIdx] = VPURT::getTileAndListIndex(queueType, numClusters, arch);
        auto execKindAndTileAndList = std::make_tuple(queueType.type, tileIdx, listIdx);

        auto numOfWorkloads = VPURT::getNumberOfWorkloads(barrierInfo.getTaskOpAtIndex(taskInd));
        int workloadIndexOffset = 0;
        if (lastTaskWorkloadRangePerQueueType.count(execKindAndTileAndList) != 0) {
            workloadIndexOffset = lastTaskWorkloadRangePerQueueType[execKindAndTileAndList].second + 1;
        }
        lastTaskWorkloadRangePerQueueType[execKindAndTileAndList].first = workloadIndexOffset;
        lastTaskWorkloadRangePerQueueType[execKindAndTileAndList].second = workloadIndexOffset + numOfWorkloads - 1;

        _log.trace("Queue {0}:{1}: Task {2} with workloads range {3}-{4}",
                   config::stringifyExecutorKind(queueType.type), queueType.id, taskInd, workloadIndexOffset,
                   workloadIndexOffset + numOfWorkloads - 1);
    });

    // Identify what execution groups are present in IR
    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    auto dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto swGroups = execGroupAnalysis.getActShvExecutionGroups();

    mlir::DenseMap<std::tuple<config::ExecutorKind, size_t, size_t>, size_t> lastTaskIndexesEnqueuedPerQueueType;

    // Traverse tasks and process EnqueueDMA and FetchDMA ops
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (auto enqueueDmaOp = mlir::dyn_cast<VPUIP::EnqueueDMAOp>(taskOp.getInnerTaskOp())) {
            // Identify EnqueueDmas indexes and task indexes that are enqueued by it. Patch or erase if they
            // correspond to DPUs/SHVs no longer present in IR
            auto pageOpt = taskOp.getWlmPage();
            VPUX_THROW_UNLESS(pageOpt.has_value(), "EnqueueDMAOp '{0}' does not have WLM page assigned",
                              taskOp.getLoc());
            auto pageInd = pageOpt.value();
            auto enqueueDmaIdx = barrierInfo.getIndex(taskOp);
            auto enqueueDmaAttr = enqueueDmaOp.getEnqueueDmaAttr();

            auto executorKind = enqueueDmaAttr.getTargetExecutorKindAttr().getValue();
            auto tileIdx = enqueueDmaAttr.getTileIdx().getValue().getZExtValue();
            auto listIdx = enqueueDmaAttr.getListIdx().getValue().getZExtValue();
            auto startTaskIdx = enqueueDmaAttr.getStartTaskIdx().getValue().getZExtValue();
            auto endTaskIdx = enqueueDmaAttr.getEndTaskIdx().getValue().getZExtValue();

            _log.trace("Found EnqueueDMAOp task {0} at page {1} for tasks {2}:{3}:{4}:{5}-{6}", enqueueDmaIdx, pageInd,
                       stringifyEnum(executorKind), tileIdx, listIdx, startTaskIdx, endTaskIdx);

            auto execKindAndTileAndList = std::make_tuple(executorKind, tileIdx, listIdx);

            if (lastTaskWorkloadRangePerQueueType.contains(execKindAndTileAndList)) {
                auto [lastTaskWorkloadStartIdx, lastTaskWorkloadEndIdx] =
                        lastTaskWorkloadRangePerQueueType[execKindAndTileAndList];

                if (endTaskIdx <= lastTaskWorkloadEndIdx) {
                    return;
                }

                if (startTaskIdx <= lastTaskWorkloadEndIdx) {
                    _log.nest().trace("Enqueue DMA needs to patched. New range: {0}-{1}", startTaskIdx,
                                      lastTaskWorkloadEndIdx);
                    auto newEndTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(&ctx), lastTaskWorkloadEndIdx);
                    auto newEnqueueDMAAttr = VPUIP::EnqueueDMAAttr::get(
                            &ctx, enqueueDmaAttr.getTargetExecutorKindAttr(), enqueueDmaAttr.getTileIdx(),
                            enqueueDmaAttr.getListIdx(), enqueueDmaAttr.getStartTaskIdx(), newEndTaskIdxAttr);
                    enqueueDmaOp.setEnqueueDmaAttrAttr(newEnqueueDMAAttr);
                    return;
                }
            }
            // Remove EnqueueDMA as it does not enqueue any task present in IR
            _log.nest().trace("Enqueue DMA needs to be replaced with SyncDma");
            builder.setInsertionPoint(enqueueDmaOp);
            builder.create<VPUIP::SyncDMAOp>(enqueueDmaOp.getLoc(), enqueueDmaOp.getInput(),
                                             enqueueDmaOp.getOutputBuff(), enqueueDmaOp.getPortAttr(),
                                             enqueueDmaOp.getIsOutOfOrder(), enqueueDmaOp.getIsCritical(),
                                             enqueueDmaOp.getDmaHwpIdAttr(), enqueueDmaOp.getProfilingMetadataAttr());

            enqueueDmaOp.erase();
            return;
        } else if (auto fetchDmaOp = mlir::dyn_cast<VPUIP::FetchDMAOp>(taskOp.getInnerTaskOp())) {
            // Identify FetchDmas corresponding to DPUs/SHVs no longer present in IR and convert them to SyncDMAs
            const auto fetchDmaAttr = fetchDmaOp.getFetchDmaAttr();
            const auto tileIdx = fetchDmaAttr.getTileIdx().getValue().getSExtValue();
            const auto listIdx = fetchDmaAttr.getListIdx().getValue().getSExtValue();
            const auto groupIdx = fetchDmaAttr.getExecGroupIdx().getValue().getSExtValue();
            const auto executorKind = fetchDmaAttr.getTargetExecutorKindAttr().getValue();

            auto queueType = VPURT::getQueueTypeFromTileAndListIndex(executorKind, tileIdx, listIdx, numClusters);

            _log.trace("Fetch DMA for {0}:{1}:{2} group {3}", stringifyEnum(executorKind), tileIdx, listIdx, groupIdx);
            auto& execGroupsMap = (executorKind == config::ExecutorKind::DPU) ? dpuGroups : swGroups;
            if (execGroupsMap.contains(queueType) && groupIdx < static_cast<int64_t>(execGroupsMap[queueType].size())) {
                // Fetch DMA is valid and corresponding group is present in IR
                return;
            }
            _log.nest().trace("Fetch DMA needs to be replaced with SyncDma");
            builder.setInsertionPoint(fetchDmaOp);
            builder.create<VPUIP::SyncDMAOp>(fetchDmaOp.getLoc(), fetchDmaOp.getInput(), fetchDmaOp.getOutputBuff(),
                                             fetchDmaOp.getPortAttr(), fetchDmaOp.getIsOutOfOrder(),
                                             fetchDmaOp.getIsCritical(), fetchDmaOp.getDmaHwpIdAttr(),
                                             fetchDmaOp.getProfilingMetadataAttr());

            fetchDmaOp.erase();
        } else if (auto barProgDmaOp = mlir::dyn_cast<VPUIP::BarProgDMAOp>(taskOp.getInnerTaskOp())) {
            // Identify Barrier Programming DMAs configuring barriers for WLM pages no longer present in IR
            auto pageOpt = taskOp.getWlmPage();
            VPUX_THROW_UNLESS(pageOpt.has_value(), "BarrierProgrammingDMAOp '{0}' does not have WLM page assigned",
                              taskOp.getLoc());
            auto pageInd = pageOpt.value();

            // Barrier Programming DMA in PageN configures barriers starting from PageN+1, ...
            // If PageN is the last in IR then this DMA is not needed anymore
            // Exception is DMA in Page0 which configures barriers starting from Page0, ...
            // and is always needed
            _log.trace("Barrier Programming DMA for WLM page {0}", pageInd);
            if (pageInd == 0 || pageInd < wlmPageLast) {
                // WLM page is present in IR
                return;
            }
            _log.nest().trace("Barrier Programming DMA needs to be replaced with SyncDma");
            builder.setInsertionPoint(barProgDmaOp);
            builder.create<VPUIP::SyncDMAOp>(barProgDmaOp.getLoc(), barProgDmaOp.getInput(),
                                             barProgDmaOp.getOutputBuff(), barProgDmaOp.getPortAttr(),
                                             barProgDmaOp.getIsOutOfOrder(), barProgDmaOp.getIsCritical(),
                                             barProgDmaOp.getDmaHwpIdAttr(), barProgDmaOp.getProfilingMetadataAttr());

            barProgDmaOp.erase();
        }
    });
}

}  // namespace

//
// createIntermediateBufferOutputPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createIntermediateBufferOutputPass(Logger log) {
    return std::make_unique<IntermediateBufferOutputPass>(log);
}
