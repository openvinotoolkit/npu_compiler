//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/compiler/utils/strings.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
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

// Loop through all TaskOps in FuncOp and assign incremental index which will be used
// for task identification. Store and return a map of index -> TaskOp.
std::map<size_t, VPURT::TaskOp> assignIndicesToTasks(mlir::func::FuncOp funcOp) {
    std::map<size_t, VPURT::TaskOp> indexToTaskMap;
    size_t opIndex = 0;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        indexToTaskMap[opIndex] = taskOp;
        // enable identification in dot and schedule trace
        taskOp->setAttr("opIndex", getIntAttr(taskOp.getContext(), opIndex));
        taskOp->setLoc(appendLoc(taskOp->getLoc(), "taskIndex={0}", opIndex));
        taskOp.getInnerTaskOp()->setLoc(appendLoc(taskOp.getInnerTaskOp()->getLoc(), "taskIndex={0}", opIndex));
        ++opIndex;
    });
    return indexToTaskMap;
}

void logTaskInfo(size_t opIndex, VPURT::TaskOp taskOp, Logger log) {
    log.trace("opIndex {0}", opIndex);
    log.trace("taskLoc {0}", taskOp.getLoc());
    log.trace("task {0}", taskOp);
}

std::set<VPURT::ConfigureBarrierOp> getUserBarriers(size_t insertionIndex, size_t printIndex,
                                                    std::map<size_t, VPURT::TaskOp>& indexToTaskMap, Logger log) {
    std::set<VPURT::ConfigureBarrierOp> usedWaitBarriers;
    // store all wait barriers - they will definitely be used
    for (size_t opIndex = 0; opIndex <= insertionIndex; ++opIndex) {
        auto taskOp = indexToTaskMap[opIndex];
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
    auto insertionTaskOp = indexToTaskMap[insertionIndex];
    if (!insertionTaskOp.getUpdateBarriers().empty()) {
        auto barrierOp = (*insertionTaskOp.getUpdateBarriers().begin()).getDefiningOp<VPURT::ConfigureBarrierOp>();
        usedWaitBarriers.insert(barrierOp);
    }

    return usedWaitBarriers;
}

int64_t findFreePhysicalBarrierId(VPURT::TaskOp insertionTaskOp) {
    std::set<int64_t> physicalBarIds;
    for (auto bar : insertionTaskOp.getUpdateBarriers()) {
        auto barrierOp = bar.getDefiningOp<VPURT::ConfigureBarrierOp>();
        physicalBarIds.insert(barrierOp.getId());
    }
    for (size_t id = 0; id < physicalBarIds.size() + 1; ++id) {
        if (physicalBarIds.find(id) == physicalBarIds.end()) {
            return id;
        }
    }
    VPUX_THROW("Failed to find free physical barrier id");
}

VPURT::ConfigureBarrierOp insertNewFinalBarrierOp(VPURT::TaskOp insertionTaskOp, mlir::func::FuncOp funcOp) {
    auto barOps = funcOp.getOps<VPURT::ConfigureBarrierOp>();

    auto lastBarOpIt = barOps.begin();
    std::advance(lastBarOpIt, std::distance(barOps.begin(), barOps.end()) - 1);
    auto lastBarOp = *lastBarOpIt;

    mlir::OpBuilder builder(lastBarOp.getOperation());
    auto idForFinalBarrier = findFreePhysicalBarrierId(insertionTaskOp);
    return builder.create<VPURT::ConfigureBarrierOp>(lastBarOp.getLoc(), idForFinalBarrier, true, false, nullptr);
}

mlir::Type insertNewCopyOut(mlir::Value targetBuffer, VPURT::TaskOp insertionTaskOp,
                            VPURT::ConfigureBarrierOp newFinalBarrierOp) {
    mlir::OpBuilder builder(insertionTaskOp.getOperation());
    // create new output buffer
    builder.setInsertionPoint(targetBuffer.getDefiningOp());
    auto memAttr = IndexedSymbolAttr::get(insertionTaskOp.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    const auto bufferType = mlir::cast<vpux::NDTypeInterface>(targetBuffer.getType());
    auto newType = mlir::MemRefType::get(bufferType.getShape(), bufferType.getElementType(),
                                         bufferType.getDimsOrder().toAffineMap(builder.getContext()), memAttr);
    // create new output buffer
    auto newBuffer = builder.create<VPURT::DeclareBufferOp>(insertionTaskOp.getLoc(), newType,
                                                            VPURT::BufferSection::NetworkOutput, 0);

    mlir::Value finalDmaWaitBar = nullptr;
    if (!insertionTaskOp.getUpdateBarriers().empty()) {
        // Use only 1 barrier to not break expectation that tasks should wait only
        // on up to 1 wait barrier - see pass SatisfyOneWaitBarrier
        finalDmaWaitBar = *insertionTaskOp.getUpdateBarriers().begin();
    }

    // create new final DMA after insertion point
    builder.setInsertionPointAfter(insertionTaskOp);
    auto newDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            builder, mlir::ValueRange(finalDmaWaitBar), mlir::ValueRange(newFinalBarrierOp.getBarrier()),
            insertionTaskOp.getLoc(), targetBuffer, newBuffer, 0, false, false, nullptr, nullptr);

    return newDMA.getType();
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
        funcOp.eraseArgument(numInputs + 1);
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
    mlir::Type elementType = mlir::FloatType::getF32(funcOp.getContext());
    if (newOutTypeND.getElementType().isF32()) {
        elementType = mlir::FloatType::getF16(funcOp.getContext());
    } else if (newOutTypeND.getElementType().isF16()) {
        elementType = mlir::FloatType::getF16(funcOp.getContext());
    } else if (auto integerInput = mlir::dyn_cast<mlir::IntegerType>(newOutTypeND.getElementType())) {
        elementType =
                mlir::IntegerType::get(funcOp.getContext(), integerInput.getWidth(), integerInput.getSignedness());
    } else if (mlir::isa<mlir::quant::QuantizedType>(newOutTypeND.getElementType())) {
        elementType = mlir::IntegerType::get(funcOp.getContext(), 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    } else {
        VPUX_THROW("Unsupported element type {0}, please add case", elementType);
    }
    const auto newOutTensorType = mlir::RankedTensorType::get(newOutTypeND.getShape(), elementType);

    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
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

void filterUsedTasks(size_t insertionIndex, std::map<size_t, VPURT::TaskOp>& indexToTaskMap) {
    // remove all operation after insertion index
    for (auto opIndex = insertionIndex + 1; opIndex < indexToTaskMap.size(); ++opIndex) {
        indexToTaskMap[opIndex].erase();
    }
}

void removeUnusedBarriers(std::set<VPURT::ConfigureBarrierOp>& toNotRemove, mlir::func::FuncOp funcOp) {
    auto barrierOps = to_small_vector(funcOp.getOps<VPURT::ConfigureBarrierOp>());
    for (auto& barrierOp : barrierOps) {
        // remove barriers with no use
        if (barrierOp.getBarrier().use_empty()) {
            barrierOp->erase();
        } else if (toNotRemove.find(barrierOp) == toNotRemove.end()) {
            barrierOp->erase();
        }
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

    _log.trace("Selected _opIndex {0}", _opIndex);
    _log.trace("Selected _bufferIndex {0}", _bufferIndex);
    _log.trace("Selected _insertionIndex {0}", _insertionIndex);

    // TODO: E#92445 unique and simple identification of the same ops with changed IR order
    // assign indexes to operation for identification
    auto indexToTaskMap = assignIndicesToTasks(funcOp);

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

    // ensure opIndex and insertionIndex exist
    if (indexToTaskMap.size() < opIndex) {
        _log.warning("Selected opIndex is not in IR {0}, max index {1}", opIndex, indexToTaskMap.size());
        return;
    }
    if (indexToTaskMap.size() < insertionIndex) {
        _log.warning("Selected insertionIndex is not in IR {0}, max index {1}", insertionIndex, indexToTaskMap.size());
        return;
    }

    const auto targetTaskOp = indexToTaskMap[opIndex].getInnerTaskOp();
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

    // retrieve used barriers by Tasks to insertion point
    auto usedBarriers = getUserBarriers(insertionIndex, printIndex, indexToTaskMap, _log);

    // create new final barrier
    auto newFinallBarrierOp = insertNewFinalBarrierOp(indexToTaskMap[insertionIndex], funcOp);
    usedBarriers.insert(newFinallBarrierOp);

    // insert new copy out for target buffer
    auto newOutType = insertNewCopyOut(uniqueVals[bufferIndex], indexToTaskMap[insertionIndex], newFinallBarrierOp);

    // update all output types
    updateOutputType(newOutType, funcOp);

    // remove tasks and unused barriers after insertion point
    filterUsedBarriers(usedBarriers, funcOp);
    filterUsedTasks(insertionIndex, indexToTaskMap);
    removeUnusedBarriers(usedBarriers, funcOp);
}

}  // namespace

//
// createIntermediateBufferOutputPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createIntermediateBufferOutputPass(Logger log) {
    return std::make_unique<IntermediateBufferOutputPass>(log);
}
