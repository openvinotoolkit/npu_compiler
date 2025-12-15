//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

// Helper structure to create outlining
// instance by using `async-deps-index`
struct OutliningInstanceIndex {
    std::set<size_t> storedFuncDepIndex;
    std::set<size_t> storedInputsIndex;
    std::set<size_t> storedOutputsIndex;
    std::set<size_t> storedOperationsIndex;
};

template <typename Range1, typename Range2>
std::set<size_t> findCommonElements(const Range1& range1, const Range2& range2) {
    std::set<size_t> commonElements;
    VPUX_THROW_UNLESS(llvm::is_sorted(range1), "Range1 is not sorted");
    VPUX_THROW_UNLESS(llvm::is_sorted(range2), "Range2 is not sorted");

    std::set_intersection(range1.begin(), range1.end(), range2.begin(), range2.end(),
                          std::inserter(commonElements, commonElements.begin()));

    return commonElements;
}

using SubFuncSlice = SmallVector<size_t>;

//
// AsyncRegionOutliningSplitter
//

class AsyncRegionOutliningSplitter {
public:
    AsyncRegionOutliningSplitter(size_t minOpsInBlock, AsyncDepsInfo& depsInfo, Logger& log)
            : _minOpsInBlock(minOpsInBlock), _depsInfo(depsInfo), _log(log) {
    }
    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp mainFunction);

private:
    void updateTokens();
    void processInputsAndDepOps();
    void processOutputOps();
    void initOutliningInstanceIndex(ArrayRef<SubFuncSlice> subFuncInstances);
    void processOperands(mlir::async::ExecuteOp execOp, OpOrderedSet& storedOperations, ValueOrderedSet& storedInputs);
    llvm::DenseMap<mlir::Value, mlir::Value> findDmaForMainFuncOutputBuffer(mlir::func::FuncOp mainFuncOp);
    void insertDmaForNetResults(llvm::DenseMap<mlir::Value, mlir::Value>& funcOutputNeedInsertSpillDma);
    SmallVector<size_t> findCutPoints();
    SmallVector<SubFuncSlice> splitOpsInMainFunToSubFuncs();
    SmallVector<OutliningInstance> createOutliningInstances();

private:
    mlir::async::ExecuteOp insertSpillWriteDmaOp(mlir::async::ExecuteOp insertAfterExecOp, mlir::Value bufferToSpill);
    mlir::async::ExecuteOp insertSpillReadDmaOp(mlir::async::ExecuteOp insertAfterExecOp, mlir::Value bufferToSpill);
    void insertSpillDmaOps(mlir::async::ExecuteOp spillExecOp);

private:
    // variables for inserting spilling operations
    /*
    -----------------------------------------------------------------------------
    |                        %exec:2 = spillingExecOp(...)               Func0  |
    |                        |                       |                          |
    |   %write0 = spillingWrite(%exec:0)     %write1 = spillingWrite(%exec:1)   |
    -----------------------------------------------------------------------------
    ---------------------------------            --------------------------------
    |                        Func1  |            |                       Func2  |
    |%read1 = spillingRead(%write0) |            |%read2 = spillingRead(%write1)|
    |           |                   |            |             |                |
    |%user1 = spillingUser0(%read1) |            |%user2 =spillingUser(%read2)  |
    ---------------------------------            --------------------------------
    */
    /*{%exec:0 : %write0}, {%exec:1 : %write1}*/
    llvm::DenseMap<mlir::Value, mlir::async::ExecuteOp> _spillingBufferAndWriteOpMap;
    /*{%exec:0 : {%user1}}, {%exec:1 : {user2}}*/
    llvm::DenseMap<mlir::Value, SmallVector<size_t>> _spillingBufferAndUserExecOpMap;
    /*{%user1 : {%read1}}, {%user2 : {%read2}}*/
    llvm::DenseMap<size_t, SmallVector<size_t>> _spillingUserExecOpAndReadOpMap;

private:
    /* record async index of function output buffer DMA operation remain in main function*/
    SmallVector<size_t> _funcOutputDMAInMainFunction = {};
    /* record outlined instances*/
    SmallVector<OutliningInstanceIndex, 4> _outliningInstancesIndex = {};
    /* minimum number of operations in an instance*/
    size_t _minOpsInBlock;
    AsyncDepsInfo& _depsInfo;
    Logger& _log;
};

// Helper function to create spill DMA async execute operation using a lambda body builder
mlir::async::ExecuteOp createSpillDMAExecOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value token,
                                            mlir::Value operand, mlir::Value targetBuffer) {
    // Create ExecOp body with a NNDMAOp and a YieldOp
    auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location odsLoc, mlir::ValueRange operands) {
        auto dmaOp = builder.create<VPUIP::NNDMAOp>(odsLoc, operands[0], targetBuffer);
        builder.create<mlir::async::YieldOp>(odsLoc, dmaOp->getResults());
    };

    // Create the execOp.
    auto execOp = builder.create<mlir::async::ExecuteOp>(loc, targetBuffer.getType(), token, operand, bodyBuilder);

    // Update executor attributes if available.
    if (!execOp.getBody()->empty()) {
        if (auto dmaOp = llvm::dyn_cast<VPUIP::NNDMAOp>(*execOp.getBody()->begin())) {
            if (auto dmaOpExecutor = mlir::dyn_cast_or_null<VPUIP::AsyncLayerOpInterface>(dmaOp.getOperation())) {
                auto executor = dmaOpExecutor.getExecutor();
                if (executor != nullptr) {
                    VPUIP::VPUIPDialect::setExecutor(execOp, executor);
                }
            }
        }
    }

    return execOp;
}

// Helper function to get swizzlingKeyAttr
mlir::IntegerAttr getSwizzlingKeyAttr(mlir::Type bufferType) {
    auto swizzlingScheme = getSwizzlingSchemeAttr(bufferType);
    if (swizzlingScheme != nullptr && swizzlingScheme.getKey().getInt() != 0) {
        return swizzlingScheme.getKey();
    }
    return nullptr;
}

// Function to insert spilling write DMA operation
// Example:
//   %0 produces a CMX buffer
//   %1 = spillingUser(%0)
// After inserting spilling write DMA operation:
//   %0 produces a CMX buffer
//   %spillWrite = DMA(%0) CMX -> DDR
//   %sprillRead = DMA(%spillWrite) DDR -> CMX
//   %1 = spillingUser(%sprillRead)
mlir::async::ExecuteOp AsyncRegionOutliningSplitter::insertSpillWriteDmaOp(mlir::async::ExecuteOp insertAfterExecOp,
                                                                           mlir::Value bufferToSpill) {
    auto spillWriteNameLoc = appendLoc(insertAfterExecOp->getLoc(), "{0}", _depsInfo.getIndex(insertAfterExecOp));
    auto createNewBufferOp = [&](mlir::OpBuilder& builder) -> mlir::Operation* {
        auto bufferToSpillType = getAsyncValueType(bufferToSpill);
        auto bufferToSpillNDType =
                mlir::isa<VPUIP::DistributedBufferType>(bufferToSpillType)
                        ? mlir::dyn_cast<vpux::NDTypeInterface>(
                                  mlir::cast<VPUIP::DistributedBufferType>(bufferToSpillType).getCompactType())
                        : mlir::dyn_cast<vpux::NDTypeInterface>(bufferToSpillType);

        auto bufferToSpillDDR = bufferToSpillNDType.changeMemSpace(VPU::MemoryKind::DDR);
        auto swizzlingKeyAttr = getSwizzlingKeyAttr(bufferToSpillType);
        if (swizzlingKeyAttr != nullptr) {
            return builder.create<VPURT::Alloc>(spillWriteNameLoc, bufferToSpillDDR, nullptr, swizzlingKeyAttr);
        }
        return builder.create<mlir::memref::AllocOp>(spillWriteNameLoc, mlir::cast<mlir::MemRefType>(bufferToSpillDDR));
    };

    mlir::OpBuilder builder(insertAfterExecOp);
    builder.setInsertionPoint(insertAfterExecOp);

    auto newBufferOp = createNewBufferOp(builder);
    auto newBufferResult = newBufferOp->getResult(0);

    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillWriteExecOp = createSpillDMAExecOp(builder, spillWriteNameLoc, insertAfterExecOp.getToken(),
                                                 bufferToSpill, newBufferResult);

    _depsInfo.insertNewExecOpToDepsMap(spillWriteExecOp);
    _depsInfo.addDependency(insertAfterExecOp, spillWriteExecOp);
    return spillWriteExecOp;
}

// Function to insert spilling read DMA operation
// Example:
//   %0 produces a CMX buffer
//   %1 = spillingUser(%0)
// After inserting spilling write DMA operation:
//   %0 produces a CMX buffer
//   %spillWrite = DMA(%0) CMX -> DDR
//   %sprillRead = DMA(%spillWrite) DDR -> CMX
//   %1 = spillingUser(%sprillRead)
mlir::async::ExecuteOp AsyncRegionOutliningSplitter::insertSpillReadDmaOp(mlir::async::ExecuteOp insertBeforeExecOp,
                                                                          mlir::Value bufferToSpill) {
    auto spillReadNameLoc = appendLoc(insertBeforeExecOp->getLoc(), "{0}", _depsInfo.getIndex(insertBeforeExecOp));
    auto createNewBufferOp = [&](mlir::OpBuilder& builder) -> mlir::Operation* {
        mlir::Operation* newBufferOp;
        auto bufferToSpillType = getAsyncValueType(bufferToSpill);
        auto swizzlingKeyAttr = getSwizzlingKeyAttr(bufferToSpillType);
        if (auto distributedBuffer = mlir::dyn_cast<VPUIP::DistributedBufferType>(bufferToSpillType)) {
            newBufferOp = builder.create<VPURT::AllocDistributed>(spillReadNameLoc, bufferToSpillType, nullptr,
                                                                  swizzlingKeyAttr);
        } else {
            if (swizzlingKeyAttr != nullptr) {
                newBufferOp =
                        builder.create<VPURT::Alloc>(spillReadNameLoc, bufferToSpillType, nullptr, swizzlingKeyAttr);
            } else {
                newBufferOp = builder.create<mlir::memref::AllocOp>(spillReadNameLoc,
                                                                    mlir::cast<mlir::MemRefType>(bufferToSpillType));
            }
        }
        return newBufferOp;
    };

    auto spillingWriteExecOp = _spillingBufferAndWriteOpMap[bufferToSpill];
    auto spillWriteResult = spillingWriteExecOp.getBodyResults()[0];
    auto spillWriteToken = spillingWriteExecOp.getToken();

    mlir::OpBuilder builder(insertBeforeExecOp);
    builder.setInsertionPointAfter(spillingWriteExecOp);

    auto newBufferOp = createNewBufferOp(builder);
    auto newBufferResult = newBufferOp->getResult(0);

    builder.setInsertionPoint(insertBeforeExecOp);
    auto spillReadExecOp =
            createSpillDMAExecOp(builder, spillReadNameLoc, spillWriteToken, spillWriteResult, newBufferResult);

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillReadExecOp);
    // Update dependency
    _depsInfo.addDependency(spillingWriteExecOp, spillReadExecOp);

    return spillReadExecOp;
}

// Function to insert spilling DMA operations for a execOp
void AsyncRegionOutliningSplitter::insertSpillDmaOps(mlir::async::ExecuteOp spillExecOp) {
    // Check if spillExecOp and userExecOp are in same outlining instance
    auto isSameInstance = [&](size_t spillExecOpIndex, size_t userExecOpIndex) {
        for (const auto& instance : _outliningInstancesIndex) {
            if (instance.storedOperationsIndex.count(spillExecOpIndex) > 0 &&
                instance.storedOperationsIndex.count(userExecOpIndex) > 0) {
                return true;
            }
        }
        return false;
    };

    auto spillExecOpIndex = _depsInfo.getIndex(spillExecOp);
    for (auto bufferToSpill : spillExecOp.getBodyResults()) {
        auto bufferToSpillUses = bufferToSpill.getUses();
        if (bufferToSpillUses.empty()) {
            continue;
        }
        // Do not insert when user is awaitOp
        if (llvm::any_of(bufferToSpillUses, [](mlir::OpOperand& use) {
                return mlir::isa<mlir::async::AwaitOp>(use.getOwner());
            })) {
            continue;
        }
        // Do not insert when bufferToSpill is already in DDR
        if (auto bufferToSpillType = mlir::dyn_cast<mlir::MemRefType>(getAsyncValueType(bufferToSpill))) {
            auto bufferToSpillNDType = mlir::dyn_cast<vpux::NDTypeInterface>(bufferToSpillType);
            if (bufferToSpillNDType && bufferToSpillNDType.getMemoryKind() == VPU::MemoryKind::DDR) {
                continue;
            }
        }
        auto spillWriteExecOp = insertSpillWriteDmaOp(spillExecOp, bufferToSpill);
        // Add spillWriteExecOp to global map as helper to insert spilling read DmaOp
        _spillingBufferAndWriteOpMap[bufferToSpill] = spillWriteExecOp;
        for (auto& use : llvm::make_early_inc_range(bufferToSpillUses)) {
            if (auto userExecOp = mlir::dyn_cast<mlir::async::ExecuteOp>(use.getOwner())) {
                auto userExecOpIndex = _depsInfo.getIndex(userExecOp);
                // If spillExecOpIndex and userExecOpIndex are in same _outliningInstancesIndex
                // do not insert spilling DMA operations, continue to next user
                if (isSameInstance(spillExecOpIndex, userExecOpIndex)) {
                    continue;
                }

                auto spillReadExecOp = insertSpillReadDmaOp(userExecOp, bufferToSpill);
                auto spillReadExecOpIndex = _depsInfo.getIndex(spillReadExecOp);
                // Update SpillWrite/Read Users operand and token dependency
                use.set(spillReadExecOp.getBodyResults()[0]);
                auto spillReadToken = spillReadExecOp.getToken();
                userExecOp.getDependenciesMutable().append(spillReadToken);
                // Update dependency
                _depsInfo.addDependency(spillReadExecOp, userExecOp);
                // Record spilling op relation
                _spillingBufferAndUserExecOpMap[bufferToSpill].push_back(userExecOpIndex);
                _spillingUserExecOpAndReadOpMap[userExecOpIndex].push_back(spillReadExecOpIndex);
            }
        }
    }
}

// Original function output has been allocated buffer, no need to add buffers again.
// This function find function output buffer DMA operations.
// If the DMA operation is only operation in the execOp body except async.yield, it
// will be remained in the main function, no need to insert spill DMA operation.
// Otherwise, record the relation of function output buffer arguments and DMA operation
// then insert spill DMA operation in insertDmaForNetResults().
// Q: Why need to insert DMA operation for function output buffer?
// A: Function output buffer arguments cannot be outlined to sub functions because aliasesInfo
//    cannot find root buffer cross function boundaries.
// Example:
// Operation set: [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Case1: operation 8 is the main function DMA, it is the only operation in the execOp body
// except async.yield, it will remain in the main function.
// Case2: operation 8 is the main function DMA, it is not the only operation in the execOp body
// except async.yield, a new DMA operation will be inserted later.
llvm::DenseMap<mlir::Value /*bufferToSpill*/, mlir::Value /*outputBuffer*/>
AsyncRegionOutliningSplitter::findDmaForMainFuncOutputBuffer(mlir::func::FuncOp mainFuncOp) {
    const auto numInputs =
            static_cast<int64_t>(mainFuncOp.getNumArguments()) - static_cast<int64_t>(mainFuncOp.getNumResults());
    VPUX_THROW_UNLESS(numInputs >= 0, "Number of inputs cannot be negative");

    llvm::DenseMap<mlir::Value, mlir::Value> funcOutputNeedInsertSpillDma;
    auto processExecOpResult = [&](mlir::Value execOpResult, mlir::Value funcArg, mlir::async::ExecuteOp execOp) {
        const vpux::ValueSourceInfo aliaseInfo(execOpResult);
        if (aliaseInfo.getRoot(execOpResult) != funcArg) {
            return;
        }

        for (auto& execOpUse : llvm::make_early_inc_range(execOpResult.getUses())) {
            auto awaitOp = mlir::dyn_cast<mlir::async::AwaitOp>(execOpUse.getOwner());
            if (awaitOp == nullptr || awaitOp.getResult() == nullptr) {
                continue;
            }
            for (auto awaitOpUser : llvm::make_early_inc_range(awaitOp.getResult().getUsers())) {
                if (!mlir::isa<mlir::func::ReturnOp>(awaitOpUser)) {
                    continue;
                }
                // Check if the execOp region only has two operations:
                // VPUIP::NNDMAOp and async.yield (async.yield is implied as it is the terminator)
                if (execOp.getBody()->getOperations().size() == 2 &&
                    mlir::isa<VPUIP::NNDMAOp>(&execOp.getBody()->front())) {
                    _funcOutputDMAInMainFunction.push_back(_depsInfo.getIndex(execOp));
                } else {
                    // Record the relation of function output buffer and execOp result
                    funcOutputNeedInsertSpillDma[execOpResult] = funcArg;
                }
            }
        }
    };

    auto processFuncOutputArgUses = [&](mlir::Value funcArg) {
        for (auto& argUse : funcArg.getUses()) {
            auto userOp = argUse.getOwner();
            auto execOp = userOp->getParentOfType<mlir::async::ExecuteOp>();
            if (execOp == nullptr) {
                continue;
            }

            for (auto execOpResult : llvm::make_early_inc_range(execOp.getBodyResults())) {
                processExecOpResult(execOpResult, funcArg, execOp);
            }
        }
    };

    for (auto funcArg : llvm::make_early_inc_range(mainFuncOp.getArguments())) {
        // Skip function's input
        if (funcArg.getArgNumber() < numInputs) {
            continue;
        }

        processFuncOutputArgUses(funcArg);
    }

    return funcOutputNeedInsertSpillDma;
}

// Before inserting spill DMA for function output buffer:
//    func.func @main(%arg0: ..., %arg1: memref<1x1x128x224xf16, @DDR>) {
//        %token_284, %bodyResults_285 = async.execute ... {
//            %98 = VPUIP.ConcatView ...
//            %99 = VPUIP.PermuteCast ...
//            %100 = VPUIP.NNDMA inputs(%99) outputs(%arg1)
//            async.yield %100 : memref<1x1x128x224xf16, @DDR>
//        }
//        %97 = async.await %bodyResults_285 : ...
//        return %97 : ...
//    }

// After inserting spilt buffer:
//    func.func @main(%arg0: ..., %arg1: memref<1x1x128x224xf16, @DDR>) {
//        %alloc_283 = memref.alloc() : memref<1x1x128x224xf16, @DDR>   -> new buffer to replace %arg1
//        %token_284, %bodyResults_285 = async.execute ... {
//            %98 = VPUIP.ConcatView ...
//            %99 = VPUIP.PermuteCast ...
//            %100 = VPUIP.NNDMA inputs(%99) outputs(%alloc_283)
//            async.yield %100 : memref<1x1x128x224xf16, @DDR>
//        }
//        // copy %bodyResults_285 into %arg1
//        %token_286, %bodyResults_287 = async.execute [%token_284] (%bodyResults_285 as %arg3) {
//            %98 = VPUIP.NNDMA inputs(%arg3) outputs(%arg1)
//            async.yield %98 : memref<1x1x128x224xf16, @DDR>
//        }
//        %97 = async.await %bodyResults_287 : ...
//        return %97 : ...
//    }
// E-166699: Supporting AliasesInfo class to find root buffer cross function boundaries
// can help eliminate the need for inserting DMA operations for function output buffers.
void AsyncRegionOutliningSplitter::insertDmaForNetResults(
        llvm::DenseMap<mlir::Value, mlir::Value>& funcOutputNeedInsertSpillDma) {
    for (auto& [bufferToSpill, funcArg] : funcOutputNeedInsertSpillDma) {
        auto execOp = bufferToSpill.getDefiningOp<mlir::async::ExecuteOp>();
        VPUX_THROW_WHEN(execOp == nullptr, "Invalid buffer to spill, {0}", bufferToSpill);
        mlir::OpBuilder builder(execOp);
        builder.setInsertionPoint(execOp);
        auto outputDmaNameLoc = appendLoc(execOp->getLoc(), "{0}", _depsInfo.getIndex(execOp));

        // Create new allocation operation
        auto newAllocOp = builder.create<mlir::memref::AllocOp>(outputDmaNameLoc,
                                                                mlir::cast<mlir::MemRefType>(funcArg.getType()));
        auto newBufferResult = newAllocOp->getResult(0);
        funcArg.replaceAllUsesWith(newBufferResult);

        // Use the helper function to create and configure the AsyncExecOp
        builder.setInsertionPointAfter(execOp);
        auto outputDmaExecOp =
                createSpillDMAExecOp(builder, outputDmaNameLoc, execOp.getToken(), bufferToSpill, newBufferResult);

        // Update users
        bufferToSpill.replaceUsesWithIf(outputDmaExecOp.getBodyResults()[0], [&](mlir::OpOperand& opOperand) {
            return mlir::isa<mlir::async::AwaitOp>(opOperand.getOwner());
        });

        // Update dependencies map and get new operation index
        _depsInfo.insertNewExecOpToDepsMap(outputDmaExecOp);

        // Update dependency
        _depsInfo.addDependency(execOp, outputDmaExecOp);

        // Record the NNDMA operation
        _funcOutputDMAInMainFunction.push_back(_depsInfo.getIndex(outputDmaExecOp));
    }
}

// Operation set: [0, 1, 2, 3, 4, 5, 6, 7]
// Operation 3 is CMX2DDR DMA operation, it should be used as a cut point
// Return [3, 7]
SmallVector<size_t> AsyncRegionOutliningSplitter::findCutPoints() {
    auto isCMX2DDRDataOp = [&](size_t opIndex) -> bool {
        auto execOp = _depsInfo.getExecuteOpAtIndex(opIndex);
        if (_depsInfo.getConsumerOps(opIndex).empty()) {
            return false;
        }
        if (vpux::getExecutorType(execOp) != VPU::ExecutorKind::DMA_NN) {
            return false;
        }
        if (auto dmaTask = vpux::getDmaTypeOp(execOp)) {
            // DMA from NN_CMX to DDR
            auto ddrMemKind = VPU::MemoryKind::DDR;
            auto dstMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getOutput().getType()).getMemoryKind();
            return ddrMemKind == dstMemSpace;
        }
        return false;
    };

    auto execOpCount = _depsInfo.getExecOpCount();
    SmallVector<size_t> cutPoints;
    for (size_t opIdx = 0; opIdx < execOpCount; ++opIdx) {
        if (isCMX2DDRDataOp(opIdx)) {
            cutPoints.push_back(opIdx);
        }
        // simple algorithm to create 2 sub functions
        if (cutPoints.size() > 0) {
            break;
        }
    }
    cutPoints.push_back(execOpCount - 1);
    return cutPoints;
}

// Operation set: [0, 1, 2, 3, 4, 5, 6, 7, 8] was split into two instances.
// operation 8 is the main function DMA, it remains in the main function
// subFuncInstances = [[0, 1, 2, 3], [4, 5, 6, 7]]
SmallVector<SubFuncSlice> AsyncRegionOutliningSplitter::splitOpsInMainFunToSubFuncs() {
    SmallVector<SubFuncSlice> subFuncInstances;
    auto cutPoints = findCutPoints();
    if (cutPoints.empty()) {
        return {};
    }
    size_t start = 0;
    for (auto cutPoint : cutPoints) {
        SubFuncSlice slice;
        for (auto opIndex = start; opIndex <= cutPoint; ++opIndex) {
            if (std::find(_funcOutputDMAInMainFunction.begin(), _funcOutputDMAInMainFunction.end(), opIndex) ==
                _funcOutputDMAInMainFunction.end()) {
                slice.push_back(opIndex);
            }
        }
        start = cutPoint + 1;
        subFuncInstances.push_back(slice);
    }
    return subFuncInstances;
}

// After splitOpsInMainFunToSubFuncs, subFuncInstances = [[0, 1, 2, 3], [4, 5, 6, 7]]
// Initialize _outliningInstancesIndex from subFuncInstances
// The operation set is:
//             [inputs] [operations]    [outputs]
// instance_1: []       [0, 1, 2, 3]    []
// instance_2: []       [4, 5, 6, 7]    []
// main_func:  []       [8]             []
void AsyncRegionOutliningSplitter::initOutliningInstanceIndex(ArrayRef<SubFuncSlice> subFuncInstances) {
    for (const auto& subFuncInstance : subFuncInstances) {
        OutliningInstanceIndex instanceIndex;
        instanceIndex.storedOperationsIndex.insert(subFuncInstance.begin(), subFuncInstance.end());
        _outliningInstancesIndex.push_back(instanceIndex);
    }
}

// Before processOutputOps(), the operation set is:
//             [inputs] [operations]    [outputs]
// instance_1: []       [0, 1, 2, 3]    []
// instance_2: []       [4, 5, 6, 7]    []
// main_func:  []       [8]             []
// _depsInfo.getConsumerOps(2) = [4], which means operation 2 is used by operation 4
// operation 4 is in the second instance, so operation 2 should be added to the output set
// Same for _depsInfo.getConsumerOps(7) = [8]
// After processOutputOps(), the operation set is:
//             [inputs] [operations]    [outputs]
// instance_1: []       [0, 1, 2, 3]    [2]
// instance_2: []       [4, 5, 6, 7]    [7]
// main_func:  []       [8]             []
void AsyncRegionOutliningSplitter::processOutputOps() {
    for (size_t currIdx = 0; currIdx < _outliningInstancesIndex.size(); ++currIdx) {
        const auto& currentInstanceOperations = _outliningInstancesIndex[currIdx].storedOperationsIndex;

        auto hasConsumersInCurrentInstance = [&](size_t opIndex) -> bool {
            auto consumerOpsIndex = _depsInfo.getConsumerOps(opIndex);
            auto commonConsumerOpsIndex = findCommonElements(consumerOpsIndex, currentInstanceOperations);
            return !commonConsumerOpsIndex.empty();
        };

        auto isOpInOtherInstances = [&](const auto& opIndex) -> bool {
            for (size_t otherIdx = currIdx + 1; otherIdx < _outliningInstancesIndex.size(); ++otherIdx) {
                if (llvm::is_contained(_funcOutputDMAInMainFunction, opIndex) ||
                    llvm::is_contained(_outliningInstancesIndex[otherIdx].storedOperationsIndex, opIndex)) {
                    return true;
                }
            }
            return false;
        };

        auto isOuterUserOp = [&](const auto& userOp) -> bool {
            if (auto userExecOp = mlir::dyn_cast<mlir::async::ExecuteOp>(userOp)) {
                auto userExecOpIndex = _depsInfo.getIndex(userExecOp);
                return isOpInOtherInstances(userExecOpIndex);
            } else if (auto awaitOp = mlir::dyn_cast<mlir::async::AwaitOp>(userOp)) {
                return awaitOp.getResult() != nullptr;
            } else {
                VPUX_THROW("Invalid user operation");
            }
        };

        auto hasOuterUse = [&](auto& execOp) -> bool {
            for (const auto& result : execOp.getBodyResults()) {
                for (auto& use : result.getUses()) {
                    if (isOuterUserOp(use.getOwner())) {
                        return true;
                    }
                }
            }
            return false;
        };

        auto addSpillOpsIntoInstances = [&](mlir::Value spillbuffer, size_t opIndex) {
            if (_spillingBufferAndWriteOpMap.count(spillbuffer) == 0) {
                return;
            }
            // For example, operation 2 is in CMX, it was consumed by
            // operation 4, which is in the second instance. It needs
            // inserting spilling DMA operation. While operation 7
            // is already in DDR, so it does not need to insert
            // Before addSpillOpsIntoInstances(), the operation set is:
            //             [inputs] [operations]    [outputs]
            // instance_1: []       [0, 1, 2, 3]    [2]
            // instance_2: []       [4, 5, 6, 7]    [7]
            // main_func:  []       [8]             []

            // After insertSpillDmaOps() and addSpillOpsIntoInstances(),
            // operation 9 is the spill write operation to bring output buffer of operation 2
            // to DDR. operation 9 should be added to the output set of current instance
            // and operation 2 should be removed from the output set. operation 2 used
            // to have operation 4 as its consumer, which requires a spill read operation
            // to load buffer from DDR to CMX. operation 10 is the spill read operation
            // which should be added to operation set of the second instance
            // At the same time, add operation 2 into deps set of second instance
            // as a record for further use
            //             [deps] [inputs] [operations]     [outputs]
            // instance_1: []     []       [0, 1, 2, 3]     [9]
            // instance_2: [2]    []       [4, 5, 6, 7, 10] [7]
            // main_func:  []     []       [8]              []

            // process spillWriteExecOp follow above example
            auto spillWriteExecOp = _spillingBufferAndWriteOpMap[spillbuffer];
            const auto spillWriteExecOpIndex = _depsInfo.getIndex(spillWriteExecOp);
            auto& currentInstance = _outliningInstancesIndex[currIdx];
            currentInstance.storedOutputsIndex.insert(spillWriteExecOpIndex);
            currentInstance.storedOperationsIndex.insert(spillWriteExecOpIndex);
            currentInstance.storedOutputsIndex.erase(opIndex);

            // process spillReadExecOp follow above example
            for (auto userExecOpIndex : _spillingBufferAndUserExecOpMap[spillbuffer]) {
                const auto spillReadExecOpIndices = _spillingUserExecOpAndReadOpMap[userExecOpIndex];
                for (auto spillReadExecOpIndex : spillReadExecOpIndices) {
                    // find which instance the userExecOpIndex belongs to
                    for (auto& instance : _outliningInstancesIndex) {
                        if (instance.storedOperationsIndex.count(userExecOpIndex) == 0) {
                            continue;
                        }
                        // Add spill read operation into instance operation set
                        // where its userExecOpIndex belongs to
                        instance.storedOperationsIndex.insert(spillReadExecOpIndex);
                        instance.storedFuncDepIndex.insert(opIndex);
                    }
                }
            }
        };

        for (const auto& opIndex : llvm::make_early_inc_range(currentInstanceOperations)) {
            auto execOp = _depsInfo.getExecuteOpAtIndex(opIndex);
            // If the operation has users from other instances or users are the main function result
            // this operation should be added to output set
            if (!hasConsumersInCurrentInstance(opIndex) || hasOuterUse(execOp)) {
                _outliningInstancesIndex[currIdx].storedOutputsIndex.insert(opIndex);
                // Insert the spilling DMA operations for sub-function
                // output buffer if it is in CMX
                insertSpillDmaOps(execOp);
                for (auto spillbuffer : execOp.getBodyResults()) {
                    addSpillOpsIntoInstances(spillbuffer, opIndex);
                }
            }
        }
    }
}

// After processOutputOps(), the operation set is:
//             [inputs] [operations]    [outputs]
// instance_1: []       [0, 1, 2, 3]    [2]
// instance_2: []       [4, 5, 6, 7]    [7]
// main_func:  []       [8]             []
// operation 2 is used by operation 4, which is in the second instance
// operation 7 is used by operation 8, which is in the main function
// So operation 2 should be added to the input set of the second instance
// and operation 7 should be added to the input set of the main function
// After processInputsAndDepOps(), the operation set is:
//             [inputs] [operations]    [outputs]
// instance_1: []       [0, 1, 2, 3]    [2]
// instance_2: [2]      [4, 5, 6, 7]    [7]
// main_func:  [7]      [8]             []
void AsyncRegionOutliningSplitter::processInputsAndDepOps() {
    for (size_t currIdx = 0; currIdx < _outliningInstancesIndex.size(); ++currIdx) {
        auto& currentInstance = _outliningInstancesIndex[currIdx];
        // If current function's output or consumer exist in other functions'
        // operation set, insert this output to input and depOp set of other functions
        for (const auto& outputIndex : currentInstance.storedOutputsIndex) {
            auto consumerOpsIndex = _depsInfo.getConsumerOps(outputIndex);
            for (size_t otherIdx = currIdx + 1; otherIdx < _outliningInstancesIndex.size(); ++otherIdx) {
                auto& otherInstance = _outliningInstancesIndex[otherIdx];
                auto hasConsumer = findCommonElements(consumerOpsIndex, otherInstance.storedOperationsIndex).size() > 0;
                auto hasOutput = otherInstance.storedOperationsIndex.erase(outputIndex) > 0;
                if (hasConsumer || hasOutput) {
                    otherInstance.storedInputsIndex.insert(outputIndex);
                    otherInstance.storedFuncDepIndex.insert(outputIndex);
                }
            }
        }
    }
}

// If operation in current instance purely served as a token and
// are not used as a operand, remove it from the token list
// before:
//    %token, %bodyResults = async.execute [%token, %token_2, %pureDepToken]
// after:
//    %token, %bodyResults = async.execute [%token, %token_2]
void AsyncRegionOutliningSplitter::updateTokens() {
    for (size_t currIdx = 0; currIdx < _outliningInstancesIndex.size(); ++currIdx) {
        auto& currentInstance = _outliningInstancesIndex[currIdx];
        for (auto opIndex : currentInstance.storedFuncDepIndex) {
            auto depOp = _depsInfo.getExecuteOpAtIndex(opIndex);
            auto depOpToken = depOp.getToken();
            auto consumerOpsIndex = _depsInfo.getConsumerOps(opIndex);
            auto commonOpsIndex = findCommonElements(consumerOpsIndex, currentInstance.storedOperationsIndex);
            for (auto consumerIndex : commonOpsIndex) {
                auto consumerOp = _depsInfo.getExecuteOpAtIndex(consumerIndex);
                auto depOpTokenPtr = llvm::find(consumerOp.getDependencies(), depOpToken);
                if (depOpTokenPtr == consumerOp.getDependencies().end()) {
                    continue;
                }
                auto depOpTokenIndex =
                        static_cast<unsigned>(std::distance(consumerOp.getDependencies().begin(), depOpTokenPtr));
                consumerOp.getDependenciesMutable().erase(depOpTokenIndex, 1);
            }
        }
    }
}

// Process the operands of ExecuteOp, find the operands which are defined outside the async region
// If operand is passed from function argument, add it to input set
// If operand is defined outside async region, it might be Const::DeclareOp or allocOps, add it to operation set
// Example:
// func.func @outlining(%arg0: ...) {
//     %2 = VPURT.AllocDistributed
//     %token_2, %bodyResults_3 = async.execute [%token] (%bodyResults as %arg3) {
//         %3 = VPUIP.NNDMA input(%arg0)                -> %arg0 is passed from function argument
//         %4 = VPUIP.ViewOp %2                         -> %2 is defined outside async region
//         %5 = VPUIP.NCEClusterTask
//                 input(%arg3: ...)
//                 weights(%3: ... )
//                 parent_input(%arg3: ... )
//                 parent_output(%4: ... )
//                 outputs(%4: ... )
//         async.yield %5
//     }
// }
void AsyncRegionOutliningSplitter::processOperands(mlir::async::ExecuteOp execOp, OpOrderedSet& storedOperations,
                                                   ValueOrderedSet& storedInputs) {
    auto* bodyBlock = execOp.getBody();
    for (auto& op : bodyBlock->getOperations()) {
        for (auto operand : op.getOperands()) {
            if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
                // If operand is function argument, add it to input set
                if (blockArg.getOwner() != bodyBlock) {
                    storedInputs.insert(blockArg);
                }
            } else {
                // If operand is defined outside async region, it might be Const::DeclareOp or allocOps
                auto parentOp = operand.getDefiningOp();
                if (parentOp && parentOp->getParentRegion() != op.getParentRegion()) {
                    storedOperations.insert(parentOp);
                }
            }
        }
    }
}

// Create outlining instances from the outlining instance index to mlir::Operation
SmallVector<OutliningInstance> AsyncRegionOutliningSplitter::createOutliningInstances() {
    // Update IR in main function before creating outlining instances
    // Remove operations that are purely served as dep token
    updateTokens();
    SmallVector<OutliningInstance> irSliceInstances;
    for (const auto& instanceIndex : _outliningInstancesIndex) {
        OpOrderedSet storedOperations;
        ValueOrderedSet storedInputs;
        ValueOrderedSet storedOutputs;

        auto transferFromIndexToStorage = [&](const auto& opIndices, auto& storage) {
            for (const auto& opIndex : opIndices) {
                auto execOp = _depsInfo.getExecuteOpAtIndex(opIndex);
                auto results = execOp.getBodyResults();
                storage.insert(results.begin(), results.end());
            }
        };

        // Process operations' operands
        for (const auto& opIndex : instanceIndex.storedOperationsIndex) {
            auto execOp = _depsInfo.getExecuteOpAtIndex(opIndex);
            storedOperations.insert(execOp);
            processOperands(execOp, storedOperations, storedInputs);
        }

        transferFromIndexToStorage(instanceIndex.storedInputsIndex, storedInputs);
        transferFromIndexToStorage(instanceIndex.storedOutputsIndex, storedOutputs);

        // Create OutliningInstance from the storage
        auto currentSlice = IRSlice();
        currentSlice.operations.insert(currentSlice.operations.end(), storedOperations.begin(), storedOperations.end());
        currentSlice.inputs.insert(currentSlice.inputs.end(), storedInputs.begin(), storedInputs.end());
        currentSlice.outputs.insert(currentSlice.outputs.end(), storedOutputs.begin(), storedOutputs.end());

        if (currentSlice.operations.empty() || currentSlice.inputs.empty() || currentSlice.outputs.empty()) {
            VPUX_THROW("At least one instance has no outputs values, which results in an empty function");
            return {};
        }

        irSliceInstances.push_back(OutliningInstance{std::move(currentSlice)});
    }
    return irSliceInstances;
}

SmallVector<OutliningInstance> AsyncRegionOutliningSplitter::getOutliningInstances(mlir::func::FuncOp mainFuncOp) {
    _depsInfo.buildConsMap();
    if (_depsInfo.getExecOpCount() <= _minOpsInBlock) {
        _log.trace("Cannot perform outlining. The number of operations is less than {0}", _minOpsInBlock);
        return {};
    }
    // Step 1:
    // Find all operations which remain in main function
    // NNDMA for func output remain in the main function by inserting spill DMA
    auto buffersNeedSpillDMA = findDmaForMainFuncOutputBuffer(mainFuncOp);

    // Step 2:
    // Split operations in main function to sub functions
    auto subFuncInstances = splitOpsInMainFunToSubFuncs();
    if (subFuncInstances.size() <= 1) {
        _log.trace("Cannot perform outlining. The number of outlining instances is 1");
        return {};
    }

    // Step 3:
    // a. Insert Spill Dma for main function output
    insertDmaForNetResults(buffersNeedSpillDMA);
    // b. Process the input/output/Ops for each outlining instance
    initOutliningInstanceIndex(subFuncInstances);
    processOutputOps();
    processInputsAndDepOps();

    // Step 4:
    // Create outlining instances from the outlining instance index
    return createOutliningInstances();
}

}  // namespace

vpux::VPUIP::FunctionOutlinerAsyncRegion::FunctionOutlinerAsyncRegion(size_t minOpsInBlock, AsyncDepsInfo& depsInfo,
                                                                      Logger log)
        : _minOpsInBlock(minOpsInBlock), _depsInfo(depsInfo), _log(log) {
}

SmallVector<OutliningInstance> vpux::VPUIP::FunctionOutlinerAsyncRegion::getOutliningTargets(
        mlir::func::FuncOp mainFunction) {
    _log.debug("Searching for outlining targets in async dialect");
    AsyncRegionOutliningSplitter asyncRegionSplitter(_minOpsInBlock, _depsInfo, _log);
    const auto outliningInstances = asyncRegionSplitter.getOutliningInstances(mainFunction);
    return outliningInstances;
}
