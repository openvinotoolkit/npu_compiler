//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <utility>

namespace vpux::HostExec {
#define GEN_PASS_DECL_OUTLINEDIMOPERATIONS
#define GEN_PASS_DEF_OUTLINEDIMOPERATIONS
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

// Struct for collecting all tensor/arith operations that have to be outlined
struct DiscoverySet {
    std::unordered_set<mlir::Operation*> discoveredOps;

    // Use BFS algorithm to traverse IR from startOp upwards and collect ops to discoveredOps
    void traverseIR(mlir::Operation* startOp) {
        std::queue<mlir::Operation*> opsToVisit;
        opsToVisit.push(startOp);
        discoveredOps.insert(startOp);
        while (!opsToVisit.empty()) {
            auto curOp = opsToVisit.front();
            opsToVisit.pop();
            for (const auto& curOpOperand : curOp->getOperands()) {
                if (curOpOperand.getDefiningOp() == nullptr) {
                    continue;
                }
                const auto [_, wasInserted] = discoveredOps.insert(curOpOperand.getDefiningOp());
                if (wasInserted) {
                    opsToVisit.push(curOpOperand.getDefiningOp());
                }
            }
        }
    }
};

class OutlineDimOperationsPass final : public HostExec::impl::OutlineDimOperationsBase<OutlineDimOperationsPass> {
    // Collect all shape operations to outline starting from shapeTensorOps
    std::vector<mlir::Operation*> collectOperationsToOutline(mlir::ArrayRef<mlir::Value> shapeTensorOps) const;

    // Collect all operands of returnOp that are shape tensor ops (tensor.from_elements) and constract a bitVector
    // of those operations that are required to be removed from the main return op.
    SmallVector<mlir::Value> collectShapeTensorOps(mlir::func::ReturnOp returnOp,
                                                   llvm::BitVector& bitVectorOfOpsToRemove) const;

    // Create a new func op
    mlir::func::FuncOp createNewFuncOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::Location loc,
                                       const std::string& funcName, mlir::ArrayRef<mlir::Type> inputTypes,
                                       mlir::ArrayRef<mlir::Type> outputTypes) const;

    // Remove outputs of funcOp's return op with bitVectorOfOpsToRemove and update funcOp's type and netInfo
    void removeFuncOutputs(mlir::func::FuncOp funcOp, const llvm::BitVector& bitVectorOfOpsToRemove,
                           mlir::func::ReturnOp returnOp, net::NetworkInfoOp netInfo) const;

    // Move operations from opsToMove to a new outlineFunc
    void outlineOperations(mlir::OpBuilder& builder, mlir::func::FuncOp mainFunc, mlir::func::FuncOp outlineFunc,
                           const std::vector<mlir::Operation*>& opsToMove,
                           mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap) const;

    // Update funcOp's signature with provided shapeTensorValues
    void updateOutlineFuncType(mlir::func::FuncOp funcOp, mlir::func::ReturnOp returnOp,
                               const SmallVector<mlir::Value>& shapeTensorValues,
                               mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap) const;

public:
    explicit OutlineDimOperationsPass(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    Logger _log;
};

std::vector<mlir::Operation*> OutlineDimOperationsPass::collectOperationsToOutline(
        mlir::ArrayRef<mlir::Value> shapeTensorOps) const {
    DiscoverySet discoveryObj;
    for (const auto& fromElemOpVal : shapeTensorOps) {
        discoveryObj.traverseIR(fromElemOpVal.getDefiningOp());
    }

    std::vector<mlir::Operation*> opsToMove(discoveryObj.discoveredOps.begin(), discoveryObj.discoveredOps.end());
    // Sort opsToMove vector to topological order for proper mapping and cloning them to a new func
    std::sort(opsToMove.begin(), opsToMove.end(), [](mlir::Operation* lhsOp, mlir::Operation* rhsOp) {
        return lhsOp->isBeforeInBlock(rhsOp);
    });

    return opsToMove;
}

SmallVector<mlir::Value> OutlineDimOperationsPass::collectShapeTensorOps(
        mlir::func::ReturnOp returnOp, llvm::BitVector& bitVectorOfOpsToRemove) const {
    SmallVector<mlir::Value> shapeTensorValues;
    // Collect all tensor.from_elements ops from ReturnOp operands of the main func
    for (auto [idx, returnOperand] : returnOp->getOpOperands() | indexed) {
        if (mlir::isa_and_nonnull<mlir::tensor::FromElementsOp, mlir::arith::ConstantOp>(
                    returnOperand.get().getDefiningOp())) {
            bitVectorOfOpsToRemove.set(idx);
            shapeTensorValues.push_back(returnOperand.get());
        }
    }

    return shapeTensorValues;
}

mlir::func::FuncOp OutlineDimOperationsPass::createNewFuncOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                                             mlir::Location loc, const std::string& funcName,
                                                             mlir::ArrayRef<mlir::Type> inputTypes,
                                                             mlir::ArrayRef<mlir::Type> outputTypes) const {
    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, outputTypes);
    return builder.create<mlir::func::FuncOp>(loc, funcName, funcType);
}

void OutlineDimOperationsPass::removeFuncOutputs(mlir::func::FuncOp funcOp,
                                                 const llvm::BitVector& bitVectorOfOpsToRemove,
                                                 mlir::func::ReturnOp returnOp, net::NetworkInfoOp netInfo) const {
    // Construct a new func type from the old one but without results that should be removed
    // and update the func with the new type
    auto newMainFuncType = funcOp.getFunctionType().getWithoutArgsAndResults({}, bitVectorOfOpsToRemove);
    funcOp.setFunctionType(newMainFuncType);

    // Update NetInfo and ReturnOp's operands of the func
    auto netInfoOutputs = netInfo.getOutputsDataInfo();
    returnOp->eraseOperands(bitVectorOfOpsToRemove);

    for (const auto idx : bitVectorOfOpsToRemove.set_bits()) {
        netInfoOutputs[idx]->erase();  // here we erase an Operation itself, not a SmallVector's element
    }
}

void OutlineDimOperationsPass::outlineOperations(mlir::OpBuilder& builder, mlir::func::FuncOp mainFunc,
                                                 mlir::func::FuncOp outlineFunc,
                                                 const std::vector<mlir::Operation*>& opsToMove,
                                                 mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap) const {
    for (size_t i = 0; i < mainFunc.getNumArguments(); ++i) {
        oldToNewMap[mainFunc.getArgument(i)] = outlineFunc.getArgument(i);
    }
    for (const auto& op : opsToMove) {
        mlir::IRMapping mapper;
        for (auto operand : op->getOperands()) {
            mapper.map(operand, oldToNewMap[operand]);
        }
        auto clonedOp = builder.clone(*op, mapper);

        clonedOp->setLoc(appendLoc(clonedOp->getLoc(), "outline"));
        for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
            oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
        }
    }
}

void OutlineDimOperationsPass::updateOutlineFuncType(mlir::func::FuncOp funcOp, mlir::func::ReturnOp returnOp,
                                                     const SmallVector<mlir::Value>& shapeTensorValues,
                                                     mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap) const {
    // Update ReturnOp and type of funcOp
    SmallVector<mlir::Value> outlinedFromElemOps;
    SmallVector<mlir::Type> outlineResultTypes(funcOp.getResultTypes());
    for (const auto& op : shapeTensorValues) {
        auto outlinedFromElemOp = oldToNewMap[op];
        outlinedFromElemOps.push_back(outlinedFromElemOp);
        outlineResultTypes.push_back(outlinedFromElemOp.getType());
    }

    funcOp.setFunctionType(
            mlir::FunctionType::get(funcOp->getContext(), funcOp.getArgumentTypes(), outlineResultTypes));
    returnOp.getOperandsMutable().append(outlinedFromElemOps);
}

void OutlineDimOperationsPass::safeRunOnModule() {
    auto module = getOperation();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFunc;
    net::NetworkInfoOp::getFromModule(module, netInfo, mainFunc);

    // Find ReturnOp in the main func
    auto mainReturnOp = findReturnOp(mainFunc);

    // Fill bitVectorOfOpsToRemove with 1 for those outputs that we want to remove from the main func later
    llvm::BitVector bitVectorOfOpsToRemove(mainFunc.getResultTypes().size());
    auto shapeTensorOps = collectShapeTensorOps(mainReturnOp, bitVectorOfOpsToRemove);
    // If shapeTensorOps is empty it means that there're no return operations with dynamic shapes
    if (shapeTensorOps.empty()) {
        return;
    }

    // Traverse the IR from all tensor.from_elements ops to their inputs
    // and collect all tensor and arith operations
    auto opsToMove = collectOperationsToOutline(shapeTensorOps);

    // An unreified IE op in the output shape computation means the shape chain was not
    // fully resolved before outlining. Such ops must implement
    // ReifyRankedShapedTypeOpInterface to be handled by resolveShapedTypeResultDims.
    llvm::SetVector<mlir::OperationName> unreifiedOpTypes;
    for (auto* op : opsToMove) {
        if (mlir::isa_and_nonnull<IE::IEDialect>(op->getDialect())) {
            unreifiedOpTypes.insert(op->getName());
        }
    }
    if (!unreifiedOpTypes.empty()) {
        auto err = mlir::emitError(mainFunc.getLoc());
        printTo(err, "Found {0} unreified IE op type(s) in @output_shape chain: ", unreifiedOpTypes.size());
        llvm::interleaveComma(unreifiedOpTypes, err);
        signalPassFailure();
        return;
    }

    mlir::OpBuilder moduleBuilder(module);
    moduleBuilder.setInsertionPoint(mainFunc);

    // Create a new outline func (with empty body, same block arguments and empty return types)
    const auto funcLoc = appendLoc(mainFunc.getLoc(), "output_shape");
    auto funcName = vpux::formatv("output_shape");
    auto outlineFunc = createNewFuncOp(moduleBuilder, mainFunc->getContext(), funcLoc, funcName.str(),
                                       mainFunc.getArgumentTypes(), {});
    config::setPureHostCompileFuncAttribute(outlineFunc);

    // Set a block to the new func and create a ReturnOp
    auto outlineBodyBlock = outlineFunc.addEntryBlock();
    auto outlineBlockBuilder = mlir::OpBuilder::atBlockEnd(outlineBodyBlock);
    auto outlineReturnOp = outlineBlockBuilder.create<mlir::func::ReturnOp>(appendLoc(outlineFunc.getLoc(), "return"));

    // Remove tensor shape ops from the main func return type, update ReturnOp and NetworkInfo
    removeFuncOutputs(mainFunc, bitVectorOfOpsToRemove, mainReturnOp, netInfo);

    // Map operations from opsToMove to a new block args and move them to the new func
    outlineBlockBuilder.setInsertionPointToStart(outlineBodyBlock);
    mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
    outlineOperations(outlineBlockBuilder, mainFunc, outlineFunc, opsToMove, oldToNewMap);

    // Update outlined func return type
    updateOutlineFuncType(outlineFunc, outlineReturnOp, shapeTensorOps, oldToNewMap);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::HostExec::createOutlineDimOperationsPass(Logger log) {
    return std::make_unique<OutlineDimOperationsPass>(std::move(log));
}
