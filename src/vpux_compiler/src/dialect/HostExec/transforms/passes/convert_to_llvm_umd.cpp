//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#ifdef _WIN32
#define NO_MINMAX
#endif

#include <cmath>

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

namespace vpux::HostExec {
#define GEN_PASS_DECL_CONVERTTOLLVMUMDCALLS
#define GEN_PASS_DEF_CONVERTTOLLVMUMDCALLS
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;
using namespace vpux::HostExec;

namespace {

struct CommandListIndexState {
    // Stores commandlist index
    uint32_t resetIndex = 0;

    // Stores the lastest pointer_to_integer and pointers of command list**.
    mlir::Value lastResult = nullptr;
    mlir::Value lastResultPtr = nullptr;

    // Stores steps size(8). Which is a byte size of pointer type
    mlir::LLVM::ConstantOp stepSizeInByte = nullptr;

    // Stores the last CallOp of submit command list
    // A fence pointer and an event pointer needs to be updated
    // with given pointers
    mlir::LLVM::CallOp lastSubmitCommandListCallOp = nullptr;

    // The max number of inputs for all sub graphs.
    // This is used to allocate an array for input pointers
    uint32_t maxNumInputs = 0;

    // The max number of outputs for all sub graphs
    // This is used to allocate an array for input pointers
    uint32_t maxNumOutputs = 0;

    // Buffer to stores input buffer pointers
    mlir::LLVM::AllocaOp inputs = nullptr;

    // Buffer to stores output buffer pointers
    mlir::LLVM::AllocaOp outputs = nullptr;

    bool commandListGroupStarted = false;

    void initialize(mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp) {
        mlir::OpBuilder builder(funcOp);
        auto& entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);

        stepSizeInByte = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64), 8);

        obtainNumMaxNumArguments(moduleOp);

        auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto constMaxNumInputs = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(),
                                                                        builder.getIntegerType(32), maxNumInputs);
        auto constMaxNumOutputs = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(),
                                                                         builder.getIntegerType(32), maxNumOutputs);
        inputs = builder.create<mlir::LLVM::AllocaOp>(builder.getUnknownLoc(), voidPtrType, voidPtrType,
                                                      constMaxNumInputs);
        outputs = builder.create<mlir::LLVM::AllocaOp>(builder.getUnknownLoc(), voidPtrType, voidPtrType,
                                                       constMaxNumOutputs);
    }

    // Update pointers of commandlist**
    mlir::Value updateCommandListIndex(mlir::OpBuilder& builder, mlir::func::FuncOp& funcOp) {
        auto loc = builder.getUnknownLoc();
        auto numArgs = funcOp.getNumArguments();
        auto cmdList = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));
        mlir::Type i64Type = builder.getI64Type();
        auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        if (resetIndex++ > 0) {
            // Increase commandlist pointer by 8 (size of pointer type)
            mlir::LLVM::AddOp addOp = builder.create<mlir::LLVM::AddOp>(loc, i64Type, lastResult, stepSizeInByte);
            lastResult = addOp.getResult();
            lastResultPtr = builder.create<mlir::LLVM::IntToPtrOp>(loc, voidPtrType, lastResult);

            return lastResultPtr;
        } else {
            // For the first commandlist pointer, no need of increasing commandlist pointer
            lastResult = builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Type, cmdList);
            lastResultPtr = builder.create<mlir::LLVM::IntToPtrOp>(loc, voidPtrType, lastResult);

            return cmdList;
        }
    }

    // Returns address of commandlist**
    mlir::Value getCommandList(mlir::func::FuncOp& funcOp) {
        auto numArgs = funcOp.getNumArguments();
        auto cmdList = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));

        // For the first subgraph, no need of commandlist pointer update is requried
        return (resetIndex > 1) ? lastResultPtr : cmdList;
    }

    // Update inference execution sync params (e.g., event or fence) for the last commandlist submission
    void finalizeCommandListIndex(mlir::ModuleOp& module, mlir::MLIRContext* ctx, mlir::func::FuncOp& funcOp) {
        // fence or event needs to be set to the last command list for host side synchronization.
        if (lastSubmitCommandListCallOp != nullptr) {
            auto numArgs = funcOp.getNumArguments();
            auto fence = funcOp.getArgument(GET_ARG_INDEX_COMMAND_FENCE(numArgs));
            auto event = funcOp.getArgument(GET_ARG_INDEX_COMMAND_EVENT(numArgs));

            constexpr uint32_t argIndexFence = 2;
            constexpr uint32_t argIndexEvent = 3;

            lastSubmitCommandListCallOp.setOperand(argIndexFence, fence);
            lastSubmitCommandListCallOp.setOperand(argIndexEvent, event);
        }

        // Add an atribute "number of subgraphs" to the main module
        mlir::Type i64Type = mlir::IntegerType::get(ctx, 64);
        mlir::IntegerAttr numSubGraphs = mlir::IntegerAttr::get(i64Type, resetIndex);

        module->setAttr(HOST_EXEC_NUM_SUBGRAPH_ATTR_NAME, numSubGraphs);
    }

    // Obtain the max inputs and outputs,
    void obtainNumMaxNumArguments(mlir::ModuleOp moduleOp) {
        maxNumInputs = 0;
        maxNumOutputs = 0;

        // Function signatures are defined as FuncOp under BinaryOp
        for (auto binaryOp : moduleOp.getOps<HostExec::BinaryOp>()) {
            for (auto func : binaryOp.getOps<mlir::func::FuncOp>()) {
                auto resultCount = func.getNumResults();

                // Note that function arguments stores both inputs and results.
                const auto inputCount = func.getNumArguments() - resultCount;

                maxNumOutputs = std::max(maxNumOutputs, resultCount);
                maxNumInputs = std::max(maxNumInputs, inputCount);
            }
        }
    }
};

// @brief Add arguments (e.g., commandlist, command queue) to an entry function for L0 function calls
void addFuncParamsForUmdFuncCall(mlir::func::FuncOp& funcOp) {
    // Update the function's return type to NoneType
    auto funcType = funcOp.getFunctionType();
    auto ctx = funcOp.getContext();

    SmallVector<mlir::Type, 8> newInputTypes;
    for (auto input : funcType.getInputs()) {
        newInputTypes.push_back(input);
    }

    // add handle to command list
    mlir::Type contextHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type deviceHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type ddiTableHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type commandListHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type commandListCountType = mlir::IntegerType::get(ctx, 64);
    mlir::Type commandQueueHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type fenceHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type eventHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);

    newInputTypes.push_back(contextHandlePtrType);
    newInputTypes.push_back(deviceHandlePtrType);
    newInputTypes.push_back(ddiTableHandlePtrType);
    newInputTypes.push_back(commandListHandlePtrType);
    newInputTypes.push_back(commandListCountType);
    newInputTypes.push_back(commandQueueHandlePtrType);
    newInputTypes.push_back(fenceHandlePtrType);
    newInputTypes.push_back(eventHandlePtrType);

    for (auto& block : funcOp.getBody()) {
        // Find the current terminator
        if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator())) {
            // Replace the return operation with a new one that has no arguments
            mlir::OpBuilder builder(returnOp);
            builder.create<mlir::func::ReturnOp>(returnOp.getLoc());
            returnOp.erase();
        }
    }

    auto newFuncType =
            mlir::FunctionType::get(funcOp.getContext(), newInputTypes, mlir::TypeRange{});  // No return types
    funcOp.setType(newFuncType);
    auto& entryBlock = funcOp.getBody().front();
    entryBlock.addArgument(contextHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(deviceHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(ddiTableHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(commandListHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(commandListCountType, funcOp->getLoc());
    entryBlock.addArgument(commandQueueHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(fenceHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(eventHandlePtrType, funcOp->getLoc());
}

//@brief create function call op to reset a command list
void createL0ResetCommandList(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                              CommandListIndexState& cmdListIndexState) {
    if (cmdListIndexState.commandListGroupStarted) {
        // Skip redudant command list reset
        // When there are multiple ExecutOp in a scf::for, there will be multiple CreateGroupOps.
        // Inference exectuion will be recorded in one command list.
        // A barrier will be added between ExecuteOp.

        return;
    }

    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    auto moduleOp = getModuleOp(op);
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));

    cmdListIndexState.commandListGroupStarted = true;
    auto cmdList = cmdListIndexState.updateCommandListIndex(rewriter, funcOp);
    createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_reset_commandlist", {cmdList}, returnType);
}

//@brief create function call op to submit command list
void createSubmitCommandList(mlir::OpBuilder& builder, mlir::ModuleOp& moduleOp,
                             CommandListIndexState& cmdListIndexState) {
    if (cmdListIndexState.commandListGroupStarted == false) {
        // ignore redudandant await/await_all op
        return;
    }

    cmdListIndexState.commandListGroupStarted = false;

    mlir::MLIRContext* ctx = builder.getContext();
    vpux::net::NetworkInfoOp netInfo;
    mlir::func::FuncOp funcOp;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);

    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));

    // Create the LLVM::mlir.constant operation representing nullptr
    auto elementPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(builder.getUnknownLoc(), elementPtrType);

    // Host side synchronization will be required for the last command list.
    // The last two arguments will be updated in updateCommandListIndex later if required.
    auto numArgs = funcOp.getNumArguments();
    auto cmdQueue = funcOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));

    auto cmdList = cmdListIndexState.getCommandList(funcOp);
    auto callOp = createLLVMFuncCallOp(builder, moduleOp, "npu_level_zero_submit_commandlist",
                                       {cmdList, cmdQueue, nullPtr, nullPtr}, returnType);

    // Mark this call op as the last one to update 3rd and 4th arguments of the last submit command list
    cmdListIndexState.lastSubmitCommandListCallOp = callOp;
}

int64_t getElementByteSize(mlir::Value sourceDesc) {
    int64_t bytes = 2;

    if (auto op = sourceDesc.getDefiningOp<mlir::memref::SubViewOp>()) {
        auto memRefType = op.getSource().getType();
        auto elementType = memRefType.getElementType();
        return (elementType.getIntOrFloatBitWidth() + 7) / 8;
    } else if (auto op = sourceDesc.getDefiningOp<mlir::memref::ViewOp>()) {
        auto memRefType = op.getSource().getType();
        auto elementType = memRefType.getElementType();
        return (elementType.getIntOrFloatBitWidth() + 7) / 8;
    } else if (auto op = sourceDesc.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        return getElementByteSize(op.getInputs()[0]);
    } else {
        auto type = sourceDesc.getType();
        if (auto memRefType = mlir::dyn_cast_or_null<mlir::MemRefType>(type)) {
            auto elementType = memRefType.getElementType();
            return (elementType.getIntOrFloatBitWidth() + 7) / 8;
        }

        return bytes;
    }
}

// Extract buffer pointers from memref descriptors
inline mlir::Value getBufferStartAddress(mlir::OpBuilder& builder, mlir::Location& loc, mlir::Value& sourceDesc) {
    // Extract source buffer pointer (allocated pointer + offset)
    // mlir::memref::Mem

    int64_t bytes = getElementByteSize(sourceDesc);

    mlir::Type i64Type = builder.getI64Type();
    mlir::LLVM::ExtractValueOp srcPtrExtractOp = builder.create<mlir::LLVM::ExtractValueOp>(loc, sourceDesc, 0);
    mlir::LLVM::ExtractValueOp srcOffsetExtractOp = builder.create<mlir::LLVM::ExtractValueOp>(loc, sourceDesc, 2);
    mlir::LLVM::PtrToIntOp srcPtrToIntOp = builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Type, srcPtrExtractOp);
    auto elementByteSize = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), i64Type, bytes);
    mlir::LLVM::MulOp byteOffsetMulOp =
            builder.create<mlir::LLVM::MulOp>(loc, i64Type, elementByteSize, srcOffsetExtractOp);
    mlir::LLVM::AddOp srcAddOp = builder.create<mlir::LLVM::AddOp>(loc, i64Type, srcPtrToIntOp, byteOffsetMulOp);

    return srcAddOp;
}

mlir::LogicalResult areAllLLVMTypes(mlir::Operation* op, mlir::ValueRange operands,
                                    mlir::ConversionPatternRewriter& rewriter) {
    if (!llvm::all_of(operands, [](mlir::Value value) {
            return mlir::LLVM::isCompatibleType(value.getType());
        })) {
        return rewriter.notifyMatchFailure(op, "Cannot convert if operands aren't of LLVM type.");
    }
    return mlir::success();
}

class LvlZeroAllocLowering final : public mlir::ConvertOpToLLVMPattern<mlir::memref::AllocOp> {
public:
    LvlZeroAllocLowering(const mlir::LLVMTypeConverter& typeConverter)
            : mlir::ConvertOpToLLVMPattern<mlir::memref::AllocOp>(typeConverter, vpux::benefitHigh) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult LvlZeroAllocLowering::matchAndRewrite(mlir::memref::AllocOp origOp, OpAdaptor adaptor,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MemRefType memrefType = origOp.getType();
    if (mlir::failed(areAllLLVMTypes(origOp, adaptor.getOperands(), rewriter)) ||
        !isConvertibleAndHasIdentityMaps(memrefType)) {
        return mlir::failure();
    }
    // Get shape of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.
    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;

    auto loc = origOp.getLoc();
    getMemRefDescriptorSizes(loc, memrefType, adaptor.getDynamicSizes(), rewriter, shape, strides, sizeBytes);
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMPointerType::get(ctx));
    mlir::func::FuncOp funcOp;
    auto moduleOp = vpux::getModuleOp(origOp);
    vpux::net::NetworkInfoOp netInfo;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    auto numArgs = funcOp.getNumArguments();
    auto context = funcOp.getArgument(GET_ARG_INDEX_CONTEXT(numArgs));
    auto allocatedPtr =
            createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_alloc", {sizeBytes, context}, returnType)
                    .getResult();
    // No alignment.
    mlir::Value alignedPtr = allocatedPtr;
    auto memrefDescriptor =
            this->createMemRefDescriptor(loc, memrefType, allocatedPtr, alignedPtr, shape, strides, rewriter);

    rewriter.replaceOp(origOp, {memrefDescriptor});

    return mlir::success();
}

class LvlZeroMemoryCopyLowering final : public mlir::ConvertOpToLLVMPattern<mlir::memref::CopyOp> {
public:
    LvlZeroMemoryCopyLowering(const mlir::LLVMTypeConverter& typeConverter, CommandListIndexState& cmdListIdxState)
            : mlir::ConvertOpToLLVMPattern<mlir::memref::CopyOp>(typeConverter, vpux::benefitHigh),
              commandListIndexState(cmdListIdxState) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::memref::CopyOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    CommandListIndexState& commandListIndexState;
};

mlir::LogicalResult LvlZeroMemoryCopyLowering ::matchAndRewrite(mlir::memref::CopyOp origOp, OpAdaptor adaptor,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    auto loc = origOp.getLoc();
    auto typeConverter = this->getTypeConverter();

    // Convert source op to LLVM pointer if necessary
    mlir::Value sourcePtr = adaptor.getSource();
    if (!mlir::LLVM::isCompatibleType(sourcePtr.getType())) {
        if (auto converted = typeConverter->materializeTargetConversion(
                    rewriter, loc, typeConverter->convertType(sourcePtr.getType()), mlir::ValueRange{sourcePtr})) {
            sourcePtr = converted;
        } else {
            return rewriter.notifyMatchFailure(origOp, "Could not convert source operand to LLVM type");
        }
    }

    // Convert target op to LLVM pointer if necessary
    mlir::Value targetPtr = adaptor.getTarget();
    if (!mlir::LLVM::isCompatibleType(targetPtr.getType())) {
        if (auto converted = typeConverter->materializeTargetConversion(
                    rewriter, loc, typeConverter->convertType(targetPtr.getType()), mlir::ValueRange{targetPtr})) {
            targetPtr = converted;
        } else {
            return rewriter.notifyMatchFailure(origOp, "Could not convert target operand to LLVM type");
        }
    }

    // Calculate the size of the memory to copy
    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;
    mlir::MemRefType targetMemrefType = mlir::cast<mlir::MemRefType>(origOp.getTarget().getType());
    getMemRefDescriptorSizes(origOp.getLoc(), targetMemrefType, {}, rewriter, shape, strides, sizeBytes);

    // Extract buffer pointers from memref descriptors
    auto src = getBufferStartAddress(rewriter, loc, sourcePtr);
    auto target = getBufferStartAddress(rewriter, loc, targetPtr);

    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
    auto moduleOp = vpux::getModuleOp(origOp);
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();

    // As of today, copy op should be considered as a sub graph
    // So, command list reset/submission is required for the copy op
    createL0ResetCommandList(origOp, rewriter, commandListIndexState);

    auto cmdList = commandListIndexState.getCommandList(funcOp);
    createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_append_memory_copy", {src, target, sizeBytes, cmdList},
                         returnType);

    createSubmitCommandList(rewriter, moduleOp, commandListIndexState);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

template <typename AsyncOp>
class AsyncOpRewriter final : public mlir::OpRewritePattern<AsyncOp> {
public:
    AsyncOpRewriter(mlir::MLIRContext* ctx, const mlir::LLVMTypeConverter& typeConverter, mlir::PatternBenefit benefit,
                    CommandListIndexState& cmdListIndexState, Logger log)
            : mlir::OpRewritePattern<AsyncOp>(ctx, benefit),
              _typeConverter(typeConverter),
              commandListIndexState(cmdListIndexState),
              _log(std::move(log)) {
        this->setDebugName("AsyncOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const final;
    const mlir::LLVMTypeConverter& _typeConverter;
    CommandListIndexState& commandListIndexState;
    Logger _log;
};

template <typename AsyncOp>
mlir::LogicalResult AsyncOpRewriter<AsyncOp>::matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const {
    auto moduleOp = vpux::getModuleOp(origOp);

    if (auto awaitOp = mlir::dyn_cast<mlir::async::AwaitOp>(*origOp)) {
        auto users = awaitOp->getUsers();
        bool awaitOpInScfFor = false;
        if (origOp->template getParentOfType<mlir::scf::ForOp>()) {
            awaitOpInScfFor = true;
            if (users.empty()) {
                rewriter.eraseOp(origOp);
                return mlir::success();
            }
        }

        if (users.empty()) {
            if (awaitOpInScfFor == false) {
                // if await is called outside of scf.for
                // command list needs to be submitted
                createSubmitCommandList(rewriter, moduleOp, commandListIndexState);
            }

            rewriter.eraseOp(origOp);
            return mlir::success();
        }

        // Async.AwaitOp
        // Replace operand of uses with operand of async.awaitop
        // as AwaitOp will be removed
        mlir::Value awaitOpOperand = awaitOp.getOperand();
        auto op = awaitOpOperand.getDefiningOp();
        if (auto executeOp = mlir::cast<mlir::async::ExecuteOp>(op)) {
            const auto results = executeOp.getResults();
            int index = -1;
            for (size_t i = 0; i < results.size(); ++i) {
                if (awaitOpOperand == results[i]) {
                    if (i == 0) {
                        _log.error("Invalid index {0} as the first result is token", i);
                        return mlir::failure();
                    }

                    // decrease index as the first result of ExecuteOp is token
                    index = static_cast<int>(i) - 1;
                    break;
                }
            }

            if (index == -1) {
                _log.error("Invalid async.AwaitOp");
                return mlir::failure();
            }

            auto moduleOp = vpux::getModuleOp(executeOp);
            for (auto& op : executeOp.getBody()->getOperations()) {
                if (auto callOp = mlir::dyn_cast<Core::NestedCallOp>(op)) {
                    auto callee = callOp->getAttrOfType<mlir::SymbolRefAttr>("callee");
                    auto root = callee.getRootReference();
                    auto fnModule = moduleOp.lookupSymbol<HostExec::BinaryOp>(root);
                    if (fnModule == nullptr) {
                        _log.error("Could not find binary op for subgraph: {0}", root.str());
                        return mlir::failure();
                    }
                    auto funcOp = fnModule.lookupSymbol<mlir::func::FuncOp>(callee.getLeafReference().str());
                    if (funcOp == nullptr) {
                        _log.error("Could not find function declaration: {0}", callee.getLeafReference().str());
                        return mlir::failure();
                    }

                    auto resultCount = funcOp.getNumResults();
                    auto inputCount = funcOp.getNumArguments();
                    auto operand = *(callOp.getOperands().begin() + (inputCount - resultCount) +
                                     static_cast<unsigned int>(index));
                    for (auto u : users) {
                        if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(u)) {
                            viewOp.setOperand(0, operand);
                        } else if (auto copyOp = mlir::dyn_cast<mlir::memref::CopyOp>(u)) {
                            copyOp.setOperand(0, operand);
                        } else if (auto nestedCallOp = mlir::dyn_cast<Core::NestedCallOp>(u)) {
                            if (awaitOpInScfFor) {
                                auto userOperands = u->getOperands();
                                auto awaitOpResult = awaitOp.getResult();
                                uint32_t index = 0;
                                for (auto userOperand : userOperands) {
                                    if (awaitOpResult == userOperand) {
                                        nestedCallOp.setOperand(index, operand);
                                        break;
                                    }
                                    ++index;
                                }
                            }
                        } else if (mlir::isa<mlir::UnrealizedConversionCastOp, Core::NestedCallOp,
                                             mlir::async::YieldOp>(u)) {
                            continue;
                        } else {
                            _log.error("Not supported user type: {0}", u->getName().getStringRef().str());
                            return mlir::failure();
                        }
                    }
                }
            }
        }

        mlir::Operation* nextOp = origOp->getNextNode();

        // If there are multiple AwaitOp for one operation with multiple outputs
        // there will be multiple AwaitOp for synchronization.
        // The last AwaitOp will be replaced with submitCommandList.
        if (mlir::isa<mlir::async::AwaitOp>(nextOp)) {
            rewriter.eraseOp(origOp);
            return mlir::success();
        }

        createSubmitCommandList(rewriter, moduleOp, commandListIndexState);
        rewriter.eraseOp(origOp);
        return mlir::success();
    } else if (mlir::isa<mlir::async::AwaitAllOp>(origOp)) {
        createSubmitCommandList(rewriter, moduleOp, commandListIndexState);
        rewriter.eraseOp(origOp);
        return mlir::success();
    } else {
        _log.warning("Unsupported async operation: {0}", origOp->getName());
    }
    return mlir::failure();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::AddToGroupOp>::matchAndRewrite(mlir::async::AddToGroupOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    rewriter.eraseOp(origOp);
    return mlir::success();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::CreateGroupOp>::matchAndRewrite(
        mlir::async::CreateGroupOp origOp, mlir::PatternRewriter& rewriter) const {
    createL0ResetCommandList(origOp, rewriter, commandListIndexState);

    rewriter.eraseOp(origOp);
    return mlir::success();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::ExecuteOp>::matchAndRewrite(mlir::async::ExecuteOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto loc = origOp.getLoc();
    mlir::func::FuncOp funcOp;
    auto moduleOp = vpux::getModuleOp(origOp);
    vpux::net::NetworkInfoOp netInfo;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);

    auto numArgs = funcOp.getNumArguments();
    auto umdContext = funcOp.getArgument(GET_ARG_INDEX_CONTEXT(numArgs));
    auto device = funcOp.getArgument(GET_ARG_INDEX_DEVICE(numArgs));
    auto ddiTable = funcOp.getArgument(GET_ARG_INDEX_DDI_TABLE(numArgs));
    auto cmdQueue = funcOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));

    // note: needs to calculate the size of the kernel function after serialization of the core.NestedModule
    if (origOp->template getParentOfType<mlir::scf::ForOp>() == nullptr) {
        createL0ResetCommandList(origOp, rewriter, commandListIndexState);
    }

    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    auto& inputs = commandListIndexState.inputs;
    auto& outputs = commandListIndexState.outputs;
    std::map<std::string, std::pair<HostExec::BinaryOp, mlir::Value>> kernels;
    for (auto& op : origOp.getBody()->getOperations()) {
        if (auto callOp = mlir::dyn_cast<Core::NestedCallOp>(op)) {
            mlir::SmallVector<mlir::Value, 1> kernelInputs, kernelOutputs;
            auto callee = callOp->getAttrOfType<mlir::SymbolRefAttr>("callee");
            auto root = callee.getRootReference();
            auto calleeNameAttr = callee.getLeafReference();
            auto kernelBinary = moduleOp.lookupSymbol<HostExec::BinaryOp>(root);
            auto rootStr = root.str();
            if (!kernelBinary) {
                _log.error("BinaryOp not found for {0}", root.str());
                return mlir::failure();
            }
            auto binaryDataOp =
                    kernelBinary.lookupSymbol<HostExec::BinaryDataOp>("serialized_" + callee.getLeafReference().str());
            if (!binaryDataOp) {
                _log.error("BinaryDataOp not found for {0}", callee.getLeafReference().str());
                return mlir::failure();
            }

            auto object = binaryDataOp.getObject();
            if (!object) {
                _log.error("Object not found in BinaryDataOp for {0}",
                           callOp.getOperation()->getName().getStringRef().str());
                return mlir::failure();
            }

            llvm::StringRef rawBytes = object.getObject().getValue();
            size_t dataSize = rawBytes.size();
            auto kernelSize = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), dataSize);

            auto resultCount = callOp.getNumResults();
            auto inputCount = callOp.getNumOperands() - resultCount;

            mlir::Value kernelGlobal;
            auto iter = kernels.find(calleeNameAttr.str());
            if (iter != kernels.end()) {
                kernelGlobal = iter->second.second;
            } else {
                auto name = callee.getLeafReference().getValue();
                auto nameAttr = mlir::StringAttr::get(origOp.getContext(), std::string(name) + "_kernel");
                kernelGlobal = mlir::LLVM::createGlobalString(loc, rewriter, nameAttr.getValue(), object.getObject(),
                                                              mlir::LLVM::Linkage::Internal);
                kernels[calleeNameAttr.str()] = std::make_pair(kernelBinary, kernelGlobal);
            }

            kernelInputs.insert(kernelInputs.begin(), callOp.getArgOperands().begin(),
                                callOp.getArgOperands().begin() + inputCount);
            kernelOutputs.insert(kernelOutputs.begin(), callOp.getArgOperands().begin() + inputCount,
                                 callOp.getArgOperands().end());

            auto numInputs =
                    rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), kernelInputs.size());
            auto numOutputs =
                    rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), kernelOutputs.size());

            auto voidPointerType = mlir::LLVM::LLVMPointerType::get(&(_typeConverter.getContext()));
            auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(loc, voidPointerType);

            // Store each output pointer as void* in the input array
            for (size_t i = 0; i < kernelInputs.size(); ++i) {
                auto idx = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), i);
                auto gep = rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, voidPtrTy, inputs, mlir::ValueRange{idx});
                auto llvmInput = kernelInputs[i];
                if (!mlir::LLVM::isCompatibleType(llvmInput.getType())) {
                    if (auto converted = _typeConverter.materializeTargetConversion(
                                rewriter, loc, _typeConverter.convertType(llvmInput.getType()),
                                mlir::ValueRange{llvmInput})) {
                        llvmInput = converted;
                    } else {
                        _log.error("Could not convert input type: {0}", llvmInput.getType());
                        return mlir::failure();
                    }
                }

                // input pointer = allocated pointer(0) + offset(2)
                auto inputPtr = getBufferStartAddress(rewriter, loc, llvmInput);
                rewriter.create<mlir::LLVM::StoreOp>(loc, inputPtr, gep);
            }

            // Store each output pointer as void* in the output array
            for (size_t i = 0; i < kernelOutputs.size(); ++i) {
                auto idx = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), i);
                auto gep =
                        rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, voidPtrTy, outputs, mlir::ValueRange{idx});
                auto llvmOutput = kernelOutputs[i];
                if (!mlir::LLVM::isCompatibleType(llvmOutput.getType())) {
                    if (auto converted = _typeConverter.materializeTargetConversion(
                                rewriter, loc, _typeConverter.convertType(llvmOutput.getType()),
                                mlir::ValueRange{llvmOutput})) {
                        llvmOutput = converted;
                    } else {
                        _log.error("Could not convert output type: {0}", llvmOutput.getType());
                        return mlir::failure();
                    }
                }

                // output pointer = allocated pointer(0) + offset(2)
                auto outputPtr = getBufferStartAddress(rewriter, loc, llvmOutput);
                rewriter.create<mlir::LLVM::StoreOp>(loc, outputPtr, gep);
            }

            auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
            auto cmdList = commandListIndexState.getCommandList(funcOp);
            createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_execute_graph",
                                 {inputs.getResult(), numInputs, outputs.getResult(), numOutputs, kernelGlobal,
                                  kernelSize, umdContext, device, ddiTable, cmdList, cmdQueue},
                                 returnType);

            // Restore stack used for descriptors
            rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSaveOp);
        }
    }

    // note
    // BinaryOp will be remove in safeRunOnModule
    // as lowering Async::AwaitOp is required function signature defined in BinaryOp

    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// ConvertToLLVMUMDCallsPass
//

class ConvertToLLVMUMDCallsPass final : public HostExec::impl::ConvertToLLVMUMDCallsBase<ConvertToLLVMUMDCallsPass> {
public:
    explicit ConvertToLLVMUMDCallsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertToLLVMUMDCallsPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = &getContext();

    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);
    mlir::LowerToLLVMOptions options(ctx);
    options.useBarePtrCallConv = true;
    mlir::LLVMTypeConverter typeConverter(ctx, options);

    mlir::func::FuncOp mainFuncOp;
    for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
        if (funcOp.getName() == "main") {
            mainFuncOp = funcOp;
            addFuncParamsForUmdFuncCall(funcOp);
        }
    }

    mlir::populateConversionTargetFromOperation(module, target, typeConverter, patterns);
    target.addIllegalOp<mlir::memref::AllocOp>();
    target.addIllegalOp<mlir::memref::CopyOp>();
    target.addIllegalOp<mlir::async::CreateGroupOp>();
    target.addIllegalOp<mlir::async::AddToGroupOp>();
    target.addIllegalOp<mlir::async::AwaitAllOp>();
    target.addIllegalOp<mlir::async::AwaitOp>();
    target.addIllegalOp<mlir::async::ExecuteOp>();
    target.addLegalOp<vpux::HostExec::BinaryOp>();
    target.addLegalOp<vpux::HostExec::BinaryDataOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<mlir::cf::AssertOp>();

    CommandListIndexState commandListIndexState;
    commandListIndexState.initialize(module, mainFuncOp);

    patterns.add<LvlZeroMemoryCopyLowering>(typeConverter, commandListIndexState);
    patterns.add<LvlZeroAllocLowering>(typeConverter);
    patterns.add<AsyncOpRewriter<mlir::async::AddToGroupOp>>(ctx, typeConverter, vpux::benefitHigh,
                                                             commandListIndexState, _log);
    patterns.add<AsyncOpRewriter<mlir::async::CreateGroupOp>>(ctx, typeConverter, vpux::benefitHigh,
                                                              commandListIndexState, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitAllOp>>(ctx, typeConverter, vpux::benefitHigh, commandListIndexState,
                                                           _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitOp>>(ctx, typeConverter, vpux::benefitHigh, commandListIndexState,
                                                        _log);

    // Note: ExecuteOp is a special case, a few conditions apply which is why it is the last pattern,
    // 1 npu_level_zero_execute_graph that all inputs and outputs are converted to LLVM types.
    // 2.It will have successor and predecessor dependencies on the other async and memref operations therefore those
    // operations should be removed or converted before this pattern is applied.

    patterns.add<AsyncOpRewriter<mlir::async::ExecuteOp>>(ctx, typeConverter, vpux::benefitLow, commandListIndexState,
                                                          _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }

    for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
        if (funcOp.getName() == "main") {
            commandListIndexState.finalizeCommandListIndex(module, ctx, funcOp);
        }
    }

    // remove all BinaryOp as they were converted into global variables
    auto binaryOps = to_small_vector(module.getOps<HostExec::BinaryOp>());
    for (auto binaryOp : binaryOps) {
        binaryOp.getOperation()->erase();
    }
}

}  // namespace

//
// createConvertToLLVMUMDCallsPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createConvertToLLVMUMDCallsPass(Logger log) {
    return std::make_unique<ConvertToLLVMUMDCallsPass>(log);
}
