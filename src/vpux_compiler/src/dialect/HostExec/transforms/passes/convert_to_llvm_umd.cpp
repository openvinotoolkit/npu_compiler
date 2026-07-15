//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#define NO_MINMAX
#endif

#include <cmath>

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

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

    bool commandListGroupStarted = false;
    bool enablePipelinedCmdListRecording = vpux::HostExec::defaultEnablePipelinedCmdListRecording;

    void initialize(mlir::func::FuncOp funcOp, bool enablePipelinedCmdListRecording) {
        mlir::OpBuilder builder(funcOp);
        auto& entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);
        this->enablePipelinedCmdListRecording = enablePipelinedCmdListRecording;
        stepSizeInByte = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64), 8);
    }

    // Update pointers of commandlist**
    void increaseCommandListIndex(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp) {
        auto loc = builder.getUnknownLoc();
        auto numArgs = funcOp.getNumArguments();
        auto cmdList = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));
        mlir::Type i64Type = builder.getI64Type();
        auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        if (!enablePipelinedCmdListRecording && resetIndex > 0) {
            resetIndex++;
            // no need of increase command list pointer
            return;
        }

        if (resetIndex++ > 0) {
            // Increase commandlist pointer by 8 (size of pointer type)
            mlir::LLVM::AddOp addOp = builder.create<mlir::LLVM::AddOp>(loc, i64Type, lastResult, stepSizeInByte);
            lastResult = addOp.getResult();
            lastResultPtr = builder.create<mlir::LLVM::IntToPtrOp>(loc, voidPtrType, lastResult);
        } else {
            // For the first commandlist pointer, no need of increasing commandlist pointer
            lastResult = builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Type, cmdList);
            lastResultPtr = builder.create<mlir::LLVM::IntToPtrOp>(loc, voidPtrType, lastResult);
        }
    }

    // Returns address of commandlist**
    mlir::Value getCommandList(mlir::func::FuncOp funcOp) {
        auto numArgs = funcOp.getNumArguments();
        auto cmdList = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));

        // For the first subgraph, no need of commandlist pointer update is requried

        return ((resetIndex > 1) && enablePipelinedCmdListRecording) ? lastResultPtr : cmdList;
    }

    // Returns commandListIndex
    mlir::Value getCommandListIndex(mlir::OpBuilder& builder) {
        int64_t index = ((resetIndex > 1) && enablePipelinedCmdListRecording) ? resetIndex - 1 : 0;
        return builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64), index);
    }

    // Update inference execution sync params (e.g., event or fence) for the last commandlist submission
    void finalizeCommandListIndex(mlir::ModuleOp module, mlir::MLIRContext* ctx, mlir::func::FuncOp funcOp) {
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

        // Add an attribute "number of subgraphs" to the main module
        mlir::Type i64Type = mlir::IntegerType::get(ctx, 64);
        mlir::IntegerAttr numSubGraphs =
                mlir::IntegerAttr::get(i64Type, ((enablePipelinedCmdListRecording == false) ? 1 : resetIndex));

        module->setAttr(HOST_EXEC_NUM_SUBGRAPH_ATTR_NAME, numSubGraphs);
    }
};

class MemRefDescriptorUtil {
public:
    // Standard indices for MemRef descriptor components
    static constexpr unsigned BASE_PTR_INDEX = 0;     // Base pointer
    static constexpr unsigned ALIGNED_PTR_INDEX = 1;  // Aligned pointer (actual data)
    static constexpr unsigned OFFSET_INDEX = 2;       // Offset
    static constexpr unsigned SIZES_INDEX = 3;        // Dimensions array
    static constexpr unsigned STRIDES_INDEX = 4;      // Strides array

    // Extract components from MemRef value
    static mlir::Value extractBasePtr(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memrefValue) {
        return builder.create<mlir::LLVM::ExtractValueOp>(loc, memrefValue, BASE_PTR_INDEX);
    }

    static mlir::Value extractAlignedPtr(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memrefValue) {
        return builder.create<mlir::LLVM::ExtractValueOp>(loc, memrefValue, ALIGNED_PTR_INDEX);
    }

    static mlir::Value extractOffset(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memrefValue) {
        return builder.create<mlir::LLVM::ExtractValueOp>(loc, memrefValue, OFFSET_INDEX);
    }

    static mlir::Value extractSizes(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memrefValue) {
        return builder.create<mlir::LLVM::ExtractValueOp>(loc, memrefValue, SIZES_INDEX);
    }

    static mlir::Value extractStrides(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memrefValue) {
        return builder.create<mlir::LLVM::ExtractValueOp>(loc, memrefValue, STRIDES_INDEX);
    }

    static mlir::Value getSizeForDim(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value sizesArray,
                                     uint32_t dimIndex) {
        auto i32Type = builder.getIntegerType(32);
        auto constDimIndex = builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, dimIndex);

        auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto gepOp = builder.create<mlir::LLVM::GEPOp>(loc, ptrType, builder.getI64Type(), sizesArray,
                                                       mlir::ValueRange{constDimIndex});

        return builder.create<mlir::LLVM::LoadOp>(loc, builder.getI64Type(), gepOp);
    }

    static mlir::Value getStrideForDim(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stridesArray,
                                       uint32_t dimIndex) {
        auto i32Type = builder.getIntegerType(32);
        auto constDimIndex = builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, dimIndex);

        auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto gepOp = builder.create<mlir::LLVM::GEPOp>(loc, ptrType, builder.getI64Type(), stridesArray,
                                                       mlir::ValueRange{constDimIndex});

        return builder.create<mlir::LLVM::LoadOp>(loc, builder.getI64Type(), gepOp);
    }

    static uint32_t getRank(mlir::Value memrefValue) {
        // Cast the type to a MemRefType.
        if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(memrefValue.getType())) {
            return memrefType.getRank();
        }

        return 0;
    }
};

int64_t getElementByteSize(mlir::Value sourceDesc) {
    int64_t bytes = 2;

    if (auto op = sourceDesc.getDefiningOp<mlir::memref::SubViewOp>()) {
        auto memRefType = op.getType();
        auto elementType = memRefType.getElementType();
        return (elementType.getIntOrFloatBitWidth() + 7) / 8;
    } else if (auto op = sourceDesc.getDefiningOp<mlir::memref::ViewOp>()) {
        auto memRefType = op.getType();
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

// Recursively accumulate offsets for subview/view/cast chains
mlir::Value accumulateOffsets(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) {
    auto i64Type = builder.getI64Type();
    mlir::Value totalOffset = MemRefDescriptorUtil::extractOffset(builder, loc, value);
    int64_t elemBytes = getElementByteSize(value);
    mlir::Value current = value;

    while (true) {
        if (auto subviewOp = current.getDefiningOp<mlir::memref::SubViewOp>()) {
            current = subviewOp.getSource();
        } else if (auto viewOp = current.getDefiningOp<mlir::memref::ViewOp>()) {
            // Accumulate byteShift (offset in bytes)
            auto byteShift = viewOp.getByteShift();
            if (byteShift) {
                auto shiftValue = builder.create<mlir::arith::IndexCastOp>(loc, i64Type, byteShift);
                // Convert byte offset to element offset if needed
                auto elemByteSize = builder.create<mlir::LLVM::ConstantOp>(loc, i64Type, elemBytes);
                auto divOp = builder.create<mlir::LLVM::SDivOp>(loc, i64Type, shiftValue, elemByteSize);
                totalOffset = builder.create<mlir::LLVM::AddOp>(loc, i64Type, totalOffset, divOp);
            }
            current = viewOp.getSource();
        } else if (auto castOp = current.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
            current = castOp.getSource();
        } else if (auto castOp = current.getDefiningOp<mlir::memref::CastOp>()) {
            current = castOp.getSource();
        } else if (auto castOp = current.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
            current = castOp.getInputs()[0];
        } else {
            break;
        }
    }
    return totalOffset;
}

// Extract buffer pointers from memref descriptors
inline mlir::Value getBufferStartAddress(mlir::OpBuilder& builder, mlir::Location& loc, mlir::Value& sourceDesc) {
    // Compute byte address of the first element: aligned_ptr + offset * elemBytes.
    //
    // We should use ALIGNED_PTR (index 1), NOT BASE_PTR (index 0), because MLIR's MemRef->LLVM
    // lowering folds the view byte shift into ALIGNED_PTR rather than into OFFSET.
    // For example, given:
    //   %alloc = memref.alloc() : memref<960032xi8>
    //   %view  = memref.view %alloc[16][] : ... to memref<1x3x400x400xf16>
    // the LLVM descriptor for %view has:
    //   BASE_PTR    = alloc base           (e.g. 0xABC0)
    //   ALIGNED_PTR = alloc base + 16      (byte shift baked in)
    //   OFFSET      = 0
    // Using BASE_PTR + OFFSET*elemBytes = 0xABC0 + 0 = 0xABC0  (WRONG)
    // Using ALIGNED_PTR + OFFSET*elemBytes = 0xABC0+16 + 0 = 0xABD0  (correct)
    //
    // An alternative would be BASE_PTR + accumulateOffsets()*elemBytes, where
    // accumulateOffsets() walks the SubViewOp/ViewOp/CastOp chain to recover the
    // byte shift as an element offset. However, in this pass the source of the
    // memref.copy has already passed through async.await/async.execute, which
    // breaks the defining-op chain that accumulateOffsets() relies on. Since
    // accumulateOffsets() does not handle async.await, it returns OFFSET=0 and
    // the byte shift is lost - the same incorrect result as raw BASE_PTR+OFFSET.
    //
    // ALIGNED_PTR is always correct here: the MLIR convention guarantees
    // "ALIGNED_PTR + OFFSET * elemBytes = first-element address" for every
    // MemRef type (plain alloc, view, subview, function argument, etc.),
    // and this invariant is preserved across async boundaries.
    int64_t bytes = getElementByteSize(sourceDesc);

    mlir::Type i64Type = builder.getI64Type();
    auto srcPtrExtractOp =
            builder.create<mlir::LLVM::ExtractValueOp>(loc, sourceDesc, MemRefDescriptorUtil::ALIGNED_PTR_INDEX);
    auto srcOffsetExtractOp =
            builder.create<mlir::LLVM::ExtractValueOp>(loc, sourceDesc, MemRefDescriptorUtil::OFFSET_INDEX);

    auto srcPtrToIntOp = builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Type, srcPtrExtractOp);
    auto elementByteSize = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), i64Type, bytes);
    auto byteOffsetMulOp = builder.create<mlir::LLVM::MulOp>(loc, i64Type, elementByteSize, srcOffsetExtractOp);
    auto srcAddOp = builder.create<mlir::LLVM::AddOp>(loc, i64Type, srcPtrToIntOp, byteOffsetMulOp);

    return srcAddOp;
}

// Manages input and output tensor information
class ModelIOManager {
public:
    // The max number of inputs / outputs for all sub graphs.
    // This is used to allocate an array for input / outputs pointers
    uint32_t maxNumInputs = 0;
    uint32_t maxNumOutputs = 0;

    // Buffers for inputs/outputs, their strides and sizes
    mlir::LLVM::AllocaOp inputDescs = nullptr;
    mlir::LLVM::AllocaOp outputDescs = nullptr;

    ModelIOManager(mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp, Logger log): _log(std::move(log)) {
        mlir::OpBuilder builder(funcOp);
        auto& entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);

        analyzeIORequirements(moduleOp);

        auto voidPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto tensorDescType = getTensorDescStructType(moduleOp.getContext());
        auto constMaxNumInputs = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(),
                                                                        builder.getIntegerType(32), maxNumInputs);
        auto constMaxNumOutputs = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(),
                                                                         builder.getIntegerType(32), maxNumOutputs);

        // Allocate buffers for input/output pointers
        inputDescs = builder.create<mlir::LLVM::AllocaOp>(builder.getUnknownLoc(), voidPtrType, tensorDescType,
                                                          constMaxNumInputs);
        outputDescs = builder.create<mlir::LLVM::AllocaOp>(builder.getUnknownLoc(), voidPtrType, tensorDescType,
                                                           constMaxNumOutputs);
    }

    // Process buffers and extract their data, size, and stride information
    bool processTensors(mlir::OpBuilder& builder, mlir::Location loc,
                        const mlir::SmallVector<mlir::Value, 1>& inputBuffers,
                        const mlir::SmallVector<mlir::Value, 1>& outputBuffers,
                        const mlir::LLVMTypeConverter& typeConverter, mlir::func::FuncOp funcOp) {
        auto& entryBlock = funcOp.getBody().front();
        // Process inputs
        for (size_t i = 0; i < inputBuffers.size(); ++i) {
            if (!processBuffer(builder, loc, inputBuffers[i], i, inputDescs, typeConverter, "input", entryBlock)) {
                return false;
            }
        }

        // Process outputs
        for (size_t i = 0; i < outputBuffers.size(); ++i) {
            if (!processBuffer(builder, loc, outputBuffers[i], i, outputDescs, typeConverter, "output", entryBlock)) {
                return false;
            }
        }

        return true;
    }

    // Get the maximum number of inputs
    uint32_t getMaxInputCount() const {
        return maxNumInputs;
    }

    // Get the maximum number of outputs
    uint32_t getMaxOutputCount() const {
        return maxNumOutputs;
    }

    // Get input/output buffers
    mlir::Value getInputBufferDescs() {
        return inputDescs.getResult();
    }
    mlir::Value getOutputBufferDescs() {
        return outputDescs.getResult();
    }

    mlir::LLVM::LLVMStructType getTensorDescStructType(mlir::MLIRContext* ctx) {
        // Note
        // If members are added/removed from struct, update HostExec::MemRefDesc
        // in params.hpp. as MemRefDesc is used to get buffer information (e.g., strides)
        SmallVector<mlir::Type, static_cast<uint32_t>(MemRefDescMemberIndex::COUNT)> members;
        auto int64Type = mlir::IntegerType::get(ctx, 64);
        auto voidPtrType = mlir::LLVM::LLVMPointerType::get(ctx);
        auto arrayType = mlir::LLVM::LLVMArrayType::get(int64Type, vpux::HostExec::MaxStrideDim);

        // The order to add members is important. See MemRefDescMemberIndex for details.
        members.push_back(voidPtrType);  // aligned ptr
        members.push_back(int64Type);    // offset
        members.push_back(int64Type);    // elementByteSize
        members.push_back(int64Type);    // dimCount
        members.push_back(int64Type);    // networkArgIndex
        members.push_back(arrayType);    // size
        members.push_back(arrayType);    // strides

        return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
    }

private:
    Logger _log;

    // Analyze model to determine max input/output requirements
    void analyzeIORequirements(mlir::ModuleOp moduleOp) {
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

    // Recursively returns the function argument index if value is derived from a function argument, else -1
    int getDerivedFuncArgIndex(mlir::Value value, mlir::Block& funcOpEntryBlock,
                               llvm::SmallPtrSetImpl<mlir::Value>& visited) {
        // Check if value is a block argument and belongs to the entry block
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            if (blockArg.getOwner() == &funcOpEntryBlock) {
                return blockArg.getArgNumber();
            }
            return -1;
        }
        if (!visited.insert(value).second) {
            return -1;  // already visited, avoid cycles
        }

        if (auto defOp = value.getDefiningOp()) {
            if (!mlir::isa<mlir::UnrealizedConversionCastOp, mlir::memref::SubViewOp, mlir::memref::ViewOp,
                           mlir::memref::CastOp>(defOp)) {
                // only view, subview and cast ops are allowed in the chain
                // for mutable command list
                return -1;
            }

            for (auto operand : defOp->getOperands()) {
                int idx = getDerivedFuncArgIndex(operand, funcOpEntryBlock, visited);
                if (idx != -1) {
                    return idx;
                }
            }
        }
        return -1;
    }

    // Process a single buffer and extract its stride information
    bool processBuffer(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value buffer, size_t index,
                       mlir::LLVM::AllocaOp bufferDescArray, const mlir::LLVMTypeConverter& typeConverter,
                       const char* bufferType, mlir::Block& funcOpEntryBlock) {
        auto llvmValue = buffer;
        // Convert to LLVM type if needed
        if (!mlir::LLVM::isCompatibleType(llvmValue.getType())) {
            if (auto converted = typeConverter.materializeTargetConversion(
                        builder, loc, typeConverter.convertType(llvmValue.getType()), mlir::ValueRange{llvmValue})) {
                llvmValue = converted;
            } else {
                _log.error("Could not convert {0} type: {1}", bufferType, llvmValue.getType());
                return false;
            }
        }

        auto i64Type = builder.getI64Type();
        auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto tensorStructType = getTensorDescStructType(builder.getContext());
        auto rank = MemRefDescriptorUtil::getRank(buffer);
        if (rank == 0) {
            _log.error("Could not get rank from {0} type", buffer.getType());
            return false;
        } else if (rank > vpux::HostExec::MaxStrideDim) {
            _log.error("MemRef rank {0} is greater than supported max {1}", rank, vpux::HostExec::MaxStrideDim);
            return false;
        }
        auto arrayType = mlir::LLVM::LLVMArrayType::get(i64Type, rank);

        auto createGetOp = [&](MemRefDescMemberIndex memrefIndex) {
            return builder.create<mlir::LLVM::GEPOp>(
                    loc, voidPtrTy, tensorStructType, bufferDescArray,
                    ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index), static_cast<int32_t>(memrefIndex)});
        };

        // Get buffer address and store it in the array
        auto basePtrExtractOp = MemRefDescriptorUtil::extractBasePtr(builder, loc, llvmValue);
        auto bufferBaseGepPtr = createGetOp(MemRefDescMemberIndex::DATA);
        builder.create<mlir::LLVM::StoreOp>(loc, basePtrExtractOp, bufferBaseGepPtr);

        auto offsetGepPtr = createGetOp(MemRefDescMemberIndex::OFFSET);
        auto accumulatedOffset = accumulateOffsets(builder, loc, llvmValue);
        builder.create<mlir::LLVM::StoreOp>(loc, accumulatedOffset, offsetGepPtr);

        auto elementByteSizeGepPtr = createGetOp(MemRefDescMemberIndex::ELEMENT_BYTE_SIZE);
        int64_t bytes = getElementByteSize(llvmValue);
        auto elementByteSize = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), i64Type, bytes);
        builder.create<mlir::LLVM::StoreOp>(loc, elementByteSize, elementByteSizeGepPtr);

        auto dimCountGepPtr = createGetOp(MemRefDescMemberIndex::DIM_COUNT);
        auto dimCount = builder.create<mlir::LLVM::ConstantOp>(loc, i64Type, rank);
        builder.create<mlir::LLVM::StoreOp>(loc, dimCount, dimCountGepPtr);

        auto networkArgumentIndexGepPtr = createGetOp(MemRefDescMemberIndex::NETWORK_ARG_INDEX);
        // This will need to be updated to support UpdateMutableCommandList
        llvm::SmallPtrSet<mlir::Value, 8> visited;
        auto idx = getDerivedFuncArgIndex(buffer, funcOpEntryBlock, visited);
        auto networkArgumentIndex = builder.create<mlir::LLVM::ConstantOp>(loc, i64Type, idx);
        builder.create<mlir::LLVM::StoreOp>(loc, networkArgumentIndex, networkArgumentIndexGepPtr);

        // Extract and store sizes information
        auto sizesGepPtr = createGetOp(MemRefDescMemberIndex::SIZES);
        auto sizesExtracted = MemRefDescriptorUtil::extractSizes(builder, loc, llvmValue);
        auto sizes = builder.create<mlir::LLVM::StoreOp>(loc, sizesExtracted, sizesGepPtr);
        sizes->setAttr("elem_type", mlir::TypeAttr::get(arrayType));

        // Pointer to temporary buffer for strides
        auto stridesGepPtr = createGetOp(MemRefDescMemberIndex::STRIDES);
        auto stridesExtracted = MemRefDescriptorUtil::extractStrides(builder, loc, llvmValue);
        auto strides = builder.create<mlir::LLVM::StoreOp>(loc, stridesExtracted, stridesGepPtr);
        strides->setAttr("elem_type", mlir::TypeAttr::get(arrayType));

        return true;
    }
};

void updateFuncTerminator(mlir::func::FuncOp funcOp) {
    for (auto& block : funcOp.getBody()) {
        // Find the current terminator
        if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator())) {
            // Replace the return operation with a new one that has no arguments
            mlir::OpBuilder builder(returnOp);
            builder.create<mlir::func::ReturnOp>(returnOp.getLoc());
            returnOp.erase();
        }
    }
    // Update functype with no return types
    auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), funcOp.getArgumentTypes(), mlir::TypeRange{});
    funcOp.setType(newFuncType);
}

// @brief Add arguments (e.g., commandlist, command queue) to an entry function for L0 function calls
void addFuncParamsForUmdFuncCall(mlir::func::FuncOp funcOp) {
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
    mlir::Type executionContextPtrType = mlir::LLVM::LLVMPointerType::get(ctx);

    newInputTypes.push_back(contextHandlePtrType);
    newInputTypes.push_back(deviceHandlePtrType);
    newInputTypes.push_back(ddiTableHandlePtrType);
    newInputTypes.push_back(commandListHandlePtrType);
    newInputTypes.push_back(commandListCountType);
    newInputTypes.push_back(commandQueueHandlePtrType);
    newInputTypes.push_back(fenceHandlePtrType);
    newInputTypes.push_back(eventHandlePtrType);
    newInputTypes.push_back(executionContextPtrType);

    auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), newInputTypes, funcOp->getResultTypes());
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
    entryBlock.addArgument(executionContextPtrType, funcOp->getLoc());
}

//@brief increase index of command lists
void increaseCommandListIndex(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                              CommandListIndexState& cmdListIndexState) {
    if (cmdListIndexState.commandListGroupStarted) {
        // Skip redudant command list reset
        // When there are multiple ExecutOp in a scf::for, there will be multiple CreateGroupOps.
        // Inference exectuion will be recorded in one command list.
        // A barrier will be added between ExecuteOp.

        return;
    }

    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();

    cmdListIndexState.commandListGroupStarted = true;
    cmdListIndexState.increaseCommandListIndex(rewriter, funcOp);
}

//@brief create function call op to submit command list
void createSubmitCommandList(mlir::OpBuilder& builder, mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp,
                             CommandListIndexState& cmdListIndexState) {
    if (cmdListIndexState.commandListGroupStarted == false) {
        // ignore redudandant await/await_all op
        return;
    }

    cmdListIndexState.commandListGroupStarted = false;

    mlir::MLIRContext* ctx = builder.getContext();

    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));

    // Create the LLVM::mlir.constant operation representing nullptr
    auto elementPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(builder.getUnknownLoc(), elementPtrType);

    // Host side synchronization will be required for the last command list.
    // The last two arguments will be updated in increaseCommandListIndex later if required.
    auto numArgs = funcOp.getNumArguments();
    auto cmdQueue = funcOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));
    auto execContext = funcOp.getArgument(GET_ARG_INDEX_COMMAND_EXECUTION_CONTEXT(numArgs));

    auto cmdList = cmdListIndexState.getCommandList(funcOp);
    auto callOp = createLLVMFuncCallOp(builder, moduleOp, "npu_level_zero_submit_commandlist",
                                       {cmdList, cmdQueue, nullPtr, nullPtr, execContext}, returnType);

    // Mark this call op as the last one to update 3rd and 4th arguments of the last submit command list
    cmdListIndexState.lastSubmitCommandListCallOp = callOp;
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

// Extract only the shape-dimension dynamic-size operands from an alloc's dynamic operand list.
// An alloc may carry extra operands (e.g. shape-buffer values) beyond the memref's actual
// dynamic dims; trimming them prevents passing unexpected values to getMemRefDescriptorSizes
SmallVector<mlir::Value, 4> trimToShapeDynamicSizes(mlir::MemRefType memrefType, mlir::ValueRange dynamicSizes) {
    const auto expectedCount = static_cast<size_t>(llvm::count(memrefType.getShape(), mlir::ShapedType::kDynamic));
    SmallVector<mlir::Value, 4> shapeSizes;
    shapeSizes.assign(dynamicSizes.begin(), dynamicSizes.begin() + std::min(expectedCount, dynamicSizes.size()));
    return shapeSizes;
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
    const auto shapeDynamicSizes = trimToShapeDynamicSizes(memrefType, adaptor.getDynamicSizes());
    getMemRefDescriptorSizes(loc, memrefType, shapeDynamicSizes, rewriter, shape, strides, sizeBytes);
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMPointerType::get(ctx));
    auto moduleOp = vpux::getModuleOp(origOp);
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();
    auto numArgs = funcOp.getNumArguments();
    auto context = funcOp.getArgument(GET_ARG_INDEX_CONTEXT(numArgs));
    auto execContext = funcOp.getArgument(GET_ARG_INDEX_COMMAND_EXECUTION_CONTEXT(numArgs));
    auto allocatedPtr = createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_alloc",
                                             {sizeBytes, context, execContext}, returnType)
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
    mlir::SmallVector<mlir::Value, 4> shapeDynamicSizes;
    auto memRefDesc = mlir::MemRefDescriptor(targetPtr);
    for (auto dim : llvm::enumerate(targetMemrefType.getShape())) {
        if (dim.value() == mlir::ShapedType::kDynamic) {
            shapeDynamicSizes.push_back(memRefDesc.size(rewriter, loc, dim.index()));
        }
    }
    getMemRefDescriptorSizes(origOp.getLoc(), targetMemrefType, shapeDynamicSizes, rewriter, shape, strides, sizeBytes);
    // Extract buffer pointers from memref descriptors
    auto src = getBufferStartAddress(rewriter, loc, sourcePtr);
    auto target = getBufferStartAddress(rewriter, loc, targetPtr);

    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();

    // As of today, copy op should be considered as a sub graph
    // So, command list reset/submission is required for the copy op
    increaseCommandListIndex(origOp, rewriter, commandListIndexState);

    auto cmdList = commandListIndexState.getCommandList(funcOp);
    auto moduleOp = vpux::getModuleOp(origOp);
    createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_append_memory_copy", {src, target, sizeBytes, cmdList},
                         returnType);

    createSubmitCommandList(rewriter, moduleOp, funcOp, commandListIndexState);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

template <typename AsyncOp>
class AsyncOpRewriter final : public mlir::OpRewritePattern<AsyncOp> {
public:
    AsyncOpRewriter(mlir::MLIRContext* ctx, const mlir::LLVMTypeConverter& typeConverter, mlir::PatternBenefit benefit,
                    CommandListIndexState& cmdListIndexState, ModelIOManager& ioManager, Logger log)
            : mlir::OpRewritePattern<AsyncOp>(ctx, benefit),
              _typeConverter(typeConverter),
              commandListIndexState(cmdListIndexState),
              modelIOManager(ioManager),
              _log(std::move(log)) {
        this->setDebugName("AsyncOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const final;
    const mlir::LLVMTypeConverter& _typeConverter;
    CommandListIndexState& commandListIndexState;
    ModelIOManager& modelIOManager;
    Logger _log;
};

template <typename AsyncOp>
mlir::LogicalResult AsyncOpRewriter<AsyncOp>::matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const {
    auto moduleOp = vpux::getModuleOp(origOp);
    auto parentFuncOp = origOp->template getParentOfType<mlir::func::FuncOp>();

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
                createSubmitCommandList(rewriter, moduleOp, parentFuncOp, commandListIndexState);
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

                    const auto callResultCount = callOp.getNumResults();
                    const auto callInputCount = callOp.getNumOperands() - callResultCount;
                    const auto funcName = funcOp.getSymName();

                    const int64_t operandIndex = static_cast<int64_t>(callInputCount) + static_cast<int64_t>(index);
                    if (operandIndex < 0 || operandIndex >= static_cast<int64_t>(callOp.getNumOperands())) {
                        _log.error("Invalid operand index {0} for '{1}' (callInputCount={2}, callResultCount={3}, "
                                   "index={4})",
                                   operandIndex, funcName, callInputCount, callResultCount, index);
                        return mlir::failure();
                    }

                    auto operand = *(callOp.getOperands().begin() + static_cast<unsigned int>(operandIndex));
                    for (auto u : users) {
                        if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(u)) {
                            viewOp.setOperand(0, operand);
                        } else if (auto copyOp = mlir::dyn_cast<mlir::memref::CopyOp>(u)) {
                            copyOp.setOperand(0, operand);
                        } else if (auto nestedCallOp = mlir::dyn_cast<Core::NestedCallOp>(u)) {
                            // Always propagate the actual value to Core::NestedCallOp users,
                            // regardless of whether the await is inside a scf.for or not.
                            // Without this, Core::NestedCallOp operands inside a later
                            // async.execute are left pointing at a dead await result, causing
                            // the framework to create zero-operand unrealized_conversion_cast
                            // materializations that survive past the conversion and trigger
                            // "unresolved materialization" errors.
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
                        } else if (mlir::isa<mlir::UnrealizedConversionCastOp, mlir::async::YieldOp>(u)) {
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

        createSubmitCommandList(rewriter, moduleOp, parentFuncOp, commandListIndexState);
        rewriter.eraseOp(origOp);
        return mlir::success();
    } else if (mlir::isa<mlir::async::AwaitAllOp>(origOp)) {
        createSubmitCommandList(rewriter, moduleOp, parentFuncOp, commandListIndexState);
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
    if (origOp->hasAttr("no_reset_cmdlist") == false) {
        increaseCommandListIndex(origOp, rewriter, commandListIndexState);
    }

    rewriter.eraseOp(origOp);
    return mlir::success();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::ExecuteOp>::matchAndRewrite(mlir::async::ExecuteOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto loc = origOp.getLoc();
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();
    auto moduleOp = vpux::getModuleOp(origOp);

    auto numArgs = funcOp.getNumArguments();
    auto umdContext = funcOp.getArgument(GET_ARG_INDEX_CONTEXT(numArgs));
    auto device = funcOp.getArgument(GET_ARG_INDEX_DEVICE(numArgs));
    auto ddiTable = funcOp.getArgument(GET_ARG_INDEX_DDI_TABLE(numArgs));
    auto commandQueue = funcOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));
    auto executionContext = funcOp.getArgument(GET_ARG_INDEX_COMMAND_EXECUTION_CONTEXT(numArgs));

    // note: needs to calculate the size of the kernel function after serialization of the core.NestedModule
    if (origOp->template getParentOfType<mlir::scf::ForOp>() == nullptr) {
        increaseCommandListIndex(origOp, rewriter, commandListIndexState);
    }

    // Process kernels
    std::map<std::string, std::tuple<HostExec::BinaryOp, mlir::Value, mlir::Value>> kernels;
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
            mlir::Value kernelName;
            auto iter = kernels.find(calleeNameAttr.str());

            if (iter != kernels.end()) {
                kernelGlobal = std::get<1>(iter->second);
                kernelName = std::get<2>(iter->second);
            } else {
                auto name = callee.getLeafReference().getValue();
                auto nameAttr = mlir::StringAttr::get(origOp.getContext(), std::string(name) + "_kernel");
                kernelName = mlir::LLVM::createGlobalString(loc, rewriter, name, nameAttr.getValue(),
                                                            mlir::LLVM::Linkage::Internal);
                kernelGlobal = mlir::LLVM::createGlobalString(loc, rewriter, nameAttr.getValue(), object.getObject(),
                                                              mlir::LLVM::Linkage::Internal);
                kernels[calleeNameAttr.str()] = std::make_tuple(kernelBinary, kernelGlobal, kernelName);
            }

            // Gather inputs and outputs
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

            // Process tensors using ModelIOManager
            if (!modelIOManager.processTensors(rewriter, loc, kernelInputs, kernelOutputs, _typeConverter, funcOp)) {
                return mlir::failure();
            }

            auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
            auto cmdList = commandListIndexState.getCommandList(funcOp);
            auto commandListIndex = commandListIndexState.getCommandListIndex(rewriter);
            createLLVMFuncCallOp(
                    rewriter, moduleOp, "npu_level_zero_execute_graph",
                    {modelIOManager.getInputBufferDescs(), numInputs, modelIOManager.getOutputBufferDescs(), numOutputs,
                     kernelName, kernelGlobal, kernelSize, umdContext, device, ddiTable, cmdList, commandListIndex,
                     commandQueue, executionContext},
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
    explicit ConvertToLLVMUMDCallsPass(bool enablePipelinedCmdListRecording, Logger log)
            : enablePipelinedCmdListRecording(enablePipelinedCmdListRecording) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    bool enablePipelinedCmdListRecording;
};

SmallVector<mlir::func::FuncOp> addFuncParamsForUmdForCallers(mlir::func::FuncOp callee) {
    SmallVector<mlir::func::FuncOp> ret;
    auto moduleOp = vpux::getModuleOp(callee);
    for (auto caller : moduleOp.getOps<mlir::func::FuncOp>()) {
        if (callee == caller) {
            continue;
        }

        auto callOps = getCallSites(callee, caller);
        if (callOps.empty()) {
            continue;
        }
        // if callee requires UMD params, then its caller also needs them as they provided by NPU plugin as a part of
        // the fat-binary execution contract
        addFuncParamsForUmdFuncCall(caller);

        // get those UMD params as they have to be passed as argumenst for all callee calls
        SmallVector<mlir::Value> umdFuncArgs;
        auto numArgs = caller.getNumArguments();
        VPUX_THROW_WHEN(numArgs < vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT,
                        "Func: {0} must have at least arguments: {1}, got: {2}", caller.getName(),
                        vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT, numArgs);
        for (size_t i = HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_CONTEXT; i < HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT;
             i++) {
            size_t funcArgIndex = numArgs - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + i;
            auto umdArg = caller.getArgument(funcArgIndex);
            umdFuncArgs.push_back(umdArg);
        }

        // Add those UMD arguments to every call of callee
        for (auto callOp : callOps) {
            auto builder = mlir::OpBuilder(callOp);
            SmallVector<mlir::Value> newCallArgs(callOp.getOperands());
            newCallArgs.append(umdFuncArgs);
            builder.create<mlir::func::CallOp>(appendLoc(callOp.getLoc(), "_extend_umd_args"), callee, newCallArgs);
        }

        for (auto callOp : callOps) {
            callOp->erase();
        }
        ret.push_back(caller);
    }
    return ret;
}

std::tuple<SmallVector<mlir::func::FuncOp>, mlir::DenseSet<mlir::func::FuncOp>> addFuncUMDParamsForAllCallers(
        ArrayRef<mlir::func::FuncOp> callees, Logger log) {
    SmallVector<mlir::func::FuncOp> topLevelCallers;
    mlir::DenseSet<mlir::func::FuncOp> allCallers;
    for (auto callee : callees) {
        log.debug("Check whether the target: {0} has caller contexts", callee.getName());
        auto callerFuncs = addFuncParamsForUmdForCallers(callee);
        log.debug("Added UMD func arguments to all found callers: {0} of the target: {1}", callerFuncs.size(),
                  callee.getName());
        allCallers.insert(callerFuncs.begin(), callerFuncs.end());
        while (!callerFuncs.empty()) {
            log.trace("More callers: {0} to process", callerFuncs.size());
            auto callee = *callerFuncs.begin();
            callerFuncs.erase(callerFuncs.begin());

            auto nextTierCallers = addFuncParamsForUmdForCallers(callee);
            std::copy(nextTierCallers.begin(), nextTierCallers.end(), std::back_inserter(callerFuncs));
            allCallers.insert(nextTierCallers.begin(), nextTierCallers.end());
            log.debug("Added UMD func arguments to parent functions: {0} of the caller: {1}, elapsed caller contexts: "
                      "{2} to process",
                      nextTierCallers.size(), callee.getName(), callerFuncs.size());

            if (nextTierCallers.empty()) {
                topLevelCallers.push_back(callee);
            }
        }
    }
    return {topLevelCallers, allCallers};
}

void insertFinalCmdListSubmissionAfterOp(mlir::scf::ForOp op, mlir::func::FuncOp parentFuncOp) {
    auto numArgs = parentFuncOp.getNumArguments();
    VPUX_THROW_UNLESS(numArgs >= HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT,
                      "Cannot insert final cmd list submission into the function: {0} which has not enough arguments: "
                      "{1}, required at least: {2}",
                      parentFuncOp.getName(), numArgs, HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT);
    auto cmdQueue = parentFuncOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));
    auto execContext = parentFuncOp.getArgument(GET_ARG_INDEX_COMMAND_EXECUTION_CONTEXT(numArgs));
    auto cmdList = parentFuncOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));
    auto fence = parentFuncOp.getArgument(GET_ARG_INDEX_COMMAND_FENCE(numArgs));
    auto event = parentFuncOp.getArgument(GET_ARG_INDEX_COMMAND_EVENT(numArgs));
    auto builder = mlir::OpBuilder(op);
    auto moduleOp = vpux::getModuleOp(parentFuncOp);
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(op.getContext()));
    builder.setInsertionPointAfter(op);
    createLLVMFuncCallOp(builder, moduleOp, "npu_level_zero_submit_commandlist",
                         {cmdList, cmdQueue, fence, event, execContext}, returnType);
}

bool removeFinalCmdListSubmission(mlir::func::FuncOp funcOp) {
    mlir::LLVM::CallOp finalCmdListSubmissionCall;
    funcOp->walk([&finalCmdListSubmissionCall](mlir::LLVM::CallOp call) {
        if (call.getCallee() == "npu_level_zero_submit_commandlist") {
            finalCmdListSubmissionCall = call;
        }
    });
    if (finalCmdListSubmissionCall) {
        finalCmdListSubmissionCall.erase();
        return true;
    }
    return false;
}

void finalizeCmdListSubmissionIfRequired(SmallVector<mlir::func::FuncOp>& targetFunctions,
                                         ArrayRef<mlir::func::FuncOp> topLevelFuncCallers,
                                         mlir::DenseSet<mlir::func::FuncOp>& allFuncCallers, Logger log) {
    /*
     * Each function in topLevelFuncCallers invokes one or more actual inference-executor functions.
     * If a top-level function invokes them repeatedly (for example, inside an scf-for loop), internal
     * command lists must not be submitted from within the inference-executor functions themselves,
     * since the submission would occur on every invocation.
     *
     * Multiple submissions of the same command list may cause incorrect behavior when:
     *   - a fence/event is signaled only on the first submission, or
     *   - each subsequent submission invalidates the previous one.
     *
     * A typical symptom is corrupted data in a batched output tensor of size N, where rows with
     * indices n < N - 1 contain garbage because only the last batch item was actually processed.
     *
     * This finalization routine scans all topLevelFuncCallers for such scf-for loops, identifies
     * the last forOp operation, and inserts the final command list submission after the last callOp
     * to the corresponding target function from targetFunctions.
     *
     * At the same time, the final command list submission is removed from the bodies of all
     * targetFunctions.
     *
     * Since all targetFunctions and their corresponding callOps are processed consistently,
     * the transformation order is irrelevant: we can either update all callers first and then
     * process target functions, or handle each caller/target pair sequentially.
     * The former approach was chosen for simplicity.
     */
    log.trace("Amend UMD callers: {0} to take over cmd list final submission", topLevelFuncCallers.size());
    std::function<bool(mlir::func::CallOp op)> isCallOpSuitableToOptimize = [&targetFunctions,
                                                                             &allFuncCallers](mlir::func::CallOp op) {
        mlir::func::FuncOp calledFunction = vpux::getCalledFunction(op);
        if (auto it = std::find(targetFunctions.begin(), targetFunctions.end(), calledFunction);
            it != targetFunctions.end()) {
            VPUX_THROW_UNLESS(it->getNumArguments() > HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT,
                              "Target function: {0} is expected to have arguments count more than: {1}, has got: {2}",
                              it->getName(), HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT, it->getNumArguments());
            return true;
        }
        if (auto it = allFuncCallers.find(calledFunction); it != allFuncCallers.end()) {
            VPUX_THROW_UNLESS(
                    it->getNumArguments() > HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT,
                    "Intermediate function: {0} is expected to have arguments count more than: {1}, has got: {2}",
                    it->getName(), HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT, it->getNumArguments());
            return true;
        }
        return false;
    };

    bool finalCmdListSubmissionRequired = false;
    for (auto f : topLevelFuncCallers) {
        log.debug("Check whether the caller function: {0} must finalize command list submission", f.getName());
        mlir::scf::ForOp finalForOp;
        f->walk([&finalForOp, isCallOpSuitableToOptimize](mlir::scf::ForOp forOp) {
            forOp->walk([&finalForOp, forOp, isCallOpSuitableToOptimize](mlir::func::CallOp callee) {
                if (isCallOpSuitableToOptimize(callee)) {
                    finalForOp = forOp;
                }
            });
        });
        if (finalForOp) {
            log.debug("Finalize cmd list submission for func: {0}", f.getName());
            insertFinalCmdListSubmissionAfterOp(finalForOp, f);
            finalCmdListSubmissionRequired = true;
        }
    }

    if (finalCmdListSubmissionRequired) {
        log.trace("Try to remove redundant cmd list submission from target functions: {0}", targetFunctions.size());
        for (auto funcOp : targetFunctions) {
            bool ret = removeFinalCmdListSubmission(funcOp);
            log.debug("Removal final cmd list submission from the function: {0}, res: {1}", funcOp.getName(), ret);
        }
    }
}

void ConvertToLLVMUMDCallsPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = &getContext();

    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);
    mlir::LowerToLLVMOptions options(ctx);
    options.useBarePtrCallConv = true;
    mlir::LLVMTypeConverter typeConverter(ctx, options);

    // Update ReturnOp for all host compile functions, but add additional arguments
    // for umd calls only for target functions
    for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
        if (config::isPureHostCompileFunc(funcOp)) {
            updateFuncTerminator(funcOp);
        }
    }

    SmallVector<mlir::func::FuncOp> targetFunctions;
    for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
        if (vpux::HostExec::isHostCompileInferenceExecFunc(funcOp)) {
            addFuncParamsForUmdFuncCall(funcOp);
            targetFunctions.push_back(funcOp);
            _log.info("Add UMD func arguments to the target: {0}, total targets processed: {1}", funcOp.getName(),
                      targetFunctions.size());
        }
    }
    if (targetFunctions.empty()) {
        _log.info("No any candidate functions detected, process the entryPoint only");
        auto entryPointFuncOp = net::getMainFunc(module);
        addFuncParamsForUmdFuncCall(entryPointFuncOp);
        targetFunctions.push_back(entryPointFuncOp);
    }

    SmallVector<mlir::func::FuncOp> topLevelFuncCallers;
    mlir::DenseSet<mlir::func::FuncOp> allFuncCallers;
    std::tie(topLevelFuncCallers, allFuncCallers) = addFuncUMDParamsForAllCallers(targetFunctions, _log);
    VPUX_THROW_WHEN(targetFunctions.size() != 1,
                    "Pass: {0} supports only processing of a single function"
                    " met the 'isHostCompileInferenceExecFunc' condition, got: {1}",
                    getName(), targetFunctions.size());
    mlir::func::FuncOp funcOpToConvert = targetFunctions[0];

    // Convert "output_shape" body's memref operations to llvm ones
    auto outputShapeFuncOp = module.lookupSymbol<mlir::func::FuncOp>("output_shape");
    if (outputShapeFuncOp != nullptr) {
        auto ctx = outputShapeFuncOp->getContext();
        mlir::LLVMConversionTarget outputShapeTarget(*ctx);
        mlir::RewritePatternSet outputShapePatterns(ctx);
        mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, outputShapePatterns);
        if (failed(applyPartialConversion(outputShapeFuncOp, outputShapeTarget, std::move(outputShapePatterns)))) {
            signalPassFailure();
            return;
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
    // Apply special conversions the target functions only.
    target.markOpRecursivelyLegal<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        bool isLegal = true;
        for (auto f : targetFunctions) {
            isLegal = isLegal && (funcOp.getSymName() != f.getSymName());
        }
        return isLegal;
    });
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<mlir::cf::AssertOp>();

    CommandListIndexState commandListIndexState;
    commandListIndexState.initialize(funcOpToConvert, enablePipelinedCmdListRecording);
    auto commandListNumber = HostExec::getHostCompileInferenceExpectedCommandListsNumber(funcOpToConvert);
    if (commandListNumber.has_value()) {
        enablePipelinedCmdListRecording = *commandListNumber != 1;
        _log.info("Desired amount of command lists requested: {0}, override enablePipelinedCmdListRecording value: {1}",
                  *commandListNumber, enablePipelinedCmdListRecording);
    }
    commandListIndexState.initialize(funcOpToConvert, enablePipelinedCmdListRecording);

    ModelIOManager ioManager(module, funcOpToConvert, _log);

    patterns.add<LvlZeroMemoryCopyLowering>(typeConverter, commandListIndexState);
    patterns.add<LvlZeroAllocLowering>(typeConverter);
    patterns.add<AsyncOpRewriter<mlir::async::AddToGroupOp>>(ctx, typeConverter, vpux::benefitHigh,
                                                             commandListIndexState, ioManager, _log);
    patterns.add<AsyncOpRewriter<mlir::async::CreateGroupOp>>(ctx, typeConverter, vpux::benefitHigh,
                                                              commandListIndexState, ioManager, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitAllOp>>(ctx, typeConverter, vpux::benefitHigh, commandListIndexState,
                                                           ioManager, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitOp>>(ctx, typeConverter, vpux::benefitHigh, commandListIndexState,
                                                        ioManager, _log);

    // Note: ExecuteOp is a special case, a few conditions apply which is why it is the last pattern,
    // 1 npu_level_zero_execute_graph that all inputs and outputs are converted to LLVM types.
    // 2.It will have successor and predecessor dependencies on the other async and memref operations therefore those
    // operations should be removed or converted before this pattern is applied.

    patterns.add<AsyncOpRewriter<mlir::async::ExecuteOp>>(ctx, typeConverter, vpux::benefitLow, commandListIndexState,
                                                          ioManager, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }

    for (auto funcOp : targetFunctions) {
        commandListIndexState.finalizeCommandListIndex(module, ctx, funcOp);
    }

    // remove all BinaryOp as they were converted into global variables
    auto binaryOps = to_small_vector(module.getOps<HostExec::BinaryOp>());
    for (auto binaryOp : binaryOps) {
        binaryOp.getOperation()->erase();
    }

    // remove redundant submit command list calls if useSingleCommandList is true
    if (enablePipelinedCmdListRecording == false) {
        bool found = false;
        for (auto funcOp : targetFunctions) {
            auto callOps = to_small_vector(funcOp.getOps<mlir::LLVM::CallOp>());
            for (int32_t i = callOps.size() - 1; i >= 0; --i) {
                auto& callOp = callOps[i];
                mlir::FlatSymbolRefAttr calleeAttr = callOp.getCalleeAttr();
                llvm::StringRef funcName = calleeAttr.getValue();
                if (funcName.str() == "npu_level_zero_submit_commandlist") {
                    if (found == false) {
                        // keep the last command list only
                        found = true;
                    } else {
                        // remove submit command list calls
                        callOps[i].getOperation()->erase();
                    }
                }
            }
        }
        // append a final submit command list call in case these targetFunctions are invoked repeatedly
        finalizeCmdListSubmissionIfRequired(targetFunctions, topLevelFuncCallers, allFuncCallers, _log);
    }
}
}  // namespace

//
// createConvertToLLVMUMDCallsPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createConvertToLLVMUMDCallsPass(bool enablePipelinedCmdListRecording,
                                                                            Logger log) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    const auto overrideEnablePipelinedCmdListOption =
            std::getenv(OVERRIDE_ENABLE_PIPELINED_COMMANDLIST_RECORDING.data());
    if (overrideEnablePipelinedCmdListOption != nullptr) {
        log.warning("The environment variable {0} is set to {1}, which overrides the default behavior of pipelined "
                    "command list recording.",
                    OVERRIDE_ENABLE_PIPELINED_COMMANDLIST_RECORDING, overrideEnablePipelinedCmdListOption);

        enablePipelinedCmdListRecording = std::string(overrideEnablePipelinedCmdListOption) != "0";
    }
#endif

    log.info("Pipelined command list recording is {0} for the ConvertToLLVMUMDCallsPass.",
             enablePipelinedCmdListRecording ? "enabled" : "disabled");
    return std::make_unique<ConvertToLLVMUMDCallsPass>(enablePipelinedCmdListRecording, log);
}
