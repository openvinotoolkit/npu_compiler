//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_ADAPTLLVMFUNCSFORSHAVE
#define GEN_PASS_DEF_ADAPTLLVMFUNCSFORSHAVE
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// AdaptLLVMFuncsForShaveBase
//

class AdaptLLVMFuncsForShavePass final :
        public ShaveCodeGen::impl::AdaptLLVMFuncsForShaveBase<AdaptLLVMFuncsForShavePass> {
public:
    explicit AdaptLLVMFuncsForShavePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;

    void adaptFuncForShave(mlir::SymbolRefAttr funcSym);
    size_t getSizeOfLLVMType(mlir::Type type);

    mlir::ModuleOp _swModule;
};

// Returns the size in bytes of different LLVM/MLIR types
size_t AdaptLLVMFuncsForShavePass::getSizeOfLLVMType(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, size_t>(type)
            .Case<mlir::LLVM::LLVMPointerType>([&](mlir::LLVM::LLVMPointerType) {
                return sizeof(uint32_t);  // shave requires 32 bit addresses
            })
            .Case<mlir::IntegerType>([](mlir::IntegerType intType) {
                return llvm::divideCeil(intType.getWidth(), 8);
            })
            .Case<mlir::FloatType>([](mlir::FloatType floatType) {
                return llvm::divideCeil(floatType.getWidth(), 8);
            })
            .Default([](auto) {
                VPUX_THROW("Unsupported input type!");
                return 0;
            });
}

void AdaptLLVMFuncsForShavePass::adaptFuncForShave(mlir::SymbolRefAttr funcSym) {
    auto funcOp = _swModule.lookupSymbol<mlir::FunctionOpInterface>(funcSym.getLeafReference());

    // Compute byte offset of each func argument
    mlir::DenseMap<mlir::BlockArgument, size_t> paramToByteOffsetMap;
    size_t offset = 0;
    for (auto funcArg : funcOp.getArguments()) {
        auto argTypeSize = getSizeOfLLVMType(funcArg.getType());
        paramToByteOffsetMap.try_emplace(funcArg, offset);
        offset += argTypeSize;
    }

    auto symName = funcOp.getName().str();
    mlir::SymbolTable symbolTable(_swModule);
    if (failed(symbolTable.rename(funcOp, llvm::formatv("__impl_{0}", symName).str()))) {
        signalPassFailure();
        return;
    }

    auto ptrType = mlir::LLVM::LLVMPointerType::get(&getContext());
    auto builder = mlir::OpBuilder(funcOp);

    // Generate a wrapper for our function that matches the management
    // kernel CC.
    SmallVector<mlir::NamedAttribute> attributes;
    auto entryFunTy = mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(&getContext()), {ptrType});

    auto entryFun =
            builder.create<mlir::LLVM::LLVMFuncOp>(funcOp.getLoc(), symName, entryFunTy, mlir::LLVM::Linkage::External,
                                                   /*dsoLocal=*/true, mlir::LLVM::CConv::C,
                                                   /*comdat=*/nullptr, attributes);
    builder.setInsertionPointToStart(entryFun.addEntryBlock());

    auto fullyEncompassingPtrArg = entryFun.getBlocks().front().getArgument(0);

    auto int8Type = vpux::getInt8Type(&getContext());
    auto int32Type = vpux::getInt32Type(&getContext());

    // All the arguments are packed into one single pointer-type arg, we need to manually extract the
    // underlying data. This is accomplished via pointer arithmetic and creating GetElementPointer & Load pairs for each
    // of the arguments.
    SmallVector<mlir::Value> callArgs;
    for (auto funcArg : funcOp.getArguments() | indexed) {
        auto offsetConstantOp = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), int32Type,
                                                                       paramToByteOffsetMap[funcArg.value()]);
        auto elementGep = builder.create<mlir::LLVM::GEPOp>(funcOp.getLoc(), ptrType, int8Type, fullyEncompassingPtrArg,
                                                            offsetConstantOp.getRes());
        auto elementLoad =
                builder.create<mlir::LLVM::LoadOp>(funcOp.getLoc(), funcArg.value().getType(), elementGep.getRes());
        callArgs.push_back(elementLoad);
    }

    // Add a call to our function.
    builder.create<mlir::LLVM::CallOp>(funcOp.getLoc(), mlir::cast<mlir::LLVM::LLVMFuncOp>(funcOp), callArgs);
    builder.create<mlir::LLVM::ReturnOp>(funcOp.getLoc(), mlir::ValueRange{});

    // The initial function should not be referenced from anywhere outside the SW module.
    // We therefore make it internal to enable inlining.
    mlir::cast<mlir::LLVM::LLVMFuncOp>(funcOp).setLinkage(mlir::LLVM::Linkage::Internal);
}

void AdaptLLVMFuncsForShavePass::safeRunOnModule() {
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, func);
    _swModule = VPUIP::getVPUSWModule(moduleOp, _log);

    func.walk([&](VPUIP::SwKernelOp swKernelOp) {
        auto swLayerFuncSym = swKernelOp.getKernelFunction();
        adaptFuncForShave(swLayerFuncSym);
    });
}

}  // namespace

//
// createAdaptLLVMFuncsForShavePass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createAdaptLLVMFuncsForShavePass(Logger log) {
    return std::make_unique<AdaptLLVMFuncsForShavePass>(log);
}
