//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// #include "vpux/compiler/dialect/VPUIP/utils.hpp"

// #include "vpux/compiler/core/attributes/shape.hpp"
// #include "vpux/compiler/core/layers.hpp"
// #include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux {
namespace VPUIP {
namespace {

mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }
    return innerModule;
}

}  // namespace

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, StringRef builtInFunctionName,
                                          const SmallVector<mlir::Type>& inputTypes, StringRef kernelEntryName,
                                          StringRef kernelSourceFileName, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    auto vpuswModule = getVPUSWModule(module, log);

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().getValue(), {builtInFlatFunction});

    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::FuncOp>(builtInFunctionName);
    if (prebuiltFunction) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto buildInOp = innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    buildInOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    buildInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    buildInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, IERT::LayerOpInterface origOp,
                                          const IERT::KernelInfo& kernelInfo, const Logger& log) {
    OpBuilderLogger builderLog(log);

    SmallString builtInFunctionName{"builtin_"};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    const auto convertToUnrankedType = [](mlir::Value operand) -> mlir::Type {
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
    };

    auto& args = kernelInfo.args;
    auto opInputs = origOp.getInputs();
    auto opResults = origOp->getResults();

    SmallVector<mlir::Type> inputTypes;
    std::transform(opInputs.begin(), opInputs.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(opResults.begin(), opResults.end(), std::back_inserter(inputTypes), convertToUnrankedType);
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [](mlir::Attribute arg) {
        return arg.getType();
    });

    return createBuiltInFunction(module, builtInFunctionName, inputTypes, kernelInfo.entryName,
                                 kernelInfo.sourceFileName, log);
}

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName{"nnActEntry"};

    // check if runtimeKernel already created
    auto runtimeKernelFunction = vpuswModule.lookupSymbol<mlir::FuncOp>(runtimeKernelName);
    if (runtimeKernelFunction) {
        log.trace("Found builtin function: {0}", runtimeKernelName);
        return;
    }

    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    // creating runtime kernel function
    const auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto runtimeFunctionOp =
            innerModuleBuilder.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), runtimeKernelName, funcType);

    // modifying attributes
    runtimeFunctionOp.sym_visibilityAttr(mlir::StringAttr::get(ctx, "private"));

    runtimeFunctionOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, runtimeKernelEntryName));

    log.trace("Added runtime kernel function: {0}", runtimeKernelEntryName);

    // creating name symbol
    auto runtimeFlatSym = mlir::SymbolRefAttr::get(ctx, runtimeKernelName);
    auto runtimeSym = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().getValue(), {runtimeFlatSym});

    static constexpr int64_t defaultStackSize = 4096;

    SmallVector<int64_t> stacksArray(4, defaultStackSize);

    //  adding runtime kernel configuration - stacks, etc
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    moduleBuilder.create<VPURT::SWRunTimeOp>(mlir::UnknownLoc::get(ctx), runtimeSym, getIntArrayAttr(ctx, stacksArray));
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  ArrayRef<mlir::Attribute> args, const Logger& log) {
    OpBuilderLogger builderLog(log);
    auto* ctx = swKernelOp.getContext();
    auto& bodyRegion = swKernelOp.body();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding args of IERT operation as constants
    SmallVector<mlir::arith::ConstantOp> constantArgs;
    for (auto&& arg : args) {
        constantArgs.push_back(swKernelBlockBuilder.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(ctx), arg));
    }

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);
    fetchOperands(constantArgs);

    swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands));
}

}  // namespace VPUIP
}  // namespace vpux
