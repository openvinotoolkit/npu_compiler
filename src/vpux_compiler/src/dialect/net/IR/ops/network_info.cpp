//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>

using namespace vpux;

inline size_t getFuncArgCount(mlir::func::FuncOp netFunc, size_t netInfoArgCount, const bool hostCompileMode) {
    auto args = netFunc.getArguments();
    if (hostCompileMode) {
        // after createConvertToLLVMUMDCallsPass,
        // more input arguments will be added for L0 wrapper function calls in LLVM code
        auto maxNetFuncArgCount = vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + netInfoArgCount;
        if (maxNetFuncArgCount == args.size()) {
            return maxNetFuncArgCount - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT;
        }
    }
    return args.size();
}

static mlir::LogicalResult checkFunctionPrototype(net::NetworkInfoOp cnnOp, mlir::func::FuncOp netFunc,
                                                  SmallVector<net::DataInfoOp>& inputsInfo,
                                                  SmallVector<net::DataInfoOp>& outputsInfo,
                                                  SmallVector<net::DataInfoOp>& profilingOutputsInfo,
                                                  const bool resultVerificationDisabled, const bool hostCompileMode) {
    const auto netFuncType = netFunc.getFunctionType();
    const auto args = netFunc.getArgumentTypes();

    const auto netInfoArgCount = inputsInfo.size() + outputsInfo.size() + profilingOutputsInfo.size();
    const auto netInfoOpOutputCount = outputsInfo.size() + profilingOutputsInfo.size();
    const auto netFuncNumResults = netFuncType.getNumResults();
    if (!resultVerificationDisabled && netFuncNumResults != netInfoOpOutputCount) {
        return errorAt(cnnOp, "entryPoint '@{0}' outputs count '{1}' doesn't match userOutputs count '{2}'",
                       cnnOp.getEntryPoint(), netFuncNumResults, outputsInfo.size());
    }

    // the number of func args which are not added for LLVM host execution
    auto netFuncArgsCounts = getFuncArgCount(netFunc, netInfoArgCount, hostCompileMode);
    auto argsEnd = args.begin() + netFuncArgsCounts;

    const auto isArgsTensorized = std::all_of(args.begin(), argsEnd, [](mlir::Type type) {
        return mlir::isa<mlir::bufferization::TensorLikeType>(type);
    });
    const auto isTensorized = (netFuncArgsCounts == inputsInfo.size()) && isArgsTensorized;
    if (isTensorized) {
        return mlir::success();
    }

    const auto isArgsBufferized = std::all_of(args.begin(), argsEnd, [](mlir::Type type) {
        return mlir::isa<mlir::bufferization::BufferLikeType>(type);
    });

    const auto isSemiBufferized = (netFuncArgsCounts == inputsInfo.size()) && isArgsBufferized;
    if (isSemiBufferized) {
        return mlir::success();
    }

    const auto isBufferized = (netFuncArgsCounts == netInfoArgCount) && isArgsBufferized;

    if (resultVerificationDisabled) {
        // E#69730: find cleaner representation of the FuncOp with no args
        return mlir::success();
    } else if (isBufferized) {
        mlir::LogicalResult res = mlir::success();
        netFunc.walk([&inputsInfo, &netFunc, &res](mlir::func::ReturnOp op) {
            const auto operands = op.getOperands();
            for (const auto ind : irange(operands.size())) {
                const auto rawInd = checked_cast<unsigned>(inputsInfo.size() + ind);

                const auto output = operands[ind];
                const auto outputBuffer = netFunc.getArgument(rawInd);

                const ValueSourceInfo info(output);
                if (info.getRoot(output) != outputBuffer) {
                    op.emitError() << "function output at index=" << ind
                                   << " should be an alias of the output buffer, but it's not";
                    res = mlir::failure();
                    break;
                }
            }
        });

        return res.failed() ? res : mlir::success();
    }

    return errorAt(cnnOp,
                   "entryPoint '@{0}' has invalid state. inputs count '{1}', results count '{2}', user inputs "
                   "count '{3}', user outputs count '{4}'",
                   cnnOp.getEntryPoint(), netFuncType.getNumInputs(), netFuncType.getNumResults(), inputsInfo.size(),
                   outputsInfo.size());
}

void net::NetworkInfoOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                               mlir::FlatSymbolRefAttr entryPoint, bool withProfiling) {
    build(builder, state, entryPoint, nullptr, static_cast<unsigned>(withProfiling ? 1 : 0));
}

mlir::LogicalResult net::NetworkInfoOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    auto& cnnOp = *this;
    const bool hostCompileMode = config::getCompilationMode(cnnOp) == config::CompilationMode::HostCompile;
    auto netFunc = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, getEntryPointAttr());

    if (netFunc == nullptr) {
        if (hostCompileMode) {
            // For host compilation, mlir::func::FuncOp is transformed to LLVMFuncOp in ConvertFuncToLLVMPass
            // So, if netFunc is null and llvmFuncOp is not null, skip netinfo verification
            // Later, revisit here if an additional pass is added to remove netinfo or transform it to something
            // global binary
            auto llvmFuncOp = symbolTable.lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(*this, getEntryPointAttr());
            if (llvmFuncOp != nullptr) {
                return mlir::success();
            }
        }
        return errorAt(*this, "entryPoint '@{0}' doesn't refer to existing Function", getEntryPoint());
    }

    auto inputsInfo = to_small_vector(this->getInputsInfo().getOps<net::DataInfoOp>());
    auto outputsInfo = to_small_vector(this->getOutputsInfo().getOps<net::DataInfoOp>());
    SmallVector<net::DataInfoOp> profilingOutputsInfo;
    if (!this->getProfilingOutputsInfo().empty()) {
        profilingOutputsInfo = to_small_vector(this->getProfilingOutputsInfo().front().getOps<net::DataInfoOp>());
    }

    const auto netFuncType = netFunc.getFunctionType();

    // E#69730: find cleaner representation of the FuncOp with no args
    const bool hoistedIOs = (netFuncType.getNumInputs() == 0) && (netFuncType.getNumResults() == 0);
    // Note: host compilation pipeline generate LLVM main function w/ no return value in ConvertToLLVMUMDCallsPass
    //       This is to alleviate output buffer verification for host compilation
    const bool resultVerificationDisabled = hoistedIOs || (hostCompileMode && (netFuncType.getResults().size() == 0));

    if (checkFunctionPrototype(cnnOp, netFunc, inputsInfo, outputsInfo, profilingOutputsInfo,
                               resultVerificationDisabled, hostCompileMode)
                .failed()) {
        return mlir::failure();
    }

    enum class PortType { Input, Output };

    const auto compareShape = [&cnnOp](NDTypeInterface funcType, NDTypeInterface userType, size_t ind,
                                       PortType portType) {
        const auto portString = portType == PortType::Input ? "input" : "output";
        if (funcType == nullptr) {
            return errorAt(cnnOp, "entryPoint '@{0}' {1} #{2} is not a 'NDTypeInterface'", cnnOp.getEntryPoint(),
                           portString, ind);
        }

        if (userType == nullptr) {
            return errorAt(cnnOp, "User {0} #{1} is not a 'NDTypeInterface'", portString, ind);
        }

        auto isDynamic = userType.getShape().isDynamic() || funcType.getShape().isDynamic();
        if (!isDynamic && funcType.getNumElements() != userType.getNumElements()) {
            return errorAt(cnnOp, "entryPoint '@{0}' {1} #{2} with type '{3}' is not compatible with user type '{4}'",
                           cnnOp.getEntryPoint(), portString, ind, funcType, userType);
        }

        return mlir::success();
    };

    if (hoistedIOs) {
        return mlir::success();
    }

    for (const auto ind : irange(inputsInfo.size())) {
        const auto funcType = mlir::dyn_cast<NDTypeInterface>(netFuncType.getInput(static_cast<uint32_t>(ind)));
        const auto userType = mlir::dyn_cast<NDTypeInterface>(inputsInfo[ind].getUserType());

        if (compareShape(funcType, userType, ind, PortType::Input).failed()) {
            return mlir::failure();
        }
    }

    if (!resultVerificationDisabled) {
        const auto args = netFunc.getArgumentTypes();
        const auto isArgsBufferized = std::all_of(args.begin(), args.end(), [](mlir::Type type) {
            return mlir::isa<mlir::bufferization::BufferLikeType>(type);
        });

        size_t argOffset = 0;
        ArrayRef<mlir::Type> outputTypes;
        if (isArgsBufferized && args.size() > inputsInfo.size()) {
            argOffset = inputsInfo.size();
            outputTypes = netFuncType.getInputs();
        } else {
            outputTypes = netFuncType.getResults();
        }

        for (const auto ind : irange(outputsInfo.size())) {
            const auto funcType = mlir::dyn_cast<NDTypeInterface>(outputTypes[ind + argOffset]);
            const auto userType = mlir::dyn_cast<NDTypeInterface>(outputsInfo[ind].getUserType());

            if (compareShape(funcType, userType, ind, PortType::Output).failed()) {
                return mlir::failure();
            }
        }
    }

    return mlir::success();
}

namespace {

mlir::LogicalResult verifyDataInfoRegion(mlir::Operation* op, mlir::Region& region, StringRef regionName) {
    if (region.getBlocks().size() != 1) {
        return errorAt(op, "'{0}' Region must contain exact 1 Block", regionName);
    }

    auto& allOps = region.front().getOperations();

    for (auto& infoOp : allOps) {
        if (!mlir::isa<net::DataInfoOp>(infoOp)) {
            return errorAt(op, "'{0}' Region must contain only DataInfo operations, got '{1}'", regionName,
                           infoOp.getName());
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult net::NetworkInfoOp::verify() {
    if (getEntryPointAttr() == nullptr) {
        return errorAt(*this, "entryPoint attribute is NULL");
    }

    if (mlir::failed(verifyDataInfoRegion(*this, getInputsInfo(), "inputInfo"))) {
        return mlir::failure();
    }
    if (mlir::failed(verifyDataInfoRegion(*this, getOutputsInfo(), "outputsInfo"))) {
        return mlir::failure();
    }

    auto outputsInfo = getOutputsDataInfo();

    if (outputsInfo.empty()) {
        return errorAt(*this, "Operation has no user outputs information");
    }

    return mlir::success();
}

size_t net::NetworkInfoOp::getNetInputsCount() {
    return getInputsInfo().front().getOperations().size();
}

SmallVector<net::DataInfoOp, 1> net::NetworkInfoOp::getInputsDataInfo() {
    return to_vector<1>(getInputsInfo().getOps<net::DataInfoOp>());
}

size_t net::NetworkInfoOp::getNetOutputsCount() {
    return getOutputsInfo().front().getOperations().size();
}

SmallVector<net::DataInfoOp, 1> net::NetworkInfoOp::getOutputsDataInfo() {
    return to_vector<1>(getOutputsInfo().getOps<net::DataInfoOp>());
}

size_t net::NetworkInfoOp::getProfilingOutputsCount() {
    if (!getProfilingOutputsInfo().empty() && !getProfilingOutputsInfo().front().empty()) {
        return getProfilingOutputsInfo().front().front().getOperations().size();
    }
    return 0;
}

SmallVector<net::DataInfoOp, 1> net::NetworkInfoOp::getProfilingOutputsDataInfo() {
    if (!getProfilingOutputsInfo().empty()) {
        return to_vector<1>(getProfilingOutputsInfo().front().getOps<net::DataInfoOp>());
    }
    return SmallVector<net::DataInfoOp, 1>();
}

void net::NetworkInfoOp::getFromModule(mlir::ModuleOp module, net::NetworkInfoOp& netInfo,
                                       mlir::func::FuncOp& netFunc) {
    auto netOps = to_small_vector(module.getOps<net::NetworkInfoOp>());

    VPUX_THROW_UNLESS(netOps.size() == 1,
                      "Can't have more than one 'net::NetworkInfoOp' Operation in Module, got '{0}'", netOps.size());

    netInfo = netOps.front();
    netFunc = module.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPointAttr());

    VPUX_THROW_UNLESS(netFunc != nullptr, "Can't find entryPoint '@{0}' for '{1}' Operation", netInfo.getEntryPoint(),
                      netInfo->getName());
}
