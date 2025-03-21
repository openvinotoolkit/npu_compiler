//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADDCOPYBETWEENSWKERNELSANDNETWORKIO
#define GEN_PASS_DEF_ADDCOPYBETWEENSWKERNELSANDNETWORKIO
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

using FunctionCalls = mlir::DenseMap<mlir::func::FuncOp, mlir::SmallVector<mlir::func::CallOp>>;
using FunctionWithSwKernelCalls =
        mlir::DenseMap<mlir::func::FuncOp, std::tuple<mlir::DenseSet<int64_t> /*shave-read-args*/,
                                                      mlir::DenseSet<int64_t> /*shave-written-args*/,
                                                      mlir::SmallVector<mlir::func::CallOp>>>;
using SwKernelWithOperandIndex =
        mlir::DenseMap<VPUIP::SwKernelOp,
                       std::tuple<mlir::DenseSet<int64_t> /*shave-read-operand-index*/,
                                  mlir::DenseSet<int64_t> /*shave-write-operand-index*/, bool /*is-tile-pattern*/>>;

mlir::BlockArgument getRootBlockArgument(mlir::Value val, const AliasesInfo& aliasesInfo) {
    // There could be pure view ops between swKernelOp's operands and BlockArgument values
    auto rootBuffers = aliasesInfo.getRoots(val);
    if (rootBuffers.size() != 1) {
        return nullptr;
    }
    return mlir::dyn_cast<mlir::BlockArgument>(*rootBuffers.begin());
}

int64_t getOperandIndex(mlir::Operation* op, mlir::Value operand) {
    auto operands = op->getOperands();
    auto it = llvm::find(operands, operand);
    VPUX_THROW_WHEN(it == operands.end(), "Could not find operand '{0}' in operation '{1}'", operand, op->getLoc());
    return std::distance(operands.begin(), it);
}

mlir::Value getCallOpOutputByOutputBufferIndex(mlir::func::CallOp callOp, mlir::func::FuncOp privateFuncOp,
                                               int64_t outputBufIdx) {
    // if the call op has multiple results, get the index of the one that is block arg
    const auto calledFuncOpNumInputs = privateFuncOp.getNumArguments() - privateFuncOp.getNumResults();
    const auto blockArgOutputIndex = outputBufIdx - calledFuncOpNumInputs;
    return callOp.getResults()[blockArgOutputIndex];
}

bool checkOperandsAreFromSlicedBlockArgument(mlir::ValueRange operands, const AliasesInfo& aliasesInfo) {
    return llvm::all_of(operands, [&](auto operand) {
        if (auto rootBlockArg = getRootBlockArgument(operand, aliasesInfo)) {
            if (auto subviewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(operand.getDefiningOp())) {
                auto sourceType = mlir::cast<vpux::NDTypeInterface>(subviewOp.getSource().getType());
                auto blockArgType = mlir::cast<vpux::NDTypeInterface>(rootBlockArg.getType());
                return sourceType.getTotalAllocSize() == blockArgType.getTotalAllocSize();
            }
        }
        return false;
    });
}

bool isSwKernelWithTilePatternBlockArgument(VPUIP::SwKernelOp swKernelOp, const AliasesInfo& aliasesInfo) {
    /*
        Check the tile pattern below, which is supposed to be happened after SwKernel tiling pass

             Input           OutputBuf
            /     \           /     \
        Subview  Subview  Subview  Subview
           \        \       /       /
                     SwKernel
                        |
                     Concat

    */
    auto allInputsAreSubview = checkOperandsAreFromSlicedBlockArgument(swKernelOp.getInputs(), aliasesInfo);
    if (!allInputsAreSubview) {
        return false;
    }
    auto allOutputBufAreSubview = checkOperandsAreFromSlicedBlockArgument(swKernelOp.getOutputBuffs(), aliasesInfo);
    if (!allOutputBufAreSubview) {
        return false;
    }
    auto usersAreConcat = llvm::all_of(swKernelOp->getUsers(), [](auto user) {
        return mlir::isa_and_nonnull<VPUIP::ConcatViewOp>(user);
    });

    return usersAreConcat;
}

bool doesCallOpHaveTilePatternBlockArgument(mlir::func::CallOp callOp, mlir::func::FuncOp privateFuncOp,
                                            const mlir::DenseSet<int64_t>& inputIdx,
                                            const mlir::DenseSet<int64_t>& outputIdx, const AliasesInfo& aliasesInfo) {
    /*
        Check the tile pattern below:

             Input           OutputBuf
            /     \           /     \
        Subview  Subview  Subview  Subview
           \        \       /       /
                   CallOp[SwKernel]
                        |
                     Concat
    */

    auto checkOperand = [&](const mlir::DenseSet<int64_t>& operandIdx) {
        SmallVector<mlir::Value> operands;
        for (auto idx : operandIdx) {
            operands.push_back(callOp->getOperand(idx));
        }
        return checkOperandsAreFromSlicedBlockArgument(operands, aliasesInfo);
    };

    if (!checkOperand(inputIdx) || !checkOperand(outputIdx)) {
        return false;
    }

    SmallVector<mlir::Value> callOpResults;
    for (auto idx : outputIdx) {
        callOpResults.push_back(getCallOpOutputByOutputBufferIndex(callOp, privateFuncOp, idx));
    }

    auto usersAreConcat = llvm::all_of(callOpResults, [](auto callOpResult) {
        return llvm::all_of(callOpResult.getUsers(), [](auto user) {
            return mlir::isa_and_nonnull<VPUIP::ConcatViewOp>(user);
        });
    });
    return usersAreConcat;
}

bool doAllCallOpsHaveTilePatternBlockArgument(const FunctionWithSwKernelCalls& functionWithSwKernelCalls,
                                              mlir::func::FuncOp privateFunc, const AliasesInfo& aliasesInfo) {
    const auto& callOpData = functionWithSwKernelCalls.at(privateFunc);
    const auto& inputIdxs = std::get<0>(callOpData);
    const auto& outputIdxs = std::get<1>(callOpData);
    const auto& callOps = std::get<2>(callOpData);
    return llvm::all_of(callOps, [&](auto callOp) {
        return doesCallOpHaveTilePatternBlockArgument(callOp, privateFunc, inputIdxs, outputIdxs, aliasesInfo);
    });
}

void updateUseWithNewValue(mlir::Value::use_range& uses, mlir::Value newValue) {
    for (auto& use : uses) {
        auto useIndex = use.getOperandNumber();
        auto useOwnerOp = use.getOwner();
        useOwnerOp->setOperand(useIndex, newValue);
    }
}

void processSwKernelWithInputBlockArg(mlir::func::FuncOp funcOp, VPUIP::SwKernelOp swKernelOp, int64_t inputOperandIdx,
                                      mlir::OpBuilder builder, mlir::DenseSet<mlir::BlockArgument>& handledBlockArgs,
                                      const AliasesInfo& aliasesInfo, bool isTilePattern) {
    auto input = swKernelOp->getOperand(inputOperandIdx);
    if (isTilePattern) {
        auto rootBlockArg = getRootBlockArgument(input, aliasesInfo);
        if (handledBlockArgs.count(rootBlockArg)) {
            return;
        }

        // create new alloc op to copy the content of the block arg
        builder.setInsertionPointToStart(&funcOp.getBody().front());
        auto newBufferType = rootBlockArg.getType();
        auto newBufferOp = builder.create<mlir::memref::AllocOp>(rootBlockArg.getLoc(),
                                                                 mlir::cast<mlir::MemRefType>(newBufferType));

        // add input copy
        const auto inputCopyLoc = appendLoc(newBufferOp->getLoc(), "_ddr_copy");
        auto inputCopyOp = builder.create<VPUIP::CopyOp>(inputCopyLoc, rootBlockArg, newBufferOp);

        // replace the uses of the block arg with the input copy
        rootBlockArg.replaceAllUsesExcept(inputCopyOp, inputCopyOp);
        handledBlockArgs.insert(rootBlockArg);
        return;
    }

    // create new alloc op to copy the content of the block arg
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto newBufferOp =
            builder.create<mlir::memref::AllocOp>(input.getLoc(), mlir::cast<mlir::MemRefType>(input.getType()));

    // add input copy
    const auto inputCopyLoc = appendLoc(newBufferOp->getLoc(), "_ddr_copy");
    builder.setInsertionPoint(swKernelOp);
    auto inputCopyOp = builder.create<VPUIP::CopyOp>(inputCopyLoc, input, newBufferOp);

    // set input copy as SwKernel operand instead of the block arg
    input.replaceUsesWithIf(inputCopyOp, [&](mlir::OpOperand& operand) {
        return operand.getOwner() == swKernelOp;
    });
}

void processSwKernelWithOutputBlockArg(mlir::func::FuncOp funcOp, VPUIP::SwKernelOp swKernelOp,
                                       int64_t outputBufOperandIdx, mlir::OpBuilder builder,
                                       mlir::DenseSet<mlir::BlockArgument>& handledBlockArgs,
                                       const AliasesInfo& aliasesInfo, bool isTilePattern) {
    auto outputBuf = swKernelOp->getOperand(outputBufOperandIdx);

    auto getSwKernelOutputByOperandIndex = [&](int64_t operandIndex) {
        // swKernelOp has the following order of operands:
        // - variadic number of inputs
        // - variadic number of dynamicInputShapes (optional)
        // - variadic number of results
        // - variadic number of dynamicOutputShapes (optional)

        auto outIndex = operandIndex - swKernelOp.getInputs().size() - swKernelOp.getDynamicInputShapes().size();
        // We need to understand which specific type of output the operand with the given OperandIdx belongs to: results
        // or dynamicOutputShapes

        if (outIndex < swKernelOp.getResults().size()) {
            // results
            return swKernelOp.getResults()[outIndex];
        }

        outIndex -= swKernelOp.getResults().size();
        if (outIndex < swKernelOp.getDynamicOutputShapes().size()) {
            // DynamicOutputShapes
            return swKernelOp.getDynamicOutputShapes()[outIndex];
        }

        // We believe it cannot be profiling output because the passes that add profiling are invoked later in the
        // compilation pipeline
        VPUX_THROW("Can't correlate OperandIdx:{0} with the SWKernel results: {1} or dynamicOutputShapes: {2}. "
                   "Preceding inputs: {3}, dynamicInputShapes: {4}",
                   outputBufOperandIdx, swKernelOp.getResults().size(), swKernelOp.getDynamicOutputShapes().size(),
                   swKernelOp.getInputs().size(), swKernelOp.getDynamicInputShapes().size());
    };
    auto swKernelOutput = getSwKernelOutputByOperandIndex(outputBufOperandIdx);

    if (isTilePattern) {
        auto rootBlockArg = getRootBlockArgument(outputBuf, aliasesInfo);
        if (handledBlockArgs.count(rootBlockArg)) {
            return;
        }

        // get the subview op that defines the output buffer
        auto subview = outputBuf.getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subview == nullptr, "Unexpected defining op for {0}", outputBuf.getLoc());
        auto outBufSource = subview.getSource();

        // get the concat user
        VPUX_THROW_UNLESS(swKernelOutput.hasOneUse(), "Expected a single use of the output buffer");
        auto concat = mlir::dyn_cast<VPUIP::ConcatViewOp>(*swKernelOutput.user_begin());
        auto concatUses = concat.getResult().getUses();

        // set the new buffer to be the operand of the subview and concat op
        builder.setInsertionPointToStart(&funcOp.getBody().front());
        auto newBufferType = outBufSource.getType();
        auto newBufferOp = builder.create<mlir::memref::AllocOp>(outBufSource.getLoc(),
                                                                 mlir::cast<mlir::MemRefType>(newBufferType));
        outBufSource.replaceAllUsesWith(newBufferOp);

        // add output copy
        builder.setInsertionPointAfter(concat);
        const auto outputCopyLoc = appendLoc(newBufferOp->getLoc(), "_ddr_copy");
        auto outputCopyOp = builder.create<VPUIP::CopyOp>(outputCopyLoc, concat, outBufSource);

        // modify all uses of the concat op to use the output copy instead
        updateUseWithNewValue(concatUses, outputCopyOp);
        handledBlockArgs.insert(rootBlockArg);
        return;
    }

    auto outputType = outputBuf.getType();
    auto swKernelOutputUses = swKernelOutput.getUses();

    // set the new buffer to be the operand of the SwKernel instead of the block arg
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto newBufferOp =
            builder.create<mlir::memref::AllocOp>(outputBuf.getLoc(), mlir::cast<mlir::MemRefType>(outputType));
    swKernelOp->setOperand(outputBufOperandIdx, newBufferOp);

    // add output copy
    const auto outputCopyLoc = appendLoc(newBufferOp->getLoc(), "_ddr_copy");
    builder.setInsertionPointAfter(swKernelOp);
    auto outputCopyOp = builder.create<VPUIP::CopyOp>(outputCopyLoc, swKernelOutput, outputBuf);

    // modify all uses of the SwKernel output to use the output copy instead
    for (auto& use : swKernelOutputUses) {
        auto useIndex = use.getOperandNumber();
        if (auto useOwnerOp = use.getOwner()) {
            useOwnerOp->setOperand(useIndex, outputCopyOp);
        }
    }
}

void addCopySwKernelWithBlockArgIOInMainFuncOp(mlir::func::FuncOp mainFuncOp, mlir::OpBuilder builder) {
    AliasesInfo aliasesInfo{mainFuncOp};

    // Find all the sw kernel ops which need insert copy
    SwKernelWithOperandIndex swKernelOps;
    mainFuncOp.walk([&](VPUIP::SwKernelOp swKernelOp) {
        auto selectBlockArgsAndGetTheirIdices = [&](mlir::OperandRange range) {
            mlir::DenseSet<int64_t> resultIdx;
            for (auto operand : range) {
                if (auto rootBlockArg = getRootBlockArgument(operand, aliasesInfo)) {
                    resultIdx.insert(getOperandIndex(swKernelOp, operand));
                }
            }
            return resultIdx;
        };

        auto inputIdx = selectBlockArgsAndGetTheirIdices(swKernelOp.getInputs());
        if (!swKernelOp.getDynamicInputShapes().empty()) {
            auto dynamicInputShapesIdx = selectBlockArgsAndGetTheirIdices(swKernelOp.getDynamicInputShapes());
            inputIdx.insert(dynamicInputShapesIdx.begin(), dynamicInputShapesIdx.end());
        }
        auto outputIdx = selectBlockArgsAndGetTheirIdices(swKernelOp.getOutputs());

        if (!inputIdx.empty() || !outputIdx.empty()) {
            auto isBlockArgTiled = isSwKernelWithTilePatternBlockArgument(swKernelOp, aliasesInfo);
            swKernelOps[swKernelOp] = std::make_tuple(inputIdx, outputIdx, isBlockArgTiled);
        }
    });

    mlir::DenseSet<mlir::BlockArgument> handledBlockArgs;
    for (auto& item : swKernelOps) {
        auto swKernelOp = item.first;
        auto inputIdx = std::get<0>(item.second);
        auto outputIdx = std::get<1>(item.second);
        auto isTilePattern = std::get<2>(item.second);
        for (auto inputOperandIdx : inputIdx) {
            processSwKernelWithInputBlockArg(mainFuncOp, swKernelOp, inputOperandIdx, builder, handledBlockArgs,
                                             aliasesInfo, isTilePattern);
        }
        for (auto outputBufOperandIdx : outputIdx) {
            processSwKernelWithOutputBlockArg(mainFuncOp, swKernelOp, outputBufOperandIdx, builder, handledBlockArgs,
                                              aliasesInfo, isTilePattern);
        }
    }
}

void processCallOpWithInputBlockArg(mlir::Value calledFuncArg,
                                    mlir::DenseMap<mlir::Value, VPUIP::CopyOp>& blockArgumentToNewCopy,
                                    mlir::func::FuncOp mainFuncOp, mlir::func::CallOp callOp, mlir::OpBuilder builder,
                                    const AliasesInfo& aliasesInfo, bool isTilePattern) {
    auto rootBlockArg = getRootBlockArgument(calledFuncArg, aliasesInfo);
    builder.setInsertionPointToStart(&mainFuncOp.getBody().front());

    if (isTilePattern) {
        if (blockArgumentToNewCopy.count(rootBlockArg)) {
            // block argument has already create a new buffer with copy op, so skip it.
            return;
        }
        // create new input buffer
        auto newBufferType = rootBlockArg.getType();
        auto newBufferOp = builder.create<mlir::memref::AllocOp>(rootBlockArg.getLoc(),
                                                                 mlir::cast<mlir::MemRefType>(newBufferType));
        auto newBufferResult = newBufferOp->getResult(0);
        // add input copy
        const auto inputCopyLoc = appendLoc(newBufferOp->getLoc(), "_ddr_copy");
        auto inputCopyOp = builder.create<VPUIP::CopyOp>(inputCopyLoc, rootBlockArg, newBufferResult);
        // set input copy as call op operand instead of the block arg
        rootBlockArg.replaceAllUsesExcept(inputCopyOp, inputCopyOp);
        blockArgumentToNewCopy.insert(std::make_pair(rootBlockArg, inputCopyOp));
        return;
    }

    // find the index of the main func block arg in the arg operands list of the call op
    auto operandIt = llvm::find(callOp.getArgOperands(), calledFuncArg);
    auto callOpArgIndex = std::distance(callOp.getArgOperands().begin(), operandIt);

    if (!blockArgumentToNewCopy.count(calledFuncArg)) {
        // create new input buffer
        auto blockArgLoc = calledFuncArg.getLoc();
        auto blockArgType = calledFuncArg.getType();
        auto newBufferOp = builder.create<mlir::memref::AllocOp>(blockArgLoc, blockArgType.cast<mlir::MemRefType>());
        auto newBufferResult = newBufferOp->getResult(0);

        // add input copy
        builder.setInsertionPoint(callOp);
        const auto inputCopyLoc = appendLoc(blockArgLoc, "_ddr_copy");
        auto inputCopyOp = builder.create<VPUIP::CopyOp>(inputCopyLoc, calledFuncArg, newBufferResult);

        // set input copy as call op operand instead of the block arg
        calledFuncArg.replaceUsesWithIf(inputCopyOp, [&](mlir::OpOperand& operand) {
            return operand.getOwner() == callOp;
        });
        blockArgumentToNewCopy.insert(std::make_pair(calledFuncArg, inputCopyOp));
    } else {
        // avoid to create a duplicated copy for the same blockArg
        callOp->setOperand(callOpArgIndex, blockArgumentToNewCopy[calledFuncArg]);
    }
}

void processCallOpWithOutputBlockArg(mlir::Value calledFuncArg,
                                     mlir::DenseMap<mlir::Value, VPUIP::CopyOp>& blockArgumentToNewCopy,
                                     mlir::func::CallOp callOp, mlir::func::FuncOp privateFuncOp,
                                     mlir::func::FuncOp mainFuncOp, mlir::OpBuilder builder,
                                     const AliasesInfo& aliasesInfo, bool isTilePattern) {
    builder.setInsertionPointToStart(&mainFuncOp.getBody().front());
    if (isTilePattern) {
        auto rootBlockArg = getRootBlockArgument(calledFuncArg, aliasesInfo);
        if (blockArgumentToNewCopy.count(rootBlockArg)) {
            return;
        }

        // get the subview op that defines the output buffer
        auto subview = calledFuncArg.getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subview == nullptr, "Unexpected defining op for {0}", calledFuncArg.getLoc());
        auto outBufSource = subview.getSource();

        auto callOpResult =
                getCallOpOutputByOutputBufferIndex(callOp, privateFuncOp, getOperandIndex(callOp, calledFuncArg));
        VPUX_THROW_UNLESS(callOpResult.hasOneUse(), "Expected a single use of the output buffer");
        auto concat = mlir::dyn_cast<VPUIP::ConcatViewOp>(*callOpResult.user_begin());
        auto concatUses = concat.getResult().getUses();

        // set the new buffer to be the operand of the subview and concat op
        builder.setInsertionPointToStart(&mainFuncOp.getBody().front());
        auto newBufferType = outBufSource.getType();
        auto newBufferOp = builder.create<mlir::memref::AllocOp>(outBufSource.getLoc(),
                                                                 mlir::cast<mlir::MemRefType>(newBufferType));
        outBufSource.replaceAllUsesWith(newBufferOp);

        // add output copy
        builder.setInsertionPointAfter(concat);
        const auto outputCopyLoc = appendLoc(newBufferOp->getLoc(), "_ddr_copy");
        auto outputCopyOp = builder.create<VPUIP::CopyOp>(outputCopyLoc, concat, outBufSource);

        // modify all uses of the concat op to use the output copy instead
        updateUseWithNewValue(concatUses, outputCopyOp);
        blockArgumentToNewCopy.insert(std::make_pair(rootBlockArg, outputCopyOp));
        return;
    }

    // create new buffer
    auto blockArgLoc = calledFuncArg.getLoc();
    auto blockArgType = calledFuncArg.getType();
    auto newBufferOp = builder.create<mlir::memref::AllocOp>(blockArgLoc, blockArgType.cast<mlir::MemRefType>());
    auto newBufferResult = newBufferOp->getResult(0);

    auto callOpArgIndex = getOperandIndex(callOp, calledFuncArg);
    // set the new buffer to be the operand of the call op instead of the block arg
    callOp->setOperand(callOpArgIndex, newBufferResult);

    auto callOpResult = getCallOpOutputByOutputBufferIndex(callOp, privateFuncOp, callOpArgIndex);
    auto callOpResultUses = callOpResult.getUses();

    // add output copy
    builder.setInsertionPointAfter(callOp);
    const auto outputCopyLoc = appendLoc(blockArgLoc, "_ddr_copy");
    auto outputCopyOp = builder.create<VPUIP::CopyOp>(outputCopyLoc, callOpResult, calledFuncArg);

    // modify all uses of the call op output to use the output copy instead
    updateUseWithNewValue(callOpResultUses, outputCopyOp);
}

void addCopyForCallOpWithBlockArgIO(FunctionWithSwKernelCalls& functionWithSwKernelCalls, mlir::func::FuncOp mainFuncOp,
                                    mlir::OpBuilder builder) {
    AliasesInfo aliasesInfo{mainFuncOp};
    for (auto& [privateFunc, callOpData] : functionWithSwKernelCalls) {
        auto inputIdxs = std::get<0>(callOpData);
        auto outputIdxs = std::get<1>(callOpData);
        auto callOps = std::get<2>(callOpData);

        mlir::DenseMap<mlir::Value, VPUIP::CopyOp> blockArgumentToNewCopy;
        auto isTilePattern =
                doAllCallOpsHaveTilePatternBlockArgument(functionWithSwKernelCalls, privateFunc, aliasesInfo);

        for (auto& callOp : callOps) {
            auto calledFuncArgs = callOp.getOperands();
            for (auto calledFuncArg : calledFuncArgs) {
                auto rootBlockArg = getRootBlockArgument(calledFuncArg, aliasesInfo);
                if (rootBlockArg == nullptr) {
                    continue;
                }

                // check the main func block arg is used as input or output
                auto callOpArgIndex = getOperandIndex(callOp, calledFuncArg);
                auto isSwKernelInputArg = inputIdxs.contains(callOpArgIndex);
                auto isSwKernelOutputArg = outputIdxs.contains(callOpArgIndex);
                if (!isSwKernelInputArg && !isSwKernelOutputArg) {
                    continue;
                }

                if (isSwKernelInputArg) {
                    processCallOpWithInputBlockArg(calledFuncArg, blockArgumentToNewCopy, mainFuncOp, callOp, builder,
                                                   aliasesInfo, isTilePattern);
                } else {
                    processCallOpWithOutputBlockArg(calledFuncArg, blockArgumentToNewCopy, callOp, privateFunc,
                                                    mainFuncOp, builder, aliasesInfo, isTilePattern);
                }
            }
        }
    }
}

//
// AddCopyBetweenSWKernelsAndNetworkIOPass
//

class AddCopyBetweenSWKernelsAndNetworkIOPass final :
        public VPUIP::impl::AddCopyBetweenSWKernelsAndNetworkIOBase<AddCopyBetweenSWKernelsAndNetworkIOPass> {
public:
    explicit AddCopyBetweenSWKernelsAndNetworkIOPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    FunctionCalls _functionCalls;
    FunctionWithSwKernelCalls _functionWithSwKernelCalls;

    void safeRunOnModule() final;
};

void AddCopyBetweenSWKernelsAndNetworkIOPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    mlir::func::FuncOp mainFuncOp;
    vpux::IE::CNNNetworkOp cnnOp;
    vpux::IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, mainFuncOp);

    mainFuncOp.walk([&](mlir::func::CallOp callOp) {
        auto calledFuncOp = vpux::getCalledFunction(callOp);
        if (calledFuncOp && calledFuncOp.isPrivate()) {
            _functionCalls[calledFuncOp].push_back(callOp);
        }
    });

    for (const auto& pair : _functionCalls) {
        auto func = pair.first;
        AliasesInfo aliasesInfo{func};

        mlir::DenseSet<int64_t> inputIndices;
        mlir::DenseSet<int64_t> outputIndices;

        func.walk([&](VPUIP::SwKernelOp swKernelOp) {
            auto swKernelInputs = swKernelOp.getInputs();
            for (auto input : swKernelInputs) {
                if (auto rootBlockArg = getRootBlockArgument(input, aliasesInfo)) {
                    inputIndices.insert(rootBlockArg.getArgNumber());
                }
            }
            auto swKernelOutput = swKernelOp.getOutputBuffs();
            for (auto output : swKernelOutput) {
                if (auto rootBlockArg = getRootBlockArgument(output, aliasesInfo)) {
                    outputIndices.insert(rootBlockArg.getArgNumber());
                }
            }
        });

        if (inputIndices.empty() && outputIndices.empty()) {
            continue;
        }
        _functionWithSwKernelCalls[func] = std::make_tuple(inputIndices, outputIndices, pair.second);
    }

    mlir::OpBuilder builder(mainFuncOp);
    addCopySwKernelWithBlockArgIOInMainFuncOp(mainFuncOp, builder);
    addCopyForCallOpWithBlockArgIO(_functionWithSwKernelCalls, mainFuncOp, builder);

    // remove the unused allocOp
    mainFuncOp.walk([&](mlir::memref::AllocOp allocOp) {
        if (allocOp->getUsers().empty()) {
            allocOp.erase();
        }
    });
}
}  // namespace

//
// createAddCopyBetweenSWKernelsAndNetworkIOPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(Logger log) {
    return std::make_unique<AddCopyBetweenSWKernelsAndNetworkIOPass>(log);
}
