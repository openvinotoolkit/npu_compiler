//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes/bounded_buffer_pass_utils.hpp"

#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

#include <iterator>
#include <optional>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNGROUPHOSTBUFFERSASFUNCARGS
#define GEN_PASS_DEF_UNGROUPHOSTBUFFERSASFUNCARGS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

// Simplified example handled by this pass.
//
// Before:
//   func @main_func0(%in: memref<?x...>) -> memref<?x...> {
//     %data = Core.ReinterpretCast(%in) : memref<?x...> -> memref<Nx...>
//     %shape = memref.alloc() : memref<3xsi32>
//     %bb = VPUIP.GroupBoundedBuffer(%data, %shape)
//     ...
//     %outData, %outShape = VPUIP.UngroupBoundedBuffer(%outBB)
//     %out = Core.ReinterpretCast(%outData) : memref<Nx...> -> memref<?x...>
//     return %out
//   }
//
// After:
//   func @main_func0(%in: memref<?x...>, %inShape: memref<3xsi32>)
//       -> (memref<?x...>, memref<3xsi32>) {
//     %data = Core.ReinterpretCast(%in)
//     %bb = VPUIP.GroupBoundedBuffer(%data, %inShape)
//     ...
//     %outData, %outShape = VPUIP.UngroupBoundedBuffer(%outBB)
//     %out = Core.ReinterpretCast(%outData)
//     return %out, %outShape
//   }
//
// Call sites are updated accordingly:
//   Core.NestedCall @main_func0(%arg) -> %res
// becomes
//   %tmpShape = memref.alloc() : memref<3xsi32>
//   Core.NestedCall @main_func0(%arg, %tmpShape) -> (%res, %resShape)

struct HiddenInputInfo {
    size_t argIndex;
    mlir::MemRefType shapeType;
    VPUIP::GroupBoundedBufferOp groupOp;
};

struct HiddenOutputInfo {
    size_t resultIndex;
    mlir::MemRefType shapeType;
    mlir::Value shapeValue;
};

struct FuncHiddenBoundaryInfo {
    SmallVector<HiddenInputInfo> inputs;
    SmallVector<HiddenOutputInfo> outputs;
};

bool isDynamicMemRefType(mlir::Type type) {
    const auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
    if (memrefType == nullptr) {
        return false;
    }

    return llvm::any_of(memrefType.getShape(), [](int64_t dim) {
        return mlir::ShapedType::isDynamic(dim);
    });
}

std::optional<HiddenInputInfo> matchHiddenInputBoundary(mlir::BlockArgument arg, size_t argIndex) {
    // Hidden input boundary pattern:
    // dynamic arg -> ReinterpretCast -> GroupBoundedBuffer(data, shape)
    for (auto* user : arg.getUsers()) {
        auto dataCast = mlir::dyn_cast<Core::ReinterpretCastOp>(user);
        if (dataCast == nullptr) {
            continue;
        }

        for (auto* castUser : dataCast.getResult().getUsers()) {
            auto groupOp = mlir::dyn_cast<VPUIP::GroupBoundedBufferOp>(castUser);
            if (groupOp == nullptr || groupOp.getData() != dataCast.getResult()) {
                continue;
            }

            auto boundedType = mlir::dyn_cast<VPUIP::BoundedBufferType>(groupOp.getResult().getType());
            if (boundedType == nullptr) {
                continue;
            }

            return HiddenInputInfo{argIndex, mlir::cast<mlir::MemRefType>(boundedType.getDynamicShape()), groupOp};
        }
    }

    return std::nullopt;
}

std::optional<HiddenOutputInfo> matchHiddenOutputBoundary(mlir::func::ReturnOp returnOp, size_t resultIndex) {
    // Hidden output boundary pattern:
    // UngroupBoundedBuffer(data, shape) -> ReinterpretCast(data) -> return
    auto outCast = returnOp.getOperand(resultIndex).getDefiningOp<Core::ReinterpretCastOp>();
    if (outCast == nullptr) {
        return std::nullopt;
    }

    auto ungroup = outCast.getInput().getDefiningOp<VPUIP::UngroupBoundedBufferOp>();
    if (ungroup == nullptr) {
        return std::nullopt;
    }

    auto boundedType = mlir::dyn_cast<VPUIP::BoundedBufferType>(ungroup.getInput().getType());
    if (boundedType == nullptr) {
        return std::nullopt;
    }

    return HiddenOutputInfo{resultIndex, mlir::cast<mlir::MemRefType>(boundedType.getDynamicShape()),
                            ungroup.getDynamicShape()};
}

FuncHiddenBoundaryInfo collectHiddenBoundaries(mlir::func::FuncOp funcOp) {
    FuncHiddenBoundaryInfo info;
    auto& entryBlock = funcOp.front();

    for (const auto argIndex : irange(funcOp.getNumArguments())) {
        auto arg = entryBlock.getArgument(argIndex);
        if (!isDynamicMemRefType(arg.getType())) {
            continue;
        }

        if (auto hiddenInput = matchHiddenInputBoundary(arg, argIndex)) {
            info.inputs.push_back(*hiddenInput);
        }
    }

    auto returnOps = funcOp.getOps<mlir::func::ReturnOp>();
    const auto numReturnOps = std::distance(returnOps.begin(), returnOps.end());
    VPUX_THROW_UNLESS(numReturnOps == 1, "Unsupported function '{0}': expected exactly one ReturnOp, got {1}",
                      funcOp.getSymName(), numReturnOps);

    auto returnOp = *returnOps.begin();
    const auto numResults = funcOp.getFunctionType().getNumResults();
    for (const auto resultIndex : irange(numResults)) {
        auto resultType = funcOp.getFunctionType().getResult(resultIndex);
        if (!isDynamicMemRefType(resultType)) {
            continue;
        }

        if (auto hiddenOutput = matchHiddenOutputBoundary(returnOp, resultIndex)) {
            info.outputs.push_back(*hiddenOutput);
        }
    }

    return info;
}

void rewriteNestedCallForHiddenBoundaries(
        Core::NestedCallOp nestedCallOp, const llvm::DenseMap<mlir::func::FuncOp, FuncHiddenBoundaryInfo>& boundaryInfo,
        mlir::OpBuilder& builder, Logger log) {
    // Rebuild nested calls with extra shape operands/results when callee boundaries were expanded
    auto callee =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(nestedCallOp, nestedCallOp.getCalleeAttr());
    if (callee == nullptr) {
        log.trace("Skip nested call at {0}: callee was not resolved", nestedCallOp.getLoc());
        return;
    }

    auto it = boundaryInfo.find(callee);
    if (it == boundaryInfo.end()) {
        log.trace("Skip nested call at {0}: callee '@{1}' has no hidden boundaries", nestedCallOp.getLoc(),
                  callee.getSymName());
        return;
    }

    const auto& info = it->second;
    log.trace("Rewrite nested call at {0} for callee '@{1}': add {2} shape operands, {3} shape results",
              nestedCallOp.getLoc(), callee.getSymName(), info.inputs.size(), info.outputs.size());
    builder.setInsertionPoint(nestedCallOp);

    SmallVector<mlir::Value> newOperands(nestedCallOp.getOperands().begin(), nestedCallOp.getOperands().end());
    for (const auto& in : info.inputs) {
        auto alloc = builder.create<mlir::memref::AllocOp>(nestedCallOp.getLoc(), in.shapeType);
        newOperands.push_back(alloc);
    }

    SmallVector<mlir::Type> newResultTypes(nestedCallOp.getResultTypes().begin(), nestedCallOp.getResultTypes().end());
    for (const auto& out : info.outputs) {
        newResultTypes.push_back(out.shapeType);
    }

    auto newNestedCall = builder.create<Core::NestedCallOp>(nestedCallOp.getLoc(), nestedCallOp.getCalleeAttr(),
                                                            newResultTypes, newOperands);
    newNestedCall->setAttrs(nestedCallOp->getAttrs());

    SmallVector<mlir::Value> replacementResults;
    replacementResults.reserve(nestedCallOp.getNumResults());
    for (const auto index : irange(nestedCallOp.getNumResults())) {
        replacementResults.push_back(newNestedCall.getResult(index));
    }

    nestedCallOp.replaceAllUsesWith(replacementResults);
    nestedCallOp.erase();
}

SmallVector<Core::NestedCallOp> collectNestedCalls(mlir::Operation* scope) {
    SmallVector<Core::NestedCallOp> nestedCalls;
    scope->walk([&](Core::NestedCallOp nestedCallOp) {
        nestedCalls.push_back(nestedCallOp);
    });
    return nestedCalls;
}

class UngroupHostBuffersAsFuncArgs final :
        public VPUIP::impl::UngroupHostBuffersAsFuncArgsBase<UngroupHostBuffersAsFuncArgs> {
public:
    explicit UngroupHostBuffersAsFuncArgs(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void UngroupHostBuffersAsFuncArgs::safeRunOnModule() {
    auto module = getOperation();
    auto ctx = module.getContext();

    if (!config::isHostCompileMode(module)) {
        return;
    }

    // The UngroupHostBuffersAsFuncArgs pass recognizes hidden buffer boundaries (input GroupBoundedBuffer / output
    // UngroupBoundedBuffer) only when a single Core.ReinterpretCast step separated the dynamic function argument (or
    // return value) from the bounded-buffer op. Canonicalizer is needed to ensure Casts fusing is done before
    // UngroupHostBuffersAsFuncArgs processes them
    mlir::RewritePatternSet castsPatterns(ctx);
    vpux::Core::ReinterpretCastOp::getCanonicalizationPatterns(castsPatterns, ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(castsPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    llvm::DenseMap<mlir::func::FuncOp, net::NetworkInfoOp> entryNetInfoByFunc;
    module.walk([&](net::NetworkInfoOp netInfo) {
        if (auto entryFunc = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                    netInfo, netInfo.getEntryPointAttr())) {
            entryNetInfoByFunc[entryFunc] = netInfo;
        }
    });

    // Collect once to avoid invalidating traversal while mutating signatures/calls
    SmallVector<mlir::func::FuncOp> funcOps;
    module.walk([&](mlir::func::FuncOp funcOp) {
        funcOps.push_back(funcOp);
    });

    llvm::DenseMap<mlir::func::FuncOp, FuncHiddenBoundaryInfo> boundaryInfo;

    // Phase 1: expand function boundaries and record per-function shape metadata
    for (auto funcOp : funcOps) {
        if (funcOp.isExternal()) {
            _log.trace("Skip external function '@{0}' at {1}", funcOp.getSymName(), funcOp.getLoc());
            continue;
        }

        auto info = collectHiddenBoundaries(funcOp);
        if (info.inputs.empty() && info.outputs.empty()) {
            _log.trace("No hidden bounded-buffer boundaries found in '@{0}'", funcOp.getSymName());
            continue;
        }

        _log.trace("Expand hidden bounded-buffer boundaries in '@{0}': {1} inputs, {2} outputs", funcOp.getSymName(),
                   info.inputs.size(), info.outputs.size());

        // New shape arguments are appended after existing args; call-site rewrites mirror this order
        auto& entryBlock = funcOp.front();

        SmallVector<mlir::Value> addedShapeArgs;
        addedShapeArgs.reserve(info.inputs.size());

        auto entryNetInfoIt = entryNetInfoByFunc.find(funcOp);
        if (entryNetInfoIt == entryNetInfoByFunc.end()) {
            _log.trace("Skip function '@{0}': no associated NetworkInfoOp", funcOp.getSymName());
            continue;
        }
        auto entryNetInfo = entryNetInfoIt->second;

        for (const auto& in : info.inputs) {
            auto shapeArg = entryBlock.insertArgument(entryBlock.getNumArguments(), in.shapeType,
                                                      appendLoc(funcOp.getLoc(), "_shape"));
            addedShapeArgs.push_back(shapeArg);
            _log.nest(2).trace("Added explicit shape argument for input {0} with type {1}", in.argIndex, in.shapeType);

            if (in.argIndex < entryNetInfo.getInputsDataInfo().size()) {
                VPUIP::addShapeTensorDataInfo(funcOp, in.shapeType, entryNetInfo.getInputsInfo().front(),
                                              entryNetInfo.getInputsDataInfo()[in.argIndex].getName(),
                                              entryNetInfo.getInputsDataInfo().size());
            }
        }

        for (const auto& [inputInfo, shapeArg] : llvm::zip(info.inputs, addedShapeArgs)) {
            auto oldShape = inputInfo.groupOp.getDynamicShape();
            inputInfo.groupOp.getDynamicShapeMutable().assign(shapeArg);
            _log.nest(2).trace("Rewired hidden input boundary for arg {0} to explicit shape argument",
                               inputInfo.argIndex);
            if (auto oldAlloc = oldShape.getDefiningOp<mlir::memref::AllocOp>()) {
                if (oldAlloc->use_empty()) {
                    oldAlloc.erase();
                }
            }
        }

        auto returnOp = *funcOp.getOps<mlir::func::ReturnOp>().begin();

        // Shape results are appended after original return values and reflected in the updated func type
        for (const auto& out : info.outputs) {
            returnOp->insertOperands(returnOp->getNumOperands(), out.shapeValue);
            _log.nest(2).trace("Added explicit shape result for output {0} with type {1}", out.resultIndex,
                               out.shapeType);

            if (out.resultIndex < entryNetInfo.getOutputsDataInfo().size()) {
                VPUIP::addShapeTensorDataInfo(funcOp, out.shapeType, entryNetInfo.getOutputsInfo().front(),
                                              entryNetInfo.getOutputsDataInfo()[out.resultIndex].getName(),
                                              entryNetInfo.getOutputsDataInfo().size());
            }
        }

        const auto newType = mlir::FunctionType::get(funcOp.getContext(), entryBlock.getArgumentTypes(),
                                                     returnOp->getOperandTypes());
        funcOp.setType(newType);

        boundaryInfo[funcOp] = std::move(info);
    }

    if (boundaryInfo.empty()) {
        _log.trace("No expanded hidden boundaries found; skip caller rewrites");
        return;
    }

    // Phase 2: rewrite callers to match updated callee signatures
    mlir::OpBuilder builder(module.getContext());
    auto nestedCalls = collectNestedCalls(module.getOperation());
    for (auto nestedCallOp : nestedCalls) {
        rewriteNestedCallForHiddenBoundaries(nestedCallOp, boundaryInfo, builder, _log.nest(2));
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupHostBuffersAsFuncArgsPass(Logger log) {
    return std::make_unique<UngroupHostBuffersAsFuncArgs>(log);
}
