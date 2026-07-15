//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/utils/func_dialect.hpp>
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dynamic_strides_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/version.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <algorithm>
#include <queue>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_LEGALIZESTRIDEDDMAS
#define GEN_PASS_DEF_LEGALIZESTRIDEDDMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
class LegalizeStridedDmasPass final : public VPUIP::impl::LegalizeStridedDmasBase<LegalizeStridedDmasPass> {
public:
    explicit LegalizeStridedDmasPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    bool areStridesCompatible(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);
    bool isCompatibleViewOp(mlir::ViewLikeOpInterface view);
    llvm::DenseSet<int> visitDmasForArgument(mlir::func::FuncOp func, int operandIdx);
    void legalizeIncompatibleReadersAndWriters(mlir::Value incompatibleVal);
    void insertLegalizationDma(mlir::Value incompatibleVal, bool hasWrites, llvm::SmallVector<mlir::Value>& writers);
    llvm::DenseSet<int> checkFunctionForWritesAndAliases(mlir::func::FuncOp func, int operandIdx, bool& hasWrites);
};

bool LegalizeStridedDmasPass::isCompatibleViewOp(mlir::ViewLikeOpInterface view) {
    // In case of strided sub views it is not possible for areStridesCompatible to
    // determine if strides are compatible, just skip it here.
    if (auto subView = mlir::dyn_cast<VPUIP::SubViewOp>(view.getOperation())) {
        if (auto staticStrides = subView.getStaticStrides()) {
            return false;
        }
    }
    auto inType = mlir::cast<vpux::NDTypeInterface>(view.getViewSource().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(view->getResult(0).getType());
    return areStridesCompatible(inType, outType);
}

bool LegalizeStridedDmasPass::areStridesCompatible(NDTypeInterface inType, NDTypeInterface outType) {
    auto inStrides = inType.getMemStrides();
    auto outStrides = outType.getMemStrides();
    auto inElemSize = inType.getElemTypeSize();
    auto outElemSize = outType.getElemTypeSize();

    return VPUIP::areStridesCompatible(inStrides, inElemSize, outStrides, outElemSize);
}

void LegalizeStridedDmasPass::insertLegalizationDma(mlir::Value incompatibleVal, bool hasWrites,
                                                    llvm::SmallVector<mlir::Value>& exitNodes) {
    auto opBuilder = mlir::OpBuilder(&getContext());
    if (auto parentOp = incompatibleVal.getDefiningOp()) {
        opBuilder.setInsertionPointAfter(parentOp);
    } else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(incompatibleVal)) {
        opBuilder.setInsertionPointToStart(blockArg.getOwner());
    }
    auto newLocation = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), "stridedDMALegalization"));
    auto newMemref = opBuilder.create<mlir::memref::AllocOp>(newLocation,
                                                             mlir::cast<mlir::MemRefType>(incompatibleVal.getType()));
    auto originalViewSource = incompatibleVal;
    auto stridedLoadDma = opBuilder.create<VPUIP::NNDMAOp>(newLocation, originalViewSource, newMemref.getMemref());
    stridedLoadDma->setAttr(vpux::stridedInputAttrName, mlir::UnitAttr::get(&getContext()));
    originalViewSource.replaceUsesWithIf(stridedLoadDma->getResult(0), [&](mlir::OpOperand& operand) {
        return operand.getOwner() != stridedLoadDma &&
               !mlir::isa<mlir::func::ReturnOp, VPUIP::ConcatViewOp>(operand.getOwner());
    });

    if (hasWrites && !exitNodes.empty()) {
        SmallVector<mlir::Value> concatInput;
        mlir::Operation* lastExitOp = nullptr;
        for (auto exitNode : exitNodes) {
            concatInput.push_back(exitNode);
            if (!mlir::isa<mlir::BlockArgument>(exitNode)) {
                auto definingOp = exitNode.getDefiningOp();
                if (lastExitOp == nullptr || !definingOp->isBeforeInBlock(lastExitOp)) {
                    lastExitOp = definingOp;
                }
            }
        }
        concatInput.push_back(stridedLoadDma->getResult(0));
        if (lastExitOp && !lastExitOp->isBeforeInBlock(stridedLoadDma)) {
            opBuilder.setInsertionPointAfter(lastExitOp);
        }
        auto writerConcat = opBuilder.create<VPUIP::ConcatViewOp>(newLocation, concatInput, newMemref.getMemref());
        auto stridedStoreDma =
                opBuilder.create<VPUIP::NNDMAOp>(newLocation, writerConcat->getResult(0), originalViewSource);
        stridedStoreDma->setAttr(vpux::stridedOutputAttrName, mlir::UnitAttr::get(&getContext()));
        originalViewSource.replaceUsesWithIf(stridedStoreDma->getResult(0), [&](mlir::OpOperand& operand) {
            return operand.getOwner() != writerConcat &&
                   mlir::isa<mlir::func::ReturnOp, VPUIP::ConcatViewOp>(operand.getOwner());
        });
    }
}

llvm::DenseSet<int> LegalizeStridedDmasPass::checkFunctionForWritesAndAliases(mlir::func::FuncOp func, int operandIdx,
                                                                              bool& hasWrites) {
    std::queue<mlir::OpOperand*> valuesToVisit;
    llvm::DenseSet<mlir::OpOperand*> visitedValues;
    llvm::DenseSet<int> aliasingOperands;
    auto funcArgument = func.getBlocks().front().getArguments()[operandIdx];
    for (auto& use : funcArgument.getUses()) {
        valuesToVisit.push(&use);
    }
    while (!valuesToVisit.empty()) {
        auto use = valuesToVisit.front();
        valuesToVisit.pop();
        if (visitedValues.contains(use)) {
            continue;
        }
        visitedValues.insert(use);
        mlir::Operation* user = use->getOwner();
        llvm::TypeSwitch<mlir::Operation*, void>(user)
                .Case<VPUIP::DMATypeOpInterface>([&](VPUIP::DMATypeOpInterface dmaOp) {
                    if (dmaOp.getOutputBuff() == use->get()) {
                        for (auto& dmaUse : dmaOp->getUses()) {
                            valuesToVisit.push(&dmaUse);
                        }
                        hasWrites = true;
                    }
                })
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewLikeOp) {
                    for (auto& viewUse : viewLikeOp->getUses()) {
                        valuesToVisit.push(&viewUse);
                    }
                })
                .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
                    auto calledFunc = getCalledFunction(callOp);
                    auto aliasingOperands =
                            checkFunctionForWritesAndAliases(calledFunc, use->getOperandNumber(), hasWrites);
                    for (auto aliasingOperandIndex : aliasingOperands) {
                        for (auto& use : callOp->getResult(aliasingOperandIndex).getUses()) {
                            valuesToVisit.push(&use);
                        }
                    }
                })
                .Case<mlir::func::ReturnOp>([&](mlir::func::ReturnOp) {
                    aliasingOperands.insert(use->getOperandNumber());
                })
                .Default([&](mlir::Operation*) {
                    VPUX_THROW("Unknown operation encountered");
                });
    }

    return aliasingOperands;
}

void LegalizeStridedDmasPass::legalizeIncompatibleReadersAndWriters(mlir::Value incompatibleVal) {
    std::queue<mlir::OpOperand*> valuesToVisit;
    llvm::DenseSet<mlir::OpOperand*> visitedValues;
    llvm::SmallVector<mlir::Value> exitNodes;
    bool hasWrites = false;
    for (auto& use : incompatibleVal.getUses()) {
        valuesToVisit.push(&use);
    }
    while (!valuesToVisit.empty()) {
        auto use = valuesToVisit.front();
        valuesToVisit.pop();
        if (visitedValues.contains(use)) {
            continue;
        }
        visitedValues.insert(use);
        auto op = use->getOwner();
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<VPUIP::DMATypeOpInterface>([&](VPUIP::DMATypeOpInterface dmaOp) {
                    auto isInput = dmaOp.getInput() == use->get();
                    if (isInput) {
                        exitNodes.push_back(use->get());
                        return;
                    } else {
                        hasWrites = true;
                        for (auto& dmaUse : dmaOp->getUses()) {
                            valuesToVisit.push(&dmaUse);
                        }
                        if (dmaOp->getUses().empty()) {
                            exitNodes.push_back(dmaOp->getResult(0));
                        }
                    }
                })
                .Case<VPUIP::ConcatViewOp>([&](VPUIP::ConcatViewOp) {
                    exitNodes.push_back(use->get());
                })
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewLikeOp) {
                    for (auto& viewUse : viewLikeOp->getUses()) {
                        valuesToVisit.push(&viewUse);
                    }
                })
                .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
                    auto calledFunc = getCalledFunction(callOp);
                    bool nestedFuncHasWrites = false;
                    auto aliasingOperands =
                            checkFunctionForWritesAndAliases(calledFunc, use->getOperandNumber(), nestedFuncHasWrites);
                    for (auto aliasingOperandIndex : aliasingOperands) {
                        for (auto& use : callOp->getResult(aliasingOperandIndex).getUses()) {
                            valuesToVisit.push(&use);
                        }
                    }
                    if (nestedFuncHasWrites) {
                        exitNodes.push_back(callOp->getResult(0));
                    }
                    hasWrites |= nestedFuncHasWrites;
                })
                .Case<mlir::func::ReturnOp>([&](mlir::func::ReturnOp) {
                    exitNodes.push_back(use->get());
                })
                .Default([&](mlir::Operation*) {
                    VPUX_THROW("Unknown operation encountered");
                });
    }

    insertLegalizationDma(incompatibleVal, hasWrites, exitNodes);
}

/*
  The function below traverses the graph in a BFS fashion and searches for transformations
  incompatible with dynamic-strides DMAs.
  The IR is assumed to meet the following constraints:
  1. There is a DMA between a function argument and a compute layer.
  2. There can be any number of ViewLikeOps between the DMA operation and the function argument.
  3. There can be a function call operation between the function argument and a DMA. However functions
     are not recursive.

  When an incompatible view or DMA op is detected algorithm will do the following:
  1. Allocate auxillary DDR buffer and copy argument data to it.
  2. Replace all uses of the incompatible value with the new auxiliary DDR buffer
  3. Check if the branch of the network writes to auxiliary buffer, if yes insert DMA
     that will transfer data back to network IO

  For example following IR:

  func(%arg) {
    ...
    %val = SomeChainOfOps(%arg)
    %incompatible_view = SomeIncompatibleView(%val)
    %compatible_view = SomeCompatibleView(%val)
    %write_compatible = DMA(%source, %compatible_view)
    %write_incompatible = DMA(%source2, %incompatible_view)
    ...
  }

  Is transformed into:

  func(%arg) {
    ...
    %val = SomeChainOfOps(%arg)

    // Legalization prologue
    %auxiliary = memref.alloc()
    %legalization = DMA {stridedInput} (%val, %auxiliary)

    %incompatible_view = SomeIncompatibleView(%legalization)
    %compatible_view = SomeCompatibleView(%legalization)
    %write_compatible = DMA(%source, %compatible_view)
    %write_incompatible = DMA(%source2, %incompatible_view)

    // Legalization epilogue
    %concat = ConcatView input(%write_compatible, %write_incompatible) output(%legalization)
    %legalization_write = DMA {stridedOutput} (%concat, %val)

    ...
  }

  Function calls are handled by calling the same algorithm recursively. This means
  that functions can't call each other (which can't happen in the current IR). The algorithm
  will also update a function even if it is called from multiple call sites, which can be
  inefficient in cases where the same function is later called with an input that is not
  function I/O. Those cases should be rare, as we mostly get multiple functions from
  vertical fusion outlining.
*/
llvm::DenseSet<int> LegalizeStridedDmasPass::visitDmasForArgument(mlir::func::FuncOp func, int operandIdx) {
    std::queue<mlir::Value> valuesToVisit;
    llvm::DenseSet<mlir::Value> visitedValues;
    llvm::DenseSet<int> aliasingOperands;
    auto funcArgument = func.getBlocks().front().getArguments()[operandIdx];
    valuesToVisit.push(funcArgument);

    while (!valuesToVisit.empty()) {
        auto value = valuesToVisit.front();
        valuesToVisit.pop();
        if (visitedValues.contains(value)) {
            continue;
        }
        visitedValues.insert(value);
        bool isLegal = true;
        std::queue<mlir::Value> nextLevel;
        for (auto& use : value.getUses()) {
            mlir::Operation* user = use.getOwner();
            llvm::TypeSwitch<mlir::Operation*, void>(user)
                    .Case<VPUIP::DMATypeOpInterface>([&](VPUIP::DMATypeOpInterface dmaOp) {
                        auto isInput = dmaOp.getInput() == use.get();
                        if (mlir::isa<VPUIP::NNDMAOp, VPUIP::ConvertDMAOp, VPUIP::PermuteDMAOp>(dmaOp.getOperation())) {
                            if (!isInput) {
                                nextLevel.push(dmaOp->getResult(0));
                            }
                        } else {
                            isLegal = false;
                        }
                    })
                    .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewLikeOp) {
                        if (isCompatibleViewOp(viewLikeOp)) {
                            nextLevel.push(viewLikeOp->getResult(0));
                        } else {
                            isLegal = false;
                        }
                    })
                    .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
                        auto calledFunc = getCalledFunction(callOp);
                        auto aliasingOperands = visitDmasForArgument(calledFunc, use.getOperandNumber());
                        for (auto aliasingOperandIndex : aliasingOperands) {
                            nextLevel.push(callOp->getResult(aliasingOperandIndex));
                        }
                    })
                    .Case<mlir::func::ReturnOp>([&](mlir::func::ReturnOp) {
                        aliasingOperands.insert(use.getOperandNumber());
                    })
                    .Default([&](mlir::Operation*) {
                        VPUX_THROW("Unknown operation encountered");
                    });
        }

        if (isLegal) {
            for (auto& use : value.getUses()) {
                mlir::Operation* user = use.getOwner();
                if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(user)) {
                    auto isInput = dmaOp.getInput() == use.get();
                    if (isInput) {
                        dmaOp->setAttr(vpux::stridedInputAttrName, mlir::UnitAttr::get(&getContext()));
                    } else {
                        dmaOp->setAttr(vpux::stridedOutputAttrName, mlir::UnitAttr::get(&getContext()));
                    }
                }
            }
            while (!nextLevel.empty()) {
                valuesToVisit.push(nextLevel.front());
                nextLevel.pop();
            }
        } else {
            legalizeIncompatibleReadersAndWriters(value);
        }
    }

    return aliasingOperands;
}

void LegalizeStridedDmasPass::safeRunOnModule() {
    auto module = getOperation();
    net::NetworkInfoOp netInfo = *module.getOps<net::NetworkInfoOp>().begin();
    auto func = module.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPoint());
    auto funcArguments = func.getBlocks().front().getArguments();
    bool hasStridedIO = false;

    for (auto it : funcArguments | indexed) {
        auto argument = it.value();
        auto index = it.index();

        if (!vpux::net::isArgStrided(module, index)) {
            continue;
        }

        auto argumentElementSize = mlir::cast<vpux::NDTypeInterface>(argument.getType()).getElemTypeSize();
        if (argumentElementSize < vpux::Bit(8)) {
            continue;
        }

        hasStridedIO = true;
        visitDmasForArgument(func, index);
    }

    if (hasStridedIO) {
        const auto currentElfVersion = config::getElfAbiVersion(module);
        const auto minimumElfVersion = config::getNPUConstraints(module.getContext()).dynamicStridesMinElfAbiVersion;
        if (currentElfVersion.has_value() && minimumElfVersion.has_value() &&
            minimumElfVersion.value() > currentElfVersion.value()) {
            config::setElfAbiVersion(module, minimumElfVersion.value());
            _log.warning("Updating ELF version to {0} due to presence of dynamic strides", minimumElfVersion.value());
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createLegalizeStridedDMAsPass(Logger log) {
    return std::make_unique<LegalizeStridedDmasPass>(log);
}
