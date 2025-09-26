//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <vpu_layer_cost_model.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MOVEREFLECTPADTOCMX
#define GEN_PASS_DEF_MOVEREFLECTPADTOCMX
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
namespace {

bool fitsIntoCMX(mlir::Value rootInput, VPU::ConcatOp* concat) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(rootInput.getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(concat->getOutput().getType());
    auto requiredCMX = inputType.getTotalAllocSize() + outputType.getTotalAllocSize();
    return requiredCMX < VPU::getTotalCMXSize(*concat);
}

void propagateReturnType(mlir::Operation*& op, mlir::OpBuilder& builder) {
    for (auto* user : op->getUsers()) {
        if (mlir::isa<VPU::ConcatOp>(user)) {
            return;
        }
        if (mlir::isa<VPU::PermuteCastOp>(user)) {
            builder.setInsertionPoint(user);
            auto permuteCastUser = mlir::cast<VPU::PermuteCastOp>(user);
            auto resultType = mlir::cast<NDTypeInterface>(permuteCastUser.getResult().getType());
            auto newResultType = resultType.changeMemSpace(
                    IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
            auto newPermuteCast = builder.create<VPU::PermuteCastOp>(
                    permuteCastUser.getLoc(), newResultType, permuteCastUser.getInput(),
                    permuteCastUser.getDstOrderAttr(), permuteCastUser.getMemPermAttr());
            permuteCastUser->getResult(0).replaceAllUsesWith(newPermuteCast);
            if (permuteCastUser.use_empty()) {
                permuteCastUser.erase();
            }
            auto newPermuteCastPtr = newPermuteCast.getOperation();
            propagateReturnType(newPermuteCastPtr, builder);
        } else {
            inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);
            propagateReturnType(user, builder);
        }
    }
}

bool checkForSameRoot(llvm::SmallVector<mlir::Operation*>& ops, mlir::Value root) {
    if (ops.empty()) {
        return false;
    }

    return llvm::all_of(ops, [&](mlir::Operation* op) {
        return mlir::cast<VPU::SliceOp>(op).getSource() == root;
    });
}

bool checkInputTypes(VPU::ConcatOp* concatOp, mlir::Operation*& inputCopy) {
    auto inputs = concatOp->getInputs();

    mlir::Operation* foundCopyOp = nullptr;
    int copyOpCount = 0;

    bool allSliceOrQuantize = true;
    bool allPermuteOrCopy = true;

    for (auto input : inputs) {
        if (mlir::isa<mlir::BlockArgument>(input)) {
            allSliceOrQuantize = false;
            allPermuteOrCopy = false;
            continue;
        }

        auto* defOp = input.getDefiningOp();

        if (mlir::isa<VPU::QuantizeCastOp>(defOp) || mlir::isa<VPU::CopyOp>(defOp)) {
            ++copyOpCount;
            if (!foundCopyOp) {
                foundCopyOp = defOp;
            }
        }

        if (!mlir::isa<VPU::SliceOp>(defOp) && !mlir::isa<VPU::QuantizeCastOp>(defOp)) {
            allSliceOrQuantize = false;
        }

        if (!mlir::isa<VPU::PermuteCastOp>(defOp) && !mlir::isa<VPU::CopyOp>(defOp)) {
            allPermuteOrCopy = false;
        }
    }

    if (copyOpCount != 1 || !foundCopyOp) {
        return false;
    }

    inputCopy = foundCopyOp;
    return allSliceOrQuantize || allPermuteOrCopy;
}

/*
 Possible patterns:
 1)
                              Copy (CMX->DDR)
                                     |
                                QuantizeCast
                   /                 |                   \
 N x     (   Slice   )    (copying the Pad input)    (   Slice   )  X M
                   \                 |                   /
                           Concat (output in DDR)

    becomes:

                               Copy (CMX->DDR)
                                     |
                                QuantizeCast
                                     |
                               Copy (DDR->CMX)
                   /                 |                   \
 N x     (   Slice   )    (copying the Pad input)    (   Slice   )  X M
                   \                 |                   /
                           Concat (output in CMX)
                                     |
                               Copy (CMX->DDR)

 2)
                                   Copy (CMX->DDR)
                     /                   |                  \
                   /           (copying the Pad input)        \
        (      Slice      )              |               (      Slice      )
        (        |        )        Copy (DDR->CMX)       (        |        )
        (        |        )              |               (        |        )
  N x   (        |        )           MaxPool            (        |        )    x M
        (        |        )              |               (        |        )
        (   PermuteCast   )        Copy (CMX->DDR)       (   PermuteCast   )
                       \                 |                   /
                               Concat (output in DDR)

    becomes:

                            ------ Copy (CMX->DDR)
                            |             |
                         --------- Copy (DDR->CMX) -------
                       /    |                              \
                     /       \                              \
                   /           (copying the Pad input)       \
        (      Slice      )              |               (      Slice      )
        (        |        )        Copy (DDR->CMX)       (        |        )
        (        |        )              |               (        |        )
  N x   (        |        )           MaxPool            (        |        )    x M
        (        |        )              |               (        |        )
        (   PermuteCast   )        Copy (CMX->DDR)       (   PermuteCast   )
                       \                 |                  /
                        \          Copy (DDR->CMX)         /
                         \               |                /
                               Concat (output in CMX)
                                         |
                                   Copy (CMX->DDR)
*/

bool matchReflectPadPatterns(const Logger& log, VPU::ConcatOp& concatOp,
                             SmallVector<mlir::Operation*>& opsWithSameSource, VPU::CopyOp& cmxToDdrCopy,
                             mlir::Operation*& inputCopy) {
    // check concat output to be in DDR
    auto concatOutput = concatOp.getOutput();
    auto concatOutputBuffType = mlir::cast<vpux::NDTypeInterface>(concatOutput.getType());
    if (concatOutputBuffType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        return false;
    }
    log.trace("Concat operation's output is in DDR.");

    // check input types
    if (!checkInputTypes(&concatOp, inputCopy)) {
        log.trace("Concat operation's input types don't match the pattern.");
        return false;
    }

    // check that the padding is done only on one dimension
    auto inputShape = to_small_vector(vpux::getShape(inputCopy->getResult(0)));
    auto outputShape = to_small_vector(vpux::getShape(concatOutput));
    auto dimToPad = -1;
    for (size_t idx = 0; idx < inputShape.size(); idx++) {
        if (outputShape[idx] != inputShape[idx]) {
            if (dimToPad != -1) {
                log.trace("Padding is done on more than one dimension.");
                return false;
            }
            dimToPad = idx;
        }
    }
    if (dimToPad == -1) {
        return false;
    }
    log.trace("Padding is done only on one dimension.");

    // check input pattern
    mlir::Value inputSource;
    auto concatInputs = concatOp.getInputs();
    auto hasPermuteCastInput = false;
    for (auto input : concatInputs) {
        auto inputOp = input.getDefiningOp();
        auto sliceInput = mlir::dyn_cast_or_null<VPU::SliceOp>(inputOp);
        auto permuteCastInput = mlir::dyn_cast_or_null<VPU::PermuteCastOp>(inputOp);
        if (sliceInput) {
            auto sliceSource = sliceInput.getSource();
            auto quantCast = mlir::dyn_cast_or_null<VPU::QuantizeCastOp>(sliceSource.getDefiningOp());
            if (!quantCast) {
                log.trace("Slice input source is not a QuantizeCast Op.");
                return false;
            }

            auto sliceOutput = sliceInput.getResult();
            auto sliceOutputShape = to_small_vector(vpux::getShape(sliceOutput));
            if (sliceOutputShape[dimToPad] != 1) {
                log.trace("Padding is not done with 1 on the padding dimension.");
                return false;
            }

            auto quantCastInput = quantCast.getInput().getDefiningOp();
            cmxToDdrCopy = mlir::dyn_cast_or_null<VPU::CopyOp>(quantCastInput);
            opsWithSameSource.push_back(inputOp);
            inputSource = sliceSource;
        } else if (permuteCastInput) {
            auto permuteCastSource = permuteCastInput.getInput().getDefiningOp();
            auto slice = mlir::dyn_cast_or_null<VPU::SliceOp>(permuteCastSource);
            if (!slice) {
                log.trace("PermuteCast input source is not a Slice Op.");
                return false;
            }

            auto permuteCastOutput = permuteCastInput.getOutput();
            auto permuteCastOutputShape = to_small_vector(vpux::getShape(permuteCastOutput));
            if (permuteCastOutputShape[dimToPad] != 1) {
                log.trace("Padding is not done with 1 on the padding dimension.");
                return false;
            }

            auto sliceSource = slice.getSource();
            cmxToDdrCopy = mlir::dyn_cast_or_null<VPU::CopyOp>(sliceSource.getDefiningOp());
            opsWithSameSource.push_back(permuteCastSource);
            inputSource = sliceSource;
            hasPermuteCastInput = true;
        } else if (mlir::isa<VPU::CopyOp>(inputOp) || mlir::isa<VPU::QuantizeCastOp>(inputOp)) {
            continue;
        } else {
            return false;
        }
    }

    if (!cmxToDdrCopy) {
        return false;
    }
    auto isCopyFromCMX = mlir::cast<vpux::NDTypeInterface>(cmxToDdrCopy.getInput().getType()).getMemoryKind() ==
                         VPU::MemoryKind::CMX_NN;
    auto isCopyToDDR = mlir::cast<vpux::NDTypeInterface>(cmxToDdrCopy.getOutput().getType()).getMemoryKind() ==
                       VPU::MemoryKind::DDR;
    if (!isCopyFromCMX || !isCopyToDDR) {
        log.trace("There is no CMX to DDR copy at the beginning of the pattern.");
        return false;
    }

    if (mlir::isa<VPU::CopyOp>(inputCopy)) {
        if (!hasPermuteCastInput) {
            return false;
        }

        auto inputCopyOp = mlir::cast<VPU::CopyOp>(inputCopy);
        auto isCopyFromCMX = mlir::cast<vpux::NDTypeInterface>(inputCopyOp.getInput().getType()).getMemoryKind() ==
                             VPU::MemoryKind::CMX_NN;
        auto isCopyToDDR = mlir::cast<vpux::NDTypeInterface>(inputCopyOp.getOutput().getType()).getMemoryKind() ==
                           VPU::MemoryKind::DDR;
        if (!isCopyFromCMX || !isCopyToDDR) {
            return false;
        }

        if (inputCopyOp != cmxToDdrCopy) {
            auto propagatedOp = inputCopyOp.getInput().getDefiningOp();
            auto propagatedOpInput = propagatedOp->getOperand(0).getDefiningOp();
            auto copyToCmx = mlir::dyn_cast_or_null<VPU::CopyOp>(propagatedOpInput);
            if (!copyToCmx || copyToCmx.getInput() != cmxToDdrCopy) {
                return false;
            }
        }
    }

    if (!checkForSameRoot(opsWithSameSource, inputSource)) {
        log.trace("The Slice ops don't have the same source op.");
        return false;
    }

    if (!fitsIntoCMX(inputSource, &concatOp)) {
        log.trace("Pattern match failed: ConcatView operation doesn't fit in CMX.");
        return false;
    }

    return true;
}

bool isMovingToCMXBeneficial(VPU::ConcatOp concatOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                             VPUNN::VPUDevice vpuDevice, int64_t numDMAPorts) {
    const auto memSpaceCMX =
            IndexedSymbolAttr::get(concatOp.getContext(), stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    auto outputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType());
    auto outputTypeCMX = outputType.changeMemSpace(memSpaceCMX);
    auto concatInputs = concatOp.getInputs();
    size_t dmasInDdrCost = 0;
    size_t dmasInCmxCost = 0;
    for (auto input : concatInputs) {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        dmasInDdrCost += getDMACost(inputType, outputType, vpuDevice, costModel, numDMAPorts);
        auto newInputType = inputType.changeMemSpace(memSpaceCMX);
        dmasInCmxCost += getDMACost(newInputType, outputTypeCMX, vpuDevice, costModel, numDMAPorts);
    }

    return dmasInCmxCost < dmasInDdrCost;
}

//
// MoveReflectPadToCMXPass
//

class MoveReflectPadToCMXPass final : public VPU::impl::MoveReflectPadToCMXBase<MoveReflectPadToCMXPass> {
public:
    explicit MoveReflectPadToCMXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveReflectPadToCMXPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto arch = config::getArch(module);
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    auto vpunnCostModel = vpux::VPU::CostModelConfig::createLayerCostModel(arch);
    auto vpuDevice = vpux::VPU::getVPUDeviceType(arch);

    func.walk([&](VPU::ConcatOp concatOp) {
        _log.trace("Found Concat operation '{0}' at '{1}'.", concatOp->getName(), concatOp->getLoc());

        mlir::OpBuilder builder(concatOp);
        auto nestedLog = _log.nest();

        VPU::CopyOp cmxToDdrCopy;
        mlir::Operation* inputCopy = nullptr;
        SmallVector<mlir::Operation*> opsWithSameSource;
        if (!matchReflectPadPatterns(nestedLog, concatOp, opsWithSameSource, cmxToDdrCopy, inputCopy)) {
            nestedLog.trace("Found wrong pattern, the op will not be moved to CMX.");
            return;
        }

        if (!isMovingToCMXBeneficial(concatOp, vpunnCostModel->get_TheoreticalDMA_cost_model_shared(), vpuDevice,
                                     numDMAPorts)) {
            nestedLog.trace("it is not beneficial to move the concat operation '{0}' at {1} to CMX.",
                            concatOp->getName(), concatOp->getLoc());
        }

        nestedLog.trace("Concat operation '{0}' at '{1}' will be moved to CMX.", concatOp->getName(),
                        concatOp->getLoc());

        // add CopyOps DDR -> CMX and infer return types
        auto handleCopy = [&](mlir::Operation* sourceOp) {
            builder.setInsertionPointAfter(sourceOp);
            auto opResult = sourceOp->getResult(0);
            const auto memSpaceCMX =
                    IndexedSymbolAttr::get(concatOp.getContext(), stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
            auto allocType = mlir::cast<NDTypeInterface>(opResult.getType());
            auto newOutputType = allocType.changeMemSpace(memSpaceCMX);
            auto copyDDRToCMX = builder.create<VPU::CopyOp>(sourceOp->getLoc(), newOutputType, opResult, memSpaceCMX);
            opResult.replaceUsesWithIf(copyDDRToCMX, [&](mlir::OpOperand& operand) {
                return llvm::is_contained(opsWithSameSource, operand.getOwner());
            });
            return copyDDRToCMX;
        };

        auto slicesSource = mlir::cast<VPU::SliceOp>(opsWithSameSource[0]).getSource().getDefiningOp();
        if (inputCopy == slicesSource) {
            opsWithSameSource.push_back(concatOp);
        } else {
            builder.setInsertionPointAfter(inputCopy);
            const auto memSpaceCMX =
                    IndexedSymbolAttr::get(concatOp.getContext(), stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
            auto opResult = inputCopy->getResult(0);
            auto allocType = mlir::cast<NDTypeInterface>(opResult.getType());
            auto newOutputType = allocType.changeMemSpace(memSpaceCMX);
            auto copyDDRToCMX = builder.create<VPU::CopyOp>(inputCopy->getLoc(), newOutputType, opResult, memSpaceCMX);
            opResult.replaceAllUsesExcept(copyDDRToCMX, copyDDRToCMX);
            propagateReturnType(inputCopy, builder);
        }

        handleCopy(slicesSource);
        for (mlir::Operation*& op : opsWithSameSource) {
            if (mlir::isa<VPU::ConcatOp>(op)) {
                continue;
            }
            inferReturnTypes(op, vpux::InferShapedTypeMode::ALL);
            propagateReturnType(op, builder);
        }

        // move concatOp output back to DDR
        builder.setInsertionPointAfter(concatOp);
        inferReturnTypes(concatOp, vpux::InferShapedTypeMode::ALL);
        auto copyOutputToDdr = builder.create<VPU::CopyOp>(concatOp.getLoc(), concatOp.getOutput());
        concatOp->getResult(0).replaceAllUsesExcept(copyOutputToDdr, copyOutputToDdr);
    });
}
}  // namespace

//
// createMoveReflectPadToCMXPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createMoveReflectPadToCMXPass(Logger log) {
    return std::make_unique<MoveReflectPadToCMXPass>(log);
}
