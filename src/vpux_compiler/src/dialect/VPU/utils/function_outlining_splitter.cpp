//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/function_outlining_splitter.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/outlining_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

using namespace vpux;

namespace {

//
// VFOutliningSplitter
//

class VFOutliningSplitter {
public:
    VFOutliningSplitter(size_t verticalFusionTileThreshold, Logger log)
            : _verticalFusionTileThreshold(verticalFusionTileThreshold), _log(log) {
    }
    SmallVector<OutliningInstance> getOutliningInstances(mlir::func::FuncOp mainFunction);

private:
    // checks for supported patterns which will be added with root op
    bool isProcessInputOp(mlir::Operation* op);
    bool isConcatWithOutlinedInputs(mlir::Operation* op);
    bool isProcessOutputOp(mlir::Operation* op);

    // recursively process operations on the input, add supported ops to the outlining instance
    void processInputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations);
    // add operation with inputs and outputs to outlining instance
    void processOperation(mlir::Operation& op, OpOrderedSet& storedOperations);
    // recursively process operations on the output, add supported ops to the outlining instance
    void processOutputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations);

    // Move storage op to instance
    void moveOpsToInstance(OpOrderedSet& instanceOps, SmallVector<OpOrderedSet>& instances, bool target = false);
    // Convert stored operations to outlining instance
    void createOutliningInstanceFromStorage(ValueOrderedSet& storedInputs, ValueOrderedSet& storedOutputs,
                                            OpOrderedSet& storedOperations, SmallVector<OutliningInstance>& instances);

private:
    mlir::DenseSet<mlir::Operation*> _outlinedOperations;
    // Thresholds for outlining, avoid creating very small functions
    size_t _verticalFusionTileThreshold;
    Logger _log;
};

bool isSupportedSkipOp(mlir::Operation* op) {
    // TODO: E#140556 include more view ops
    return mlir::isa<VPU::QuantizeCastOp, VPU::ShapeCastOp, VPU::AffineReshapeOp, VPU::PermuteCastOp, VPU::ExpandOp>(
            op);
}

mlir::Operation* trySearchForRootOp(mlir::Operation* op) {
    if (!isSupportedSkipOp(op)) {
        return nullptr;
    }
    if (!op->getResult(0).hasOneUse()) {
        return nullptr;
    }
    return *op->getResult(0).getUsers().begin();
}

bool VFOutliningSplitter::isProcessInputOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }

    if (mlir::isa<VPU::SliceOp>(op)) {
        return isProcessInputOp(op->getOperand(0).getDefiningOp());
    }

    if (VPU::isConstOperandOp(op)) {
        return true;
    }

    if (_outlinedOperations.find(op) != _outlinedOperations.end()) {
        return false;
    }

    if (auto tryOp = trySearchForRootOp(op)) {
        // Do no add 'ViewOp -> Return' to current outlining instance
        return !mlir::isa<mlir::func::ReturnOp>(tryOp);
    }

    // TODO: E#140556 generic view ops
    if (mlir::isa<VPU::SliceOp>(op)) {
        // Only add 'BlockArg -> Slice' to current outlining instance
        return mlir::isa<mlir::BlockArgument>(op->getOperand(0));
    }

    return false;
}

bool VFOutliningSplitter::isConcatWithOutlinedInputs(mlir::Operation* op) {
    if (!mlir::isa<VPU::ConcatOp>(op)) {
        return false;
    }
    for (auto operand : op->getOperands()) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp == nullptr) {
            continue;
        }
        if (_outlinedOperations.contains(parentOp)) {
            continue;
        }
        return false;
    }
    return true;
}

bool VFOutliningSplitter::isProcessOutputOp(mlir::Operation* op) {
    // skip supported ops, search for root op
    while (auto tryOp = trySearchForRootOp(op)) {
        if (mlir::isa<mlir::func::ReturnOp>(tryOp)) {
            return true;
        }
        op = tryOp;
    }

    // TODO: E#140556 generic view ops
    if (mlir::isa<VPU::SliceOp>(op)) {
        // 'BlockArg -> Slice' case handled in 'isProcessInputOp'
        if (mlir::isa<mlir::BlockArgument>(op->getOperand(0))) {
            return false;
        }
        if (op->getOperand(0).getDefiningOp<VPU::ConcatOp>() != nullptr) {
            // 'Concat -> Slice' or 'Concat -> Concat' patterns will
            // be processed recursively
            for (auto userOp : op->getOperand(0).getUsers()) {
                if (!mlir::isa<VPU::SliceOp, VPU::ConcatOp>(userOp)) {
                    // slice input has other uses, do not process
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

void VFOutliningSplitter::processOperation(mlir::Operation& op, OpOrderedSet& storedOperations) {
    if (storedOperations.find(&op) != storedOperations.end()) {
        return;
    }
    storedOperations.insert(&op);
    _outlinedOperations.insert(&op);
}

void VFOutliningSplitter::processInputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations) {
    // recursively add supported ops to the outlining instance
    for (auto operand : op.getOperands()) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp == nullptr) {
            continue;
        }
        if (isProcessInputOp(parentOp)) {
            processInputsRecursively(*parentOp, storedOperations);
            processOperation(*parentOp, storedOperations);
        }
    }
}

void VFOutliningSplitter::processOutputsRecursively(mlir::Operation& op, OpOrderedSet& storedOperations) {
    // recursively add supported ops to the outlining instance
    mlir::SmallVector<mlir::Operation*> stepOps;
    for (auto result : op.getResults()) {
        for (auto userOp : result.getUsers()) {
            if (_outlinedOperations.find(userOp) != _outlinedOperations.end()) {
                continue;
            }
            if (isProcessOutputOp(userOp)) {
                stepOps.push_back(userOp);
            } else if (isConcatWithOutlinedInputs(userOp)) {
                stepOps.push_back(userOp);
            }
        }
    }
    // BFS to preserve order of operations
    for (auto stepOp : stepOps) {
        processOperation(*stepOp, storedOperations);
        processOutputsRecursively(*stepOp, storedOperations);
    }
}

void VFOutliningSplitter::createOutliningInstanceFromStorage(ValueOrderedSet& storedInputs,
                                                             ValueOrderedSet& storedOutputs,
                                                             OpOrderedSet& storedOperations,
                                                             SmallVector<OutliningInstance>& instances) {
    auto currentSlice = IRSlice();
    for (auto op : storedOperations) {
        currentSlice.operations.push_back(op);
    }

    for (auto operand : llvm::make_early_inc_range(storedInputs)) {
        if (storedOutputs.find(operand) != storedOutputs.end()) {
            storedInputs.erase(operand);
        }
    }

    for (auto operand : llvm::make_early_inc_range(storedOutputs)) {
        const auto hasOutsideUser = llvm::any_of(operand.getUsers(), [&](mlir::Operation* op) {
            return storedOperations.find(op) == storedOperations.end();
        });
        if (!hasOutsideUser) {
            storedOutputs.erase(operand);
        } else if (mlir::isa<VPU::SparseTensorType>(operand.getType())) {
            // TODO: E#140551 support GroupSparseTensorOp as function arg
            return;
        } else if (auto constOp = operand.getDefiningOp<Const::DeclareOp>()) {
            if (Const::hasSparsifyTransformation(constOp)) {
                // If a constant op has sparsify transformation, its user must be the VPU.GroupSparseTensor op
                // TODO: E#173628 duplicating the Const::DeclareOp and VPU.GroupSparseTensorOp for function outlining
                return;
            }
        }
    }

    for (auto inputs : storedInputs) {
        currentSlice.inputs.push_back(inputs);
    }
    for (auto outputs : storedOutputs) {
        currentSlice.outputs.push_back(outputs);
    }

    if (currentSlice.operations.empty() || currentSlice.inputs.empty() || currentSlice.outputs.empty()) {
        _log.error("At least one instance has no outputs values, which results in an empty function");
        return;
    }

    // stored ops moved into outlined instance
    storedInputs.clear();
    storedOutputs.clear();
    storedOperations.clear();
    instances.push_back(OutliningInstance{std::move(currentSlice)});
}

void VFOutliningSplitter::moveOpsToInstance(OpOrderedSet& instanceOps, SmallVector<OpOrderedSet>& instances,
                                            bool target) {
    if (!target) {
        // check if only skip ops in instance
        const auto onlySkipOps = llvm::all_of(instanceOps, [&](mlir::Operation* op) {
            return isSupportedSkipOp(op);
        });

        // try to add ops to parent instances
        if (onlySkipOps) {
            auto firstOp = *instanceOps.begin();
            if (auto parentOp = firstOp->getOperand(0).getDefiningOp()) {
                for (auto& instance : instances) {
                    if (instance.find(parentOp) != instance.end()) {
                        // found parent instance
                        instance.insert(instanceOps.begin(), instanceOps.end());
                        instanceOps.clear();
                        return;
                    }
                }
            }
        }
    }

    // move to new instance
    instances.push_back(instanceOps);
    instanceOps.clear();
}

SmallVector<OutliningInstance> VFOutliningSplitter::getOutliningInstances(mlir::func::FuncOp netFunc) {
    // High level overview:
    // Walk though operations in IR and add them to stored ops with inputs and outputs
    // Recursively add supported inputs and outputs for the operations to storage.
    // If outlining instance was found (vertical fusion operation satisfying tile threshold),
    // outlining instance need to be created, creating outlining instance can have 3 cases:

    // CASE1. There are some operation stored and current operation is vertical fusion outlining instance
    //          1. Create outlining instance for the stored operations
    //          2. Add VF operation to storage, recursively add supported input and output ops
    //          3. Create outlining instance for the VF operation

    // CASE3. Current operation is vertical fusion outlining instance
    //          1. Add VF operation to storage, recursively add supported input and output ops
    //          2. Create outlining instance for the VF operation

    // CASE2. ReturnOp is reached in the main function:
    //          1. Recursively add supported input operations to the ReturnOp
    //          2. If there are any operations stored - create outlining instance for the last operations.

    SmallVector<OpOrderedSet> opInstances;
    OpOrderedSet instanceOps;

    const auto isOpInCurrentOutliningInstance = [&](mlir::Operation* op) {
        return instanceOps.find(op) != instanceOps.end();
    };

    const auto isParallelConcatInput = [&](mlir::Operation* op, bool tiledOnMultiDims) {
        /*                          ... ...  Op
                                      \  |  /  \
        Check for pattern:      ...    Concat  Concat
                                  \    /
                                   VFOp
        */
        // parallel concat sequences can be well optimized, but require:
        // "producers and consumers in the same function"
        // TODO: E#140555 optimize concat across functions
        const bool vfOpInStorage = isOpInCurrentOutliningInstance(op);
        for (auto input : op->getOperands()) {
            if (auto concatOp = input.getDefiningOp<VPU::ConcatOp>()) {
                const bool concatOpInStorage = isOpInCurrentOutliningInstance(concatOp);
                for (auto concatInput : concatOp.getInputs()) {
                    for (auto user : concatInput.getUsers()) {
                        if (user == concatOp || !mlir::isa_and_nonnull<VPU::ConcatOp>(user)) {
                            continue;
                        }
                        if (!tiledOnMultiDims && vfOpInStorage == concatOpInStorage &&
                            concatOpInStorage == isOpInCurrentOutliningInstance(user)) {
                            // if VF is tiled on multiple dimensions, need to further avoid separating VF ops into
                            // different function ops
                            continue;
                        }
                        // VFOp is a consumer of parallel concat, can not outline since
                        // would result in parallel concat consumers in different function
                        return true;
                    }
                }
            }
        }
        return false;
    };

    /*
        Given pattern:
            -> Concat / tiled compute op -> Slice (on K) -> VerticalFusion tiling-on-K { ... }
        If we outline right between the Slice and the Verticalfusion block we may lose the opportunity to optimize
        a DDR2DDR DMA transaction.

        At VPUIP, this pattern may translate to:
            outlined func {n x Copy -> Concat -> Slice -> Copy (DDR2DDR, remove strides) } -> outlined VF func

        The DDR2DDR copy that removes the strides cannot be optimized out, as the outlined block should return compact
        strides.

        Without outlining, the pattern may translate to:
            n x Copy -> Concat -> Slice (on K) -> Copy (DDR2DDR, remove strides) -> Slice (for tiling, on K) -> Copy
       (DDR2CMX)

        Copy optimization patterns should be able to merge the Slice -> Copy -> Slice pattern, thus removing
        the DDR2DDR copy.

        Other VF blocks can appear between the slice producer and the VF consumer. Keep this whole interval
        together. Rejecting only the consumer as an outlining candidate does not prevent an earlier split.

        Ticket for improvement: E#215360
    */
    const auto getPotentialDDR2DDRCopyProducers = [&](VPU::VerticalFusionOp vfOp) {
        OpOrderedSet sliceProducers;
        const auto tilingStrategy = parseIntArrayAttr<size_t>(vfOp.getTilingStrategy());
        for (auto operand : vfOp.getOperands()) {
            auto sliceOp = mlir::dyn_cast_if_present<VPU::SliceOp>(operand.getDefiningOp());
            if (sliceOp == nullptr) {
                continue;
            }

            auto sourceOp = sliceOp.getSource().getDefiningOp();
            if (auto concatOp = mlir::dyn_cast_if_present<VPU::ConcatOp>(sourceOp)) {
                // If Concat can fit in CMX, there is a good chance it will be moved. Thus, the DDR2DDR copy
                // could be optimized out by fusing it to the CMX2DDR transaction.
                if (concatOp.fitIntoCMX(mlir::cast<NDTypeInterface>(concatOp.getOutput().getType()))) {
                    continue;
                }
            } else if (auto tilingOp = mlir::dyn_cast_if_present<VPU::TilingBuilderOpInterface>(sourceOp)) {
                auto producerTilingAttr = tilingOp->getAttrOfType<mlir::ArrayAttr>("tilingStrategy");
                if (producerTilingAttr == nullptr) {
                    continue;
                }
                const auto producerTiling = parseIntArrayAttr<size_t>(producerTilingAttr);
                if (llvm::none_of(producerTiling, [](auto tiles) {
                        return tiles > 1;
                    })) {
                    continue;
                }
            } else {
                continue;
            }

            auto inputShape = getShape(sliceOp.getSource());
            auto outputShape = getShape(sliceOp.getResult());

            SmallVector<Dim> diffInOutSizeDims;
            const auto ioShapes = zip(inputShape, outputShape);
            for (const auto& ioShape : ioShapes | indexed) {
                const auto inSize = std::get<0>(ioShape.value());
                const auto outSize = std::get<1>(ioShape.value());
                if (inSize != outSize) {
                    diffInOutSizeDims.push_back(Dim(ioShape.index()));
                }
            }

            for (const auto& sliceDim : diffInOutSizeDims) {
                if (tilingStrategy[sliceDim.ind()] > 1) {
                    // slice is on the same axis as tiling, can not outline since would result in slice and VF ops
                    // in different functions
                    sliceProducers.insert(sourceOp);
                    break;
                }
            }
        }
        return sliceProducers;
    };

    const auto isVfOutliningInstanceCandidate = [&](mlir::Operation* op, VPU::VerticalFusionOp lastDependentVF) {
        auto vfOp = mlir::dyn_cast_if_present<VPU::VerticalFusionOp>(op);
        if (vfOp == nullptr) {
            return false;
        }

        const auto tilingStrategy = parseIntArrayAttr<size_t>(vfOp.getTilingStrategy());
        const auto numTiles =
                std::accumulate(tilingStrategy.begin(), tilingStrategy.end(), size_t(1), std::multiplies<size_t>());
        if (numTiles < _verticalFusionTileThreshold) {
            return false;
        }

        // An active slice dependency blocks outlining until the walk passes its last VF consumer.
        const bool hasPendingSliceConsumer = lastDependentVF != nullptr && !lastDependentVF->isBeforeInBlock(vfOp);
        if (hasPendingSliceConsumer) {
            _log.nest().trace("Outlining VerticalFusion block may potentially leave DDR to DDR copies unoptimized; "
                              "do not outline.");
            return false;
        }

        const auto tiledOnMultiDims = llvm::count_if(tilingStrategy, [](auto value) {
                                          return value > 1;
                                      }) > 1;
        return !isParallelConcatInput(op, tiledOnMultiDims);
    };

    // Discover dependencies before outlining: a slice consumer can appear much later than its producer.
    // Block-order iteration makes the last assignment record the latest consumer for each producer.
    mlir::DenseMap<mlir::Operation*, VPU::VerticalFusionOp> lastDependentVFs;
    for (auto vfOp : netFunc.getOps<VPU::VerticalFusionOp>()) {
        for (auto producer : getPotentialDDR2DDRCopyProducers(vfOp)) {
            lastDependentVFs[producer] = vfOp;
        }
    }

    // Activate protection at the producer and extend the endpoint when dependency intervals overlap.
    const auto updateLastDependentVF = [&lastDependentVFs](mlir::Operation* op,
                                                           VPU::VerticalFusionOp& lastDependentVF) {
        const auto dependentVF = lastDependentVFs.find(op);
        if (dependentVF == lastDependentVFs.end()) {
            return;
        }
        if (lastDependentVF == nullptr || lastDependentVF->isBeforeInBlock(dependentVF->second)) {
            lastDependentVF = dependentVF->second;
        }
    };

    // Track the furthest consumer of any producer encountered by the main walk.
    // No reset is needed: candidates strictly after this consumer are no longer protected.
    VPU::VerticalFusionOp lastDependentVF = nullptr;
    for (auto& op : netFunc.getOps()) {
        // Update before skipping processed operations: recursive collection may have visited the producer already.
        updateLastDependentVF(&op, lastDependentVF);

        if (_outlinedOperations.find(&op) != _outlinedOperations.end()) {
            // skip already outlined operations
            continue;
        }
        if (isProcessInputOp(&op) || isProcessOutputOp(&op)) {
            // process input and output operations will be added
            // with their root operation to outlining instance
            continue;
        }

        // Building outlining instance CASE 1
        if (!instanceOps.empty() && isVfOutliningInstanceCandidate(&op, lastDependentVF)) {
            // operation exist in outlining instance and current op is target for outlining
            // need to add new instance from the previous operations in storage
            // and instance for current op will be build in CASE 3
            moveOpsToInstance(instanceOps, opInstances);
        }

        // Building outlining instance CASE 2
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            // recurse and add input ops
            processInputsRecursively(op, instanceOps);
            if (!instanceOps.empty()) {
                // add final outlining instance for operations linked to results of function
                moveOpsToInstance(instanceOps, opInstances);
            }
            // result op does not need to be added, new ops will be created during outlining
            continue;
        }

        // add operation(s) to storage of current outlining instance
        processInputsRecursively(op, instanceOps);
        processOperation(op, instanceOps);
        processOutputsRecursively(op, instanceOps);

        // Building outlining instance CASE 3
        if (isVfOutliningInstanceCandidate(&op, lastDependentVF)) {
            // current op is target for outlining, add new instance
            moveOpsToInstance(instanceOps, opInstances, true);
        }
    }

    // convert to outlining instances
    SmallVector<OutliningInstance> instances;
    OpOrderedSet storedOperations;
    ValueOrderedSet storedInputs;
    ValueOrderedSet storedOutputs;

    for (auto& opInstance : opInstances) {
        storedOperations.insert(opInstance.begin(), opInstance.end());
        for (auto& op : storedOperations) {
            for (auto operand : op->getOperands()) {
                storedInputs.insert(operand);
            }
            for (auto result : op->getResults()) {
                storedOutputs.insert(result);
            }
        }
        createOutliningInstanceFromStorage(storedInputs, storedOutputs, storedOperations, instances);
    }

    return instances;
}

}  // namespace

vpux::VPU::FunctionOutlinerVerticalFusion::FunctionOutlinerVerticalFusion(size_t numInstanceThreshold,
                                                                          size_t verticalFusionTileThreshold,
                                                                          Logger log)
        : _numInstanceThreshold(numInstanceThreshold),
          _verticalFusionTileThreshold(verticalFusionTileThreshold),
          _log(log) {
    _log.setName("function-outliner-vertical-fusion");
}

SmallVector<OutliningInstance> vpux::VPU::FunctionOutlinerVerticalFusion::getOutliningTargets(
        mlir::func::FuncOp mainFunction) {
    _log.debug("Searching for outlining targets with a vertical fusion split strategy");
    VFOutliningSplitter vfSplitter(_verticalFusionTileThreshold, _log);
    const auto outliningInstances = vfSplitter.getOutliningInstances(mainFunction);
    if (outliningInstances.size() < _numInstanceThreshold) {
        return {};
    }
    printOutliningInstances(outliningInstances, _log);
    return outliningInstances;
}
