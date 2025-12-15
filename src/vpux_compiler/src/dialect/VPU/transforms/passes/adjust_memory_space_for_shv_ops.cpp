//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTMEMORYSPACEFORSHVOPS
#define GEN_PASS_DEF_ADJUSTMEMORYSPACEFORSHVOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {
struct CMXIndices {
    mlir::DenseSet<size_t> operands;
    mlir::DenseSet<size_t> results;
};
}  // namespace

namespace {

class AdjustMemorySpaceForSHVOpsPass final :
        public VPU::impl::AdjustMemorySpaceForSHVOpsBase<AdjustMemorySpaceForSHVOpsPass> {
public:
    explicit AdjustMemorySpaceForSHVOpsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto funcOp = getOperation();
        auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();

        std::vector<mlir::Operation*> shaveOps;
        funcOp.walk([&](VPUIP::SoftwareLayerOpInterface op) {
            // Some operations have multiple bufferization paths, one of them being on Shave
            // Skip the operations that will not be lowered to Shave
            if (!willBeBufferizedToShave(op)) {
                return;
            }
            // Skip operations that have distributed types, as their data is already placed in CMX
            auto hasDistributedType = [](mlir::Value value) {
                if (auto distributedIf = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(value.getType())) {
                    return distributedIf.containsDistributedTypes();
                }
                return false;
            };
            if (llvm::any_of(op->getOperands(), hasDistributedType) ||
                llvm::any_of(op->getResults(), hasDistributedType)) {
                return;
            }
            shaveOps.push_back(op);
        });

        const auto totalAvailableCMX = VPU::getTotalCMXSize(moduleOp).count();
        const auto memSpaceCMX = IndexedSymbolAttr::get(&getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
        for (auto op : shaveOps) {
            moveIntoCMX(op, memSpaceCMX, totalAvailableCMX);
        }
    }

    bool willBeBufferizedToShave(mlir::Operation* op) const {
        return llvm::TypeSwitch<mlir::Operation*, bool>(op)
                .Case<VPU::ConcatOp>([&](VPU::ConcatOp concatOp) {
                    return !canBeBufferizedToCopies(concatOp);
                })
                .Case<VPU::StridedSliceOp>([&](VPU::StridedSliceOp stridedSliceOp) {
                    return !canBeBufferizedToCopies(stridedSliceOp);
                })
                .Case<VPU::PermuteCastOp>([&](VPU::PermuteCastOp permuteCastOp) {
                    return !canBeBufferizedToCast(permuteCastOp);
                })
                .Case<VPU::ConvertOp>([&](VPU::ConvertOp convertOp) {
                    return !isConvertSupportedOnDMA(convertOp);
                })
                .Default([](mlir::Operation*) {
                    return true;
                });
    }

    void moveIntoCMX(mlir::Operation* op, IndexedSymbolAttr memSpaceCMX, int64_t totalAvailableCMX) {
        const auto copyIntoMemSpace = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value,
                                         vpux::IndexedSymbolAttr dstMemSpace) -> mlir::Value {
            return builder.createOrFold<VPU::CopyOp>(loc, value, dstMemSpace);
        };

        _log.trace("Moving '{0}' at '{1}' into '{2}'", op->getName(), op->getLoc(), memSpaceCMX);

        const auto cmxIndices = getOptimalCMXPlacement(op, totalAvailableCMX);

        mlir::OpBuilder builder(op);
        for (auto& operand : llvm::make_early_inc_range(op->getOpOperands())) {
            if (cmxIndices.operands.find(operand.getOperandNumber()) == cmxIndices.operands.end()) {
                _log.nest().trace("Operand {0} should remain in DDR, according to the mapping",
                                  operand.getOperandNumber());
                continue;
            }
            auto input = operand.get();
            auto origInType = mlir::cast<vpux::NDTypeInterface>(input.getType());
            if (origInType.getMemSpace() == memSpaceCMX) {
                _log.nest().trace("Operand {0} is already in CMX", operand.getOperandNumber());
                continue;
            }
            _log.nest().trace("Moving operand {0} to CMX", operand.getOperandNumber());
            const auto copiedInput = copyIntoMemSpace(
                    builder, appendLoc(op->getLoc(), "input-{0}-CMX", operand.getOperandNumber()), input, memSpaceCMX);
            operand.set(copiedInput);
        }

        for (auto result : llvm::make_early_inc_range(op->getOpResults())) {
            if (cmxIndices.results.find(result.getResultNumber()) == cmxIndices.results.end()) {
                _log.nest().trace("Result {0} should remain in DDR, according to the mapping",
                                  result.getResultNumber());
                continue;
            }
            const auto origOutType = mlir::cast<vpux::NDTypeInterface>(result.getType());
            const auto origOutMemSpace = origOutType.getMemSpace();
            if (origOutMemSpace == memSpaceCMX) {
                _log.nest().trace("Result {0} is already in CMX", result.getResultNumber());
                continue;
            }
            _log.nest().trace("Moving result {0} to CMX", result.getResultNumber());
            builder.setInsertionPointAfter(op);
            const auto newOutType = origOutType.changeMemSpace(memSpaceCMX);
            result.setType(newOutType);
            const auto copiedOutput =
                    copyIntoMemSpace(builder, appendLoc(op->getLoc(), "output-DDR"), result, origOutMemSpace);
            result.replaceAllUsesExcept(copiedOutput, copiedOutput.getDefiningOp());
        }
    }

    /**
     * @brief Find which operands and results of the operation fit into CMX
     * @details The method identifies the operands and results of the operation which fit into CMX. In the best case
     * scenario, all of the values fit. Otherwise, the maximal subset is identified such that the CMX utilization is
     * maximized.
     * @param op The operation whose operands and results are analyzed
     * @param totalAvailableCMX the upper limit of CMX memory available for the operands and results
     * @return a structure containing the indices of the operands and results that fit in CMX
     */
    CMXIndices getOptimalCMXPlacement(mlir::Operation* op, int64_t totalAvailableCMX) {
        _log.nest().trace("Searching for optimal CMX placement");

        // Create an array of all the inputs and outputs and index them
        const auto inputs = op->getOperands();
        const auto outputs = op->getResults();
        SmallVector<mlir::Value> mergedVals;
        mergedVals.reserve(inputs.size() + outputs.size());
        mergedVals.insert(mergedVals.end(), inputs.begin(), inputs.end());
        mergedVals.insert(mergedVals.end(), outputs.begin(), outputs.end());
        SmallVector<size_t> idxVec(inputs.size() + outputs.size(), 0);
        std::iota(idxVec.begin(), idxVec.end(), 0);

        mlir::DenseSet<size_t> inputsForCMX;
        mlir::DenseSet<size_t> outputsForCMX;

        SmallVector<Byte> ioCmxSizes;
        for (const auto& val : mergedVals) {
            ioCmxSizes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getTotalAllocSize());
        }

        Byte defaultCMXOffsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT);
        Byte defaultCMXSizeAlignment = Byte(1);

        auto requiredSizeForAllIO = vpux::calculateAlignedBuffersMemoryRequirement(
                                            ioCmxSizes, defaultCMXOffsetAlignment, defaultCMXSizeAlignment)
                                            .count();
        // If all inputs and outputs already fit in CMX, this is already optimal
        if (requiredSizeForAllIO <= totalAvailableCMX) {
            _log.nest(2).trace("All the inputs and outputs will fit in CMX. Total size: {0} bytes",
                               requiredSizeForAllIO);
            auto inputsSize = inputs.size();
            for (size_t i = 0; i < inputsSize; ++i) {
                inputsForCMX.insert(i);
            }
            auto outputsSize = outputs.size();
            for (size_t i = 0; i < outputsSize; ++i) {
                outputsForCMX.insert(i);
            }
            return CMXIndices{inputsForCMX, outputsForCMX};
        }

        // Not all inputs and outputs fit in the available CMX space. Find the maximal subset that fits,
        // such that we use CMX as much as possible.
        SmallVector<std::pair<SmallVector<size_t>, int64_t>> subsets;
        SmallVector<size_t> aux;

        auto genSubsets = [](const auto& idxVec, auto& subsets, auto& aux, size_t currentIdx,
                             auto& genSubsetsFunc) -> void {
            subsets.push_back({aux, 0});
            for (auto i{currentIdx}; i < idxVec.size(); ++i) {
                aux.push_back(idxVec[i]);
                genSubsetsFunc(idxVec, subsets, aux, i + 1, genSubsetsFunc);
                aux.pop_back();
            }
        };

        // Generate all subsets
        genSubsets(idxVec, subsets, aux, 0, genSubsets);

        // For each subset, compute the total necessary NNCMX size
        for (auto& p : subsets) {
            const auto& idxVec = p.first;
            SmallVector<Byte> bufferSizes;
            bufferSizes.reserve(idxVec.size());
            for (const auto& idx : idxVec) {
                bufferSizes.push_back(mlir::cast<vpux::NDTypeInterface>(mergedVals[idx].getType()).getTotalAllocSize());
            }
            p.second = vpux::calculateAlignedBuffersMemoryRequirement(bufferSizes, defaultCMXOffsetAlignment,
                                                                      defaultCMXSizeAlignment)
                               .count();
        }

        std::sort(subsets.begin(), subsets.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second < rhs.second;
        });

        // Find the subset that uses the most CMX without going over the limit
        std::optional<size_t> subsetIdx;
        for (size_t idx = 0; idx < subsets.size(); ++idx) {
            if (subsets[idx].second > totalAvailableCMX) {
                if (idx > 0) {
                    subsetIdx = idx - 1;
                }
                break;
            }
        }
        if (!subsetIdx.has_value()) {
            return CMXIndices{inputsForCMX, outputsForCMX};
        }

        const auto maxValidCmxUsageSubsetIdx = std::min(subsetIdx.value(), subsets.size() - 1);
        for (const auto& idx : subsets[maxValidCmxUsageSubsetIdx].first) {
            if (idx < inputs.size()) {
                inputsForCMX.insert(idx);
            } else {
                outputsForCMX.insert(idx - inputs.size());
            }
        }

        if (_log.isActive(LogLevel::Trace)) {
            _log.nest(2).trace("Following inputs and outputs will be mapped to CMX:");
            _log.nest(2).trace("Inputs:");
            for (const auto& i : inputsForCMX) {
                _log.nest(3).trace("'{0}'", i);
            }
            _log.nest(2).trace("Outputs:");
            for (const auto& o : outputsForCMX) {
                _log.nest(3).trace("'{0}'", o);
            }
        }

        return CMXIndices{inputsForCMX, outputsForCMX};
    }
};

}  // namespace

//
// createAdjustMemorySpaceForSHVOpsPass
//

std::unique_ptr<mlir::Pass> VPU::createAdjustMemorySpaceForSHVOpsPass(const Logger& log) {
    return std::make_unique<AdjustMemorySpaceForSHVOpsPass>(log);
}
