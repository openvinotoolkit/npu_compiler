//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
using namespace vpux;

namespace {

//
// getOptimalCMXPlacement
//

std::pair<mlir::DenseSet<size_t>, mlir::DenseSet<size_t>> getOptimalCMXPlacement(ArrayRef<mlir::Value> inputs,
                                                                                 ArrayRef<mlir::Value> outputs,
                                                                                 Byte reservedMem,
                                                                                 mlir::ModuleOp module,
                                                                                 const Logger& log) {
    VPUX_THROW_WHEN(inputs.empty() && outputs.empty(), "Received empty input and output arrays!");
    auto nestedLog2 = log.nest(2);
    auto nestedLog3 = log.nest(3);
    // Create an array of all the inputs and outputs and index them
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
        ioCmxSizes.push_back(val.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize());
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? VPU::getTotalCMXSize(module).count()
                                                          : VPU::getTotalCMXFragmentationAwareSize(module).count();

    nestedLog2.trace("Total available CMX size: {0} bytes", totalAvailableCMXSize);

    Byte defaultCMXOffsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT);
    Byte defaultCMXSizeAlignment = Byte(1);

    auto requiredSizeForAllIO = vpux::calculateAlignedBuffersMemoryRequirement(ioCmxSizes, defaultCMXOffsetAlignment,
                                                                               defaultCMXSizeAlignment)
                                        .count();
    // If all inputs and outputs already fit in NNCMX, this is already optimal
    if (requiredSizeForAllIO + reservedMem.count() <= totalAvailableCMXSize) {
        nestedLog2.trace("All the inputs and outputs will fit in CMX. Total size: {0} bytes", requiredSizeForAllIO);
        auto inputsSize = inputs.size();
        for (size_t i = 0; i < inputsSize; ++i) {
            inputsForCMX.insert(i);
        }
        auto outputsSize = outputs.size();
        for (size_t i = 0; i < outputsSize; ++i) {
            outputsForCMX.insert(i);
        }
        return {inputsForCMX, outputsForCMX};
    }

    // Not all inputs and outputs fit in the available NNCMX space. Find the maximal subset that fits,
    // such that we use NNCMX as much as possible.
    SmallVector<std::pair<SmallVector<size_t>, int64_t>> subsets;
    SmallVector<size_t> aux;

    auto genSubsets = [](const auto& idxVec, auto& subsets, auto& aux, size_t currentIdx,
                         auto& genSubsetsFunc) -> void {
        subsets.push_back({aux, 0});
        for (size_t i = currentIdx; i < idxVec.size(); ++i) {
            aux.push_back(idxVec[i]);
            genSubsetsFunc(idxVec, subsets, aux, i + 1, genSubsetsFunc);
            aux.pop_back();
        }
    };

    // Generate all subsets
    genSubsets(idxVec, subsets, aux, 0, genSubsets);

    // For each subset, compute the total necesssary NNCMX size
    for (auto& p : subsets) {
        const auto& idxVec = p.first;
        SmallVector<Byte> bufferSizes;
        bufferSizes.reserve(idxVec.size());
        for (const auto& idx : idxVec) {
            bufferSizes.push_back(mergedVals[idx].getType().cast<vpux::NDTypeInterface>().getTotalAllocSize());
        }
        p.second = vpux::calculateAlignedBuffersMemoryRequirement(bufferSizes, defaultCMXOffsetAlignment,
                                                                  defaultCMXSizeAlignment)
                           .count() +
                   reservedMem.count();
    }

    std::sort(subsets.begin(), subsets.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });

    // Find the subset that uses the most NNCMX without going over the limit
    std::optional<size_t> subsetIdx;
    for (size_t idx = 0; idx < subsets.size(); ++idx) {
        if (subsets[idx].second > totalAvailableCMXSize) {
            if (idx > 0) {
                subsetIdx = idx - 1;
            }
            break;
        }
    }
    if (!subsetIdx.has_value()) {
        return {inputsForCMX, outputsForCMX};
    }

    const auto maxValidCmxUsageSubsetIdx = std::min(subsetIdx.value(), subsets.size() - 1);

    for (const auto& idx : subsets[maxValidCmxUsageSubsetIdx].first) {
        if (idx < inputs.size()) {
            inputsForCMX.insert(idx);
        } else {
            outputsForCMX.insert(idx - inputs.size());
        }
    }

    nestedLog2.trace("Following inputs and outputs will be mapped to CMX:");

    nestedLog2.trace("Inputs:");
    for (const auto& i : inputsForCMX) {
        nestedLog3.trace("'{0}'", i);
    }

    nestedLog2.trace("Outputs:");
    for (const auto& o : outputsForCMX) {
        nestedLog3.trace("'{0}'", o);
    }

    return {inputsForCMX, outputsForCMX};
}

//
// isDMAConvertibleSwOp
//

bool isDMAConvertibleSwOp(VPUIP::SoftwareLayerOpInterface swOp) {
    return mlir::isa<VPU::MemPermuteOp, VPU::SpaceToDepthOp, VPU::DepthToSpaceOp, VPU::PerAxisTileOp>(swOp);
}

}  // namespace

//
// bufferizeSWLayerOp
//

mlir::LogicalResult vpux::bufferizeSWLayerOp(mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op,
                                             ArrayRef<mlir::Value> newOperands, vpux::Logger log) {
    auto* ctx = op->getContext();
    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    SmallVector<mlir::Value> opResults(op->getResults().begin(), op->getResults().end());
    auto idxForCMX = getOptimalCMXPlacement(newOperands, opResults, Byte(0), module, log);

    SmallVector<mlir::Value> swKernelOperands;
    for (size_t i = 0; i < newOperands.size(); ++i) {
        if (idxForCMX.first.count(i) == 0) {
            // Operand should remain in DDR according to mapping
            swKernelOperands.push_back(newOperands[i]);
        } else {
            log.trace("Create CMX buffer and copy operation for input: {0}", newOperands[i].getLoc());
            const auto outputBuffer =
                    allocateBuffer(log, newOperands[i].getLoc(), rewriter, newOperands[i], memSpaceCMX);
            auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), newOperands[i], outputBuffer);
            swKernelOperands.push_back(copyOp.getOutput());
        }
    }

    SmallVector<mlir::Value> swKernelResults;
    for (size_t i = 0; i < op->getResults().size(); ++i) {
        if (idxForCMX.second.count(i) == 0) {
            log.trace("Create DDR buffer for output: {0}", op->getResults()[i].getLoc());
            const auto outputBuffer = allocateBuffer(log, op->getLoc(), rewriter, op->getResults()[i], nullptr);
            swKernelResults.push_back(outputBuffer);
        } else {
            log.trace("Create CMX buffer for output: {0}", op->getResults()[i].getLoc());
            const auto outputBuffer = allocateBuffer(log, op->getLoc(), rewriter, op->getResults()[i], memSpaceCMX);
            swKernelResults.push_back(outputBuffer);
        }
    }

    VPUIP::createRuntimeKernelDefinition(module, log.nest(), VPU::getArch(op));

    // TODO : tile 0
    const int64_t tileIndex = 0;
    auto genericSwLayerOp = mlir::dyn_cast<VPU::GenericSwLayerOp>(op);
    auto builtInFunction = genericSwLayerOp
                                   ? genericSwLayerOp.getCallee()
                                   : VPUIP::createBuiltInFunction(module, layerOp, swKernelOperands, swKernelResults,
                                                                  swLayerOp.getKernelInfo(), log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), swKernelOperands, swKernelResults,
                                                         builtInFunction, getIntAttr(ctx, tileIndex));

    vpux::VPUIP::initSwKernel(swKernelOp, swKernelOperands, swKernelResults, swLayerOp.getKernelInfo().args,
                              log.nest());

    const auto moveSwOpToCMX = [&]() {
        // Go through all inputs and outputs that were mapped to DDR and map them to NNCMX
        if (idxForCMX.first.size() == swKernelOp.getInputs().size() &&
            idxForCMX.second.size() == swKernelOp.getResults().size()) {
            return;
        }

        SmallVector<mlir::Value> cmxOperands;
        cmxOperands.reserve(swKernelOperands.size());
        if (idxForCMX.first.size() != swKernelOp.getInputs().size()) {
            for (const auto& operand : swKernelOperands) {
                if (operand.getType().cast<vpux::NDTypeInterface>().getMemSpace() == memSpaceCMX) {
                    cmxOperands.push_back(operand);
                } else {
                    const auto outputBuffer = allocateBuffer(log, operand.getLoc(), rewriter, operand, memSpaceCMX);
                    auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), operand, outputBuffer);
                    cmxOperands.push_back(copyOp.getOutput());
                }
            }
        } else {
            cmxOperands.append(swKernelOperands.begin(), swKernelOperands.end());
        }

        SmallVector<mlir::Value> cmxResults;
        cmxResults.reserve(swKernelResults.size());
        if (idxForCMX.second.size() != swKernelOp.getResults().size()) {
            for (const auto& result : swKernelResults) {
                cmxResults.push_back(result);
                if (result.getType().cast<vpux::NDTypeInterface>().getMemSpace() != memSpaceCMX) {
                    cmxResults.back().setType(
                            result.getType().cast<vpux::NDTypeInterface>().changeMemSpace(memSpaceCMX));
                }
            }
        } else {
            cmxResults.append(swKernelResults.begin(), swKernelResults.end());
        }

        auto parentModule = swKernelOp->getParentOfType<mlir::ModuleOp>();
        VPUX_THROW_UNLESS(parentModule, "Sw Kernel Op {0} has no parent Module Op", swKernelOp);
        auto kernelFunc = parentModule.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
        if (kernelFunc) {
            kernelFunc.erase();
        }

        rewriter.eraseOp(swKernelOp);

        if (genericSwLayerOp == nullptr) {
            builtInFunction = createBuiltInFunction(module, layerOp, cmxOperands, cmxResults, swLayerOp.getKernelInfo(),
                                                    log.nest());
        }

        swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), cmxOperands, cmxResults, builtInFunction,
                                                        getIntAttr(ctx, tileIndex));

        vpux::VPUIP::initSwKernel(swKernelOp, cmxOperands, cmxResults, swLayerOp.getKernelInfo().args, log.nest());
    };

    if (isDMAConvertibleSwOp(mlir::dyn_cast<vpux::VPUIP::SoftwareLayerOpInterface>(op)) &&
        vpux::VPUIP::isLegalAndBeneficialConvertToDMA(swKernelOp, log)) {
        log.trace("SW Kernel will be converted to DMA Operation: {0}", swKernelOp);
        moveSwOpToCMX();
    }

    log.trace("Added kernel operation: {0}", swKernelOp);

    SmallVector<mlir::Value> finalResults;
    for (auto&& result : swKernelOp.getResults()) {
        if (result.getType().cast<vpux::NDTypeInterface>().getMemSpace() == memSpaceCMX) {
            // Copy outputs that were mapped to CMX back to DDR
            log.trace("Create DDR buffer for output: {0}", result.getLoc());
            const auto outputBuffer = allocateBuffer(log, op->getLoc(), rewriter, result, nullptr);
            auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), result, outputBuffer);
            finalResults.push_back(copyOp.getOutput());
        } else {
            finalResults.push_back(result);
        }
    }

    log.trace("Replace origin op {0} with new outputs from SW Kernel {1}", op->getLoc(), finalResults);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, finalResults);
    return mlir::success();
}

//
// bufferizeSWLayerOpInNceClusterTiling
//

mlir::LogicalResult vpux::bufferizeSWLayerOpInNceClusterTiling(mlir::RewriterBase& rewriter, mlir::ModuleOp module,
                                                               mlir::Operation* op, ArrayRef<mlir::Value> newOperands,
                                                               vpux::Logger log) {
    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    VPUIP::createRuntimeKernelDefinition(module, log.nest(), VPU::getArch(op));

    auto outputBuffers = allocateBuffers(log, op->getLoc(), rewriter, op->getResults(),
                                         /*individualBuffers=*/true);
    // actual tile index will be corrected as part of unroll NCEClusterTiling pass, this index will be dropped
    const int64_t tileIndex = 0;
    auto genericSwLayerOp = mlir::dyn_cast<VPU::GenericSwLayerOp>(op);
    auto builtInFunction = genericSwLayerOp ? genericSwLayerOp.getCallee()
                                            : createBuiltInFunction(module, layerOp, newOperands, outputBuffers,
                                                                    swLayerOp.getKernelInfo(), log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), newOperands, outputBuffers, builtInFunction,
                                                         getIntAttr(op->getContext(), tileIndex));
    vpux::VPUIP::initSwKernel(swKernelOp, newOperands, outputBuffers, swLayerOp.getKernelInfo().args, log.nest());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, swKernelOp.getResults());
    return mlir::success();
}

namespace {

//
// ConcatOpBufferizeModel
//

class ConcatOpBufferizeModel : public BufferizableOpInterfaceExternalModelBase<ConcatOpBufferizeModel, VPU::ConcatOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::ConcatOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::ConcatOp::Adaptor adaptor) const;
};

bool isLegalConcatOp(VPU::ConcatOp concatOp) {
    const auto outputType = concatOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    return outputType.getShape().isStatic();
}

mlir::LogicalResult ConcatOpBufferizeModel::bufferizeImpl(VPU::ConcatOp origOp, mlir::RewriterBase& rewriter,
                                                          const mlir::bufferization::BufferizationOptions& options,
                                                          VPU::ConcatOp::Adaptor adaptor) const {
    if (isLegalConcatOp(origOp)) {
        return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
    }
    SoftwareLayerOpBufferizeModel<VPU::ConcatOp> concatOpSoftwareModel;
    return concatOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
}

//
// StridedSliceOpBufferizeModel
//

class StridedSliceOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<StridedSliceOpBufferizeModel, VPU::StridedSliceOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::StridedSliceOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::StridedSliceOp::Adaptor adaptor) const;
};

bool isLegalStridedSliceOp(VPU::StridedSliceOp stridedSliceOp) {
    if (IE::hasDynamicTensors(stridedSliceOp)) {
        return false;
    }

    auto attrToVector = [&](mlir::ArrayAttr attr) {
        if (attr == nullptr) {
            return SmallVector<int64_t>{};
        }
        return parseIntArrayAttr<int64_t>(attr);
    };

    const auto beginsVec = attrToVector(stridedSliceOp.getBeginsAttrAttr());
    const auto stridesVec = attrToVector(stridedSliceOp.getStridesAttrAttr());

    if (beginsVec.size() != stridesVec.size()) {
        return false;
    }

    if (beginsVec.empty() || stridesVec.empty()) {
        return false;
    }

    // This is an oversimplified way of computing the required striding level and does not account parameters that can
    // modify the striding level, such as ends, begins_mask, ends_mask, strides_mask etc.
    int64_t stridingLevel = 0;
    for (size_t index = 0; index < beginsVec.size(); ++index) {
        if (beginsVec[index] != 0 || stridesVec[index] > 1 || stridesVec[index] < -1) {
            ++stridingLevel;
        }
    }

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(VPU::getArch(stridedSliceOp));

    return stridingLevel <= dmaEngineLimits.getMaxStrideCount();
}

mlir::LogicalResult StridedSliceOpBufferizeModel::bufferizeImpl(
        VPU::StridedSliceOp origOp, mlir::RewriterBase& rewriter,
        const mlir::bufferization::BufferizationOptions& options, VPU::StridedSliceOp::Adaptor adaptor) const {
    if (isLegalStridedSliceOp(origOp)) {
        return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
    }

    SoftwareLayerOpBufferizeModel<VPU::StridedSliceOp> stridedSliceOpSoftwareModel;
    return stridedSliceOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
}

//
// PermuteCastOpBufferizeModel
//

class PermuteCastOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<PermuteCastOpBufferizeModel, VPU::PermuteCastOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::PermuteCastOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::PermuteCastOp::Adaptor adaptor) const;
};

bool isLegalPermuteCastOp(VPU::PermuteCastOp op) {
    const auto inputShape = getShape(op.getInput());
    const auto outputShape = getShape(op.getOutput());
    return inputShape.isStatic() && outputShape.isStatic();
}

mlir::LogicalResult PermuteCastOpBufferizeModel::bufferizeImpl(VPU::PermuteCastOp origOp, mlir::RewriterBase& rewriter,
                                                               const mlir::bufferization::BufferizationOptions& options,
                                                               VPU::PermuteCastOp::Adaptor adaptor) const {
    if (isLegalPermuteCastOp(origOp)) {
        return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
    }

    SoftwareLayerOpBufferizeModel<VPU::PermuteCastOp> permuteCastOpSoftwareModel;
    return permuteCastOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
}

}  // namespace

//
// registerSoftwareLayerBufferizableOpInterfaces
//

void vpux::registerSoftwareLayerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::StridedSliceOp::attachInterface<StridedSliceOpBufferizeModel>(*ctx);
        VPU::PermuteCastOp::attachInterface<PermuteCastOpBufferizeModel>(*ctx);
        VPU::SigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SigmoidOp>>(*ctx);
        VPU::HardSigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::HardSigmoidOp>>(*ctx);
        VPU::GridSampleOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GridSampleOp>>(*ctx);
        VPU::SoftMaxOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SoftMaxOp>>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogSoftmaxOp>>(*ctx);
        VPU::HSwishOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::HSwishOp>>(*ctx);
        VPU::MVNOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVNOp>>(*ctx);
        VPU::MVN1SumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN1SumOp>>(*ctx);
        VPU::MVN1MeanVarOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN1MeanVarOp>>(*ctx);
        VPU::MVN1NormalizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN1NormalizeOp>>(*ctx);
        VPU::MVN6Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MVN6Op>>(*ctx);
        VPU::InterpolateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::InterpolateOp>>(*ctx);
        VPU::ScatterNDUpdateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ScatterNDUpdateOp>>(*ctx);
        VPU::EluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EluOp>>(*ctx);
        VPU::SeluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SeluOp>>(*ctx);
        VPU::ClampOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ClampOp>>(*ctx);
        VPU::FullyConnectedOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FullyConnectedOp>>(*ctx);
        VPU::MatMulOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MatMulOp>>(*ctx);
        VPU::SqrtOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SqrtOp>>(*ctx);
        VPU::CeilingOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CeilingOp>>(*ctx);
        VPU::NormalizeL2Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NormalizeL2Op>>(*ctx);
        VPU::CumSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CumSumOp>>(*ctx);
        VPU::EyeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EyeOp>>(*ctx);
        VPU::DetectionOutputNormalizeOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputNormalizeOp>>(*ctx);
        VPU::DetectionOutputDecodeBoxesOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputDecodeBoxesOp>>(*ctx);
        VPU::DetectionOutputSortOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DetectionOutputSortOp>>(*ctx);
        VPU::DetectionOutputNmsCaffeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DetectionOutputNmsCaffeOp>>(
                *ctx);
        VPU::DetectionOutputCollectResultsOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::DetectionOutputCollectResultsOp>>(*ctx);
        VPU::DivideOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DivideOp>>(*ctx);
        VPU::MultiplyOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MultiplyOp>>(*ctx);
        VPU::AddOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AddOp>>(*ctx);
        VPU::SubtractOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SubtractOp>>(*ctx);
        VPU::PowerOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PowerOp>>(*ctx);
        VPU::MinimumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MinimumOp>>(*ctx);
        VPU::MaximumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MaximumOp>>(*ctx);
        VPU::ExpOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ExpOp>>(*ctx);
        VPU::RegionYoloOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RegionYoloOp>>(*ctx);
        VPU::GatherOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherOp>>(*ctx);
        VPU::GatherElementsOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherElementsOp>>(*ctx);
        VPU::GatherNDOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherNDOp>>(*ctx);
        VPU::GatherTreeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GatherTreeOp>>(*ctx);
        VPU::ConditionalCopyOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ConditionalCopyOp>>(*ctx);
        VPU::LoopSelectOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LoopSelectOp>>(*ctx);
        VPU::GroupNormalizationOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GroupNormalizationOp>>(*ctx);
        VPU::TanOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TanOp>>(*ctx);
        VPU::TanhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TanhOp>>(*ctx);
        VPU::SinOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SinOp>>(*ctx);
        VPU::CosOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CosOp>>(*ctx);
        VPU::SinhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SinhOp>>(*ctx);
        VPU::EmbeddingSegmentsSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EmbeddingSegmentsSumOp>>(*ctx);
        VPU::CoshOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CoshOp>>(*ctx);
        VPU::AsinOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AsinOp>>(*ctx);
        VPU::AcosOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AcosOp>>(*ctx);
        VPU::AtanOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AtanOp>>(*ctx);
        VPU::AsinhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AsinhOp>>(*ctx);
        VPU::AcoshOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AcoshOp>>(*ctx);
        VPU::AtanhOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AtanhOp>>(*ctx);
        VPU::TopKOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TopKOp>>(*ctx);
        VPU::LRNOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LRNOp>>(*ctx);
        VPU::MemPermuteOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MemPermuteOp>>(*ctx);
        VPU::PadOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PadOp>>(*ctx);
        VPU::DepthToSpaceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DepthToSpaceOp>>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SpaceToDepthOp>>(*ctx);
        VPU::SpaceToBatch::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SpaceToBatch>>(*ctx);
        VPU::BatchToSpace::attachInterface<SoftwareLayerOpBufferizeModel<VPU::BatchToSpace>>(*ctx);
        VPU::AvgPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AvgPoolOp>>(*ctx);
        VPU::AdaptiveAvgPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AdaptiveAvgPoolOp>>(*ctx);
        VPU::AdaptiveMaxPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AdaptiveMaxPoolOp>>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FakeQuantizeOp>>(*ctx);
        VPU::QuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::QuantizeOp>>(*ctx);
        VPU::DequantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DequantizeOp>>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicQuantizeOp>>(*ctx);
        VPU::DynamicDequantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicDequantizeOp>>(*ctx);
        VPU::PReluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PReluOp>>(*ctx);
        VPU::ExtractImagePatchesOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ExtractImagePatchesOp>>(*ctx);
        VPU::LeakyReluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LeakyReluOp>>(*ctx);
        VPU::MishOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MishOp>>(*ctx);
        VPU::TileOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::TileOp>>(*ctx);
        VPU::ReLUOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReLUOp>>(*ctx);
        VPU::YuvToRgbOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::YuvToRgbOp>>(*ctx);
        VPU::RandomUniformOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RandomUniformOp>>(*ctx);
        VPU::OneHotOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::OneHotOp>>(*ctx);
        VPU::ReorgYoloOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReorgYoloOp>>(*ctx);
        VPU::ProposalOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ProposalOp>>(*ctx);
        VPU::ScatterUpdateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ScatterUpdateOp>>(*ctx);
        VPU::ScatterElementsUpdateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ScatterElementsUpdateOp>>(
                *ctx);
        VPU::ReverseOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReverseOp>>(*ctx);
        VPU::ReverseSequenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReverseSequenceOp>>(*ctx);
        VPU::FloorModOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FloorModOp>>(*ctx);
        VPU::ModOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ModOp>>(*ctx);
        VPU::EqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EqualOp>>(*ctx);
        VPU::GreaterOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GreaterOp>>(*ctx);
        VPU::GreaterEqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GreaterEqualOp>>(*ctx);
        VPU::LessOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LessOp>>(*ctx);
        VPU::LessEqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LessEqualOp>>(*ctx);
        VPU::LogicalOrOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogicalOrOp>>(*ctx);
        VPU::BitwiseAndOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::BitwiseAndOp>>(*ctx);
        VPU::BitwiseOrOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::BitwiseOrOp>>(*ctx);
        VPU::BitwiseXorOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::BitwiseXorOp>>(*ctx);
        VPU::BitwiseNotOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::BitwiseNotOp>>(*ctx);
        VPU::HSigmoidOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::HSigmoidOp>>(*ctx);
        VPU::LogicalXorOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogicalXorOp>>(*ctx);
        VPU::LogicalNotOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogicalNotOp>>(*ctx);
        VPU::AndOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AndOp>>(*ctx);
        VPU::NotEqualOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NotEqualOp>>(*ctx);
        VPU::ReduceL1Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceL1Op>>(*ctx);
        VPU::ReduceSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceSumOp>>(*ctx);
        VPU::ReduceMeanOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMeanOp>>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceLogicalAndOp>>(*ctx);
        VPU::ReduceMaxOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMaxOp>>(*ctx);
        VPU::ReduceMinOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMinOp>>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceLogicalOrOp>>(*ctx);
        VPU::ReduceL2Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceL2Op>>(*ctx);
        VPU::ReduceProdOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceProdOp>>(*ctx);
        VPU::NegativeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NegativeOp>>(*ctx);
        VPU::NonMaxSuppressionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NonMaxSuppressionOp>>(*ctx);
        VPU::ROIAlignOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ROIAlignOp>>(*ctx);
        VPU::ROIPoolingOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ROIPoolingOp>>(*ctx);
        VPU::PSROIPoolingOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PSROIPoolingOp>>(*ctx);
        VPU::PermuteQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PermuteQuantizeOp>>(*ctx);
        VPU::LogOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LogOp>>(*ctx);
        VPU::FloorOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FloorOp>>(*ctx);
        VPU::RoundOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RoundOp>>(*ctx);
        VPU::SignOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SignOp>>(*ctx);
        VPU::SwishOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SwishOp>>(*ctx);
        VPU::SelectOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SelectOp>>(*ctx);
        VPU::EmbeddingBagOffsetsSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EmbeddingBagOffsetsSumOp>>(
                *ctx);
        VPU::GRUSequenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUSequenceOp>>(*ctx);
        VPU::EmbeddingBagPackedSumOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::EmbeddingBagPackedSumOp>>(
                *ctx);
        VPU::GRUSequenceFirstPartOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUSequenceFirstPartOp>>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUSequenceLastPartOp>>(*ctx);
        VPU::LSTMCellOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LSTMCellOp>>(*ctx);
        VPU::LSTMGatesOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LSTMGatesOp>>(*ctx);
        VPU::LSTMSequenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::LSTMSequenceOp>>(*ctx);
        VPU::ErfOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ErfOp>>(*ctx);
        VPU::BucketizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::BucketizeOp>>(*ctx);
        VPU::MaxPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MaxPoolOp>>(*ctx);
        VPU::MaxPool8Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::MaxPool8Op>>(*ctx);
        VPU::RollOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RollOp>>(*ctx);
        VPU::CTCGreedyDecoderSeqLenOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CTCGreedyDecoderSeqLenOp>>(
                *ctx);
        VPU::AbsOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AbsOp>>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SquaredDifferenceOp>>(*ctx);
        VPU::CTCGreedyDecoderOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::CTCGreedyDecoderOp>>(*ctx);
        VPU::GeluOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GeluOp>>(*ctx);
        VPU::SoftPlusOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SoftPlusOp>>(*ctx);
        VPU::ConvolutionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ConvolutionOp>>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GroupConvolutionOp>>(*ctx);
        VPU::DFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DFTOp>>(*ctx);
        VPU::RDFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RDFTOp>>(*ctx);
        VPU::IDFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::IDFTOp>>(*ctx);
        VPU::IRDFTOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::IRDFTOp>>(*ctx);
        VPU::RDFTUncutOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RDFTUncutOp>>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::IRDFTLastAxisOp>>(*ctx);
        VPU::ExperimentalDetectronROIFeatureExtractorOp::attachInterface<
                SoftwareLayerOpBufferizeModel<VPU::ExperimentalDetectronROIFeatureExtractorOp>>(*ctx);
        VPU::AccumulateOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AccumulateOp>>(*ctx);
        VPU::RangeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RangeOp>>(*ctx);
        VPU::NonZeroOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::NonZeroOp>>(*ctx);
        VPU::DynamicReshapeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicReshapeOp>>(*ctx);
        VPU::DynamicTileOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicTileOp>>(*ctx);
        VPU::ConcatOp::attachInterface<ConcatOpBufferizeModel>(*ctx);
        VPU::RMSOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RMSOp>>(*ctx);
        VPU::InverseOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::InverseOp>>(*ctx);
        VPU::DeformableConvolutionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DeformableConvolutionOp>>(
                *ctx);
        VPU::DynamicExpandOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicExpandOp>>(*ctx);
        VPU::PopulateWeightTableOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PopulateWeightTableOp>>(*ctx);
        VPU::GenericSwLayerOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GenericSwLayerOp>>(*ctx);
        VPU::RoPEOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RoPEOp>>(*ctx);
    });
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
}
