//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
using namespace vpux;

bool vpux::canBeBufferizedToCopies(VPU::ConcatOp concatOp) {
    auto outType = concatOp.getOutput().getType();
    return !mlir::isa<Core::DynamicDimsMaskTensorType>(outType);
}

bool vpux::canBeBufferizedToCopies(VPU::StridedSliceOp stridedSliceOp) {
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

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(stridedSliceOp));

    return stridingLevel <= dmaEngineLimits.getMaxStrideCount();
}

bool vpux::canBeBufferizedToCast(VPU::PermuteCastOp op) {
    const auto inputType = op.getInput().getType();
    const auto outputType = op.getOutput().getType();
    return !mlir::isa<Core::DynamicDimsMaskTensorType>(inputType) &&
           !mlir::isa<Core::DynamicDimsMaskTensorType>(outputType);
}

namespace {

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
    // NOTE: If you are implementing a new SW layer and getting a cast error here,
    // you need to attach SoftwareLayerOpModel to your operation
    // src/vpux_compiler/src/dialect/VPUIP/IR/ops.cpp
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    SmallVector<mlir::Value> swKernelResults;
    for (auto result : op->getResults()) {
        const auto memSpace = mlir::cast<NDTypeInterface>(result.getType()).getMemSpace();
        const auto outputBuffer = allocateBuffer(log, op->getLoc(), rewriter, result, memSpace);
        swKernelResults.push_back(outputBuffer);
    }

    VPUIP::createRuntimeKernelDefinition(module, log.nest(), config::getArch(op));

    // TODO : tile 0
    const int64_t tileIndex = 0;
    auto genericSwLayerOp = mlir::dyn_cast<VPU::GenericSwLayerOp>(op);
    auto builtInFunction = genericSwLayerOp
                                   ? genericSwLayerOp.getCallee()
                                   : VPUIP::createBuiltInFunction(module, layerOp, newOperands, swKernelResults,
                                                                  swLayerOp.getKernelInfo(), log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), newOperands, swKernelResults, builtInFunction,
                                                         getIntAttr(ctx, tileIndex));

    vpux::VPUIP::initSwKernel(swKernelOp, newOperands, swKernelResults, swLayerOp.getKernelInfo().args, log.nest(),
                              /*swKernelRunOp=*/nullptr);

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    const auto moveSwOpToCMX = [&]() {
        SmallVector<mlir::Value> cmxOperands;
        cmxOperands.reserve(newOperands.size());
        for (const auto& operand : newOperands) {
            if (mlir::cast<vpux::NDTypeInterface>(operand.getType()).getMemSpace() == memSpaceCMX) {
                cmxOperands.push_back(operand);
            } else {
                const auto outputBuffer = allocateBuffer(log, operand.getLoc(), rewriter, operand, memSpaceCMX);
                auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), operand, outputBuffer);
                cmxOperands.push_back(copyOp.getOutput());
            }
        }

        SmallVector<mlir::Value> cmxResults;
        cmxResults.reserve(swKernelResults.size());
        for (const auto& result : swKernelResults) {
            cmxResults.push_back(result);
            if (mlir::cast<vpux::NDTypeInterface>(result.getType()).getMemSpace() != memSpaceCMX) {
                cmxResults.back().setType(
                        mlir::cast<vpux::NDTypeInterface>(result.getType()).changeMemSpace(memSpaceCMX));
            }
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

        vpux::VPUIP::initSwKernel(swKernelOp, cmxOperands, cmxResults, swLayerOp.getKernelInfo().args, log.nest(),
                                  /*swKernelRunOp=*/nullptr);

        SmallVector<mlir::Value> newResults;
        for (auto&& result : swKernelOp.getResults()) {
            const auto origResultType = mlir::cast<NDTypeInterface>(op->getResult(result.getResultNumber()).getType());
            const auto newResultType = mlir::cast<NDTypeInterface>(result.getType());
            if (origResultType.getMemSpace() != memSpaceCMX && newResultType.getMemSpace() == memSpaceCMX &&
                !op->getResult(result.getResultNumber()).use_empty()) {
                // Copy outputs that were mapped to CMX back to DDR
                log.trace("Create DDR buffer for output: {0}", result.getLoc());
                const auto outputBuffer = allocateBuffer(log, op->getLoc(), rewriter, result, nullptr);
                auto copyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), result, outputBuffer);
                newResults.push_back(copyOp.getOutput());
            } else {
                newResults.push_back(result);
            }
        }
        return newResults;
    };

    auto finalResults = SmallVector<mlir::Value>(swKernelOp->getResults());
    if (isDMAConvertibleSwOp(mlir::dyn_cast<vpux::VPUIP::SoftwareLayerOpInterface>(op)) &&
        vpux::VPUIP::isLegalAndBeneficialConvertToDMA(swKernelOp, log)) {
        log.trace("SW Kernel will be converted to DMA Operation: {0}", swKernelOp);
        finalResults = moveSwOpToCMX();
    }

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, finalResults);
    return mlir::success();
}

//
// bufferizeDistributedSWLayerOp
//

mlir::LogicalResult vpux::bufferizeDistributedSWLayerOp(mlir::RewriterBase& rewriter, mlir::ModuleOp module,
                                                        mlir::Operation* op, ArrayRef<mlir::Value> newOperands,
                                                        vpux::Logger log) {
    auto layerOp = mlir::cast<VPU::LayerOpInterface>(op);
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(op);

    VPUIP::createRuntimeKernelDefinition(module, log.nest(), config::getArch(op));

    auto outputBuffers = allocateBuffers(log, op->getLoc(), rewriter, op->getResults(),
                                         /*individualBuffers=*/true);
    // The actual tile index will be corrected as part of UnrollDistributedOpsPass; this index will be dropped
    const int64_t tileIndex = 0;
    auto genericSwLayerOp = mlir::dyn_cast<VPU::GenericSwLayerOp>(op);
    auto builtInFunction = genericSwLayerOp ? genericSwLayerOp.getCallee()
                                            : createBuiltInFunction(module, layerOp, newOperands, outputBuffers,
                                                                    swLayerOp.getKernelInfo(), log.nest());

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(op->getLoc(), newOperands, outputBuffers, builtInFunction,
                                                         getIntAttr(op->getContext(), tileIndex));
    vpux::VPUIP::initSwKernel(swKernelOp, newOperands, outputBuffers, swLayerOp.getKernelInfo().args, log.nest(),
                              /*swKernelRunOp=*/nullptr);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, swKernelOp.getResults());
    return mlir::success();
}

namespace {

class ConcatOpBufferizeModel : public BufferizableOpInterfaceExternalModelBase<ConcatOpBufferizeModel, VPU::ConcatOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::ConcatOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::ConcatOp::Adaptor& adaptor) const {
        if (canBeBufferizedToCopies(origOp)) {
            return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
        }
        SoftwareLayerOpBufferizeModel<VPU::ConcatOp> concatOpSoftwareModel;
        return concatOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
    }
};

class StridedSliceOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<StridedSliceOpBufferizeModel, VPU::StridedSliceOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::StridedSliceOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::StridedSliceOp::Adaptor& adaptor) const {
        if (canBeBufferizedToCopies(origOp)) {
            return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
        }

        SoftwareLayerOpBufferizeModel<VPU::StridedSliceOp> stridedSliceOpSoftwareModel;
        return stridedSliceOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
    }
};

class PermuteCastOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<PermuteCastOpBufferizeModel, VPU::PermuteCastOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::PermuteCastOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::PermuteCastOp::Adaptor& adaptor) const {
        if (canBeBufferizedToCast(origOp)) {
            return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
        }

        SoftwareLayerOpBufferizeModel<VPU::PermuteCastOp> permuteCastOpSoftwareModel;
        return permuteCastOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
    }
};

class ConvertOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<ConvertOpBufferizeModel, VPU::ConvertOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::ConvertOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::ConvertOp::Adaptor& adaptor) const {
        // If conversion can be done on DMA, bufferize it to DMA operation.
        if (isConvertSupportedOnDMA<VPU::ConvertOp>(origOp)) {
            return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
        }

        // If ConvertOp can not be converted to DMA operation, bufferize it to software layer operation instead.
        SoftwareLayerOpBufferizeModel<VPU::ConvertOp> convertOpSoftwareModel;
        return convertOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
    }
};

class FlashSDPAModel : public BufferizableOpInterfaceExternalModelBase<FlashSDPAModel, VPU::FlashSDPAOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::FlashSDPAOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::FlashSDPAOp::Adaptor& adaptor) const;
};

mlir::LogicalResult FlashSDPAModel::bufferizeImpl(VPU::FlashSDPAOp flashSdpaOp, mlir::RewriterBase& rewriter,
                                                  const mlir::bufferization::BufferizationOptions&,
                                                  VPU::FlashSDPAOp::Adaptor&) const {
    auto log = Logger::global().nest("one-shot-bufferize-FlashSDPAOp", 0);
    log.trace("Got {0} at {1}", flashSdpaOp->getName(), flashSdpaOp->getLoc());

    auto bufferizedOperands = vpux::bufferizeOperands(rewriter, flashSdpaOp->getOperands());

    auto module = flashSdpaOp->getParentOfType<mlir::ModuleOp>();
    if (module == nullptr) {
        return errorAt(flashSdpaOp->getLoc(), "Operation {0} has no parent Module Op", flashSdpaOp->getName());
    }
    VPUIP::createRuntimeKernelDefinition(module, log.nest(), config::getArch(flashSdpaOp));

    // The last output is aliased with the first input.
    // It will re-use the input's buffer, instead of allocating a separate one
    auto tensorsToBufferize = flashSdpaOp->getResults().drop_back();
    auto outputBuffers = allocateBuffers(log, flashSdpaOp->getLoc(), rewriter, tensorsToBufferize,
                                         /*individualBuffers=*/true);
    auto queryBuffer = bufferizedOperands.front();
    outputBuffers.push_back(queryBuffer);

    auto layerOp = mlir::cast<VPU::LayerOpInterface>(flashSdpaOp.getOperation());
    auto swLayerOp = mlir::cast<VPUIP::SoftwareLayerOpInterface>(flashSdpaOp.getOperation());
    auto builtInFunction = createBuiltInFunction(module, layerOp, bufferizedOperands, outputBuffers,
                                                 swLayerOp.getKernelInfo(), log.nest());

    auto tileIndexAttr = getIntAttr(flashSdpaOp->getContext(), 0);
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(flashSdpaOp->getLoc(), bufferizedOperands, outputBuffers,
                                                         builtInFunction, tileIndexAttr);

    vpux::VPUIP::initSwKernel(swKernelOp, bufferizedOperands, outputBuffers, swLayerOp.getKernelInfo().args, log.nest(),
                              /*swKernelRunOp=*/nullptr);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, flashSdpaOp, swKernelOp.getResults());
    return mlir::success();
}

}  // namespace

//
// registerSoftwareLayerBufferizableOpInterfaces
//

void vpux::registerSoftwareLayerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::ConcatOp::attachInterface<ConcatOpBufferizeModel>(*ctx);
        VPU::StridedSliceOp::attachInterface<StridedSliceOpBufferizeModel>(*ctx);
        VPU::PermuteCastOp::attachInterface<PermuteCastOpBufferizeModel>(*ctx);
        VPU::ConvertOp::attachInterface<ConvertOpBufferizeModel>(*ctx);
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
        VPU::AvgPool16Op::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AvgPool16Op>>(*ctx);
        VPU::AdaptiveAvgPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AdaptiveAvgPoolOp>>(*ctx);
        VPU::AdaptiveMaxPoolOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::AdaptiveMaxPoolOp>>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FakeQuantizeOp>>(*ctx);
        VPU::FakeConvertOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::FakeConvertOp>>(*ctx);
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
        VPU::ReduceMeanSquareOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ReduceMeanSquareOp>>(*ctx);
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
        VPU::GRUGatesOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GRUGatesOp>>(*ctx);
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
        VPU::RMSOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RMSOp>>(*ctx);
        VPU::InverseOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::InverseOp>>(*ctx);
        VPU::DeformableConvolutionOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DeformableConvolutionOp>>(
                *ctx);
        VPU::DynamicExpandOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicExpandOp>>(*ctx);
        VPU::PopulateWeightTableOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::PopulateWeightTableOp>>(*ctx);
        VPU::GenericSwLayerOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::GenericSwLayerOp>>(*ctx);
        VPU::ExternalKernelOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ExternalKernelOp>>(*ctx);
        VPU::RoPEOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::RoPEOp>>(*ctx);
        VPU::SDPAOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SDPAOp>>(*ctx);
        VPU::SDPAExtendedOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::SDPAExtendedOp>>(*ctx);
        VPU::DynamicDataMaskOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::DynamicDataMaskOp>>(*ctx);
        VPU::FlashSDPAOp::attachInterface<FlashSDPAModel>(*ctx);
    });
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
}
