//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/utils/mpe_engine_info_utils.hpp"
#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/interfaces/multi_consumer_strategy.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/IR/BuiltinTypes.h>

namespace vpux::IE::arch60xx {

namespace {

template <typename TargetOp>
mlir::Operation* findProducingOpThroughViewLikes(mlir::Operation* definingOp) {
    auto* current = definingOp;
    while (current != nullptr && !mlir::isa<TargetOp>(current)) {
        if (!IE::isElemTypeTransparentViewLikeOp(current)) {
            return definingOp;
        }
        current = current->getOperand(0).getDefiningOp();
    }
    return current != nullptr ? current : definingOp;
}

}  // namespace

mlir::Operation* findProducingFakeQuantizeThroughViewLikes(mlir::Operation* definingOp) {
    return findProducingOpThroughViewLikes<IE::FakeQuantizeOp>(definingOp);
}

mlir::Operation* findProducingDynamicDequantizeThroughViewLikes(mlir::Operation* definingOp) {
    return findProducingOpThroughViewLikes<IE::DynamicDequantizeOp>(definingOp);
}

namespace {

bool meetsConvolutionDCIMRequirements(IE::ConvolutionOp convOp) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(convOp.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(convOp.getFilter().getType());
    if (inputType.getRank() != 4 || filterType.getRank() != 4) {
        return false;
    }

    Logger log = Logger::global();
    log.setName("mpe-engine");
    const auto W = inputType.getShape()[Dims4D::Act::W];
    const auto padsEnd = parseIntArrayAttr<int64_t>(convOp.getPadsEnd());
    const int64_t padRight = padsEnd.back();

    const auto definingOp = findProducingFakeQuantizeThroughViewLikes(convOp.getInput().getDefiningOp());

    // The layout transformation pass is subsequent to the pass that splits FakeQuantizeOps into separate Quantize -
    // Dequantize layers
    if (!mlir::isa_and_present<IE::FakeQuantizeOp>(definingOp)) {
        // Only NHWC input layout is supported
        if (inputType.getDimsOrder() != DimsOrder::NHWC) {
            return false;
        }
    }

    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(definingOp)) {
        if (config::getArch(convOp) == config::ArchKind::NPU60XX &&
            config::getRevisionID(convOp) >= config::RevisionID::REVISION_A0 &&
            config::getRevisionID(convOp) < config::RevisionID::REVISION_B) {
            const auto qType = IE::getQuantizedTypeFromFakeQuantize(fqOp);
            const bool hasInvalid16BitWithZeroPoint = qType != nullptr && qType.getStorageTypeIntegralWidth() == 16 &&
                                                      !IE::areAllQuantTypeZeroPointsEqualToZero(qType);
            if (hasInvalid16BitWithZeroPoint) {
                return false;
            }
        }
    }

    // Workaround for Erratum 18039532808:
    if (W + padRight > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        log.trace("Unsupported for NPU60XX! Erratum 18039532808: W ({0}) + pad_right ({1}) must be <= {2}", W, padRight,
                  VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
        return false;
    }

    return true;
}

}  // namespace

bool isEligibleForDCIM(mlir::Operation* op, bool checkChannelAlignment) {
    if (!mlir::isa<IE::ConvolutionOp, IE::MatMulOp, IE::FullyConnectedOp>(op) || !config::isDCIMEnabled(op)) {
        return false;
    }
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        if (!meetsConvolutionDCIMRequirements(convOp)) {
            return false;
        }
    }

    const auto info = getDcimMappingInfo(op);

    return passesSWHeuristicForDCIM(info, checkChannelAlignment) && respectHWConstraintsForDCIM(info);
}

mlir::Attribute makeMPEEngineModeAttr(mlir::Operation* op, bool useDCIM, mlir::Attribute weightZp,
                                      mlir::Attribute activationZp) {
    const auto mode = useDCIM ? VPU::MPEEngine60XXMode::DCIM : VPU::MPEEngine60XXMode::SCL;
    return VPU::MPEEngine60XXAttr::get(op->getContext(), VPU::MPEEngine60XXModeAttr::get(op->getContext(), mode),
                                       weightZp, activationZp);
}

mlir::Attribute getConvMPEEngineMode(mlir::Operation* op, bool dcimEligible, mlir::Attribute weightZp,
                                     mlir::Attribute activationZp) {
    auto convOp = mlir::cast<IE::ConvolutionOp>(op);
    Logger log = Logger::global();
    log.setName("mpe-engine");
    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    const bool useDCIM = dcimEligible && !VPU::NCECompressConvolutionOp::isSupported(convOp, logCb,
                                                                                     /*checkLayout=*/true,
                                                                                     /*checkChannelAlignment=*/true);
    return makeMPEEngineModeAttr(op, useDCIM, weightZp, activationZp);
}

mlir::Type getDcimIntStorageType(mlir::Type elemType) {
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        if (auto storageType = mlir::dyn_cast<mlir::IntegerType>(qType.getStorageType())) {
            const auto sign = qType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
            return mlir::IntegerType::get(elemType.getContext(), storageType.getWidth(), sign);
        }
    } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        if (intType.isSigned() || intType.isUnsigned()) {
            return intType;
        }
    }
    return nullptr;
}

void setDcimFakeQuantizeFacts(DcimMappingInfo& info) {
    if (info.inputFakeQuantizeOp != nullptr) {
        const auto levelsOrLowFpType = info.inputFakeQuantizeOp.getLevelsOrLowFpTypeAttr();
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(levelsOrLowFpType)) {
            info.inputFqLevels = intAttr.getInt();
        }
        info.inputFqIsPerTensor = IE::isPerTensorFQ({info.inputFakeQuantizeOp});
    }
}

// If definingOp dequantizes its input (Dequantize or DynamicDequantize), reports the pre-dequant raw type and
// returns true.
static bool setRawTypeIfDequantizeLike(mlir::Operation* definingOp, vpux::NDTypeInterface& operandType) {
    if (!mlir::isa_and_present<IE::DequantizeOp, IE::DynamicDequantizeOp>(definingOp)) {
        return false;
    }
    operandType = mlir::cast<vpux::NDTypeInterface>(definingOp->getOperand(0).getType());
    return true;
}

void setDcimInputProducerInfo(mlir::Value operand, vpux::NDTypeInterface& operandType, bool& hasQuantizeProducer,
                              IE::FakeQuantizeOp& fakeQuantizeOp) {
    auto definingOp = operand.getDefiningOp();
    if (setRawTypeIfDequantizeLike(definingOp, operandType)) {
        hasQuantizeProducer = true;
    } else {
        if (!mlir::isa_and_present<IE::FakeQuantizeOp>(definingOp)) {
            definingOp = findProducingFakeQuantizeThroughViewLikes(definingOp);
        }
        fakeQuantizeOp = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(definingOp);
        hasQuantizeProducer = fakeQuantizeOp != nullptr;
    }
}

void setDcimFilterProducerInfo(mlir::Value operand, vpux::NDTypeInterface& operandType, bool& hasQuantizeProducer,
                               bool& dynamicDequantizeScaleIsPerTensor) {
    auto definingOp = findProducingDynamicDequantizeThroughViewLikes(operand.getDefiningOp());
    if (!setRawTypeIfDequantizeLike(definingOp, operandType)) {
        return;
    }
    auto dynDeqOp = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(definingOp);
    if (dynDeqOp == nullptr) {
        return;
    }
    hasQuantizeProducer = true;
    const auto scaleShape = getShape(dynDeqOp.getScale());
    dynamicDequantizeScaleIsPerTensor = llvm::none_of(scaleShape, [](auto dim) {
        return dim > 1;
    });
}

void setDcimFilterWaiInfo(mlir::Operation* nceOp, DcimMappingInfo& info) {
    mlir::quant::QuantizedType waiQuantType = nullptr;
    auto weights = IE::findNCEOpWeightsAsInputOrConst(nceOp, &waiQuantType);
    if (mlir::failed(weights)) {
        return;
    }
    info.filterIsWeightsAsInput = mlir::isa<mlir::BlockArgument>(weights.value());
    info.filterWaiQuantType = waiQuantType;
}

DcimMappingInfo getDcimMappingInfo(mlir::Operation* op) {
    DcimMappingInfo info;
    if (mlir::isa<IE::MatMulOp>(op)) {
        info.opKind = DcimOpKind::MatMul;
        info.transposeA = mlir::cast<IE::MatMulOp>(op).getTransposeA();
    } else if (mlir::isa<IE::FullyConnectedOp>(op)) {
        info.opKind = DcimOpKind::FullyConnected;
    } else if (mlir::isa<IE::ConvolutionOp>(op)) {
        info.opKind = DcimOpKind::Convolution;
    } else {
        VPUX_THROW("Unsupported operation '{0}' for DCIM mapping info", op->getName());
    }

    info.inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
    info.filterType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType());
    info.outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

    setDcimInputProducerInfo(op->getOperand(0), info.inputType, info.inputHasQuantizeProducer,
                             info.inputFakeQuantizeOp);
    setDcimFilterProducerInfo(op->getOperand(1), info.filterType, info.filterHasQuantizeProducer,
                              info.filterDynDeqScaleIsPerTensor);
    setDcimFakeQuantizeFacts(info);
    setDcimFilterWaiInfo(op, info);

    info.dcimEnabled = config::isDCIMEnabled(op);
    info.powerOptimized = config::isPowerOptimized(op);
    return info;
}

DcimMappingInfo getProspectiveConvDcimMappingInfo(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                                  vpux::NDTypeInterface outputType, bool dcimEnabled,
                                                  bool powerOptimized, bool inputHasQuantizeProducer,
                                                  bool filterHasQuantizeProducer) {
    DcimMappingInfo info;
    info.opKind = DcimOpKind::Convolution;
    info.inputType = inputType;
    info.filterType = filterType;
    info.outputType = outputType;
    info.inputHasQuantizeProducer = inputHasQuantizeProducer;
    info.filterHasQuantizeProducer = filterHasQuantizeProducer;
    info.dcimEnabled = dcimEnabled;
    info.powerOptimized = powerOptimized;
    return info;
}

int32_t getInputChannelCount(const DcimMappingInfo& info) {
    if (info.opKind == DcimOpKind::MatMul || info.opKind == DcimOpKind::FullyConnected) {
        const auto inputDims = info.inputType.getShape().toValues();
        const auto inputRank = info.inputType.getRank();
        return inputDims[info.transposeA ? Dim(inputRank - 2) : Dim(inputRank - 1)];
    }

    return info.inputType.getShape()[Dims4D::Act::C];
}

int32_t getOutputChannelCount(const DcimMappingInfo& info) {
    if (info.opKind == DcimOpKind::MatMul || info.opKind == DcimOpKind::FullyConnected) {
        const auto outDims = info.outputType.getShape().toValues();
        const auto outRank = info.outputType.getRank();
        return outDims[Dim(outRank - 1)];
    }

    return info.outputType.getShape()[Dims4D::Act::C];
}

int32_t getInputHeight(const DcimMappingInfo& info) {
    if (info.opKind == DcimOpKind::MatMul || info.opKind == DcimOpKind::FullyConnected) {
        const auto outDims = info.outputType.getShape().toValues();
        const auto outRank = info.outputType.getRank();
        return outDims[Dim(outRank - 2)];
    }

    return info.inputType.getShape()[Dims4D::Act::H];
}

int32_t getInputWidth(const DcimMappingInfo& info) {
    if (info.opKind == DcimOpKind::MatMul || info.opKind == DcimOpKind::FullyConnected) {
        return 1;
    }

    return info.inputType.getShape()[Dims4D::Act::W];
}

bool isRegressedFullyConnectedDCIMShape(int32_t IC, int32_t OC, int32_t IHW) {
    // Temporary workaround for regressed fully connected DCIM shapes.
    // Shapes empirically found to regress end-to-end latency under DCIM due to unmodeled DMA/schedule overhead.
    if (IC == 768 && OC == 768 && IHW == 64) {
        return true;
    }
    if (IHW >= 128 && ((IC == 512 && OC >= 688 && IHW >= 128) || (IC == 512 && OC <= 128 && IHW >= 512) ||
                       (IC == 256 && OC >= 1024 && IHW >= 256))) {
        return true;
    }
    if (IHW <= 196 && ((IC == 480 && OC == 112) || (IC == 672 && OC == 112) || (IC == 112 && OC == 672))) {
        return true;
    }
    if (IC == 1536 && OC == 384) {
        return true;
    }
    return false;
}

bool passesSWHeuristicForDCIM(const DcimMappingInfo& info, bool checkChannelAlignment) {
    const auto IC = getInputChannelCount(info);
    const auto OC = getOutputChannelCount(info);
    const auto IH = getInputHeight(info);
    const auto IW = getInputWidth(info);
    const auto IHW = IH * IW;

    if (IHW >= 6400) {
        return false;
    }
    // From testing there was an observation that fullyConnected layers benefit from a higher channel alignment
    // requirement.
    const auto channelAlignment = info.opKind == DcimOpKind::FullyConnected ? 32 : 16;
    if (checkChannelAlignment && (IC % channelAlignment != 0 || OC % channelAlignment != 0)) {
        return false;
    }

    const auto isMatrixLike = info.opKind == DcimOpKind::MatMul || info.opKind == DcimOpKind::FullyConnected;
    const auto filterShape = info.filterType.getShape();
    const auto isOneByOne = !isMatrixLike && filterShape.size() >= 4 && filterShape[Dims4D::Filter::KY] == 1 &&
                            filterShape[Dims4D::Filter::KX] == 1;
    if ((isMatrixLike || isOneByOne) && isRegressedFullyConnectedDCIMShape(IC, OC, IHW)) {
        return false;
    }
    if (OC > 32 && IHW >= 32) {
        if (IC <= 64 && IH >= 4 && IW >= 4) {
            return false;
        }
    } else if (OC <= 32 && IHW >= 32) {
        if (IC < 128 && IH >= 4 && IW >= 4) {
            return false;
        }
    } else if (OC <= 32 && IHW < 32) {
        return false;
    } else { /* if (OC > 32 && IHW < 32) { */
        if (IHW <= 4) {
            return false;
        }
    }

    return true;
}

bool respectHWConstraintsForDCIM(const DcimMappingInfo& info) {
    Logger log = Logger::global();

    const auto IC = getInputChannelCount(info);
    // input channels must be >= 16
    if (IC < 16) {
        log.trace("Expected at least 16 input channels, but got {0}", IC);
        return false;
    }
    const auto inElem = info.inputType.getElementType();
    const auto wtElem = info.filterType.getElementType();

    if (mlir::isa<mlir::quant::QuantileType>(wtElem)) {
        return false;
    }

    if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(wtElem)) {
        auto quantized = mlir::cast<mlir::quant::QuantizedType>(wtElem);
        if (mlir::isa<mlir::quant::QuantileType>(quantized.getStorageType())) {
            return false;
        }
    }

    // only per-workload weight zeropoints are supported
    if (const auto perAxis = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(wtElem)) {
        if (!vpux::areAllZeroPointsEqual(perAxis)) {
            return false;
        }
    } else if (!info.filterHasQuantizeProducer && !mlir::isa<mlir::quant::UniformQuantizedType>(wtElem) &&
               info.filterWaiQuantType == nullptr) {
        return false;
    }
    if (const auto waiPerAxis =
                mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(info.filterWaiQuantType)) {
        if (!vpux::areAllZeroPointsEqual(waiPerAxis)) {
            return false;
        }
    }

    // dCIM valid quantize patterns: I16xI8, U16xU8, I8xI8, U8xU8. There is no support for cross data types (e.g I16 x
    // U8). The activation / weights storage types are exposed either directly (quantized-quantized) or through a
    // Dequantize / DynamicDequantize producer (DQ - DQ, DQ - DDQ)
    const auto actStorageType = getDcimIntStorageType(inElem);
    auto wtStorageType = getDcimIntStorageType(wtElem);
    if (wtStorageType == nullptr && info.filterWaiQuantType != nullptr) {
        wtStorageType = getDcimIntStorageType(info.filterWaiQuantType);
    }
    if (actStorageType != nullptr && wtStorageType != nullptr) {
        return IE::checkQuantizePattern(actStorageType, wtStorageType, {{16, 8}, {8, 8}});
    }

    // FakeQuantize - DynamicDequantize
    if (info.inputFqIsPerTensor && info.inputFakeQuantizeOp != nullptr && wtStorageType != nullptr &&
        info.filterHasQuantizeProducer) {
        const auto actBits = (info.inputFqLevels > 256) ? 16 : 8;
        const auto wtBits = mlir::cast<mlir::IntegerType>(wtStorageType).getWidth();
        if ((actBits == 16 || actBits == 8) && wtBits == 8) {
            if (info.filterIsWeightsAsInput && mlir::cast<mlir::IntegerType>(wtStorageType).isSignedInteger() &&
                (!IE::canActivationMatchSignedWaiWeight(info.inputFakeQuantizeOp, log) ||
                 !IE::canFakeQuantizeFoldUpstream(info.inputFakeQuantizeOp, false, log))) {
                return false;
            }
            return true;
        }
    }
    return false;
}

bool dcimChannelAlignmentSatisfied(mlir::Operation* op, const DcimMappingInfo& info) {
    auto alignedChannelOpInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
    if (alignedChannelOpInterface == nullptr) {
        return false;
    }
    const auto inputChannelAlignment = alignedChannelOpInterface.getInputChannelAlignment();
    const auto outputChannelAlignment = alignedChannelOpInterface.getOutputChannelAlignment();

    const auto inputChannelPad = alignValUp(static_cast<int64_t>(getInputChannelCount(info)), inputChannelAlignment);
    const auto outputChannelPad = alignValUp(static_cast<int64_t>(getOutputChannelCount(info)), outputChannelAlignment);
    return inputChannelPad % 32 == 0 && outputChannelPad % 32 == 0;
}

bool isMappableToDCIM(mlir::Operation* op, bool checkChannelAlignment) {
    if (!isEligibleForDCIM(op, checkChannelAlignment)) {
        return false;
    }
    if (checkChannelAlignment) {
        return true;
    }
    if (mlir::isa<IE::FullyConnectedOp>(op)) {
        return true;
    }
    return dcimChannelAlignmentSatisfied(op, getDcimMappingInfo(op));
}

bool isMappedToDCIM(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }

    const auto maybeMpeEngine = nceOp.getMpeEngine();
    if (!maybeMpeEngine) {
        return false;
    }

    const auto mpeEngine = *maybeMpeEngine;
    if (mpeEngine == nullptr) {
        return false;
    }

    auto mpeEngineModeAttr = mlir::dyn_cast<VPU::MPEEngine60XXAttr>(mpeEngine);
    if (mpeEngineModeAttr == nullptr) {
        return false;
    }

    return mpeEngineModeAttr.getMode().getValue() == VPU::MPEEngine60XXMode::DCIM;
}

bool passesSWHeuristicForDCIM(mlir::Operation* rootOp, bool checkChannelAlignment) {
    return passesSWHeuristicForDCIM(getDcimMappingInfo(rootOp), checkChannelAlignment);
}

bool respectHWConstraintsForDCIM(mlir::Operation* rootOp) {
    return respectHWConstraintsForDCIM(getDcimMappingInfo(rootOp));
}

}  // namespace vpux::IE::arch60xx
