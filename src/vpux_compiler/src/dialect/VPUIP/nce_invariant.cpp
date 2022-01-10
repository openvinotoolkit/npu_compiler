//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// verifyConvChannels
//

int64_t vpux::VPUIP::NCEInvariant::getChannelAlignment(mlir::Type elemType) {
    const Bit typeSizeInBits = getElemTypeSize(elemType);
    return std::max<int64_t>(128 / typeSizeInBits.count(), 16);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvChannels(mlir::Location loc, mlir::ShapedType filterType,
                                                                  Logger log) {
    log.setName("NCEInvariant");

    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", loc, filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = getShape(filterType);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];

    if (OC % getChannelAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Convolution output channels are not aligned", loc);
        return mlir::failure();
    }
    if (IC % getChannelAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Convolution input channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::ConvolutionOp origOp, Logger log) {
    return verifyConvChannels(origOp->getLoc(), origOp.filter().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::ConvolutionOp origOp, Logger log) {
    return verifyConvChannels(origOp->getLoc(), origOp.filter().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyPoolChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                                  Logger log) {
    log.setName("NCEInvariant");

    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    const auto inputShape = getShape(inputType);
    const auto IC = inputShape[Dims4D::Act::C];

    if (IC % getChannelAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Pooling channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyEltwiseChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwiseChannels(mlir::Location loc,
                                                                     mlir::ShapedType firstInputType,
                                                                     mlir::ShapedType secondInputType, Logger log) {
    log.setName("NCEInvariant");
    if (firstInputType.getRank() != 4) {
        log.trace("[{0}] Eltwise input1 shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    if (secondInputType.getRank() != 4) {
        log.trace("[{0}] Eltwise input2 shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    const auto firstInputShape = getShape(firstInputType);
    const auto secondInputShape = getShape(secondInputType);
    const auto firstIC = firstInputShape[Dims4D::Act::C];
    const auto secondIC = secondInputShape[Dims4D::Act::C];

    if (firstIC % getChannelAlignment(firstInputType.getElementType()) != 0) {
        log.trace("[{0}] Eltwise input1 channels are not aligned", loc);
        return mlir::failure();
    }

    if (secondIC % getChannelAlignment(secondInputType.getElementType()) != 0) {
        log.trace("[{0}] Eltwise input2 channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AddOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::AddOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MultiplyOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::MultiplyOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::SubtractOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::SubtractOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AndOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::AndOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

//
// verifyGroupConvChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyGroupConvChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                                       mlir::ShapedType filterType, Logger log) {
    log.setName("NCEInvariant");

    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", loc, filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = getShape(filterType);
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", loc);
        return mlir::failure();
    }

    const auto inputShape = getShape(inputType);
    const auto inputChan = inputShape[Dims4D::Act::C];
    const auto OC = filterShape[Dims4D::Filter::OC];
    if (OC != inputChan) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", loc, OC, inputChan);
        return mlir::failure();
    }

    if (OC % getChannelAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Group Convolution output channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::GroupConvolutionOp origOp, Logger log) {
    return verifyGroupConvChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::ShapedType>(),
                                   origOp.filter().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::GroupConvolutionOp origOp, Logger log) {
    return verifyGroupConvChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::ShapedType>(),
                                   origOp.filter().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyConvCMX
//

namespace {

Byte getCMXSizeForTiling(mlir::ModuleOp module) {
    auto cmxRes = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    VPUX_THROW_UNLESS(cmxRes != nullptr, "Can't get information about {0} memory", VPU::MemoryKind::CMX_NN);

    // This function is used to determine the best tile size. It tries to put maximum data in CMX.
    // Available CMX memory is decreased by two profilingBufferSize even if profiling is disabled
    // because we want to get exactly same compiled networks with profiling enabled and disabled.
    // Two buffer sizes are required in case when profiling allocates new buffer and old buffer
    // is still not disposed. Second buffer can be treated as an optimisation that prevents spilling.
    const int64_t profilingBufferSize =
            vpux::VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE + vpux::VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE;
    return cmxRes.size() - Byte(2 * profilingBufferSize);
}

Byte getRequiredCMXForTiling(ArrayRef<mlir::ShapedType> operands, int64_t numChannels) {
    Byte requiredCMX(0);

    for (const auto& operand : operands) {
        requiredCMX += getTotalSize(operand);
    }

    requiredCMX += numChannels * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             mlir::ShapedType inputType, mlir::ShapedType filterType,
                                                             mlir::ShapedType outputType, Logger log) {
    log.setName("NCEInvariant");

    const auto filterShape = getShape(filterType);
    // consider alignment when calculating required CMX
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto requiredCMX = getRequiredCMXForTiling({inputType, filterType, outputType}, OC);

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Convolution, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::ConvolutionOp origOp, Logger log) {
    return verifyConvCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                         origOp.input().getType().cast<mlir::ShapedType>(),
                         origOp.filter().getType().cast<mlir::ShapedType>(),
                         origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::ConvolutionOp origOp, Logger log) {
    return verifyConvCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                         origOp.input().getType().cast<mlir::ShapedType>(),
                         origOp.filter().getType().cast<mlir::ShapedType>(),
                         origOp.output().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyPoolCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             mlir::ShapedType inputType, mlir::ShapedType outputType,
                                                             mlir::ArrayAttr kernelSize, mlir::ArrayAttr kernelStrides,
                                                             Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: {0}", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelSize.size());

    const auto inputShape = getShape(inputType);
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            kernelSizeVals, kernelStridesVals[Dims4D::Strides::X], inputType.getElementType(), IC);

    const auto requiredCMX = getRequiredCMXForTiling({inputType, outputType}, IC) + activationWindowSize * 1_Byte;

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Pooling, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::MaxPoolOp origOp, Logger log) {
    return verifyPoolCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                         origOp.input().getType().cast<mlir::ShapedType>(),
                         origOp.output().getType().cast<mlir::ShapedType>(), origOp.kernel_size(), origOp.strides(),
                         log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::MaxPoolOp origOp, Logger log) {
    return verifyPoolCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                         origOp.input().getType().cast<mlir::ShapedType>(),
                         origOp.output().getType().cast<mlir::ShapedType>(), origOp.kernel_size(), origOp.strides(),
                         log);
}

//
// verifyEltwiseCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module,
                                                                mlir::ShapedType firstInputType,
                                                                mlir::ShapedType secondInputType,
                                                                mlir::ShapedType outputType, Logger log) {
    log.setName("NCEInvariant");

    const auto requiredCMX = getRequiredCMXForTiling({firstInputType, secondInputType, outputType}, 0);

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Eltwise, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::AddOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::MultiplyOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::SubtractOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::AndOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::AddOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::MultiplyOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::SubtractOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::AndOp origOp, Logger log) {
    return verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                            origOp.input1().getType().cast<mlir::ShapedType>(),
                            origOp.input2().getType().cast<mlir::ShapedType>(),
                            origOp.output().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyGroupConvCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                                  mlir::ShapedType inputType,
                                                                  mlir::ShapedType filterType,
                                                                  mlir::ShapedType outputType,
                                                                  mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelStrides.size());

    const auto filterShape = getShape(filterType);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Setting more than 16 groups results in worse accuracy on KMB.
    // FIXME verify CMX is not a proper place for this. But it is required to fail CMX check during tiling.
    const auto depthwiseOutChanCount = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());

    if (OC % depthwiseOutChanCount != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  depthwiseOutChanCount, OC);
        return mlir::failure();
    }

    const Shape kernelSizeVals{KY, KX};
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            kernelSizeVals, kernelStridesVals[Dims4D::Strides::X], inputType.getElementType(), OC);

    // consider alignment when calculating required CMX
    const auto depthwiseConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % depthwiseConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    const int64_t alignment = (remainder > 0) ? (depthwiseConvAlignment - remainder) : 0;
    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, filtersPerInChan * KY * KX + alignment};
    const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());

    const auto requiredCMX =
            getRequiredCMXForTiling({inputType, alignedFilterType, outputType}, OC) + activationWindowSize * 1_Byte;

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Depthwise Convolution, available '{1}', required '{2}'", loc,
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IE::GroupConvolutionOp origOp, Logger log) {
    return verifyGroupConvCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                              origOp.input().getType().cast<mlir::ShapedType>(),
                              origOp.filter().getType().cast<mlir::ShapedType>(),
                              origOp.output().getType().cast<mlir::ShapedType>(), origOp.strides(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::GroupConvolutionOp origOp, Logger log) {
    return verifyGroupConvCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                              origOp.input().getType().cast<mlir::ShapedType>(),
                              origOp.filter().getType().cast<mlir::ShapedType>(),
                              origOp.output().getType().cast<mlir::ShapedType>(), origOp.strides(), log);
}

SmallVector<mlir::ShapedType> getTileTypes(IE::ConvolutionOp origOp, const TileInfo& outTile) {
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
    auto tileConf = vpux::backInferConvTile(outTile, getShape(origOp.input()), getShape(origOp.filter()), origBiasShape,
                                            origOp.strides(), origOp.pads_begin(), origOp.pads_end());

    SmallVector<mlir::ShapedType> tileTypes;

    tileTypes.push_back(getDenseTileType(origOp.input().getType().cast<mlir::ShapedType>(), tileConf.tiles[0].offsets,
                                         tileConf.tiles[0].shape));
    tileTypes.push_back(getDenseTileType(origOp.filter().getType().cast<mlir::ShapedType>(), tileConf.tiles[1].offsets,
                                         tileConf.tiles[1].shape));
    tileTypes.push_back(getDenseTileType(origOp.getType().cast<mlir::ShapedType>(), outTile.offsets, outTile.shape));

    return tileTypes;
}

// verifyPrefetchCMX

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::ConvolutionOp origOp, vpux::OutputTiling tiling,
                                                                 Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;
    if (isWeightPrefetch && curTile.axis[Dims4D::Act::H] > 1) {
        // Nested tiling is not supported for prefetch tiling
        return mlir::failure();
    }

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    const auto curOC = getShape(curTileTypes[1])[Dims4D::Filter::OC];

    const auto curInputTileType = curTileTypes[0];
    const auto curFilterTileType = curTileTypes[1];
    const auto curOutputTileType = curTileTypes[2];

    const auto nextInputTileType = nextTileTypes[0];
    const auto nextFilterTileType = nextTileTypes[1];

    if (isWeightPrefetch) {
        const auto nextOC = getShape(nextFilterTileType)[Dims4D::Filter::OC];
        requiredCMX = getRequiredCMXForTiling(
                {// Computing current tile, prefetch the next tile.
                 curInputTileType, curFilterTileType, curOutputTileType, nextFilterTileType},
                curOC + nextOC);
    } else {
        requiredCMX = getRequiredCMXForTiling(
                {curInputTileType, curFilterTileType, curOutputTileType, nextInputTileType}, curOC);
    }
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

SmallVector<mlir::ShapedType> getTileTypes(IE::MaxPoolOp origOp, const TileInfo& outTile) {
    auto tileConf = vpux::backInferPoolTile(outTile, getShape(origOp.input()), origOp.kernel_size(), origOp.strides(),
                                            origOp.pads_begin(), origOp.pads_end());

    SmallVector<mlir::ShapedType> tileTypes;

    tileTypes.push_back(getDenseTileType(origOp.input().getType().cast<mlir::ShapedType>(), tileConf.tiles[0].offsets,
                                         tileConf.tiles[0].shape));
    tileTypes.push_back(getDenseTileType(origOp.getType().cast<mlir::ShapedType>(), outTile.offsets, outTile.shape));

    return tileTypes;
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::MaxPoolOp origOp, vpux::OutputTiling tiling,
                                                                 Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;
    if (isWeightPrefetch && curTile.axis[Dims4D::Act::H] > 1) {
        // Nested tiling is not supported for prefetch tiling
        return mlir::failure();
    }

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    const auto curIC = getShape(curTileTypes[0])[Dims4D::Act::C];
    const auto nextIC = getShape(nextTileTypes[0])[Dims4D::Act::C];

    const auto curInputTileType = curTileTypes[0];
    const auto curOutputTileType = curTileTypes[1];

    const auto nextInputTileType = nextTileTypes[0];

    //  Consider tiling does not change the element type
    const auto inType = origOp.input().getType().cast<mlir::RankedTensorType>();
    const auto curActivationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            Shape(parseIntArrayAttr<int64_t>(origOp.kernel_sizeAttr())),
            parseIntArrayAttr<int64_t>(origOp.stridesAttr())[0], inType.getElementType(), curIC);
    const auto nextActivationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            Shape(parseIntArrayAttr<int64_t>(origOp.kernel_sizeAttr())),
            parseIntArrayAttr<int64_t>(origOp.stridesAttr())[0], inType.getElementType(), nextIC);

    requiredCMX = getRequiredCMXForTiling({curInputTileType, curOutputTileType, nextInputTileType}, curIC + nextIC) +
                  Byte(curActivationWindowSize + nextActivationWindowSize);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

SmallVector<mlir::ShapedType> getTileTypes(IE::GroupConvolutionOp origOp, const TileInfo& outTile) {
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
    auto tileConf =
            vpux::backInferGroupConvTile(outTile, getShape(origOp.input()), getShape(origOp.filter()), origBiasShape,
                                         origOp.strides(), origOp.pads_begin(), origOp.pads_end());

    SmallVector<mlir::ShapedType> tileTypes;

    tileTypes.push_back(getDenseTileType(origOp.input().getType().cast<mlir::ShapedType>(), tileConf.tiles[0].offsets,
                                         tileConf.tiles[0].shape));
    tileTypes.push_back(getDenseTileType(origOp.filter().getType().cast<mlir::ShapedType>(), tileConf.tiles[1].offsets,
                                         tileConf.tiles[1].shape));
    tileTypes.push_back(getDenseTileType(origOp.getType().cast<mlir::ShapedType>(), outTile.offsets, outTile.shape));

    return tileTypes;
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::GroupConvolutionOp origOp,
                                                                 vpux::OutputTiling tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;
    if (isWeightPrefetch && curTile.axis[Dims4D::Act::H] > 1) {
        // Nested tiling is not supported for prefetch tiling
        return mlir::failure();
    }

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    const auto curOC = getShape(curTileTypes[1])[Dims4D::Filter::OC];

    const auto curInputTileType = curTileTypes[0];
    const auto curFilterTileType = curTileTypes[1];
    const auto curOutputTileType = curTileTypes[2];

    const auto nextInputTileType = nextTileTypes[0];
    const auto nextFilterTileType = nextTileTypes[1];

    const auto curIC = getShape(curInputTileType)[Dims4D::Act::C];
    const auto nextIC = getShape(nextInputTileType)[Dims4D::Act::C];
    const auto inType = origOp.input().getType().cast<mlir::RankedTensorType>();

    const auto kernelSizeVals = Shape(SmallVector<int64_t>{getShape(curFilterTileType)[Dims4D::Filter::KX],
                                                           getShape(curFilterTileType)[Dims4D::Filter::KY]});

    const auto curActivationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            kernelSizeVals, parseIntArrayAttr<int64_t>(origOp.stridesAttr())[0], inType.getElementType(), curIC);
    const auto nextActivationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            kernelSizeVals, parseIntArrayAttr<int64_t>(origOp.stridesAttr())[0], inType.getElementType(), nextIC);

    if (isWeightPrefetch) {
        const auto nextOC = getShape(nextFilterTileType)[Dims4D::Filter::OC];
        requiredCMX =
                getRequiredCMXForTiling(
                        {// Computing current tile, prefetch the next tile.
                         curInputTileType, curFilterTileType, curOutputTileType, nextInputTileType, nextFilterTileType},
                        curOC + nextOC) +
                Byte(curActivationWindowSize + nextActivationWindowSize);
    } else {
        requiredCMX = getRequiredCMXForTiling(
                              {curInputTileType, curFilterTileType, curOutputTileType, nextInputTileType}, curOC) +
                      Byte(curActivationWindowSize + nextActivationWindowSize);
    }
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyEltwisePrefetchCMX
//

SmallVector<mlir::ShapedType> getTileTypes(mlir::Operation* op, const TileInfo& outTile) {
    auto tileConf = vpux::IE::backInferEltwiseTile(op, outTile);

    SmallVector<mlir::ShapedType> tileTypes;

    tileTypes.push_back(getDenseTileType(op->getOperand(0).getType().cast<mlir::ShapedType>(),
                                         tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(getDenseTileType(op->getOperand(1).getType().cast<mlir::ShapedType>(),
                                         tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            getDenseTileType(op->getResult(0).getType().cast<mlir::ShapedType>(), outTile.offsets, outTile.shape));

    return tileTypes;
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwisePrefetchCMX(mlir::Operation* op, vpux::OutputTiling tiling,
                                                                        Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = getTileTypes(op, curTile);
    const auto& nextTileTypes = getTileTypes(op, nextTile);

    const auto curInputTileType1 = curTileTypes[0];
    const auto curInputTileType2 = curTileTypes[1];
    const auto curOutputTileType = curTileTypes[2];

    const auto nextInputTileType1 = nextTileTypes[0];
    const auto nextInputTileType2 = nextTileTypes[1];

    requiredCMX = getRequiredCMXForTiling(
            {curInputTileType1, curInputTileType2, curOutputTileType, nextInputTileType1, nextInputTileType2}, 0);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'", op->getLoc(),
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::AddOp origOp, vpux::OutputTiling tiling,
                                                                 Logger log) {
    return verifyEltwisePrefetchCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::MultiplyOp origOp, vpux::OutputTiling tiling,
                                                                 Logger log) {
    return verifyEltwisePrefetchCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::SubtractOp origOp, vpux::OutputTiling tiling,
                                                                 Logger log) {
    return verifyEltwisePrefetchCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(IE::AndOp origOp, vpux::OutputTiling tiling,
                                                                 Logger log) {
    return verifyEltwisePrefetchCMX(origOp.getOperation(), tiling, log);
}

//
// verifyKernel
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(mlir::Location loc, int64_t KY, int64_t KX, int64_t SY,
                                                            int64_t SX, int64_t padTop, int64_t padBottom,
                                                            int64_t padLeft, int64_t padRight, VPU::ArchKind arch,
                                                            Logger log) {
    log.setName("NCEInvariant");

    static const int32_t NCE_MAX_KERNEL_SIZE = 11;
    static const int32_t NCE_MAX_STRIDE_SIZE = 8;

    if (KY > NCE_MAX_KERNEL_SIZE || KY <= 0) {
        log.trace("[{0}] Unsupported kernel height dimension '{1}', must be in range [1, {2}]", loc, KY,
                  NCE_MAX_KERNEL_SIZE);
        return mlir::failure();
    }
    if (KX > NCE_MAX_KERNEL_SIZE || KX <= 0) {
        log.trace("[{0}] Unsupported kernel width dimension '{1}', must be in range [1, {2}]", loc, KX,
                  NCE_MAX_KERNEL_SIZE);
        return mlir::failure();
    }

    if (SX != SY && arch != VPU::ArchKind::MTL) {
        log.trace("[{0}] Asymmetric strides are not supported", loc);
        return mlir::failure();
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        log.trace("[{0}] Unsupported stride height dimension '{1}', must be in range [1, {2}]", loc, SY,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        log.trace("[{0}] Unsupported stride width dimension '{1}', must be in range [1, {2}]", loc, SX,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    if (padTop < 0 || (padTop > 1 && padTop > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padTop, KY / 2);
        return mlir::failure();
    }
    if (padBottom < 0 || (padBottom > 1 && padBottom > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padBottom, KY / 2);
        return mlir::failure();
    }
    if (padLeft < 0 || (padLeft > 1 && padLeft > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padLeft, KX / 2);
        return mlir::failure();
    }
    if (padRight < 0 || (padRight > 1 && padRight > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padRight, KX / 2);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::MaxPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    if (kernelSize[0] != kernelSize[1]) {
        log.trace("[{0}] Assymetric kernel is not supported", origOp->getLoc());
        return mlir::failure();
    }
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::MaxPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    if (kernelSize[0] != kernelSize[1]) {
        log.trace("[{0}] Assymetric kernel is not supported", origOp->getLoc());
        return mlir::failure();
    }
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

//
// verifyEltwiseKernel
//

static mlir::LogicalResult verifyEltwiseKernel(mlir::ShapedType input1, mlir::ShapedType input2,
                                               mlir::ShapedType output, const bool allowDifferentScales = false,
                                               const bool allowDifferentZp = true) {
    // Eltwise add is expected to have the same shapes for all operands
    if (input1.getRank() != 4 || input2.getRank() != 4 || output.getRank() != 4) {
        return mlir::failure();
    }

    if (input1.getShape() != input2.getShape())
        return mlir::failure();

    // Output type can differ from input type. In case of quantization
    // this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    auto input1ElemType = input1.getElementType();
    auto input2ElemType = input2.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (input1ElemType != input2ElemType) {
            return mlir::failure();
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        auto qInput1 = input1ElemType.cast<mlir::quant::UniformQuantizedType>();
        auto qInput2 = input2ElemType.cast<mlir::quant::UniformQuantizedType>();

        if (qInput1.getExpressedType() != qInput2.getExpressedType() ||
            qInput1.getStorageType() != qInput2.getStorageType() || qInput1.isSigned() != qInput2.isSigned()) {
            return mlir::failure();
        }

        if (!allowDifferentZp && qInput1.getZeroPoint() != qInput2.getZeroPoint())
            return mlir::failure();

        if (!allowDifferentScales && qInput1.getScale() != qInput2.getScale())
            return mlir::failure();
    } else {
        VPUX_THROW("Unsupported inputs type. in1='{0}', in2='{1}'", input1ElemType, input2ElemType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::AddOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::AddOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::MultiplyOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType, true);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::MultiplyOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType, true);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::SubtractOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::SubtractOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::AndOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::AndOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<mlir::ShapedType>();
    auto input2Type = origOp.input2().getType().cast<mlir::ShapedType>();
    auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::GroupConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }
    if (origOp.filter().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!origOp.groups().hasValue()) {
        log.trace("[{0}] Grouped convolution does not have groups", origOp->getLoc());
        return mlir::failure();
    }
    if (origOp.groups().getValue() != OC) {
        log.trace("[{0}] Unsupported group size: '{1}' expected '{2}'", origOp->getLoc(), origOp.groups(), OC);
        return mlir::failure();
    }
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", origOp->getLoc());
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.input());
    const auto IC = inputShape[Dims4D::Act::C];
    if (OC != IC) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", origOp->getLoc(), OC, IC);
        return mlir::failure();
    }

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::GroupConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<mlir::ShapedType>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!origOp.groups().hasValue()) {
        log.trace("[{0}] Grouped convolution does not have groups", origOp->getLoc());
        return mlir::failure();
    }
    if (origOp.groups().getValue() != OC) {
        log.trace("[{0}] Unsupported group size: '{1}' expected '{2}'", origOp->getLoc(), origOp.groups(), OC);
        return mlir::failure();
    }

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

//
// verifyOp
//

namespace {

template <class ConcreteOp>
mlir::LogicalResult verifyConcreteOp(ConcreteOp origOp, Logger log) {
    const auto inputShape = getShape(origOp->getOperand(0));
    if (inputShape[Dims4D::Act::N] != 1) {
        log.trace("Input has unsupported batch: {0}", inputShape[Dims4D::Act::N]);
        return mlir::failure();
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(origOp, log))) {
        return mlir::failure();
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyChannels(origOp, log))) {
        return mlir::failure();
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyCMX(origOp, log))) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyOp(mlir::Operation* op, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IERT::ConvolutionOp>([&](IERT::ConvolutionOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::MaxPoolOp>([&](IERT::MaxPoolOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::AddOp>([&](IERT::AddOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::MultiplyOp>([&](IERT::MultiplyOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::SubtractOp>([&](IERT::SubtractOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::AndOp>([&](IERT::AndOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::GroupConvolutionOp>([&](IERT::GroupConvolutionOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Default([](mlir::Operation* unknownOp) -> mlir::LogicalResult {
                VPUX_THROW("Operation '{0}' at '{1}' is not supported by the NCE", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}
