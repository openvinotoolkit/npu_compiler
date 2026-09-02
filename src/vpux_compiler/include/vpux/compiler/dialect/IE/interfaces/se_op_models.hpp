//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Common SEOpInterface external model implementations shared across architectures.
// Each model fully implements `isSupported()` with the operation-specific checks
// that determine whether the operation can be lowered to an NCE operation using
// Storage Element pointers.
//
// Architecture-specific models (e.g. SETransposedConvOpModel for NPU37XX) remain
// in the corresponding arch file.

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <cstdint>

namespace vpux::IE {

// Interpolate: checks mode, axes, channel count, padding, kernel size, alignment, layout
template <class MainOpType>
class SEInterpolateOpModel final :
        public IE::SEOpInterface::ExternalModel<SEInterpolateOpModel<MainOpType>, MainOpType> {
public:
    bool isSupported(mlir::Operation* op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool checkBatch) const {
        auto concreteOp = mlir::cast<MainOpType>(op);
        const auto attr = concreteOp.getAttr();
        if (attr == nullptr) {
            logCb(formatv("Missing Interpolate configuration information"));
            return false;
        }

        auto inputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getInput().getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getOutput().getType());
        auto inputShape = inputType.getShape();
        auto outputShape = outputType.getShape();

        if (checkBatch && inputShape[Dims4D::Act::N] != 1) {
            return false;
        }
        if (inputShape.isDynamic() || outputShape.isDynamic()) {
            logCb(formatv("Dynamic shapes are not supported. Input shape {0}, output shape {1}", inputShape,
                          outputShape));
            return false;
        }
        // TODO E#71403: remove dimension check
        auto exceedsDimLimit = [](int64_t dim) {
            return dim > VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
        };
        if (llvm::any_of(inputShape, exceedsDimLimit) || llvm::any_of(outputShape, exceedsDimLimit)) {
            logCb(formatv("Dimension sizes over {0} are not supported. Input shape {1}, output shape {2}",
                          VPU::NCEInvariant::VPU_DIMENSION_LIMIT, inputShape, outputShape));
            return false;
        }

        // Antialias is not supported
        if (attr.getAntialias() != nullptr && attr.getAntialias().getValue() == true) {
            logCb(formatv("Antialias is not supported"));
            return false;
        }

        const auto attrModeValue = attr.getMode().getValue();
        const auto attrCoordMode = attr.getCoordMode();
        const auto attrCoordModeValue = attrCoordMode.getValue();

        // Only 4D interpolates are supported and the interpolation axes must be H and/or W
        auto potentialScales = VPU::getNCEInterpolateScales(inputType, outputType, attrCoordMode);
        if (!potentialScales.has_value()) {
            return false;
        }

        if (inputShape[Dims4D::Act::C] < 8) {
            // Interpolate layers with fewer than 8 channels may perform better on SHAVE than on DPU #E100988.
            // A better cost model can be introduced in the future to clearly identify which scenarios
            // receive a hit in performance when executed on DPU
            const auto totalSpatialSize = inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W] +
                                          outputShape[Dims4D::Act::H] * outputShape[Dims4D::Act::W];
            logCb(formatv("Interpolate has less than 8 channels: {0}", inputShape[Dims4D::Act::C]));
            if (attrModeValue == IE::InterpolateMode::NEAREST) {
                // Experiments in E#156089 validated that, for nearest mode with total spatial size >= 1320720 (e.g.
                // 512x512, scale=2) DPU solution always has better performance even when channels < 8
                if (totalSpatialSize < 1320720) {
                    return false;
                }
            } else {
                // Experiments in E#228181 showed that many cases for bilinear interpolate showed better performance
                // with SEP. However, measuring the same interpolate in isolation differed in performance compared to
                // executing it within a network (e.g. 4x160x160 -> 4x640x640 is better in isolation, worse in network),
                // while other cases showed better performance in both isolation and network (e.g. 4x256x256 ->
                // 4x1024x1024). The current threshold is focused on these examples.
                if (totalSpatialSize < (256 * 256) + (1024 * 1024)) {
                    return false;
                }
            }
        }

        // Check for the supported modes and their coordinate transformation restrictions
        if (attrModeValue == IE::InterpolateMode::NEAREST) {
            // TODO E#83681: Add support for NEAREST ALIGN_CORNERS mode
            if (attrCoordModeValue == IE::InterpolateCoordMode::ALIGN_CORNERS) {
                logCb(formatv("Coordinate transformation mode {0} is not yet supported", attrCoordModeValue));
                return false;
            }
        } else if (attrModeValue == IE::InterpolateMode::LINEAR || attrModeValue == IE::InterpolateMode::LINEAR_ONNX) {
            // TODO E#107568: Add support for LINEAR TF_HALF_PIXEL_FOR_NN mode
            if (attrCoordModeValue == IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN) {
                logCb(formatv("Bilinear InterpolateOp with coordinate transformation mode {0} is not yet supported",
                              attrCoordModeValue));
                return false;
            }
        } else {
            logCb(formatv("Mode {0} is not supported", attrModeValue));
            return false;
        }

        // Only interpolate ops without padding are supported
        auto hasNonZeroPads = [&](mlir::ArrayAttr padsAttr) -> bool {
            if (padsAttr == nullptr) {
                return false;
            }
            auto pads = parseIntArrayAttr<int64_t>(padsAttr);
            return llvm::any_of(pads, [](int64_t pad) {
                return pad != 0;
            });
        };
        if (hasNonZeroPads(attr.getPadsBegin()) || hasNonZeroPads(attr.getPadsEnd())) {
            logCb(formatv("Padding is not supported"));
            return false;
        }

        const auto scales = potentialScales.value();
        // kernelSize must be in range [1:MAX_KERNEL_SIZE]
        const auto kernelSize =
                VPU::getNCEInterpolateKernelSize(scales, VPU::getNCEInterpolateModeAttr(attr.getMode()), attrCoordMode);
        const auto maxKernelSize = config::getNPUConstraints(op->getContext()).maxKernelSize;
        VPUX_THROW_WHEN(maxKernelSize <= 0, "Invalid maxKernelSize {0}", maxKernelSize);
        for (auto kernel : kernelSize) {
            if (kernel > maxKernelSize || kernel <= 0) {
                logCb(formatv("Only kernel size less than {0} are supported for nce interpolate. Got kernel Size "
                              "{1}",
                              maxKernelSize, kernel));
                return false;
            }
        }

        if (checkChannelAlignment) {
            if (!VPU::NCEInvariant::isInputActTypeSupported(
                        inputType, vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType()),
                        /*supportsInputActCompression=*/false) ||
                !VPU::NCEInvariant::isOutputActTypeSupported(
                        op, outputType, vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType()))) {
                logCb(formatv("Misaligned tensor shape"));
                return false;
            }
        }

        if (checkLayout) {
            const auto arch = config::getArch(op);
            if (!VPU::NCEInvariant::checkLayouts({inputType}, {outputType}, arch, 1, logCb)) {
                return false;
            }
        }

        return true;
    }
};

// PadOp: validates mode, pad values, spatial padding, channel threshold, then delegates to NCE conv check.
template <class MainOpType, bool HasSparsityMapSupport = true>
class SEPadOpModel final :
        public IE::SEOpInterface::ExternalModel<SEPadOpModel<MainOpType, HasSparsityMapSupport>, MainOpType> {
public:
    bool isSupported(mlir::Operation* op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool /*checkBatch*/) const {
        auto concreteOp = mlir::cast<MainOpType>(op);
        auto inputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getInput().getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getOutput().getType());
        auto* ctx = concreteOp.getContext();

        if (inputType.getShape().size() != 4) {
            logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
            return false;
        }
        if (outputType.getShape().size() != 4) {
            logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
            return false;
        }

        const auto padsBeginAttr = concreteOp.getPadsBeginAttrAttr();
        const auto padsEndAttr = concreteOp.getPadsEndAttrAttr();
        if (padsBeginAttr == nullptr || padsEndAttr == nullptr) {
            logCb(formatv("Only constant pads begin and pads end are supported"));
            return false;
        }

        if (concreteOp.getMode() == IE::PadMode::CONSTANT) {
            if constexpr (!HasSparsityMapSupport) {
                logCb(formatv("CONSTANT pad mode requires a sparsity map not supported on this architecture"));
                return false;
            }
            const auto padValueAttr = concreteOp.getPadValueAttrAttr();
            if (padValueAttr == nullptr) {
                logCb(formatv("PadMode with CONSTANT should have constant pad value"));
                return false;
            }
            const auto padValue = padValueAttr.getValue().convertToDouble();
            if (!isDoubleEqual(padValue, 0.f)) {
                logCb(formatv("Only CONSTANT mode with pad value '0' is supported"));
                return false;
            }
        }

        const auto inputShape = inputType.getShape();
        const auto padsBegin = parseIntArrayAttr<int64_t>(padsBeginAttr);
        const auto padsEnd = parseIntArrayAttr<int64_t>(padsEndAttr);

        // Check for negative padding values - SEP does not support negative pads currently.
        auto hasNegativePad = [](ArrayRef<int64_t> padsValue) {
            return llvm::any_of(padsValue, [](int64_t pad) {
                return pad < 0;
            });
        };
        if (hasNegativePad(padsBegin) || hasNegativePad(padsEnd)) {
            logCb(formatv("Negative padding values are not supported"));
            return false;
        }

        auto isSpatialPadding = [](ArrayRef<int64_t> padsValue) {
            return padsValue[Dims4D::Act::N.ind()] == 0 && padsValue[Dims4D::Act::C.ind()] == 0;
        };
        if (!isSpatialPadding(padsBegin) || !isSpatialPadding(padsEnd)) {
            logCb(formatv("Only spatial padding is supported"));
            return false;
        }

        const auto newY = inputShape[Dims4D::Act::H] + padsBegin[Dims4D::Act::H.ind()] + padsEnd[Dims4D::Act::H.ind()];
        const auto newX = inputShape[Dims4D::Act::W] + padsBegin[Dims4D::Act::W.ind()] + padsEnd[Dims4D::Act::W.ind()];
        const Shape newInputShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], newY, newX};
        inputType = inputType.changeShape(newInputShape);

        // SEP PadOp will get fused into the next Convolution, if that exists, or it will be converted
        // into a 1x1 Convolution, input sparsity will be enabled and storage element pointers will be used.
        // E-163345 shows that the overhead of storage element table reads from CMX will bring a significant
        // performance regression when Conv Input Channel number is low.
        const auto& strategyFactory = IE::getIEStrategyFactory(ctx);
        const auto sePadICPerfThresholdVerifier = strategyFactory->getSEPadICPerfThresholdVerifier();
        if (!sePadICPerfThresholdVerifier->isBeneficialForPerformance(newInputShape, op)) {
            return false;
        }

        auto weightShape =
                Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::C], /*KY=*/1, /*KX=*/1});
        mlir::Type elemType = mlir::Float16Type::get(ctx);
        if (mlir::isa<mlir::quant::QuantizedType>(inputType.getElementType())) {
            elemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/static_cast<double>(1.0f), /*zeroPoint=*/0, /*storageTypeMin=*/0,
                    /*storageTypeMax=*/255);
        }
        const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
        const auto weightsType =
                mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(weightShape.raw(), elemType, tensorAttr));

        const auto dilations = SmallVector<int64_t>{1, 1};
        const auto pads = PadInfo(0, 0, 0, 0);

        // When SEP Pad Op is enabled, it will be converted to NCEConvolution
        // with kernel size, strides, and dilations set to [1, 1]
        // This is to verify that it can meet the NCEConvolution HW requirements, such as channel alignment
        // and layout.
        return VPU::isNCEConvSupported(op, inputType, weightsType, outputType, dilations, /*KY=*/1, /*KX=*/1,
                                       /*SY=*/1, /*SX=*/1, pads, checkLayout, checkChannelAlignment, logCb);
    }
};

// RollOp: validates rank and axes, then delegates to NCE conv check
template <class MainOpType>
class SERollOpModel final : public IE::SEOpInterface::ExternalModel<SERollOpModel<MainOpType>, MainOpType> {
public:
    bool isSupported(mlir::Operation* op, vpux::LogCb logCb, bool checkLayout, bool /*checkChannelAlignment*/,
                     bool /*checkBatch*/) const {
        auto concreteOp = mlir::cast<MainOpType>(op);
        auto inputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getData().getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getOutput().getType());
        auto* ctx = concreteOp.getContext();
        const auto inputShape = inputType.getShape();

        if (inputShape.size() != 4 || outputType.getRank() != 4) {
            logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputShape.size()));
            return false;
        }

        auto shiftAndAxesOrFail = IE::getShiftAndAxesForRollOp(concreteOp.getLoc(), concreteOp.getShift(),
                                                               concreteOp.getAxes(), inputShape);
        if (mlir::failed(shiftAndAxesOrFail)) {
            return false;
        }
        const auto axes = shiftAndAxesOrFail.value().axes;

        if (axes.size() != 2) {
            logCb(formatv("{0} dimensions to roll", axes.size()));
            return false;
        }
        if (axes[0] != Dims4D::Act::H.ind() || axes[1] != Dims4D::Act::W.ind()) {
            logCb(formatv("it's not spatial rolling"));
            return false;
        }

        const int64_t KY = 1;
        const int64_t KX = 1;

        auto weightShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::C], KY, KX});
        mlir::Type elemType = mlir::Float16Type::get(ctx);
        if (mlir::isa<mlir::quant::QuantizedType>(inputType.getElementType())) {
            elemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/static_cast<double>(1.0f), /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
        }
        const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
        const auto weightsType =
                mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(weightShape.raw(), elemType, tensorAttr));

        const int64_t SY = 1;
        const int64_t SX = 1;

        PadInfo pads(0, 0, 0, 0);
        const auto dilations = SmallVector<int64_t>{1, 1};

        return VPU::isNCEConvSupported(op, inputType, weightsType, outputType, dilations, KY, KX, SY, SX, pads,
                                       checkLayout,
                                       /*checkChannelAlignment*/ true, logCb);
    }
};

// DilatedGroupConv: validates dilation, padding compatibility, then delegates to NCE conv check
template <class MainOpType>
class SEDilatedGroupConvOpModel final :
        public IE::SEOpInterface::ExternalModel<SEDilatedGroupConvOpModel<MainOpType>, MainOpType> {
public:
    bool isSupported(mlir::Operation* op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool /*checkBatch*/) const {
        // Dilated GroupConv SE path requires the experimental SE flag
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        if (!config::hasEnableExperimentalSEPtrsOperations(moduleOp)) {
            return false;
        }

        auto concreteOp = mlir::cast<MainOpType>(op);

        if (!VPU::areConvInputOutputs4d(concreteOp, logCb)) {
            return false;
        }
        auto dilations = parseIntArrayAttr<int64_t>(concreteOp.getDilations());
        const auto dilationY = dilations[Dims4D::Dilation::Y.ind()];
        const auto dilationX = dilations[Dims4D::Dilation::X.ind()];
        if (dilationY == 1 && dilationX == 1) {
            return false;
        }
        auto pads = PadInfo(concreteOp.getPadsBegin(), concreteOp.getPadsEnd());
        if (!VPU::isSupportedSEPDilatedConvPadding(pads, dilations)) {
            return false;
        }
        const auto filterShape = getShape(concreteOp.getFilter());
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getInput().getType());
        const auto filterType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getFilter().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getOutput().getType());

        const auto KY = filterShape[Dims4D::Filter::KY];
        const auto KX = filterShape[Dims4D::Filter::KX];
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(concreteOp.getStrides()));
        const auto SY = kernelStrides[Dims4D::Strides::Y];
        const auto SX = kernelStrides[Dims4D::Strides::X];

        pads = VPU::shrinkPadsForDilatedConvolution(pads, dilations);

        // Normal isNCEConvSupported can be used to check if Op can be run on NCE,
        // when padding and dilation is adjusted
        dilations[Dims4D::Dilation::X.ind()] = 1;
        dilations[Dims4D::Dilation::Y.ind()] = 1;

        return VPU::isNCEConvSupported(op, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads,
                                       checkLayout, checkChannelAlignment, logCb,
                                       /*supportsInputActCompression*/ false);
    }
};

}  // namespace vpux::IE
