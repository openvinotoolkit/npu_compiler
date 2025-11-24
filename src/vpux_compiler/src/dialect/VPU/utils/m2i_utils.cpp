//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::M2iColorFmt VPU::IEtoM2iColorFmt(IE::ColorFmt fmt) {
    if (fmt == IE::ColorFmt::NV12) {  // semi-planar
        return VPU::M2iColorFmt::SP_NV12_8;
    } else if (fmt == IE::ColorFmt::I420) {  // planar
        return VPU::M2iColorFmt::PL_YUV420_8;
    } else if (fmt == IE::ColorFmt::RGB) {  // C-minor
        return VPU::M2iColorFmt::IL_RGB888;
    } else if (fmt == IE::ColorFmt::BGR) {  // C-minor
        return VPU::M2iColorFmt::IL_RGB888;
    } else {
        VPUX_THROW("IEtoM2iColorFmt: unsupported format {0}", fmt);
    }
}

VPU::M2iInterp VPU::IEtoM2iInterpMode(IE::InterpolateMode mode) {
    if (mode == IE::InterpolateMode::NEAREST) {  // nearest neighbour
        return VPU::M2iInterp::NEAREST;
    } else if (mode == IE::InterpolateMode::LINEAR) {  // bilinear
        return VPU::M2iInterp::BILINEAR;
    }

    VPUX_THROW("IEtoM2iInterpMode: unsupported format {0}", mode);
}

long VPU::getM2iLineStride(NDTypeInterface ndType, size_t dimW) {
    auto shape = ndType.getShape().raw();
    auto lineStride = static_cast<long>(shape[dimW] * ndType.getElemTypeSize().to<Byte>().count());
    return lineStride;
}

// Line stride must be multiple of 16
bool VPU::isM2iLineStrideSupported(long lineStride) {
    return lineStride % 16 == 0;
}

// YCbCr format: 0 - Cb before Cr, 1 - Cr before Cb
// RGB format:   0 - RGB, 1 - BGR.
bool VPU::getM2iColorOrderReg(IE::ColorFmt fmt) {
    if (fmt == IE::ColorFmt::NV12) {
        return false;
    } else if (fmt == IE::ColorFmt::I420) {
        return false;
    } else if (fmt == IE::ColorFmt::RGB) {
        return false;
    } else if (fmt == IE::ColorFmt::BGR) {
        return true;
    } else {
        VPUX_THROW("IEtoM2iColorFmt: unsupported format {0}", fmt);
    }
}

// YCbCr: 0 - Y before Cb/Cr, 1 - Y after Cb/Cr
bool VPU::getM2iLumaOrderReg(IE::ColorFmt fmt) {
    if (fmt == IE::ColorFmt::NV12) {
        return false;
    } else if (fmt == IE::ColorFmt::I420) {
        return false;
    } else if (fmt == IE::ColorFmt::RGB) {
        return false;
    } else if (fmt == IE::ColorFmt::BGR) {
        return false;
    } else {
        VPUX_THROW("IEtoM2iColorFmt: unsupported format {0}", fmt);
    }
}

uint32_t VPU::getM2iFixedPointScaleFactor(uint32_t input, uint32_t output, uint16_t fractionalBits) {
    return static_cast<uint32_t>(std::round(static_cast<float>(input << fractionalBits) / static_cast<float>(output)));
}

uint32_t VPU::getM2iFixedPointTilingRegister(double input, uint16_t fractionalBits) {
    return static_cast<uint32_t>(round(input * (1 << fractionalBits)));
}

std::vector<double> VPU::getM2INormCoeffs(VPU::M2INormOp origOp) {
    // Build the {A,B,C,D} M2I coefs from {gamma, beta, mean, variance, eps}
    const auto gamma = parseFPArrayAttr<double>(origOp.getGammaValueAttr());
    const auto beta = parseFPArrayAttr<double>(origOp.getBetaValueAttr());
    const auto mean = parseFPArrayAttr<double>(origOp.getMeanValueAttr());
    const auto var = parseFPArrayAttr<double>(origOp.getVarianceValueAttr());
    const auto eps = origOp.getEpsAttr().getValueAsDouble();
    const auto numChannels = gamma.size();

    // {A,B,C,D} coefs for each channel
    std::vector<double> normCoefs;
    for (size_t i = 0; i < numChannels; i++) {
        normCoefs.push_back(gamma[i]);
        normCoefs.push_back(mean[i]);
        normCoefs.push_back(std::sqrt(var[i] + eps));
        normCoefs.push_back(beta[i]);
    }
    return normCoefs;
}

mlir::ArrayAttr VPU::getM2INormCoeffsAttr(mlir::MLIRContext* ctx, VPU::M2INormOp origOp) {
    return getFPArrayAttr(ctx, getM2INormCoeffs(origOp));
}

bool VPU::isM2IBatchNormSupported(mlir::Value input, mlir::Value output, LogCb logCb) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(output.getType());

    // Norm only defined for FP, and M2I only supports fp16
    auto iType = inType.getElementType();
    auto oType = outType.getElementType();

    if (!iType.isF16()) {
        logCb(llvm::formatv("Op only supports F16 input, got {0}", iType));
        return false;
    }

    if (!oType.isF16()) {
        logCb(llvm::formatv("Op only supports F16 output, got {0}", oType));
        return false;
    }

    const auto rank = inType.getShape().size();
    if (rank != 4) {
        logCb(llvm::formatv("Op only supports 4D shape, got {0}", rank));
        return false;
    }

    const auto lnStride = getM2iLineStride(inType, Dims4D::Act::W.ind());
    if (!VPU::isM2iLineStrideSupported(lnStride)) {
        logCb(llvm::formatv("Convert to M2I : line-stride NOT multiple of 16, got {0}", lnStride));
        return false;
    }

    return true;
}

template <typename InputOp>
bool VPU::isM2IResizeSupported(InputOp op, LogCb logCb, bool checkFp16Interleaved) {
    mlir::Value input = op.getInput();
    mlir::Value output = op.getOutput();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(output.getType());

    const auto iType = inType.getElementType();
    if (!(iType.isUnsignedInteger(8) || iType.isF16())) {
        logCb(llvm::formatv("Op only supports UI8/F16 input, got {0}", iType));
        return false;
    }
    const auto oType = outType.getElementType();
    if (!(oType.isUnsignedInteger(8) || oType.isF16())) {
        logCb(llvm::formatv("Op only supports UI8/F16 output, got {0}", oType));
        return false;
    }

    const auto shapeCalcMode = op.getAttr().getShapeCalcMode().getValue();
    if (shapeCalcMode != IE::InterpolateCalcMode::SIZES) {
        logCb(llvm::formatv("Op only implements 'sizes' mode, got {0}", shapeCalcMode));
        return false;
    }

    const auto sizesSize = op.getSizesAttrAttr().size();
    if (sizesSize != 2) {
        logCb(llvm::formatv("M2I can only resize 2D images, got {0}D", sizesSize));
        return false;
    }

    const auto axesSize = op.getAxesAttrAttr().size();
    if (sizesSize != axesSize) {
        logCb(llvm::formatv("Interpolate sizes/axes attr must have same size, got {0}, {1}", sizesSize, axesSize));
        return false;
    }

    const auto axes = parseIntArrayAttr<int64_t>(op.getAxesAttrAttr());
    const auto Waxis = axes[1];  // H(0),W(1)
    const auto iStride = getM2iLineStride(inType, Waxis);
    if (!VPU::isM2iLineStrideSupported(iStride)) {
        logCb(llvm::formatv("Input line-stride NOT multiple of 16, got {0}", iStride));
        return false;
    }
    const auto oStride = getM2iLineStride(outType, Waxis);
    if (!VPU::isM2iLineStrideSupported(oStride)) {
        logCb(llvm::formatv("Output line-stride NOT multiple of 16, got {0}", oStride));
        return false;
    }
    // Check consecutive axes
    if (axes[0] != (axes[1] - 1)) {
        logCb(llvm::formatv("Axes need to be consecutive values, got {0}, {1}", axes[0], axes[1]));
        return false;
    }

    const auto interpMode = op.getAttr().getMode().getValue();
    const auto coordMode = op.getAttr().getCoordMode().getValue();
    const auto nearestMode = op.getAttr().getNearestMode().getValue();
    if (interpMode == IE::InterpolateMode::NEAREST) {
        if (!(coordMode == IE::InterpolateCoordMode::ASYMMETRIC && nearestMode == IE::InterpolateNearestMode::FLOOR)) {
            logCb(llvm::formatv("M2I-HW in nearest neighbour mode only supports ASYMMETRIC coord mode and FLOOR "
                                "rounding mode, got {0} and {1}",
                                coordMode, nearestMode));
            return false;
        }
    } else if (interpMode == IE::InterpolateMode::LINEAR) {
        if (!(coordMode == IE::InterpolateCoordMode::HALF_PIXEL)) {
            logCb(llvm::formatv("M2I-HW in bilinear mode only supports HALF_PIXEL coord mode, got {0}", coordMode));
            return false;
        }
    } else {
        logCb(llvm::formatv("M2I-HW only supports nearest/linear interpolate, got {0}", interpMode));
        return false;
    }

    if (checkFp16Interleaved) {
        // Interleaved fp16 not supported by M2I-HW
        const auto lastAxis = axes[axesSize - 1];
        const auto lastDim = outType.getRank() - 1;
        if (oType.isF16() && (lastAxis != lastDim)) {
            logCb(llvm::formatv(
                    "Interleaved fp16 not supported by M2I-HW, expecting last_axis == last_dim, got {0} != {1}",
                    lastAxis, lastDim));
            return false;
        }
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(op.getAttr().getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.getAttr().getPadsEnd());

    auto isNotZero = [](auto val) {
        return val != 0;
    };

    if (llvm::any_of(padsBegin, isNotZero) || llvm::any_of(padsEnd, isNotZero)) {
        logCb(llvm::formatv("Op does not support pads"));
        return false;
    }

    return true;
}

// Template instantiation
template bool VPU::isM2IResizeSupported<IE::InterpolateOp>(IE::InterpolateOp op, LogCb logCb,
                                                           bool checkFp16Interleaved);
template bool VPU::isM2IResizeSupported<VPU::InterpolateOp>(VPU::InterpolateOp op, LogCb logCb,
                                                            bool checkFp16Interleaved);
