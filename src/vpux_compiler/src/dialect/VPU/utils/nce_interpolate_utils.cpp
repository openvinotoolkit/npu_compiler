//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <algorithm>

using namespace vpux;
using namespace VPU;

VPU::NCEInterpolateModeAttr VPU::getNCEInterpolateModeAttr(IE::InterpolateModeAttr origModeAttr) {
    if (origModeAttr == nullptr) {
        return nullptr;
    }

    auto ctx = origModeAttr.getContext();
    auto origMode = origModeAttr.getValue();
    switch (origMode) {
    case IE::InterpolateMode::NEAREST:
        return VPU::NCEInterpolateModeAttr::get(ctx, VPU::NCEInterpolateMode::NEAREST);
    case IE::InterpolateMode::LINEAR:
    case IE::InterpolateMode::LINEAR_ONNX:
        return VPU::NCEInterpolateModeAttr::get(ctx, VPU::NCEInterpolateMode::BILINEAR);
    default:
        return nullptr;
    }
}

bool VPU::isSupportedNCEInterpolateScales(ArrayRef<double> scales, vpux::LogCb logCb) {
    if (scales.size() != 4) {
        logCb(formatv("Only 4D scales is supported. Got {0}D", scales.size()));
        return false;
    }

    if (!isDoubleEqual(scales[Dims4D::Act::N.ind()], 1.0)) {
        logCb(formatv("Interpolation over axis {0} is not supported", Dims4D::Act::N.ind()));
        return false;
    }

    if (!isDoubleEqual(scales[Dims4D::Act::C.ind()], 1.0)) {
        logCb(formatv("Interpolation over axis {0} is not supported", Dims4D::Act::C.ind()));
        return false;
    }

    return std::all_of(scales.begin(), scales.end(), [&](const auto scale) {
        if (!isDoubleEqual(std::floor(scale), scale)) {
            logCb(formatv("Only integer scales are supported. Got scale {0}", scale));
            return false;
        }
        return true;
    });
}

std::optional<SmallVector<double>> VPU::getNCEInterpolateScales(NDTypeInterface inputType, NDTypeInterface outputType,
                                                                IE::InterpolateCoordModeAttr coordModeAttr) {
    const auto inputShape = inputType.getShape();
    const auto outputShape = outputType.getShape();
    VPUX_THROW_UNLESS(inputShape.size() == outputShape.size(),
                      "Input and output should have the same rank. Got {0}D input, {1}D output", inputShape.size(),
                      outputShape.size());

    SmallVector<double> scales(inputShape.size());
    std::transform(inputShape.begin(), inputShape.end(), outputShape.begin(), scales.begin(),
                   [](const auto inSize, const auto outSize) {
                       return static_cast<double>(outSize) / static_cast<double>(inSize);
                   });

    VPUX_THROW_UNLESS(coordModeAttr != nullptr, "CoordMode attribute is None");
    if (coordModeAttr.getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) {
        const auto inputShape = to_small_vector(inputType.getShape());
        const auto outputShape = to_small_vector(outputType.getShape());
        for (size_t dim = 0; dim < scales.size(); dim++) {
            if (!isDoubleEqual(scales[dim], 1.0)) {
                auto newInSize = isDoubleEqual(inputShape[dim], 1.0) ? 1.0 : (inputShape[dim] - 1.0);
                auto newOutSize = isDoubleEqual(outputShape[dim], 1.0) ? 1.0 : (outputShape[dim] - 1.0);
                scales[dim] = newOutSize / newInSize;
            }
        }
    }

    if (isSupportedNCEInterpolateScales(scales)) {
        return scales;
    }

    return std::nullopt;
}

SmallVector<int64_t> VPU::getNCEInterpolateFactors(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                   IE::InterpolateCoordModeAttr coordModeAttr) {
    VPUX_THROW_UNLESS(scales.size() == 4, "Scales should have rank 4, but got rank {0}", scales.size());
    VPUX_THROW_UNLESS(modeAttr != nullptr, "Mode attribute is None");
    VPUX_THROW_UNLESS(coordModeAttr != nullptr, "CoordMode attribute is None");

    const auto mode = modeAttr.getValue();
    const auto coordMode = coordModeAttr.getValue();
    auto scaleH = static_cast<int64_t>(scales[Dims4D::Act::H.ind()]);
    auto scaleW = static_cast<int64_t>(scales[Dims4D::Act::W.ind()]);
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        return SmallVector<int64_t>{scaleH, scaleW};
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        switch (coordMode) {
        case IE::InterpolateCoordMode::HALF_PIXEL:
        case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL:
        case IE::InterpolateCoordMode::ASYMMETRIC:
        case IE::InterpolateCoordMode::ALIGN_CORNERS:
        case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
            return SmallVector<int64_t>{scaleH, scaleW};
        default:
            VPUX_THROW("Get unsupported Interpolate coordMode '{0}'", coordMode);
        }
    }
    VPUX_THROW("Get unsupported NCEInterpolate Mode '{0}'", mode);
}

SmallVector<int64_t> VPU::getNCEInterpolatePadsBegin(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                     IE::InterpolateCoordModeAttr coordModeAttr) {
    VPUX_THROW_UNLESS(scales.size() == 4, "Scales should have rank 4, but got rank {0}", scales.size());
    VPUX_THROW_UNLESS(modeAttr != nullptr, "Mode attribute is None");
    VPUX_THROW_UNLESS(coordModeAttr != nullptr, "CoordMode attribute is None");

    const auto mode = modeAttr.getValue();
    const auto coordMode = coordModeAttr.getValue();
    auto scaleH = static_cast<int64_t>(scales[Dims4D::Act::H.ind()]);
    auto scaleW = static_cast<int64_t>(scales[Dims4D::Act::W.ind()]);
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        return SmallVector<int64_t>{0, 0};
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        switch (coordMode) {
        case IE::InterpolateCoordMode::HALF_PIXEL:
        case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL: {
            const int64_t isEvenScaleH = (scaleH % 2 == 0) ? 1 : 0;
            const int64_t isEvenScaleW = (scaleW % 2 == 0) ? 1 : 0;

            scaleH = ((scaleH - 1) >> 1) + isEvenScaleH;
            scaleW = ((scaleW - 1) >> 1) + isEvenScaleW;

            return SmallVector<int64_t>{scaleH, scaleW};
        }
        case IE::InterpolateCoordMode::ASYMMETRIC:
        case IE::InterpolateCoordMode::ALIGN_CORNERS:
            return SmallVector<int64_t>{0, 0};
        case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
            return SmallVector<int64_t>{1, 1};
        default:
            VPUX_THROW("Get unsupported Interpolate coordMode '{0}'", coordMode);
        }
    }
    VPUX_THROW("Get unsupported NCEInterpolate Mode '{0}'", mode);
}

SmallVector<int64_t> VPU::getNCEInterpolatePadsEnd(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                   IE::InterpolateCoordModeAttr coordModeAttr) {
    VPUX_THROW_UNLESS(scales.size() == 4, "Scales should have rank 4, but got rank {0}", scales.size());
    VPUX_THROW_UNLESS(modeAttr != nullptr, "Mode attribute is None");
    VPUX_THROW_UNLESS(coordModeAttr != nullptr, "CoordMode attribute is None");

    const auto mode = modeAttr.getValue();
    const auto coordMode = coordModeAttr.getValue();
    auto scaleH = static_cast<int64_t>(scales[Dims4D::Act::H.ind()]);
    auto scaleW = static_cast<int64_t>(scales[Dims4D::Act::W.ind()]);
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        return SmallVector<int64_t>{0, 0};
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        switch (coordMode) {
        case IE::InterpolateCoordMode::HALF_PIXEL:
        case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL: {
            const int64_t isEvenScaleH = (scaleH % 2 == 0) ? 1 : 0;
            const int64_t isEvenScaleW = (scaleW % 2 == 0) ? 1 : 0;

            scaleH = ((scaleH - 1) >> 1) + isEvenScaleH;
            scaleW = ((scaleW - 1) >> 1) + isEvenScaleW;

            return SmallVector<int64_t>{scaleH, scaleW};
        }
        case IE::InterpolateCoordMode::ASYMMETRIC:
        case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
            return SmallVector<int64_t>{scaleH - 1, scaleW - 1};
        case IE::InterpolateCoordMode::ALIGN_CORNERS:
            return SmallVector<int64_t>{0, 0};
        default:
            VPUX_THROW("Get unsupported Interpolate coordMode '{0}'", coordMode);
        }
    }
    VPUX_THROW("Get unsupported NCEInterpolate Mode '{0}'", mode);
}

SmallVector<int64_t> VPU::getNCEInterpolateKernelSize(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                      IE::InterpolateCoordModeAttr coordModeAttr) {
    VPUX_THROW_UNLESS(scales.size() == 4, "Scales should have rank 4, but got rank {0}", scales.size());
    VPUX_THROW_UNLESS(modeAttr != nullptr, "Mode attribute is None");
    VPUX_THROW_UNLESS(coordModeAttr != nullptr, "CoordMode attribute is None");

    const auto mode = modeAttr.getValue();
    const auto coordMode = coordModeAttr.getValue();
    auto scaleH = static_cast<int64_t>(scales[Dims4D::Act::H.ind()]);
    auto scaleW = static_cast<int64_t>(scales[Dims4D::Act::W.ind()]);
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        return SmallVector<int64_t>{1, 1};
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        switch (coordMode) {
        case IE::InterpolateCoordMode::HALF_PIXEL:
        case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL: {
            const bool isEvenScaleH = scaleH % 2 == 0;
            const bool isEvenScaleW = scaleW % 2 == 0;

            scaleH += isEvenScaleH;
            scaleW += isEvenScaleW;

            return SmallVector<int64_t>{scaleH, scaleW};
        }
        case IE::InterpolateCoordMode::ASYMMETRIC:
        case IE::InterpolateCoordMode::ALIGN_CORNERS:
        case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
            return SmallVector<int64_t>{scaleH, scaleW};
        default:
            VPUX_THROW("Get unsupported Interpolate coordMode '{0}'", coordMode);
        }
    }
    VPUX_THROW("Get unsupported NCEInterpolate Mode '{0}'", mode);
}

SmallVector<int64_t> VPU::getNCEInterpolateStrides(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                   IE::InterpolateCoordModeAttr coordModeAttr) {
    VPUX_THROW_UNLESS(scales.size() == 4, "Scales should have rank 4, but got rank {0}", scales.size());
    VPUX_THROW_UNLESS(modeAttr != nullptr, "Mode attribute is None");
    VPUX_THROW_UNLESS(coordModeAttr != nullptr, "CoordMode attribute is None");

    const auto mode = modeAttr.getValue();
    const auto coordMode = coordModeAttr.getValue();
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        return SmallVector<int64_t>{1, 1};
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        switch (coordMode) {
        case IE::InterpolateCoordMode::HALF_PIXEL:
        case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL:
        case IE::InterpolateCoordMode::ASYMMETRIC:
        case IE::InterpolateCoordMode::ALIGN_CORNERS:
            return SmallVector<int64_t>{1, 1};
        case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
            return SmallVector<int64_t>{2, 2};
        default:
            VPUX_THROW("Get unsupported Interpolate coordMode '{0}'", coordMode);
        }
    }
    VPUX_THROW("Get unsupported NCEInterpolate Mode '{0}'", mode);
}
