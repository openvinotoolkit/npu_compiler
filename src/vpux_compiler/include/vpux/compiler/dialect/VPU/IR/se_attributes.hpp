//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include <cstdint>

namespace vpux {
namespace VPU {

// SEInterpolateAttr
constexpr int64_t SE_INTERPOLATE_FACTOR_H = 0;
constexpr int64_t SE_INTERPOLATE_FACTOR_W = 1;

constexpr int64_t SE_INTERPOLATE_KERNEL_Y = 0;
constexpr int64_t SE_INTERPOLATE_KERNEL_X = 1;

constexpr int64_t SE_INTERPOLATE_STRIDE_Y = 0;
constexpr int64_t SE_INTERPOLATE_STRIDE_X = 1;

// SEUpsamplingAttr
constexpr int64_t SE_UPSAMPLING_FACTOR_H = 0;
constexpr int64_t SE_UPSAMPLING_FACTOR_W = 1;

constexpr int64_t SE_PAD_LEFT = 0;
constexpr int64_t SE_PAD_TOP = 1;
constexpr int64_t SE_PAD_RIGHT = 2;
constexpr int64_t SE_PAD_BOTTOM = 3;

// SERollAttr
constexpr int64_t SE_ROLL_SPATIAL_H = 0;
constexpr int64_t SE_ROLL_SPATIAL_W = 1;

// SEDilatedConvAttr
constexpr int64_t SE_DILATEDCONV_DILATION_Y = 0;
constexpr int64_t SE_DILATEDCONV_DILATION_X = 1;

constexpr int64_t SE_DILATEDCONV_KERNEL_H = 0;
constexpr int64_t SE_DILATEDCONV_KERNEL_W = 1;

constexpr int64_t SE_DILATEDCONV_STRIDE_Y = 0;
constexpr int64_t SE_DILATEDCONV_STRIDE_X = 1;

namespace DilationUtils {

struct DilationFactors {
    int64_t dilateY;
    int64_t dilateX;
};

inline DilationFactors extractDilationFactors(mlir::ArrayAttr dilationFactorsAttr) {
    VPUX_THROW_UNLESS(dilationFactorsAttr != nullptr, "SEDilatedConvAttr: missing dilation attribute");
    auto dilationFactors = parseIntArrayAttr<int64_t>(dilationFactorsAttr);

    VPUX_THROW_WHEN(dilationFactors.size() != 2,
                    "SEDilatedConvAttr: dilationFactors should have rank of 2, but got rank {0}",
                    dilationFactors.size());

    return {dilationFactors[VPU::SE_DILATEDCONV_DILATION_Y], dilationFactors[VPU::SE_DILATEDCONV_DILATION_X]};
}

struct DilationKernelSize {
    int64_t kernelH;
    int64_t kernelW;
};

inline DilationKernelSize extractDilationKernelSize(mlir::ArrayAttr dilationKernelSizeAttr) {
    VPUX_THROW_UNLESS(dilationKernelSizeAttr != nullptr, "SEDilatedConvAttr: missing kernel size attribute");
    auto dilationKernelSize = parseIntArrayAttr<int64_t>(dilationKernelSizeAttr);

    VPUX_THROW_WHEN(dilationKernelSize.size() != 2,
                    "SEDilatedConvAttr: strides should have rank of 2, but got rank {0}", dilationKernelSize.size());

    return {dilationKernelSize[VPU::SE_DILATEDCONV_KERNEL_H], dilationKernelSize[VPU::SE_DILATEDCONV_KERNEL_W]};
}

struct DilationStrides {
    int64_t strideY;
    int64_t strideX;
};

inline DilationStrides extractDilationStrides(mlir::ArrayAttr dilationStridesAttr) {
    VPUX_THROW_UNLESS(dilationStridesAttr != nullptr, "SEDilatedConvAttr: missing strides attribute");
    auto dilationStrides = parseIntArrayAttr<int64_t>(dilationStridesAttr);

    VPUX_THROW_WHEN(dilationStrides.size() != 2,
                    "SEDilatedConvAttr: dilationStrides should have rank of 2, but got rank {0}",
                    dilationStrides.size());

    return {dilationStrides[VPU::SE_DILATEDCONV_STRIDE_Y], dilationStrides[VPU::SE_DILATEDCONV_STRIDE_X]};
}

struct DilationDataOffsets {
    int64_t offsetN;
    int64_t offsetC;
    int64_t offsetH;
    int64_t offsetW;
};

inline DilationDataOffsets extractDilationDataOffsets(mlir::ArrayAttr dilationDataOffsetsAttr) {
    VPUX_THROW_UNLESS(dilationDataOffsetsAttr != nullptr, "SEDilatedConvAttr: missing data offsets attribute");
    auto dilationDataOffsets = parseIntArrayAttr<int64_t>(dilationDataOffsetsAttr);

    VPUX_THROW_WHEN(dilationDataOffsets.size() != 4,
                    "SEDilatedConvAttr: dilationDataOffsets should have rank of 4, but got rank {0}",
                    dilationDataOffsets.size());

    return {dilationDataOffsets[Dims4D::Act::N.ind()], dilationDataOffsets[Dims4D::Act::C.ind()],
            dilationDataOffsets[Dims4D::Act::H.ind()], dilationDataOffsets[Dims4D::Act::W.ind()]};
}

struct DilationDataSizes {
    int64_t sizeN;
    int64_t sizeC;
    int64_t sizeH;
    int64_t sizeW;
};

inline DilationDataSizes extractDilationDataSizes(mlir::ArrayAttr dilationDataSizesAttr) {
    VPUX_THROW_UNLESS(dilationDataSizesAttr != nullptr, "SEDilatedConvAttr: missing data sizes attribute");
    auto dilationDataSizes = parseIntArrayAttr<int64_t>(dilationDataSizesAttr);

    VPUX_THROW_WHEN(dilationDataSizes.size() != 4,
                    "SEDilatedConvAttr: dilationDataSizes should have rank of 4, but got rank {0}",
                    dilationDataSizes.size());

    return {dilationDataSizes[Dims4D::Act::N.ind()], dilationDataSizes[Dims4D::Act::C.ind()],
            dilationDataSizes[Dims4D::Act::H.ind()], dilationDataSizes[Dims4D::Act::W.ind()]};
}
}  // namespace DilationUtils

}  // namespace VPU
}  // namespace vpux
