//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {
// M2I is provided with fixed point scaling factors and tiling offsets registers, with a (15,17) format (32 bits and 17
// fractional bits)
constexpr uint16_t M2I_SCALE_FACTOR_FIXED_POINT_BITS = 32;
constexpr uint16_t M2I_SCALE_FACTOR_FRACTIONAL_BITS = 17;
constexpr uint16_t M2I_TILING_REG_FIXED_POINT_BITS = 32;
constexpr uint16_t M2I_TILING_REG_FRACTIONAL_BITS = 17;

VPU::M2iColorFmt IEtoM2iColorFmt(IE::ColorFmt fmt);
VPU::M2iInterp IEtoM2iInterpMode(IE::InterpolateMode mode);
long getM2iLineStride(NDTypeInterface ndType, size_t dimW);
bool isM2iLineStrideSupported(long lineStride);
bool getM2iColorOrderReg(IE::ColorFmt fmt);
bool getM2iLumaOrderReg(IE::ColorFmt fmt);
uint32_t getM2iFixedPointScaleFactor(uint32_t input, uint32_t output, uint16_t fractionalBits);
uint32_t getM2iFixedPointTilingRegister(double input, uint16_t fractionalBits);
std::vector<double> getM2INormCoeffs(VPU::M2INormOp origOp);
mlir::ArrayAttr getM2INormCoeffsAttr(mlir::MLIRContext* ctx, VPU::M2INormOp origOp);
bool isM2IBatchNormSupported(mlir::Value input, mlir::Value output, LogCb logCb);
template <typename InputOp>
bool isM2IResizeSupported(InputOp op, LogCb logCb, bool checkFp16Interleaved = false);

}  // namespace VPU
}  // namespace vpux
