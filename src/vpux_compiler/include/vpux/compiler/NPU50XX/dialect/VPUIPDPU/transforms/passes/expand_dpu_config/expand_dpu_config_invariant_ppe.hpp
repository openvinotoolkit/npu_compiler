//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIPDPU::arch50xx::PPE {

struct PPETask {
    struct PPEFixedFunction {
        VPU::PPEMode ppeMode = VPU::PPEMode::NOOP;
        float fpClampLow = std::numeric_limits<float>::lowest();
        float fpClampHigh = std::numeric_limits<float>::max();
    } fixedFunction;
    PPEIntRoundMode rounding = PPEIntRoundMode::RNE;
    std::optional<float> fpScaleData;
    SmallVector<float> fpPreluAlpha = {1.f};
    float fpAdder = 0.f;
    std::optional<float> fpBias;
};

struct PPEConfig {
    struct BiasAdd {
        std::optional<float> biasStatic;
    } biasAdd;
    struct ScaleMult {
        std::optional<float> scaleStatic;
    } scaleMult;
    struct SprLut {
        PPEsprLUTMode enableLookUpTable = PPEsprLUTMode::OFF;
    } sprLUT;
    struct PreluMult {
        std::optional<float> preluAlpha;
    } preluMult;
    struct Clamp {
        std::optional<float> clampLow;
        std::optional<float> clampHigh;
    } clamp;
    struct Convert {
        PPEFpConvertMode convertMode = PPEFpConvertMode::NONE;
        std::optional<PPEFpConvClampMode> clampMode;
        std::optional<PPEFpConvFTZMode> ftzMode;
        std::optional<PPEFpConvBf16RoundMode> bf16RoundMode;
    } convert;
    struct ZeroPointOffset {
        int64_t zeroPointStatic = 0;
    } zeroPointOffset;
};

mlir::FailureOr<PPETask> evalPPETasks(const Logger& log, mlir::Region& ppeRegion);
mlir::LogicalResult buildPPEConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const PPEConfig& config, mlir::Value weightsTable);
mlir::LogicalResult configurePPE(PPEConfig& config, mlir::Type outDataType, const vpux::NDTypeInterface& inActType,
                                 VPUIP::NCETaskType dpuTaskType, const arch50xx::PPE::PPETask& ppeTask,
                                 bool isWeightTableProvided, bool isSprLookUpTableProvided);

}  // namespace vpux::VPUIPDPU::arch50xx::PPE
