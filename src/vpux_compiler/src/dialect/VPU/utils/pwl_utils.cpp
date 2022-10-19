//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/utils/pwl_utils.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// The PWL table is currently only used for U8 computation.
// When PWL is enabled, the hardware compute pipeline looks as follows:
//
//   Compute_stage -> [i32] -> Requant_stage * ((in_sc * wt_sc) / out_sc) -> CLAMP
//          [i13] -> PWL -> [i13] -> PostShift_stage << >> -> [i8] -> + ZP -> [u8]
//
// The PWL table requires fixed input and output quantization ranges for i13 data.
// For example, the pre-trained Sigmoid works best with [-4.0, 4.0] input range
// and [0.0, 1.0] output range. To ensure the correct range reaches the PWL, we
// need to map [-4.0, 4.0] to [-4096, 4095] (i13). This can be achieved in the
// requantization stage by dropping the original operation's quantization and
// enforcing the one needed by the PWL: setting the output scale (out_sc) to 1/1024
// and the clamp values to [-4096, 4095].
//
// The output of the PWL table for the Sigmoid will contain values between [0, 4095],
// which map to [0.0, 1.0] when dequantized. The post-shift stage helps translate the
// i13 results to i8/u8. Therefore, [0, 4095] >> 4 results in [0, 255]. The zero-point
// is then set to zero since the data is u8. The consumer operations need to account
// for this enforced quantization as well, so zero-point 0 and scale 1/255 are used to
// represent the [0.0, 1.0] float interval.
//
// The same behavior occurs for Tanh, with the mention that the output of the PWL table
// will contain values between [-4096, 4095] (corresponding to float [-1, 1]). Applying
// post-shift [-4096, 4095] >> 5 leads to [-128, 127], so a zero-point of 128 is added
// to obtain u8 results.
//
// NOTE: The scale and zero-point values mentioned above are fine-tuned to best fit the
// hardcoded PWL values for the Sigmoid and Tanh operations; they have been observed to
// lead to the best average accuracy.
//

const EnumMap<VPU::PPEMode, VPU::PwlQuantReqs> vpux::VPU::pwlQuantReqs = {
        {VPU::PPEMode::SIGMOID, {{-4.0, 4.0, 1.0 / 1015.6875, 0, 4}, {0.0, 1.0, 1.0 / 249.0, 3, 0}}},  //
        {VPU::PPEMode::TANH, {{-4.0, 4.0, 1.0 / 903.5, 128, 5}, {-1.0, 1.0, 1.0 / 127.0, 128, 0}}}     //
};

VPU::PwlQuantReqs vpux::VPU::getPwlQuantReqs(VPU::PPEMode ppeType) {
    const auto quantReqs = pwlQuantReqs.find(ppeType);
    VPUX_THROW_UNLESS(quantReqs != pwlQuantReqs.cend(), "Missing quantization requirements for PWL post-op {0}",
                      ppeType);
    return quantReqs->second;
}

int64_t vpux::VPU::getPwlPostShift(const VPU::PPEMode ppeType) {
    const auto quantReqs = VPU::getPwlQuantReqs(ppeType);
    return quantReqs.input.postShift;
}

int64_t vpux::VPU::getPwlClamp(const mlir::Type inElemType, const mlir::Type outElemType, const VPU::PPEMode ppeType,
                               const bool getMin) {
    constexpr int64_t CLAMP_MIN = -4096;
    constexpr int64_t CLAMP_MAX = 4095;

    // Input type defines the compute type
    if (inElemType.template isa<mlir::FloatType>()) {
        return getMin ? CLAMP_MIN : CLAMP_MAX;
    }

    const auto quantReqs = VPU::getPwlQuantReqs(ppeType);
    const auto rMin = quantReqs.input.rMin;
    const auto rMax = quantReqs.input.rMax;

    if (const auto qElemType = outElemType.template dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto scale = qElemType.getScale();
        const auto actualMin = std::max(CLAMP_MIN, static_cast<int64_t>(std::ceil(rMin / scale)));
        const auto actualMax = std::min(CLAMP_MAX, static_cast<int64_t>(std::floor(rMax / scale)));
        return getMin ? actualMin : actualMax;
    }
    VPUX_THROW("Unexpected output element type: {0}", outElemType);
}
