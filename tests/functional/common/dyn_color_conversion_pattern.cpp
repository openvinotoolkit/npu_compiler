//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/dyn_color_conversion_pattern.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::test::subgraph {

namespace {

// Helper: create a scalar FakeQuantize with symmetric input/output ranges
std::shared_ptr<ov::Node> makeFQ(const std::shared_ptr<ov::Node>& input, float lo, float hi, int64_t levels) {
    auto loConst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {lo});
    auto hiConst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {hi});
    auto loConst2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {lo});
    auto hiConst2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {hi});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, loConst, hiConst, loConst2, hiConst2, levels);
}

}  // namespace

std::shared_ptr<ov::Node> buildDynYuvToRgbPattern(const std::shared_ptr<ov::op::v0::Parameter>& yInput,
                                                  const std::shared_ptr<ov::op::v0::Parameter>& uvInput) {
    // Y path: f16 -> Convert(ui8) -> Convert(f32) -> ShapeOf -> Gather([0,3,1,2]) -> DynamicReshape -> [1,1,H,W]
    auto yToU8 = std::make_shared<ov::op::v0::Convert>(yInput, ov::element::u8);
    auto yToF32 = std::make_shared<ov::op::v0::Convert>(yToU8, ov::element::f32);

    auto yShapeOf = std::make_shared<ov::op::v3::ShapeOf>(yToF32, ov::element::i64);
    auto gatherIndices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
    auto gatherAxis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto yGather = std::make_shared<ov::op::v8::Gather>(yShapeOf, gatherIndices, gatherAxis);
    auto yReshaped = std::make_shared<ov::op::v1::Reshape>(yToF32, yGather, true);

    // UV path: f16 -> Convert(ui8) -> Convert(f32) -> Transpose([0,3,1,2]) -> Interpolate(2x) -> FQ
    auto uvToU8 = std::make_shared<ov::op::v0::Convert>(uvInput, ov::element::u8);
    auto uvToF32 = std::make_shared<ov::op::v0::Convert>(uvToU8, ov::element::f32);

    auto transposeOrder =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
    auto uvTransposed = std::make_shared<ov::op::v1::Transpose>(uvToF32, transposeOrder);

    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;

    auto interpSizes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 1, 1, 1});
    auto interpScales =
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f});
    auto interpolated = std::make_shared<ov::op::v4::Interpolate>(uvTransposed, interpSizes, interpScales, attrs);

    // FakeQuantize on UV path: [0, 239.704999]
    auto uvFQ = makeFQ(interpolated, 0.0f, 239.704999f, 256);

    // Concat Y and UV channels along axis=1 (NCHW): [1,3,H,W]
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{yReshaped, uvFQ}, 1);

    // FakeQuantize on concat: [0, 239.704999]
    auto concatFQ = makeFQ(concat, 0.0f, 239.704999f, 256);

    // Conv weights: ui8 constants dequantized via Convert(f32) -> Subtract(73) -> Multiply(4.352326e-5).
    //
    // FuseColorConversion discards this matrix (it only sniffs RGB vs BGR from it) and replaces the whole
    // Concat->Conv->Add->FQ chain with IE.YuvToRgb, whose SHAVE kernel hardcodes BT.601 limited range
    // (sw_runtime_kernels/kernels/src/asm/50xx/nv12_to_rgb.asm). So these coefficients must be exactly
    // BT.601 * scaleFactor, otherwise the CPU reference and the NPU result diverge.
    //
    // scaleFactor is 1/255 here: the kernel emits [0, 255] and the output FakeQuantize below is [0, 1].
    // Row order is BGR, which is what detectOutputColorFormat() infers from |w[8]| > |w[2]|.
    //
    //           Y                U                V              u8 code (zp=73)
    //   B   1.164/255        2.018/255          0                178, 255,  73
    //   G   1.164/255       -0.391/255      -0.813/255           178,  38,   0
    //   R   1.164/255            0           1.596/255           178,  73, 217
    //
    // The zero point and scale are the ones the source model was calibrated with; they happen to fit
    // BT.601/255 almost exactly (zp 73 vs the ideal 73.23, max dequantization error 1.1e-5).
    std::vector<uint8_t> weightsU8 = {178, 255, 73, 178, 38, 0, 178, 73, 217};
    auto weightsConst = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{3, 3, 1, 1}, weightsU8);
    auto weightsF32 = std::make_shared<ov::op::v0::Convert>(weightsConst, ov::element::f32);
    auto zeroPoint = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {73.0f});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {4.3523261e-5f});
    auto weightsSub = std::make_shared<ov::op::v1::Subtract>(weightsF32, zeroPoint);
    auto weightsDequant = std::make_shared<ov::op::v1::Multiply>(weightsSub, scale);

    auto convolution = std::make_shared<ov::op::v1::Convolution>(concatFQ, weightsDequant, ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    // Bias: the BT.601 offsets folded over Y-16 / UV-128, scaled by the same 1/255.
    //   B: (-1.164*16 - 2.018*128) / 255 = -276.928 / 255
    //   G: (-1.164*16 + 0.391*128 + 0.813*128) / 255 =  135.488 / 255
    //   R: (-1.164*16 - 1.596*128) / 255 = -222.912 / 255
    // FuseColorConversion derives the YuvToRgb scale attribute from these, as
    // mean(bias[i] / standardBias[i]), so they must use the same scaleFactor as the weights above.
    std::vector<float> biasValues = {-1.08599216f, 0.53132549f, -0.87416471f};
    auto biasConst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, biasValues);
    auto add = std::make_shared<ov::op::v1::Add>(convolution, biasConst);

    // Output FakeQuantize: [0, 1]
    auto outputFQ = makeFQ(add, 0.0f, 1.0f, 256);

    // Transpose NCHW [1,3,H,W] -> NHWC [1,H,W,3] to match YuvToRgb output layout
    auto nchw2nhwcOrder =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    auto toNhwc = std::make_shared<ov::op::v1::Transpose>(outputFQ, nchw2nhwcOrder);

    // Convert f32 -> f16 for output
    auto toF16 = std::make_shared<ov::op::v0::Convert>(toNhwc, ov::element::f16);

    return toF16;
}

}  // namespace ov::test::subgraph
