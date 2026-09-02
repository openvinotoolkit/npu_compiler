//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test {

// E#218933: a u16 network input with a preprocessing scale.
//
//   [input u16] -> (Convert f32) -> (Multiply scale) -> (FakeQuantize) -> [output]
//
// The Convert + Multiply lowers to a u16 activation quantized type plus a `dequantize` act-shave
// kernel (u16 -> f16). That kernel had no u16 implementation for __npu_major__ < 4 (silent garbage
// at runtime), and for 16-bit storage the scale/zero params are packed as F32 - the per-axis buffer
// offsets must account for that (paramsBpp), otherwise per-channel scales are mis-indexed.
//
// The subgraph isolates that path (no convolution, to avoid unrelated quantized-conv/padding noise).
// The NPU keeps the dequantized grid while the fp reference applies the FakeQuantize's 256-level grid
// (step = 166/255 ~= 0.65), so a correct dequantize still differs from the reference by up to about
// half a step (~0.33) plus f16 rounding. abs_threshold=0.5 covers that with margin (it is below a full
// step, by design), while a broken kernel (zeros/garbage/mis-indexed scales) is far outside it.

namespace {
constexpr float kAbsThreshold = 0.5f;
constexpr float kRelThreshold = 0.1f;
constexpr size_t kLevels = 256;
}  // namespace

// Per-tensor: a single scalar scale -> per-tensor u16 dequantize.
class DequantizeU16TestCommon : public VpuOv2LayerTest {
    void SetUp() override {
        rel_threshold = kRelThreshold;
        abs_threshold = kAbsThreshold;

        const ov::Shape inputShape{1, 4, 64, 64};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::u16, inputDynamicShapes.front())};
        const auto convertIn = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::f32);

        const auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {0.25f});
        const auto mul = std::make_shared<ov::op::v1::Multiply>(convertIn, scale);

        const std::vector<float> low = {0.0f};
        const std::vector<float> high = {166.0f};
        const auto fq = ov::test::utils::make_fake_quantize(mul, ov::element::f32, kLevels, {}, low, high, low, high);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fq)};
        function = std::make_shared<ov::Model>(results, params, "DequantizeU16");
    }
};

// Per-axis: a per-channel scale -> per-axis u16 dequantize (nParams > 1). Exercises the paramsBpp
// per-channel offset/indexing; distinct per-channel scales make a mis-indexed scale detectable.
class DequantizeU16PerAxisTestCommon : public VpuOv2LayerTest {
    void SetUp() override {
        rel_threshold = kRelThreshold;
        abs_threshold = kAbsThreshold;

        const ov::Shape inputShape{1, 4, 64, 64};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::u16, inputDynamicShapes.front())};
        const auto convertIn = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::f32);

        // Per-channel scale over the channel axis (C=4), values deliberately distinct.
        const auto scale =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 1, 1}, {0.25f, 0.5f, 0.75f, 1.0f});
        const auto mul = std::make_shared<ov::op::v1::Multiply>(convertIn, scale);

        const std::vector<float> low = {0.0f};
        const std::vector<float> high = {166.0f};
        const auto fq = ov::test::utils::make_fake_quantize(mul, ov::element::f32, kLevels, {}, low, high, low, high);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fq)};
        function = std::make_shared<ov::Model>(results, params, "DequantizeU16PerAxis");
    }
};

//
// Per-tensor
//
TEST_F(DequantizeU16TestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(DequantizeU16TestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(DequantizeU16TestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(DequantizeU16TestCommon, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

//
// Per-axis
//
TEST_F(DequantizeU16PerAxisTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(DequantizeU16PerAxisTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(DequantizeU16PerAxisTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(DequantizeU16PerAxisTestCommon, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test
