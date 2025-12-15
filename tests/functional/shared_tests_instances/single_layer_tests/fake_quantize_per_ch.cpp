// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vpu_ov2_layer_test.hpp"

#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Test purpose: have a 'FakeQuantize' split into 'Quantize' + 'Dequantize'
// In HW-pipeline, 'Quantize' will run on DPU, 'Dequantize' on Shave
class FakeQuantPerChLayerTest_NPU3720 : virtual public VpuOv2LayerTest, public testing::WithParamInterface<ov::Shape> {
    void SetUp() override {
        ov::Shape shape = GetParam();
        inType = outType = ov::element::f16;
        const auto C = shape[1];

        init_input_shapes(static_shapes_to_test_representation({shape}));

        const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape))};

        const size_t levels = 256;

        // Different low/high limits per channel (must include 0 in range, so it splits into Q/DQ)
        // In 'DefaultHW' pipeline:
        // 'Quantize' will run on DPU
        // 'Dequantize' will run as SW-kernel
        std::vector<float> lo(C);
        std::vector<float> hi(C);
        for (size_t i = 0; i < C; ++i) {
            lo[i] = 0.0f;
            hi[i] = 8.0f + 0.2f * i * (i % 2 ? -1 : +1);
            if (hi[i] < lo[i]) {
                hi[i] = 8.0f + 0.2f * i;
            }
        }

        const auto dataFq =
                ov::test::utils::make_fake_quantize(params[0], inType, levels, {1, C, 1, 1}, lo, hi, lo, hi);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(dataFq)};
        function = std::make_shared<ov::Model>(results, params, "FakeQuantPerCh");

        rel_threshold = 0.2f;
    }
};

class FakeQuantPerChLayerTestConfig_NPU3720 : public FakeQuantPerChLayerTest_NPU3720 {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "merge-fake-quant=false";
    }
};

class FakeQuantPerChLayerTest_NPU4000 : public FakeQuantPerChLayerTestConfig_NPU3720 {};
class FakeQuantPerChLayerTest_NPU5010 : public FakeQuantPerChLayerTestConfig_NPU3720 {};

typedef std::tuple<ov::Shape, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
        FakeQuantPerChCustomLimitsTestParams;

// Test purpose: checking the functional results of the FQ Operation executed on shave for different ZPs for both
// input and output
class FakeQuantPerChCustomLimitsLayerTestCommon :
        virtual public VpuOv2LayerTest,
        public testing::WithParamInterface<FakeQuantPerChCustomLimitsTestParams> {
    void SetUp() override {
        ov::Shape shape;
        std::vector<float> ho;
        std::vector<float> lo;
        std::vector<float> hi;
        std::vector<float> li;
        inType = outType = ov::element::f16;
        std::tie(shape, ho, lo, hi, li) = this->GetParam();

        init_input_shapes(static_shapes_to_test_representation({shape}));

        const auto C = shape[1];

        const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape))};

        const size_t levels = 256;

        const auto dataFq =
                ov::test::utils::make_fake_quantize(params[0], inType, levels, {1, C, 1, 1}, li, hi, lo, ho);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(dataFq)};
        function = std::make_shared<ov::Model>(results, params, "FakeQuantPerCh");
    }
};

class FakeQuantPerChCustomLimitsLayerTest_NPU3720 : public FakeQuantPerChCustomLimitsLayerTestCommon {};

TEST_P(FakeQuantPerChLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FakeQuantPerChLayerTestConfig_NPU3720, SW) {
    rel_threshold = 0.001;
    abs_threshold = 0.2;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(FakeQuantPerChLayerTest_NPU4000, SW) {
    rel_threshold = 0.001;
    abs_threshold = 0.2;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(FakeQuantPerChLayerTest_NPU5010, SW) {
    rel_threshold = 0.001;
    abs_threshold = 0.2;
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
TEST_P(FakeQuantPerChCustomLimitsLayerTest_NPU3720, SW) {
    rel_threshold = 0.1;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {
const std::vector<ov::Shape> shapesHW = {
        {1, 16, 8, 32},
        {1, 32, 16, 8},
};

const std::vector<ov::Shape> shapesSW = {{1, 1, 1, 100}, {1, 3, 8, 32}, {1, 8, 3, 21},  {1, 13, 16, 8},
                                         {1, 16, 3, 5},  {1, 21, 2, 3}, {1, 32, 8, 16}, {1, 64, 3, 4}};

const std::vector<ov::Shape> shapesSWcustomLimits = {{1, 3, 199, 199}};

const std::vector<ov::Shape> shapesTiling = {
        {1, 64, 128, 100}, {1, 128, 68, 164},  // aclnet
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_FakeQuantPerCh, FakeQuantPerChLayerTest_NPU3720,
                         ::testing::ValuesIn(shapesHW));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantPerCh, FakeQuantPerChLayerTestConfig_NPU3720, ::testing::ValuesIn(shapesSW));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantPerCh, FakeQuantPerChLayerTest_NPU4000, ::testing::ValuesIn(shapesSW));
INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantPerCh, FakeQuantPerChLayerTest_NPU5010, ::testing::ValuesIn(shapesSW));
INSTANTIATE_TEST_SUITE_P(smoke_tiling_FakeQuantPerCh, FakeQuantPerChLayerTestConfig_NPU3720,
                         ::testing::ValuesIn(shapesTiling));

//{outHigh, outLow, inHigh, inLow}
// testing per-channel quantization with different ZPs for output
INSTANTIATE_TEST_SUITE_P(
        smoke_customLimits_FakeQuantPerCh1, FakeQuantPerChCustomLimitsLayerTest_NPU3720,
        ::testing::Combine(::testing::ValuesIn(shapesSWcustomLimits),
                           ::testing::Values(std::vector<float>{+2.63867188}),
                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125}),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02}),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02})));

// testing per-channel quantization with different ZPs for output and input
INSTANTIATE_TEST_SUITE_P(
        smoke_customLimits_FakeQuantPerCh2, FakeQuantPerChCustomLimitsLayerTest_NPU3720,
        ::testing::Combine(::testing::ValuesIn(shapesSWcustomLimits),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02}),
                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125}),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02}),
                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125})));

// testing per-channel quantization with different ZPs for input
INSTANTIATE_TEST_SUITE_P(smoke_customLimits_FakeQuantPerCh3, FakeQuantPerChCustomLimitsLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(shapesSWcustomLimits),
                                            ::testing::Values(std::vector<float>{+2.63867188}),
                                            ::testing::Values(std::vector<float>{-2.63867188}),
                                            ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02,
                                                                                 +2.780000e+02}),
                                            ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125})));
}  // namespace
