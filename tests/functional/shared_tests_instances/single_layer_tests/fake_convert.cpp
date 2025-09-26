//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/fake_convert.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/fake_convert.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class FakeConvertLayerTestCommon : public FakeConvertLayerTest, virtual public VpuOv2LayerTest {
    // Override base impl to create Scale/Shift inputs as ::Parameter,
    // to avoid mapping on DPU/FakeQuantize
    void SetUp() override {
        FakeConvertParams params = this->GetParam();
        std::vector<InputShape> data_shapes;
        Shape scale_shape, shift_shape;
        element::Type_t data_prec, dst_prec;
        bool default_shift;
        std::tie(data_shapes, scale_shape, shift_shape, data_prec, dst_prec, default_shift, targetDevice) = params;

        std::vector<ov::Shape> param_shapes;
        ov::Shape data_shape = data_shapes[0].second[0];
        param_shapes.push_back(data_shape);
        param_shapes.push_back(scale_shape);
        if (!default_shift) {
            param_shapes.push_back(shift_shape);
        }
        init_input_shapes(ov::test::static_shapes_to_test_representation(param_shapes));

        if (data_prec == ov::element::f16) {
            configuration.insert(ov::hint::inference_precision(ov::element::f16));
        } else if (data_prec == ov::element::bf16) {
            configuration.insert(ov::hint::inference_precision(ov::element::bf16));
        }

        const auto data = std::make_shared<ov::op::v0::Parameter>(data_prec, inputDynamicShapes.front());
        const auto scale = std::make_shared<ov::op::v0::Parameter>(data_prec, scale_shape);
        const auto shift = std::make_shared<ov::op::v0::Parameter>(data_prec, shift_shape);

        const auto fake_convert = default_shift
                                          ? std::make_shared<ov::op::v13::FakeConvert>(data, scale, dst_prec)
                                          : std::make_shared<ov::op::v13::FakeConvert>(data, scale, shift, dst_prec);
        auto inputs = ParameterVector{data, scale};
        if (!default_shift) {
            inputs.push_back(shift);
        }
        function = std::make_shared<ov::Model>(NodeVector{fake_convert}, inputs);
    }

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(FakeConvertLayerTestCommon, NPU3720_SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        skip << "SW-kernel incomplete for NPU 3720";
    });
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(FakeConvertLayerTestCommon, NPU4000_SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        skip << "SW-kernel incomplete for NPU 4000";
    });
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<ov::Shape>> shapes = {{{1, 256, 11, 31}}};
const std::vector<ov::element::Type> data_precisions = {ov::element::f16};
const std::vector<ov::element::Type> destination_precisions = {ov::element::f8e4m3, ov::element::f8e5m2};
const std::vector<bool> default_shift = {true, false};

const auto genParams = [](auto input, auto scaleShift) {
    return ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(input)),
                              ::testing::Values(scaleShift),  // scale
                              ::testing::Values(scaleShift),  // shift (must have same shape as scale)
                              ::testing::ValuesIn(data_precisions), ::testing::ValuesIn(destination_precisions),
                              ::testing::ValuesIn(default_shift), ::testing::Values(test_utils::TARGET_DEVICE));
};

const auto params1 = genParams(shapes[0], ov::Shape{1});
const auto params2 = genParams(shapes[0], ov::Shape{1, 256, 1, 1});
const auto params3 = genParams(shapes[0], ov::Shape{1, 1, 1, 31});

// E#161218 : flat zero output when going through 'ConsolidateActivationFP8QuantizationPass'
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_FakeConvert1, FakeConvertLayerTestCommon, params1,
                         FakeConvertLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeConvert2, FakeConvertLayerTestCommon, params2,
                         FakeConvertLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FakeConvert3, FakeConvertLayerTestCommon, params3,
                         FakeConvertLayerTestCommon::getTestCaseName);

}  // namespace
