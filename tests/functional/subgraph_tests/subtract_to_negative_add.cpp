//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/subtract.hpp"

namespace ov::test {

struct SubtractTestParams {
    ov::Shape input1_shape;
    ov::Shape input2_shape;
};

//
//         [DeclareOp]
//             |
//         [Subtract] --- [DeclareOp]
//             |
//          [output]
//

class SubtractSubGraphTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(params[0], params[1]);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(subtract)};

        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shape_const_inputs, SubtractSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_subtract_diff_shape_const_inputs, SubtractSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 1},  // input1 shape
                                 {1, 1, 2, 4},  // input2 shape
                         }),
                         SubtractSubGraphTest_NPU3720::getTestCaseName);

//
//         [DeclareOp]
//             |
//       [FakeQuantize]
//             |
//         [Subtract] --- [FakeQuantize] --- [DeclareOp]
//             |
//       [FakeQuantize]
//             |
//          [output]
//

class SubtractFqSubGraphTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const size_t dataLevels = 255;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};

        const auto input1 = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {}, dataLow,
                                                                dataHigh, dataLow, dataHigh);
        const auto input2 = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {}, dataLow,
                                                                dataHigh, dataLow, dataHigh);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(input2, input2);
        const auto subtractFq = ov::test::utils::make_fake_quantize(subtract, ov::element::f16, dataLevels, {}, dataLow,
                                                                    dataHigh, dataLow, dataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(subtractFq)};

        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractFqSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shape_const_fq_inputs, SubtractFqSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractFqSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_subtract_diff_shape_const_fq_inputs, SubtractFqSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 4},  // input1 shape
                                 {1, 1, 2, 1},  // input2 shape
                         }),
                         SubtractFqSubGraphTest_NPU3720::getTestCaseName);

//
//         [DeclareOp]
//             |
//           [ReLU]
//             |
//         [Subtract] --- [ReLU] --- [DeclareOp]
//             |
//          [output]
//

class SubtractActInputsSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(params[0]);
        const auto relu2 = std::make_shared<ov::op::v0::Relu>(params[1]);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, relu2);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(subtract)};

        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractActInputsSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shapes_act_inputs, SubtractActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractActInputsSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_subtract_diff_shapes_act_inputs, SubtractActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 4},  // input1 shape
                                 {1, 1, 2, 1},  // input2 shape
                         }),
                         SubtractActInputsSubGraphTest_NPU3720::getTestCaseName);

//
//         [DeclareOp]
//             |
//           [ReLU]
//             |
//       [FakeQuantize]
//             |
//         [Subtract] --- [FakeQuantize] --- [ReLU] --- [DeclareOp]
//             |
//       [Convolution] --- [FakeQuantize] (weights) (applied to each Subtract operand)
//             |
//           [ReLU]
//             |
//       [FakeQuantize]
//             |
//          [output]
//

class SubtractFqActInputsSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    // Keep input1 >= input2: HW doesn't clamp negative broadcast-Subtract results to 0
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor input1 = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                    targetInputStaticShapes[0], 50, 150, 1);
        ov::Tensor input2 = ov::test::utils::create_and_fill_tensor(funcInputs[1].get_element_type(),
                                                                    targetInputStaticShapes[1], 50, 0, 1);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), input1});
        VpuOv2LayerTest::inputs.insert({funcInputs[1].get_node_shared_ptr(), input2});
    }
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const size_t dataLevels = 255;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};

        // Quantize each Subtract input via its own Conv so the FQ fuses into a PPE instead of
        // being stripped (broadcast Subtract can't become a quantized NCE.Eltwise)
        const auto buildQuantizedIdentityConv = [&](const ov::Output<ov::Node>& input) {
            const auto channels = input.get_shape().at(1);
            const auto convWShape = ov::Shape{channels, channels, 1, 1};
            const auto convWeights = ov::op::v0::Constant::create(ov::element::f16, convWShape,
                                                                  std::vector<float>(ov::shape_size(convWShape), 1.0f));
            const auto convWeightsFq = ov::test::utils::make_fake_quantize(convWeights, ov::element::f16, dataLevels,
                                                                           {}, {0.0f}, {1.0f}, {0.0f}, {1.0f});
            const auto conv = std::make_shared<ov::op::v1::Convolution>(input, convWeightsFq, ov::Strides{1, 1},
                                                                        ov::CoordinateDiff{0, 0},
                                                                        ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
            const auto convRelu = std::make_shared<ov::op::v0::Relu>(conv);
            return ov::test::utils::make_fake_quantize(convRelu, ov::element::f16, dataLevels, {}, dataLow, dataHigh,
                                                       dataLow, dataHigh);
        };

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(params[0]);
        const auto relu1Fq = ov::test::utils::make_fake_quantize(relu1, ov::element::f16, dataLevels, {}, dataLow,
                                                                 dataHigh, dataLow, dataHigh);
        const auto input1Quantized = buildQuantizedIdentityConv(relu1Fq->output(0));

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(params[1]);
        const auto relu2Fq = ov::test::utils::make_fake_quantize(relu2, ov::element::f16, dataLevels, {}, dataLow,
                                                                 dataHigh, dataLow, dataHigh);
        const auto input2Quantized = buildQuantizedIdentityConv(relu2Fq->output(0));

        const auto subtract =
                std::make_shared<ov::op::v1::Subtract>(input1Quantized->output(0), input2Quantized->output(0));

        // Quantized Conv + activation + output FQ so the FQ fuses into the Conv PPE
        const auto subtractChannels = subtract->get_output_shape(0).at(1);
        const auto convWShape = ov::Shape{subtractChannels, subtractChannels, 1, 1};
        const auto convWeights = ov::op::v0::Constant::create(ov::element::f16, convWShape,
                                                              std::vector<float>(ov::shape_size(convWShape), 0.5f));
        const auto convWeightsFq = ov::test::utils::make_fake_quantize(convWeights, ov::element::f16, dataLevels, {},
                                                                       {0.0f}, {0.5f}, {0.0f}, {0.5f});
        const auto conv = std::make_shared<ov::op::v1::Convolution>(subtract, convWeightsFq, ov::Strides{1, 1},
                                                                    ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                                    ov::Strides{1, 1});
        const auto convRelu = std::make_shared<ov::op::v0::Relu>(conv);
        const auto outFq = ov::test::utils::make_fake_quantize(convRelu, ov::element::f16, dataLevels, {}, dataLow,
                                                               dataHigh, dataLow, dataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractFqActInputsSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shapes_fq_act_inputs, SubtractFqActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractFqActInputsSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_subtract_diff_shapes_fq_act_inputs, SubtractFqActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 1},  // input1 shape
                                 {1, 1, 2, 4},  // input2 shape
                         }),
                         SubtractFqActInputsSubGraphTest_NPU3720::getTestCaseName);

}  // namespace ov::test
