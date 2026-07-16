//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::test {

struct DecomposeSoftmaxInSdpaParams {
    ov::Shape inputShape;
    size_t expandedChannels;
    ov::element::Type dataType;
};

class DecomposeSoftmaxInSdpaCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<DecomposeSoftmaxInSdpaParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0.0;
        in_data.range = 1.0;
        in_data.resolution = 32768;

        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], in_data);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    ov::Output<ov::Node> buildConvLayer(const ov::Output<ov::Node>& input, size_t inChannels,
                                        size_t outChannels) const {
        const auto weightsShape = ov::Shape{outChannels, inChannels, 1, 1};
        std::vector<float> weightsData(outChannels * inChannels, 0.0f);
        for (size_t i = 0; i < outChannels; ++i) {
            weightsData[i * inChannels + (i % inChannels)] = 1.0f;
        }
        const auto weights = ov::op::v0::Constant::create(inType, weightsShape, weightsData)->get_default_output();
        return std::make_shared<ov::op::v1::Convolution>(input, weights,
                                                         /*strides=*/ov::Strides{1, 1},
                                                         /*pads_begin=*/ov::CoordinateDiff{0, 0},
                                                         /*pads_end=*/ov::CoordinateDiff{0, 0},
                                                         /*dilations=*/ov::Strides{1, 1})
                ->get_default_output();
    }

    void SetUp() override {
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape;
        const auto inChannels = inputShape[1];
        const auto expandedChannels = testParams.expandedChannels;
        inType = outType = testParams.dataType;

        const ov::Shape nhwcShape = {inputShape[0], inputShape[2], inputShape[3], inputShape[1]};
        init_input_shapes(ov::test::static_shapes_to_test_representation({nhwcShape}));
        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(inputShape));

        const auto conv1 = buildConvLayer(input, inChannels, expandedChannels);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(conv1, /*axis=*/1)->get_default_output();
        const auto conv2 = buildConvLayer(softmax, expandedChannels, inChannels);

        const ov::ResultVector outputs{std::make_shared<ov::op::v0::Result>(conv2)};
        function = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input}, "DecomposeSoftmaxInSdpa");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NCHW"));
        preProc.output(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(0).model().set_layout(ov::Layout("NCHW"));
        function = preProc.build();

        abs_threshold = 0.5;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<DecomposeSoftmaxInSdpaParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }
};

TEST_P(DecomposeSoftmaxInSdpaCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-softmax-decomposition=true";
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(precommit_DecomposeSoftmaxInSdpa, DecomposeSoftmaxInSdpaCommon,
                         ::testing::Values(DecomposeSoftmaxInSdpaParams{{1, 48, 1024, 4}, 4096, ov::element::f16}),
                         DecomposeSoftmaxInSdpaCommon::getTestCaseName);

}  // namespace ov::test
