//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/opsets/opset1_decl.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"

#include "pretty_test_arguments.hpp"
#include "vpux/utils/core/checked_cast.hpp"
namespace ov::test {

using DynamicConvBiasConvParams = std::tuple<ov::test::InputShape, ov::element::Type, ov::Strides>;

class DynamicConvBiasConvTest : public testing::WithParamInterface<DynamicConvBiasConvParams>, public VpuOv2LayerTest {
    void configure_model() override {
        // tests should fail without this option enabled
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-apply-dynamic-boundary-correction=true";
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& inputStaticShape = targetInputStaticShapes[0];

        ov::Tensor inputTensor =
                ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), inputStaticShape, 3, 0);

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        const auto& [dataShape, type, strides] = this->GetParam();

        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(type, inputDynamicShapes.at(0));

        const size_t outputChannelNum = 3;
        const auto conv1 = buildConv(param, strides, outputChannelNum);
        const auto addConst = ov::op::v0::Constant::create(type, ov::Shape{1, outputChannelNum, 1, 1},
                                                           std::vector<float>(outputChannelNum, 3.f));
        const auto convWithBias = std::make_shared<ov::op::v1::Add>(conv1, addConst);
        const auto conv2 = buildConv(convWithBias, strides, outputChannelNum);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(conv2->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicConvBiasConv");
        auto preProc = ov::preprocess::PrePostProcessor(function);

        preProc.input().tensor().set_layout("NCHW");
        preProc.input().model().set_layout("NCHW");
        preProc.output().tensor().set_layout("NCHW");
        preProc.output().model().set_layout("NCHW");
        function = preProc.build();
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param, const ov::Strides& strides,
                                        size_t outputChannelNum) {
        const auto inputShape = param.get_partial_shape();

        VPUX_THROW_UNLESS(inputShape[1].is_static(), "Conv input C dim can't be dynamic");

        const auto inputC = inputShape[1].get_length();
        const auto weightsSize = inputC * outputChannelNum * 3 * 3;

        const auto weightsShape = ov::Shape{outputChannelNum, vpux::checked_cast<uint64_t>(inputC), 3, 3};

        std::vector<float> values(weightsSize, 2.f);
        const auto weights = ov::op::v0::Constant::create(param.get_element_type(), weightsShape, values);

        const ov::CoordinateDiff padsBegin = ov::CoordinateDiff(std::vector<ptrdiff_t>{1, 1});
        const ov::CoordinateDiff padsEnd = ov::CoordinateDiff(std::vector<ptrdiff_t>{1, 1});
        const ov::Strides dilations = ov::Strides(std::vector<size_t>{1, 1});
        auto conv2dNode = std::make_shared<ov::op::v1::Convolution>(param, weights->output(0), strides, padsBegin,
                                                                    padsEnd, dilations);

        return conv2dNode;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicConvBiasConvParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        const auto& [dataShape, type, strides] = obj.param;

        result << "InferShapes" << sep;
        for (auto i : dataShape.second) {
            result << i << sep;
        }
        result << strides << sep;
        result << "Type_" << type;
        return result.str();
    }
};

TEST_P(DynamicConvBiasConvTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicConvBiasConvTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<ov::test::InputShape> inShapes = {
        generateTestShape(1, 3, 32_Dyn, 16),      // dynamic H
        generateTestShape(1, 3, 16, 32_Dyn),      // dynamic W
        generateTestShape(1, 3, 32_Dyn, 32_Dyn),  // dynamic HW
};

const std::vector<ov::element::Type> inPrecision = {ov::element::f32, ov::element::f16, ov::element::i32};

const std::vector<ov::Strides> convStrides = {std::vector<size_t>{1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_ApplyBoundaryCorrection, DynamicConvBiasConvTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inPrecision),
                                            ::testing::ValuesIn(convStrides)),
                         DynamicConvBiasConvTest::getTestCaseName);

// Tracking number [E#156910]
const std::vector<ov::Strides> convStridesFailed = {std::vector<size_t>{2, 2}};
INSTANTIATE_TEST_SUITE_P(DISABLED_ApplyBoundaryCorrection, DynamicConvBiasConvTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inPrecision),
                                            ::testing::ValuesIn(convStridesFailed)),
                         DynamicConvBiasConvTest::getTestCaseName);

}  // namespace ov::test
